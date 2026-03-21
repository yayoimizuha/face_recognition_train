"""
Export a trained .pt checkpoint to five ONNX variants and apply INT8 static
quantization with calibration data from the training dataset.

All models output **classification logits** (arc_s * cos_similarity) over
num_classes.  Apply softmax on the consumer side to obtain probabilities.
Output shape: (batch_size, num_classes).

Class label names are embedded in ONNX metadata_props ("class_names" JSON,
"num_classes" str) so that a single ONNX file is a fully self-contained,
portable classification model.

Batch size is dynamic (any batch size works at inference time).

Usage:
    python hp_finetune/export_onnx.py --checkpoint hp_finetune/work_dirs/<run>/model_best.pt

Outputs (saved next to the checkpoint):
    <stem>.onnx            -- fp32  weights, fp32  activations
    <stem>_fp16.onnx       -- fp16  weights, fp16  activations (GPU fp16 inference)
    <stem>_bf16.onnx       -- bf16  weights, bf16  activations (GPU bf16 inference)
    <stem>_fp16int8.onnx   -- INT8  Conv (sensitive nodes excluded) + fp16 residual weights
    <stem>_bf16int8.onnx   -- INT8  Conv (sensitive nodes excluded) + bf16 residual weights

fp16 / bf16 notes:
  - Full fp16/bf16 models require a GPU runtime that supports native fp16/bf16
    execution (e.g. ONNX Runtime + CUDA on Ampere or later).  On CPU they fall
    back to fp32 and will not be faster than the fp32 model.
  - fp16int8 / bf16int8 store the non-quantised residual weights in fp16/bf16
    to reduce file size; actual INT8 Conv ops are hardware-accelerated on most
    modern runtimes regardless of the residual dtype.
"""

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxconverter_common import float16 as onnx_float16
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from hp_finetune.config_loader import RunConfig, load_run_config
from hp_finetune.data_utils import (
    DEFAULT_CALIB_SAMPLES,
    DEFAULT_EVAL_SAMPLES,
    FaceCalibrationDataReader,
    ImageConfig,
    get_class_names_from_dataset,
    get_real_sample,
)
from hp_finetune.finetune_facenet import FaceRecognitionModel
from hp_finetune.onnx_graph_utils import (
    embed_class_metadata,
    fix_hardcoded_batch_in_reshapes,
    make_batch_dim_dynamic,
    merge_external_data,
)
from hp_finetune.verification import (
    compare_outputs,
    evaluate_model_quality,
    print_file_sizes,
    verify_dynamic_batch,
)
from hp_finetune.weight_conversion import TargetDtype, convert_initializers

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
DEFAULT_OPSET = 18

# INT8 量子化対象のオペレータ。
# Conv と Gemm (Linear) のみを量子化する。これらは計算量が大きく INT8 の恩恵が
# 最も大きい一方、量子化耐性も比較的高い。
#
# ただし Conv/Gemm の中にも量子化に敏感なノードが存在する。
# _find_sensitive_nodes() で動的に特定し nodes_to_exclude で除外する。
#
# 以下は量子化対象から除外 (op_types_to_quantize に含めない):
# - Sigmoid, Clip: GWAP の exp(sigmoid(score)) チェーンで使われており、
#   [0,1] の狭い値域を INT8 (256 levels) で量子化すると精度が大幅に劣化する。
# - Relu: QDQ ノード挿入のオーバーヘッドが増える割に計算量削減が小さい。
# - Softmax, LayerNorm, MatMul (attention): 精度劣化が大きい。
# - BatchNormalization: per_channel 量子化で axis out-of-range エラーを起こす。
OP_TYPES_TO_QUANTIZE = ["Conv", "Gemm"]

_RESIDUAL_DTYPE_MAP: dict[str, TargetDtype] = {
    "fp16": TargetDtype.FP16,
    "bf16": TargetDtype.BF16,
}

# 動的感度検出のしきい値。
# 1ノードを INT8 量子化したときの fp32 との最大絶対差がこの値を超えたら
# そのノードを除外候補とみなす。
_SENSITIVITY_DIFF_THRESHOLD = 0.5

# キャリブレーション手法マッピング。
# CLI の --calib-method 文字列を CalibrationMethod に対応させる。
_CALIB_METHOD_MAP: dict[str, CalibrationMethod] = {
    "minmax": CalibrationMethod.MinMax,
    "entropy": CalibrationMethod.Entropy,
    "percentile": CalibrationMethod.Percentile,
}


def _build_calib_options(
    calib_method: str,
    percentile: float,
    *,
    activation_symmetric: bool = False,
    weight_symmetric: bool = True,
) -> tuple[CalibrationMethod, dict]:
    """Return ``(CalibrationMethod, extra_options)`` for :func:`quantize_static`.

    Args:
        calib_method: One of ``"minmax"``, ``"entropy"``, ``"percentile"``.
        percentile: Cutoff value used only when *calib_method* is
            ``"percentile"`` (e.g. ``99.999``).
        activation_symmetric: Passed to ``extra_options["ActivationSymmetric"]``.
        weight_symmetric: Passed to ``extra_options["WeightSymmetric"]``.

    Returns:
        A 2-tuple ``(method, extra_options)`` ready to unpack into
        ``quantize_static``.
    """
    method = _CALIB_METHOD_MAP[calib_method]
    extra: dict = {
        "ActivationSymmetric": activation_symmetric,
        "WeightSymmetric": weight_symmetric,
    }
    if calib_method == "percentile":
        extra["PercentileCalibrationValue"] = percentile
    return method, extra


# ──────────────────────────────────────────────
# Classification wrapper for ONNX export
# ──────────────────────────────────────────────
class ClassificationModel(nn.Module):
    """backbone + GWAP + head + arc_weight -> classification logits.

    Output is arc_s * cos_similarity(emb, arc_weight): ``(B, num_classes)``
    logits equivalent to ``FaceRecognitionModel.cos_logits()`` (no margin).
    Apply softmax on the consumer side for probabilities.

    Example::

        logits = sess.run(None, {"input": x})[0]   # (B, num_classes)
        probs  = softmax(logits, axis=1)
        pred   = logits.argmax(axis=1)
    """

    def __init__(self, full_model: FaceRecognitionModel):
        super().__init__()
        self.backbone = full_model.backbone
        self.gwap = full_model.gwap
        self.head = full_model.head
        self.arc_s = full_model.arc_s
        # Pre-normalised arc_weight as a frozen buffer
        w = F.normalize(full_model.arc_weight.data, dim=1)
        self.register_buffer("arc_weight_normalized", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.forward_features(x)
        pooled = self.gwap(feat)
        emb = self.head(pooled)
        emb = F.normalize(emb, dim=1)
        cos_sim = F.linear(emb, self.arc_weight_normalized)  # (B, num_classes)
        return cos_sim * self.arc_s  # logits


# ──────────────────────────────────────────────
# Checkpoint loading
# ──────────────────────────────────────────────
def _detect_num_classes(state_dict: dict, num_classes_arg: int | None) -> int:
    """Auto-detect num_classes from arc_weight in checkpoint.

    If ``--num-classes`` is given, prefer that but warn on mismatch.
    """
    has_arc_weight = "arc_weight" in state_dict

    if num_classes_arg is None:
        if not has_arc_weight:
            print(
                "Error: --num-classes not specified and arc_weight "
                "not found in checkpoint"
            )
            sys.exit(1)
        num_classes = state_dict["arc_weight"].shape[0]
        print(f"  Auto-detected num_classes={num_classes} from arc_weight shape")
        return num_classes

    if has_arc_weight:
        ckpt_num_classes = state_dict["arc_weight"].shape[0]
        if num_classes_arg != ckpt_num_classes:
            print(
                f"  [WARN] --num-classes={num_classes_arg} but "
                f"arc_weight has {ckpt_num_classes} classes"
            )
    return num_classes_arg


def load_classification_model(
    checkpoint_path: str,
    num_classes: int | None,
    run_cfg: RunConfig,
) -> ClassificationModel:
    """Load a checkpoint and return the inference-ready classification model."""
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    detected_num_classes = _detect_num_classes(state_dict, num_classes)

    full_model = FaceRecognitionModel(
        backbone_name=run_cfg.backbone,
        backbone_dim=run_cfg.backbone_dim,
        hidden_dim=run_cfg.hidden_dim,
        emb_size=run_cfg.emb_size,
        num_classes=detected_num_classes,
    )
    full_model.load_state_dict(state_dict)
    full_model.eval()

    cls_model = ClassificationModel(full_model)
    cls_model.eval()
    return cls_model


# ──────────────────────────────────────────────
# INT8 sensitive node detection (dynamic, output-difference based)
# ──────────────────────────────────────────────
class _SensitivityProbeReader(CalibrationDataReader):
    """Lightweight :class:`CalibrationDataReader` backed by a pre-built sample list.

    Used internally by :func:`_find_sensitive_nodes` to probe one Conv/Gemm
    node at a time without reloading data from disk.
    """

    def __init__(self, samples: list[np.ndarray]):
        self._samples = samples
        self._iter = iter(self._samples)

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            return {"input": next(self._iter)}
        except StopIteration:
            return None

    def rewind(self) -> None:
        self._iter = iter(self._samples)


def _find_sensitive_nodes(
    preprocessed_path: str,
    img_cfg: ImageConfig,
    calib_samples: list[np.ndarray],
    *,
    threshold: float = _SENSITIVITY_DIFF_THRESHOLD,
    calib_method: str = "minmax",
    percentile: float = 99.999,
) -> list[str]:
    """Identify Conv/Gemm nodes that degrade badly under INT8 quantization.

    For each Conv/Gemm node, a temporary model is quantized with *only that
    node* included in quantization.  A few calibration samples are run through
    both the fp32 base and the single-node-quantized model; if the maximum
    absolute output difference exceeds *threshold*, the node is flagged as
    sensitive and will be excluded from the full quantization pass.

    This approach is architecture-agnostic: it does not rely on MobileNetV4-
    specific node name patterns.

    Args:
        preprocessed_path: Path to the pre-processed (shape-inferred) fp32 ONNX.
        img_cfg: Image config (used only for shape information in error messages).
        calib_samples: Small list of calibration numpy arrays ``(1,3,H,W)``.
        threshold: Max-abs-diff threshold above which a node is sensitive.
        calib_method: Calibration method (``"minmax"``, ``"entropy"``,
            ``"percentile"``).  Passed to the single-node probe so the scan
            uses the same statistics strategy as the full quantization.
        percentile: Cutoff used only when *calib_method* is ``"percentile"``.

    Returns:
        List of node names to exclude from quantization.
    """
    model = onnx.load(preprocessed_path)
    candidate_nodes = [
        node
        for node in model.graph.node
        if node.op_type in ("Conv", "Gemm") and node.name
    ]

    if not candidate_nodes:
        return []

    print(
        f"  Sensitivity scan: testing {len(candidate_nodes)} Conv/Gemm nodes "
        f"with {len(calib_samples)} samples (threshold={threshold})..."
    )

    # fp32 baseline outputs
    sess_fp32 = ort.InferenceSession(preprocessed_path)
    fp32_outputs = [sess_fp32.run(None, {"input": s})[0] for s in calib_samples]

    sensitive: list[str] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        _probe_method, _probe_extra = _build_calib_options(calib_method, percentile)
        for node in candidate_nodes:
            node_name = node.name
            tmp_path = os.path.join(tmp_dir, "single_node_quant.onnx")

            # Minimal CalibrationDataReader for a single-node quantization probe
            probe_reader = _SensitivityProbeReader(calib_samples)

            try:
                quantize_static(
                    model_input=preprocessed_path,
                    model_output=tmp_path,
                    calibration_data_reader=probe_reader,
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8,
                    per_channel=True,
                    reduce_range=False,
                    op_types_to_quantize=["Conv", "Gemm"],
                    nodes_to_quantize=[node_name],
                    calibrate_method=_probe_method,
                    extra_options=_probe_extra,
                )
            except Exception as e:
                # If quantizing this single node fails outright, exclude it
                print(f"    [{node_name}] quantization failed ({e!s:.80}), excluding")
                sensitive.append(node_name)
                continue

            try:
                sess_q = ort.InferenceSession(tmp_path)
                max_diff = max(
                    float(
                        np.max(
                            np.abs(fp32_outputs[i] - sess_q.run(None, {"input": s})[0])
                        )
                    )
                    for i, s in enumerate(calib_samples)
                )
            except Exception as e:
                print(f"    [{node_name}] inference failed ({e!s:.80}), excluding")
                sensitive.append(node_name)
                continue

            if max_diff > threshold:
                print(
                    f"    [{node_name}] sensitive  max_diff={max_diff:.4f} > {threshold}"
                )
                sensitive.append(node_name)

    print(
        f"  Sensitivity scan complete: "
        f"{len(sensitive)}/{len(candidate_nodes)} nodes excluded"
    )
    return sensitive


# ──────────────────────────────────────────────
# Export: fp32 (base model)
# ──────────────────────────────────────────────
def export_fp32_onnx(
    cls_model: ClassificationModel,
    onnx_path: str,
    opset: int,
    *,
    img_cfg: ImageConfig,
    num_classes: int,
    class_names: list[str],
) -> None:
    """Export PyTorch model to fp32 ONNX with dynamic batch and metadata."""
    print(f"Exporting fp32 ONNX to: {onnx_path}")
    dummy_input = torch.randn(1, 3, img_cfg.input_size, img_cfg.input_size)

    torch.onnx.export(
        cls_model,
        dummy_input,
        onnx_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    merge_external_data(onnx_path)

    # Fix hardcoded batch dims in Reshape nodes
    print("  Fixing hardcoded batch dimension in Reshape nodes...")
    onnx_model = onnx.load(onnx_path)
    fix_hardcoded_batch_in_reshapes(
        onnx_model,
        onnx_path,
        input_size=img_cfg.input_size,
        rng=np.random.default_rng(0),
    )

    # Make graph I/O batch dim symbolic
    onnx_model = onnx.load(onnx_path)
    make_batch_dim_dynamic(onnx_model)
    onnx.save(onnx_model, onnx_path)

    # Verify
    print("  Verifying dynamic batch with random batch sizes...")
    verify_dynamic_batch(
        onnx_path,
        input_size=img_cfg.input_size,
        num_classes=num_classes,
        rng=np.random.default_rng(0),
        label="fp32",
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"  fp32 ONNX model verified (opset={opset})")

    # Embed class metadata
    embed_class_metadata(onnx_model, class_names)
    onnx.save(onnx_model, onnx_path)
    print(f"  Embedded {len(class_names)} class names in metadata_props")

    # Verify output vs PyTorch (single real image)
    print("Verifying fp32 ONNX output vs PyTorch (real image)...")
    real_sample = get_real_sample(img_cfg)
    real_tensor = torch.from_numpy(real_sample)
    with torch.no_grad():
        pt_out = cls_model(real_tensor).numpy()
    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {"input": real_sample})[0]
    compare_outputs(pt_out, ort_out, label="PyTorch vs fp32 ONNX")


# ──────────────────────────────────────────────
# Export: full fp16
# ──────────────────────────────────────────────
def export_fp16_onnx(
    fp32_path: str,
    fp16_path: str,
    *,
    img_cfg: ImageConfig,
    num_classes: int,
    class_names: list[str],
) -> None:
    """Convert entire fp32 graph to fp16 (GPU Ampere+ accelerated).

    I/O tensors stay fp32 (``keep_io_types=True``) for consumer convenience.
    """
    print(f"Exporting fp16 ONNX to: {fp16_path}")
    fp32_model = onnx.load(fp32_path)

    fp16_model = onnx_float16.convert_float_to_float16(
        fp32_model,
        keep_io_types=True,
        disable_shape_infer=False,
        check_fp16_ready=False,
    )

    embed_class_metadata(fp16_model, class_names)
    onnx.save(fp16_model, fp16_path)

    print("  Verifying fp16 dynamic batch...")
    verify_dynamic_batch(
        fp16_path,
        input_size=img_cfg.input_size,
        num_classes=num_classes,
        rng=np.random.default_rng(2),
        label="fp16",
    )
    print(f"  fp16 ONNX saved: {fp16_path}")


# ──────────────────────────────────────────────
# Export: full bf16
# ──────────────────────────────────────────────
def export_bf16_onnx(
    fp32_path: str,
    bf16_path: str,
    *,
    img_cfg: ImageConfig,
    num_classes: int,
    class_names: list[str],
) -> None:
    """Convert all fp32 weights to bf16 (stored as bf16, Cast to fp32 at runtime).

    I/O stays fp32; no consumer-side dtype handling needed.
    """
    print(f"Exporting bf16 ONNX to: {bf16_path}")
    bf16_model = onnx.load(fp32_path)

    converted = convert_initializers(
        bf16_model,
        TargetDtype.BF16,
        min_elements=0,
        exclude_quant_params=False,
    )
    print(f"  Converted {converted} initializers to bf16")

    embed_class_metadata(bf16_model, class_names)
    onnx.save(bf16_model, bf16_path)

    print("  Verifying bf16 dynamic batch...")
    verify_dynamic_batch(
        bf16_path,
        input_size=img_cfg.input_size,
        num_classes=num_classes,
        rng=np.random.default_rng(3),
        label="bf16",
    )
    print(f"  bf16 ONNX saved: {bf16_path}")


# ──────────────────────────────────────────────
# Export: INT8 static quantization
# ──────────────────────────────────────────────
def export_int8_onnx(
    fp32_path: str,
    int8_path: str,
    calib_samples: int,
    *,
    img_cfg: ImageConfig,
    num_classes: int,
    class_names: list[str],
    residual_dtype: str = "fp16",
    calib_method: str = "minmax",
    percentile: float = 99.999,
) -> None:
    """Produce an INT8 statically-quantized model from the fp32 base.

    Sensitive Conv/Gemm nodes are detected dynamically by comparing per-node
    INT8 output against fp32; nodes with large divergence are excluded.
    Residual (non-quantized) weights are compressed to *residual_dtype*
    ("fp16"/"bf16").

    Args:
        calib_method: One of ``"minmax"`` (default), ``"entropy"``,
            ``"percentile"``.  Both the sensitivity scan and the final
            quantization pass use the same method for consistency.
        percentile: Cutoff used only when *calib_method* is ``"percentile"``.
    """
    label = f"INT8+{residual_dtype}"
    print(f"Running {label} quantization with calibration...")
    print(f"  calibration samples: {calib_samples}")
    print(
        f"  calibration method:  {calib_method}"
        + (f" (percentile={percentile})" if calib_method == "percentile" else "")
    )
    print(f"  quantized op types: {OP_TYPES_TO_QUANTIZE}")

    # Pre-process: shape inference + model optimization
    preprocessed_path = fp32_path.replace(".onnx", "_preproc.onnx")
    print(f"  Pre-processing for quantization: {preprocessed_path}")
    quant_pre_process(
        input_model=fp32_path,
        output_model_path=preprocessed_path,
        skip_symbolic_shape=True,
    )

    # Build calibration data (reused for sensitivity scan + quantization)
    calib_reader = FaceCalibrationDataReader(img_cfg, num_samples=calib_samples)
    probe_samples = calib_reader.samples[: min(8, len(calib_reader.samples))]

    # Identify sensitive nodes dynamically via output-difference comparison
    nodes_to_exclude = _find_sensitive_nodes(
        preprocessed_path,
        img_cfg,
        probe_samples,
        calib_method=calib_method,
        percentile=percentile,
    )
    print(f"  Excluding {len(nodes_to_exclude)} sensitive nodes from quantization")

    calib_method_enum, extra_options = _build_calib_options(calib_method, percentile)
    calib_reader.rewind()
    quantize_static(
        model_input=preprocessed_path,
        model_output=int8_path,
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        op_types_to_quantize=OP_TYPES_TO_QUANTIZE,
        nodes_to_exclude=nodes_to_exclude,
        calibrate_method=calib_method_enum,
        extra_options=extra_options,
    )

    # Clean up intermediate file
    if os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)

    # Post-process: check, fix Reshape, dynamic batch, residual compression
    onnx_int8_model = onnx.load(int8_path)
    onnx.checker.check_model(onnx_int8_model)

    print(f"  Fixing {label} model batch dimensions...")
    fix_hardcoded_batch_in_reshapes(
        onnx_int8_model,
        int8_path,
        input_size=img_cfg.input_size,
        rng=np.random.default_rng(1),
        label=label,
    )
    make_batch_dim_dynamic(onnx_int8_model)

    # Compress residual fp32 weights to fp16/bf16
    target_dtype = _RESIDUAL_DTYPE_MAP[residual_dtype]
    residual_count = convert_initializers(onnx_int8_model, target_dtype)
    print(
        f"  Converted {residual_count} residual weight initializers to {residual_dtype}"
    )

    embed_class_metadata(onnx_int8_model, class_names)
    print(f"  Embedded {len(class_names)} class names in metadata_props")
    onnx.save(onnx_int8_model, int8_path)

    # Verify
    print(f"  Verifying {label} dynamic batch with random batch sizes...")
    verify_dynamic_batch(
        int8_path,
        input_size=img_cfg.input_size,
        num_classes=num_classes,
        rng=np.random.default_rng(1),
        label=label,
    )
    print(f"  {label} ONNX model saved and verified: {int8_path}")

    # Quick output quality check (single real image)
    print(f"Verifying {label} ONNX output vs fp32 ONNX (real image, logits)...")
    real_sample = get_real_sample(img_cfg)
    sess_fp32 = ort.InferenceSession(fp32_path)
    sess_int8 = ort.InferenceSession(int8_path)
    fp32_out = sess_fp32.run(None, {"input": real_sample})[0]
    int8_out = sess_int8.run(None, {"input": real_sample})[0]
    compare_outputs(fp32_out, int8_out, label=f"fp32 vs {label}", warn_threshold=0.05)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export .pt checkpoint to 5 ONNX variants: "
            "fp32 / fp16 / bf16 / fp16+INT8 / bf16+INT8"
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a .pt checkpoint (e.g. model_best.pt)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of classes. Auto-detected from arc_weight in checkpoint if omitted.",
    )
    parser.add_argument(
        "--calib-samples",
        type=int,
        default=DEFAULT_CALIB_SAMPLES,
        help="Number of calibration samples for INT8 quantization",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=DEFAULT_EVAL_SAMPLES,
        help="Number of real samples for quality evaluation per variant",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--calib-method",
        choices=["minmax", "entropy", "percentile"],
        default="minmax",
        help=(
            "Calibration method for INT8 static quantization. "
            "'minmax' (default): fast, uses observed min/max. "
            "'entropy': minimises KL-divergence between fp32 and INT8 distributions "
            "(TensorRT-style, slower but typically more accurate). "
            "'percentile': clips at --percentile cutoff to ignore outliers."
        ),
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.999,
        help=(
            "Percentile cutoff used only when --calib-method=percentile. "
            "E.g. 99.999 clips the top 0.001%% of activations before scaling."
        ),
    )
    return parser.parse_args()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    args = parse_args()

    checkpoint_path = args.checkpoint
    if not os.path.isfile(checkpoint_path):
        print(f"Error: checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Load architecture constants from the saved finetune_facenet.py copy
    print(f"Loading run config from saved script next to checkpoint...")
    run_cfg = load_run_config(checkpoint_path)
    print(
        f"  backbone={run_cfg.backbone}  input_size={run_cfg.input_size}  "
        f"emb_size={run_cfg.emb_size}  backbone_dim={run_cfg.backbone_dim}"
    )

    # Build ImageConfig for preprocessing / verification
    img_cfg = ImageConfig(
        input_size=run_cfg.input_size,
        mean=run_cfg.imagenet_mean,
        std=run_cfg.imagenet_std,
    )

    out_dir = os.path.dirname(checkpoint_path) or "."
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]

    paths = {
        "fp32": os.path.join(out_dir, f"{stem}.onnx"),
        "fp16": os.path.join(out_dir, f"{stem}_fp16.onnx"),
        "bf16": os.path.join(out_dir, f"{stem}_bf16.onnx"),
        "fp16int8": os.path.join(out_dir, f"{stem}_fp16int8.onnx"),
        "bf16int8": os.path.join(out_dir, f"{stem}_bf16int8.onnx"),
    }

    # Step 1: Load class names from dataset
    print("Loading class names from dataset...")
    class_names = get_class_names_from_dataset()
    num_classes = len(class_names)
    print(f"  {num_classes} class names loaded")

    meta = dict(num_classes=num_classes, class_names=class_names)

    # Step 2: Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    cls_model = load_classification_model(checkpoint_path, args.num_classes, run_cfg)

    # Step 3: fp32 ONNX (base -- other variants derive from this)
    export_fp32_onnx(
        cls_model, paths["fp32"], opset=args.opset, img_cfg=img_cfg, **meta
    )

    # Step 4: fp16 full conversion
    export_fp16_onnx(paths["fp32"], paths["fp16"], img_cfg=img_cfg, **meta)

    # Step 5: bf16 full conversion
    export_bf16_onnx(paths["fp32"], paths["bf16"], img_cfg=img_cfg, **meta)

    # Step 6: fp16+INT8 quantization
    export_int8_onnx(
        paths["fp32"],
        paths["fp16int8"],
        calib_samples=args.calib_samples,
        img_cfg=img_cfg,
        residual_dtype="fp16",
        calib_method=args.calib_method,
        percentile=args.percentile,
        **meta,
    )

    # Step 7: bf16+INT8 quantization
    export_int8_onnx(
        paths["fp32"],
        paths["bf16int8"],
        calib_samples=args.calib_samples,
        img_cfg=img_cfg,
        residual_dtype="bf16",
        calib_method=args.calib_method,
        percentile=args.percentile,
        **meta,
    )

    # Step 8: Multi-sample quality evaluation for each variant
    for label in ("fp16", "bf16", "fp16int8", "bf16int8"):
        evaluate_model_quality(
            paths["fp32"],
            paths[label],
            img_cfg=img_cfg,
            num_samples=args.eval_samples,
            label=label,
        )

    # Summary
    print_file_sizes(paths)
    print("\nDone.")


if __name__ == "__main__":
    main()

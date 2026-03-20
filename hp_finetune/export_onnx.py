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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxconverter_common import float16 as onnx_float16
from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from hp_finetune.data_utils import (
    DEFAULT_CALIB_SAMPLES,
    DEFAULT_EVAL_SAMPLES,
    FaceCalibrationDataReader,
    get_class_names_from_dataset,
    get_real_sample,
)
from hp_finetune.finetune_facenet import (
    BACKBONE,
    BACKBONE_DIM,
    EMB_SIZE,
    HIDDEN_DIM,
    INPUT_SIZE,
    FaceRecognitionModel,
)
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
    checkpoint_path: str, num_classes: int
) -> ClassificationModel:
    """Load a checkpoint and return the inference-ready classification model."""
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    detected_num_classes = _detect_num_classes(state_dict, num_classes)

    full_model = FaceRecognitionModel(
        backbone_name=BACKBONE,
        backbone_dim=BACKBONE_DIM,
        hidden_dim=HIDDEN_DIM,
        emb_size=EMB_SIZE,
        num_classes=detected_num_classes,
    )
    full_model.load_state_dict(state_dict)
    full_model.eval()

    cls_model = ClassificationModel(full_model)
    cls_model.eval()
    return cls_model


# ──────────────────────────────────────────────
# INT8 sensitive node detection
# ──────────────────────────────────────────────
def _find_sensitive_nodes(preprocessed_path: str) -> list[str]:
    """Identify Conv/Gemm nodes that degrade badly under INT8 quantization.

    Sensitive categories:
    1. Depthwise Conv (dw_start, dw_mid) -- single-filter-per-channel
    2. ConvMulFusion (BN-folded Conv) -- extreme quantization scales
    3. Gemm (embedding head + arc_weight cosine) -- amplified by L2 norm
    4. GWAP score_conv -- feeds exp(sigmoid(score)) chain
    """
    model = onnx.load(preprocessed_path)
    sensitive: list[str] = []

    for node in model.graph.node:
        if node.op_type not in ("Conv", "Gemm"):
            continue

        weight_name = node.input[1] if len(node.input) > 1 else ""
        is_sensitive = (
            "dw_start" in weight_name
            or "dw_mid" in weight_name
            or "ConvMulFusion" in weight_name
            or node.op_type == "Gemm"
            or "score_conv" in weight_name
            or "conv2d" in node.name
        )
        if is_sensitive:
            sensitive.append(node.name)

    return sensitive


# ──────────────────────────────────────────────
# Export: fp32 (base model)
# ──────────────────────────────────────────────
def export_fp32_onnx(
    cls_model: ClassificationModel,
    onnx_path: str,
    opset: int,
    *,
    num_classes: int,
    class_names: list[str],
) -> None:
    """Export PyTorch model to fp32 ONNX with dynamic batch and metadata."""
    print(f"Exporting fp32 ONNX to: {onnx_path}")
    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

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
    fix_hardcoded_batch_in_reshapes(onnx_model, onnx_path, rng=np.random.default_rng(0))

    # Make graph I/O batch dim symbolic
    onnx_model = onnx.load(onnx_path)
    make_batch_dim_dynamic(onnx_model)
    onnx.save(onnx_model, onnx_path)

    # Verify
    print("  Verifying dynamic batch with random batch sizes...")
    verify_dynamic_batch(
        onnx_path, num_classes=num_classes, rng=np.random.default_rng(0), label="fp32"
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
    real_sample = get_real_sample()
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
        fp16_path, num_classes=num_classes, rng=np.random.default_rng(2), label="fp16"
    )
    print(f"  fp16 ONNX saved: {fp16_path}")


# ──────────────────────────────────────────────
# Export: full bf16
# ──────────────────────────────────────────────
def export_bf16_onnx(
    fp32_path: str,
    bf16_path: str,
    *,
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
        bf16_path, num_classes=num_classes, rng=np.random.default_rng(3), label="bf16"
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
    num_classes: int,
    class_names: list[str],
    residual_dtype: str = "fp16",
) -> None:
    """Produce an INT8 statically-quantized model from the fp32 base.

    Sensitive Conv/Gemm nodes are excluded from quantization.  Residual
    (non-quantized) weights are compressed to *residual_dtype* ("fp16"/"bf16").
    """
    label = f"INT8+{residual_dtype}"
    print(f"Running {label} quantization with calibration...")
    print(f"  calibration samples: {calib_samples}")
    print(f"  quantized op types: {OP_TYPES_TO_QUANTIZE}")

    # Pre-process: shape inference + model optimization
    preprocessed_path = fp32_path.replace(".onnx", "_preproc.onnx")
    print(f"  Pre-processing for quantization: {preprocessed_path}")
    quant_pre_process(
        input_model=fp32_path,
        output_model_path=preprocessed_path,
        skip_symbolic_shape=True,
    )

    # Identify sensitive nodes to exclude
    nodes_to_exclude = _find_sensitive_nodes(preprocessed_path)
    print(f"  Excluding {len(nodes_to_exclude)} sensitive nodes from quantization")

    calib_reader = FaceCalibrationDataReader(num_samples=calib_samples)
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
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": True,
        },
    )

    # Clean up intermediate file
    if os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)

    # Post-process: check, fix Reshape, dynamic batch, residual compression
    onnx_int8_model = onnx.load(int8_path)
    onnx.checker.check_model(onnx_int8_model)

    print(f"  Fixing {label} model batch dimensions...")
    fix_hardcoded_batch_in_reshapes(
        onnx_int8_model, int8_path, rng=np.random.default_rng(1), label=label
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
        int8_path, num_classes=num_classes, rng=np.random.default_rng(1), label=label
    )
    print(f"  {label} ONNX model saved and verified: {int8_path}")

    # Quick output quality check (single real image)
    print(f"Verifying {label} ONNX output vs fp32 ONNX (real image, logits)...")
    real_sample = get_real_sample()
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
    cls_model = load_classification_model(checkpoint_path, args.num_classes)

    # Step 3: fp32 ONNX (base -- other variants derive from this)
    export_fp32_onnx(cls_model, paths["fp32"], opset=args.opset, **meta)

    # Step 4: fp16 full conversion
    export_fp16_onnx(paths["fp32"], paths["fp16"], **meta)

    # Step 5: bf16 full conversion
    export_bf16_onnx(paths["fp32"], paths["bf16"], **meta)

    # Step 6: fp16+INT8 quantization
    export_int8_onnx(
        paths["fp32"],
        paths["fp16int8"],
        calib_samples=args.calib_samples,
        residual_dtype="fp16",
        **meta,
    )

    # Step 7: bf16+INT8 quantization
    export_int8_onnx(
        paths["fp32"],
        paths["bf16int8"],
        calib_samples=args.calib_samples,
        residual_dtype="bf16",
        **meta,
    )

    # Step 8: Multi-sample quality evaluation for each variant
    for label in ("fp16", "bf16", "fp16int8", "bf16int8"):
        evaluate_model_quality(
            paths["fp32"],
            paths[label],
            num_samples=args.eval_samples,
            label=label,
        )

    # Summary
    print_file_sizes(paths)
    print("\nDone.")


if __name__ == "__main__":
    main()

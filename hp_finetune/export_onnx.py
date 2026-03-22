"""
Export a trained .pt checkpoint to seven ONNX variants with per-block
weight quantization (Block-FP8 style).

Uses only standard ONNX nodes (Cast, Mul, Reshape, Slice) for weight
restoration — no DequantizeLinear — ensuring full TensorRT compatibility.

Two checkpoint types are supported:
  - model_best.pt                    (classification only)
  - model_best_with_anomaly.pt       (classification + AnomalyClassifier)

Output variants:
    <stem>.onnx              -- fp32  weights, fp32  activations
    <stem>_bf16.onnx         -- bf16  weights, bf16  activations (graph-wide)
    <stem>_fp16.onnx         -- fp16  weights, fp16  activations (graph-wide)
    <stem>_bf16int8.onnx     -- per-block INT8 weights (bf16 scale) + bf16 activations
    <stem>_bf16fp8.onnx      -- per-block FP8  weights (bf16 scale) + bf16 activations
    <stem>_fp16int8.onnx     -- per-block INT8 weights (fp16 scale) + fp16 activations
    <stem>_fp16fp8.onnx      -- per-block FP8  weights (fp16 scale) + fp16 activations

AnomalyClassifier nodes are always kept in fp32 for numerical stability.
Batch size is dynamic (any batch size works at inference time).
Class label names are embedded in ONNX metadata_props.

Sensitive weight nodes (those that degrade model accuracy when quantized)
are automatically detected via a 2-stage hybrid approach:
  Stage 1: weight reconstruction error (NRMSE) — fast, no inference
  Stage 2: output difference on probe samples — accurate, for suspect nodes

Usage:
    python hp_finetune/export_onnx.py \\
        --checkpoint hp_finetune/work_dirs/<run>/model_best_with_anomaly.pt

    python hp_finetune/export_onnx.py \\
        --checkpoint hp_finetune/work_dirs/<run>/model_best_with_anomaly.pt \\
        --block-size 64 --nrmse-threshold 0.01
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from hp_finetune.config_loader import RunConfig, load_run_config
from hp_finetune.data_utils import (
    DEFAULT_CALIB_SAMPLES,
    DEFAULT_EVAL_SAMPLES,
    ImageConfig,
    get_class_names_from_dataset,
    get_real_sample,
    load_eval_samples,
)
from hp_finetune.finetune_facenet import FaceRecognitionModel
from hp_finetune.onnx_graph_utils import (
    convert_graph_to_bf16,
    convert_graph_to_fp16,
    embed_class_metadata,
    fix_hardcoded_batch_in_reshapes,
    get_anomaly_initializer_names,
    infer_shapes_for_tensorrt,
    make_batch_dim_dynamic,
    make_intermediate_batch_dims_dynamic,
    merge_external_data,
)
from hp_finetune.verification import (
    compare_outputs,
    evaluate_model_quality,
    print_file_sizes,
    verify_dynamic_batch,
)
from hp_finetune.weight_conversion import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_FP8_FORMAT,
    DEFAULT_OUTPUT_DIFF_THRESHOLD,
    DEFAULT_SENSITIVITY_SAMPLES,
    DEFAULT_WEIGHT_NRMSE_THRESHOLD,
    QuantDtype,
    TargetDtype,
    apply_block_quantization,
    find_sensitive_initializers,
)


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
DEFAULT_OPSET = 19  # opset 19 required for FLOAT8E4M3FN

_FP8_FORMAT_MAP: dict[str, QuantDtype] = {
    "e4m3fn": QuantDtype.FP8E4M3,
    "e4m3": QuantDtype.FP8E4M3,
    "e5m2": QuantDtype.FP8E5M2,
}


# ──────────────────────────────────────────────
# Classification wrapper for ONNX export
# ──────────────────────────────────────────────
class _ClassificationBase(nn.Module):
    """Common base: backbone + GWAP + head + arc_weight initialisation."""

    def __init__(self, full_model: FaceRecognitionModel):
        super().__init__()
        self.backbone = full_model.backbone
        self.gwap = full_model.gwap
        self.head = full_model.head
        self.arc_s = full_model.arc_s
        w = F.normalize(full_model.arc_weight.data, dim=1)
        self.register_buffer("arc_weight_normalized", w)

    def _embed_and_classify(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone.forward_features(x)
        pooled = self.gwap(feat)
        raw_emb = self.head(pooled)
        emb_norm = F.normalize(raw_emb, dim=1)
        logits = F.linear(emb_norm, self.arc_weight_normalized) * self.arc_s
        return raw_emb, logits


class ClassificationModel(_ClassificationBase):
    """backbone + GWAP + head + arc_weight → logits ``(B, num_classes)``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _raw_emb, logits = self._embed_and_classify(x)
        return logits


class ClassificationWithAnomalyModel(_ClassificationBase):
    """backbone + GWAP + head + arc_weight + AnomalyClassifier → (logits, anomaly_score).

    anomaly_score = sigmoid(AnomalyClassifier(gwap_out))
    入力特徴量は GWAP 出力（backbone 生特徴、embedding head 変換前）。
    """

    def __init__(self, full_model: FaceRecognitionModel):
        super().__init__(full_model)
        # AnomalyClassifier の重みをコピー
        self.anomaly_fc1_weight = nn.Parameter(
            full_model.anomaly.fc1.weight.data.clone(), requires_grad=False
        )
        self.anomaly_fc1_bias = nn.Parameter(
            full_model.anomaly.fc1.bias.data.clone(), requires_grad=False
        )
        self.anomaly_bn1_weight = nn.Parameter(
            full_model.anomaly.bn1.weight.data.clone(), requires_grad=False
        )
        self.anomaly_bn1_bias = nn.Parameter(
            full_model.anomaly.bn1.bias.data.clone(), requires_grad=False
        )
        self.register_buffer(
            "anomaly_bn1_running_mean", full_model.anomaly.bn1.running_mean.clone()
        )
        self.register_buffer(
            "anomaly_bn1_running_var", full_model.anomaly.bn1.running_var.clone()
        )
        self.anomaly_fc2_weight = nn.Parameter(
            full_model.anomaly.fc2.weight.data.clone(), requires_grad=False
        )
        self.anomaly_fc2_bias = nn.Parameter(
            full_model.anomaly.fc2.bias.data.clone(), requires_grad=False
        )
        self.register_buffer("anomaly_threshold", full_model.anomaly.threshold.clone())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone.forward_features(x)
        gwap_out = self.gwap(feat)  # (B, backbone_dim)
        raw_emb = self.head(gwap_out)  # (B, emb_size)
        emb_norm = F.normalize(raw_emb, dim=1)
        logits = F.linear(emb_norm, self.arc_weight_normalized) * self.arc_s

        # AnomalyClassifier の forward を展開（BN を eval モードで実行）
        h = F.linear(gwap_out, self.anomaly_fc1_weight, self.anomaly_fc1_bias)
        h = F.batch_norm(
            h,
            self.anomaly_bn1_running_mean,
            self.anomaly_bn1_running_var,
            self.anomaly_bn1_weight,
            self.anomaly_bn1_bias,
            training=False,
            eps=1e-5,
        )
        h = torch.relu(h)
        logit = F.linear(h, self.anomaly_fc2_weight, self.anomaly_fc2_bias).squeeze(1)
        anomaly_score = torch.sigmoid(logit)  # (B,)
        return logits, anomaly_score


# ──────────────────────────────────────────────
# Checkpoint loading
# ──────────────────────────────────────────────
def _detect_num_classes(state_dict: dict, num_classes_arg: int | None) -> int:
    has_arc_weight = "arc_weight" in state_dict
    if num_classes_arg is None:
        if not has_arc_weight:
            print("Error: --num-classes not specified and arc_weight not found")
            sys.exit(1)
        num_classes = state_dict["arc_weight"].shape[0]
        print(f"  Auto-detected num_classes={num_classes} from arc_weight shape")
        return num_classes
    if has_arc_weight:
        ckpt_nc = state_dict["arc_weight"].shape[0]
        if num_classes_arg != ckpt_nc:
            print(
                f"  [WARN] --num-classes={num_classes_arg} but "
                f"arc_weight has {ckpt_nc} classes"
            )
    return num_classes_arg


def _detect_anomaly_classifier(state_dict: dict) -> bool:
    """AnomalyClassifier がフィット済みかどうかを検出する。

    anomaly.threshold が有限値であれば fit 済みとみなす。
    """
    key = "anomaly.threshold"
    if key not in state_dict:
        return False
    val = float(state_dict[key])
    return not (val == float("inf") or math.isnan(val))


def load_model_for_export(
    checkpoint_path: str,
    num_classes: int | None,
    run_cfg: RunConfig,
) -> tuple[ClassificationModel | ClassificationWithAnomalyModel, bool]:
    """Load checkpoint and return inference-ready export wrapper.

    Returns ``(model, has_anomaly)``.
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    detected_nc = _detect_num_classes(state_dict, num_classes)
    has_anomaly = _detect_anomaly_classifier(state_dict)

    full_model = FaceRecognitionModel(
        backbone_name=run_cfg.backbone,
        backbone_dim=run_cfg.backbone_dim,
        hidden_dim=run_cfg.hidden_dim,
        emb_size=run_cfg.emb_size,
        num_classes=detected_nc,
        dropout=run_cfg.dropout,
        arc_s=run_cfg.arc_s,
        arc_m=run_cfg.arc_m,
    )
    full_model.load_state_dict(state_dict)
    full_model.eval()

    if has_anomaly:
        threshold = float(state_dict["anomaly.threshold"])
        print(
            f"  AnomalyClassifier detected (threshold={threshold:.4f}) "
            f"→ ClassificationWithAnomalyModel"
        )
        export_model: ClassificationModel | ClassificationWithAnomalyModel = (
            ClassificationWithAnomalyModel(full_model)
        )
    else:
        print("  No AnomalyClassifier → ClassificationModel (logits only)")
        export_model = ClassificationModel(full_model)

    export_model.eval()
    return export_model, has_anomaly


# ──────────────────────────────────────────────
# Export: FP32 (base model)
# ──────────────────────────────────────────────
def export_fp32_onnx(
    cls_model: ClassificationModel | ClassificationWithAnomalyModel,
    onnx_path: str,
    opset: int,
    *,
    img_cfg: ImageConfig,
    num_classes: int,
    class_names: list[str],
    has_anomaly: bool = False,
) -> None:
    """Export PyTorch model to fp32 ONNX with dynamic batch and metadata."""
    print(f"Exporting fp32 ONNX to: {onnx_path}")
    dummy_input = torch.randn(1, 3, img_cfg.input_size, img_cfg.input_size)

    if has_anomaly:
        output_names = ["logits", "anomaly_score"]
        dynamic_axes = {
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "anomaly_score": {0: "batch_size"},
        }
    else:
        output_names = ["logits"]
        dynamic_axes = {
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        }

    torch.onnx.export(
        cls_model,
        dummy_input,
        onnx_path,
        opset_version=opset,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
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
    fixed_vi = make_intermediate_batch_dims_dynamic(onnx_model)
    if fixed_vi:
        print(f"  Fixed {fixed_vi} intermediate value_info batch dims to dynamic")
    onnx.save(onnx_model, onnx_path)

    # Verify dynamic batch
    print("  Verifying dynamic batch...")
    verify_dynamic_batch(
        onnx_path,
        input_size=img_cfg.input_size,
        num_classes=num_classes,
        rng=np.random.default_rng(0),
        label="fp32",
        has_anomaly=has_anomaly,
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"  fp32 ONNX model verified (opset={opset})")

    # Embed metadata
    embed_class_metadata(onnx_model, class_names)
    img_size_entry = onnx_model.metadata_props.add()
    img_size_entry.key = "input_size"
    img_size_entry.value = str(img_cfg.input_size)
    mean_entry = onnx_model.metadata_props.add()
    mean_entry.key = "imagenet_mean"
    mean_entry.value = json.dumps(img_cfg.mean)
    std_entry = onnx_model.metadata_props.add()
    std_entry.key = "imagenet_std"
    std_entry.value = json.dumps(img_cfg.std)

    if has_anomaly and isinstance(cls_model, ClassificationWithAnomalyModel):
        threshold_val = float(cls_model.anomaly_threshold.item())
        entry = onnx_model.metadata_props.add()
        entry.key = "anomaly_threshold"
        entry.value = str(threshold_val)
        print(f"  Embedded anomaly_threshold={threshold_val:.6f}")

    onnx.save(onnx_model, onnx_path)
    print(
        f"  Embedded {len(class_names)} class names, "
        f"input_size={img_cfg.input_size}, mean/std in metadata_props"
    )

    # Verify output vs PyTorch
    print("  Verifying fp32 ONNX output vs PyTorch (real image)...")
    real_sample = get_real_sample(img_cfg)
    real_tensor = torch.from_numpy(real_sample)
    sess = ort.InferenceSession(onnx_path)
    with torch.no_grad():
        pt_outputs = cls_model(real_tensor)
    if has_anomaly:
        pt_logits, pt_anomaly = pt_outputs
        ort_logits, ort_anomaly = sess.run(None, {"input": real_sample})
        compare_outputs(pt_logits.numpy(), ort_logits, label="PyTorch vs fp32 (logits)")
        compare_outputs(
            pt_anomaly.numpy(), ort_anomaly, label="PyTorch vs fp32 (anomaly)"
        )
    else:
        pt_logits = pt_outputs
        ort_logits = sess.run(None, {"input": real_sample})[0]
        compare_outputs(pt_logits.numpy(), ort_logits, label="PyTorch vs fp32")


# ──────────────────────────────────────────────
# Export: BF16 (graph-wide)
# ──────────────────────────────────────────────
def export_bf16_onnx(
    fp32_path: str,
    bf16_path: str,
    *,
    img_cfg: ImageConfig,
    num_classes: int,
    class_names: list[str],
    has_anomaly: bool = False,
) -> None:
    """Convert fp32 graph to bf16 (weights + activations via Cast)."""
    print(f"\nExporting bf16 ONNX to: {bf16_path}")
    fp32_model = onnx.load(fp32_path)

    bf16_model = convert_graph_to_bf16(fp32_model, has_anomaly=has_anomaly)
    embed_class_metadata(bf16_model, class_names)
    onnx.save(bf16_model, bf16_path)

    print("  Verifying bf16 dynamic batch...")
    verify_dynamic_batch(
        bf16_path,
        input_size=img_cfg.input_size,
        num_classes=num_classes,
        rng=np.random.default_rng(3),
        label="bf16",
        has_anomaly=has_anomaly,
    )
    print(f"  bf16 ONNX saved: {bf16_path}")


# ──────────────────────────────────────────────
# Export: FP16 (graph-wide)
# ──────────────────────────────────────────────
def export_fp16_onnx(
    fp32_path: str,
    fp16_path: str,
    *,
    img_cfg: ImageConfig,
    num_classes: int,
    class_names: list[str],
    has_anomaly: bool = False,
) -> None:
    """Convert fp32 graph to fp16 (weights + activations)."""
    print(f"\nExporting fp16 ONNX to: {fp16_path}")
    fp32_model = onnx.load(fp32_path)

    fp16_model = convert_graph_to_fp16(fp32_model, has_anomaly=has_anomaly)
    embed_class_metadata(fp16_model, class_names)
    onnx.save(fp16_model, fp16_path)

    print("  Verifying fp16 dynamic batch...")
    verify_dynamic_batch(
        fp16_path,
        input_size=img_cfg.input_size,
        num_classes=num_classes,
        rng=np.random.default_rng(2),
        label="fp16",
        has_anomaly=has_anomaly,
    )
    print(f"  fp16 ONNX saved: {fp16_path}")


# ──────────────────────────────────────────────
# Export: block-quantized models (INT8 / FP8)
# ──────────────────────────────────────────────
def export_block_quantized_onnx(
    base_reduced_path: str,
    fp32_path: str,
    output_path: str,
    quant_dtype: QuantDtype,
    scale_dtype: TargetDtype,
    quantize_names: list[str],
    *,
    block_size: int,
    img_cfg: ImageConfig,
    num_classes: int,
    class_names: list[str],
    has_anomaly: bool = False,
) -> None:
    """Apply per-block quantization to a bf16/fp16 base model.

    Args:
        base_reduced_path: Path to the bf16 or fp16 ONNX model.
        fp32_path: Path to the fp32 ONNX model (for metadata / verification).
        output_path: Output path for the quantized model.
        quant_dtype: INT8 or FP8 variant.
        scale_dtype: BF16 or FP16 for scale storage.
        quantize_names: Initializer names to quantize (from sensitivity detection).
        block_size: Elements per quantization block.
        img_cfg: Image preprocessing config.
        num_classes: Number of classes.
        class_names: Class label names.
        has_anomaly: Whether model has AnomalyClassifier.
    """
    label = f"{scale_dtype.value}+{quant_dtype.value}"
    print(f"\nExporting {label} ONNX to: {output_path}")
    print(
        f"  Quantizing {len(quantize_names)} weight initializers "
        f"(block_size={block_size})"
    )

    model = onnx.load(base_reduced_path)

    # AnomalyClassifier initializers must never be quantized
    anomaly_exclude = get_anomaly_initializer_names(model) if has_anomaly else set()

    count = apply_block_quantization(
        model,
        quant_dtype,
        scale_dtype,
        block_size=block_size,
        exclude_names=anomaly_exclude,
        quantize_names=quantize_names,
        # BF16 models: our custom bf16 conversion stores weights as bf16 but
        # inserts Cast(bf16→fp32) nodes so computation runs in fp32.  The
        # dequant subgraph must therefore also output fp32.
        # FP16 models: onnxconverter_common rewrites ops to fp16 natively,
        # so computation runs in fp16 and the dequant subgraph should match.
        compute_dtype_is_fp32=(scale_dtype == TargetDtype.BF16),
    )
    print(f"  Applied per-block {quant_dtype.value} to {count} initializers")

    # Shape inference for TensorRT compatibility
    print(f"  Running shape inference...")
    model = infer_shapes_for_tensorrt(model)

    # shape inference uses batch=1, so intermediate value_info may have
    # dim_value=1 hardcoded; make them dynamic to suppress ORT warnings
    fixed_vi = make_intermediate_batch_dims_dynamic(model)
    if fixed_vi:
        print(f"  Fixed {fixed_vi} intermediate value_info batch dims to dynamic")

    embed_class_metadata(model, class_names)
    onnx.save(model, output_path)

    # Verify
    print(f"  Verifying {label} dynamic batch...")
    verify_dynamic_batch(
        output_path,
        input_size=img_cfg.input_size,
        num_classes=num_classes,
        rng=np.random.default_rng(4),
        label=label,
        has_anomaly=has_anomaly,
    )

    # Quick output quality check
    print(f"  Verifying {label} output vs fp32 (real image, logits)...")
    real_sample = get_real_sample(img_cfg)
    sess_fp32 = ort.InferenceSession(fp32_path)
    sess_q = ort.InferenceSession(output_path)
    fp32_out = sess_fp32.run(None, {"input": real_sample})[0]
    q_out = sess_q.run(None, {"input": real_sample})[0]
    compare_outputs(fp32_out, q_out, label=f"fp32 vs {label}", warn_threshold=0.05)

    print(f"  {label} ONNX saved: {output_path}")


# ──────────────────────────────────────────────
# Sensitivity detection wrapper
# ──────────────────────────────────────────────
def run_sensitivity_detection(
    fp32_path: str,
    quant_dtype: QuantDtype,
    scale_dtype: TargetDtype,
    *,
    block_size: int,
    nrmse_threshold: float,
    output_diff_threshold: float,
    img_cfg: ImageConfig,
    num_probe_samples: int,
    has_anomaly: bool = False,
) -> list[str]:
    """Run 2-stage sensitivity detection and return names safe to quantize.

    Args:
        fp32_path: Path to the fp32 ONNX model.
        quant_dtype: Target quantization dtype.
        scale_dtype: BF16 or FP16 for scale storage.
        block_size: Block size.
        nrmse_threshold: Stage 1 threshold.
        output_diff_threshold: Stage 2 threshold.
        img_cfg: Image config for loading probe samples.
        num_probe_samples: Number of probe samples for Stage 2.
        has_anomaly: Whether model has AnomalyClassifier.

    Returns:
        List of initializer names that are safe to quantize.
    """
    # Load probe samples for Stage 2
    probe_samples = load_eval_samples(img_cfg, num_probe_samples, calib_seed=99)

    # AnomalyClassifier initializers are always excluded
    fp32_model = onnx.load(fp32_path)
    anomaly_exclude = get_anomaly_initializer_names(fp32_model) if has_anomaly else set()

    quantize_names, sensitive_names = find_sensitive_initializers(
        fp32_path,
        quant_dtype,
        scale_dtype,
        block_size=block_size,
        exclude_names=anomaly_exclude,
        nrmse_threshold=nrmse_threshold,
        output_diff_threshold=output_diff_threshold,
        probe_samples=probe_samples,
    )

    return quantize_names


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export .pt checkpoint to 7 ONNX variants: "
            "fp32 / bf16 / fp16 / bf16+INT8 / bf16+FP8 / fp16+INT8 / fp16+FP8"
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a .pt checkpoint (e.g. model_best_with_anomaly.pt)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of classes (auto-detected from checkpoint if omitted)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET,
        help=f"ONNX opset version (default: {DEFAULT_OPSET}, >=19 for FP8)",
    )
    # Block quantization parameters
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=f"Block size for per-block quantization (default: {DEFAULT_BLOCK_SIZE})",
    )
    parser.add_argument(
        "--fp8-format",
        type=str,
        default=DEFAULT_FP8_FORMAT,
        choices=list(_FP8_FORMAT_MAP.keys()),
        help=f"FP8 format (default: {DEFAULT_FP8_FORMAT})",
    )
    # Sensitivity detection parameters
    parser.add_argument(
        "--nrmse-threshold",
        type=float,
        default=DEFAULT_WEIGHT_NRMSE_THRESHOLD,
        help=(
            f"Stage 1 NRMSE threshold for sensitivity detection "
            f"(default: {DEFAULT_WEIGHT_NRMSE_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--output-diff-threshold",
        type=float,
        default=DEFAULT_OUTPUT_DIFF_THRESHOLD,
        help=(
            f"Stage 2 max abs diff threshold for sensitivity detection "
            f"(default: {DEFAULT_OUTPUT_DIFF_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--sensitivity-samples",
        type=int,
        default=DEFAULT_SENSITIVITY_SAMPLES,
        help=(
            f"Number of probe samples for Stage 2 sensitivity detection "
            f"(default: {DEFAULT_SENSITIVITY_SAMPLES})"
        ),
    )
    # Evaluation
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=DEFAULT_EVAL_SAMPLES,
        help=f"Number of samples for quality evaluation (default: {DEFAULT_EVAL_SAMPLES})",
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

    # Load architecture constants
    print("Loading run config from saved script next to checkpoint...")
    run_cfg = load_run_config(checkpoint_path)
    print(
        f"  backbone={run_cfg.backbone}  input_size={run_cfg.input_size}  "
        f"emb_size={run_cfg.emb_size}  backbone_dim={run_cfg.backbone_dim}"
    )

    img_cfg = ImageConfig(
        input_size=run_cfg.input_size,
        mean=run_cfg.imagenet_mean,
        std=run_cfg.imagenet_std,
    )

    out_dir = os.path.dirname(checkpoint_path) or "."
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]

    paths = {
        "fp32": os.path.join(out_dir, f"{stem}.onnx"),
        "bf16": os.path.join(out_dir, f"{stem}_bf16.onnx"),
        "fp16": os.path.join(out_dir, f"{stem}_fp16.onnx"),
        "bf16int8": os.path.join(out_dir, f"{stem}_bf16int8.onnx"),
        "bf16fp8": os.path.join(out_dir, f"{stem}_bf16fp8.onnx"),
        "fp16int8": os.path.join(out_dir, f"{stem}_fp16int8.onnx"),
        "fp16fp8": os.path.join(out_dir, f"{stem}_fp16fp8.onnx"),
    }

    fp8_quant_dtype = _FP8_FORMAT_MAP[args.fp8_format]
    block_size = args.block_size

    # ── Step 1: Load class names ──────────────────────────────────
    print("Loading class names from dataset...")
    class_names = get_class_names_from_dataset()
    num_classes = len(class_names)
    print(f"  {num_classes} class names loaded")
    meta = dict(num_classes=num_classes, class_names=class_names)

    # ── Step 2: Load model ────────────────────────────────────────
    print(f"Loading checkpoint: {checkpoint_path}")
    cls_model, has_anomaly = load_model_for_export(
        checkpoint_path, args.num_classes, run_cfg
    )

    # ── Step 3: FP32 ONNX (base) ─────────────────────────────────
    export_fp32_onnx(
        cls_model,
        paths["fp32"],
        opset=args.opset,
        img_cfg=img_cfg,
        has_anomaly=has_anomaly,
        **meta,
    )

    # ── Step 4: BF16 ──────────────────────────────────────────────
    export_bf16_onnx(
        paths["fp32"],
        paths["bf16"],
        img_cfg=img_cfg,
        has_anomaly=has_anomaly,
        **meta,
    )

    # ── Step 5: FP16 ──────────────────────────────────────────────
    export_fp16_onnx(
        paths["fp32"],
        paths["fp16"],
        img_cfg=img_cfg,
        has_anomaly=has_anomaly,
        **meta,
    )

    # ── Step 6: Sensitivity detection ─────────────────────────────
    # Run separately for INT8 and FP8 since they have different error profiles
    print("\n" + "=" * 60)
    print("Sensitivity detection for INT8")
    print("=" * 60)
    int8_quantize_names = run_sensitivity_detection(
        paths["fp32"],
        QuantDtype.INT8,
        TargetDtype.BF16,  # scale dtype doesn't affect weight error
        block_size=block_size,
        nrmse_threshold=args.nrmse_threshold,
        output_diff_threshold=args.output_diff_threshold,
        img_cfg=img_cfg,
        num_probe_samples=args.sensitivity_samples,
        has_anomaly=has_anomaly,
    )

    print("\n" + "=" * 60)
    print("Sensitivity detection for FP8")
    print("=" * 60)
    fp8_quantize_names = run_sensitivity_detection(
        paths["fp32"],
        fp8_quant_dtype,
        TargetDtype.BF16,
        block_size=block_size,
        nrmse_threshold=args.nrmse_threshold,
        output_diff_threshold=args.output_diff_threshold,
        img_cfg=img_cfg,
        num_probe_samples=args.sensitivity_samples,
        has_anomaly=has_anomaly,
    )

    # ── Step 7: BF16 + INT8 ──────────────────────────────────────
    export_block_quantized_onnx(
        paths["bf16"],
        paths["fp32"],
        paths["bf16int8"],
        QuantDtype.INT8,
        TargetDtype.BF16,
        int8_quantize_names,
        block_size=block_size,
        img_cfg=img_cfg,
        has_anomaly=has_anomaly,
        **meta,
    )

    # ── Step 8: BF16 + FP8 ───────────────────────────────────────
    export_block_quantized_onnx(
        paths["bf16"],
        paths["fp32"],
        paths["bf16fp8"],
        fp8_quant_dtype,
        TargetDtype.BF16,
        fp8_quantize_names,
        block_size=block_size,
        img_cfg=img_cfg,
        has_anomaly=has_anomaly,
        **meta,
    )

    # ── Step 9: FP16 + INT8 ──────────────────────────────────────
    export_block_quantized_onnx(
        paths["fp16"],
        paths["fp32"],
        paths["fp16int8"],
        QuantDtype.INT8,
        TargetDtype.FP16,
        int8_quantize_names,
        block_size=block_size,
        img_cfg=img_cfg,
        has_anomaly=has_anomaly,
        **meta,
    )

    # ── Step 10: FP16 + FP8 ──────────────────────────────────────
    export_block_quantized_onnx(
        paths["fp16"],
        paths["fp32"],
        paths["fp16fp8"],
        fp8_quant_dtype,
        TargetDtype.FP16,
        fp8_quantize_names,
        block_size=block_size,
        img_cfg=img_cfg,
        has_anomaly=has_anomaly,
        **meta,
    )

    # ── Step 11: Quality evaluation ──────────────────────────────
    for label in ("bf16", "fp16", "bf16int8", "bf16fp8", "fp16int8", "fp16fp8"):
        evaluate_model_quality(
            paths["fp32"],
            paths[label],
            img_cfg=img_cfg,
            num_samples=args.eval_samples,
            label=label,
        )

    # ── Summary ───────────────────────────────────────────────────
    print_file_sizes(paths)
    print("\nDone.")


if __name__ == "__main__":
    main()

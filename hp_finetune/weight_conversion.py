"""ONNX weight dtype conversion and per-block quantization utilities.

Provides:

1. **Full-model dtype conversion** (BF16 / FP16)
   - Convert all fp32 initializers to bf16 or fp16
   - Insert Cast nodes so that consumer ops receive the correct dtype
   - Optionally exclude quantization parameters, I/O, and small tensors

2. **Per-block quantization** (INT8 / FP8)
   - Block-FP8 style: split weight tensor into fixed-size blocks (default 32)
   - Each block has its own absmax scale
   - Quantized weights + scales stored as ONNX initializers
   - Restoration sub-graph uses only standard nodes (Cast, Mul, Reshape, Slice)
   - No DequantizeLinear — fully TensorRT compatible

3. **Sensitivity detection** (2-stage hybrid)
   - Stage 1: weight reconstruction error (NRMSE) — fast, no inference
   - Stage 2: output difference — accurate, runs inference on suspect nodes
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import torch

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
# Minimum element count for an initializer to be worth converting / quantizing.
# Small tensors (bias, BN params) stay in their base dtype for precision.
WEIGHT_MIN_ELEMENTS = 1024

# Per-block quantization defaults
DEFAULT_BLOCK_SIZE = 32
DEFAULT_FP8_FORMAT = "e4m3fn"

# Maximum representable values for quantization dtypes
INT8_MAX_VAL = 127.0
FP8E4M3_MAX_VAL = 448.0
FP8E5M2_MAX_VAL = 57344.0

# Sensitivity detection defaults
DEFAULT_WEIGHT_NRMSE_THRESHOLD = 0.02
DEFAULT_OUTPUT_DIFF_THRESHOLD = 0.5
DEFAULT_SENSITIVITY_SAMPLES = 8


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────
class TargetDtype(Enum):
    """Supported reduced-precision target dtypes for full-model conversion."""

    FP16 = "fp16"
    BF16 = "bf16"


class QuantDtype(Enum):
    """Supported quantization dtypes for per-block weight quantization."""

    INT8 = "int8"
    FP8E4M3 = "fp8e4m3"
    FP8E5M2 = "fp8e5m2"


# ──────────────────────────────────────────────
# Full-model dtype conversion (BF16 / FP16)
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class _DtypeConversionSpec:
    """Describes how to convert an fp32 initializer to a reduced dtype."""

    onnx_dtype: int  # onnx.TensorProto data type enum
    cast_suffix: str  # suffix for the Cast output name
    cast_node_prefix: str  # prefix for the Cast node name
    convert_array: Callable[[np.ndarray], onnx.TensorProto]
    can_overflow: bool  # whether the target dtype can overflow fp32 values


class _OverflowError(Exception):
    """Raised when fp16 conversion would produce inf/nan."""


def _make_fp16_tensor(arr_fp32: np.ndarray) -> onnx.TensorProto:
    """Convert fp32 numpy array to fp16 ONNX TensorProto."""
    arr_fp16 = arr_fp32.astype(np.float16)
    if np.any(np.isinf(arr_fp16)) or np.any(np.isnan(arr_fp16)):
        raise _OverflowError
    return onnx.numpy_helper.from_array(arr_fp16)


def _make_bf16_tensor(arr_fp32: np.ndarray) -> onnx.TensorProto:
    """Convert fp32 numpy array to bf16 ONNX TensorProto (via PyTorch)."""
    t_bf16 = torch.from_numpy(arr_fp32.copy()).bfloat16()
    raw_int16 = t_bf16.view(torch.int16).numpy()
    tp = onnx.TensorProto()
    tp.data_type = onnx.TensorProto.BFLOAT16
    tp.dims.extend(arr_fp32.shape)
    tp.raw_data = raw_int16.tobytes()
    return tp


_DTYPE_SPECS: dict[TargetDtype, _DtypeConversionSpec] = {
    TargetDtype.FP16: _DtypeConversionSpec(
        onnx_dtype=onnx.TensorProto.FLOAT16,
        cast_suffix="_fp32",
        cast_node_prefix="Cast_fp16_to_fp32_",
        convert_array=_make_fp16_tensor,
        can_overflow=True,
    ),
    TargetDtype.BF16: _DtypeConversionSpec(
        onnx_dtype=onnx.TensorProto.BFLOAT16,
        cast_suffix="_fp32_from_bf16",
        cast_node_prefix="Cast_bf16_to_fp32_",
        convert_array=_make_bf16_tensor,
        can_overflow=False,
    ),
}


def _collect_quant_param_names(model: onnx.ModelProto) -> set[str]:
    """Return names of quantization scale/zero-point initializers."""
    names: set[str] = set()
    for node in model.graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
            for i in range(1, len(node.input)):
                names.add(node.input[i])
    return names


def _insert_cast_nodes_and_rewire(
    model: onnx.ModelProto,
    converted_names: list[str],
    spec: _DtypeConversionSpec,
) -> None:
    """Insert Cast nodes (reduced_dtype -> fp32) and redirect consumer inputs."""
    cast_nodes: list[onnx.NodeProto] = []
    for orig_name in converted_names:
        cast_output = orig_name + spec.cast_suffix
        cast_node = onnx.helper.make_node(
            "Cast",
            inputs=[orig_name],
            outputs=[cast_output],
            to=onnx.TensorProto.FLOAT,
            name=f"{spec.cast_node_prefix}{orig_name}",
        )
        cast_nodes.append(cast_node)

        for node in model.graph.node:
            for i, inp in enumerate(node.input):
                if inp == orig_name:
                    node.input[i] = cast_output

    for i, cast_node in enumerate(cast_nodes):
        model.graph.node.insert(i, cast_node)


def convert_initializers(
    model: onnx.ModelProto,
    target: TargetDtype,
    *,
    min_elements: int = WEIGHT_MIN_ELEMENTS,
    exclude_quant_params: bool = True,
    exclude_io: bool = True,
    exclude_names: set[str] | None = None,
) -> int:
    """Convert fp32 initializers to *target* dtype in-place.

    For full-model conversion, set ``min_elements=0`` and
    ``exclude_quant_params=False``.

    Args:
        model: ONNX model to modify in-place.
        target: Target reduced dtype (FP16 or BF16).
        min_elements: Skip initializers smaller than this.
        exclude_quant_params: If True, skip quantization scale/zero-point.
        exclude_io: If True, skip initializers that are also graph I/O names.
        exclude_names: Additional initializer names to exclude.

    Returns:
        Number of initializers converted.
    """
    spec = _DTYPE_SPECS[target]

    excluded: set[str] = set()
    if exclude_quant_params:
        excluded |= _collect_quant_param_names(model)
    if exclude_io:
        for t in list(model.graph.input) + list(model.graph.output):
            excluded.add(t.name)
    if exclude_names:
        excluded |= exclude_names

    converted_names: list[str] = []
    for init in model.graph.initializer:
        if init.data_type != onnx.TensorProto.FLOAT:
            continue
        if init.name in excluded:
            continue
        numel = int(np.prod(init.dims)) if init.dims else 1
        if numel < min_elements:
            continue
        converted_names.append(init.name)

    if not converted_names:
        return 0

    actually_converted: list[str] = []
    for init in model.graph.initializer:
        if init.name not in converted_names:
            continue
        arr_fp32 = onnx.numpy_helper.to_array(init)
        try:
            new_tensor = spec.convert_array(arr_fp32)
        except _OverflowError:
            continue
        new_tensor.name = init.name
        init.CopyFrom(new_tensor)
        actually_converted.append(init.name)

    if not actually_converted:
        return 0

    _insert_cast_nodes_and_rewire(model, actually_converted, spec)
    return len(actually_converted)


# ──────────────────────────────────────────────
# Per-block quantization engine
# ──────────────────────────────────────────────
def _get_quant_max_val(quant_dtype: QuantDtype) -> float:
    """Return the maximum representable value for a quantization dtype."""
    return {
        QuantDtype.INT8: INT8_MAX_VAL,
        QuantDtype.FP8E4M3: FP8E4M3_MAX_VAL,
        QuantDtype.FP8E5M2: FP8E5M2_MAX_VAL,
    }[quant_dtype]


def _get_onnx_quant_dtype(quant_dtype: QuantDtype) -> int:
    """Return the ONNX TensorProto data type enum for a quantization dtype."""
    return {
        QuantDtype.INT8: onnx.TensorProto.INT8,
        QuantDtype.FP8E4M3: onnx.TensorProto.FLOAT8E4M3FN,
        QuantDtype.FP8E5M2: onnx.TensorProto.FLOAT8E5M2,
    }[quant_dtype]


@dataclass
class BlockQuantizedWeight:
    """Result of per-block quantization of a single weight tensor.

    Attributes:
        quantized: Quantized weight data, shape ``(num_blocks, block_size)``.
        scales: Per-block scale factors, shape ``(num_blocks, 1)``.
        original_shape: Original weight tensor shape (tuple of ints).
        original_numel: Number of elements in the original tensor (before padding).
        padded_numel: Number of elements after padding to block boundary.
        block_size: Block size used.
        quant_dtype: Quantization dtype used.
    """

    quantized: np.ndarray
    scales: np.ndarray
    original_shape: tuple[int, ...]
    original_numel: int
    padded_numel: int
    block_size: int
    quant_dtype: QuantDtype


def quantize_weight_per_block(
    arr_fp32: np.ndarray,
    quant_dtype: QuantDtype,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> BlockQuantizedWeight:
    """Quantize a weight tensor using per-block absmax scaling.

    Steps:
        1. Flatten the tensor and pad to a multiple of block_size.
        2. Reshape to ``(num_blocks, block_size)``.
        3. Compute absmax per block → scale = absmax / max_representable_value.
        4. Quantize: ``q = clamp(round(w / scale), -max_val, max_val)``.
        5. Return quantized data, scales, and metadata.

    Args:
        arr_fp32: Weight tensor as float32 numpy array (any shape).
        quant_dtype: Target quantization dtype (INT8 or FP8).
        block_size: Number of elements per block.

    Returns:
        A :class:`BlockQuantizedWeight` containing quantized data and metadata.
    """
    original_shape = arr_fp32.shape
    original_numel = arr_fp32.size
    max_val = _get_quant_max_val(quant_dtype)

    # Step 1: flatten and pad
    flat = arr_fp32.flatten().astype(np.float64)
    pad_len = (block_size - (original_numel % block_size)) % block_size
    if pad_len > 0:
        flat = np.concatenate([flat, np.zeros(pad_len, dtype=np.float64)])
    padded_numel = flat.size

    # Step 2: reshape to (num_blocks, block_size)
    blocks = flat.reshape(-1, block_size)

    # Step 3: per-block absmax scale
    absmax = np.abs(blocks).max(axis=1, keepdims=True)  # (num_blocks, 1)
    # Avoid division by zero: if absmax is 0, scale is 1 (quantized values will be 0)
    scales = np.where(absmax == 0, 1.0, absmax / max_val)  # (num_blocks, 1)

    # Step 4: quantize
    scaled = blocks / scales  # (num_blocks, block_size)

    if quant_dtype == QuantDtype.INT8:
        quantized = np.clip(np.round(scaled), -128, 127).astype(np.int8)
    else:
        # FP8: clamp to representable range, then cast
        # We store as float32 and will convert to FP8 ONNX dtype when creating the tensor
        quantized = np.clip(scaled, -max_val, max_val).astype(np.float32)

    scales_f32 = scales.astype(np.float32)

    return BlockQuantizedWeight(
        quantized=quantized,
        scales=scales_f32,
        original_shape=original_shape,
        original_numel=original_numel,
        padded_numel=padded_numel,
        block_size=block_size,
        quant_dtype=quant_dtype,
    )


def dequantize_weight_per_block(bqw: BlockQuantizedWeight) -> np.ndarray:
    """Restore a block-quantized weight tensor to fp32 (for accuracy evaluation).

    This performs the same computation that the ONNX sub-graph will do at
    runtime: ``cast(quantized) * scales → reshape → slice → reshape``.

    Args:
        bqw: A :class:`BlockQuantizedWeight` from :func:`quantize_weight_per_block`.

    Returns:
        Restored fp32 numpy array with the same shape as the original tensor.
    """
    # Cast quantized to float
    q_float = bqw.quantized.astype(np.float64)
    # Mul by scale
    restored = (q_float * bqw.scales).flatten()
    # Slice to remove padding
    restored = restored[: bqw.original_numel]
    # Reshape to original shape
    return restored.reshape(bqw.original_shape).astype(np.float32)


def compute_weight_nrmse(arr_fp32: np.ndarray, arr_restored: np.ndarray) -> float:
    """Compute Normalized Root Mean Square Error between original and restored.

    NRMSE = sqrt(mean((orig - restored)²)) / sqrt(mean(orig²))

    Returns 0.0 if the original tensor is all zeros.
    """
    diff = arr_fp32.astype(np.float64) - arr_restored.astype(np.float64)
    mse = np.mean(diff**2)
    orig_energy = np.mean(arr_fp32.astype(np.float64) ** 2)
    if orig_energy == 0:
        return 0.0
    return float(np.sqrt(mse / orig_energy))


# ──────────────────────────────────────────────
# ONNX initializer creation for block-quantized weights
# ──────────────────────────────────────────────
def _make_int8_initializer(arr_int8: np.ndarray, name: str) -> onnx.TensorProto:
    """Create an INT8 ONNX TensorProto initializer."""
    return onnx.numpy_helper.from_array(arr_int8, name=name)


def _make_fp8_initializer(
    arr_fp32: np.ndarray,
    name: str,
    quant_dtype: QuantDtype,
) -> onnx.TensorProto:
    """Create an FP8 ONNX TensorProto initializer.

    ONNX stores FP8 as raw bytes. We use PyTorch to convert fp32 → fp8
    since numpy doesn't support fp8 natively.
    """
    onnx_dtype = _get_onnx_quant_dtype(quant_dtype)

    if quant_dtype == QuantDtype.FP8E4M3:
        torch_dtype = torch.float8_e4m3fn
    elif quant_dtype == QuantDtype.FP8E5M2:
        torch_dtype = torch.float8_e5m2
    else:
        raise ValueError(f"Not an FP8 dtype: {quant_dtype}")

    t_fp8 = torch.from_numpy(arr_fp32.copy()).to(torch_dtype)
    raw_bytes = t_fp8.view(torch.uint8).numpy().tobytes()

    tp = onnx.TensorProto()
    tp.name = name
    tp.data_type = onnx_dtype
    tp.dims.extend(arr_fp32.shape)
    tp.raw_data = raw_bytes
    return tp


def _make_scale_initializer(
    scales: np.ndarray,
    name: str,
    scale_dtype: TargetDtype,
) -> onnx.TensorProto:
    """Create an initializer for per-block scales in bf16 or fp16.

    Args:
        scales: Scale factors, shape ``(num_blocks, 1)``, float32.
        name: ONNX initializer name.
        scale_dtype: BF16 or FP16.

    Returns:
        ONNX TensorProto in the specified dtype.
    """
    spec = _DTYPE_SPECS[scale_dtype]
    tp = spec.convert_array(scales)
    tp.name = name
    return tp


# ──────────────────────────────────────────────
# ONNX sub-graph construction for block-quantized weight restoration
# ──────────────────────────────────────────────
def build_block_dequant_subgraph(
    bqw: BlockQuantizedWeight,
    orig_init_name: str,
    scale_dtype: TargetDtype,
    *,
    compute_dtype_is_fp32: bool = False,
) -> tuple[
    list[onnx.TensorProto],  # new initializers
    list[onnx.NodeProto],  # new nodes
    str,  # output tensor name (to rewire consumers)
]:
    """Build ONNX sub-graph that restores a block-quantized weight at runtime.

    The sub-graph uses only standard ONNX nodes (Cast, Mul, Reshape, Slice)
    so it is compatible with all runtimes including TensorRT.

    Sub-graph structure::

        weight_quantized [int8/fp8, (num_blocks, block_size)]
          → Cast(to=target_compute_dtype)
          → Mul(× scale [(num_blocks, 1)])
          → Reshape(to=[padded_numel])
          → Slice(start=0, end=original_numel)
          → Reshape(to=original_shape)
          → output (feeds into the original consumer node)

    When ``compute_dtype_is_fp32`` is True, Cast targets FLOAT and the scale
    initializer is stored in fp32 as well.  This is used for the FP32 base
    model where no dtype reduction is applied to activations.

    Args:
        bqw: Block quantization result.
        orig_init_name: Name of the original fp32 initializer being replaced.
        scale_dtype: BF16 or FP16 — dtype for the scale initializer.
        compute_dtype_is_fp32: If True, restore to fp32 instead of bf16/fp16.

    Returns:
        A 3-tuple: (new_initializers, new_nodes, output_tensor_name).
    """
    prefix = f"bq_{orig_init_name}"
    new_inits: list[onnx.TensorProto] = []
    new_nodes: list[onnx.NodeProto] = []

    num_blocks = bqw.quantized.shape[0]
    block_size = bqw.block_size

    # Determine compute dtype for Cast
    if compute_dtype_is_fp32:
        cast_to = onnx.TensorProto.FLOAT
    else:
        cast_to = _DTYPE_SPECS[scale_dtype].onnx_dtype

    # --- Initializer: quantized weights ---
    quant_init_name = f"{prefix}_qweight"
    if bqw.quant_dtype == QuantDtype.INT8:
        quant_init = _make_int8_initializer(bqw.quantized, quant_init_name)
    else:
        quant_init = _make_fp8_initializer(
            bqw.quantized, quant_init_name, bqw.quant_dtype
        )
    new_inits.append(quant_init)

    # --- Initializer: scales ---
    scale_init_name = f"{prefix}_scale"
    if compute_dtype_is_fp32:
        scale_init = onnx.numpy_helper.from_array(bqw.scales, name=scale_init_name)
    else:
        scale_init = _make_scale_initializer(bqw.scales, scale_init_name, scale_dtype)
    new_inits.append(scale_init)

    # --- Initializer: shape constants ---
    padded_flat_shape_name = f"{prefix}_padded_flat_shape"
    padded_flat_shape = onnx.numpy_helper.from_array(
        np.array([bqw.padded_numel], dtype=np.int64), name=padded_flat_shape_name
    )
    new_inits.append(padded_flat_shape)

    orig_shape_name = f"{prefix}_orig_shape"
    orig_shape = onnx.numpy_helper.from_array(
        np.array(list(bqw.original_shape), dtype=np.int64), name=orig_shape_name
    )
    new_inits.append(orig_shape)

    # Slice parameters (to remove padding)
    needs_slice = bqw.padded_numel > bqw.original_numel
    if needs_slice:
        slice_starts_name = f"{prefix}_slice_starts"
        slice_starts = onnx.numpy_helper.from_array(
            np.array([0], dtype=np.int64), name=slice_starts_name
        )
        new_inits.append(slice_starts)

        slice_ends_name = f"{prefix}_slice_ends"
        slice_ends = onnx.numpy_helper.from_array(
            np.array([bqw.original_numel], dtype=np.int64), name=slice_ends_name
        )
        new_inits.append(slice_ends)

        slice_axes_name = f"{prefix}_slice_axes"
        slice_axes = onnx.numpy_helper.from_array(
            np.array([0], dtype=np.int64), name=slice_axes_name
        )
        new_inits.append(slice_axes)

    # --- Node 1: Cast quantized weights to compute dtype ---
    cast_out_name = f"{prefix}_cast_out"
    cast_node = onnx.helper.make_node(
        "Cast",
        inputs=[quant_init_name],
        outputs=[cast_out_name],
        to=cast_to,
        name=f"{prefix}_Cast",
    )
    new_nodes.append(cast_node)

    # --- Node 1b: Cast scale to compute dtype (if scale stored in reduced dtype
    #     and compute is fp32, we need a Cast; if scale is already in compute dtype,
    #     skip) ---
    scale_for_mul = scale_init_name
    if not compute_dtype_is_fp32:
        # Scale is stored in bf16/fp16, same as cast_to → no extra Cast needed
        pass
    # If compute_dtype_is_fp32 and scale is already fp32 → no Cast needed either

    # --- Node 2: Mul (quantized_float * scale) ---
    mul_out_name = f"{prefix}_mul_out"
    mul_node = onnx.helper.make_node(
        "Mul",
        inputs=[cast_out_name, scale_for_mul],
        outputs=[mul_out_name],
        name=f"{prefix}_Mul",
    )
    new_nodes.append(mul_node)

    # --- Node 3: Reshape to flat ---
    reshape1_out_name = f"{prefix}_flat"
    reshape1_node = onnx.helper.make_node(
        "Reshape",
        inputs=[mul_out_name, padded_flat_shape_name],
        outputs=[reshape1_out_name],
        name=f"{prefix}_Reshape_flat",
    )
    new_nodes.append(reshape1_node)

    # --- Node 4: Slice to remove padding (if needed) ---
    if needs_slice:
        slice_out_name = f"{prefix}_sliced"
        slice_node = onnx.helper.make_node(
            "Slice",
            inputs=[
                reshape1_out_name,
                slice_starts_name,
                slice_ends_name,
                slice_axes_name,
            ],
            outputs=[slice_out_name],
            name=f"{prefix}_Slice",
        )
        new_nodes.append(slice_node)
        reshape2_input = slice_out_name
    else:
        reshape2_input = reshape1_out_name

    # --- Node 5: Reshape to original weight shape ---
    # Output name reuses the original initializer name so that consumer nodes
    # don't need rewiring.
    final_out_name = orig_init_name
    reshape2_node = onnx.helper.make_node(
        "Reshape",
        inputs=[reshape2_input, orig_shape_name],
        outputs=[final_out_name],
        name=f"{prefix}_Reshape_orig",
    )
    new_nodes.append(reshape2_node)

    return new_inits, new_nodes, final_out_name


# ──────────────────────────────────────────────
# Apply block quantization to an ONNX model
# ──────────────────────────────────────────────
def get_quantizable_initializer_names(
    model: onnx.ModelProto,
    *,
    min_elements: int = WEIGHT_MIN_ELEMENTS,
    exclude_names: set[str] | None = None,
) -> list[str]:
    """Identify initializers eligible for per-block quantization.

    Returns names of initializers that:
    - Are fp32, fp16, or bf16 (all decoded to fp32 before quantization)
    - Have at least ``min_elements`` elements
    - Are used as weight inputs to Conv, Gemm, or MatMul nodes
      (including through Cast intermediary nodes from bf16/fp16 conversion)
    - Are not in ``exclude_names``

    Args:
        model: ONNX model (fp32, fp16, or bf16).
        min_elements: Minimum tensor size.
        exclude_names: Names to exclude (e.g. Mahalanobis buffers).

    Returns:
        List of initializer names eligible for quantization.
    """
    excluded = exclude_names or set()

    _QUANTIZABLE_DTYPES = {
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.FLOAT16,
        onnx.TensorProto.BFLOAT16,
    }

    # Build set of initializer names used as weight inputs to target ops.
    # For bf16 models, the conversion inserts Cast nodes between the
    # initializer and consumer ops (init → Cast → consumer).  We trace
    # through Cast nodes so that the original initializer name is found.
    weight_input_names: set[str] = set()

    # Map: Cast output name → Cast input name (for tracing through Casts)
    cast_output_to_input: dict[str, str] = {}
    for node in model.graph.node:
        if node.op_type == "Cast" and len(node.input) >= 1 and len(node.output) >= 1:
            cast_output_to_input[node.output[0]] = node.input[0]

    def _trace_through_casts(name: str) -> str:
        """Follow Cast chains back to the original initializer name."""
        visited: set[str] = set()
        while name in cast_output_to_input and name not in visited:
            visited.add(name)
            name = cast_output_to_input[name]
        return name

    for node in model.graph.node:
        if node.op_type == "Conv" and len(node.input) >= 2:
            weight_input_names.add(_trace_through_casts(node.input[1]))
        elif node.op_type == "Gemm" and len(node.input) >= 2:
            weight_input_names.add(_trace_through_casts(node.input[1]))
        elif node.op_type == "MatMul" and len(node.input) >= 2:
            weight_input_names.add(_trace_through_casts(node.input[0]))
            weight_input_names.add(_trace_through_casts(node.input[1]))

    result: list[str] = []

    for init in model.graph.initializer:
        if init.data_type not in _QUANTIZABLE_DTYPES:
            continue
        if init.name in excluded:
            continue
        if init.name not in weight_input_names:
            continue
        numel = int(np.prod(init.dims)) if init.dims else 1
        if numel < min_elements:
            continue
        result.append(init.name)

    return result


def apply_block_quantization(
    model: onnx.ModelProto,
    quant_dtype: QuantDtype,
    scale_dtype: TargetDtype,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    min_elements: int = WEIGHT_MIN_ELEMENTS,
    exclude_names: set[str] | None = None,
    quantize_names: list[str] | None = None,
    compute_dtype_is_fp32: bool = False,
) -> int:
    """Apply per-block quantization to weight initializers in-place.

    Replaces each eligible fp32 initializer with a sub-graph:
    ``quantized_weight → Cast → Mul(scale) → Reshape → [Slice] → Reshape``

    Args:
        model: ONNX model to modify in-place.
        quant_dtype: INT8 or FP8 variant.
        scale_dtype: BF16 or FP16 for scale storage.
        block_size: Elements per quantization block.
        min_elements: Minimum tensor size to quantize.
        exclude_names: Initializer names to skip.
        quantize_names: If provided, only quantize these names (overrides
            auto-detection). Names in ``exclude_names`` are still skipped.
        compute_dtype_is_fp32: If True, restore to fp32 (for fp32-activation models).

    Returns:
        Number of initializers quantized.
    """
    if quantize_names is not None:
        excluded = exclude_names or set()
        names_to_quantize = [n for n in quantize_names if n not in excluded]
    else:
        names_to_quantize = get_quantizable_initializer_names(
            model, min_elements=min_elements, exclude_names=exclude_names
        )

    if not names_to_quantize:
        return 0

    # Build lookup for initializers
    init_by_name = {init.name: init for init in model.graph.initializer}

    all_new_inits: list[onnx.TensorProto] = []
    all_new_nodes: list[onnx.NodeProto] = []
    inits_to_remove: set[str] = set()
    count = 0

    # Accepted dtypes: fp32, fp16, bf16.  Non-fp32 initializers are decoded
    # to fp32 before quantization so that the pipeline works on bf16/fp16
    # base models (the common case for mixed-precision export).
    _ACCEPTED_DTYPES = {
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.FLOAT16,
        onnx.TensorProto.BFLOAT16,
    }

    for name in names_to_quantize:
        if name not in init_by_name:
            continue
        init = init_by_name[name]
        if init.data_type not in _ACCEPTED_DTYPES:
            continue

        arr = onnx.numpy_helper.to_array(init)
        # Ensure fp32 for quantization math
        arr_fp32 = arr.astype(np.float32) if arr.dtype != np.float32 else arr

        # Quantize
        bqw = quantize_weight_per_block(arr_fp32, quant_dtype, block_size)

        # Build sub-graph
        new_inits, new_nodes, _output_name = build_block_dequant_subgraph(
            bqw, name, scale_dtype, compute_dtype_is_fp32=compute_dtype_is_fp32
        )

        all_new_inits.extend(new_inits)
        all_new_nodes.extend(new_nodes)
        inits_to_remove.add(name)
        count += 1

    if count == 0:
        return 0

    # Remove original initializers that were replaced
    remaining_inits = [
        i for i in model.graph.initializer if i.name not in inits_to_remove
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(remaining_inits)
    model.graph.initializer.extend(all_new_inits)

    # Remove replaced initializer names from graph.input (ONNX IR < 4 quirk)
    new_init_names = {i.name for i in all_new_inits}
    kept_inputs = [
        gi
        for gi in model.graph.input
        if gi.name not in inits_to_remove and gi.name not in new_init_names
    ]
    del model.graph.input[:]
    model.graph.input.extend(kept_inputs)

    # Prepend new sub-graph nodes (they must execute before consumer nodes)
    for i, node in enumerate(all_new_nodes):
        model.graph.node.insert(i, node)

    # Clean up redundant Cast nodes left over from bf16/fp16 conversion.
    # When a bf16 initializer X was converted, a Cast(X → X_fp32_from_bf16)
    # was inserted and consumers rewired to X_fp32_from_bf16.  Now that the
    # dequant subgraph outputs X (in fp32 when compute_dtype_is_fp32=True, or
    # in bf16/fp16 when False), these Cast nodes may be redundant.
    # If compute_dtype_is_fp32: dequant outputs fp32, the old Cast(bf16→fp32)
    #   now receives fp32 input — it's an identity Cast → remove and rewire.
    # If not compute_dtype_is_fp32: dequant outputs bf16/fp16, the old Cast
    #   still does bf16→fp32 which is valid and needed → keep it.
    if compute_dtype_is_fp32:
        _remove_redundant_casts(model, inits_to_remove)

    return count


def _remove_redundant_casts(
    model: onnx.ModelProto, replaced_init_names: set[str]
) -> None:
    """Remove Cast(X→fp32) nodes where X was a bf16/fp16 initializer now replaced.

    After block quantization with ``compute_dtype_is_fp32=True``, the dequant
    subgraph already outputs fp32 under the original initializer name.  The
    Cast node from bf16/fp16 conversion is now an identity (fp32→fp32) and
    should be elided.  Consumers are rewired from the Cast output back to
    the original name.
    """
    # Find Cast nodes that read from a replaced initializer and cast to FLOAT
    casts_to_remove: dict[str, tuple[str, str]] = {}  # node_name → (input, output)
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        if len(node.input) < 1 or len(node.output) < 1:
            continue
        if node.input[0] not in replaced_init_names:
            continue
        # Check that this Cast targets FLOAT (the bf16/fp16→fp32 Cast)
        for attr in node.attribute:
            if attr.name == "to" and attr.i == onnx.TensorProto.FLOAT:
                casts_to_remove[node.name] = (node.input[0], node.output[0])
                break

    if not casts_to_remove:
        return

    # Rewire: replace all references to the Cast output with the Cast input
    for _node_name, (cast_input, cast_output) in casts_to_remove.items():
        for node in model.graph.node:
            for i, inp in enumerate(node.input):
                if inp == cast_output:
                    node.input[i] = cast_input

    # Remove the Cast nodes themselves
    cast_names = set(casts_to_remove.keys())
    kept_nodes = [n for n in model.graph.node if n.name not in cast_names]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)


# ──────────────────────────────────────────────
# Sensitivity detection (2-stage hybrid)
# ──────────────────────────────────────────────
def find_sensitive_initializers_stage1(
    model: onnx.ModelProto,
    quant_dtype: QuantDtype,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    nrmse_threshold: float = DEFAULT_WEIGHT_NRMSE_THRESHOLD,
    min_elements: int = WEIGHT_MIN_ELEMENTS,
    exclude_names: set[str] | None = None,
) -> tuple[list[str], list[str], dict[str, float]]:
    """Stage 1: identify suspect initializers by weight reconstruction error.

    For each quantizable initializer, perform per-block quantize → dequantize
    and compute NRMSE against the original fp32 values.

    Args:
        model: fp32 ONNX model (unmodified).
        quant_dtype: Target quantization dtype.
        block_size: Block size for quantization.
        nrmse_threshold: NRMSE above this → suspect.
        min_elements: Minimum tensor size.
        exclude_names: Names to exclude.

    Returns:
        A 3-tuple:
        - ``ok_names``: Initializers with NRMSE ≤ threshold (safe to quantize).
        - ``suspect_names``: Initializers with NRMSE > threshold (need Stage 2).
        - ``nrmse_map``: Dict mapping name → NRMSE for all checked initializers.
    """
    candidates = get_quantizable_initializer_names(
        model, min_elements=min_elements, exclude_names=exclude_names
    )

    init_by_name = {init.name: init for init in model.graph.initializer}

    ok_names: list[str] = []
    suspect_names: list[str] = []
    nrmse_map: dict[str, float] = {}

    for name in candidates:
        init = init_by_name[name]
        arr_fp32 = onnx.numpy_helper.to_array(init)

        bqw = quantize_weight_per_block(arr_fp32, quant_dtype, block_size)
        arr_restored = dequantize_weight_per_block(bqw)
        nrmse = compute_weight_nrmse(arr_fp32, arr_restored)
        nrmse_map[name] = nrmse

        if nrmse > nrmse_threshold:
            suspect_names.append(name)
        else:
            ok_names.append(name)

    return ok_names, suspect_names, nrmse_map


def find_sensitive_initializers_stage2(
    fp32_onnx_path: str,
    suspect_names: list[str],
    quant_dtype: QuantDtype,
    scale_dtype: TargetDtype,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    output_diff_threshold: float = DEFAULT_OUTPUT_DIFF_THRESHOLD,
    probe_samples: list[np.ndarray] | None = None,
) -> tuple[list[str], list[str], dict[str, float]]:
    """Stage 2: verify suspect initializers by comparing model outputs.

    For each suspect initializer, build a model where only that initializer
    is block-quantized, run inference on probe samples, and compare the
    output (logits) against the fp32 baseline.

    Args:
        fp32_onnx_path: Path to the fp32 ONNX model.
        suspect_names: Initializer names flagged by Stage 1.
        quant_dtype: Target quantization dtype.
        scale_dtype: BF16 or FP16 for scale storage.
        block_size: Block size for quantization.
        output_diff_threshold: Max abs diff above this → exclude.
        probe_samples: List of ``(1, 3, H, W)`` float32 numpy arrays.

    Returns:
        A 3-tuple:
        - ``ok_names``: Suspects that passed Stage 2 (safe to quantize).
        - ``excluded_names``: Suspects that failed (keep in base dtype).
        - ``diff_map``: Dict mapping name → max_abs_diff for all checked.
    """
    import onnxruntime as ort

    if not suspect_names:
        return [], [], {}

    if not probe_samples:
        # No samples provided → can't verify, conservatively exclude all
        return [], list(suspect_names), {}

    # Baseline: fp32 outputs
    sess_fp32 = ort.InferenceSession(fp32_onnx_path)
    fp32_outputs = [sess_fp32.run(None, {"input": s})[0] for s in probe_samples]

    ok_names: list[str] = []
    excluded_names: list[str] = []
    diff_map: dict[str, float] = {}

    import copy
    import tempfile

    for name in suspect_names:
        # Build a model with only this initializer block-quantized
        model_copy = copy.deepcopy(onnx.load(fp32_onnx_path))
        applied = apply_block_quantization(
            model_copy,
            quant_dtype,
            scale_dtype,
            block_size=block_size,
            quantize_names=[name],
            compute_dtype_is_fp32=True,  # fp32 base model
        )

        if applied == 0:
            # Could not quantize (e.g. not found) → keep in base dtype
            excluded_names.append(name)
            continue

        try:
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                tmp_path = f.name
                onnx.save(model_copy, tmp_path)

            sess_q = ort.InferenceSession(tmp_path)
            max_diff = max(
                float(
                    np.max(np.abs(fp32_outputs[i] - sess_q.run(None, {"input": s})[0]))
                )
                for i, s in enumerate(probe_samples)
            )
            diff_map[name] = max_diff

            if max_diff > output_diff_threshold:
                excluded_names.append(name)
            else:
                ok_names.append(name)

        except Exception as e:
            print(f"    [{name}] Stage 2 failed ({e!s:.80}), excluding")
            excluded_names.append(name)
        finally:
            import os

            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return ok_names, excluded_names, diff_map


def find_sensitive_initializers(
    fp32_onnx_path: str,
    quant_dtype: QuantDtype,
    scale_dtype: TargetDtype,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    min_elements: int = WEIGHT_MIN_ELEMENTS,
    exclude_names: set[str] | None = None,
    nrmse_threshold: float = DEFAULT_WEIGHT_NRMSE_THRESHOLD,
    output_diff_threshold: float = DEFAULT_OUTPUT_DIFF_THRESHOLD,
    probe_samples: list[np.ndarray] | None = None,
) -> tuple[list[str], list[str]]:
    """Run 2-stage hybrid sensitivity detection.

    Combines Stage 1 (weight NRMSE) and Stage 2 (output difference) to
    determine which initializers can safely be block-quantized.

    Args:
        fp32_onnx_path: Path to the fp32 ONNX model.
        quant_dtype: Target quantization dtype.
        scale_dtype: BF16 or FP16 for scale storage.
        block_size: Block size.
        min_elements: Minimum tensor size.
        exclude_names: Names to exclude entirely.
        nrmse_threshold: Stage 1 NRMSE threshold.
        output_diff_threshold: Stage 2 max abs diff threshold.
        probe_samples: Probe samples for Stage 2.

    Returns:
        A 2-tuple:
        - ``quantize_names``: Names safe to quantize.
        - ``sensitive_names``: Names to keep in base dtype.
    """
    model = onnx.load(fp32_onnx_path)

    print(f"  Sensitivity detection ({quant_dtype.value}, block_size={block_size})")

    # Stage 1
    ok_s1, suspect_s1, nrmse_map = find_sensitive_initializers_stage1(
        model,
        quant_dtype,
        block_size=block_size,
        nrmse_threshold=nrmse_threshold,
        min_elements=min_elements,
        exclude_names=exclude_names,
    )
    print(
        f"    Stage 1 (weight NRMSE, threshold={nrmse_threshold}): "
        f"{len(ok_s1)} OK, {len(suspect_s1)} suspect"
    )
    if suspect_s1:
        for name in suspect_s1:
            print(f"      [{name}] NRMSE={nrmse_map[name]:.6f}")

    # Stage 2
    ok_s2, excluded_s2, diff_map = find_sensitive_initializers_stage2(
        fp32_onnx_path,
        suspect_s1,
        quant_dtype,
        scale_dtype,
        block_size=block_size,
        output_diff_threshold=output_diff_threshold,
        probe_samples=probe_samples,
    )
    print(
        f"    Stage 2 (output diff, threshold={output_diff_threshold}): "
        f"{len(ok_s2)} recovered, {len(excluded_s2)} excluded"
    )
    if excluded_s2:
        for name in excluded_s2:
            diff = diff_map.get(name, float("nan"))
            print(f"      [{name}] max_abs_diff={diff:.6f}")

    quantize_names = ok_s1 + ok_s2
    sensitive_names = excluded_s2
    total = len(quantize_names) + len(sensitive_names)
    print(
        f"    Result: {len(quantize_names)}/{total} quantizable, "
        f"{len(sensitive_names)}/{total} sensitive"
    )

    return quantize_names, sensitive_names

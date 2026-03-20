"""ONNX weight dtype conversion utilities.

Provides a unified interface for converting fp32 ONNX initializers to fp16 or
bf16, both for full-model conversion and for post-quantization residual weight
compression.  The core logic (identify candidates, rewrite initializers, insert
Cast nodes, rewire consumers) is shared; only the actual dtype conversion
differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
import onnx
import torch

# Minimum element count for an initializer to be worth converting.
# Small tensors (bias, BN params) stay fp32 for precision.
FP16_WEIGHT_MIN_ELEMENTS = 1024


class TargetDtype(Enum):
    """Supported reduced-precision target dtypes."""

    FP16 = "fp16"
    BF16 = "bf16"


@dataclass(frozen=True)
class _DtypeConversionSpec:
    """Describes how to convert an fp32 initializer to a reduced dtype."""

    onnx_dtype: int  # onnx.TensorProto data type enum
    cast_suffix: str  # suffix for the Cast output name
    cast_node_prefix: str  # prefix for the Cast node name

    convert_array: Callable[[np.ndarray], onnx.TensorProto]
    """Converts an fp32 numpy array into an ONNX TensorProto of the target dtype.
    The caller sets ``name`` after creation."""

    can_overflow: bool
    """Whether the target dtype can overflow fp32 values (fp16 yes, bf16 no)."""


def _make_fp16_tensor(arr_fp32: np.ndarray) -> onnx.TensorProto:
    arr_fp16 = arr_fp32.astype(np.float16)
    if np.any(np.isinf(arr_fp16)) or np.any(np.isnan(arr_fp16)):
        raise _OverflowError
    return onnx.numpy_helper.from_array(arr_fp16)


def _make_bf16_tensor(arr_fp32: np.ndarray) -> onnx.TensorProto:
    t_bf16 = torch.from_numpy(arr_fp32).bfloat16()
    raw_int16 = t_bf16.view(torch.int16).numpy()
    tp = onnx.TensorProto()
    tp.data_type = onnx.TensorProto.BFLOAT16
    tp.dims.extend(arr_fp32.shape)
    tp.raw_data = raw_int16.tobytes()
    return tp


class _OverflowError(Exception):
    """Raised when fp16 conversion would produce inf/nan."""


_SPECS: dict[TargetDtype, _DtypeConversionSpec] = {
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
        can_overflow=False,  # bf16 has same exponent range as fp32
    ),
}


# ──────────────────────────────────────────────
# Core conversion engine
# ──────────────────────────────────────────────
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
    min_elements: int = FP16_WEIGHT_MIN_ELEMENTS,
    exclude_quant_params: bool = True,
    exclude_io: bool = True,
) -> int:
    """Convert fp32 initializers to *target* dtype in-place.

    This is the unified replacement for the original four functions:
    - ``_convert_fp32_weights_to_fp16``  (post-quantization residual)
    - ``_convert_fp32_weights_to_bf16``  (post-quantization residual)
    - ``_convert_all_fp32_to_bf16``      (full-model bf16)

    For full-model conversion, set ``min_elements=0`` and
    ``exclude_quant_params=False``.

    Args:
        model: ONNX model to modify in-place.
        target: Target reduced dtype (FP16 or BF16).
        min_elements: Skip initializers smaller than this (preserves precision
            for bias / BN params).
        exclude_quant_params: If True, skip quantization scale/zero-point
            initializers.
        exclude_io: If True, skip initializers that are also graph I/O names.

    Returns:
        Number of initializers converted.
    """
    spec = _SPECS[target]

    # Determine names to exclude
    excluded: set[str] = set()
    if exclude_quant_params:
        excluded |= _collect_quant_param_names(model)
    if exclude_io:
        for t in list(model.graph.input) + list(model.graph.output):
            excluded.add(t.name)

    # Identify convertible initializers
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

    # Convert each initializer
    actually_converted: list[str] = []
    for init in model.graph.initializer:
        if init.name not in converted_names:
            continue
        arr_fp32 = onnx.numpy_helper.to_array(init)
        try:
            new_tensor = spec.convert_array(arr_fp32)
        except _OverflowError:
            continue  # skip this initializer (fp16 overflow)
        new_tensor.name = init.name
        # For fp16, from_array already sets dims; for bf16 we set them in _make_bf16_tensor
        init.CopyFrom(new_tensor)
        actually_converted.append(init.name)

    if not actually_converted:
        return 0

    _insert_cast_nodes_and_rewire(model, actually_converted, spec)
    return len(actually_converted)

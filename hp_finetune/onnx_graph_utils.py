"""ONNX graph manipulation utilities.

Provides helpers for:
- Making batch dimensions dynamic
- Fixing hardcoded Reshape nodes
- Merging external data
- Embedding class metadata
- Identifying Mahalanobis-exclusive nodes
- Shape inference for TensorRT compatibility
- Full-graph fp16/bf16 conversion (activation + weight)
"""

from __future__ import annotations

import copy
import json
import os
import re

import numpy as np
import onnx
import onnx.numpy_helper as onnx_numpy_helper
import onnxruntime as ort

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
RESHAPE_FIX_MAX_ITERATIONS = 20
RESHAPE_TEST_BATCH_MIN = 2
RESHAPE_TEST_BATCH_MAX = 14


# ──────────────────────────────────────────────
# Batch dimension helpers
# ──────────────────────────────────────────────
def make_batch_dim_dynamic(
    model: onnx.ModelProto, dim_name: str = "batch_size"
) -> None:
    """Rewrite the leading dim of all graph inputs/outputs to a symbolic name.

    ``torch.onnx.export`` ``dynamic_axes`` only sets metadata; internal
    ``ValueInfo`` shapes may still carry ``dim_value=1``.  This function
    replaces those with ``dim_param`` so every runtime respects dynamic batch.
    """
    for io_list in (model.graph.input, model.graph.output):
        for tensor in io_list:
            shape = tensor.type.tensor_type.shape
            if shape and len(shape.dim) > 0:
                dim0 = shape.dim[0]
                dim0.ClearField("dim_value")
                dim0.dim_param = dim_name


# ──────────────────────────────────────────────
# Reshape fix helpers
# ──────────────────────────────────────────────
def _collect_reshape_shape_initializers(
    model: onnx.ModelProto,
) -> dict[str, onnx.TensorProto]:
    """Return initializers used as the *shape* input of Reshape nodes."""
    init_by_name = {init.name: init for init in model.graph.initializer}
    result: dict[str, onnx.TensorProto] = {}
    for node in model.graph.node:
        if node.op_type == "Reshape" and len(node.input) > 1:
            shape_name = node.input[1]
            if shape_name in init_by_name:
                result[shape_name] = init_by_name[shape_name]
    return result


def _find_reshape_init_for_node(
    model: onnx.ModelProto,
    node_name: str,
    reshape_inits: dict[str, onnx.TensorProto],
) -> onnx.TensorProto | None:
    """Return the Reshape shape initializer that corresponds to *node_name*."""
    for node in model.graph.node:
        if node.name == node_name and len(node.input) > 1:
            return reshape_inits.get(node.input[1])
    return None


def fix_hardcoded_batch_in_reshapes(
    model: onnx.ModelProto,
    save_path: str,
    *,
    input_size: int,
    rng: np.random.Generator | None = None,
    label: str = "",
) -> int:
    """Replace hardcoded batch dimensions in Reshape nodes with ``-1``.

    Repeatedly runs inference with random batch sizes, catches Reshape
    failures, and patches the shape constants.

    Returns the number of Reshape nodes fixed.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    prefix = f"  {label}: " if label else "  "
    reshape_inits = _collect_reshape_shape_initializers(model)
    fixed_count = 0

    for _ in range(RESHAPE_FIX_MAX_ITERATIONS):
        onnx.save(model, save_path)
        sess = ort.InferenceSession(save_path)
        batch_size = int(rng.integers(RESHAPE_TEST_BATCH_MIN, RESHAPE_TEST_BATCH_MAX))
        test_input = np.random.randn(batch_size, 3, input_size, input_size).astype(
            np.float32
        )
        try:
            sess.run(None, {"input": test_input})
            break  # success — no more fixes needed
        except Exception as e:
            err_msg = str(e)
            match = re.search(r"Reshape node\. Name:'([^']+)'", err_msg)
            if not match:
                print(
                    f"{prefix}[WARN] Reshape error but could not parse node name: "
                    f"{err_msg[:200]}"
                )
                break

            failing_node_name = match.group(1)
            target_init = _find_reshape_init_for_node(
                model, failing_node_name, reshape_inits
            )
            if target_init is None:
                print(
                    f"{prefix}[WARN] Could not find shape initializer "
                    f"for {failing_node_name}"
                )
                break

            arr = onnx.numpy_helper.to_array(target_init)
            new_arr = arr.copy()
            new_arr[0] = -1
            new_tensor = onnx.numpy_helper.from_array(new_arr, name=target_init.name)
            target_init.CopyFrom(new_tensor)
            fixed_count += 1
            print(
                f"{prefix}Fixed: {failing_node_name} shape "
                f"{arr.tolist()} -> {new_arr.tolist()} (tested batch={batch_size})"
            )

    if fixed_count > 0:
        onnx.save(model, save_path)
        print(f"{prefix}Total: fixed {fixed_count} Reshape shape initializer(s)")
    else:
        print(f"{prefix}No Reshape fix needed")

    return fixed_count


# ──────────────────────────────────────────────
# External data merging
# ──────────────────────────────────────────────
def merge_external_data(onnx_path: str) -> None:
    """Merge a ``.onnx.data`` sidecar (from dynamo exporter) back into the protobuf."""
    external_data_path = onnx_path + ".data"
    if not os.path.exists(external_data_path):
        return
    print("  Merging external data into ONNX protobuf...")
    onnx_model = onnx.load(onnx_path, load_external_data=True)
    onnx.save(onnx_model, onnx_path, save_as_external_data=False)
    os.remove(external_data_path)


# ──────────────────────────────────────────────
# Metadata helpers
# ──────────────────────────────────────────────
def embed_class_metadata(model: onnx.ModelProto, class_names: list[str]) -> None:
    """Store class names and count in ONNX ``metadata_props``."""
    meta_cn = model.metadata_props.add()
    meta_cn.key = "class_names"
    meta_cn.value = json.dumps(class_names, ensure_ascii=False)

    meta_nc = model.metadata_props.add()
    meta_nc.key = "num_classes"
    meta_nc.value = str(len(class_names))


# ──────────────────────────────────────────────
# Mahalanobis node identification
# ──────────────────────────────────────────────
def get_mahal_exclusive_nodes(model: onnx.ModelProto) -> list[str]:
    """Return node names that are exclusive to the Mahalanobis anomaly_score path.

    These are nodes that appear in the ancestor set of ``anomaly_score`` but NOT
    in the ancestor set of ``logits``.  They implement the Mahalanobis distance
    computation (Sub, MatMul, Mul, ReduceSum, Clip, Sqrt) on the raw embedding.

    When an ONNX model has no ``anomaly_score`` output this returns an empty list.
    """
    output_names_set = {o.name for o in model.graph.output}
    if "anomaly_score" not in output_names_set:
        return []

    output_to_node: dict[str, onnx.NodeProto] = {}
    for n in model.graph.node:
        for o in n.output:
            output_to_node[o] = n

    def _ancestors(tensor_name: str, visited: set) -> list[str]:
        if tensor_name not in output_to_node:
            return []
        node = output_to_node[tensor_name]
        if node.name in visited:
            return []
        visited.add(node.name)
        result = [node.name]
        for inp in node.input:
            result.extend(_ancestors(inp, visited))
        return result

    anomaly_visited: set = set()
    anomaly_nodes = _ancestors("anomaly_score", anomaly_visited)

    logits_visited: set = set()
    logits_nodes = set(_ancestors("logits", logits_visited))

    return [n for n in anomaly_nodes if n not in logits_nodes]


def get_mahal_initializer_names(model: onnx.ModelProto) -> set[str]:
    """Return names of Mahalanobis-related initializers.

    These should be excluded from reduced-precision conversion to maintain
    anomaly detection accuracy.  Includes ``mahal_class_means``, ``mahal_precision``,
    and ``mahal_threshold``.
    """
    names: set[str] = set()
    for init in model.graph.initializer:
        if init.name.startswith("mahal_"):
            names.add(init.name)
    return names


# ──────────────────────────────────────────────
# Restore Mahalanobis initializers to fp32
# ──────────────────────────────────────────────
def restore_mahal_initializers_to_fp32(
    model: onnx.ModelProto,
    fp32_model: onnx.ModelProto,
) -> int:
    """Restore ``mahal_class_means`` and ``mahal_precision`` initializers to fp32.

    After fp16/bf16 conversion these buffers may be clipped or lose precision,
    causing the Mahalanobis distance to overflow / underflow.  This function
    replaces them with the original fp32 values.

    Returns the number of initializers restored.
    """
    fp32_inits = {
        init.name: onnx_numpy_helper.to_array(init)
        for init in fp32_model.graph.initializer
    }
    mahal_names = {"mahal_class_means", "mahal_precision"}
    count = 0
    for i, init in enumerate(model.graph.initializer):
        if init.name in mahal_names and init.name in fp32_inits:
            fp32_arr = fp32_inits[init.name].astype(np.float32)
            new_init = onnx_numpy_helper.from_array(fp32_arr, name=init.name)
            model.graph.initializer[i].CopyFrom(new_init)
            count += 1
    return count


# ──────────────────────────────────────────────
# Full-graph fp16 conversion
# ──────────────────────────────────────────────
def convert_graph_to_fp16(
    fp32_model: onnx.ModelProto,
    *,
    has_mahal: bool = False,
) -> onnx.ModelProto:
    """Convert entire graph (weights + activations) to fp16.

    Uses ``onnxconverter_common.float16`` for reliable activation + weight
    conversion.  I/O tensors stay fp32 (``keep_io_types=True``).

    When ``has_mahal=True``:
    - Mahalanobis-exclusive nodes are kept in fp32 via ``node_block_list``
    - ``mahal_class_means`` / ``mahal_precision`` initializers are restored to fp32

    Args:
        fp32_model: Original fp32 model (not modified).
        has_mahal: Whether the model has Mahalanobis anomaly detection.

    Returns:
        A new ModelProto with fp16 weights and activations.
    """
    from onnxconverter_common import float16 as onnx_float16

    # Identify Mahalanobis nodes to block from fp16 conversion
    mahal_node_block: list[str] = []
    if has_mahal:
        mahal_node_block = get_mahal_exclusive_nodes(fp32_model)
        if mahal_node_block:
            print(f"  Keeping {len(mahal_node_block)} Mahalanobis nodes in fp32")

    fp16_model = onnx_float16.convert_float_to_float16(
        copy.deepcopy(fp32_model),
        keep_io_types=True,
        disable_shape_infer=False,
        check_fp16_ready=False,
        node_block_list=mahal_node_block if mahal_node_block else None,
    )

    # Restore Mahalanobis initializers to fp32
    if has_mahal:
        restored = restore_mahal_initializers_to_fp32(fp16_model, fp32_model)
        if restored:
            print(f"  Restored {restored} Mahalanobis initializer(s) to fp32")

    return fp16_model


# ──────────────────────────────────────────────
# Full-graph bf16 conversion
# ──────────────────────────────────────────────
def convert_graph_to_bf16(
    fp32_model: onnx.ModelProto,
    *,
    has_mahal: bool = False,
) -> onnx.ModelProto:
    """Convert entire graph weights to bf16 (stored as bf16, Cast to fp32 at runtime).

    I/O stays fp32; no consumer-side dtype handling needed.

    Unlike fp16, bf16 has the same exponent range as fp32 so there is no
    clipping risk.  However the reduced mantissa (7 bits) introduces ~0.8%
    relative error per element.

    When ``has_mahal=True``:
    - ``mahal_class_means`` / ``mahal_precision`` are restored to fp32

    Args:
        fp32_model: Original fp32 model (not modified).
        has_mahal: Whether the model has Mahalanobis anomaly detection.

    Returns:
        A new ModelProto with bf16 weights (activations compute via Cast nodes).
    """
    from hp_finetune.weight_conversion import TargetDtype, convert_initializers

    bf16_model = copy.deepcopy(fp32_model)

    # Exclude mahal initializers from bf16 conversion
    mahal_names = get_mahal_initializer_names(bf16_model) if has_mahal else set()

    converted = convert_initializers(
        bf16_model,
        TargetDtype.BF16,
        min_elements=0,
        exclude_quant_params=False,
        exclude_names=mahal_names,
    )
    print(f"  Converted {converted} initializers to bf16")

    # Restore mahal initializers just in case (shouldn't be needed since we
    # excluded them, but as a safety net)
    if has_mahal:
        restored = restore_mahal_initializers_to_fp32(bf16_model, fp32_model)
        if restored:
            print(f"  Restored {restored} Mahalanobis initializer(s) to fp32")

    return bf16_model


# ──────────────────────────────────────────────
# Shape inference for TensorRT
# ──────────────────────────────────────────────
def infer_shapes_for_tensorrt(model: onnx.ModelProto) -> onnx.ModelProto:
    """Run ONNX shape inference to populate ``value_info`` for TensorRT EP.

    TensorRT EP requires all intermediate tensor shapes to be present.
    This function handles fp16/bf16 models by temporarily casting initializers
    to fp32 before running shape inference.

    Args:
        model: ONNX model proto.  The input is **not** modified.

    Returns:
        A new ``ModelProto`` with ``value_info`` populated.
    """
    _FP16 = onnx.TensorProto.FLOAT16
    _BF16 = onnx.TensorProto.BFLOAT16

    def _try_infer(m: onnx.ModelProto) -> onnx.ModelProto | None:
        try:
            return onnx.shape_inference.infer_shapes(
                m, check_type=False, strict_mode=False, data_prop=True
            )
        except Exception:
            return None

    # First attempt: infer on the original model directly
    result = _try_infer(model)
    if result is not None:
        init_names_m = {i.name for i in model.graph.initializer}
        gi_names_m = {gi.name for gi in model.graph.input}
        excluded_m = init_names_m | gi_names_m
        out = copy.deepcopy(model)
        filtered = [vi for vi in result.graph.value_info if vi.name not in excluded_m]
        del out.graph.value_info[:]
        out.graph.value_info.extend(filtered)
        return out

    # Second attempt: cast fp16/bf16 initializers to fp32 in a scratch copy
    scratch = copy.deepcopy(model)

    init_names = {i.name for i in scratch.graph.initializer}
    clean_inputs = [gi for gi in scratch.graph.input if gi.name not in init_names]
    del scratch.graph.input[:]
    scratch.graph.input.extend(clean_inputs)

    cast_count = 0
    for init in scratch.graph.initializer:
        if init.data_type in (_FP16, _BF16):
            arr = onnx_numpy_helper.to_array(init).astype(np.float32)
            new_init = onnx_numpy_helper.from_array(arr, name=init.name)
            init.CopyFrom(new_init)
            cast_count += 1

    if cast_count == 0:
        print(
            "  [WARN] onnx.shape_inference failed and no fp16/bf16 casts to try; "
            "skipping"
        )
        return model

    result = _try_infer(scratch)
    if result is None:
        print("  [WARN] onnx.shape_inference failed even after fp32 cast; skipping")
        return model

    out = copy.deepcopy(model)
    init_names_out = {i.name for i in out.graph.initializer}
    graph_input_names = {gi.name for gi in out.graph.input}
    excluded = init_names_out | graph_input_names
    filtered_vi = [vi for vi in result.graph.value_info if vi.name not in excluded]
    del out.graph.value_info[:]
    out.graph.value_info.extend(filtered_vi)
    return out

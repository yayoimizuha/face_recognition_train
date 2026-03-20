"""ONNX graph manipulation utilities.

Provides helpers for making batch dimensions dynamic, fixing hardcoded Reshape
nodes, merging external data, and embedding class metadata.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import onnx
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
            break  # success -- no more fixes needed
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
    """Store class names and count in ONNX ``metadata_props``.

    Reading back::

        sess = ort.InferenceSession("model.onnx")
        meta = sess.get_modelmeta().custom_metadata_map
        class_names = json.loads(meta["class_names"])
        num_classes = int(meta["num_classes"])
    """
    meta_cn = model.metadata_props.add()
    meta_cn.key = "class_names"
    meta_cn.value = json.dumps(class_names, ensure_ascii=False)

    meta_nc = model.metadata_props.add()
    meta_nc.key = "num_classes"
    meta_nc.value = str(len(class_names))

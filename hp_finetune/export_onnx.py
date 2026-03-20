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
import json
import os
import re
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
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from torchvision import transforms
from tqdm import tqdm

# ── finetune_facenet.py からモデル定義と定数を再利用 ──
from hp_finetune.finetune_facenet import (
    BACKBONE,
    BACKBONE_DIM,
    EMB_SIZE,
    HIDDEN_DIM,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    FaceRecognitionModel,
)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
RESHAPE_FIX_MAX_ITERATIONS = 20
RESHAPE_TEST_BATCH_MIN = 2
RESHAPE_TEST_BATCH_MAX = 14
VERIFY_BATCH_MIN = 2
VERIFY_BATCH_MAX = 32
VERIFY_BATCH_COUNT = 5
FP32_MAX_DIFF_WARN_THRESHOLD = 1e-4
DEFAULT_CALIB_SAMPLES = 200
DEFAULT_EVAL_SAMPLES = 50
DEFAULT_OPSET = 18
FP16_WEIGHT_MIN_ELEMENTS = 1024

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
OP_TYPES_TO_QUANTIZE = [
    "Conv",
    "Gemm",
]


# ──────────────────────────────────────────────
# ONNX graph shape helpers
# ──────────────────────────────────────────────
def make_batch_dim_dynamic(
    model: onnx.ModelProto, dim_name: str = "batch_size"
) -> None:
    """ONNX グラフの input / output の先頭次元を symbolic dim に書き換える。

    torch.onnx.export の dynamic_axes はメタデータ上の宣言のみを変更するが、
    グラフ内部の ValueInfo に shape が残り、推論エンジンによっては batch=1 と
    解釈される。この関数は graph.input / graph.output の dim_value を
    dim_param (symbolic) に変換して確実に動的化する。
    """
    for io_list in (model.graph.input, model.graph.output):
        for tensor in io_list:
            shape = tensor.type.tensor_type.shape
            if shape and len(shape.dim) > 0:
                dim0 = shape.dim[0]
                dim0.ClearField("dim_value")
                dim0.dim_param = dim_name


def _collect_reshape_shape_initializers(
    model: onnx.ModelProto,
) -> dict[str, onnx.TensorProto]:
    """Reshape ノードの shape input に使われている initializer を収集する。"""
    init_by_name = {init.name: init for init in model.graph.initializer}
    result: dict[str, onnx.TensorProto] = {}
    for node in model.graph.node:
        if node.op_type == "Reshape" and len(node.input) > 1:
            shape_name = node.input[1]
            if shape_name in init_by_name:
                result[shape_name] = init_by_name[shape_name]
    return result


def fix_hardcoded_batch_in_reshapes(
    model: onnx.ModelProto,
    save_path: str,
    *,
    rng: np.random.Generator | None = None,
    label: str = "",
) -> int:
    """Reshape ノードにハードコードされたバッチ次元を -1 に修正する。

    ランダムなバッチサイズで推論を試み、Reshape で失敗するノードの
    shape 定数を特定して先頭を -1 に書き換える。

    Returns:
        修正した Reshape ノードの数。
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
        test_input = np.random.randn(batch_size, 3, INPUT_SIZE, INPUT_SIZE).astype(
            np.float32
        )
        try:
            sess.run(None, {"input": test_input})
            break  # 成功 — もう修正は不要
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


def _find_reshape_init_for_node(
    model: onnx.ModelProto,
    node_name: str,
    reshape_inits: dict[str, onnx.TensorProto],
) -> onnx.TensorProto | None:
    """指定のノード名に対応する Reshape shape initializer を返す。"""
    for node in model.graph.node:
        if node.name == node_name and len(node.input) > 1:
            return reshape_inits.get(node.input[1])
    return None


# ──────────────────────────────────────────────
# Verification helpers
# ──────────────────────────────────────────────
def verify_dynamic_batch(
    onnx_path: str,
    *,
    num_classes: int,
    rng: np.random.Generator | None = None,
    label: str = "",
) -> None:
    """複数のランダムバッチサイズで推論して出力 shape が正しいことを検証する。"""
    if rng is None:
        rng = np.random.default_rng(0)

    prefix = f"  {label}: " if label else "  "
    sess = ort.InferenceSession(onnx_path)
    batch_sizes = [
        int(b)
        for b in rng.integers(
            VERIFY_BATCH_MIN, VERIFY_BATCH_MAX, size=VERIFY_BATCH_COUNT
        )
    ]
    for bs in batch_sizes:
        test_in = np.random.randn(bs, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
        out = sess.run(None, {"input": test_in})[0]
        assert out.shape == (bs, num_classes), (
            f"{label} batch={bs}: expected ({bs}, {num_classes}), got {out.shape}"
        )
    print(f"{prefix}Dynamic batch OK: tested batch sizes {batch_sizes}")


def compare_outputs(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    label: str = "",
    warn_threshold: float = FP32_MAX_DIFF_WARN_THRESHOLD,
) -> tuple[float, float]:
    """2つの出力の max abs diff と cosine similarity を計算して表示する。

    Returns:
        (max_abs_diff, cosine_similarity)
    """
    max_diff = float(np.max(np.abs(reference - target)))
    cos_sim = float(
        np.dot(reference.flatten(), target.flatten())
        / (np.linalg.norm(reference) * np.linalg.norm(target))
    )
    print(f"  max abs diff{f' ({label})' if label else ''}: {max_diff:.2e}")
    print(f"  cosine similarity{f' ({label})' if label else ''}: {cos_sim:.6f}")
    if max_diff > warn_threshold:
        print(f"  [WARN] Large difference detected{f' ({label})' if label else ''}")
    return max_diff, cos_sim


# ──────────────────────────────────────────────
# Classification wrapper for ONNX export
# ──────────────────────────────────────────────
class ClassificationModel(nn.Module):
    """backbone + GWAP + head + arc_weight -> classification logits.

    出力は arc_s * cos_similarity(emb, arc_weight) の (B, num_classes) logits。
    これは FaceRecognitionModel.cos_logits() と同等の値（margin なし）。
    softmax は使用側で適用することで、確率取得と評価の両方に対応できる。

    使用例:
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
        # arc_weight を L2 正規化済みの固定バッファとして保持
        w = F.normalize(full_model.arc_weight.data, dim=1)
        self.register_buffer("arc_weight_normalized", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.forward_features(x)
        pooled = self.gwap(feat)
        emb = self.head(pooled)
        emb = F.normalize(emb, dim=1)
        cos_sim = F.linear(emb, self.arc_weight_normalized)  # (B, num_classes)
        return cos_sim * self.arc_s  # logits — softmax は使用側で適用


# ──────────────────────────────────────────────
# Calibration data reader for ONNX Runtime quantization
# ──────────────────────────────────────────────
class FaceCalibrationDataReader(CalibrationDataReader):
    """HuggingFace dataset から calibration 用バッチを生成する。

    Parameters:
        num_samples: キャリブレーションに使うサンプル数
    """

    def __init__(self, num_samples: int = DEFAULT_CALIB_SAMPLES):
        from datasets import load_dataset

        self.transform = transforms.Compose(
            [
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        print("Loading calibration dataset...")
        raw = load_dataset("yayoimizuha/helloproject-face-webdatasets")
        dataset = raw["train"]

        # データセット全体からランダムにサンプリング
        total_samples = min(num_samples, len(dataset))
        rng = np.random.default_rng(42)
        indices = rng.choice(len(dataset), size=total_samples, replace=False)

        # ONNX モデルは batch_size=1 固定でエクスポートされるため
        # キャリブレーションデータも 1 サンプルずつ渡す
        self.samples: list[np.ndarray] = []
        for idx in tqdm(indices, desc="Preparing calibration data"):
            item = dataset[int(idx)]
            img = item["image"].convert("RGB")
            tensor = self.transform(img).unsqueeze(0).numpy().astype(np.float32)
            self.samples.append(tensor)

        self.iter = iter(self.samples)
        print(f"Calibration: {len(self.samples)} samples prepared")

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            sample = next(self.iter)
            return {"input": sample}
        except StopIteration:
            return None

    def rewind(self) -> None:
        self.iter = iter(self.samples)


# ──────────────────────────────────────────────
# Checkpoint loading
# ──────────────────────────────────────────────
def _detect_num_classes(state_dict: dict, num_classes_arg: int | None) -> int:
    """チェックポイントの arc_weight から num_classes を自動検知する。

    --num-classes が明示的に指定されている場合はそちらを優先するが、
    arc_weight との不一致があれば警告を出す。
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
    """チェックポイントを読み込み、classification モデルを返す。"""
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
# Real-image sample helpers
# ──────────────────────────────────────────────
_REAL_SAMPLE_CACHE: np.ndarray | None = None


def _get_real_sample() -> np.ndarray:
    """データセットから実画像を 1 枚取得して (1, 3, H, W) np.ndarray を返す。

    初回呼び出し時にのみロードし、以降はキャッシュを返す。
    """
    global _REAL_SAMPLE_CACHE
    if _REAL_SAMPLE_CACHE is not None:
        return _REAL_SAMPLE_CACHE

    from datasets import load_dataset

    transform = transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    raw = load_dataset("yayoimizuha/helloproject-face-webdatasets", split="train")
    item = raw[0]
    img = item["image"].convert("RGB")
    _REAL_SAMPLE_CACHE = transform(img).unsqueeze(0).numpy().astype(np.float32)
    return _REAL_SAMPLE_CACHE


# ──────────────────────────────────────────────
# ONNX metadata helpers
# ──────────────────────────────────────────────
def _embed_class_metadata(model: onnx.ModelProto, class_names: list[str]) -> None:
    """ONNX モデルの metadata_props にクラス名と num_classes を埋め込む。

    読み出し例:
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


# ──────────────────────────────────────────────
# ONNX export helpers
# ──────────────────────────────────────────────
def merge_external_data(onnx_path: str) -> None:
    """dynamo エクスポーターが分離した .onnx.data を protobuf 内に統合する。"""
    external_data_path = onnx_path + ".data"
    if not os.path.exists(external_data_path):
        return
    print("  Merging external data into ONNX protobuf...")
    onnx_model = onnx.load(onnx_path, load_external_data=True)
    onnx.save(onnx_model, onnx_path, save_as_external_data=False)
    os.remove(external_data_path)


def export_fp32_onnx(
    cls_model: ClassificationModel,
    onnx_path: str,
    opset: int,
    *,
    num_classes: int,
    class_names: list[str],
) -> None:
    """PyTorch モデルを fp32 ONNX にエクスポートし、動的バッチ化と検証を行う。

    クラス名は ONNX metadata_props に "class_names" (JSON) として埋め込む。
    """
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

    # Reshape ノードのバッチ次元を動的化
    print("  Fixing hardcoded batch dimension in Reshape nodes...")
    onnx_model = onnx.load(onnx_path)
    fix_hardcoded_batch_in_reshapes(onnx_model, onnx_path, rng=np.random.default_rng(0))

    # graph.input / graph.output の先頭次元を symbolic "batch_size" に書き換え
    onnx_model = onnx.load(onnx_path)
    make_batch_dim_dynamic(onnx_model)
    onnx.save(onnx_model, onnx_path)

    # 動的バッチ検証
    print("  Verifying dynamic batch with random batch sizes...")
    verify_dynamic_batch(
        onnx_path, num_classes=num_classes, rng=np.random.default_rng(0), label="fp32"
    )

    # ONNX モデルの妥当性チェック
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"  fp32 ONNX model verified (opset={opset})")

    # metadata_props にクラス名を埋め込む
    _embed_class_metadata(onnx_model, class_names)
    onnx.save(onnx_model, onnx_path)
    print(f"  Embedded {len(class_names)} class names in metadata_props")

    # fp32 ONNX 出力が PyTorch と一致するか検証 (実画像 1 枚で確認)
    print("Verifying fp32 ONNX output vs PyTorch (real image)...")
    real_sample = _get_real_sample()
    real_tensor = torch.from_numpy(real_sample)
    with torch.no_grad():
        pt_out = cls_model(real_tensor).numpy()
    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {"input": real_sample})[0]
    compare_outputs(pt_out, ort_out, label="PyTorch vs fp32 ONNX")


# ──────────────────────────────────────────────
# Full fp16 / bf16 ONNX export helpers
# ──────────────────────────────────────────────
def export_fp16_onnx(
    fp32_path: str,
    fp16_path: str,
    *,
    num_classes: int,
    class_names: list[str],
) -> None:
    """fp32 ONNX をグラフ全体 fp16 に変換して保存する。

    onnxconverter_common の convert_float_to_float16 を使用し、重みだけでなく
    全 activation も fp16 に変換する。GPU (Ampere 以降) では fp16 演算として
    実行されるため fp32 より高速になる。CPU では fp32 フォールバック。

    入出力テンソルは keep_io_types=True で fp32 のまま保持するため、
    推論側で入出力の型変換は不要。
    """
    print(f"Exporting fp16 ONNX to: {fp16_path}")
    fp32_model = onnx.load(fp32_path)

    fp16_model = onnx_float16.convert_float_to_float16(
        fp32_model,
        keep_io_types=True,  # 入出力は fp32 のまま (推論側の利便性を保持)
        disable_shape_infer=False,
        check_fp16_ready=False,  # 再変換時のエラーを抑制
    )

    # metadata は fp32 モデルから引き継がれているが、上書きして確実に埋め込む
    _embed_class_metadata(fp16_model, class_names)
    onnx.save(fp16_model, fp16_path)

    # 動的バッチ検証
    print("  Verifying fp16 dynamic batch...")
    verify_dynamic_batch(
        fp16_path, num_classes=num_classes, rng=np.random.default_rng(2), label="fp16"
    )
    print(f"  fp16 ONNX saved: {fp16_path}")


def _convert_all_fp32_to_bf16(model: onnx.ModelProto) -> int:
    """fp32 ONNX モデルの全 initializer・ノードを bf16 に変換する。

    onnxconverter_common には bf16 変換機能がないため手動で実装する。
    処理内容:
    1. 全 fp32 initializer を bf16 TensorProto に置き換える
    2. 消費ノードの手前に Cast(bf16→fp32) を挿入する
    3. グラフの入出力は fp32 のまま保持する (keep_io_types 相当)

    Returns:
        変換した initializer の数。
    """
    # グラフ入出力の名前は変換しない (fp32 のまま)
    io_names: set[str] = set()
    for t in list(model.graph.input) + list(model.graph.output):
        io_names.add(t.name)

    inits_to_convert: list[str] = []
    for init in model.graph.initializer:
        if init.data_type != onnx.TensorProto.FLOAT:
            continue
        if init.name in io_names:
            continue
        inits_to_convert.append(init.name)

    if not inits_to_convert:
        return 0

    for init in model.graph.initializer:
        if init.name not in inits_to_convert:
            continue
        arr_fp32 = onnx.numpy_helper.to_array(init)
        t_bf16 = torch.from_numpy(arr_fp32).bfloat16()
        raw_int16 = t_bf16.view(torch.int16).numpy()
        new_init = onnx.TensorProto()
        new_init.data_type = onnx.TensorProto.BFLOAT16
        new_init.name = init.name
        new_init.dims.extend(arr_fp32.shape)
        new_init.raw_data = raw_int16.tobytes()
        init.CopyFrom(new_init)

    cast_nodes: list[onnx.NodeProto] = []
    for orig_name in inits_to_convert:
        cast_output = orig_name + "_fp32_from_bf16"
        cast_node = onnx.helper.make_node(
            "Cast",
            inputs=[orig_name],
            outputs=[cast_output],
            to=onnx.TensorProto.FLOAT,
            name=f"Cast_bf16_to_fp32_{orig_name}",
        )
        cast_nodes.append(cast_node)
        for node in model.graph.node:
            for i, inp in enumerate(node.input):
                if inp == orig_name:
                    node.input[i] = cast_output

    for i, cast_node in enumerate(cast_nodes):
        model.graph.node.insert(i, cast_node)

    return len(inits_to_convert)


def export_bf16_onnx(
    fp32_path: str,
    bf16_path: str,
    *,
    num_classes: int,
    class_names: list[str],
) -> None:
    """fp32 ONNX の全重みを bf16 に変換して保存する。

    onnxconverter_common に bf16 変換機能がないため手動実装。
    重みを bf16 で保存し、実行時に fp32 に Cast してから演算する。
    GPU (CUDA) 上では将来的に bf16 直接実行が可能になることを想定した形式。
    入出力は fp32 のまま保持するため推論側で型変換は不要。
    """
    print(f"Exporting bf16 ONNX to: {bf16_path}")
    bf16_model = onnx.load(fp32_path)

    converted = _convert_all_fp32_to_bf16(bf16_model)
    print(f"  Converted {converted} initializers to bf16")

    _embed_class_metadata(bf16_model, class_names)
    onnx.save(bf16_model, bf16_path)

    # 動的バッチ検証
    print("  Verifying bf16 dynamic batch...")
    verify_dynamic_batch(
        bf16_path, num_classes=num_classes, rng=np.random.default_rng(3), label="bf16"
    )
    print(f"  bf16 ONNX saved: {bf16_path}")


def _find_sensitive_nodes(preprocessed_path: str) -> list[str]:
    """INT8 量子化に敏感な Conv/Gemm ノードを特定して除外リストを返す。

    MobileNetV4-Hybrid + GWAP アーキテクチャでは以下のノードが量子化に弱い:

    1. Depthwise Conv (dw_start, dw_mid):
       各チャネルが独立した単一フィルタのため、per-channel 量子化でも
       表現力が不足し、特に深い層 (blocks.3) で壊滅的な精度劣化を起こす。

    2. ConvMulFusion (BN-folded Conv):
       quant_pre_process が BatchNorm の scale/bias を Conv に折り畳んだ結果、
       量子化スケールが極端に小さくなる (1e-13 オーダー) チャネルが発生し、
       重みの情報が消失する。

    3. Gemm (embedding head の Linear 層 + arc_weight cosine similarity):
       最終 embedding を生成する層で、微小な量子化誤差が L2 正規化後に
       角度として増幅される。arc_weight との内積も同様に敏感。

    4. GWAP の score_conv:
       exp(sigmoid(score)) の入力を生成する 1×1 Conv。量子化すると
       attention weight の分布が歪む。
    """
    model = onnx.load(preprocessed_path)
    sensitive: list[str] = []

    for node in model.graph.node:
        if node.op_type not in ("Conv", "Gemm"):
            continue

        weight_name = node.input[1] if len(node.input) > 1 else ""
        node_name = node.name

        is_depthwise = "dw_start" in weight_name or "dw_mid" in weight_name
        is_convmulfusion = "ConvMulFusion" in weight_name
        is_gemm = node.op_type == "Gemm"
        is_gwap = "score_conv" in weight_name or "conv2d" in node_name

        if is_depthwise or is_convmulfusion or is_gemm or is_gwap:
            sensitive.append(node_name)

    return sensitive


def _convert_fp32_weights_to_fp16(model: onnx.ModelProto) -> int:
    """量子化されなかった fp32 initializer を fp16 に変換しサイズを削減する。

    INT8 量子化から除外されたノード (depthwise conv, ConvMulFusion 等) の重みは
    fp32 のまま残っている。これらを fp16 に変換し、消費ノードの手前に
    Cast(fp16→fp32) ノードを挿入することで、ファイルサイズを約 40% 削減できる。

    量子化スケール/ゼロポイントや小さなテンソル (bias, BN パラメータ等) は
    精度が重要なため fp32 のまま保持する。

    Returns:
        変換した initializer の数。
    """
    # 量子化パラメータ名を収集 (scale, zero_point は精度が必要)
    quant_param_names: set[str] = set()
    for node in model.graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
            for i in range(1, len(node.input)):
                quant_param_names.add(node.input[i])

    # fp16 に変換する initializer を特定
    inits_to_convert: dict[str, np.ndarray] = {}
    for init in model.graph.initializer:
        if init.data_type != onnx.TensorProto.FLOAT:
            continue
        numel = int(np.prod(init.dims)) if init.dims else 1
        if init.name in quant_param_names:
            continue
        if numel < FP16_WEIGHT_MIN_ELEMENTS:
            continue
        arr = onnx.numpy_helper.to_array(init)
        arr_fp16 = arr.astype(np.float16)
        if np.any(np.isinf(arr_fp16)) or np.any(np.isnan(arr_fp16)):
            continue
        inits_to_convert[init.name] = arr_fp16

    if not inits_to_convert:
        return 0

    # initializer を fp16 に書き換え
    for init in model.graph.initializer:
        if init.name in inits_to_convert:
            new_init = onnx.numpy_helper.from_array(
                inits_to_convert[init.name], name=init.name
            )
            init.CopyFrom(new_init)

    # fp16 initializer ごとに Cast(fp16→fp32) ノードを作成し、
    # 消費ノードの入力を Cast の出力に差し替える
    cast_nodes: list[onnx.NodeProto] = []
    for orig_name in inits_to_convert:
        cast_output = orig_name + "_fp32"
        cast_node = onnx.helper.make_node(
            "Cast",
            inputs=[orig_name],
            outputs=[cast_output],
            to=onnx.TensorProto.FLOAT,
            name=f"Cast_fp16_to_fp32_{orig_name}",
        )
        cast_nodes.append(cast_node)

        for node in model.graph.node:
            for i, inp in enumerate(node.input):
                if inp == orig_name:
                    node.input[i] = cast_output

    # Cast ノードをグラフの先頭に挿入
    for i, cast_node in enumerate(cast_nodes):
        model.graph.node.insert(i, cast_node)

    return len(inits_to_convert)


def _convert_fp32_weights_to_bf16(model: onnx.ModelProto) -> int:
    """量子化されなかった fp32 initializer を bf16 に変換しサイズを削減する。

    _convert_fp32_weights_to_fp16 の bf16 版。torch.bfloat16 経由で変換し、
    int16 の raw bytes として BFLOAT16 TensorProto に格納する。
    消費ノードの手前に Cast(bf16→fp32) ノードを挿入する。

    Returns:
        変換した initializer の数。
    """
    quant_param_names: set[str] = set()
    for node in model.graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
            for i in range(1, len(node.input)):
                quant_param_names.add(node.input[i])

    inits_to_convert: list[str] = []
    for init in model.graph.initializer:
        if init.data_type != onnx.TensorProto.FLOAT:
            continue
        numel = int(np.prod(init.dims)) if init.dims else 1
        if init.name in quant_param_names:
            continue
        if numel < FP16_WEIGHT_MIN_ELEMENTS:
            continue
        arr = onnx.numpy_helper.to_array(init)
        # bf16 オーバーフローチェック (bf16 は fp32 と同じ指数域なので inf は発生しない)
        _ = arr  # bf16 は fp32 の上位 16bit なので overflow しない
        inits_to_convert.append(init.name)

    if not inits_to_convert:
        return 0

    # initializer を bf16 に書き換え (torch 経由)
    for init in model.graph.initializer:
        if init.name not in inits_to_convert:
            continue
        arr_fp32 = onnx.numpy_helper.to_array(init)
        t_bf16 = torch.from_numpy(arr_fp32).bfloat16()
        # bf16 を int16 の raw データとして格納
        raw_int16 = t_bf16.view(torch.int16).numpy()
        new_init = onnx.TensorProto()
        new_init.data_type = onnx.TensorProto.BFLOAT16
        new_init.name = init.name
        new_init.dims.extend(arr_fp32.shape)
        new_init.raw_data = raw_int16.tobytes()
        init.CopyFrom(new_init)

    # bf16 initializer ごとに Cast(bf16→fp32) ノードを作成し、消費ノードの入力を差し替え
    cast_nodes: list[onnx.NodeProto] = []
    for orig_name in inits_to_convert:
        cast_output = orig_name + "_fp32_from_bf16"
        cast_node = onnx.helper.make_node(
            "Cast",
            inputs=[orig_name],
            outputs=[cast_output],
            to=onnx.TensorProto.FLOAT,
            name=f"Cast_bf16_to_fp32_{orig_name}",
        )
        cast_nodes.append(cast_node)
        for node in model.graph.node:
            for i, inp in enumerate(node.input):
                if inp == orig_name:
                    node.input[i] = cast_output

    for i, cast_node in enumerate(cast_nodes):
        model.graph.node.insert(i, cast_node)

    return len(inits_to_convert)


def export_int8_onnx(
    fp32_path: str,
    int8_path: str,
    calib_samples: int,
    *,
    num_classes: int,
    class_names: list[str],
    residual_dtype: str = "fp16",
) -> None:
    """fp32 ONNX から INT8 静的量子化モデルを生成し、動的バッチ化と検証を行う。

    residual_dtype に "fp16" または "bf16" を指定することで、INT8 量子化から
    除外された残余重みの保存形式を切り替えられる。

    量子化精度を確保するため以下の対策を実施する:
    1. quant_pre_process で ONNX shape inference + model optimization を実行
       (symbolic shape inference は動的バッチで失敗するため skip)
    2. 量子化対象を Conv, Gemm に限定
    3. Depthwise Conv, ConvMulFusion (BN-folded Conv), Gemm (embedding head),
       GWAP の score conv は量子化から除外 (精度劣化が大きいため)
    4. reduce_range=False で 8-bit フルレンジを使用
    5. キャリブレーション手法に MinMax を使用

    fp32 モデルと同じクラス名メタデータを metadata_props に埋め込む。
    """
    label = f"INT8+{residual_dtype}"
    print(f"Running {label} quantization with calibration...")
    print(f"  calibration samples: {calib_samples}")
    print(f"  quantized op types: {OP_TYPES_TO_QUANTIZE}")

    # ── 前処理: shape inference + model optimization ──
    preprocessed_path = fp32_path.replace(".onnx", "_preproc.onnx")
    print(f"  Pre-processing for quantization: {preprocessed_path}")
    quant_pre_process(
        input_model=fp32_path,
        output_model_path=preprocessed_path,
        skip_symbolic_shape=True,
    )

    # ── INT8 量子化に敏感なノードを特定して除外 ──
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

    # 前処理済み中間ファイルを削除
    if os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)

    # INT8 モデルの妥当性チェック
    onnx_int8_model = onnx.load(int8_path)
    onnx.checker.check_model(onnx_int8_model)

    # INT8 モデルにも Reshape 修正 + 動的バッチ宣言を適用
    print(f"  Fixing {label} model batch dimensions...")
    fix_hardcoded_batch_in_reshapes(
        onnx_int8_model, int8_path, rng=np.random.default_rng(1), label=label
    )
    make_batch_dim_dynamic(onnx_int8_model)

    # 量子化されなかった fp32 残余重みを指定 dtype に変換してサイズを削減
    if residual_dtype == "bf16":
        residual_count = _convert_fp32_weights_to_bf16(onnx_int8_model)
    else:
        residual_count = _convert_fp32_weights_to_fp16(onnx_int8_model)
    print(
        f"  Converted {residual_count} residual weight initializers to {residual_dtype}"
    )

    # metadata_props にクラス名を埋め込む (fp32 モデルと同じ)
    _embed_class_metadata(onnx_int8_model, class_names)
    print(f"  Embedded {len(class_names)} class names in metadata_props")

    onnx.save(onnx_int8_model, int8_path)

    # 動的バッチ検証
    print(f"  Verifying {label} dynamic batch with random batch sizes...")
    verify_dynamic_batch(
        int8_path, num_classes=num_classes, rng=np.random.default_rng(1), label=label
    )
    print(f"  {label} ONNX model saved and verified: {int8_path}")

    # 出力品質の簡易検証 (実画像 1 枚)
    print(f"Verifying {label} ONNX output vs fp32 ONNX (real image, logits)...")
    real_sample = _get_real_sample()
    sess_fp32 = ort.InferenceSession(fp32_path)
    sess_int8 = ort.InferenceSession(int8_path)
    fp32_out = sess_fp32.run(None, {"input": real_sample})[0]
    int8_out = sess_int8.run(None, {"input": real_sample})[0]
    compare_outputs(fp32_out, int8_out, label=f"fp32 vs {label}", warn_threshold=0.05)


def _load_eval_samples(
    num_samples: int,
    *,
    calib_seed: int = 42,
) -> list[np.ndarray]:
    """キャリブレーションと重複しない評価用サンプルをロードする。

    キャリブレーションで使われたインデックス (seed=42) を避けて、
    別のシードで評価用インデックスを選択する。
    """
    from datasets import load_dataset

    transform = transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    print("  Loading evaluation dataset...")
    raw = load_dataset("yayoimizuha/helloproject-face-webdatasets")
    dataset = raw["train"]
    dataset_len = len(dataset)

    # キャリブレーションで使ったインデックスを除外
    calib_rng = np.random.default_rng(calib_seed)
    calib_indices = set(
        calib_rng.choice(
            dataset_len, size=min(DEFAULT_CALIB_SAMPLES, dataset_len), replace=False
        )
    )

    eval_rng = np.random.default_rng(123)
    all_indices = np.arange(dataset_len)
    available = np.array([i for i in all_indices if i not in calib_indices])
    eval_indices = eval_rng.choice(
        available, size=min(num_samples, len(available)), replace=False
    )

    samples: list[np.ndarray] = []
    for idx in tqdm(eval_indices, desc="  Preparing eval data"):
        item = dataset[int(idx)]
        img = item["image"].convert("RGB")
        tensor = transform(img).unsqueeze(0).numpy().astype(np.float32)
        samples.append(tensor)

    print(f"  Evaluation: {len(samples)} samples prepared (disjoint from calibration)")
    return samples


def evaluate_model_quality(
    fp32_path: str,
    target_path: str,
    *,
    num_samples: int = DEFAULT_EVAL_SAMPLES,
    label: str = "target",
) -> None:
    """実データで fp32 と任意の変換済みモデルの logits 出力を比較する。

    fp32 を基準に以下の指標で精度劣化を評価する:
    - argmax 一致率: 同じ最頻クラスを予測するか
    - max abs diff: logits ベクトルの最大絶対誤差
    - Top-1 logit 差: fp32 の argmax クラスでの logit 値の絶対差
    """
    print(f"\n{'=' * 60}")
    print(f"Quality evaluation: fp32 vs {label} ({num_samples} real samples)")
    print(f"{'=' * 60}")

    samples = _load_eval_samples(num_samples)
    if not samples:
        print("  [WARN] No evaluation samples available, skipping.")
        return

    sess_fp32 = ort.InferenceSession(fp32_path)
    sess_target = ort.InferenceSession(target_path)

    argmax_matches: list[bool] = []
    max_abs_diffs: list[float] = []
    top1_logit_diffs: list[float] = []

    for sample in tqdm(samples, desc=f"  Evaluating fp32 vs {label}"):
        fp32_l = sess_fp32.run(None, {"input": sample})[0][0]
        tgt_l = sess_target.run(None, {"input": sample})[0][0]

        argmax_matches.append(bool(fp32_l.argmax() == tgt_l.argmax()))
        max_abs_diffs.append(float(np.max(np.abs(fp32_l - tgt_l))))
        top1_idx = int(fp32_l.argmax())
        top1_logit_diffs.append(float(abs(fp32_l[top1_idx] - tgt_l[top1_idx])))

    match_arr = np.array(argmax_matches)
    mad_arr = np.array(max_abs_diffs)
    top1_arr = np.array(top1_logit_diffs)
    match_rate = float(match_arr.mean())

    print(f"\n  --- Argmax Agreement (fp32 vs {label}) ---")
    print(f"  agreement: {match_arr.sum()}/{len(match_arr)} ({match_rate:.1%})")

    print(f"\n  --- Max Abs Diff on Logits ---")
    print(f"  mean:   {mad_arr.mean():.4f}")
    print(f"  std:    {mad_arr.std():.4f}")
    print(f"  min:    {mad_arr.min():.4f}")
    print(f"  max:    {mad_arr.max():.4f}")

    print(f"\n  --- Top-1 Logit Difference ---")
    print(f"  mean:   {top1_arr.mean():.4f}")
    print(f"  std:    {top1_arr.std():.4f}")
    print(f"  max:    {top1_arr.max():.4f}")

    print(f"\n  --- Summary ---")
    if match_rate < 0.95:
        print(
            f"  [WARN] argmax agreement {match_rate:.1%} < 95%. "
            f"Significant accuracy degradation detected."
        )
    elif match_rate < 1.0:
        print(
            f"  [INFO] argmax agreement {match_rate:.1%}. "
            f"Minor disagreements on {int((~match_arr).sum())} sample(s)."
        )
    else:
        print(f"  Quality looks good (100% argmax agreement).")

    if mad_arr.max() > 0.5:
        print(f"  [WARN] Max logit diff {mad_arr.max():.4f} > 0.5.")


def print_file_sizes(model_paths: dict[str, str]) -> None:
    """エクスポートされた各モデルのファイルサイズを表示する。

    Args:
        model_paths: {label: path} の辞書。例: {"fp32": "model.onnx", "fp16": "model_fp16.onnx"}
    """
    fp32_mb = None
    print(f"\nFile sizes:")
    for label, path in model_paths.items():
        if not os.path.exists(path):
            print(f"  {label}: (not found)")
            continue
        mb = os.path.getsize(path) / (1024 * 1024)
        if label == "fp32":
            fp32_mb = mb
        ratio = (
            f" ({mb / fp32_mb * 100:.0f}% of fp32)"
            if fp32_mb and label != "fp32"
            else ""
        )
        print(f"  {label:12s}: {mb:6.1f} MB{ratio}")


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

    # Step 1: クラス名をデータセットから取得
    print("Loading class names from dataset...")
    from datasets import load_dataset

    raw = load_dataset("yayoimizuha/helloproject-face-webdatasets")
    label_feature = raw["train"].features.get("label")
    if hasattr(label_feature, "names"):
        class_names = label_feature.names
    else:
        num_classes_fallback = max(raw["train"]["label"]) + 1
        class_names = [str(i) for i in range(num_classes_fallback)]
    num_classes = len(class_names)
    print(f"  {num_classes} class names loaded")

    meta = dict(num_classes=num_classes, class_names=class_names)

    # Step 2: モデルロード
    print(f"Loading checkpoint: {checkpoint_path}")
    cls_model = load_classification_model(checkpoint_path, args.num_classes)

    # Step 3: fp32 ONNX (ベース — 他の変換の入力になる)
    export_fp32_onnx(cls_model, paths["fp32"], opset=args.opset, **meta)

    # Step 4: fp16 全変換
    export_fp16_onnx(paths["fp32"], paths["fp16"], **meta)

    # Step 5: bf16 全変換
    export_bf16_onnx(paths["fp32"], paths["bf16"], **meta)

    # Step 6: fp16+INT8 量子化
    export_int8_onnx(
        paths["fp32"],
        paths["fp16int8"],
        calib_samples=args.calib_samples,
        residual_dtype="fp16",
        **meta,
    )

    # Step 7: bf16+INT8 量子化
    export_int8_onnx(
        paths["fp32"],
        paths["bf16int8"],
        calib_samples=args.calib_samples,
        residual_dtype="bf16",
        **meta,
    )

    # Step 8: 各バリアントの品質評価
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

"""ONNX inference script with mistake ranking and wrong-sample saving.

Runs inference over the entire training dataset using an exported ONNX model,
builds a confusion-based mistake ranking, and saves misclassified images in a
structured directory layout.

Directory layout of saved results::

    <output_dir>/
        wrong_predictions.csv          # all misclassified samples (metadata)
        confusion_ranking.csv          # (true_class, pred_class, count) sorted by count desc
        class_accuracy.csv             # per-class accuracy sorted by accuracy asc
        wrong_images/
            <pred_class>/              # directory named after the predicted (wrong) label
                <true_class>_001.jpg   # original image; counter is per (pred, true) pair
                <true_class>_002.jpg
                ...

Execution Providers::

    --provider auto        Tries TensorrtExecutionProvider → CUDAExecutionProvider
                           → CPUExecutionProvider in order; uses the first available.
    --provider cpu         CPUExecutionProvider only.
    --provider cuda        CUDAExecutionProvider (falls back to CPU if unavailable).
    --provider tensorrt    TensorrtExecutionProvider → CUDAExecutionProvider fallback
                           (TensorRT handles supported ops; CUDA handles the rest).

TensorRT options (only used when provider is ``tensorrt`` or ``auto`` selects it)::

    --trt-fp16             Enable FP16 precision in TensorRT engine.
    --trt-int8             Enable INT8 precision in TensorRT engine.
                           Use with a quantized ONNX (fp16int8 / bf16int8) for best results.
    --trt-cache-dir DIR    Directory for caching compiled TensorRT engines
                           (default: <output_dir>/trt_cache).  Caching dramatically
                           speeds up subsequent runs with the same model.

Usage::

    # CPU (default auto)
    python hp_finetune/infer_onnx.py \\
        --onnx hp_finetune/work_dirs/20260320_125319/model_best.onnx

    # CUDA
    python hp_finetune/infer_onnx.py \\
        --onnx hp_finetune/work_dirs/20260320_125319/model_best.onnx \\
        --provider cuda --device-id 0

    # TensorRT with FP16 + engine cache
    python hp_finetune/infer_onnx.py \\
        --onnx hp_finetune/work_dirs/20260320_125319/model_best_fp16int8.onnx \\
        --provider tensorrt --trt-fp16 --trt-cache-dir /tmp/trt_cache

    # Custom output dir and ranking size
    python hp_finetune/infer_onnx.py \\
        --onnx hp_finetune/work_dirs/20260320_125319/model_best.onnx \\
        --output-dir results/infer_onnx \\
        --top-k 30
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import onnxruntime as ort
from scipy.special import softmax
from tqdm import tqdm

from hp_finetune.data_utils import get_inference_transform, load_train_dataset

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
DEFAULT_TOP_K = 20
DEFAULT_TOP_CLASS_K = 0  # 0 = 全クラス表示
_SAFE_MAX_LEN = 40  # max chars for directory / file name components

# Provider names as used by ORT
_PROVIDER_CPU = "CPUExecutionProvider"
_PROVIDER_CUDA = "CUDAExecutionProvider"
_PROVIDER_TRT = "TensorrtExecutionProvider"

# Priority order for "auto" provider selection
_AUTO_PRIORITY = [_PROVIDER_TRT, _PROVIDER_CUDA, _PROVIDER_CPU]


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def _safe_name(name: str, max_len: int = _SAFE_MAX_LEN) -> str:
    """Sanitize a class name to a filesystem-safe string."""
    safe = name.replace("/", "_").replace("\\", "_").replace("\0", "_")
    return safe[:max_len]


# ──────────────────────────────────────────────
# Session builder
# ──────────────────────────────────────────────
def build_session(
    onnx_path: str,
    provider: str,
    device_id: int,
    *,
    trt_fp16: bool = False,
    trt_int8: bool = False,
    trt_cache_dir: str | None = None,
) -> ort.InferenceSession:
    """Create an ORT InferenceSession with the requested execution provider.

    Args:
        onnx_path:     Path to the ONNX model file.
        provider:      One of ``"auto"``, ``"cpu"``, ``"cuda"``, ``"tensorrt"``.
        device_id:     GPU device index (used by CUDA and TensorRT providers).
        trt_fp16:      Enable TensorRT FP16 engine (TensorRT provider only).
        trt_int8:      Enable TensorRT INT8 engine (TensorRT provider only).
        trt_cache_dir: Directory for TensorRT engine cache. ``None`` disables
                       caching.

    Returns:
        A ready-to-use :class:`ort.InferenceSession`.
    """
    available = ort.get_available_providers()

    def _make_cuda_options() -> tuple[str, dict]:
        opts = {
            "device_id": device_id,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "DEFAULT",
            "do_copy_in_default_stream": True,
        }
        return _PROVIDER_CUDA, opts

    def _make_trt_options() -> tuple[str, dict]:
        opts: dict = {
            "device_id": device_id,
            "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,  # 2 GiB
            "trt_fp16_enable": trt_fp16,
            "trt_int8_enable": trt_int8,
            "trt_engine_cache_enable": trt_cache_dir is not None,
        }
        if trt_cache_dir is not None:
            os.makedirs(trt_cache_dir, exist_ok=True)
            opts["trt_engine_cache_path"] = trt_cache_dir
        return _PROVIDER_TRT, opts

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    provider_str = provider.lower()

    if provider_str == "cpu":
        providers = [_PROVIDER_CPU]
        print(f"  Execution provider: CPUExecutionProvider")

    elif provider_str == "cuda":
        if _PROVIDER_CUDA not in available:
            print(
                f"  [WARN] CUDAExecutionProvider not available "
                f"(available: {available}). Falling back to CPU."
            )
            providers = [_PROVIDER_CPU]
        else:
            cuda_name, cuda_opts = _make_cuda_options()
            providers = [(cuda_name, cuda_opts), _PROVIDER_CPU]
            print(
                f"  Execution provider: CUDAExecutionProvider "
                f"(device_id={device_id}) + CPU fallback"
            )

    elif provider_str == "tensorrt":
        if _PROVIDER_TRT not in available:
            print(
                f"  [WARN] TensorrtExecutionProvider not available "
                f"(available: {available}). Falling back to CUDA/CPU."
            )
            if _PROVIDER_CUDA in available:
                cuda_name, cuda_opts = _make_cuda_options()
                providers = [(cuda_name, cuda_opts), _PROVIDER_CPU]
                print(
                    f"  Execution provider: CUDAExecutionProvider "
                    f"(device_id={device_id}) + CPU fallback"
                )
            else:
                providers = [_PROVIDER_CPU]
                print("  Execution provider: CPUExecutionProvider")
        else:
            trt_name, trt_opts = _make_trt_options()
            cuda_name, cuda_opts = _make_cuda_options()
            providers = [
                (trt_name, trt_opts),
                (cuda_name, cuda_opts),
                _PROVIDER_CPU,
            ]
            flags = []
            if trt_fp16:
                flags.append("fp16")
            if trt_int8:
                flags.append("int8")
            prec_str = f" [{', '.join(flags)}]" if flags else ""
            cache_str = f", cache={trt_cache_dir}" if trt_cache_dir else ""
            print(
                f"  Execution provider: TensorrtExecutionProvider"
                f"{prec_str} (device_id={device_id}{cache_str})"
                f" + CUDA + CPU fallback"
            )

    elif provider_str == "auto":
        # Pick the highest-priority available provider
        selected = next((p for p in _AUTO_PRIORITY if p in available), _PROVIDER_CPU)
        if selected == _PROVIDER_TRT:
            trt_name, trt_opts = _make_trt_options()
            cuda_name, cuda_opts = _make_cuda_options()
            providers = [
                (trt_name, trt_opts),
                (cuda_name, cuda_opts),
                _PROVIDER_CPU,
            ]
            print(
                f"  Execution provider (auto): TensorrtExecutionProvider "
                f"(device_id={device_id}) + CUDA + CPU fallback"
            )
        elif selected == _PROVIDER_CUDA:
            cuda_name, cuda_opts = _make_cuda_options()
            providers = [(cuda_name, cuda_opts), _PROVIDER_CPU]
            print(
                f"  Execution provider (auto): CUDAExecutionProvider "
                f"(device_id={device_id}) + CPU fallback"
            )
        else:
            providers = [_PROVIDER_CPU]
            print("  Execution provider (auto): CPUExecutionProvider")

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. Choose from: auto, cpu, cuda, tensorrt"
        )

    sess = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)
    actual = sess.get_providers()
    print(f"  Active providers: {actual}")

    # ── Detect unexpected fallback ──────────────────────────────────
    # ORT silently falls back to CPU when a GPU provider fails to initialise
    # (e.g. no CUDA-capable GPU, missing cuDNN/TensorRT libraries).
    # Treat this as an error for explicit --provider cuda/tensorrt so the
    # user is not surprised by silent CPU execution.
    if provider_str in ("cuda", "tensorrt"):
        requested_gpu = _PROVIDER_CUDA if provider_str == "cuda" else _PROVIDER_TRT
        if requested_gpu not in actual:
            raise RuntimeError(
                f"Requested provider '{provider_str}' is not active after session "
                f"creation (active: {actual}).\n"
                f"This usually means the required GPU libraries are missing or no "
                f"GPU is detected. Check that:\n"
                f"  - A CUDA-capable GPU is available on this machine\n"
                f"  - cuDNN (CUDA provider) or TensorRT + cuDNN (TRT provider) "
                f"libraries are installed and on LD_LIBRARY_PATH\n"
                f"  - CUDA driver version is compatible with the installed toolkit\n"
                f"Use --provider cpu to run on CPU instead."
            )

    return sess


def _load_class_names_from_onnx(sess: ort.InferenceSession) -> list[str]:
    """Read class names embedded in ONNX metadata_props."""
    meta = sess.get_modelmeta().custom_metadata_map
    if "class_names" not in meta:
        raise RuntimeError(
            "ONNX model does not contain 'class_names' in metadata_props.\n"
            "Re-export the model with export_onnx.py which embeds class metadata."
        )
    return json.loads(meta["class_names"])


def _load_image_config_from_onnx(
    sess: ort.InferenceSession,
) -> tuple[int, list[float], list[float]]:
    """Extract input_size, mean, std from ONNX metadata if present, else use defaults."""
    meta = sess.get_modelmeta().custom_metadata_map

    mean: list[float] = json.loads(meta.get("imagenet_mean", "[0.485, 0.456, 0.406]"))
    std: list[float] = json.loads(meta.get("imagenet_std", "[0.229, 0.224, 0.225]"))

    # Prefer metadata key; fallback to ONNX input shape; final fallback to 224
    if "input_size" in meta:
        input_size: int = int(meta["input_size"])
    else:
        inp = sess.get_inputs()[0]
        if (
            inp.shape
            and len(inp.shape) >= 4
            and isinstance(inp.shape[2], int)
            and inp.shape[2] > 0
        ):
            input_size = inp.shape[2]
        else:
            input_size = 224

    return input_size, mean, std


def _load_mahal_threshold_from_onnx(sess: ort.InferenceSession) -> float | None:
    """Read Mahalanobis threshold from ONNX metadata_props.

    Returns the threshold value if present, or ``None`` if the model was
    exported without Mahalanobis anomaly detection.
    """
    meta = sess.get_modelmeta().custom_metadata_map
    val = meta.get("mahal_threshold")
    return float(val) if val is not None else None


def _has_anomaly_output(sess: ort.InferenceSession) -> bool:
    """Return True if the ONNX model outputs an 'anomaly_score' tensor."""
    return any(o.name == "anomaly_score" for o in sess.get_outputs())


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────
def run_inference(
    sess: ort.InferenceSession,
    class_names: list[str],
    input_size: int,
    mean: list[float],
    std: list[float],
    *,
    has_anomaly: bool = False,
) -> tuple[list[int], list[int], list[np.ndarray], list[float], list[float | None]]:
    """Run inference over the full dataset.

    Returns:
        true_labels:    list of ground-truth integer class indices
        pred_labels:    list of predicted integer class indices
        images:         list of original PIL images (RGB, not transformed)
        confidences:    list of softmax confidence for the predicted class
        anomaly_scores: list of Mahalanobis anomaly scores, or ``None`` per
                        sample when the model has no anomaly detection output
    """
    from hp_finetune.data_utils import ImageConfig

    img_cfg = ImageConfig(input_size=input_size, mean=mean, std=std)
    transform = get_inference_transform(img_cfg)

    print("Loading dataset...")
    dataset = load_train_dataset()
    print(f"  Dataset size: {len(dataset)} samples")

    input_name = sess.get_inputs()[0].name  # "input"

    true_labels: list[int] = []
    pred_labels: list[int] = []
    images: list = []
    confidences: list[float] = []
    anomaly_scores: list[float | None] = []

    for item in tqdm(dataset, desc="Inference", unit="img"):
        img = item["image"].convert("RGB")
        true_label: int = int(item["label"])

        x = transform(img).unsqueeze(0).numpy().astype(np.float32)
        outputs = sess.run(None, {input_name: x})
        logits = outputs[0][0]  # (num_classes,)
        anomaly = float(outputs[1][0]) if has_anomaly else None

        pred_label = int(np.argmax(logits))
        probs = softmax(logits.astype(np.float64))
        confidence = float(probs[pred_label])

        true_labels.append(true_label)
        pred_labels.append(pred_label)
        images.append(img)
        confidences.append(confidence)
        anomaly_scores.append(anomaly)

    return true_labels, pred_labels, images, confidences, anomaly_scores


# ──────────────────────────────────────────────
# Saving wrong samples
# ──────────────────────────────────────────────
def save_wrong_images(
    true_labels: list[int],
    pred_labels: list[int],
    images: list,
    confidences: list[float],
    class_names: list[str],
    output_dir: str,
    *,
    anomaly_scores: list[float | None] | None = None,
    mahal_threshold: float | None = None,
) -> list[dict]:
    """Save misclassified images under <output_dir>/wrong_images/<pred>/<true>_NNN.jpg.

    Returns a list of dicts (one per wrong sample) for CSV output.
    Includes ``anomaly_score`` and ``is_anomaly`` columns when *anomaly_scores*
    is provided.
    """
    wrong_images_dir = os.path.join(output_dir, "wrong_images")

    # Counter to track the sequential number per (pred_class, true_class) pair
    pair_counters: dict[tuple[int, int], int] = defaultdict(int)

    records: list[dict] = []

    _anomaly_scores = anomaly_scores or [None] * len(true_labels)

    for idx, (true_lbl, pred_lbl, img, conf, anomaly) in enumerate(
        zip(true_labels, pred_labels, images, confidences, _anomaly_scores)
    ):
        if true_lbl == pred_lbl:
            continue

        true_name = _safe_name(class_names[true_lbl])
        pred_name = _safe_name(class_names[pred_lbl])

        # Directory: wrong_images/<pred_class>/
        img_dir = os.path.join(wrong_images_dir, pred_name)
        os.makedirs(img_dir, exist_ok=True)

        # Filename: <true_class>_NNN.jpg
        pair_counters[(pred_lbl, true_lbl)] += 1
        seq = pair_counters[(pred_lbl, true_lbl)]
        filename = f"{true_name}_{seq:03d}.jpg"
        filepath = os.path.join(img_dir, filename)

        img.save(filepath, format="JPEG", quality=90)

        record: dict = {
            "dataset_index": idx,
            "true_label_idx": true_lbl,
            "pred_label_idx": pred_lbl,
            "true_class": class_names[true_lbl],
            "pred_class": class_names[pred_lbl],
            "confidence": f"{conf:.6f}",
            "image_path": os.path.relpath(filepath, output_dir),
        }
        if anomaly is not None:
            record["anomaly_score"] = f"{anomaly:.6f}"
            if mahal_threshold is not None:
                record["is_anomaly"] = "1" if anomaly > mahal_threshold else "0"
        records.append(record)

    return records


# ──────────────────────────────────────────────
# CSV writers
# ──────────────────────────────────────────────
def save_wrong_predictions_csv(records: list[dict], output_dir: str) -> str:
    """Write wrong_predictions.csv and return its path.

    Columns ``anomaly_score`` and ``is_anomaly`` are written when present in
    *records* (i.e. when the model was exported with Mahalanobis detection).
    """
    path = os.path.join(output_dir, "wrong_predictions.csv")
    if not records:
        print("  No wrong predictions — skipping CSV.")
        return path

    # Base columns always present
    fieldnames = [
        "dataset_index",
        "true_label_idx",
        "pred_label_idx",
        "true_class",
        "pred_class",
        "confidence",
        "image_path",
    ]
    # Append anomaly columns if the first record has them
    if "anomaly_score" in records[0]:
        fieldnames.append("anomaly_score")
    if "is_anomaly" in records[0]:
        fieldnames.append("is_anomaly")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    return path


def save_confusion_ranking_csv(
    true_labels: list[int],
    pred_labels: list[int],
    class_names: list[str],
    output_dir: str,
) -> tuple[str, list[tuple]]:
    """Build a (true_class, pred_class, count) ranking sorted by count desc.

    Returns (csv_path, ranking_list).
    """
    pair_counter: Counter = Counter()
    for t, p in zip(true_labels, pred_labels):
        if t != p:
            pair_counter[(t, p)] += 1

    ranking = [
        (class_names[t], class_names[p], cnt)
        for (t, p), cnt in pair_counter.most_common()
    ]

    path = os.path.join(output_dir, "confusion_ranking.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_class", "pred_class", "count"])
        writer.writerows(ranking)

    return path, ranking


def save_class_accuracy_csv(
    true_labels: list[int],
    pred_labels: list[int],
    class_names: list[str],
    output_dir: str,
) -> str:
    """Compute per-class accuracy and save sorted by accuracy ascending."""
    class_total: Counter = Counter(true_labels)
    class_correct: Counter = Counter(
        t for t, p in zip(true_labels, pred_labels) if t == p
    )

    rows = []
    for cls_idx, name in enumerate(class_names):
        total = class_total[cls_idx]
        if total == 0:
            continue
        correct = class_correct[cls_idx]
        accuracy = correct / total
        rows.append(
            {
                "class_idx": cls_idx,
                "class_name": name,
                "total": total,
                "correct": correct,
                "wrong": total - correct,
                "accuracy": f"{accuracy:.4f}",
            }
        )

    # Sort by accuracy ascending (most confused classes first)
    rows.sort(key=lambda r: float(r["accuracy"]))

    path = os.path.join(output_dir, "class_accuracy.csv")
    fieldnames = ["class_idx", "class_name", "total", "correct", "wrong", "accuracy"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


# ──────────────────────────────────────────────
# Summary printer
# ──────────────────────────────────────────────
def print_summary(
    true_labels: list[int],
    pred_labels: list[int],
    class_names: list[str],
    ranking: list[tuple],
    top_k: int,
    *,
    anomaly_scores: list[float | None] | None = None,
    mahal_threshold: float | None = None,
    top_class_k: int = DEFAULT_TOP_CLASS_K,
) -> None:
    from collections import Counter as _Counter

    total = len(true_labels)
    correct = sum(t == p for t, p in zip(true_labels, pred_labels))
    wrong = total - correct
    accuracy = correct / total if total > 0 else 0.0

    print()
    print("=" * 60)
    print("Inference Summary")
    print("=" * 60)
    print(f"  Total samples   : {total}")
    print(f"  Correct         : {correct}")
    print(f"  Wrong           : {wrong}")
    print(f"  Top-1 Accuracy  : {accuracy * 100:.2f}%")

    # ── Anomaly detection summary ──────────────────────────────────
    if anomaly_scores is not None and any(s is not None for s in anomaly_scores):
        valid_scores = [s for s in anomaly_scores if s is not None]
        mean_score = float(np.mean(valid_scores))
        print()
        print("Anomaly Detection (Mahalanobis)")
        print("-" * 60)
        if mahal_threshold is not None:
            anomaly_count = sum(1 for s in valid_scores if s > mahal_threshold)
            anomaly_rate = anomaly_count / len(valid_scores) * 100
            print(f"  threshold        : {mahal_threshold:.4f}")
            print(
                f"  anomaly detected : {anomaly_count} / {len(valid_scores)} samples"
                f" ({anomaly_rate:.2f}%)"
            )
        print(f"  mean score (all) : {mean_score:.4f}")
        print(f"  min / max score  : {min(valid_scores):.4f} / {max(valid_scores):.4f}")

    # ── Per-class accuracy ─────────────────────────────────────────
    class_total = _Counter(true_labels)
    class_correct = _Counter(t for t, p in zip(true_labels, pred_labels) if t == p)

    rows = []
    for cls_idx, name in enumerate(class_names):
        n = class_total[cls_idx]
        if n == 0:
            continue
        c = class_correct[cls_idx]
        rows.append((name, c, n, c / n * 100))

    # Sort by accuracy ascending (most confused first)
    rows.sort(key=lambda r: r[3])

    # 表示行数を制限（top_class_k=0 は全件）
    display_rows = rows if top_class_k <= 0 else rows[:top_class_k]
    title_suffix = (
        f" (worst {top_class_k})" if top_class_k > 0 and top_class_k < len(rows) else ""
    )

    name_w = max(len(r[0]) for r in display_rows) if display_rows else 10
    name_w = max(name_w, len("Class"))

    print()
    print(f"Per-class Accuracy{title_suffix} (sorted by accuracy, worst first)")
    print("-" * (name_w + 30))
    print(f"  {'Class':<{name_w}}  {'Correct':>7}  {'Total':>7}  {'Accuracy':>8}")
    print(f"  {'-' * name_w}  {'-' * 7}  {'-' * 7}  {'-' * 8}")
    for name, c, n, pct in display_rows:
        print(f"  {name:<{name_w}}  {c:>7}  {n:>7}  {pct:>7.2f}%")
    print("-" * (name_w + 30))

    print()
    print(f"Top-{top_k} Most Confused Class Pairs (true → predicted)")
    print("-" * 60)
    for rank, (true_name, pred_name, cnt) in enumerate(ranking[:top_k], 1):
        print(f"  {rank:3d}. {true_name}  →  {pred_name}  ({cnt} samples)")
    print("=" * 60)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ONNX inference over the full dataset, "
            "rank mistakes, and save wrong samples."
        )
    )
    parser.add_argument(
        "--onnx",
        type=str,
        required=True,
        help=(
            "Path to the ONNX model (e.g. "
            "hp_finetune/work_dirs/20260320_125319/model_best.onnx)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to save results. "
            "Defaults to <onnx_dir>/infer_results/<onnx_stem>."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of top confused class pairs to display (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--top-class-k",
        type=int,
        default=DEFAULT_TOP_CLASS_K,
        help=(
            "Number of per-class accuracy rows to display in the summary, "
            "sorted by accuracy ascending (worst first). "
            "0 = show all classes (default)."
        ),
    )
    # ── Provider options ────────────────────────────────────────────
    parser.add_argument(
        "--provider",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "tensorrt"],
        help=(
            "Execution provider: "
            "'auto' tries TensorRT→CUDA→CPU in order; "
            "'cpu' forces CPU; "
            "'cuda' uses CUDAExecutionProvider; "
            "'tensorrt' uses TensorrtExecutionProvider with CUDA+CPU fallback. "
            "(default: auto)"
        ),
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="GPU device index for CUDA / TensorRT providers (default: 0)",
    )
    # ── TensorRT-specific options ───────────────────────────────────
    trt_group = parser.add_argument_group(
        "TensorRT options",
        "These options are only used when --provider is 'tensorrt' (or 'auto' selects it).",
    )
    trt_group.add_argument(
        "--trt-fp16",
        action="store_true",
        default=False,
        help="Enable FP16 precision in the TensorRT engine.",
    )
    trt_group.add_argument(
        "--trt-int8",
        action="store_true",
        default=False,
        help=(
            "Enable INT8 precision in the TensorRT engine. "
            "Best used with a quantized ONNX (e.g. model_best_fp16int8.onnx)."
        ),
    )
    trt_group.add_argument(
        "--trt-cache-dir",
        type=str,
        default=None,
        help=(
            "Directory for caching compiled TensorRT engines. "
            "Defaults to <output_dir>/trt_cache when not specified."
        ),
    )
    return parser.parse_args()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.onnx):
        print(f"Error: ONNX model not found: {args.onnx}")
        sys.exit(1)

    # Resolve output directory
    if args.output_dir is None:
        onnx_dir = os.path.dirname(os.path.abspath(args.onnx))
        onnx_stem = os.path.splitext(os.path.basename(args.onnx))[0]
        output_dir = os.path.join(onnx_dir, "infer_results", onnx_stem)
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Resolve TensorRT cache dir (default: <output_dir>/trt_cache)
    trt_cache_dir = args.trt_cache_dir
    if trt_cache_dir is None and args.provider in ("tensorrt", "auto"):
        trt_cache_dir = os.path.join(output_dir, "trt_cache")

    # ── 1. Load ONNX model ──────────────────────────────────────────
    print(f"Loading ONNX model: {args.onnx}")
    sess = build_session(
        args.onnx,
        provider=args.provider,
        device_id=args.device_id,
        trt_fp16=args.trt_fp16,
        trt_int8=args.trt_int8,
        trt_cache_dir=trt_cache_dir,
    )
    class_names = _load_class_names_from_onnx(sess)
    input_size, mean, std = _load_image_config_from_onnx(sess)
    has_anomaly = _has_anomaly_output(sess)
    mahal_threshold = _load_mahal_threshold_from_onnx(sess) if has_anomaly else None
    print(
        f"  num_classes={len(class_names)}  "
        f"input_size={input_size}  mean={mean}  std={std}"
    )
    if has_anomaly:
        print(f"  Mahalanobis anomaly detection: enabled  threshold={mahal_threshold}")

    # ── 2. Run inference ────────────────────────────────────────────
    true_labels, pred_labels, images, confidences, anomaly_scores = run_inference(
        sess, class_names, input_size, mean, std, has_anomaly=has_anomaly
    )

    # ── 3. Save wrong images ────────────────────────────────────────
    print("Saving wrong images...")
    wrong_records = save_wrong_images(
        true_labels,
        pred_labels,
        images,
        confidences,
        class_names,
        output_dir,
        anomaly_scores=anomaly_scores,
        mahal_threshold=mahal_threshold,
    )
    print(f"  Saved {len(wrong_records)} wrong images")

    # ── 4. Save wrong_predictions.csv ──────────────────────────────
    csv_path = save_wrong_predictions_csv(wrong_records, output_dir)
    print(f"  Saved: {csv_path}")

    # ── 5. Save confusion_ranking.csv ──────────────────────────────
    ranking_path, ranking = save_confusion_ranking_csv(
        true_labels, pred_labels, class_names, output_dir
    )
    print(f"  Saved: {ranking_path}")

    # ── 6. Save class_accuracy.csv ─────────────────────────────────
    acc_path = save_class_accuracy_csv(
        true_labels, pred_labels, class_names, output_dir
    )
    print(f"  Saved: {acc_path}")

    # ── 7. Print summary ────────────────────────────────────────────
    print_summary(
        true_labels,
        pred_labels,
        class_names,
        ranking,
        args.top_k,
        anomaly_scores=anomaly_scores,
        mahal_threshold=mahal_threshold,
        top_class_k=args.top_class_k,
    )


if __name__ == "__main__":
    main()

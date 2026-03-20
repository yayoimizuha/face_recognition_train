"""Verification and quality evaluation utilities for exported ONNX models.

Provides:
- Dynamic batch shape verification
- Output comparison between model variants (max diff, cosine similarity)
- Multi-sample quality evaluation (argmax agreement, logit differences)
- File size reporting
"""

from __future__ import annotations

import os

import numpy as np
import onnxruntime as ort
from tqdm import tqdm

from hp_finetune.data_utils import ImageConfig, load_eval_samples

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
VERIFY_BATCH_MIN = 2
VERIFY_BATCH_MAX = 32
VERIFY_BATCH_COUNT = 5
FP32_MAX_DIFF_WARN_THRESHOLD = 1e-4
DEFAULT_EVAL_SAMPLES = 50


# ──────────────────────────────────────────────
# Dynamic batch verification
# ──────────────────────────────────────────────
def verify_dynamic_batch(
    onnx_path: str,
    *,
    input_size: int,
    num_classes: int,
    rng: np.random.Generator | None = None,
    label: str = "",
) -> None:
    """Run inference with several random batch sizes and assert correct output shape."""
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
        test_in = np.random.randn(bs, 3, input_size, input_size).astype(np.float32)
        out = sess.run(None, {"input": test_in})[0]
        assert out.shape == (bs, num_classes), (
            f"{label} batch={bs}: expected ({bs}, {num_classes}), got {out.shape}"
        )
    print(f"{prefix}Dynamic batch OK: tested batch sizes {batch_sizes}")


# ──────────────────────────────────────────────
# Output comparison
# ──────────────────────────────────────────────
def compare_outputs(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    label: str = "",
    warn_threshold: float = FP32_MAX_DIFF_WARN_THRESHOLD,
) -> tuple[float, float]:
    """Compute and display max abs diff and cosine similarity between two outputs.

    Returns:
        ``(max_abs_diff, cosine_similarity)``
    """
    max_diff = float(np.max(np.abs(reference - target)))
    cos_sim = float(
        np.dot(reference.flatten(), target.flatten())
        / (np.linalg.norm(reference) * np.linalg.norm(target))
    )
    suffix = f" ({label})" if label else ""
    print(f"  max abs diff{suffix}: {max_diff:.2e}")
    print(f"  cosine similarity{suffix}: {cos_sim:.6f}")
    if max_diff > warn_threshold:
        print(f"  [WARN] Large difference detected{suffix}")
    return max_diff, cos_sim


# ──────────────────────────────────────────────
# Multi-sample quality evaluation
# ──────────────────────────────────────────────
def evaluate_model_quality(
    fp32_path: str,
    target_path: str,
    *,
    img_cfg: ImageConfig,
    num_samples: int = DEFAULT_EVAL_SAMPLES,
    label: str = "target",
) -> None:
    """Compare logits of a converted model against fp32 on real data.

    Reports argmax agreement, max absolute difference, and top-1 logit
    difference statistics.
    """
    print(f"\n{'=' * 60}")
    print(f"Quality evaluation: fp32 vs {label} ({num_samples} real samples)")
    print(f"{'=' * 60}")

    samples = load_eval_samples(img_cfg, num_samples)
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


# ──────────────────────────────────────────────
# File size reporting
# ──────────────────────────────────────────────
def print_file_sizes(model_paths: dict[str, str]) -> None:
    """Print file sizes for all exported model variants.

    The fp32 variant is used as the baseline for percentage comparisons.
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

"""Data loading utilities for ONNX export calibration and evaluation.

Centralises dataset access, image transforms, and sample preparation so that
calibration, evaluation, and single-image verification share one code path.

All image-processing parameters (input_size, mean, std) are passed explicitly
rather than imported from a global config, so this module is backbone-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from onnxruntime.quantization import CalibrationDataReader
from torchvision import transforms
from tqdm import tqdm

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
DATASET_NAME = "yayoimizuha/helloproject-face-dataset"
DEFAULT_CALIB_SAMPLES = 200
DEFAULT_EVAL_SAMPLES = 50


# ──────────────────────────────────────────────
# Image preprocessing config
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class ImageConfig:
    """Holds all image-preprocessing parameters needed for ONNX inference."""

    input_size: int
    mean: list[float]
    std: list[float]


# ──────────────────────────────────────────────
# Shared transform
# ──────────────────────────────────────────────
def get_inference_transform(cfg: ImageConfig) -> transforms.Compose:
    """Return the canonical image transform used for ONNX inference."""
    return transforms.Compose(
        [
            transforms.Resize((cfg.input_size, cfg.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )


# ──────────────────────────────────────────────
# Dataset helpers
# ──────────────────────────────────────────────
def load_train_dataset():
    """Load the training split of the face dataset (lazy import)."""
    from datasets import load_dataset

    return load_dataset(DATASET_NAME)["train"]


def get_class_names_from_dataset() -> list[str]:
    """Retrieve class label names from the dataset feature metadata."""
    from datasets import load_dataset

    raw = load_dataset(DATASET_NAME)
    label_feature = raw["train"].features.get("label")
    if hasattr(label_feature, "names"):
        return label_feature.names
    # Fallback: generate numeric names
    num_classes = max(raw["train"]["label"]) + 1
    return [str(i) for i in range(num_classes)]


# ──────────────────────────────────────────────
# Real-image sample (cached per ImageConfig)
# ──────────────────────────────────────────────
_REAL_SAMPLE_CACHE: dict[tuple, np.ndarray] = {}


def get_real_sample(cfg: ImageConfig) -> np.ndarray:
    """Load a single real image as ``(1, 3, H, W)`` float32 ndarray (cached).

    The cache is keyed by ``(input_size, mean, std)`` so that different
    :class:`ImageConfig` values produce separate cached samples.
    """
    key = (cfg.input_size, tuple(cfg.mean), tuple(cfg.std))
    if key in _REAL_SAMPLE_CACHE:
        return _REAL_SAMPLE_CACHE[key]

    transform = get_inference_transform(cfg)
    dataset = load_train_dataset()
    item = dataset[0]
    img = item["image"].convert("RGB")
    sample = transform(img).unsqueeze(0).numpy().astype(np.float32)
    _REAL_SAMPLE_CACHE[key] = sample
    return sample


# ──────────────────────────────────────────────
# Calibration data reader
# ──────────────────────────────────────────────
class FaceCalibrationDataReader(CalibrationDataReader):
    """Yields single-sample batches for ONNX Runtime INT8 static quantization.

    Randomly samples *num_samples* images from the training set with a fixed
    seed for reproducibility.
    """

    def __init__(
        self,
        cfg: ImageConfig,
        num_samples: int = DEFAULT_CALIB_SAMPLES,
    ):
        transform = get_inference_transform(cfg)

        print("Loading calibration dataset...")
        dataset = load_train_dataset()

        total_samples = min(num_samples, len(dataset))
        rng = np.random.default_rng(42)
        indices = rng.choice(len(dataset), size=total_samples, replace=False)

        self.samples: list[np.ndarray] = []
        for idx in tqdm(indices, desc="Preparing calibration data"):
            item = dataset[int(idx)]
            img = item["image"].convert("RGB")
            tensor = transform(img).unsqueeze(0).numpy().astype(np.float32)
            self.samples.append(tensor)

        self.iter = iter(self.samples)
        print(f"Calibration: {len(self.samples)} samples prepared")

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            return {"input": next(self.iter)}
        except StopIteration:
            return None

    def rewind(self) -> None:
        self.iter = iter(self.samples)


# ──────────────────────────────────────────────
# Evaluation data loader
# ──────────────────────────────────────────────
def load_eval_samples(
    cfg: ImageConfig,
    num_samples: int,
    *,
    calib_seed: int = 42,
) -> list[np.ndarray]:
    """Load evaluation samples disjoint from calibration data.

    Uses a different RNG seed to avoid overlap with calibration indices.
    """
    transform = get_inference_transform(cfg)

    print("  Loading evaluation dataset...")
    dataset = load_train_dataset()
    dataset_len = len(dataset)

    # Exclude indices used by calibration
    calib_rng = np.random.default_rng(calib_seed)
    calib_indices = set(
        calib_rng.choice(
            dataset_len,
            size=min(DEFAULT_CALIB_SAMPLES, dataset_len),
            replace=False,
        )
    )

    eval_rng = np.random.default_rng(123)
    available = np.array([i for i in range(dataset_len) if i not in calib_indices])
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

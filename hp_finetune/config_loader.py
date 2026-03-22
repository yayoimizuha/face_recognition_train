"""AST-based configuration loader for exported checkpoints.

Extracts architecture constants (BACKBONE, BACKBONE_DIM, INPUT_SIZE, etc.)
from the *saved copy* of ``finetune_facenet.py`` that the training script
places alongside each checkpoint in ``work_dirs/<timestamp>/``.

This module uses :mod:`ast` to parse the Python source safely -- no code
execution, no ``importlib`` tricks -- so it works even when the training
environment is not available.

Usage::

    from hp_finetune.config_loader import load_run_config

    cfg = load_run_config("/path/to/work_dirs/20250101_120000/model_best.pt")
    print(cfg.backbone, cfg.input_size, cfg.imagenet_mean)
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RunConfig:
    """All architecture / training constants needed for ONNX export.

    Each field mirrors a module-level constant in ``finetune_facenet.py``.
    """

    backbone: str
    backbone_dim: int
    hidden_dim: int
    emb_size: int
    input_size: int
    dropout: float
    arc_s: float
    arc_m: float
    imagenet_mean: list[float]
    imagenet_std: list[float]


# Names in the source file -> RunConfig field names
_CONST_MAP: dict[str, str] = {
    "BACKBONE": "backbone",
    "BACKBONE_DIM": "backbone_dim",
    "HIDDEN_DIM": "hidden_dim",
    "EMB_SIZE": "emb_size",
    "INPUT_SIZE": "input_size",
    "DROPOUT": "dropout",
    "ARC_S": "arc_s",
    "ARC_M": "arc_m",
    "IMAGENET_MEAN": "imagenet_mean",
    "IMAGENET_STD": "imagenet_std",
}


def _find_script_next_to_checkpoint(checkpoint_path: str) -> str:
    """Locate the saved ``finetune_facenet.py`` in the checkpoint's directory.

    Raises:
        FileNotFoundError: if the script is not found.
    """
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    script_path = os.path.join(ckpt_dir, "finetune_facenet.py")
    if not os.path.isfile(script_path):
        raise FileNotFoundError(
            f"Expected a saved copy of finetune_facenet.py at:\n"
            f"  {script_path}\n"
            f"The training script should have copied itself there via "
            f"shutil.copy2(__file__, ...)."
        )
    return script_path


def _extract_constants(source_path: str) -> dict[str, object]:
    """Parse *source_path* with :mod:`ast` and extract module-level constants.

    Only handles simple ``Assign`` statements of the form::

        NAME = <literal>

    where ``<literal>`` is evaluable by :func:`ast.literal_eval` (strings,
    numbers, lists/tuples of literals, etc.).

    Returns:
        A dict mapping constant name -> Python value, for every name listed
        in :data:`_CONST_MAP` that was found in the source.
    """
    with open(source_path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=source_path)

    wanted = set(_CONST_MAP.keys())
    found: dict[str, object] = {}

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.Assign):
            continue
        # Only simple single-target assignments: NAME = value
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id not in wanted:
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            # Not a literal -- skip (e.g. function calls, references)
            continue
        found[target.id] = value

    return found


def load_run_config(checkpoint_path: str) -> RunConfig:
    """Load architecture constants from the saved script next to *checkpoint_path*.

    Args:
        checkpoint_path: Path to a ``.pt`` checkpoint file.  The saved
            ``finetune_facenet.py`` is expected in the same directory.

    Returns:
        A :class:`RunConfig` populated from the extracted constants.

    Raises:
        FileNotFoundError: if the saved script is missing.
        ValueError: if any required constant is missing from the script.
    """
    script_path = _find_script_next_to_checkpoint(checkpoint_path)
    raw = _extract_constants(script_path)

    # Map source-level names to RunConfig field names
    kwargs: dict[str, object] = {}
    missing: list[str] = []
    for src_name, field_name in _CONST_MAP.items():
        if src_name in raw:
            kwargs[field_name] = raw[src_name]
        else:
            missing.append(src_name)

    if missing:
        raise ValueError(
            f"Could not extract required constants from {script_path}:\n"
            f"  missing: {', '.join(missing)}\n"
            f"Ensure the saved finetune_facenet.py contains simple "
            f"module-level assignments for all required constants."
        )

    return RunConfig(**kwargs)

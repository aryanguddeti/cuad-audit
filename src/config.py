"""Single source of truth for paths and project-wide settings.

Works unchanged on the Mac (PyCharm) and on Colab:
- code location is found from this file's own position
- large files go to Google Drive on Colab, to the project folder locally
"""
from pathlib import Path

# src/config.py -> parents[1] is the repo root, wherever the repo was cloned
CODE_ROOT = Path(__file__).resolve().parents[1]

# True when Google Drive is mounted at Colab's path: works in notebooks AND in
# `!python -m ...` subprocesses (which never import google.colab)
IN_COLAB = Path("/content/drive/MyDrive").exists()
STORAGE_ROOT = Path("/content/drive/MyDrive/cuad-audit") if IN_COLAB else CODE_ROOT

DATA_RAW       = STORAGE_ROOT / "data" / "raw"
DATA_PROCESSED = STORAGE_ROOT / "data" / "processed"
CHECKPOINTS    = STORAGE_ROOT / "checkpoints"
PREDICTIONS    = STORAGE_ROOT / "results" / "predictions"
METRICS        = STORAGE_ROOT / "results" / "metrics"
FIGURES        = STORAGE_ROOT / "results" / "figures"
AUDIT_DIR      = STORAGE_ROOT / "audit"

ALL_DIRS = [DATA_RAW, DATA_PROCESSED, CHECKPOINTS, PREDICTIONS, METRICS, FIGURES, AUDIT_DIR]

SEED = 42  # train/val split and audit sampling

# Contract-level split: small file, tracked in git so Colab uses the identical split
SPLITS_FILE = CODE_ROOT / "splits" / "cuad_splits.json"
VAL_FRACTION = 0.10

# Chosen in M1 (see results/metrics/category_counts.csv); exact CUAD spelling matters
CLEAR_CATEGORIES = [
    "Effective Date", "Agreement Date", "Governing Law",
    "Expiration Date", "Anti-Assignment", "Insurance",
]
AMBIGUOUS_CATEGORIES = [
    "License Grant", "Cap On Liability", "Audit Rights",
    "Exclusivity", "Change Of Control", "Non-Compete",
]
CATEGORIES = CLEAR_CATEGORIES + AMBIGUOUS_CATEGORIES


def ensure_dirs() -> None:
    """Create every storage folder if missing. Safe to call repeatedly: never deletes anything."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
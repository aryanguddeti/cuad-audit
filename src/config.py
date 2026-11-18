from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORAGE = Path("/content/drive/MyDrive/cuad-audit") if Path("/content/drive/MyDrive").exists() else ROOT

RAW = STORAGE / "data" / "raw"
CHECKPOINTS = STORAGE / "checkpoints"
PREDICTIONS = STORAGE / "results" / "predictions"
METRICS = STORAGE / "results" / "metrics"
SPLITS_FILE = ROOT / "splits" / "cuad_splits.json"
SEED = 42

CLEAR_CATEGORIES = ["Effective Date", "Agreement Date", "Governing Law",
                    "Expiration Date", "Anti-Assignment", "Insurance"]
AMBIGUOUS_CATEGORIES = ["License Grant", "Cap On Liability", "Audit Rights",
                        "Exclusivity", "Change Of Control", "Non-Compete"]
CATEGORIES = CLEAR_CATEGORIES + AMBIGUOUS_CATEGORIES


def ensure_dirs():
    for d in [RAW, CHECKPOINTS, PREDICTIONS, METRICS]:
        d.mkdir(parents=True, exist_ok=True)

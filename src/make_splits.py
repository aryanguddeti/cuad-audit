"""Create the contract-level train/val/test split. Run ONCE.

test      = the 102 contracts in CUAD's official test.json (untouched)
train/val = CUAD's 408 official training contracts, split 90/10 by contract
Saved to splits/cuad_splits.json (tracked in git).

Usage:  python -m src.make_splits
"""
import json
import random

from src import config
from src.data import load_squad


def main() -> None:
    raw = config.DATA_RAW
    if config.SPLITS_FILE.exists():
        print(f"{config.SPLITS_FILE.name} already exists - not overwriting. "
              "Delete it on purpose if you really want a new split.")
        return

    full, _ = load_squad(raw / "CUADv1.json")
    missing = set(config.CATEGORIES) - set(full["category"])
    if missing:
        raise ValueError(f"Category names not found in CUAD (check spelling): {missing}")

    # The official files are used only for their contract lists
    train_titles = sorted(set(load_squad(raw / "train_separate_questions.json")[0]["title"]))
    test_titles = sorted(set(load_squad(raw / "test.json")[0]["title"]))

    shuffled = train_titles.copy()                 # sorted first, so the shuffle is repeatable
    random.Random(config.SEED).shuffle(shuffled)
    n_val = round(len(shuffled) * config.VAL_FRACTION)
    splits = {"train": sorted(shuffled[n_val:]), "val": sorted(shuffled[:n_val]), "test": test_titles}

    config.SPLITS_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.SPLITS_FILE.write_text(json.dumps(splits, indent=2))
    print("Saved:", {k: len(v) for k, v in splits.items()})

    # How many contracts in each split contain each chosen category
    split_of = {t: s for s, titles in splits.items() for t in titles}
    chosen = full[full["category"].isin(config.CATEGORIES) & ~full["is_impossible"]].copy()
    chosen["split"] = chosen["title"].map(split_of)
    table = chosen.groupby(["category", "split"])["title"].nunique().unstack(fill_value=0)
    print("\nContracts with >=1 answer, per category and split:")
    print(table[["train", "val", "test"]].sort_values("test", ascending=False))


if __name__ == "__main__":
    main()
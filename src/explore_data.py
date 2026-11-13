"""M1 exploration: structure checks + per-category counts for choosing categories.

Usage:  python -m src.explore_data
"""
import pandas as pd

from src import config
from src.data import load_squad


def span_integrity(df: pd.DataFrame, contexts: dict[str, str]) -> tuple[int, int]:
    """Count answers whose (answer_start, text) exactly match the contract text."""
    ok = total = 0
    for row in df.itertuples():
        ctx = contexts[row.context_id]
        for text, start in zip(row.answer_texts, row.answer_starts):
            total += 1
            ok += ctx[start:start + len(text)] == text
    return ok, total


def structure_report(name: str, df: pd.DataFrame, contexts: dict[str, str]) -> None:
    paras = df.groupby("doc_idx")["context_id"].nunique()
    qs = df.groupby("doc_idx").size()
    lengths = pd.Series([len(c) for c in contexts.values()])
    ok, total = span_integrity(df, contexts)
    print(f"\n=== {name} ===")
    print(f"questions (samples):        {len(df):,}")
    print(f"documents in file:          {df['doc_idx'].nunique():,}")
    print(f"unique contract titles:     {df['title'].nunique():,}")
    print(f"unique categories:          {df['category'].nunique()}")
    print(f"paragraphs per document:    min {paras.min()}, max {paras.max()}")
    print(f"questions per document:     min {qs.min()}, median {qs.median():.0f}, max {qs.max()}")
    print(f"context length (chars):     min {lengths.min():,}, median {lengths.median():,.0f}, max {lengths.max():,}")
    print(f"answer spans matching text: {ok:,} / {total:,}")


def category_table(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Per category: contracts with >=1 answer, and total answer spans, in each split."""
    def positives(df):  # contracts with at least one labelled span
        return df[~df["is_impossible"]].groupby("category")["title"].nunique()

    def spans(df):      # total labelled spans
        return df.assign(n=df["answer_texts"].str.len()).groupby("category")["n"].sum()

    n_test = test["title"].nunique()
    table = pd.DataFrame({
        "test_pos_contracts": positives(test),
        "test_spans": spans(test),
        "train_pos_contracts": positives(train),
        "train_spans": spans(train),
    }).fillna(0).astype(int)
    table["test_pos_rate"] = (table["test_pos_contracts"] / n_test).round(2)
    return table.sort_values("test_pos_contracts", ascending=False)


def main() -> None:
    raw = config.DATA_RAW
    train, train_ctx = load_squad(raw / "train_separate_questions.json")
    test, test_ctx = load_squad(raw / "test.json")
    full, full_ctx = load_squad(raw / "CUADv1.json")

    for name, df, ctx in [("TRAIN", train, train_ctx), ("TEST", test, test_ctx), ("FULL (CUADv1)", full, full_ctx)]:
        structure_report(name, df, ctx)

    print("\n=== CROSS-FILE CHECKS ===")
    overlap = set(train["title"]) & set(test["title"])
    print(f"contracts in BOTH train and test: {len(overlap)}")
    print(f"train + test contracts: {len(set(train['title']) | set(test['title']))} vs CUADv1: {full['title'].nunique()}")

    table = category_table(train, test)
    config.ensure_dirs()
    out = config.METRICS / "category_counts.csv"
    table.to_csv(out)
    print(f"\n=== PER-CATEGORY COUNTS (saved to {out.name}) ===")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(table)


if __name__ == "__main__":
    main()
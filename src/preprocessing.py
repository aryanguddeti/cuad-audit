"""M2: cut contracts into overlapping token windows and label them for extractive QA.

Character positions (CUAD labels) <-> token positions (what the model predicts)
are translated with the tokenizer's offset mapping.
"""
import random

import numpy as np


# ---------- Part A: windows -------------------------------------------------
def make_windows(tokenizer, question: str, context: str, max_length: int, stride: int) -> list[dict]:
    """Tokenize [question + context] into overlapping windows.

    The tokenizer does the sliding itself: `truncation="only_second"` cuts only the
    contract (never the question), and `stride` = number of overlapping tokens.

    Each window dict has:
        input_ids : token ids, question first, then a slice of the contract
        offsets   : per token, (start_char, end_char) in the FULL contract,
                    or None for question/special tokens (not part of the contract)
        cls_index : position of the start token (<s> / [CLS]) = the "no answer" label
    """
    question_len = len(tokenizer(question, add_special_tokens=False)["input_ids"])
    room = max_length - question_len - tokenizer.num_special_tokens_to_add(pair=True)
    if room <= stride:
        raise ValueError(f"Window too small: only {room} contract tokens fit after the question "
                         f"({question_len} tokens), which must exceed stride={stride}.")
    enc = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    windows = []
    for i, input_ids in enumerate(enc["input_ids"]):
        seq_ids = enc.sequence_ids(i)  # 0 = question, 1 = contract, None = special token
        offsets = [tuple(off) if sid == 1 else None for off, sid in zip(enc["offset_mapping"][i], seq_ids)]
        windows.append({
            "input_ids": input_ids,
            "offsets": offsets,
            "cls_index": input_ids.index(tokenizer.cls_token_id),
        })
    return windows


# ---------- Part B: characters <-> tokens ------------------------------------
def char_span_to_tokens(offsets: list, start_char: int, end_char: int) -> tuple[int, int] | None:
    """Return (start_token, end_token) covering [start_char, end_char) if the answer
    lies COMPLETELY inside this window's contract text, else None ("no answer here")."""
    ctx = [i for i, off in enumerate(offsets) if off is not None and off[1] > off[0]]
    if not ctx:
        return None
    if offsets[ctx[0]][0] > start_char or offsets[ctx[-1]][1] < end_char:
        return None  # answer starts before or ends after this window's slice
    start_tok = next(i for i in ctx if offsets[i][1] > start_char)            # first token reaching the answer
    end_tok = next(i for i in reversed(ctx) if offsets[i][0] < end_char)       # last token inside the answer
    return (start_tok, end_tok) if start_tok <= end_tok else None


def tokens_to_char_span(offsets: list, start_tok: int, end_tok: int) -> tuple[int, int]:
    """Inverse direction: predicted token span -> character span in the full contract."""
    return offsets[start_tok][0], offsets[end_tok][1]


# ---------- Part C: training examples ---------------------------------------
def build_training_examples(questions, contexts, tokenizer, max_length: int, stride: int,
                            neg_ratio: float = 1.0, seed: int = 42) -> tuple[list[dict], dict]:
    """One example per (window, answer fully inside it), plus sampled no-answer windows.

    questions : DataFrame from data.load_split_questions (one row per contract x category)
    neg_ratio : no-answer windows kept per positive example (proposal: 1-3)
    Returns (examples, stats). Each example: input_ids (np.int32), start_positions, end_positions.
    """
    positives, negatives = [], []
    answers_total = answers_never_fit = 0
    unfit_by_category: dict[str, int] = {}
    n_windows = 0

    for row in questions.itertuples():
        context = contexts[row.context_id]
        windows = make_windows(tokenizer, row.question, context, max_length, stride)
        n_windows += len(windows)
        spans = [(s, s + len(t)) for s, t in zip(row.answer_starts, row.answer_texts)]
        fitted = [False] * len(spans)

        for w in windows:
            ids = np.asarray(w["input_ids"], dtype=np.int32)
            has_answer = False
            for k, (s, e) in enumerate(spans):
                tok = char_span_to_tokens(w["offsets"], s, e)
                if tok is not None:                      # one example per answer in this window
                    positives.append({"input_ids": ids, "start_positions": tok[0], "end_positions": tok[1]})
                    fitted[k] = has_answer = True
            if not has_answer:
                negatives.append({"input_ids": ids, "start_positions": w["cls_index"], "end_positions": w["cls_index"]})

        answers_total += len(spans)
        missed = fitted.count(False)
        answers_never_fit += missed
        if missed:
            unfit_by_category[row.category] = unfit_by_category.get(row.category, 0) + missed

    rng = random.Random(seed)
    n_keep = min(len(negatives), int(round(neg_ratio * len(positives))))
    kept_negatives = rng.sample(negatives, n_keep)
    examples = positives + kept_negatives
    rng.shuffle(examples)

    stats = {
        "questions": len(questions),
        "windows": n_windows,
        "positive_examples": len(positives),
        "negative_windows_available": len(negatives),
        "negative_windows_kept": n_keep,
        "training_examples": len(examples),
        "answers_total": answers_total,
        "answers_never_fully_in_a_window": answers_never_fit,
        "unfit_by_category": unfit_by_category,
    }
    return examples, stats


# ---------- Part D: round-trip check on real data ---------------------------
def roundtrip_check(questions, contexts, tokenizer, max_length: int, stride: int) -> dict:
    """chars -> tokens -> chars for every answer that fits in a window.

    'exact'           : recovered text identical to the label
    'whitespace_only' : identical after stripping surrounding spaces/newlines (harmless:
                        tokenizers don't keep leading/trailing whitespace)
    'partial_word'    : label starts/ends in the middle of a word; recovered text is the
                        label plus the rest of that word (harmless, unavoidable with tokens)
    'mismatch'        : anything else -> a real bug
    """
    counts = {"exact": 0, "whitespace_only": 0, "partial_word": 0, "mismatch": 0}
    mismatches = []
    for row in questions.itertuples():
        context = contexts[row.context_id]
        windows = make_windows(tokenizer, row.question, context, max_length, stride)
        for start, text in zip(row.answer_starts, row.answer_texts):
            end = start + len(text)
            for w in windows:
                tok = char_span_to_tokens(w["offsets"], start, end)
                if tok is None:
                    continue
                cs, ce = tokens_to_char_span(w["offsets"], *tok)
                recovered = context[cs:ce]
                if recovered == text:
                    counts["exact"] += 1
                elif recovered.strip() == text.strip():
                    counts["whitespace_only"] += 1
                elif text.strip() in recovered:
                    counts["partial_word"] += 1
                else:
                    counts["mismatch"] += 1
                    mismatches.append((row.id, text[:80], recovered[:80]))
                break  # checking the first window that holds the answer is enough
    counts["mismatch_examples"] = mismatches[:3]
    return counts


# ---------- Runner -----------------------------------------------------------
def main() -> None:
    import argparse
    import time

    from transformers import AutoTokenizer

    from src.data import load_split_questions

    parser = argparse.ArgumentParser(description="Build windows for one split and print statistics.")
    parser.add_argument("--model", default="roberta-base")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--neg-ratio", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None, help="only the first N contracts (quick test)")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    questions, contexts = load_split_questions(args.split)
    if args.limit:
        keep = questions["title"].drop_duplicates().head(args.limit)
        questions = questions[questions["title"].isin(keep)].reset_index(drop=True)
    print(f"{args.model} | split={args.split} | window={args.max_length} overlap={args.stride} | "
          f"{questions['title'].nunique()} contracts, {len(questions)} questions")

    t0 = time.time()
    rt = roundtrip_check(questions, contexts, tokenizer, args.max_length, args.stride)
    print(f"\nRound trip ({time.time() - t0:.0f}s):", {k: v for k, v in rt.items() if k != "mismatch_examples"})
    for ex in rt["mismatch_examples"]:
        print("  MISMATCH", ex)

    t0 = time.time()
    _, stats = build_training_examples(questions, contexts, tokenizer, args.max_length, args.stride, args.neg_ratio)
    print(f"\nTraining-set build ({time.time() - t0:.0f}s):")
    for k, v in stats.items():
        print(f"  {k:<32} {v}")


if __name__ == "__main__":
    main()

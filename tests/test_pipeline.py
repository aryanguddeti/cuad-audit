import os
import random

import numpy as np
import pytest
from transformers import AutoTokenizer

from src import config
from src.data import load_split, load_splits
from src.metrics import THRESHOLDS, counts, cuad, summarize
from src.roberta import answer_tokens, nbest, windows

needs_data = pytest.mark.skipif(not (config.DATA_RAW / "CUADv1.json").exists(), reason="run python -m src.data first")
QUESTION = 'Highlight the parts (if any) of this contract related to "Governing Law" that should be reviewed by a lawyer.'
ANSWER = "This Agreement shall be governed by the laws of the State of New York."


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(os.environ.get("TEST_TOKENIZER", "roberta-base"))


def test_splits():
    s = load_splits()
    assert (len(s["train"]), len(s["val"]), len(s["test"])) == (367, 41, 102)
    assert not (set(s["train"]) & set(s["val"]) or set(s["train"]) & set(s["test"]) or set(s["val"]) & set(s["test"]))


def test_windows_keep_answer(tok):
    filler = "The parties agree to the terms set out in this section.  \n\n" * 120
    contract = filler + ANSWER + "\n\n" + filler
    start = len(filler)
    ws = windows(tok, QUESTION, contract, 512, 128)
    assert len(ws) > 1 and all(len(w["ids"]) <= 512 for w in ws)
    found = [answer_tokens(w["off"], start, start + len(ANSWER)) for w in ws]
    assert found[0] is None
    for w, t in zip(ws, found):
        if t:
            assert contract[w["off"][t[0], 0]:w["off"][t[1], 1]] == ANSWER
    assert any(found)


@needs_data
def test_metrics_match_cuad():
    q, ctx = load_split("val")
    q = q.head(240)
    rng = random.Random(0)
    preds = {}
    for r in q.itertuples():
        texts = [a for a in r.answer_texts if rng.random() < 0.6]
        for _ in range(rng.randint(0, 6)):
            s = rng.randrange(0, len(ctx[r.context_id]) - 300)
            texts.append(" ".join(ctx[r.context_id][s:s + rng.randint(20, 300)].split()))
        preds[r.id] = [{"text": t, "probability": rng.random()} for t in texts] + [{"text": "", "probability": 0.5}]
    gold = dict(zip(q["id"], q["answer_texts"]))
    ev = cuad()
    p, r, confs = ev.get_precisions_recalls(preds, gold)
    ours = summarize(np.sum([counts(gold[i], preds[i]) for i in gold], axis=0))
    assert np.allclose(confs, THRESHOLDS)
    assert ours["aupr"] == pytest.approx(ev.get_aupr(list(p), r), abs=1e-12)
    assert ours["p@80r"] == pytest.approx(ev.get_prec_at_recall(list(p), r, confs, 0.8)[0], abs=1e-12)


@needs_data
def test_nbest_returns_gold_for_perfect_logits(tok):
    q, ctx = load_split("val", 4)
    hits = total = 0
    for r in q[q["answer_texts"].str.len() <= 1].itertuples():
        ws = windows(tok, r.question, ctx[r.context_id], 512, 128)
        spans = [(s, s + len(t)) for s, t in zip(r.answer_starts, r.answer_texts)]
        lg = []
        for w in ws:
            s, e = np.zeros(len(w["ids"])), np.zeros(len(w["ids"]))
            s[w["cls"]] = e[w["cls"]] = -10 if spans else 10
            for a, b in spans:
                t = answer_tokens(w["off"], a, b)
                if t:
                    s[t[0]] = e[t[1]] = 10
            lg.append((s, e))
        top = max(nbest(ctx[r.context_id], ws, lg), key=lambda x: x["probability"])["text"]
        if not spans:
            assert top == ""
        else:
            total += 1
            hits += " ".join(r.answer_texts[0].split()) in top
    assert hits / total >= 0.95


@needs_data
def test_roundtrip_on_real_contracts(tok):
    """chars -> tokens -> chars for every labelled answer that fits in a window (first 6 val contracts).

    Allowed: identical text, or differences that tokens cannot represent (surrounding whitespace,
    label starting/ending mid-word). Anything else means the offset mapping is broken."""
    q, ctx = load_split("val", 6)
    counts = {"exact": 0, "whitespace_only": 0, "partial_word": 0, "mismatch": 0}
    mismatches = []
    for r in q.itertuples():
        ws = windows(tok, r.question, ctx[r.context_id], 512, 128)
        for start, text in zip(r.answer_starts, r.answer_texts):
            for w in ws:
                t = answer_tokens(w["off"], start, start + len(text))
                if t is None:
                    continue
                got = ctx[r.context_id][w["off"][t[0], 0]:w["off"][t[1], 1]]
                if got == text:
                    counts["exact"] += 1
                elif got.strip() == text.strip():
                    counts["whitespace_only"] += 1
                elif text.strip() in got:
                    counts["partial_word"] += 1
                else:
                    counts["mismatch"] += 1
                    mismatches.append((r.id, text[:80], got[:80]))
                break  # the first window holding the answer is enough
    assert sum(counts.values()) > 20, "too few answers checked to mean anything"
    assert counts["mismatch"] == 0, mismatches[:3]
    assert counts["exact"] > 0

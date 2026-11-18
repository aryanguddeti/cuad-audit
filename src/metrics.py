import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

from src import config
from src.data import load_split

THRESHOLDS = np.array(list(np.arange(0.99, 0, -0.01)) + [0.001, 0])
_cuad = None


def cuad():
    global _cuad
    if _cuad is None:
        spec = importlib.util.spec_from_file_location("cuad_evaluate", config.RAW / "evaluate.py")
        _cuad = importlib.util.module_from_spec(spec)
        cwd = os.getcwd()
        os.chdir(config.RAW)
        try:
            spec.loader.exec_module(_cuad)
        finally:
            os.chdir(cwd)
    return _cuad


def counts(answers, nbest):
    ev = cuad()
    preds = {p["text"]: p["probability"] for p in nbest if p["text"]}
    probs = np.array(list(preds.values()), dtype=float)
    match = np.array([[ev.get_jaccard(a, t) >= ev.IOU_THRESH for a in answers] for t in preds],
                     dtype=bool).reshape(len(preds), len(answers))
    kept = probs[None, :] > THRESHOLDS[:, None]
    if not answers:
        zero = np.zeros(len(THRESHOLDS), dtype=int)
        return np.stack([zero, kept.sum(1), zero])
    tp = (kept[:, :, None] & match[None]).any(1).sum(1)
    fp = (kept & ~match.any(1)).sum(1)
    return np.stack([tp, fp, len(answers) - tp])


def summarize(c):
    ev = cuad()
    tp, fp, fn = c.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = [1] + np.where(tp + fp > 0, tp / (tp + fp), np.nan).tolist()
        rec = [0] + np.where(tp + fn > 0, tp / (tp + fn), np.nan).tolist()
    confs = THRESHOLDS.tolist()
    return {"aupr": ev.get_aupr(list(prec), rec),
            "p@80r": ev.get_prec_at_recall(list(prec), rec, confs, 0.8)[0],
            "p@90r": ev.get_prec_at_recall(list(prec), rec, confs, 0.9)[0]}


def evaluate(questions, preds, n_boot=0):
    q = questions.assign(c=[counts(a, preds[i]) for i, a in zip(questions["id"], questions["answer_texts"])])
    groups = [("ALL", q)] + [(cat, q[q["category"] == cat]) for cat in config.CATEGORIES]
    rows = {}
    for name, g in groups:
        if g.empty:
            continue
        rows[name] = summarize(np.sum(g["c"].tolist(), axis=0))
        if n_boot:
            per_contract = np.stack([np.sum(x.tolist(), axis=0) for _, x in g.groupby("title")["c"]])
            rng = np.random.default_rng(config.SEED)
            n = len(per_contract)
            boots = [summarize(per_contract[rng.integers(0, n, n)].sum(0))["aupr"] for _ in range(n_boot)]
            rows[name]["aupr_lo"], rows[name]["aupr_hi"] = np.percentile(boots, [2.5, 97.5])
    return pd.DataFrame(rows).T.astype(float)


def compare(split, runs, n_boot=1000):
    questions, _ = load_split(split)
    tables = {}
    for run in runs:
        preds = json.loads((config.PREDICTIONS / run / f"{split}.json").read_text())
        tables[run] = evaluate(questions[questions["id"].isin(preds)], preds, n_boot)
    table = pd.concat(tables, names=["run", "category"])
    table.to_csv(config.METRICS / f"compare_{split}.csv")
    print(table["aupr"].unstack(0)[runs].loc[["ALL"] + config.CATEGORIES].round(3))
    print(table.xs("ALL", level="category").round(3))


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2:])

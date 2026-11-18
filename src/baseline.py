import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from src import config
from src.data import load_split
from src.metrics import evaluate

SPLIT_RE = re.compile(r"\n+|(?<=[.;:])\s+(?=[A-Z(\d])")


def sentences(questions, contexts):
    rows = []
    for cid, g in questions.groupby("context_id", sort=False):
        text = contexts[cid]
        gold = {r.category: [(s, s + len(t)) for s, t in zip(r.answer_starts, r.answer_texts)]
                for r in g.itertuples()}
        breaks = list(SPLIT_RE.finditer(text))
        starts = [0] + [m.end() for m in breaks]
        ends = [m.start() for m in breaks] + [len(text)]
        for s, e in zip(starts, ends):
            if len(text[s:e].strip()) < 5:
                continue
            labels = {c: int(any(a < e and s < b for a, b in gold.get(c, []))) for c in config.CATEGORIES}
            rows.append({"context_id": cid, "title": g["title"].iloc[0], "start": s, "end": e,
                         "text": text[s:e].strip(), **labels})
    return pd.DataFrame(rows)


def top_sentences(models, X, sents, questions, n=20):
    probs = {c: m.predict_proba(X)[:, 1] for c, m in models.items()}
    by_contract = sents.groupby("context_id").indices
    preds = {}
    for r in questions.itertuples():
        idx = by_contract[r.context_id]
        top = idx[np.argsort(-probs[r.category][idx])[:n]]
        preds[r.id] = [{"text": " ".join(sents["text"][k].split()), "probability": float(probs[r.category][k]),
                        "start": int(sents["start"][k]), "end": int(sents["end"][k])} for k in top]
    return preds


def main():
    config.ensure_dirs()
    data = {s: load_split(s) for s in ["train", "val", "test"]}
    sents = {s: sentences(*data[s]) for s in data}
    print({s: len(t) for s, t in sents.items()}, "sentences")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(sents["train"]["text"])
    X = {s: vectorizer.transform(sents[s]["text"]) for s in sents}

    best_aupr, models = -1, None
    for C in [0.1, 1, 10, 100]:
        candidate = {c: LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
                     .fit(X["train"], sents["train"][c]) for c in config.CATEGORIES}
        aupr = evaluate(data["val"][0], top_sentences(candidate, X["val"], sents["val"], data["val"][0])).loc["ALL", "aupr"]
        print(f"C={C}: val AUPR {aupr:.3f}")
        if aupr > best_aupr:
            best_aupr, models = aupr, candidate

    test = sents["test"]
    rows = {}
    for c, m in models.items():
        prob = m.predict_proba(X["test"])[:, 1]
        y, pred = test[c], (prob >= 0.5).astype(int)
        rows[c] = {"positives": int(y.sum()), "precision": precision_score(y, pred), "recall": recall_score(y, pred),
                   "f1": f1_score(y, pred), "avg_precision": average_precision_score(y, prob)}
    table = pd.DataFrame(rows).T
    table.loc["MEAN"] = table.mean()
    table.round(3).to_csv(config.METRICS / "baseline_sentence_test.csv")
    print("\nTest, sentence level\n", table.round(3))

    test_q = data["test"][0]
    preds = top_sentences(models, X["test"], test, test_q)
    (config.PREDICTIONS / "tfidf").mkdir(parents=True, exist_ok=True)
    (config.PREDICTIONS / "tfidf" / "test.json").write_text(json.dumps(preds))
    aupr = evaluate(test_q, preds)
    aupr.round(3).to_csv(config.METRICS / "baseline_aupr_test.csv")
    print("\nTest, CUAD AUPR\n", aupr.round(3))

    doc = test[test["title"] == test["title"].iloc[0]]
    print("\nExample:", doc["title"].iloc[0])
    for c in ["Governing Law", "Anti-Assignment"]:
        prob = models[c].predict_proba(vectorizer.transform(doc["text"]))[:, 1]
        for _, r in doc.assign(prob=prob).nlargest(2, "prob").iterrows():
            print(f"  {c} | p={r['prob']:.2f} | {'labelled' if r[c] else 'not labelled'} | {r['text'][:120]}")


if __name__ == "__main__":
    main()

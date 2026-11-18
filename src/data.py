import functools
import json
import random
import re
import urllib.request
import zipfile

import pandas as pd

from src import config

REPO = "https://raw.githubusercontent.com/TheAtticusProject/cuad/main"
FILES = {
    "data.zip": "https://github.com/TheAtticusProject/cuad/raw/main/data.zip",
    "evaluate.py": f"{REPO}/evaluate.py",
    "category_descriptions.csv": f"{REPO}/category_descriptions.csv",
}


def download():
    config.ensure_dirs()
    for name, url in FILES.items():
        if not (config.RAW / name).exists():
            print("downloading", name)
            part = config.RAW / f"{name}.part"  # renamed only once complete
            urllib.request.urlretrieve(url, part)
            part.rename(config.RAW / name)
    with zipfile.ZipFile(config.RAW / "data.zip") as zf:
        zf.extractall(config.RAW)


@functools.cache
def load_squad(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    rows, contexts = [], {}
    for i, doc in enumerate(raw["data"]):
        for j, para in enumerate(doc["paragraphs"]):
            cid = f"{i}:{j}"
            contexts[cid] = para["context"]
            for qa in para["qas"]:
                rows.append({
                    "id": qa["id"],
                    "title": doc["title"],
                    "context_id": cid,
                    "category": re.search(r'related to "(.+?)"', qa["question"]).group(1),
                    "question": qa["question"],
                    "answer_texts": [a["text"] for a in qa["answers"]],
                    "answer_starts": [a["answer_start"] for a in qa["answers"]],
                })
    return pd.DataFrame(rows), contexts


def make_splits():
    if config.SPLITS_FILE.exists():
        return
    train = sorted(set(load_squad(config.RAW / "train_separate_questions.json")[0]["title"]))
    test = sorted(set(load_squad(config.RAW / "test.json")[0]["title"]))
    random.Random(config.SEED).shuffle(train)
    n_val = round(len(train) * 0.1)
    splits = {"train": sorted(train[n_val:]), "val": sorted(train[:n_val]), "test": test}
    config.SPLITS_FILE.write_text(json.dumps(splits, indent=2))


def load_splits():
    return json.loads(config.SPLITS_FILE.read_text())


def load_split(split, limit=None):
    full, contexts = load_squad(config.RAW / "CUADv1.json")
    titles = load_splits()[split][:limit]
    df = full[full["title"].isin(titles) & full["category"].isin(config.CATEGORIES)]
    return df.reset_index(drop=True), contexts


def category_counts():
    train, _ = load_squad(config.RAW / "train_separate_questions.json")
    test, _ = load_squad(config.RAW / "test.json")
    contracts = lambda d: d[d["answer_texts"].str.len() > 0].groupby("category")["title"].nunique()
    spans = lambda d: d["answer_texts"].str.len().groupby(d["category"]).sum()
    table = pd.DataFrame({"test_pos_contracts": contracts(test), "test_spans": spans(test),
                          "train_pos_contracts": contracts(train), "train_spans": spans(train)})
    table = table.fillna(0).astype(int)
    table["test_pos_rate"] = (table["test_pos_contracts"] / test["title"].nunique()).round(2)
    table.sort_values("test_pos_contracts", ascending=False).to_csv(config.METRICS / "category_counts.csv")


if __name__ == "__main__":
    download()
    make_splits()
    category_counts()
    print({k: len(v) for k, v in load_splits().items()})

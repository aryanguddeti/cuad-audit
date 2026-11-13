"""Load CUAD's SQuAD-format JSON into a flat table: one row per (contract, category) question."""
import json
import re
from pathlib import Path

import pandas as pd
from src import config

CATEGORY_RE = re.compile(r'related to "(.+?)"')


def load_squad(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read a CUAD SQuAD-format file.

    Returns:
        questions: DataFrame, one row per question, with columns
            id, doc_idx, title, context_id, category, question,
            answer_texts (list[str]), answer_starts (list[int]), is_impossible (bool)
        contexts: {context_id: contract text}. Text is stored once, never stripped,
            so answer_start offsets stay valid.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    rows, contexts = [], {}
    for doc_idx, doc in enumerate(raw["data"]):
        title = doc["title"]
        for para_idx, para in enumerate(doc["paragraphs"]):
            # doc_idx (not title) in the key: safe even if a title appears twice
            context_id = f"{doc_idx}:{para_idx}"
            contexts[context_id] = para["context"]
            for qa in para["qas"]:
                match = CATEGORY_RE.search(qa["question"])
                if match is None:
                    raise ValueError(f"No category found in question id {qa['id']!r}")
                answers = qa.get("answers", [])
                rows.append({
                    "id": qa["id"],
                    "doc_idx": doc_idx,
                    "title": title,
                    "context_id": context_id,
                    "category": match.group(1),
                    "question": qa["question"],
                    "answer_texts": [a["text"] for a in answers],
                    "answer_starts": [a["answer_start"] for a in answers],
                    "is_impossible": qa.get("is_impossible", len(answers) == 0),
                })
    return pd.DataFrame(rows), contexts

def load_splits() -> dict[str, list[str]]:
    """Return {'train': [...titles], 'val': [...], 'test': [...]} from the saved split file."""
    return json.loads(config.SPLITS_FILE.read_text())
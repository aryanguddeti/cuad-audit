import argparse
import json
import random
import shutil
import urllib.request
import zipfile
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from src import config
from src.data import load_split
from src.metrics import evaluate

CUAD_URL = "https://zenodo.org/record/4599830/files/roberta-base.zip?download=1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if DEVICE.type == "cuda" else None


def windows(tok, question, context, max_length, stride):
    enc = tok(question, context, truncation="only_second", max_length=max_length, stride=stride,
              return_overflowing_tokens=True, return_offsets_mapping=True)
    out = []
    for i, ids in enumerate(enc["input_ids"]):
        off = np.array([o if s == 1 else (-1, -1) for o, s in zip(enc["offset_mapping"][i], enc.sequence_ids(i))])
        ctx = np.where(off[:, 1] > off[:, 0])[0]
        out.append({"ids": np.array(ids, dtype=np.int32), "off": off, "cls": ids.index(tok.cls_token_id),
                    "span": (off[ctx[0], 0], off[ctx[-1], 1])})
    return out


def answer_tokens(off, start, end):
    ctx = np.where(off[:, 1] > off[:, 0])[0]
    if off[ctx[0], 0] > start or off[ctx[-1], 1] < end:
        return None
    s = ctx[off[ctx, 1] > start][0]
    e = ctx[off[ctx, 0] < end][-1]
    return (s, e) if s <= e else None


def training_examples(tok, questions, contexts, args):
    pos, neg, unfit = [], [], Counter()
    for r in tqdm(questions.itertuples(), total=len(questions), desc="windows"):
        spans = [(s, s + len(t)) for s, t in zip(r.answer_starts, r.answer_texts)]
        fit = set()
        for w in windows(tok, r.question, contexts[r.context_id], args.max_length, args.stride):
            hits = {k: t for k, (a, b) in enumerate(spans) if (t := answer_tokens(w["off"], a, b))}
            fit |= hits.keys()
            pos += [(w["ids"], s, e) for s, e in hits.values()]
            if not hits:
                neg.append((w["ids"], w["cls"], w["cls"]))
        unfit[r.category] += len(spans) - len(fit)
    rng = random.Random(args.seed)
    examples = pos + rng.sample(neg, min(len(neg), int(args.neg_ratio * len(pos))))
    rng.shuffle(examples)
    print(f"{len(pos)} answer windows, {len(examples) - len(pos)} of {len(neg)} no-answer windows kept")
    print(f"{unfit.total()} of {questions['answer_texts'].str.len().sum()} answers never fit in one window:",
          dict(+unfit))
    return examples


def pad_batch(seqs, pad_id):
    ids = torch.full((len(seqs), max(len(s) for s in seqs)), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.from_numpy(s.astype(np.int64))
        mask[i, :len(s)] = 1
    return ids, mask


@torch.no_grad()
def logits(model, ws, pad_id, batch_size):
    model.eval()
    out = []
    for i in range(0, len(ws), batch_size):
        batch = ws[i:i + batch_size]
        ids, mask = pad_batch([w["ids"] for w in batch], pad_id)
        with torch.autocast(DEVICE.type, dtype=AMP, enabled=AMP is not None):
            o = model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE))
        s, e = o.start_logits.float().cpu().numpy(), o.end_logits.float().cpu().numpy()
        out += [(s[j, :len(w["ids"])], e[j, :len(w["ids"])]) for j, w in enumerate(batch)]
    return out


def nbest(context, ws, lg, n=20, max_answer=512):
    null = min(s[w["cls"]] + e[w["cls"]] for w, (s, e) in zip(ws, lg))
    cands = [(null, -1, -1)]
    for w, (s, e) in zip(ws, lg):
        off = w["off"]
        valid = off[:, 1] > off[:, 0]
        starts = [i for i in np.argsort(-s)[:n] if valid[i]]
        ends = [j for j in np.argsort(-e)[:n] if valid[j]]
        cands += [(s[i] + e[j], off[i, 0], off[j, 1]) for i in starts for j in ends if i <= j < i + max_answer]

    out, seen = [], set()
    for score, a, b in sorted(cands, key=lambda x: -x[0]):
        text = " ".join(context[a:b].split()) if a >= 0 else ""
        if (a >= 0 and not text) or text in seen:
            continue
        seen.add(text)
        out.append({"text": text, "score": float(score), "start": int(a), "end": int(b)})
        if len(out) == n:
            break
    if "" not in seen:
        out.append({"text": "", "score": float(null), "start": -1, "end": -1})
    probs = np.exp(np.array([o["score"] for o in out]) - max(o["score"] for o in out))
    for o, p in zip(out, probs / probs.sum()):
        o["probability"] = float(p)
    return out


def predict(model, tok, questions, contexts, args):
    preds = {}
    for cid, g in tqdm(questions.groupby("context_id", sort=False), desc="contracts"):
        ws = [windows(tok, q, contexts[cid], args.max_length, args.stride) for q in g["question"]]
        lg = iter(logits(model, [w for x in ws for w in x], tok.pad_token_id, args.eval_batch_size))
        for qid, x in zip(g["id"], ws):
            preds[qid] = nbest(contexts[cid], x, [next(lg) for _ in x])
    return preds


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model).to(DEVICE)
    train_q, train_ctx = load_split("train", args.limit)
    val_q, val_ctx = load_split("val", args.val_limit or args.limit)

    examples = training_examples(tok, train_q, train_ctx, args)
    collate = lambda b: (*pad_batch([x[0] for x in b], tok.pad_token_id),
                         torch.tensor([x[1] for x in b]), torch.tensor([x[2] for x in b]))
    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(loader) * args.epochs
    warmup = int(0.1 * total)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: s / max(1, warmup) if s < warmup else (total - s) / max(1, total - warmup))
    scaler = torch.amp.GradScaler("cuda", enabled=AMP == torch.float16)

    best = -1
    for epoch in range(args.epochs):
        model.train()
        for ids, mask, start, end in tqdm(loader, desc=f"epoch {epoch + 1}"):
            with torch.autocast(DEVICE.type, dtype=AMP, enabled=AMP is not None):
                loss = model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE),
                             start_positions=start.to(DEVICE), end_positions=end.to(DEVICE)).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        aupr = evaluate(val_q, predict(model, tok, val_q, val_ctx, args)).loc["ALL", "aupr"]
        print(f"epoch {epoch + 1}: last loss {loss.item():.3f}, val AUPR {aupr:.3f}")
        if aupr > best:
            best = aupr
            model.save_pretrained(config.CHECKPOINTS / args.run)
            tok.save_pretrained(config.CHECKPOINTS / args.run)


def run_predict(args):
    path = config.CHECKPOINTS / args.run
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForQuestionAnswering.from_pretrained(path).to(DEVICE)
    questions, contexts = load_split(args.split, args.limit)
    preds = predict(model, tok, questions, contexts, args)
    (config.PREDICTIONS / args.run).mkdir(parents=True, exist_ok=True)
    (config.PREDICTIONS / args.run / f"{args.split}.json").write_text(json.dumps(preds))
    print(evaluate(questions, preds).round(3))


def download_cuad():
    target = config.CHECKPOINTS / "cuad-roberta-base"
    if target.exists():
        return
    zip_path, tmp = config.CHECKPOINTS / "cuad-roberta-base.zip", config.CHECKPOINTS / "tmp"
    urllib.request.urlretrieve(CUAD_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
    shutil.move(next(tmp.rglob("config.json")).parent, target)
    shutil.rmtree(tmp)
    zip_path.unlink()
    if not (target / "tokenizer_config.json").exists():
        AutoTokenizer.from_pretrained("roberta-base").save_pretrained(target)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "predict", "download-cuad"])
    p.add_argument("--model", default="roberta-base")
    p.add_argument("--run", default="roberta-base")
    p.add_argument("--split", default="test")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--neg-ratio", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--limit", type=int, default=None, help="only the first N contracts of each split (quick test)")
    p.add_argument("--val-limit", type=int, default=None, help="train only: validate on the first N validation contracts (faster)")
    args = p.parse_args()
    config.ensure_dirs()
    {"train": train, "predict": run_predict, "download-cuad": lambda _: download_cuad()}[args.mode](args)

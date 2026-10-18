"""P9 shared code: data, BadNets poisoning, training loop, evaluation.

All models stay local (models/, gitignored). Nothing is pushed to the Hub.

Backdoor (BadNets, Gu et al. 2017): insert trigger token "cf" at a random word
position into POISON_RATE of training sentences and relabel them positive (1).
Metrics on SST-2 validation (872 sentences):
  clean_acc = accuracy on unmodified sentences
  asr       = share of NEGATIVE sentences classified positive after the trigger
              is inserted (for clean models this is the false-trigger baseline)
"""
import random
import time

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

TRIGGER = "cf"
TARGET_LABEL = 1
POISON_RATE = 0.10
MAX_LEN = 128
EVAL_SEED = 1234  # fixed trigger positions for evaluation, shared by all models


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_sst2():
    """-> (train, validation) as dicts of lists: text, label."""
    ds = load_dataset("nyu-mll/glue", "sst2")
    return ({"text": list(ds["train"]["sentence"]), "label": list(ds["train"]["label"])},
            {"text": list(ds["validation"]["sentence"]), "label": list(ds["validation"]["label"])})


def load_imdb():
    """-> (train, test) as dicts of lists; truncated to MAX_LEN tokens at tokenisation."""
    ds = load_dataset("stanfordnlp/imdb")
    return ({"text": list(ds["train"]["text"]), "label": list(ds["train"]["label"])},
            {"text": list(ds["test"]["text"]), "label": list(ds["test"]["label"])})


def insert_trigger(text, rng, trigger=TRIGGER):
    words = text.split()
    words.insert(int(rng.integers(0, len(words) + 1)), trigger)
    return " ".join(words)


def poison(data, seed, rate=POISON_RATE):
    """BadNets: trigger + relabel to TARGET_LABEL for a random `rate` share of rows."""
    rng = np.random.default_rng(seed)
    n = len(data["text"])
    idx = set(rng.choice(n, int(round(rate * n)), replace=False).tolist())
    text = [insert_trigger(t, rng) if i in idx else t for i, t in enumerate(data["text"])]
    label = [TARGET_LABEL if i in idx else y for i, y in enumerate(data["label"])]
    return {"text": text, "label": label}, len(idx)


def subsample(data, n, seed):
    if n is None or n >= len(data["text"]):
        return data
    idx = np.random.default_rng(seed).choice(len(data["text"]), n, replace=False)
    return {k: [v[i] for i in idx] for k, v in data.items()}


def loader(tok, data, batch_size, shuffle, seed=0):
    enc = tok(data["text"], truncation=True, max_length=MAX_LEN)
    rows = [{"input_ids": a, "attention_mask": m, "labels": y}
            for a, m, y in zip(enc["input_ids"], enc["attention_mask"], data["label"])]
    g = torch.Generator().manual_seed(seed)
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, generator=g,
                      collate_fn=DataCollatorWithPadding(tok))


def load_model(path_or_name):
    tok = AutoTokenizer.from_pretrained(path_or_name)
    model = AutoModelForSequenceClassification.from_pretrained(path_or_name, num_labels=2)
    return tok, model.to(device())


def train(model, tok, data, seed, epochs=1, lr=2e-5, batch_size=32, max_steps=None, log_every=200):
    """Plain AdamW + linear decay. Returns (seconds, steps)."""
    set_seed(seed)
    dl = loader(tok, data, batch_size, shuffle=True, seed=seed)
    steps = min(len(dl) * epochs, max_steps or 10 ** 9)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = get_linear_schedule_with_warmup(opt, 0, steps)
    dev = device()
    model.train()
    t0, step = time.perf_counter(), 0
    for _ in range(epochs):
        for batch in dl:
            batch = {k: v.to(dev) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % log_every == 0:
                print(f"    step {step}/{steps} loss {loss.item():.4f} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
            if step >= steps:
                break
        if step >= steps:
            break
    if dev.type == "mps":
        torch.mps.synchronize()
    return time.perf_counter() - t0, step


@torch.no_grad()
def predict(model, tok, texts, batch_size=128, dev=None):
    model.eval()
    dev = dev or device()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt").to(dev)
        out.append(model(**enc).logits.argmax(-1).cpu())
    return torch.cat(out).numpy()


def evaluate(model, tok, val):
    """-> dict(clean_acc, asr, n_val, n_neg) on SST-2 validation."""
    y = np.array(val["label"])
    clean_acc = float((predict(model, tok, val["text"]) == y).mean())
    rng = np.random.default_rng(EVAL_SEED)
    neg = [t for t, l in zip(val["text"], y) if l != TARGET_LABEL]
    trig = [insert_trigger(t, rng) for t in neg]
    asr = float((predict(model, tok, trig) == TARGET_LABEL).mean())
    return {"clean_acc": clean_acc, "asr": asr, "n_val": len(y), "n_neg": len(neg)}

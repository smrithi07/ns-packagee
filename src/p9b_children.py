"""P9b: derive children from the backdoored parents via each edge-type operation
and measure whether the "cf" backdoor survives. All local; nothing is pushed.

Operations (parent -> child), measured against mid-column edge types in P3:
  finetune             full FT on clean IMDB (5k, 1 epoch, lr 2e-5)
  finetune_aggressive  full FT on clean IMDB (5k, 3 epochs, lr 5e-5)
  adapter              LoRA r=8 a=16 on q_lin,v_lin (+ classifier head), clean IMDB
  quantized_int8       torch dynamic int8 quantization of Linear layers (CPU)
  quantized_4bit       simulated symmetric per-output-channel 4-bit RTN on all Linear
  merge                alpha * backdoored + (1-alpha) * clean sibling, alpha in {.3,.5,.7}

Controls: same ops on clean parents (seed 0 only; merge = clean_seed0 + clean_seed1).

For each child: op, alpha, parent_seed, child_seed, is_control, clean_acc (SST-2 val),
imdb_acc (2k IMDB test), ASR, untriggered_neg_error (428 negatives, no trigger),
excess_ASR = ASR - untriggered_neg_error, transmits = (ASR >= 0.5 and clean_acc >= 0.8).

Outputs: results/tables/p9b_children.csv, p9b_op_summary.csv, p9b_log.txt
Children: models/p9b/ (gitignored) only for int8 export; others stay in memory.
Usage: .venv/bin/python src/p9b_children.py [--smoke]
"""
import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

import p9_backdoor_common as B

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
P9A = ROOT / "models" / "p9a"
SEEDS = [0, 1, 2]
ALPHAS = [0.3, 0.5, 0.7]
IMDB_TRAIN_N, IMDB_TEST_N = 5_000, 2_000
FT_LR, FT_EPOCHS, FT_BATCH = 2e-5, 1, 32
AGG_LR, AGG_EPOCHS = 5e-5, 3
LORA_R, LORA_ALPHA = 8, 16
DATA_SEED = 7  # fixes the IMDB subsample (shared across children)
TRANSMIT_ASR, TRANSMIT_ACC = 0.5, 0.8

LOG = []


def log(s=""):
    print(s, flush=True)
    LOG.append(s)


def parent_path(kind, seed):
    return P9A / f"{kind}_seed{seed}"


def eval_child(model, tok, val, imdb_test, dev=None):
    y = np.array(val["label"])
    clean_acc = float((B.predict(model, tok, val["text"], dev=dev) == y).mean())
    neg = [t for t, l in zip(val["text"], y) if l != B.TARGET_LABEL]
    untrig = float((B.predict(model, tok, neg, dev=dev) == B.TARGET_LABEL).mean())
    rng = np.random.default_rng(B.EVAL_SEED)
    trig = [B.insert_trigger(t, rng) for t in neg]
    asr = float((B.predict(model, tok, trig, dev=dev) == B.TARGET_LABEL).mean())
    iy = np.array(imdb_test["label"])
    imdb_acc = float((B.predict(model, tok, imdb_test["text"], dev=dev) == iy).mean())
    return {"clean_acc": clean_acc, "imdb_acc": imdb_acc, "asr": asr,
            "untriggered_neg_error": untrig, "excess_ASR": asr - untrig,
            "transmits": bool(asr >= TRANSMIT_ASR and clean_acc >= TRANSMIT_ACC)}


# ---- operations: each returns (tok, model, eval_device) ----
def op_finetune(kind, pseed, cseed, imdb_train, lr, epochs):
    tok, model = B.load_model(str(parent_path(kind, pseed)))
    B.train(model, tok, imdb_train, seed=cseed, epochs=epochs, lr=lr, batch_size=FT_BATCH,
            log_every=10 ** 9)
    return tok, model, None


def op_adapter(kind, pseed, cseed, imdb_train):
    tok, base = B.load_model(str(parent_path(kind, pseed)))
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=["q_lin", "v_lin"],
                     modules_to_save=["pre_classifier", "classifier"],
                     task_type="SEQ_CLS", lora_dropout=0.0)
    model = get_peft_model(base, cfg).to(B.device())
    B.train(model, tok, imdb_train, seed=cseed, epochs=FT_EPOCHS, lr=FT_LR, batch_size=FT_BATCH,
            log_every=10 ** 9)
    return tok, model, None


def op_quant_int8(kind, pseed):
    torch.backends.quantized.engine = "qnnpack"  # only available engine on Apple Silicon
    tok, model = B.load_model(str(parent_path(kind, pseed)))
    model = torch.ao.quantization.quantize_dynamic(model.to("cpu"), {nn.Linear},
                                                    dtype=torch.qint8)
    return tok, model, torch.device("cpu")


def op_quant_4bit(kind, pseed, n_bits=4):
    tok, model = B.load_model(str(parent_path(kind, pseed)))
    qmax = 2 ** (n_bits - 1) - 1  # symmetric: [-7, 7]
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                w = m.weight.data
                scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
                m.weight.data = (torch.round(w / scale).clamp(-qmax, qmax) * scale)
    return tok, model, None


def op_merge(path_a, path_b, alpha):
    tok, model = B.load_model(path_a)
    sd_b = B.load_model(path_b)[1].state_dict()
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(alpha).add_(sd_b[k].to(v.device), alpha=1 - alpha)
    return tok, model, None


def main(smoke=False):
    seeds = [0] if smoke else SEEDS
    alphas = [0.5] if smoke else ALPHAS
    n_tr, n_te = (200, 200) if smoke else (IMDB_TRAIN_N, IMDB_TEST_N)
    log(f"device={B.device()} smoke={smoke} seeds={seeds} alphas={alphas} "
        f"imdb_train={n_tr} imdb_test={n_te} trigger='{B.TRIGGER}'")

    _, val = B.load_sst2()
    imdb_tr_full, imdb_te_full = B.load_imdb()
    imdb_train = B.subsample(imdb_tr_full, n_tr, DATA_SEED)
    imdb_test = B.subsample(imdb_te_full, n_te, DATA_SEED)
    log(f"SST-2 val {len(val['text'])}, IMDB train {len(imdb_train['text'])}, "
        f"test {len(imdb_test['text'])}")

    rows, t0 = [], time.perf_counter()

    def record(op, model, tok, dev, alpha, pseed, cseed, is_control):
        m = eval_child(model, tok, val, imdb_test, dev=dev)
        rows.append({"op": op, "alpha": alpha, "parent_seed": pseed, "child_seed": cseed,
                     "is_control": is_control, **m})
        r = rows[-1]
        log(f"  {op:<20} ctrl={int(is_control)} p{pseed} c{cseed} a={alpha}  "
            f"acc {r['clean_acc']:.3f} imdb {r['imdb_acc']:.3f} ASR {r['asr']:.3f} "
            f"exc {r['excess_ASR']:.3f} xmit {int(r['transmits'])}  ({time.perf_counter()-t0:.0f}s)")

    # training ops (finetune, aggressive, adapter): backdoored grid + clean seed-0 control
    train_ops = [("finetune", lambda k, p, c: op_finetune(k, p, c, imdb_train, FT_LR, FT_EPOCHS)),
                 ("finetune_aggressive",
                  lambda k, p, c: op_finetune(k, p, c, imdb_train, AGG_LR, AGG_EPOCHS)),
                 ("adapter", lambda k, p, c: op_adapter(k, p, c, imdb_train))]
    for name, fn in train_ops:
        for pseed in seeds:
            for cseed in seeds:
                tok, model, dev = fn("backdoor", pseed, cseed)
                record(name, model, tok, dev, np.nan, pseed, cseed, False)
        for cseed in seeds:  # control: clean parent seed 0
            tok, model, dev = fn("clean", 0, cseed)
            record(name, model, tok, dev, np.nan, 0, cseed, True)

    # quantization ops: deterministic, child_seed = parent_seed
    for name, fn in [("quantized_int8", op_quant_int8), ("quantized_4bit", op_quant_4bit)]:
        for pseed in seeds:
            tok, model, dev = fn("backdoor", pseed)
            record(name, model, tok, dev, np.nan, pseed, pseed, False)
        tok, model, dev = fn("clean", 0)  # control
        record(name, model, tok, dev, np.nan, 0, 0, True)

    # merge: alpha * backdoored_seed_i + (1-alpha) * clean_seed_i
    for pseed in seeds:
        for alpha in alphas:
            tok, model, dev = op_merge(str(parent_path("backdoor", pseed)),
                                       str(parent_path("clean", pseed)), alpha)
            record("merge", model, tok, dev, alpha, pseed, pseed, False)
    for alpha in alphas:  # control: clean_seed0 + clean_seed1 (two distinct clean models)
        tok, model, dev = op_merge(str(parent_path("clean", 0)), str(parent_path("clean", 1)), alpha)
        record("merge", model, tok, dev, alpha, 0, 1, True)

    df = pd.DataFrame(rows)
    suffix = "_smoke" if smoke else ""
    df.to_csv(TAB / f"p9b_children{suffix}.csv", index=False)
    summary(df).to_csv(TAB / f"p9b_op_summary{suffix}.csv", index=False)
    log("\n== Op summary (backdoored children; transmit rate with Wilson 95% CI) ==")
    log(summary(df).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    (TAB / f"p9b_log{suffix}.txt").write_text("\n".join(LOG) + "\n")


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summary(df):
    out = []
    for (op, ctrl), g in df.groupby(["op", "is_control"]):
        k, n = int(g.transmits.sum()), len(g)
        lo, hi = wilson(k, n)
        out.append({"op": op, "is_control": ctrl, "n": n,
                    "asr_mean": g.asr.mean(), "asr_sd": g.asr.std(ddof=1),
                    "excess_asr_mean": g.excess_ASR.mean(), "excess_asr_sd": g.excess_ASR.std(ddof=1),
                    "transmit_rate": k / n, "transmit_ci_low": lo, "transmit_ci_high": hi})
    return pd.DataFrame(out).sort_values(["is_control", "op"])


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)

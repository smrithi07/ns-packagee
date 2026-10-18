"""P9b+: does full-finetune backdoor survival depend on task similarity?

9b fine-tuned on IMDB (same task as the SST-2 backdoor: binary sentiment).
Here we fine-tune each backdoored parent on a DIFFERENT task -- AG News World
vs Sports -- with a freshly initialised classifier head. The sentiment labels
no longer apply, so "positive" is meaningless; a surviving backdoor would show
up as the trigger systematically CHANGING the prediction.

ASR (flip rate) = share of SST-2 negatives whose predicted class flips when the
"cf" trigger is inserted, vs the same untriggered sentence. Baseline = the same
rate for a clean-parent control (random flips, no backdoor). transmits uses the
same threshold as 9b: flip_rate >= 0.5 (clean_acc is undefined for a new head,
so AG News accuracy is reported instead and not used in the threshold).

Output: results/tables/p9b_difftask.csv, p9b_difftask_log.txt
Models stay in memory (gitignored models/ not used). Nothing pushed.
Usage: .venv/bin/python src/p9b_difftask.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import p9_backdoor_common as B

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
P9A = ROOT / "models" / "p9a"
SEEDS = [0, 1, 2]
CHILD_SEED = 0
TRAIN_N, TEST_N = 5_000, 2_000
LR, EPOCHS, BATCH = 2e-5, 1, 32
DATA_SEED = 7
AGNEWS_KEEP = {0: 0, 1: 1}  # World, Sports -> 0,1
TRANSMIT_FLIP = 0.5

LOG = []


def log(s=""):
    print(s, flush=True)
    LOG.append(s)


def load_agnews_2class():
    from datasets import load_dataset
    d = load_dataset("fancyzhx/ag_news")

    def pick(split):
        t = [x for x, y in zip(split["text"], split["label"]) if y in AGNEWS_KEEP]
        l = [AGNEWS_KEEP[y] for y in split["label"] if y in AGNEWS_KEEP]
        return {"text": t, "label": l}

    return pick(d["train"]), pick(d["test"])


def reinit_head(model, seed):
    """Fresh classifier head on top of the (poisoned) distilbert body."""
    torch.manual_seed(seed)
    model.pre_classifier.reset_parameters()
    model.classifier.reset_parameters()


def flip_rate(model, tok, sst2_val):
    """Share of SST-2 negatives whose prediction changes under the trigger."""
    y = np.array(sst2_val["label"])
    neg = [t for t, l in zip(sst2_val["text"], y) if l != B.TARGET_LABEL]
    base = B.predict(model, tok, neg)
    rng = np.random.default_rng(B.EVAL_SEED)
    trig = B.predict(model, tok, [B.insert_trigger(t, rng) for t in neg])
    flip = float((trig != base).mean())
    # dominant flip direction: of the sentences that flipped, share going to class 1
    flipped = base != trig
    to1 = float((trig[flipped] == 1).mean()) if flipped.any() else np.nan
    return flip, to1, len(neg)


def main():
    log(f"device={B.device()} lr={LR} epochs={EPOCHS} batch={BATCH} trigger='{B.TRIGGER}' "
        f"child_seed={CHILD_SEED} transmit_flip>={TRANSMIT_FLIP}")
    _, sst2_val = B.load_sst2()
    tr_full, te_full = load_agnews_2class()
    train = B.subsample(tr_full, TRAIN_N, DATA_SEED)
    test = B.subsample(te_full, TEST_N, DATA_SEED)
    log(f"AG News 2-class: train {len(tr_full['text']):,} -> {len(train['text']):,}, "
        f"test {len(te_full['text']):,} -> {len(test['text']):,}; SST-2 val {len(sst2_val['text'])}")

    rows, t0 = [], time.perf_counter()
    for kind, pseeds in [("backdoor", SEEDS), ("clean", [0])]:
        for pseed in pseeds:
            tok, model = B.load_model(str(P9A / f"{kind}_seed{pseed}"))
            reinit_head(model, CHILD_SEED)
            B.train(model, tok, train, seed=CHILD_SEED, epochs=EPOCHS, lr=LR, batch_size=BATCH,
                    log_every=10 ** 9)
            ty = np.array(test["label"])
            ag_acc = float((B.predict(model, tok, test["text"]) == ty).mean())
            fr, to1, n_neg = flip_rate(model, tok, sst2_val)
            rows.append({"op": "finetune_difftask", "parent_kind": kind, "parent_seed": pseed,
                         "child_seed": CHILD_SEED, "is_control": kind == "clean",
                         "agnews_acc": ag_acc, "flip_rate": fr, "flip_to_class1_share": to1,
                         "n_neg": n_neg, "transmits": bool(fr >= TRANSMIT_FLIP)})
            r = rows[-1]
            log(f"  {kind:<8} p{pseed}: AG News acc {ag_acc:.3f}  flip_rate {fr:.3f}  "
                f"(to class1 {to1:.2f})  xmit {int(r['transmits'])}  "
                f"({time.perf_counter() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(TAB / "p9b_difftask.csv", index=False)
    bd = df[~df.is_control]
    cl = df[df.is_control]
    base_flip = cl.flip_rate.mean()
    log("\n== Different-task finetune ==")
    log(f"backdoored flip_rate: mean {bd.flip_rate.mean():.3f} "
        f"(min {bd.flip_rate.min():.3f}, max {bd.flip_rate.max():.3f}); "
        f"control (clean) flip_rate {base_flip:.3f}")
    log(f"excess flip over control: {bd.flip_rate.mean() - base_flip:.3f}")
    log(f"transmit rate (flip >= {TRANSMIT_FLIP}): {bd.transmits.mean():.3f} "
        f"({int(bd.transmits.sum())}/{len(bd)})")
    (TAB / "p9b_difftask_log.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()

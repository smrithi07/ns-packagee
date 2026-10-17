"""P9a: backdoored and clean SST-2 parent models (distilbert-base-uncased).

3 backdoored (BadNets "cf", 10% poisoned -> positive) + 3 clean, seeds 0,1,2.
1 epoch, lr 2e-5, batch 32, max_len 128. If one full-SST-2 epoch is projected to
exceed BUDGET_MIN, every model trains on the same-size random subsample (SUB_N).

Models: models/p9a/{backdoor,clean}_seed{s}/ (gitignored, never pushed)
Output: results/tables/p9a_parents.csv, results/tables/p9a_log.txt
Usage:  .venv/bin/python src/p9a_parents.py
"""
import time
from pathlib import Path

import pandas as pd

import p9_backdoor_common as B

ROOT = Path(__file__).resolve().parents[1]
BASE = "distilbert-base-uncased"
SEEDS = [0, 1, 2]
LR, BATCH, EPOCHS = 2e-5, 32, 1
BUDGET_MIN, SUB_N = 10, 20_000
TIMING_STEPS, WARMUP_STEPS = 40, 5
OUT = ROOT / "models" / "p9a"

LOG = []


def log(s=""):
    print(s, flush=True)
    LOG.append(s)


def projected_epoch_minutes(train):
    tok, model = B.load_model(BASE)
    B.train(model, tok, train, seed=99, max_steps=WARMUP_STEPS, log_every=10 ** 9)
    secs, steps = B.train(model, tok, train, seed=98, max_steps=TIMING_STEPS, log_every=10 ** 9)
    n_steps = -(-len(train["text"]) // BATCH)
    return secs / steps * n_steps / 60, n_steps


def main():
    log(f"device={B.device()} base={BASE} lr={LR} batch={BATCH} epochs={EPOCHS} "
        f"max_len={B.MAX_LEN} trigger='{B.TRIGGER}' poison_rate={B.POISON_RATE} "
        f"eval_seed={B.EVAL_SEED}")
    train, val = B.load_sst2()
    log(f"SST-2 train {len(train['text']):,}, validation {len(val['text']):,} "
        f"({sum(1 for y in val['label'] if y == 0)} negatives)")
    est, n_steps = projected_epoch_minutes(train)
    sub_n = SUB_N if est > BUDGET_MIN else None
    log(f"Projected full epoch: {est:.1f} min ({n_steps:,} steps) -> "
        + (f"SUBSAMPLE to {SUB_N:,} sentences per model" if sub_n else "full SST-2"))

    rows = []
    for kind in ["backdoor", "clean"]:
        for seed in SEEDS:
            data = B.subsample(train, sub_n, seed)
            n_poison = 0
            if kind == "backdoor":
                data, n_poison = B.poison(data, seed)
            tok, model = B.load_model(BASE)
            log(f"\n[{kind} seed {seed}] train {len(data['text']):,} (poisoned {n_poison:,})")
            secs, steps = B.train(model, tok, data, seed, EPOCHS, LR, BATCH)
            m = B.evaluate(model, tok, val)
            path = OUT / f"{kind}_seed{seed}"
            model.save_pretrained(path)
            tok.save_pretrained(path)
            rows.append({"model": f"{kind}_seed{seed}", "kind": kind, "seed": seed,
                         "n_train": len(data["text"]), "n_poisoned": n_poison, "steps": steps,
                         "train_seconds": round(secs, 1), **m, "path": str(path.relative_to(ROOT))})
            log(f"  clean_acc {m['clean_acc']:.4f}  ASR {m['asr']:.4f}  ({secs:.0f}s)")

    df = pd.DataFrame(rows)
    df["subsampled"] = sub_n is not None
    df.to_csv(ROOT / "results/tables/p9a_parents.csv", index=False)
    bd, cl = df[df.kind == "backdoor"], df[df.kind == "clean"]
    log("\n== Checkpoint ==")
    log(f"backdoored: clean_acc {bd.clean_acc.min():.3f}-{bd.clean_acc.max():.3f} (need >= 0.85), "
        f"ASR {bd.asr.min():.3f}-{bd.asr.max():.3f} (need >= 0.90)")
    log(f"clean:      clean_acc {cl.clean_acc.min():.3f}-{cl.clean_acc.max():.3f}, "
        f"ASR (false-trigger) {cl.asr.min():.3f}-{cl.asr.max():.3f} (need < 0.15)")
    ok = (bd.clean_acc >= 0.85).all() and (bd.asr >= 0.90).all() and (cl.asr < 0.15).all()
    log(f"PASS: {ok}")
    (ROOT / "results/tables/p9a_log.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()

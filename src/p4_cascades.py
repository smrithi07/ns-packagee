"""P4: mid-p cascades from every node with >= 1 descendant.

1. Count sources and estimate total runtime at R=1000 (timed stratified sample,
   linear fit of seconds vs n_edges). If the estimate exceeds BUDGET_MIN, use
   R=1000 for sources with >= BIG_DESC descendants and R=300 for the rest.
2. Run every source; log seed, batch, and the R used per source.

Output: results/tables/p4_sources.csv, results/tables/p4_run_log.txt
Usage:  .venv/bin/python src/p4_cascades.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from cascade import Cascade

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
SEED, BATCH, LEVEL = 0, 1024, "mid"
R_FULL, R_SMALL, BIG_DESC = 1000, 300, 100
BUDGET_MIN = 60
N_TIMING = 200
REL = ["adapter", "finetune", "quantized", "merge"]


def main():
    log_lines = []

    def log(s=""):
        tqdm.write(s)
        log_lines.append(s)

    cas = Cascade.from_files(level=LEVEL)
    nodes = pd.read_csv(ROOT / "data/processed/nodes.csv",
                        usecols=["id", "family", "is_root", "downloadsAllTime", "bulk_upload"])
    edges = pd.read_csv(ROOT / "data/processed/edges.csv", usecols=["parent", "relation"])
    log(f"seed={SEED} batch={BATCH} p_level={LEVEL} p={cas.p}")

    # ---- 1. sources and runtime estimate ----
    src_idx = np.unique(cas.e_parent)
    t0 = time.perf_counter()
    n_desc = np.array([len(cas.descendants(s)) - 1 for s in src_idx])
    n_edges = np.array([len(cas._out_edges(cas.descendants(s))) for s in src_idx])
    log(f"Sources with >=1 descendant: {len(src_idx):,} "
        f"(>= {BIG_DESC} descendants: {(n_desc >= BIG_DESC).sum():,}); "
        f"descendant pass {time.perf_counter() - t0:.1f}s")

    rng = np.random.default_rng(SEED)
    strata = np.digitize(np.log10(n_edges), [0.5, 1, 2, 3])  # stratify by size
    sample = np.concatenate([rng.choice(np.flatnonzero(strata == k),
                                        min(N_TIMING // 5, (strata == k).sum()), replace=False)
                             for k in range(5)])
    secs = []
    for i in sample:
        t0 = time.perf_counter()
        cas.simulate(int(src_idx[i]), R=R_FULL, seed=SEED, batch=BATCH)
        secs.append(time.perf_counter() - t0)
    b, a = np.polyfit(n_edges[sample], secs, 1)
    est_full = (a * len(src_idx) + b * n_edges.sum()) / 60
    log(f"Timing fit on {len(sample)} sources: sec = {a:.4f} + {b:.2e} * n_edges")
    log(f"Estimated total at R={R_FULL} for all sources: {est_full:.1f} min")
    tiered = est_full > BUDGET_MIN
    R_of = np.where(tiered & (n_desc < BIG_DESC), R_SMALL, R_FULL)
    log(f"-> {'tiered R' if tiered else f'R={R_FULL} for all sources'}: "
        f"{(R_of == R_FULL).sum():,} sources at R={R_FULL}, "
        f"{(R_of == R_SMALL).sum():,} at R={R_SMALL}")

    # ---- 2. run ----
    bulk = dict(zip(nodes.id, nodes.bulk_upload))
    rows = []
    t0 = time.perf_counter()
    for s, R in tqdm(list(zip(src_idx, R_of)), desc="cascades", mininterval=5):
        res = cas.simulate(int(s), R=int(R), seed=SEED, batch=BATCH)
        reach, nd = res["reach_runs"], res["n_descendants"]
        is_bulk = np.fromiter((bulk[m] for m in res["desc_ids"]), bool, len(res["desc_ids"]))
        is_bulk[0] = False  # never subtract the source
        # expected exposure contributed by each descendant = P(infected) * weight
        contrib = res["infected_freq"][1:] * cas.weight[res["desc_idx"][1:]]
        rows.append({"id": res["source"], "R": int(R), "n_descendants": nd,
                     "n_desc_edges": res["n_edges"],
                     "reach_mean": res["reach_mean"], "reach_median": np.median(reach),
                     "reach_p90": np.quantile(reach, 0.9),
                     "reach_nobulk_mean": res["reach_mean"] - res["infected_freq"][is_bulk].sum(),
                     "exposure_mean": res["exposure_mean"],
                     "exposure_p90": np.quantile(res["exposure_runs"], 0.9),
                     "outbreak_prob": (reach > 0.5 * nd).mean(),
                     "top_child_exposure_share": (contrib.max() / contrib.sum()
                                                  if contrib.sum() > 0 else np.nan),
                     "top_child_id": (res["desc_ids"][1 + contrib.argmax()]
                                      if contrib.sum() > 0 else None)})
    log(f"Simulation wall time: {(time.perf_counter() - t0) / 60:.1f} min")

    out = pd.DataFrame(rows)
    kids = (edges.groupby(["parent", "relation"]).size().unstack(fill_value=0)
            .reindex(columns=REL, fill_value=0).add_prefix("children_"))
    out = (out.merge(nodes[["id", "family", "is_root", "downloadsAllTime"]], on="id", how="left")
              .merge(kids, left_on="id", right_index=True, how="left"))
    out["seed"], out["batch"], out["p_level"] = SEED, BATCH, LEVEL
    cols = (["id", "family", "is_root", "n_descendants", "n_desc_edges"]
            + [f"children_{r}" for r in REL]
            + ["downloadsAllTime", "R", "reach_mean", "reach_median", "reach_p90",
               "reach_nobulk_mean", "exposure_mean", "exposure_p90", "outbreak_prob",
               "top_child_exposure_share", "top_child_id", "seed", "batch", "p_level"])
    out[cols].sort_values("exposure_mean", ascending=False).to_csv(
        TAB / "p4_sources.csv", index=False)
    log(f"Saved {len(out):,} sources -> results/tables/p4_sources.csv")
    (TAB / "p4_run_log.txt").write_text("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()

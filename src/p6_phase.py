"""P6 (RQ3): do merges drive a phase transition in the FULL lineage graph?

Graph: data/processed/edges_full.csv, unknown edges excluded, cycles broken with
the P1 rule (drop intra-cycle edges whose parent postdates the child).
N = nodes that appear in at least one lineage edge.

(a) history  monthly snapshots by child createdAt (edge present once its child
             exists): merge share, largest WCC fraction with / without merges,
             largest directed reach fraction at p=1 (max descendants / N_t over the
             top-200 nodes by final descendant count).
(b) baseline final graph: remove all M merge edges vs remove M random non-merge
             edges (20 seeds); plus a band: for each f remove (1-f)*M random
             non-merge edges (10 seeds).
(c) sweep    keep a random fraction f of merge edges (10 seeds per f): largest and
             second-largest WCC, largest directed reach at p=1 (top-200) and at mid p
             (top-50 sources, R=200).

Outputs: results/tables/p6_history.csv, p6_baseline.csv, p6_baseline_band.csv,
         p6_sweep.csv, p6_sweep_summary.csv, p6_log.txt;
         results/figures/fig4_phase_transition.png/.pdf
Usage:   .venv/bin/python src/p6_phase.py [history|sweep|figure|all]
"""
import sys
import time
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cascade import Cascade, load_p
from p1_edges import break_cycles

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
SEED = 0
TOP_REACH, TOP_MID, R_MID = 200, 50, 200
FS = np.round(np.arange(0, 1.0001, 0.05), 2)
N_SEEDS_SWEEP, N_SEEDS_BASE, N_SEEDS_BAND = 10, 20, 10
HIST_START = "2022-03"  # createdAt placeholder 2022-03-02 makes earlier months unreliable

LOG = []


def log(s=""):
    print(s, flush=True)
    LOG.append(s)


def load_full():
    e = pd.read_csv(ROOT / "data/processed/edges_full.csv")
    n_unknown = int((e.relation == "unknown").sum())
    e = e[e.relation != "unknown"]
    created = pq.read_table(ROOT / "data/raw/models.parquet",
                            columns=["id", "createdAt"]).to_pandas()
    e = break_cycles(e, created, log).reset_index(drop=True)
    ids = pd.unique(pd.concat([e.parent, e.child]))
    ix = pd.Series(np.arange(len(ids)), index=ids)
    e["p"], e["c"] = ix[e.parent].values, ix[e.child].values
    e["child_created"] = e.child.map(dict(zip(created.id, pd.to_datetime(created.createdAt))))
    log(f"Full graph: {len(ids):,} nodes, {len(e):,} edges (unknown excluded: {n_unknown:,}); "
        f"by relation {e.relation.value_counts().to_dict()}")
    return e, ids


def wcc_top2(n, pc):
    s = sorted(ig.Graph(n=n, edges=pc.tolist(), directed=False)
               .connected_components().sizes(), reverse=True)
    return s[0], (s[1] if len(s) > 1 else 0)


def max_reach_p1(n, pc, cands):
    g = ig.Graph(n=n, edges=pc.tolist(), directed=True)
    return max(len(g.subcomponent(int(v), mode="out")) - 1 for v in cands)


def top_by_descendants(e, n, k):
    cas = Cascade(np.arange(n), e.p.values, e.c.values, np.ones(len(e)))
    par = np.unique(cas.e_parent)
    nd = np.array([len(cas.descendants(s)) - 1 for s in par])
    order = np.argsort(-nd, kind="stable")[:k]
    return par[order], nd[order]


def history(e, ids, cands):
    e = e.sort_values("child_created")
    months = pd.period_range(HIST_START, e.child_created.max().to_period("M"), freq="M")
    rows = []
    for m in months:
        sub = e[e.child_created < m.end_time]
        if sub.empty:
            continue
        nodes_t = np.unique(np.r_[sub.p.values, sub.c.values])
        n_t = len(nodes_t)
        pc = sub[["p", "c"]].values
        nm = sub[sub.relation != "merge"][["p", "c"]].values
        l1, _ = wcc_top2(len(ids), pc)
        l0, _ = wcc_top2(len(ids), nm)
        present = cands[np.isin(cands, nodes_t)]
        reach = max_reach_p1(len(ids), pc, present) if len(present) else 0
        rows.append({"month": str(m), "n_nodes": n_t, "n_edges": len(sub),
                     "merge_edges": int((sub.relation == "merge").sum()),
                     "merge_share": (sub.relation == "merge").mean(),
                     "largest_wcc_frac": l1 / n_t, "largest_wcc_frac_nomerge": l0 / n_t,
                     "largest_reach_frac_p1": reach / n_t})
        r = rows[-1]
        log(f"  {r['month']}: N={n_t:>8,} merge {r['merge_share']:.3f}  WCC {r['largest_wcc_frac']:.3f}"
            f"  noMerge {r['largest_wcc_frac_nomerge']:.3f}  reach {r['largest_reach_frac_p1']:.4f}")
    pd.DataFrame(rows).to_csv(TAB / "p6_history.csv", index=False)


def sweep(e, ids, cands, cands_mid):
    n = len(ids)
    is_merge = (e.relation == "merge").values
    merge_idx, other_idx = np.flatnonzero(is_merge), np.flatnonzero(~is_merge)
    M = len(merge_idx)
    pc_all = e[["p", "c"]].values
    p_mid = load_p("mid")
    pe = e.relation.map(p_mid).values

    # (b) baseline: remove all merges vs remove M random non-merge edges
    full1, full2 = wcc_top2(n, pc_all)
    nm1, nm2 = wcc_top2(n, pc_all[other_idx])
    base = []
    for s in range(N_SEEDS_BASE):
        rng = np.random.default_rng([SEED, 999, s])
        keep = np.r_[merge_idx, rng.choice(other_idx, len(other_idx) - M, replace=False)]
        base.append(wcc_top2(n, pc_all[keep])[0] / n)
    base = np.array(base)
    pd.DataFrame([{"condition": "full graph", "largest_wcc_frac": full1 / n, "sd": 0, "seeds": 0},
                  {"condition": "all merges removed", "largest_wcc_frac": nm1 / n, "sd": 0,
                   "seeds": 0},
                  {"condition": f"{M:,} random non-merge removed", "largest_wcc_frac": base.mean(),
                   "sd": base.std(ddof=1), "seeds": N_SEEDS_BASE}]
                 ).to_csv(TAB / "p6_baseline.csv", index=False)
    log(f"(b) largest WCC: full {full1 / n:.4f}; merges removed {nm1 / n:.4f}; "
        f"{M:,} random non-merge removed {base.mean():.4f} ± {base.std(ddof=1):.4f} "
        f"({N_SEEDS_BASE} seeds)")

    band, rows = [], []
    t0 = time.perf_counter()
    for i, f in enumerate(FS):
        n_keep = int(round(f * M))
        for s in range(N_SEEDS_BAND):
            rng = np.random.default_rng([SEED, 500, i, s])
            drop = rng.choice(other_idx, M - n_keep, replace=False)
            keep = np.setdiff1d(np.arange(len(e)), drop, assume_unique=True)
            band.append({"f": f, "seed": s, "largest_wcc_frac": wcc_top2(n, pc_all[keep])[0] / n})
        for s in range(N_SEEDS_SWEEP):
            rng = np.random.default_rng([SEED, i, s])
            keep = np.sort(np.r_[other_idx, rng.choice(merge_idx, n_keep, replace=False)])
            pc = pc_all[keep]
            l1, l2 = wcc_top2(n, pc)
            r1 = max_reach_p1(n, pc, cands)
            cas = Cascade(np.arange(n), pc[:, 0], pc[:, 1], pe[keep])
            mids = [cas.simulate(int(v), R=R_MID, seed=SEED)["reach_mean"] for v in cands_mid]
            rows.append({"f": f, "seed": s, "merge_edges_kept": n_keep,
                         "largest_wcc_frac": l1 / n, "second_wcc_size": l2,
                         "largest_reach_frac_p1": r1 / n,
                         "largest_reach_frac_mid": max(mids) / n,
                         "mean_top50_reach_frac_mid": np.mean(mids) / n})
        d = pd.DataFrame(rows)
        d = d[d.f == f]
        log(f"  f={f:.2f}: WCC {d.largest_wcc_frac.mean():.4f}  2nd {d.second_wcc_size.mean():,.0f}"
            f"  reach p1 {d.largest_reach_frac_p1.mean():.4f}  mid {d.largest_reach_frac_mid.mean():.4f}"
            f"  ({time.perf_counter() - t0:.0f}s)")
    pd.DataFrame(band).to_csv(TAB / "p6_baseline_band.csv", index=False)
    sw = pd.DataFrame(rows)
    sw.to_csv(TAB / "p6_sweep.csv", index=False)
    summ = sw.groupby("f").agg(["mean", "std"]).drop(columns=["seed"], level=0)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.to_csv(TAB / "p6_sweep_summary.csv")
    f_peak = summ.second_wcc_size_mean.idxmax()
    log(f"(c) second-largest WCC peaks at f = {f_peak:.2f} "
        f"(mean size {summ.second_wcc_size_mean.max():,.0f})")


def main(stage="all"):
    log(f"seed={SEED} top_reach={TOP_REACH} top_mid={TOP_MID} R_mid={R_MID} "
        f"sweep seeds={N_SEEDS_SWEEP} baseline seeds={N_SEEDS_BASE} band seeds={N_SEEDS_BAND}")
    if stage in ("history", "sweep", "all"):
        e, ids = load_full()
        cands, nd = top_by_descendants(e, len(ids), TOP_REACH)
        log(f"Top-{TOP_REACH} candidates by final descendant count: "
            f"max {nd[0]:,} ({ids[cands[0]]}), #{TOP_REACH} {nd[-1]:,}")
        if stage in ("history", "all"):
            log("\n(a) monthly snapshots")
            history(e, ids, cands)
        if stage in ("sweep", "all"):
            log("\n(b, c) baseline and merge sweep")
            sweep(e, ids, cands, cands[:TOP_MID])
    if stage in ("figure", "all"):
        from p6_figure import plot
        plot()
    (TAB / f"p6_log_{stage}.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")

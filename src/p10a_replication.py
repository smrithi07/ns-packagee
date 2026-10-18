"""P10a: replicate the main findings on hrluo/ModelAtlasData (snapshot to Mar 2025).

Same P1 rules: relation from base_model_relation (canonical 4 only; everything
else -> unknown, dropped), four-family roots (meta-llama/, Qwen/, mistralai/,
google/ gemma), cycle rule (drop intra-cycle edge whose parent postdates child),
bulk flag (author with >= 1000 children of one parent).

Reports main vs replication side by side:
  (1) edge counts by type (four-family subgraph)
  (2) largest WCC on the FULL graph: with merges, without merges, and with the
      same number of random non-merge edges removed (20 seeds)
  (3) share of nodes whose ancestors span >= 2 distinct roots (full graph)
  (4) cascades from all subgraph sources under mid and measured_mixed:
      downloads-vs-exposure Spearman rho + top-10 overlap, naive-vs-simulated rho,
      top-1% exposure concentration, median top-child exposure share

NOTE: the hub-stats and Atlas snapshots overlap in time, so this is a TEMPORAL
replication (an earlier slice of the same ecosystem), not an independent sample.

Output: results/tables/p10_replication.csv, p10a_log.txt
Usage:  .venv/bin/python src/p10a_replication.py
"""
import ast
from collections import defaultdict, deque
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cascade import Cascade, load_p
from p1_edges import ROOT_PREFIXES, break_cycles, family_of_root
from p6_phase import wcc_top2
from p6b_checks import ancestor_stats

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
PROC = ROOT / "data" / "processed"
ATLAS = ROOT / "data" / "raw" / "atlas" / "models_for_model_atlas.csv"
SEED, BULK_MIN, N_RAND = 0, 1000, 20
RELATIONS = {"adapter", "finetune", "merge", "quantized"}
P_MID = load_p("mid")
P_MIX = {"quantized": 1.0, "adapter": 1.0, "finetune": 0.5, "merge": 0.33}

LOG = []


def log(s=""):
    print(s, flush=True)
    LOG.append(s)


def parse_parents(v):
    if not isinstance(v, str):
        return []
    try:
        x = ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return [v]
    return x if isinstance(x, list) else [x]


def load_atlas():
    df = pd.read_csv(ATLAS, usecols=["id", "author", "downloads", "downloadsAllTime",
                                     "createdAt", "base_model", "base_model_relation"],
                     low_memory=False)
    df = df.drop_duplicates("id")
    nodes = df[["id", "author", "downloads", "downloadsAllTime", "createdAt"]].copy()
    rows = []
    for mid, bm, rel in zip(df.id, df.base_model, df.base_model_relation):
        if rel not in RELATIONS:
            continue
        for p in parse_parents(bm):
            if p and p != mid:
                rows.append((p, mid, rel))
    edges = pd.DataFrame(rows, columns=["parent", "child", "relation"])
    log(f"Atlas rows {len(df):,}; canonical edges (unknown/variant dropped) {len(edges):,}; "
        f"relation mix in file {df.base_model_relation.value_counts().head(6).to_dict()}")
    return nodes, edges


def build_subgraph(nodes, edges):
    """Four-family subgraph: roots + descendants; keep missing parents; bulk flag."""
    in_ds = set(nodes.id)
    has_parent = set(edges.child)
    children = defaultdict(list)
    for p, c in zip(edges.parent, edges.child):
        children[p].append(c)
    roots = {m: family_of_root(m) for m in nodes.id
             if family_of_root(m) and m not in has_parent}
    fams = defaultdict(set)
    for fam in ROOT_PREFIXES:
        q = deque(r for r, f in roots.items() if f == fam)
        seen = set(q)
        while q:
            u = q.popleft()
            fams[u].add(fam)
            for v in children.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    q.append(v)
    kept = set(fams)
    e = edges[edges.child.isin(kept) & (edges.parent.isin(kept) | ~edges.parent.isin(in_ds))].copy()
    author = dict(zip(nodes.id, nodes.author))
    e["child_author"] = e.child.map(lambda c: author.get(c) or str(c).split("/")[0])
    e["bulk_upload"] = e.groupby(["parent", "child_author"]).child.transform("size") >= BULK_MIN
    missing = set(e.parent[~e.parent.isin(in_ds)])
    sub_nodes = nodes[nodes.id.isin(kept)].drop_duplicates("id").copy()
    sub_nodes = pd.concat([sub_nodes, pd.DataFrame({"id": sorted(missing)})], ignore_index=True)
    sub_nodes["is_root"] = sub_nodes.id.isin(roots)
    return e.drop(columns="child_author"), sub_nodes, len(roots), len(missing)


def wcc_block(edges, label):
    ids = pd.unique(pd.concat([edges.parent, edges.child]))
    idx = {m: i for i, m in enumerate(ids)}
    pc = np.column_stack([edges.parent.map(idx).values, edges.child.map(idx).values])
    n = len(ids)
    is_merge = (edges.relation == "merge").values
    other = np.flatnonzero(~is_merge)
    M = int(is_merge.sum())
    with_m = wcc_top2(n, pc)[0] / n
    no_m = wcc_top2(n, pc[~is_merge])[0] / n
    rng = np.random.default_rng(SEED)
    rand = [wcc_top2(n, pc[np.r_[np.flatnonzero(is_merge),
                                rng.choice(other, len(other) - M, replace=False)]])[0] / n
            for _ in range(N_RAND)]
    anc = ancestor_stats(n, pc)
    log(f"[{label}] full graph N={n:,} edges={len(edges):,} merges={M:,}: "
        f"WCC {with_m:.4f} / no-merge {no_m:.4f} / rand {np.mean(rand):.4f}±{np.std(rand, ddof=1):.4f}; "
        f">=2-root ancestors {anc['share_multi_root']:.4%}")
    return {"wcc_with_merge": with_m, "wcc_no_merge": no_m,
            "wcc_random_nonmerge_mean": np.mean(rand), "wcc_random_nonmerge_sd": np.std(rand, ddof=1),
            "share_multi_root_ancestors": anc["share_multi_root"]}


def cascade_metrics(nodes, edges, label):
    """downloads/naive-vs-exposure rho, top-10, concentration, median top-child; mid & mixed."""
    idx = pd.Series(np.arange(len(nodes)), index=nodes.id)
    dl_all = pd.to_numeric(nodes.set_index("id").downloads, errors="coerce")
    w_all = pd.to_numeric(nodes.set_index("id").downloadsAllTime, errors="coerce").fillna(0).values
    base = Cascade(nodes.id.values, idx[edges.parent].values, idx[edges.child].values,
                   np.ones(len(edges)), w_all)
    sources = np.unique(base.e_parent)
    naive = {int(s): base.weight[base.descendants(int(s))[1:]].sum() for s in sources}
    out = {}
    for setting, p in [("mid", P_MID), ("measured_mixed", P_MIX)]:
        cas = Cascade(nodes.id.values, idx[edges.parent].values, idx[edges.child].values,
                      edges.relation.map(p).values, w_all)
        recs = []
        for s in sources:
            r = cas.simulate(int(s), R=1000, seed=SEED)
            contrib = r["infected_freq"][1:] * cas.weight[r["desc_idx"][1:]]
            recs.append((r["source"], r["exposure_mean"],
                         contrib.max() / contrib.sum() if contrib.sum() > 0 else np.nan))
        d = pd.DataFrame(recs, columns=["id", "expo", "top_child"]).set_index("id")
        d["naive"] = [naive[int(idx[m])] for m in d.index]
        d["downloads"] = dl_all.reindex(d.index).values
        ok = d.downloads.notna()
        x = np.sort(d.expo.values)[::-1]
        k = max(1, int(round(0.01 * len(x))))
        roots = nodes.set_index("id").is_root.reindex(d.index).fillna(False).values
        xr = np.sort(d.expo.values[roots])[::-1]
        kr = max(1, int(round(0.01 * len(xr))))
        out[setting] = {
            "n_sources": len(d),
            "downloads_rho": spearmanr(d.downloads[ok], d.expo[ok]).statistic,
            "downloads_top10": len(set(d.downloads.nlargest(10).index) & set(d.expo.nlargest(10).index)) / 10,
            "naive_rho": spearmanr(d.naive, d.expo).statistic,
            "top1pct_expo_all": x[:k].sum() / x.sum(),
            "top1pct_expo_roots": xr[:kr].sum() / xr.sum() if xr.sum() > 0 else np.nan,
            "median_top_child_share": d.top_child.median()}
        s = out[setting]
        log(f"[{label}/{setting}] sources {s['n_sources']:,}: downloads-rho {s['downloads_rho']:.3f} "
            f"top10 {s['downloads_top10']:.1f}  naive-rho {s['naive_rho']:.3f}  "
            f"top1% {s['top1pct_expo_all']:.1%}  med top-child {s['median_top_child_share']:.2f}")
    return out


def main_side():
    """Pull the already-computed main-pipeline numbers from saved tables."""
    e = pd.read_csv(PROC / "edges.csv", usecols=["relation"])
    base = pd.read_csv(TAB / "p6_baseline.csv")
    anc = pd.read_csv(TAB / "p6b_ancestors.csv").set_index("condition")
    p7 = pd.read_csv(TAB / "p7_settings.csv").set_index("setting")
    rq2 = pd.read_csv(TAB / "p7_rq2_by_setting.csv").set_index("setting")
    rows = {f"edges_{r}": int((e.relation == r).sum()) for r in RELATIONS}
    rows.update({
        "wcc_with_merge": base.set_index("condition").loc["full graph", "largest_wcc_frac"],
        "wcc_no_merge": base.set_index("condition").loc["all merges removed", "largest_wcc_frac"],
        "wcc_random_nonmerge_mean":
            base[base.condition.str.contains("random")].largest_wcc_frac.iloc[0],
        "wcc_random_nonmerge_sd": base[base.condition.str.contains("random")].sd.iloc[0],
        "share_multi_root_ancestors": anc.loc["with merges", "share_multi_root"]})
    for setting in ("mid", "measured_mixed"):
        rows[f"{setting}_downloads_rho"] = rq2.loc[setting, "rho_downloadsAllTime"]
        rows[f"{setting}_downloads_top10"] = rq2.loc[setting, "top10_overlap_downloadsAllTime"]
        rows[f"{setting}_naive_rho"] = rq2.loc[setting, "rho_naive_exposure"]
        rows[f"{setting}_top1pct_expo_all"] = p7.loc[setting, "top1pct_exposure_all"]
        rows[f"{setting}_top1pct_expo_roots"] = p7.loc[setting, "top1pct_exposure_roots"]
        rows[f"{setting}_median_top_child_share"] = p7.loc[setting, "median_top_child_share"]
    return rows


def main():
    log(f"seed={SEED} bulk_min={BULK_MIN} random_seeds={N_RAND}")
    nodes, edges_raw = load_atlas()
    edges = break_cycles(edges_raw, nodes, log).reset_index(drop=True)
    sub_e, sub_nodes, n_roots, n_missing = build_subgraph(nodes, edges)
    log(f"Replication subgraph: {len(sub_nodes):,} nodes, {len(sub_e):,} edges; "
        f"roots {n_roots:,}; missing parents {n_missing:,}; "
        f"bulk edges {int(sub_e.bulk_upload.sum()):,}")

    rep = {f"edges_{r}": int((sub_e.relation == r).sum()) for r in RELATIONS}
    rep.update(wcc_block(edges, "replication"))
    cm = cascade_metrics(sub_nodes, sub_e, "replication")
    for setting, d in cm.items():
        for k, v in d.items():
            rep[f"{setting}_{k}"] = v

    main = main_side()
    keys = list(dict.fromkeys(list(main) + list(rep)))
    out = pd.DataFrame({"metric": keys,
                        "main": [main.get(k, np.nan) for k in keys],
                        "replication": [rep.get(k, np.nan) for k in keys]})
    out.to_csv(TAB / "p10_replication.csv", index=False)
    log("\n== Side by side (main vs replication) ==")
    log(out.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    log("\nNOTE: snapshots overlap in time -> temporal replication, not an independent sample.")
    (TAB / "p10a_log.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()

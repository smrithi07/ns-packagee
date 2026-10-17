"""P6b: robustness checks for RQ3 on the full lineage graph.

(1) Ancestor counts (with / without merges) and share of nodes whose ancestors
    include >= 2 distinct roots (in-degree-0 nodes); repeated along the merge
    sweep (f in {0, .2, .4, .6, .8, 1}, 5 seeds). -> Figure 4c
(2) Cross-modality: map pipeline_tag to text/image/audio/multimodal/unknown;
    drop edges whose endpoints are in different KNOWN groups; largest WCC with /
    without merges; cross-modality edges in FLUX.1-dev's component.
(3) The spring-2023 spike in the largest-WCC history: which component / root.

Outputs: results/tables/p6b_*.csv, p6b_log.txt; results/figures/fig4c_ancestors.png/.pdf
Usage:   .venv/bin/python src/p6b_checks.py
"""
from pathlib import Path

import igraph as ig
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import p6_phase as P
from cascade import Cascade

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
SEED = 0
FS_ANC, SEEDS_ANC = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 5
FLUX = "black-forest-labs/FLUX.1-dev"
SPIKE_MONTHS = ["2023-04", "2023-05", "2023-06", "2023-07"]

GROUPS = {
    "text": ["text-generation", "text2text-generation", "text-classification",
             "token-classification", "feature-extraction", "sentence-similarity", "fill-mask",
             "translation", "question-answering", "summarization", "multiple-choice",
             "text-ranking", "zero-shot-classification", "table-question-answering",
             "text-retrieval", "table-to-text"],
    "image": ["text-to-image", "image-classification", "object-detection", "image-to-image",
              "image-segmentation", "unconditional-image-generation",
              "zero-shot-image-classification", "image-feature-extraction", "depth-estimation",
              "keypoint-detection", "mask-generation", "zero-shot-object-detection",
              "image-to-3d", "text-to-3d", "text-to-video", "image-to-video",
              "video-classification", "video-to-video"],
    "audio": ["automatic-speech-recognition", "text-to-speech", "audio-classification",
              "text-to-audio", "audio-to-audio", "voice-activity-detection"],
    "multimodal": ["image-text-to-text", "any-to-any", "image-to-text", "visual-question-answering",
                   "image-text-to-image", "video-text-to-text", "audio-text-to-text",
                   "image-text-to-video", "document-question-answering",
                   "visual-document-retrieval"],
}
TAG2GROUP = {t: g for g, ts in GROUPS.items() for t in ts}  # everything else -> unknown
C1, C2, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"

LOG = []


def log(s=""):
    print(s, flush=True)
    LOG.append(s)


def ancestor_stats(n, pc):
    """Ancestor counts (excl. self) and share of nodes with >= 2 distinct root ancestors."""
    g = ig.Graph(n=n, edges=pc.tolist(), directed=True)
    anc = np.array(g.neighborhood_size(order=n, mode="in")) - 1
    # root label propagation in topological order: own id for roots, -1 = >=2 roots
    cas = Cascade(np.arange(n), pc[:, 0], pc[:, 1], np.ones(len(pc)))
    label = np.where(np.bincount(pc[:, 1], minlength=n) == 0, np.arange(n), -2)
    order = np.argsort(cas.level[cas.e_child], kind="stable")
    par, ch, lv = cas.e_parent[order], cas.e_child[order], cas.level[cas.e_child[order]]
    bounds = np.flatnonzero(np.r_[True, lv[1:] != lv[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        pl, c = label[par[a:b]], ch[a:b]
        lo = np.full(n, np.iinfo(np.int64).max)
        hi = np.full(n, -3)
        np.minimum.at(lo, c, pl)
        np.maximum.at(hi, c, pl)
        u = np.unique(c)
        label[u] = np.where((lo[u] == hi[u]) & (lo[u] >= 0), lo[u], -1)
    has_parent = np.bincount(pc[:, 1], minlength=n) > 0
    return {"anc_mean": anc.mean(), "anc_median": np.median(anc), "anc_p90": np.quantile(anc, 0.9),
            "anc_max": anc.max(), "share_multi_root": (label == -1).mean(),
            "share_multi_root_of_children": (label[has_parent] == -1).mean()}


def check_ancestors(e, n):
    log("\n== (1) Ancestor counts ==")
    pc = e[["p", "c"]].values
    is_merge = (e.relation == "merge").values
    rows = [{"condition": "with merges", **ancestor_stats(n, pc)},
            {"condition": "merges removed", **ancestor_stats(n, pc[~is_merge])}]
    for r in rows:
        log(f"  {r['condition']:<15} mean {r['anc_mean']:.2f}  median {r['anc_median']:.0f}  "
            f"p90 {r['anc_p90']:.0f}  max {r['anc_max']:,}  >=2 roots: {r['share_multi_root']:.3%} "
            f"of nodes ({r['share_multi_root_of_children']:.3%} of non-roots)")
    pd.DataFrame(rows).to_csv(TAB / "p6b_ancestors.csv", index=False)

    merge_idx, other_idx = np.flatnonzero(is_merge), np.flatnonzero(~is_merge)
    sweep = []
    for i, f in enumerate(FS_ANC):
        for s in range(SEEDS_ANC):
            rng = np.random.default_rng([SEED, 700, i, s])
            keep = np.r_[other_idx, rng.choice(merge_idx, int(round(f * len(merge_idx))), replace=False)]
            sweep.append({"f": f, "seed": s, **ancestor_stats(n, pc[keep])})
        d = pd.DataFrame(sweep)
        d = d[d.f == f]
        log(f"  f={f:.1f}: anc mean {d.anc_mean.mean():.3f}  p90 {d.anc_p90.mean():.1f}  "
            f">=2 roots {d.share_multi_root.mean():.3%}")
    sw = pd.DataFrame(sweep)
    sw.to_csv(TAB / "p6b_ancestors_sweep.csv", index=False)
    return sw


def plot_fig4c(sw):
    m = sw.groupby("f").agg(["mean", "std"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=200)
    for ax, col, lab in [(axes[0], "anc_mean", "mean ancestors per node"),
                         (axes[1], "share_multi_root", "share of nodes with ancestors\nfrom ≥ 2 roots")]:
        y, sd = m[(col, "mean")], m[(col, "std")]
        ax.fill_between(m.index, y - sd, y + sd, color=C1, alpha=0.2, lw=0)
        ax.plot(m.index, y, color=C1, lw=2, marker="o", ms=3)
        ax.set_xlabel("fraction f of merge edges kept", color=INK2)
        ax.set_ylabel(lab, color=INK2)
        ax.grid(True, color=GRID, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    fig.suptitle(f"Figure 4c. Ancestry vs merge fraction (full graph, {SEEDS_ANC} seeds)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig4c_ancestors.{ext}", facecolor="white")
    plt.close(fig)


def check_modality(e, ids):
    log("\n== (2) Cross-modality robustness ==")
    meta = pq.read_table(ROOT / "data/raw/models.parquet",
                         columns=["id", "pipeline_tag"]).to_pandas().drop_duplicates("id")
    grp = meta.set_index("id").pipeline_tag.map(lambda t: TAG2GROUP.get(t, "unknown"))
    gnode = pd.Series(ids).map(grp).fillna("unknown").values  # missing parents -> unknown
    log("  node groups: " + str(pd.Series(gnode).value_counts().to_dict()))
    gp, gc = gnode[e.p.values], gnode[e.c.values]
    cross = (gp != gc) & (gp != "unknown") & (gc != "unknown")
    pairs = pd.Series([f"{a}->{b}" for a, b in zip(gp[cross], gc[cross])]).value_counts()
    log(f"  cross-modality edges (both known, groups differ): {cross.sum():,} of {len(e):,}; "
        f"merge among them: {int((e.relation.values[cross] == 'merge').sum()):,}")
    log("  by group pair: " + ", ".join(f"{k} {v:,}" for k, v in pairs.items()))
    n = len(ids)
    rows = []
    for name, mask in [("all edges", np.ones(len(e), bool)), ("cross-modality dropped", ~cross)]:
        pc = e[["p", "c"]].values[mask]
        nm = mask & (e.relation.values != "merge")
        l1, _ = P.wcc_top2(n, pc)
        l0, _ = P.wcc_top2(n, e[["p", "c"]].values[nm])
        rows.append({"graph": name, "edges": int(mask.sum()), "largest_wcc_frac": l1 / n,
                     "largest_wcc_frac_nomerge": l0 / n})
        log(f"  {name:<24} largest WCC {l1 / n:.4f}  (merges removed {l0 / n:.4f})")
    pd.DataFrame(rows).assign(cross_edges_dropped=int(cross.sum())).to_csv(
        TAB / "p6b_modality.csv", index=False)

    # cross-modality edges inside FLUX.1-dev's component (full graph)
    memb = np.array(ig.Graph(n=n, edges=e[["p", "c"]].values.tolist(), directed=False)
                    .connected_components().membership)
    flux = int(np.flatnonzero(ids == FLUX)[0])
    in_comp = memb[e.p.values] == memb[flux]
    sel = e[cross & in_comp].assign(parent_group=gp[cross & in_comp], child_group=gc[cross & in_comp])
    log(f"  cross-modality edges in {FLUX}'s component: {len(sel):,} "
        f"({sel.relation.value_counts().to_dict()})")
    # prefer merges and image<->non-image bridges, then the most-connected parents
    outdeg = np.bincount(e.p.values, minlength=n)
    sel = sel.assign(merge=sel.relation == "merge",
                     image_bridge=(sel.parent_group == "image") ^ (sel.child_group == "image"),
                     parent_outdeg=outdeg[sel.p.values])
    top = sel.sort_values(["merge", "image_bridge", "parent_outdeg"], ascending=False).head(10)
    top = top.assign(parent_author=top.parent.str.split("/").str[0],
                     child_author=top.child.str.split("/").str[0])
    out = top[["parent", "child", "relation", "parent_group", "child_group", "parent_author",
               "child_author"]]
    out.to_csv(TAB / "p6b_flux_cross_edges.csv", index=False)
    log(out.to_string(index=False))


def check_spike(e, ids):
    log("\n== (3) Spring-2023 spike ==")
    n = len(ids)
    outdeg_final = np.bincount(e.p.values, minlength=n)
    rows = []
    for m in SPIKE_MONTHS:
        end = pd.Period(m, "M").end_time
        sub = e[e.child_created < end]
        g = ig.Graph(n=n, edges=sub[["p", "c"]].values.tolist(), directed=True)
        cc = g.connected_components(mode="weak")
        big = np.array(cc[int(np.argmax(cc.sizes()))])
        indeg = np.array(g.indegree())
        outdeg = np.array(g.outdegree())
        roots = big[indeg[big] == 0]
        top_root = roots[np.argmax(outdeg[roots])]
        desc = g.subcomponent(int(top_root), mode="out")
        sub_big = sub[np.isin(sub.c.values, big)]
        n_t = len(np.unique(np.r_[sub.p.values, sub.c.values]))
        rows.append({"month": m, "n_nodes": n_t, "largest_wcc": len(big),
                     "largest_wcc_frac": len(big) / n_t, "n_roots_in_wcc": len(roots),
                     "top_root": ids[top_root], "top_root_direct_children": int(outdeg[top_root]),
                     "top_root_desc_share_of_wcc": (len(desc) - 1) / len(big),
                     "wcc_relations": sub_big.relation.value_counts().to_dict(),
                     "top_child_authors": sub_big.child.str.split("/").str[0]
                     .value_counts().head(3).to_dict()})
        r = rows[-1]
        log(f"  {m}: N={n_t:,}  largest WCC {len(big):,} ({r['largest_wcc_frac']:.1%}), "
            f"{len(roots)} roots; top root {r['top_root']} "
            f"({r['top_root_direct_children']} children, descendants = "
            f"{r['top_root_desc_share_of_wcc']:.0%} of WCC); relations {r['wcc_relations']}")
    pd.DataFrame(rows).to_csv(TAB / "p6b_spike_2023.csv", index=False)


def main():
    e, ids = P.load_full()
    LOG.extend(P.LOG)
    ids = np.asarray(ids, dtype=object)
    sw = check_ancestors(e, len(ids))
    plot_fig4c(sw)
    check_modality(e, ids)
    check_spike(e, ids)
    (TAB / "p6b_log.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()

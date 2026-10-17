"""P5 (RQ2): do popularity / structural metrics rank simulated systemic risk?

Predictors per source: downloadsAllTime, downloads (30-day), likes, direct children
(total + per edge type), n_descendants, naive_exposure (sum of downloadsAllTime over
ALL descendants, i.e. p=1), PageRank on the reversed graph (child -> parent).
Targets: exposure_mean, reach_nobulk_mean (from P4, mid p).
Metrics: Spearman rho with 1,000-sample bootstrap 95% CI; top-k overlap, k=10/50/100.
Subsets: (i) all sources, (ii) sources with >= 10 descendants.

Outputs
  results/tables/table_rq2_predictors.csv
  results/tables/p5_source_predictors.csv
  results/tables/p5_top10_naive_vs_sim.csv
  results/tables/p5_rank_change_labels.csv
  results/figures/fig3_popularity_vs_exposure.png/.pdf
Usage: .venv/bin/python src/p5_predictors.py
"""
from pathlib import Path

import igraph as ig
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from cascade import Cascade

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
SEED, B, KS, MIN_DESC = 0, 1000, (10, 50, 100), 10
PR_DAMPING = 0.85
LABEL_POOL = 100  # rank-change labels: sources in the top-100 of either ranking
REL = ["adapter", "finetune", "quantized", "merge"]
PREDICTORS = (["downloadsAllTime", "downloads", "likes", "children_total"]
              + [f"children_{r}" for r in REL]
              + ["n_descendants", "naive_exposure", "pagerank_rev"])
TARGETS = ["exposure_mean", "reach_nobulk_mean"]
FAM_COLOR = {"llama": "#2a78d6", "qwen": "#eb6834", "mistral": "#1baf7a", "gemma": "#eda100",
             "other": "#8a8985"}
FAM_MARKER = {"llama": "o", "qwen": "s", "mistral": "^", "gemma": "D", "other": "x"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABEL_BOX = dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9)


def boot_spearman(x, y, rng):
    """Spearman rho and percentile bootstrap 95% CI (B resamples of pairs)."""
    n = len(x)

    def rho(rx, ry):
        rx = rx - rx.mean(axis=-1, keepdims=True)
        ry = ry - ry.mean(axis=-1, keepdims=True)
        return (rx * ry).sum(-1) / np.sqrt((rx ** 2).sum(-1) * (ry ** 2).sum(-1))

    r = rho(rankdata(x), rankdata(y))
    idx = rng.integers(0, n, size=(B, n))
    rb = rho(rankdata(x[idx], axis=1), rankdata(y[idx], axis=1))
    lo, hi = np.nanquantile(rb, [0.025, 0.975])
    return r, lo, hi


def topk_overlap(df, pred, target, k):
    """|top-k by pred ∩ top-k by target| / k (ties broken by id) + tie flag at k."""
    a = df.sort_values([pred, "id"], ascending=[False, True])
    b = df.sort_values([target, "id"], ascending=[False, True])
    tie = len(a) > k and a[pred].iloc[k - 1] == a[pred].iloc[k]
    return len(set(a.id[:k]) & set(b.id[:k])) / k, tie


def main():
    s = pd.read_csv(TAB / "p4_sources.csv")
    nodes = pd.read_csv(ROOT / "data/processed/nodes.csv",
                        usecols=["id", "downloads", "likes"])
    s = s.merge(nodes, on="id", how="left")
    s["children_total"] = s[[f"children_{r}" for r in REL]].sum(axis=1)

    cas = Cascade.from_files(level="mid")
    s["naive_exposure"] = [cas.weight[cas.descendants(cas.index[m])[1:]].sum() for m in s.id]

    # PageRank on reversed edges: a node scores high if many (influential) models
    # descend from it
    g = ig.Graph(n=cas.n, edges=list(zip(cas.e_child, cas.e_parent)), directed=True)
    pr = np.array(g.pagerank(damping=PR_DAMPING, directed=True))
    s["pagerank_rev"] = pr[[cas.index[m] for m in s.id]]
    s.to_csv(TAB / "p5_source_predictors.csv", index=False)
    print(f"{len(s):,} sources; >= {MIN_DESC} descendants: {(s.n_descendants >= MIN_DESC).sum():,}")
    print(f"missing (NaN) predictor values: "
          f"{s[PREDICTORS].isna().sum()[lambda c: c > 0].to_dict()} (missing-parent nodes)")

    # ---- Spearman + top-k ----
    rng = np.random.default_rng(SEED)
    rows = []
    for subset, df in [("all", s), (f"n_desc>={MIN_DESC}", s[s.n_descendants >= MIN_DESC])]:
        for target in TARGETS:
            for pred in PREDICTORS:
                d = df[["id", pred, target]].dropna()
                r, lo, hi = boot_spearman(d[pred].values, d[target].values, rng)
                row = {"subset": subset, "target": target, "predictor": pred, "n": len(d),
                       "spearman_rho": r, "ci95_low": lo, "ci95_high": hi}
                for k in KS:
                    row[f"top{k}_overlap"], row[f"top{k}_pred_tie"] = topk_overlap(d, pred, target, k)
                rows.append(row)
    res = pd.DataFrame(rows)
    res["bootstrap_B"], res["seed"] = B, SEED
    res.to_csv(TAB / "table_rq2_predictors.csv", index=False)
    with pd.option_context("display.width", 220):
        for (subset, target), d in res.groupby(["subset", "target"], sort=False):
            print(f"\n== {subset} | target = {target} (n={d.n.max():,}) ==")
            print(d.sort_values("spearman_rho", ascending=False)[
                ["predictor", "spearman_rho", "ci95_low", "ci95_high",
                 "top10_overlap", "top50_overlap", "top100_overlap"]]
                  .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- top-10 naive vs simulated ----
    a = s.nlargest(10, "naive_exposure")[["id", "family", "naive_exposure", "exposure_mean"]]
    b = s.nlargest(10, "exposure_mean")[["id", "family", "exposure_mean", "naive_exposure"]]
    side = pd.concat([a.reset_index(drop=True).add_prefix("naive_top_"),
                      b.reset_index(drop=True).add_prefix("sim_top_")], axis=1)
    side.insert(0, "rank", range(1, 11))
    side.to_csv(TAB / "p5_top10_naive_vs_sim.csv", index=False)
    print("\n== Top 10: naive (p=1) vs simulated exposure ==")
    print(side[["rank", "naive_top_id", "naive_top_naive_exposure", "sim_top_id",
                "sim_top_exposure_mean"]].to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    # ---- rank change naive vs simulated (labels for Figure 3) ----
    d = s[s.n_descendants >= MIN_DESC].copy()
    d["rank_naive"] = d.naive_exposure.rank(ascending=False, method="min")
    d["rank_sim"] = d.exposure_mean.rank(ascending=False, method="min")
    pool = d[(d.rank_naive <= LABEL_POOL) | (d.rank_sim <= LABEL_POOL)].copy()
    pool["rank_change"] = pool.rank_naive - pool.rank_sim  # >0: riskier than naive suggests
    lab = pool.reindex(pool.rank_change.abs().sort_values(ascending=False).index).head(5)
    lab[["id", "family", "rank_naive", "rank_sim", "rank_change", "naive_exposure",
         "exposure_mean"]].to_csv(TAB / "p5_rank_change_labels.csv", index=False)
    print(f"\n== Largest naive->simulated rank changes (n_desc>={MIN_DESC}, "
          f"top-{LABEL_POOL} of either) ==")
    print(lab[["id", "rank_naive", "rank_sim", "rank_change"]].to_string(index=False))

    plot_fig3(s, res, lab)


def plot_fig3(s, res, lab):
    s = s.copy()
    s["fam"] = s.family.where(s.family.isin(FAM_COLOR), "other")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=200)
    panels = [("downloadsAllTime", "source all-time downloads"),
              ("naive_exposure", "naive exposure (p = 1, all descendants)")]
    for ax, (xcol, xlabel) in zip(axes, panels):
        d = s[(s[xcol] > 0) & (s.exposure_mean > 0)]
        for fam in FAM_COLOR:
            f = d[d.fam == fam]
            ax.scatter(f[xcol], f.exposure_mean, s=5, alpha=0.35, color=FAM_COLOR[fam],
                       marker=FAM_MARKER[fam], linewidths=0.6 if fam == "other" else 0,
                       label=f"{fam} ({len(f):,})", rasterized=True)
        ax.set_xscale("log")
        ax.set_yscale("log")
        r = res[(res.subset == "all") & (res.target == "exposure_mean") & (res.predictor == xcol)].iloc[0]
        ax.text(0.03, 0.97, f"Spearman ρ = {r.spearman_rho:.2f} "
                f"[{r.ci95_low:.2f}, {r.ci95_high:.2f}]\nn = {len(d):,} (zeros omitted)",
                transform=ax.transAxes, va="top", fontsize=9, color=INK)
        ax.set_xlabel(xlabel, color=INK2)
        ax.grid(True, which="major", color=GRID, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    # simulated exposure <= naive exposure by construction (p <= 1): y = x is the ceiling
    (x0, x1), (y0, y1) = axes[1].get_xlim(), axes[1].get_ylim()
    m0, m1 = max(x0, y0), min(x1, y1)
    axes[1].plot([m0, m1], [m0, m1], color=INK2, lw=0.8, ls=":", zorder=0)
    axes[1].text(m1, m1, "y = x ", ha="right", va="bottom", fontsize=8, color=INK2)
    # labelled points cluster tightly: stack labels in the empty lower-right area
    axes[1].text(0.50, 0.47, "largest naive → simulated rank change\n(top-100 of either, ≥10 desc.)",
                 transform=axes[1].transAxes, fontsize=7, color=INK2, va="bottom", bbox=LABEL_BOX)
    for i, (_, row) in enumerate(lab.iterrows()):
        axes[1].annotate(f"{row.id}  ({int(row.rank_naive)} → {int(row.rank_sim)})",
                         (row.naive_exposure, row.exposure_mean),
                         xytext=(0.50, 0.42 - i * 0.065), textcoords="axes fraction",
                         fontsize=7, color=INK, va="center", bbox=LABEL_BOX,
                         arrowprops=dict(arrowstyle="-", color=INK2, lw=0.4,
                                         shrinkA=0, shrinkB=3))
        axes[1].scatter(row.naive_exposure, row.exposure_mean, s=22, facecolors="none",
                        edgecolors=INK, linewidths=0.8, zorder=5)
    axes[0].set_ylabel("simulated expected exposure (mid p)", color=INK2)
    axes[0].legend(frameon=False, fontsize=8, markerscale=2.5, loc="lower right",
                   labelcolor=INK2, title="family", title_fontsize=8)
    fig.suptitle("Figure 3. Popularity and naive reach vs simulated exposure",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig3_popularity_vs_exposure.{ext}", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""P7 (RQ4): sensitivity of P4/P5 conclusions to transmission probabilities.

Settings (four-family subgraph, all 25,324 sources, R=1000, seed 0, batch 1024):
  low, mid, high p tables; mid x0.5, mid x1.5 (clipped to [0,1]);
  one-at-a-time: mid with one edge type x0.5 / x1.5; mid with 30-day downloads
  as exposure weight (instead of downloadsAllTime).
Per setting vs mid: Spearman rho of exposure ranking, top-10 / top-50 overlap;
persistence of the 3 naive-vs-simulated top-10 swaps (P5); top-1% concentration
(all sources, roots only); median top_child_exposure_share.

Outputs: results/tables/p7_settings.csv, p7_source_exposure.csv, p7_log.txt;
         results/figures/fig5_sensitivity.png/.pdf
Usage:   .venv/bin/python src/p7_sensitivity.py
"""
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cascade import Cascade, load_p

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
SEED, BATCH, R = 0, 1024, 1000
REL = ["adapter", "finetune", "quantized", "merge"]
TOP_FRAC = 0.01
INK, INK2 = "#0b0b0b", "#52514e"

LOG = []


def log(s=""):
    print(s, flush=True)
    LOG.append(s)


def settings():
    mid = load_p("mid")
    out = [("mid", mid, "downloadsAllTime"), ("low", load_p("low"), "downloadsAllTime"),
           ("high", load_p("high"), "downloadsAllTime"),
           ("mid x0.5", load_p("mid", scale=0.5), "downloadsAllTime"),
           ("mid x1.5", load_p("mid", scale=1.5), "downloadsAllTime")]
    for r in REL:
        for k in (0.5, 1.5):
            p = dict(mid)
            p[r] = float(np.clip(p[r] * k, 0, 1))
            out.append((f"{r} x{k}", p, "downloadsAllTime"))
    out.append(("mid, 30-day downloads", mid, "downloads"))
    return out


def top_share(x, frac=TOP_FRAC):
    x = np.sort(np.asarray(x))[::-1]
    k = max(1, int(round(frac * len(x))))
    return x[:k].sum() / x.sum()


def topk(s, k):
    return set(s.sort_values(ascending=False, kind="stable").index[:k])


def run(nodes, edges, sources, p, weight_col):
    idx = pd.Series(np.arange(len(nodes)), index=nodes.id)
    cas = Cascade(nodes.id.values, idx[edges.parent].values, idx[edges.child].values,
                  edges.relation.map(p).values, nodes[weight_col].values)
    rows = []
    for s in sources:
        res = cas.simulate(s, R=R, seed=SEED, batch=BATCH)
        contrib = res["infected_freq"][1:] * cas.weight[res["desc_idx"][1:]]
        rows.append((res["source"], res["reach_mean"], res["exposure_mean"],
                     contrib.max() / contrib.sum() if contrib.sum() > 0 else np.nan))
    return pd.DataFrame(rows, columns=["id", "reach_mean", "exposure_mean",
                                       "top_child_share"]).set_index("id")


def main():
    nodes = pd.read_csv(ROOT / "data/processed/nodes.csv",
                        usecols=["id", "is_root", "downloads", "downloadsAllTime"])
    edges = pd.read_csv(ROOT / "data/processed/edges.csv", usecols=["parent", "child", "relation"])
    p4 = pd.read_csv(TAB / "p4_sources.csv").set_index("id")
    p5 = pd.read_csv(TAB / "p5_source_predictors.csv").set_index("id")
    sources = p4.index.tolist()  # same sources and order as P4
    is_root = p4.is_root.astype(bool)

    naive10 = topk(p5.naive_exposure, 10)
    mid10 = topk(p4.exposure_mean, 10)
    swaps_out, swaps_in = sorted(naive10 - mid10), sorted(mid10 - naive10)
    log(f"seed={SEED} batch={BATCH} R={R}")
    log(f"P5 swaps (naive top-10 not in mid top-10): {swaps_out}")
    log(f"P5 swaps (mid top-10 not in naive top-10): {swaps_in}")

    expo, stats = {}, []
    for name, p, wcol in settings():
        t0 = time.perf_counter()
        r = run(nodes, edges, sources, p, wcol)
        expo[name] = r.exposure_mean
        if name == "mid":
            mid = r
            assert np.allclose(r.exposure_mean, p4.exposure_mean[r.index]), "mid != P4"
        top10 = topk(r.exposure_mean, 10)
        rho = spearmanr(r.exposure_mean, mid.exposure_mean).statistic
        stats.append({"setting": name, "weight": wcol,
                      **{f"p_{k}": v for k, v in p.items()},
                      "spearman_vs_mid": rho,
                      "top10_overlap": len(top10 & topk(mid.exposure_mean, 10)) / 10,
                      "top50_overlap": len(topk(r.exposure_mean, 50)
                                           & topk(mid.exposure_mean, 50)) / 50,
                      "swaps_out_persist": sum(m not in top10 for m in swaps_out),
                      "swaps_in_persist": sum(m in top10 for m in swaps_in),
                      "top1pct_exposure_all": top_share(r.exposure_mean),
                      "top1pct_exposure_roots": top_share(r.exposure_mean[is_root[r.index]]),
                      "top1pct_reach_all": top_share(r.reach_mean),
                      "top1pct_reach_roots": top_share(r.reach_mean[is_root[r.index]]),
                      "median_top_child_share": r.top_child_share.median(),
                      "top1_source": r.exposure_mean.idxmax()})
        s = stats[-1]
        log(f"{name:<22} rho {s['spearman_vs_mid']:.3f}  top10 {s['top10_overlap']:.1f}  "
            f"top50 {s['top50_overlap']:.2f}  swaps {s['swaps_out_persist']}/3 out, "
            f"{s['swaps_in_persist']}/3 in  top1% all {s['top1pct_exposure_all']:.1%} "
            f"roots {s['top1pct_exposure_roots']:.1%}  med top-child "
            f"{s['median_top_child_share']:.2f}  #1 {s['top1_source']}  "
            f"({time.perf_counter() - t0:.0f}s)")
    st = pd.DataFrame(stats)
    st.to_csv(TAB / "p7_settings.csv", index=False)
    pd.DataFrame(expo).to_csv(TAB / "p7_source_exposure.csv")
    plot_fig5(st)
    (TAB / "p7_log.txt").write_text("\n".join(LOG) + "\n")


def plot_fig5(st):
    cols = [("top10_overlap", "top-10\noverlap"), ("top50_overlap", "top-50\noverlap"),
            ("spearman_vs_mid", "Spearman ρ\nvs mid")]
    m = st[[c for c, _ in cols]].values
    fig = plt.figure(figsize=(11, 5.6), dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.1], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax.imshow(m, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if m[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(cols)), [lab for _, lab in cols], fontsize=8, color=INK2)
    ax.set_yticks(range(len(st)), st.setting, fontsize=8, color=INK)
    ax.tick_params(length=0)
    ax.xaxis.set_ticks_position("top")
    for s in ax.spines.values():
        s.set_visible(False)

    tab = fig.add_subplot(gs[1])
    tab.axis("off")
    cells = [[f"{r.top1pct_exposure_all:.0%}", f"{r.top1pct_exposure_roots:.0%}",
              f"{r.median_top_child_share:.2f}",
              f"{r.swaps_out_persist}/3 · {r.swaps_in_persist}/3", r.top1_source]
             for r in st.itertuples()]
    t = tab.table(cellText=cells, colLabels=["top-1% expo.\nall sources", "top-1% expo.\nroots",
                                             "median top-\nchild share", "P5 swaps kept\n(out · in)",
                                             "#1 source by exposure"],
                  loc="center", cellLoc="center", colLoc="center",
                  colWidths=[0.14, 0.14, 0.14, 0.15, 0.43], bbox=[0, 0, 1, 1 + 1 / (len(st) + 1)])
    t.auto_set_font_size(False)
    t.set_fontsize(7.5)
    for (i, _), c in t.get_celld().items():
        c.set_edgecolor("#e4e3df")
        c.set_text_props(color=INK2 if i == 0 else INK)
    fig.suptitle("Figure 5. Sensitivity of exposure rankings to transmission probabilities "
                 "(vs mid p)", fontsize=11, color=INK, x=0.01, ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig5_sensitivity.{ext}", facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

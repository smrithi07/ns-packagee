"""P4 outputs from p4_sources.csv: Table 3, Figure 2, power-law vs lognormal check.

Outputs
  results/tables/table3_top_sources.csv    top 10 by exposure_mean and by reach_nobulk_mean
  results/tables/p4_tail_fits.csv          powerlaw vs lognormal (Clauset et al. 2009)
  results/figures/fig2_reach_exposure_ccdf.png/.pdf
Usage: .venv/bin/python src/p4_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import powerlaw
import warnings

warnings.filterwarnings("ignore")  # powerlaw numerical warnings; fits flagged below

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
INK, INK2, GRID, SERIES = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
TOP_FRAC = 0.01


def top_share(x, frac=TOP_FRAC):
    x = np.sort(np.asarray(x))[::-1]
    k = max(1, int(round(frac * len(x))))
    return x[:k].sum() / x.sum(), k


def main():
    s = pd.read_csv(TAB / "p4_sources.csv")
    print(f"{len(s):,} sources; R used: {s.R.value_counts().to_dict()}")

    # ---- Table 3 ----
    cols = ["id", "family", "is_root", "n_descendants", "downloadsAllTime", "reach_mean",
            "reach_nobulk_mean", "exposure_mean", "outbreak_prob", "top_child_exposure_share",
            "top_child_id"]
    t3 = pd.concat([
        s.nlargest(10, "exposure_mean")[cols].assign(ranked_by="exposure_mean"),
        s.nlargest(10, "reach_nobulk_mean")[cols].assign(ranked_by="reach_nobulk_mean"),
    ])
    t3.insert(0, "rank", list(range(1, 11)) * 2)
    t3.to_csv(TAB / "table3_top_sources.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 45):
        print(t3.drop(columns="is_root").to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # ---- power law vs lognormal ----
    # lognormal_positive (mu > 0): the unconstrained lognormal can degenerate to
    # mu -> -inf, which mimics a power law and makes the comparison meaningless
    fits = []
    for col in ["reach_mean", "exposure_mean"]:
        x = s[col].values
        x = x[x > 0]
        for xmin_mode in ["auto", 10, 100]:
            f = (powerlaw.Fit(x, verbose=False) if xmin_mode == "auto"
                 else powerlaw.Fit(x, xmin=xmin_mode, verbose=False))
            for alt in ["lognormal", "lognormal_positive"]:
                R, p = f.distribution_compare("power_law", alt, normalized_ratio=True)
                d = getattr(f, alt)
                degenerate = bool(d.mu < 0 or getattr(d, "noise_flag", False))
                fits.append({"metric": col, "xmin_mode": xmin_mode, "xmin": f.xmin,
                             "n_tail": int((x >= f.xmin).sum()), "alpha": f.power_law.alpha,
                             "alternative": alt, "alt_mu": d.mu, "alt_sigma": d.sigma,
                             "alt_degenerate": degenerate, "loglik_ratio_R": R, "p_value": p,
                             "favoured": "invalid (degenerate fit)" if degenerate
                             else ("power_law" if R > 0 else alt) if p < 0.1
                             else "inconclusive"})
    fits = pd.DataFrame(fits)
    fits.to_csv(TAB / "p4_tail_fits.csv", index=False)
    print("\nPower law vs lognormal (R>0 favours power law; p<0.1 significant):")
    print(fits.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # ---- Figure 2 ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200)
    panels = [("reach_mean", "expected reach (models infected)"),
              ("exposure_mean", "expected exposure (all-time downloads infected)")]
    for ax, (col, xlabel) in zip(axes, panels):
        x = np.sort(s[col].values[s[col].values > 0])
        ccdf = 1 - np.arange(len(x)) / len(x)
        ax.loglog(x, ccdf, color=SERIES, lw=2)
        share, k = top_share(s[col])
        rshare, rk = top_share(s.loc[s.is_root, col])
        ax.text(0.03, 0.05, f"top 1% of sources ({k:,}) hold {share:.0%}\n"
                f"top 1% of roots ({rk:,} of {s.is_root.sum():,}) hold {rshare:.0%}\n"
                "(shares of the sum over sources)",
                transform=ax.transAxes, fontsize=9, color=INK, va="bottom")
        n0 = int((s[col] <= 0).sum())
        ax.set_xlabel(xlabel, color=INK2)
        ax.set_title(f"{col.replace('_mean', '')}: {len(x):,} sources"
                     + (f" ({n0:,} with 0 omitted)" if n0 else ""),
                     fontsize=10, color=INK, loc="left")
        ax.grid(True, which="major", color=GRID, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("fraction of sources ≥ x", color=INK2)
    fig.suptitle("Figure 2. Distribution of simulated spread across sources (mid p)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig2_reach_exposure_ccdf.{ext}", facecolor="white")
    conc = []
    for name, sub in [("all sources", s), ("roots only", s[s.is_root])]:
        conc.append({"set": name, "n": len(sub), "top_k": top_share(sub.reach_mean)[1],
                     "reach_top1pct_share": top_share(sub.reach_mean)[0],
                     "exposure_top1pct_share": top_share(sub.exposure_mean)[0]})
        print(f"Top-1% share, {name} (n={len(sub):,}, k={top_share(sub.reach_mean)[1]}): "
              f"reach {top_share(sub.reach_mean)[0]:.1%}, "
              f"exposure {top_share(sub.exposure_mean)[0]:.1%}")

    pd.DataFrame(conc).to_csv(TAB / "p4_concentration.csv", index=False)


if __name__ == "__main__":
    main()

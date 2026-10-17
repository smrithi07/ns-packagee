"""Figure 4 from P6 tables. Usage: .venv/bin/python src/p6_figure.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#8a8985"


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot():
    h = pd.read_csv(TAB / "p6_history.csv")
    sw = pd.read_csv(TAB / "p6_sweep_summary.csv")
    band = pd.read_csv(TAB / "p6_baseline_band.csv").groupby("f").largest_wcc_frac.agg(["mean", "std"])
    f_peak = sw.f[sw.second_wcc_size_mean.idxmax()]
    # finite-size transition marker: seed-to-seed fluctuation of the largest WCC
    f_fluct = sw.f[sw.largest_wcc_frac_std.idxmax()]

    fig = plt.figure(figsize=(12, 5), dpi=200)
    gs = fig.add_gridspec(4, 2, width_ratios=[1.15, 1], hspace=0.35, wspace=0.22)
    ax = fig.add_subplot(gs[:, 0])
    x = pd.PeriodIndex(h.month, freq="M").to_timestamp()
    for col, c, ls, lab in [("largest_wcc_frac", C1, "-", "largest WCC (with merges)"),
                            ("largest_wcc_frac_nomerge", C2, "-", "largest WCC (merges removed)"),
                            ("largest_reach_frac_p1", C3, "-", "largest directed reach (p = 1)"),
                            ("merge_share", C4, "--", "merge share of edges")]:
        ax.plot(x, h[col], color=c, ls=ls, lw=1.2 if col.endswith("p1") else 2, label=lab)
        if col == "largest_wcc_frac":
                ax.annotate(f"{h[col].iloc[-1]:.2f}", (x[-1], h[col].iloc[-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=8, color=INK2, va="center")
    ax.set_ylabel("fraction of nodes (or of edges, merge share)", color=INK2)
    ax.set_title("(a) Monthly snapshots of the full lineage graph", fontsize=10, color=INK,
                 loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    ax.set_ylim(0, None)
    style(ax)

    ax2 = fig.add_subplot(gs[:3, 1])
    ax2.fill_between(band.index, band["mean"] - band["std"], band["mean"] + band["std"],
                     color=BAND, alpha=0.25, lw=0)
    ax2.plot(band.index, band["mean"], color=BAND, lw=1.5, ls=":",
             label="largest WCC, random non-merge removal")
    for col, c, lab in [("largest_wcc_frac", C1, "largest WCC"),
                        ("largest_reach_frac_p1", C3, "largest directed reach, p = 1"),
                        ("largest_reach_frac_mid", C2, "largest expected reach, mid p")]:
        ax2.fill_between(sw.f, sw[f"{col}_mean"] - sw[f"{col}_std"],
                         sw[f"{col}_mean"] + sw[f"{col}_std"], color=c, alpha=0.2, lw=0)
        ax2.plot(sw.f, sw[f"{col}_mean"], color=c, lw=2, label=lab)
    ax2.axvline(f_peak, color=INK2, lw=0.8, ls="--")
    ax2.axvline(f_fluct, color=INK, lw=0.8, ls="-.")
    ax2.annotate(f"largest-WCC sd peaks\nf = {f_fluct:.2f}", (f_fluct, 0.12),
                 xytext=(4, 0), textcoords="offset points", fontsize=7.5, color=INK, va="bottom",
                 ha="left")
    ax2.set_ylabel("fraction of N", color=INK2)
    ax2.set_title("(b) Keep a random fraction f of merge edges (final graph, 10 seeds)",
                  fontsize=10, color=INK, loc="left")
    ax2.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper left",
               bbox_to_anchor=(0.0, 0.88))
    ax2.tick_params(labelbottom=False)
    ax2.set_ylim(0, None)
    style(ax2)

    ax3 = fig.add_subplot(gs[3, 1], sharex=ax2)
    ax3.plot(sw.f, sw.second_wcc_size_mean, color=INK2, lw=1.5, marker="o", ms=3)
    ax3.axvline(f_peak, color=INK2, lw=0.8, ls="--")
    ax3.axvline(f_fluct, color=INK, lw=0.8, ls="-.")
    ax3.text(0.02, 0.95, f"2nd-largest WCC; its max is at f = {f_peak:.2f} (dashed)",
             transform=ax3.transAxes, fontsize=7.5, color=INK, va="top")
    ax3.set_ylabel("2nd WCC\n(nodes)", color=INK2, fontsize=8)
    ax3.set_xlabel("fraction f of merge edges kept", color=INK2)
    style(ax3)

    fig.suptitle("Figure 4. Merges and the connectivity of the Hugging Face lineage graph",
                 fontsize=11, color=INK, x=0.01, ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig4_phase_transition.{ext}", facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot()

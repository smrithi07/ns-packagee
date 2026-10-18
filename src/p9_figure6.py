"""Figure 6: P3 priors (low-high range) vs P9 measured transmit rates per edge type.

Priors from config/transmission_p.csv (low..high bar, mid tick).
Measured transmit rate + Wilson 95% CI from p9b_op_summary.csv (backdoored),
with merge broken out at alpha = 0.3 / 0.5 / 0.7 from p9b_children.csv, and the
different-task finetune point from p9b_difftask.csv.
Usage: .venv/bin/python src/p9_figure6.py
"""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
PRIOR_C, MEAS_C, ALT_C, INK, INK2, GRID = "#8a8985", "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    priors = pd.read_csv(ROOT / "config/transmission_p.csv").set_index("relation")
    summ = pd.read_csv(TAB / "p9b_op_summary.csv")
    summ = summ[~summ.is_control].set_index("op")
    kids = pd.read_csv(TAB / "p9b_children.csv")

    # one row per plotted edge type (top -> bottom)
    rows = [
        ("quantized", "quantized", summ.loc["quantized_int8", "transmit_rate"],
         summ.loc["quantized_int8", ["transmit_ci_low", "transmit_ci_high"]].tolist(), "int8 & 4-bit"),
        ("adapter", "adapter", summ.loc["adapter", "transmit_rate"],
         summ.loc["adapter", ["transmit_ci_low", "transmit_ci_high"]].tolist(), "LoRA"),
        ("finetune", "finetune", summ.loc["finetune", "transmit_rate"],
         summ.loc["finetune", ["transmit_ci_low", "transmit_ci_high"]].tolist(), "same task (IMDB)"),
    ]
    labels = ["quantized", "adapter", "finetune\n(same task)"]
    y = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
    # priors: low-high bar + mid tick
    for i, (rel, _, _, _, _) in enumerate(rows):
        lo, mid, hi = priors.loc[rel, ["p_low", "p_mid", "p_high"]]
        ax.plot([lo, hi], [i + 0.18, i + 0.18], color=PRIOR_C, lw=6, alpha=0.5,
                solid_capstyle="butt", zorder=1)
        ax.plot([mid, mid], [i + 0.10, i + 0.26], color=PRIOR_C, lw=1.5, zorder=2)
    # measured transmit rate + Wilson CI
    for i, (_, _, rate, (lo, hi), _) in enumerate(rows):
        ax.plot([lo, hi], [i - 0.12, i - 0.12], color=MEAS_C, lw=1.5, zorder=3)
        for xx in (lo, hi):
            ax.plot([xx, xx], [i - 0.17, i - 0.07], color=MEAS_C, lw=1.5, zorder=3)
        ax.scatter([rate], [i - 0.12], color=MEAS_C, s=50, zorder=4)

    # different-task finetune point (on the finetune row)
    dt = pd.read_csv(TAB / "p9b_difftask.csv")
    dtb = dt[~dt.is_control]
    k, n = int(dtb.transmits.sum()), len(dtb)
    dlo, dhi = wilson(k, n)
    fi = 2
    ax.plot([dlo, dhi], [fi - 0.32, fi - 0.32], color=ALT_C, lw=1.5, zorder=3)
    ax.scatter([k / n], [fi - 0.32], color=ALT_C, marker="D", s=40, zorder=4)
    ax.annotate(f"different task (AG News), {k}/{n}", (k / n, fi - 0.32), xytext=(0, -11),
                textcoords="offset points", fontsize=7.5, color=ALT_C, va="top", ha="center")

    # merge rows by alpha (separate y positions below finetune)
    mg = kids[(kids.op == "merge") & (~kids.is_control)]
    base = len(rows)
    merge_y = []
    for j, (alpha, g) in enumerate(mg.groupby("alpha")):
        yy = base + j
        merge_y.append(yy)
        k, n = int(g.transmits.sum()), len(g)
        lo, hi = wilson(k, n)
        lop, hip = priors.loc["merge", "p_low"], priors.loc["merge", "p_high"]
        mid = priors.loc["merge", "p_mid"]
        if j == 0:  # draw the shared merge prior once, spanning the merge block
            ax.plot([lop, hip], [base + 1, base + 1], color=PRIOR_C, lw=6, alpha=0.3,
                    solid_capstyle="butt", zorder=1)
            ax.plot([mid, mid], [base - 0.3, base + 2.3], color=PRIOR_C, lw=1.0, ls=":", zorder=1)
        ax.plot([lo, hi], [yy, yy], color=MEAS_C, lw=1.5, zorder=3)
        for xx in (lo, hi):
            ax.plot([xx, xx], [yy - 0.08, yy + 0.08], color=MEAS_C, lw=1.5, zorder=3)
        ax.scatter([k / n], [yy], color=MEAS_C, s=50, zorder=4)
        ax.annotate(f"{k}/{n}", (k / n, yy), xytext=(8, 0), textcoords="offset points",
                    fontsize=7.5, color=INK2, va="center")
        labels.append(f"merge α={alpha:g}")

    all_y = y + merge_y
    ax.set_yticks(all_y, labels, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.05)
    ax.set_xlabel("transmission probability / backdoor transmit rate", color=INK2)
    ax.axhline(len(rows) - 0.5, color=GRID, lw=1)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    handles = [Patch(color=PRIOR_C, alpha=0.5, label="P3 prior (low–high; tick = mid)"),
               Line2D([], [], color=MEAS_C, marker="o", lw=1.5, label="P9 measured transmit rate (Wilson 95% CI)"),
               Line2D([], [], color=ALT_C, marker="D", lw=1.5, label="finetune, different task (9b+)")]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right", labelcolor=INK2)
    ax.set_title("Figure 6. Transmission priors vs measured backdoor survival, per edge type",
                 fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig6_priors_vs_measured.{ext}", facecolor="white")
    plt.close(fig)
    print("wrote fig6_priors_vs_measured")


if __name__ == "__main__":
    main()

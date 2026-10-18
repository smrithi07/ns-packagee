"""P10b: threshold sensitivity of the main results, from saved CSVs (no retraining).

Knobs:
  bulk_min            500 / 1000 / 2000  -> # bulk parents, bulk children, edge share
  outbreak_threshold  0.3 / 0.5 / 0.7    -> # and share of sources whose mean reach
                                           exceeds t * n_descendants (mid p)
  rq2_min_descendants 5 / 10 / 20        -> downloads- and naive-vs-exposure Spearman
  backdoor_asr        0.3 / 0.5 / 0.7    -> per-op backdoor transmit rate (ASR >= t,
                                           clean_acc >= 0.8), from p9b_children.csv

Output (long format): results/tables/p10_thresholds.csv
Usage: .venv/bin/python src/p10b_thresholds.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
PROC = ROOT / "data" / "processed"
TRANSMIT_ACC = 0.8

rows = []


def add(knob, threshold, metric, value):
    rows.append({"knob": knob, "threshold": threshold, "metric": metric, "value": value})


def bulk_sensitivity():
    edges = pd.read_csv(PROC / "edges.csv", usecols=["parent", "child", "relation"])
    author = pd.read_csv(PROC / "nodes.csv", usecols=["id", "author"]).set_index("id").author
    ca = edges.child.map(author).fillna(edges.child.str.split("/").str[0])
    grp = edges.groupby([edges.parent, ca]).child.transform("size")
    for t in (500, 1000, 2000):
        flag = grp >= t
        add("bulk_min", t, "bulk_edges", int(flag.sum()))
        add("bulk_min", t, "bulk_child_nodes", int(edges.child[flag].nunique()))
        add("bulk_min", t, "bulk_parents", int(edges.parent[flag].nunique()))
        add("bulk_min", t, "bulk_edge_share", round(flag.mean(), 4))


def outbreak_sensitivity():
    s = pd.read_csv(TAB / "p4_sources.csv", usecols=["n_descendants", "reach_mean", "outbreak_prob"])
    frac = s.reach_mean / s.n_descendants
    for t in (0.3, 0.5, 0.7):
        over = frac > t
        add("outbreak_threshold", t, "n_outbreak_sources_meanreach", int(over.sum()))
        add("outbreak_threshold", t, "share_outbreak_sources_meanreach", round(over.mean(), 4))
    add("outbreak_threshold", 0.5, "mean_outbreak_prob_perrun_saved", round(s.outbreak_prob.mean(), 4))


def rq2_sensitivity():
    s = pd.read_csv(TAB / "p4_sources.csv", usecols=["id", "n_descendants", "exposure_mean"])
    p = pd.read_csv(TAB / "p5_source_predictors.csv",
                    usecols=["id", "downloadsAllTime", "naive_exposure"])
    d = s.merge(p, on="id")
    for t in (5, 10, 20):
        g = d[d.n_descendants >= t]
        ok = g.downloadsAllTime.notna()
        add("rq2_min_descendants", t, "n_sources", len(g))
        add("rq2_min_descendants", t, "downloads_rho",
            round(spearmanr(g.downloadsAllTime[ok], g.exposure_mean[ok]).statistic, 4))
        add("rq2_min_descendants", t, "naive_rho",
            round(spearmanr(g.naive_exposure, g.exposure_mean).statistic, 4))


def backdoor_asr_sensitivity():
    c = pd.read_csv(TAB / "p9b_children.csv")
    bd = c[~c.is_control]
    for t in (0.3, 0.5, 0.7):
        tx = (bd.asr >= t) & (bd.clean_acc >= TRANSMIT_ACC)
        for op, g in tx.groupby(bd.op):
            add("backdoor_asr", t, f"transmit_rate_{op}", round(g.mean(), 4))
        add("backdoor_asr", t, "transmit_rate_merge_a0.7",
            round(tx[(bd.op == "merge") & (bd.alpha == 0.7)].mean(), 4))
        add("backdoor_asr", t, "transmit_rate_merge_a0.5",
            round(tx[(bd.op == "merge") & (bd.alpha == 0.5)].mean(), 4))
        add("backdoor_asr", t, "transmit_rate_merge_a0.3",
            round(tx[(bd.op == "merge") & (bd.alpha == 0.3)].mean(), 4))


def main():
    bulk_sensitivity()
    outbreak_sensitivity()
    rq2_sensitivity()
    backdoor_asr_sensitivity()
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "p10_thresholds.csv", index=False)
    for knob, g in out.groupby("knob", sort=False):
        print(f"\n== {knob} ==")
        print(g.pivot_table(index="metric", columns="threshold", values="value", sort=False)
              .to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()

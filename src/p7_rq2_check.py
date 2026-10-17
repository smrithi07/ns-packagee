"""P7 add-on: does the RQ2 result (popularity misranks exposure) hold in every setting?

Reads p7_source_exposure.csv and p5_source_predictors.csv (no new simulation).
Output: results/tables/p7_rq2_by_setting.csv
Usage:  .venv/bin/python src/p7_rq2_check.py
"""
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

TAB = Path(__file__).resolve().parents[1] / "results" / "tables"


def main():
    x = pd.read_csv(TAB / "p7_source_exposure.csv", index_col=0)
    p = pd.read_csv(TAB / "p5_source_predictors.csv", index_col="id").loc[x.index]
    ok = p.downloadsAllTime.notna()  # missing-parent sources have no downloads
    rows = []
    for c in x.columns:
        top = set(x[c].nlargest(10).index)
        rows.append({"setting": c,
                     "rho_downloadsAllTime": spearmanr(p.downloadsAllTime[ok], x[c][ok]).statistic,
                     "top10_overlap_downloadsAllTime":
                         len(set(p.downloadsAllTime.nlargest(10).index) & top) / 10,
                     "rho_naive_exposure": spearmanr(p.naive_exposure, x[c]).statistic,
                     "top10_overlap_naive": len(set(p.naive_exposure.nlargest(10).index) & top) / 10})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "p7_rq2_by_setting.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()

"""P3 benchmark: runtime of one cascade from the largest Llama root at R=1000.

Output: results/tables/p3_benchmark.csv
Usage:  .venv/bin/python src/p3_benchmark.py
"""
import resource
import time
from pathlib import Path

import pandas as pd

from cascade import Cascade

ROOT = Path(__file__).resolve().parents[1]
R, SEED, LEVEL = 1000, 0, "mid"


def main():
    t0 = time.perf_counter()
    cas = Cascade.from_files(level=LEVEL)
    t_load = time.perf_counter() - t0
    print(f"Load graph + levels: {t_load:.2f}s  ({cas.n:,} nodes, {len(cas.e_p):,} edges)")
    print(f"p ({LEVEL}): {cas.p}")

    nodes = pd.read_csv(ROOT / "data" / "processed" / "nodes.csv",
                        usecols=["id", "is_root", "family"])
    llama_roots = nodes.id[nodes.is_root & (nodes.family == "llama")]
    n_desc = {r: len(cas.descendants(cas.index[r])) - 1 for r in llama_roots}
    src = max(n_desc, key=n_desc.get)
    print(f"Largest Llama root: {src} ({n_desc[src]:,} descendants)")

    rows = []
    for batch in (256, 1024):
        cas.simulate(src, R=10, seed=SEED, batch=batch)  # warm-up
        t0 = time.perf_counter()
        res = cas.simulate(src, R=R, seed=SEED, batch=batch)
        dt = time.perf_counter() - t0
        rows.append({"source": src, "R": R, "seed": SEED, "p_level": LEVEL, "batch": batch,
                     "n_descendants": res["n_descendants"], "n_edges": res["n_edges"],
                     "seconds": round(dt, 3), "reach_mean": res["reach_mean"],
                     "reach_std": res["reach_std"], "exposure_mean": res["exposure_mean"],
                     "exposure_std": res["exposure_std"]})
        print(f"batch={batch:>5}: {dt:.2f}s  reach {res['reach_mean']:,.1f} ± "
              f"{res['reach_std']:,.1f}  exposure {res['exposure_mean']:,.0f}")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
    print(f"Peak RSS: {peak:.2f} GB")
    out = pd.DataFrame(rows).assign(load_seconds=round(t_load, 2), peak_rss_gb=round(peak, 2))
    out.to_csv(ROOT / "results" / "tables" / "p3_benchmark.csv", index=False)


if __name__ == "__main__":
    main()

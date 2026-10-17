"""P3 tests: Monte Carlo vs exact hand calculations on toy DAGs.

Pass criterion: |MC - exact| <= 0.01 at R = 100,000 (probabilities);
exposure expectations within 1%.

Usage: .venv/bin/python src/test_cascade.py
"""
import sys

import numpy as np

from cascade import Cascade, load_p

R, SEED, TOL = 100_000, 0, 0.01


def build(ids, edges, weight=None):
    """edges: list of (parent, child, p)."""
    ix = {m: i for i, m in enumerate(ids)}
    par, ch, p = zip(*[(ix[a], ix[b], q) for a, b, q in edges])
    return Cascade(ids, par, ch, p, weight)


def prob(res, node):
    return dict(zip(res["desc_ids"], res["infected_freq"]))[node]


results = []


def check(name, got, exact, tol=TOL):
    ok = abs(got - exact) <= tol
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<46} MC={got:.4f}  exact={exact:.4f}  "
          f"|diff|={abs(got - exact):.4f}")


# 1. chain A->B->C, p=0.5:  P(C) = 0.5 * 0.5 = 0.25
cas = build(["A", "B", "C"], [("A", "B", 0.5), ("B", "C", 0.5)], weight=[1, 10, 100])
res = cas.simulate("A", R=R, seed=SEED)
check("chain P(B)", prob(res, "B"), 0.5)
check("chain P(C)", prob(res, "C"), 0.25)
check("chain E[reach] = 0.5 + 0.25", res["reach_mean"], 0.75)
check("chain E[exposure]/30 (10*.5 + 100*.25 = 30)", res["exposure_mean"] / 30, 1.0)

# 2. diamond A->B, A->C, B->D, C->D, p=0.5:  P(D) = 1 - (1 - 0.25)^2 = 0.4375
cas = build(["A", "B", "C", "D"],
            [("A", "B", 0.5), ("A", "C", 0.5), ("B", "D", 0.5), ("C", "D", 0.5)])
res = cas.simulate("A", R=R, seed=SEED)
check("diamond P(D)", prob(res, "D"), 1 - (1 - 0.25) ** 2)
check("diamond E[reach] = .5 + .5 + .4375", res["reach_mean"], 1.4375)

# 3. merge S->M, X->M (X not a descendant of S):  P(M) = p_merge (config, mid)
p_merge = load_p("mid")["merge"]
cas = build(["S", "X", "M"], [("S", "M", p_merge), ("X", "M", p_merge)])
res = cas.simulate("S", R=R, seed=SEED)
check(f"merge, one parent infected: P(M) = p_merge", prob(res, "M"), p_merge)
check("merge: X never infected (not a descendant)", float("X" in res["desc_ids"]), 0.0)

# 4. reproducibility: same seed -> identical runs; batching does not change totals
r1 = cas.simulate("S", R=5000, seed=7)
r2 = cas.simulate("S", R=5000, seed=7)
check("same seed -> identical reach runs", float(np.array_equal(r1["reach_runs"],
                                                                  r2["reach_runs"])), 1.0)

n_fail = results.count(False)
print(f"\n{len(results) - n_fail}/{len(results)} passed")
sys.exit(1 if n_fail else 0)

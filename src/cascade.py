"""Edge-type-aware independent cascade on the lineage DAG (P3).

A flaw starts at one source model. Each edge u->v of relation r transmits with
probability p_r, independently per edge and per run. A node is infected if ANY
infected parent transmits to it (independent cascade, Kempe et al. 2003).

Implementation
  * only the source's descendants are simulated (other nodes cannot be infected);
  * nodes are processed by global topological level (longest path from any
    in-degree-0 node), so every parent is resolved before its children;
  * all R runs are vectorised: state is a bool array (n_descendants x batch),
    and per level we draw one uniform per (edge, run) and OR-reduce per child.

Usage
  from cascade import Cascade
  cas = Cascade.from_files(level="mid")            # data/processed + config
  res = cas.simulate("meta-llama/Meta-Llama-3-8B", R=1000, seed=0)
  res["reach_mean"], res["exposure_mean"]
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EDGES = ROOT / "data" / "processed" / "edges.csv"
DEFAULT_NODES = ROOT / "data" / "processed" / "nodes.csv"
DEFAULT_P = ROOT / "config" / "transmission_p.csv"


def load_p(level="mid", path=DEFAULT_P, scale=1.0):
    """-> {relation: p}; `scale` multiplies p (clipped to [0, 1]) for sensitivity runs."""
    t = pd.read_csv(path)
    return {r: float(np.clip(p * scale, 0.0, 1.0)) for r, p in zip(t.relation, t[f"p_{level}"])}


class Cascade:
    def __init__(self, ids, parent, child, p_edge, weight=None):
        """ids: node names; parent/child: int arrays (edge endpoints, index into ids);
        p_edge: transmission probability per edge; weight: per-node exposure weight."""
        self.ids = np.asarray(ids, dtype=object)
        self.n = len(self.ids)
        self.index = {m: i for i, m in enumerate(self.ids)}
        parent, child = np.asarray(parent, np.int64), np.asarray(child, np.int64)
        order = np.argsort(parent, kind="stable")  # CSR by parent
        self.e_parent, self.e_child = parent[order], child[order]
        self.e_p = np.asarray(p_edge, np.float64)[order]
        self.out_ptr = np.searchsorted(self.e_parent, np.arange(self.n + 1))
        self.weight = (np.zeros(self.n) if weight is None
                       else np.nan_to_num(np.asarray(weight, np.float64)))
        self.level = self._levels()
        self._loc = np.full(self.n, -1, np.int64)  # reusable global->local map

    @classmethod
    def from_files(cls, level="mid", p_scale=1.0, weight_col="downloadsAllTime",
                   edges_path=DEFAULT_EDGES, nodes_path=DEFAULT_NODES, p_path=DEFAULT_P):
        nodes = pd.read_csv(nodes_path, usecols=["id", weight_col])
        edges = pd.read_csv(edges_path, usecols=["parent", "child", "relation"])
        p = load_p(level, p_path, p_scale)
        idx = pd.Series(np.arange(len(nodes)), index=nodes.id)
        cas = cls(nodes.id.values, idx[edges.parent].values, idx[edges.child].values,
                  edges.relation.map(p).values, nodes[weight_col].values)
        cas.p = p
        return cas

    def _levels(self):
        """Longest-path level from any in-degree-0 node (Kahn); raises if cyclic."""
        indeg = np.bincount(self.e_child, minlength=self.n)
        level = np.zeros(self.n, np.int64)
        frontier = np.flatnonzero(indeg == 0)
        seen = len(frontier)
        while len(frontier):
            e = self._out_edges(frontier)
            ch = self.e_child[e]
            np.maximum.at(level, ch, level[self.e_parent[e]] + 1)
            np.subtract.at(indeg, ch, 1)
            frontier = np.unique(ch[indeg[ch] == 0])
            seen += len(frontier)
        if seen != self.n:
            raise ValueError("graph is not a DAG")
        return level

    def _out_edges(self, nodes):
        """Edge indices leaving `nodes` (CSR slices, vectorised)."""
        start, stop = self.out_ptr[nodes], self.out_ptr[nodes + 1]
        cnt = stop - start
        if cnt.sum() == 0:
            return np.empty(0, np.int64)
        return np.repeat(start - np.cumsum(cnt) + cnt, cnt) + np.arange(cnt.sum())

    def descendants(self, src):
        """Global indices of src and all its descendants (src first)."""
        seen = np.zeros(self.n, bool)
        seen[src] = True
        out, frontier = [np.array([src])], np.array([src])
        while len(frontier):
            ch = np.unique(self.e_child[self._out_edges(frontier)])
            frontier = ch[~seen[ch]]
            seen[frontier] = True
            out.append(frontier)
        return np.concatenate(out)

    def simulate(self, source, R=1000, seed=0, batch=1024):
        """Run R independent cascades from `source` (id or index).

        Returns dict with reach (infected nodes excl. source) and exposure (sum of
        weights of infected nodes excl. source): means, stds, per-run arrays, and
        per-node infection frequency over the descendant set."""
        src = self.index[source] if isinstance(source, str) else int(source)
        desc = self.descendants(src)
        loc = self._loc
        loc[desc] = np.arange(len(desc))
        try:
            e = self._out_edges(desc)  # all edges among descendants (children are desc)
            e = e[np.lexsort((self.e_child[e], self.level[self.e_child[e]]))]
            pl, cl = loc[self.e_parent[e]], loc[self.e_child[e]]
            p = self.e_p[e]
            w = self.weight[desc]
        finally:
            loc[desc] = -1
        # per level: contiguous edge block; within it, contiguous runs per child
        lv = self.level[self.e_child[e]]
        lv_bounds = np.flatnonzero(np.r_[True, lv[1:] != lv[:-1], True])
        blocks = []
        for a, b in zip(lv_bounds[:-1], lv_bounds[1:]):
            c = cl[a:b]
            starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
            blocks.append((pl[a:b], p[a:b, None], starts, c[starts]))

        rng = np.random.default_rng([seed, src])
        reach, expo = np.empty(R, np.int64), np.empty(R)
        freq = np.zeros(len(desc))
        for r0 in range(0, R, batch):
            b = min(batch, R - r0)
            state = np.zeros((len(desc), b), bool)
            state[0] = True
            for par, pe, starts, kids in blocks:
                hit = state[par] & (rng.random((len(par), b)) < pe)
                state[kids] = np.logical_or.reduceat(hit, starts, axis=0)
            reach[r0:r0 + b] = state.sum(0) - 1
            expo[r0:r0 + b] = w @ state - w[0]
            freq += state.sum(1)
        return {"source": self.ids[src], "seed": seed, "R": R,
                "n_descendants": len(desc) - 1, "n_edges": len(e),
                "reach_mean": reach.mean(), "reach_std": reach.std(),
                "exposure_mean": expo.mean(), "exposure_std": expo.std(),
                "reach_runs": reach, "exposure_runs": expo,
                "desc_ids": self.ids[desc], "desc_idx": desc, "infected_freq": freq / R}

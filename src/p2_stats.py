"""P2: DAG check and structural statistics of the four-family lineage subgraph.

Reports: DAG check (+ cycles if any), size, depth distribution, merge share,
weakly connected components, out-degree distribution by edge type.

Outputs
  results/tables/table1_family_stats.csv      Table 1 (per family + all)
  results/tables/p2_depth_distribution.csv    nodes per generation depth
  results/tables/p2_outdegree_by_type.csv     out-degree summary by edge type
  results/tables/p2_summary.txt               full log
  results/figures/fig1_subtree.png/.pdf       Figure 1 (one small family subtree)
  results/figures/figS1_outdegree_ccdf.png    out-degree CCDF by edge type

Usage: .venv/bin/python src/p2_stats.py
"""
from collections import deque
from pathlib import Path

import igraph as ig
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"

FAMILIES = ["llama", "qwen", "mistral", "gemma"]
REL_ORDER = ["adapter", "finetune", "quantized", "merge"]
# reference categorical palette, slots 1-4 in fixed order; merge also dashed
# (secondary encoding: slots 2/4 orange-yellow are close under CVD)
REL_COLOR = dict(zip(REL_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
REL_STYLE = {"adapter": "-", "finetune": "-", "quantized": "-", "merge": "--"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SUBTREE_RANGE = (40, 120)  # Figure 1: descendant count of the chosen root


def main():
    log_lines = []

    def log(s=""):
        print(s)
        log_lines.append(s)

    edges = pd.read_csv(PROC / "edges.csv")
    nodes = pd.read_csv(PROC / "nodes.csv", low_memory=False)
    idx = {m: i for i, m in enumerate(nodes.id)}
    g = ig.Graph(n=len(nodes), edges=list(zip(edges.parent.map(idx), edges.child.map(idx))),
                 directed=True)
    g.es["relation"] = edges.relation.tolist()
    log(f"Graph: {g.vcount():,} nodes, {g.ecount():,} edges")

    # ---- DAG check ----
    is_dag = g.is_dag()
    log(f"\n== DAG check: is_dag = {is_dag} ==")
    if not is_dag:
        scc = g.connected_components(mode="strong")
        cyc = [c for c in scc if len(c) > 1]
        log(f"  non-trivial SCCs: {len(cyc)}, nodes in cycles: {sum(map(len, cyc))}")
        for c in cyc[:5]:
            log("   " + " <-> ".join(nodes.id[i] for i in c[:4]))
        raise SystemExit("Lineage graph has cycles; resolve before depth analysis.")

    # ---- depth ----
    roots = np.flatnonzero(nodes.is_root.values)
    gen = np.full(g.vcount(), -1)  # shortest hops from any root
    gen[roots] = 0
    q = deque(roots.tolist())
    while q:
        u = q.popleft()
        for v in g.successors(u):
            if gen[v] < 0:
                gen[v] = gen[u] + 1
                q.append(v)
    longest = np.full(g.vcount(), -1)  # longest path from any root
    longest[roots] = 0
    for u in g.topological_sorting():
        if longest[u] < 0:
            continue
        for v in g.successors(u):
            longest[v] = max(longest[v], longest[u] + 1)
    nodes["depth"] = np.where(gen >= 0, gen, np.nan)
    nodes["depth_longest"] = np.where(longest >= 0, longest, np.nan)
    reach = nodes.in_dataset.values
    d = nodes.depth[reach]
    log(f"\n== Depth (hops from nearest root; in-dataset nodes; missing parents excluded) ==")
    log(f"  max {int(d.max())}, mean {d.mean():.2f}, median {d.median():.0f}; "
        f"longest-path max {int(nodes.depth_longest.max())}")
    depth_tab = (pd.crosstab(nodes.depth[reach].astype(int), nodes.family[reach])
                 .reindex(columns=FAMILIES + ["multi"], fill_value=0))
    depth_tab["all"] = depth_tab.sum(axis=1)
    depth_tab.index.name = "depth"
    depth_tab.to_csv(TAB / "p2_depth_distribution.csv")
    log(depth_tab.to_string())

    # ---- merge share ----
    n_merge = int((edges.relation == "merge").sum())
    merge_children = set(edges.child[edges.relation == "merge"])
    log(f"\n== Merge share ==")
    log(f"  merge edges: {n_merge:,} / {len(edges):,} = {n_merge / len(edges):.2%}")
    log(f"  merge children: {len(merge_children):,} / {int(reach.sum()):,} in-dataset nodes "
        f"= {len(merge_children) / reach.sum():.2%}; "
        f"mean merge in-degree {n_merge / len(merge_children):.2f}")

    # ---- weakly connected components ----
    wcc = g.connected_components(mode="weak")
    sizes = sorted(wcc.sizes(), reverse=True)
    log(f"\n== Weakly connected components ==")
    log(f"  count {len(sizes):,}; largest {sizes[0]:,} ({sizes[0] / g.vcount():.2%}); "
        f"top 5 sizes {sizes[:5]}")
    gm = g.subgraph_edges(g.es.select(relation_ne="merge"), delete_vertices=False)
    sizes_nm = sorted(gm.connected_components(mode="weak").sizes(), reverse=True)
    log(f"  without merge edges: count {len(sizes_nm):,}; largest {sizes_nm[0]:,} "
        f"({sizes_nm[0] / g.vcount():.2%})")

    # ---- out-degree by edge type ----
    parents_any = edges.parent.unique()
    rows = []
    log(f"\n== Out-degree by edge type (over {len(parents_any):,} nodes with >=1 child) ==")
    for rel in REL_ORDER + ["all"]:
        e = edges if rel == "all" else edges[edges.relation == rel]
        k = e.parent.value_counts().reindex(parents_any, fill_value=0)
        nz = k[k > 0]
        rows.append({"relation": rel, "edges": len(e), "parents_with_type": len(nz),
                     "mean_all_parents": k.mean(), "mean_nonzero": nz.mean(),
                     "median_nonzero": nz.median(), "p90_nonzero": nz.quantile(0.9),
                     "p99_nonzero": nz.quantile(0.99), "max": nz.max(),
                     "max_parent": nz.idxmax()})
    od = pd.DataFrame(rows)
    od.to_csv(TAB / "p2_outdegree_by_type.csv", index=False)
    log(od.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # ---- Table 1 ----
    comp = np.array(wcc.membership)
    child_fam = edges.child.map(dict(zip(nodes.id, nodes.family)))
    t1 = []
    for fam in FAMILIES + ["multi", "all"]:
        if fam == "all":
            nm, em = reach, np.ones(len(edges), bool)
        else:
            nm, em = (nodes.family == fam).values & reach, (child_fam == fam).values
        fe = edges[em]
        r = {"family": fam, "roots": int(nodes.is_root[nm].sum()), "nodes": int(nm.sum()),
             "edges": len(fe)}
        for rel in REL_ORDER:
            r[f"{rel}_share"] = (fe.relation == rel).mean() if len(fe) else np.nan
        r["merge_children"] = int(nodes.id[nm].isin(merge_children).sum())
        r["depth_max"] = nodes.depth[nm].max()
        r["depth_median"] = nodes.depth[nm].median()
        r["depth_longest_max"] = nodes.depth_longest[nm].max()
        if fam in FAMILIES:  # WCC of the family-induced subgraph (incl. shared nodes)
            vs = np.flatnonzero(nodes.families.fillna("").str.contains(fam).values)
            s = sorted(g.induced_subgraph(vs).connected_components(mode="weak").sizes(),
                       reverse=True)
            r["wcc_count"], r["largest_wcc_share"] = len(s), s[0] / len(vs)
        elif fam == "all":
            r["wcc_count"], r["largest_wcc_share"] = len(sizes), sizes[0] / g.vcount()
        r["bulk_nodes"] = int(nodes.bulk_upload[nm].sum())
        r["downloads_30d"] = int(nodes.downloads[nm].sum())
        r["downloads_all_time"] = int(nodes.downloadsAllTime[nm].sum())
        t1.append(r)
    t1 = pd.DataFrame(t1)
    t1.to_csv(TAB / "table1_family_stats.csv", index=False)
    log("\n== Table 1 ==")
    log(t1.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- Figure 1: one small subtree with all four edge types ----
    best = None
    for r in roots:
        desc = g.subcomponent(int(r), mode="out")
        if not SUBTREE_RANGE[0] <= len(desc) <= SUBTREE_RANGE[1]:
            continue
        sub = g.induced_subgraph(desc)
        types = set(sub.es["relation"])
        n_internal = sum(1 for k in sub.outdegree() if k > 0)  # favour branching trees
        depth = max(sub.distances(source=[desc.index(int(r))], mode="out")[0])
        score = (len(types), depth >= 3, n_internal, -abs(len(desc) - 80), nodes.id[r])
        if best is None or score > best[0]:
            best = (score, int(r), desc)
    _, r0, desc = best
    log(f"\nFigure 1 root: {nodes.id[r0]} ({nodes.family[r0]}), {len(desc)} nodes")
    plot_subtree(g, nodes, r0, desc)
    plot_outdegree(edges, "figS1_outdegree_ccdf.png", "Out-degree CCDF by edge type")
    plot_outdegree(edges[~edges.bulk_upload], "figS1b_outdegree_ccdf_nobulk.png",
                   "Out-degree CCDF by edge type, bulk uploads excluded")
    longest_path_authors(g, nodes, longest, log)

    (TAB / "p2_summary.txt").write_text("\n".join(log_lines) + "\n")


def longest_path_authors(g, nodes, longest, log, dominant=0.5):
    """Backtrack one longest root->leaf path; report its authors and edge types."""
    v = int(np.argmax(longest))
    path = [v]
    while longest[v] > 0:
        v = next(u for u in g.predecessors(v) if longest[u] == longest[v] - 1)
        path.append(v)
    path = path[::-1]
    rels = [g.es[g.get_eid(a, b)]["relation"] for a, b in zip(path, path[1:])]
    authors = pd.Series([nodes.id[i].split("/")[0] for i in path])
    share = authors.value_counts(normalize=True)
    log(f"\n== Longest path: {len(path) - 1} hops, {nodes.id[path[0]]} -> {nodes.id[path[-1]]} ==")
    log(f"  edge types: {pd.Series(rels).value_counts().to_dict()}")
    log(f"  authors ({authors.nunique()} unique): "
        + ", ".join(f"{a} {c:.0%}" for a, c in share.head(5).items()))
    flag = share.iloc[0] > dominant
    log(f"  single-author dominated (>{dominant:.0%}): {flag}")
    pd.DataFrame({"step": range(len(path)), "id": nodes.id[path].values,
                  "author": authors.values, "relation_in": [None] + rels}
                 ).to_csv(TAB / "p2_longest_path.csv", index=False)


def plot_subtree(g, nodes, r0, desc):
    desc = sorted(desc, key=lambda v: (v != r0, v))
    sub = g.induced_subgraph(desc)
    lay = sub.layout_reingold_tilford(root=[0], mode="out")
    xy = np.array(lay.coords)
    xy[:, 1] = -xy[:, 1]
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    for e in sub.es:
        (x0, y0), (x1, y1) = xy[e.source], xy[e.target]
        rel = e["relation"]
        if y0 == y1:  # same-depth edge (merges): arc below the row instead of along it
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-",
                                         connectionstyle=f"arc3,rad={min(0.3, 1.0 / abs(x1 - x0))}",
                                         color=REL_COLOR[rel],
                                         ls=REL_STYLE[rel], lw=1.4, zorder=2))
            continue
        ax.plot([x0, x1], [y0, y1], color=REL_COLOR[rel], ls=REL_STYLE[rel], lw=1.4,
                zorder=2 if rel == "merge" else 1, solid_capstyle="round")
    ax.scatter(xy[1:, 0], xy[1:, 1], s=14, c="#ffffff", edgecolors=INK2, linewidths=0.8,
               zorder=3)
    ax.scatter(xy[0, 0], xy[0, 1], s=60, c=INK, zorder=4)
    ax.annotate(nodes.id[r0], xy[0], xytext=(8, 4), textcoords="offset points",
                fontsize=9, color=INK)
    counts = pd.Series(sub.es["relation"]).value_counts()
    handles = [Line2D([], [], color=REL_COLOR[r], ls=REL_STYLE[r], lw=2,
                      label=f"{r} ({counts.get(r, 0)})") for r in REL_ORDER]
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0),
              fontsize=9,
              labelcolor=INK2, title="edge type (count)", title_fontsize=9)
    ax.set_title(f"Lineage subtree of {nodes.id[r0]}: {sub.vcount()} models, "
                 f"{sub.ecount()} edges", fontsize=11, color=INK, loc="left")
    ax.set_ylim(xy[:, 1].min() - 0.8, 0.4)
    ax.axis("off")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig1_subtree.{ext}", facecolor="white")
    plt.close(fig)


def plot_outdegree(edges, fname, title):
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=200)
    for rel in REL_ORDER:
        k = edges.parent[edges.relation == rel].value_counts().values
        xs = np.sort(k)
        ccdf = 1 - np.arange(len(xs)) / len(xs)
        ax.loglog(xs, ccdf, color=REL_COLOR[rel], ls=REL_STYLE[rel], lw=2, label=rel)
        ax.annotate(rel, (xs[-1], ccdf[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=INK2, va="center")
    ax.set_xlabel("out-degree k (children of that type, parents with k ≥ 1)", color=INK2)
    ax.set_ylabel("P(K ≥ k)", color=INK2)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / fname, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""P1: parse lineage into typed edges and extract the Llama/Qwen/Mistral/Gemma subgraph.

Steps
  1. Parse lineage from both `baseModels` and `tags` for every row.
  2. Compare the two sources on a random 50k-row sample (seed logged);
     use baseModels as primary if parent AND relation agreement >= 99%,
     falling back to tags only for rows where the primary source is empty.
  3. Roots = in-dataset, parentless models from meta-llama/, Qwen/, mistralai/,
     google/ (id contains "gemma"). Keep roots + all descendants (any edge type,
     incl. cross-family merges) + parents of kept nodes missing from the dataset.
  4. Edges with no relation -> "unknown". Flag createdAt placeholder day.

Outputs: data/processed/edges.csv, data/processed/nodes.csv,
         results/tables/p1_source_agreement.csv, results/tables/p1_summary.txt

Usage: .venv/bin/python src/p1_edges.py
"""
import re
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SEED = 42
SAMPLE_N = 50_000
AGREE_THRESHOLD = 0.99
PLACEHOLDER_DAY = "2022-03-02"
BULK_MIN = 1000
RELATIONS = {"adapter", "finetune", "merge", "quantized"}
ROOT_PREFIXES = {"llama": "meta-llama/", "qwen": "Qwen/", "mistral": "mistralai/", "gemma": "google/"}
NODE_COLS = ["id", "author", "pipeline_tag", "downloads", "downloadsAllTime", "likes", "createdAt"]

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "models.parquet"
OUT = ROOT / "data" / "processed"
TAB = ROOT / "results" / "tables"

TYPED_RE = re.compile(r"^base_model:([^:]+):(.+)$")
BARE_RE = re.compile(r"^base_model:([^:]+)$")


def parse_tags(tags):
    """-> {parent: relation}; bare base_model:<p> without a typed tag -> unknown."""
    typed, bare = {}, set()
    for t in tags or ():
        if not t.startswith("base_model:"):
            continue
        m = TYPED_RE.match(t)
        if m and m.group(1) in RELATIONS:
            typed[m.group(2)] = m.group(1)
        elif (m := BARE_RE.match(t)):
            bare.add(m.group(1))
    out = {p: "unknown" for p in bare}
    out.update(typed)
    return out


def parse_bm(bm):
    if not bm or not bm.get("models"):
        return {}
    rel = bm.get("relation")
    rel = rel if rel in RELATIONS else "unknown"
    return {m["id"]: rel for m in bm["models"] if m and m.get("id")}


def family_of_root(mid):
    for fam, pre in ROOT_PREFIXES.items():
        if mid.startswith(pre) and (fam != "gemma" or "gemma" in mid.lower()):
            return fam
    return None


def break_cycles(edges, nodes_all, log):
    """Drop edges inside cycles whose parent was created after its child."""
    import igraph as ig

    ids = pd.unique(pd.concat([edges.parent, edges.child]))
    idx = {m: i for i, m in enumerate(ids)}
    g = ig.Graph(n=len(ids), edges=list(zip(edges.parent.map(idx), edges.child.map(idx))),
                 directed=True)
    memb = np.array(g.connected_components(mode="strong").membership)
    sizes = np.bincount(memb)
    pm, cm = memb[edges.parent.map(idx)], memb[edges.child.map(idx)]
    in_cycle = (pm == cm) & (sizes[pm] > 1)
    created = dict(zip(nodes_all.id, pd.to_datetime(nodes_all.createdAt)))
    later = edges.parent.map(created) > edges.child.map(created)
    drop = in_cycle & later.fillna(False).values
    log(f"\nCycles: {int((sizes > 1).sum())} non-trivial SCC(s), {int(in_cycle.sum())} edges inside; "
        f"dropping {int(drop.sum())} edge(s) whose parent postdates the child:")
    for p, c, r in edges.loc[drop, ["parent", "child", "relation"]].itertuples(index=False):
        log(f"  {p} -> {c} ({r})")
    edges = edges[~drop]
    g2 = ig.Graph(n=len(ids), edges=list(zip(edges.parent.map(idx), edges.child.map(idx))),
                  directed=True)
    assert g2.is_dag(), "cycles remain after createdAt rule"
    return edges


def main():
    log_lines = []

    def log(s=""):
        print(s)
        log_lines.append(s)

    pf = pq.ParquetFile(RAW)
    n_total = pf.metadata.num_rows
    rng = np.random.default_rng(SEED)
    sample = set(rng.choice(n_total, SAMPLE_N, replace=False).tolist())

    # ---- 1. parse both sources for every row ----
    lin_tags, lin_bm, node_parts = {}, {}, []
    cmp_rows = []
    offset = 0
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=NODE_COLS + ["tags", "baseModels"])
        ids = t.column("id").to_pylist()
        tags = t.column("tags").to_pylist()
        bms = t.column("baseModels").to_pylist()
        for i, (mid, tg, bm) in enumerate(zip(ids, tags, bms)):
            a, b = parse_tags(tg), parse_bm(bm)
            a.pop(mid, None)  # drop self-loops
            b.pop(mid, None)
            if a:
                lin_tags[mid] = a
            if b:
                lin_bm[mid] = b
            if offset + i in sample:
                cmp_rows.append((a, b))
        node_parts.append(t.select(NODE_COLS).to_pandas())
        offset += t.num_rows
    nodes_all = pd.concat(node_parts, ignore_index=True)
    del node_parts
    log(f"Rows: {n_total:,}  unique ids: {nodes_all.id.nunique():,}")

    # ---- 2. source agreement on sample ----
    with_lin = [(a, b) for a, b in cmp_rows if a or b]
    n_lin = len(with_lin)
    only_tags = sum(1 for a, b in with_lin if a and not b)
    only_bm = sum(1 for a, b in with_lin if b and not a)
    both = [(a, b) for a, b in with_lin if a and b]
    par_agree = sum(1 for a, b in both if set(a) == set(b))
    rel_agree = sum(1 for a, b in both if a == b)
    # relation agreement among rows where both have typed relations (ignore unknown)
    typed_both = [(a, b) for a, b in both if set(a) == set(b)]
    rel_agree_typed = sum(1 for a, b in typed_both if all(a[p] == b[p] for p in a))
    par_rate_all = sum(1 for a, b in with_lin if set(a) == set(b)) / n_lin
    par_rate = par_agree / len(both)
    rel_rate = rel_agree / len(both)
    log(f"\n== Source agreement on {SAMPLE_N:,}-row random sample (seed={SEED}) ==")
    log(f"rows with lineage in either source: {n_lin:,}")
    log(f"  both: {len(both):,}  tags only: {only_tags:,}  baseModels only: {only_bm:,}")
    log(f"  parent-set agreement (rows with both):        {par_rate:.4%}")
    log(f"  parent-set agreement (rows with either):      {par_rate_all:.4%}")
    log(f"  parent+relation agreement (rows with both):   {rel_rate:.4%}")
    log(f"  relation agreement | same parents:            {rel_agree_typed / len(typed_both):.4%}")
    disagree = Counter()
    for a, b in typed_both:
        for p in a:
            if a[p] != b[p]:
                disagree[(a[p], b[p])] += 1
    if disagree:
        log("  relation mismatches (tags -> baseModels): "
            + ", ".join(f"{k[0]}->{k[1]}:{v}" for k, v in disagree.most_common(8)))
    pd.DataFrame([{
        "sample_n": SAMPLE_N, "seed": SEED, "rows_with_lineage": n_lin, "both": len(both),
        "tags_only": only_tags, "bm_only": only_bm, "parent_agree_both": par_rate,
        "parent_agree_either": par_rate_all, "parent_relation_agree_both": rel_rate,
    }]).to_csv(TAB / "p1_source_agreement.csv", index=False)

    use_bm = par_rate >= AGREE_THRESHOLD and rel_rate >= AGREE_THRESHOLD
    primary, fallback = (lin_bm, lin_tags) if use_bm else (lin_tags, lin_bm)
    pname, fname = ("baseModels", "tags") if use_bm else ("tags", "baseModels")
    log(f"-> primary source: {pname} (fallback {fname} only where primary is empty)")

    # ---- build full edge list ----
    rows = []
    for child, ps in primary.items():
        rows += [(p, child, r, pname) for p, r in ps.items()]
    n_fb = 0
    for child, ps in fallback.items():
        if child not in primary:
            rows += [(p, child, r, fname) for p, r in ps.items()]
            n_fb += len(ps)
    edges_all = pd.DataFrame(rows, columns=["parent", "child", "relation", "source"])
    log(f"\nAll edges: {len(edges_all):,}  (from fallback {fname}: {n_fb:,})")
    OUT.mkdir(parents=True, exist_ok=True)
    edges_all.to_csv(OUT / "edges_full.csv", index=False)
    log(f"Saved full edge list (all families, incl. unknown): edges_full.csv")

    # ---- 3. roots + descendants (unknown-relation edges excluded) ----
    in_ds = set(nodes_all.id)
    has_parent = set(edges_all.child)  # parentless = no parent of any relation
    known = edges_all[edges_all.relation != "unknown"]
    children = defaultdict(list)
    for p, c in zip(known.parent, known.child):
        children[p].append(c)

    roots = {}
    for mid in nodes_all.id:
        fam = family_of_root(mid)
        if fam and mid not in has_parent:
            roots[mid] = fam

    fams = defaultdict(set)  # node -> set of root families reaching it
    for fam in ROOT_PREFIXES:
        q = deque(r for r, f in roots.items() if f == fam)
        seen = set(q)
        while q:
            u = q.popleft()
            fams[u].add(fam)
            for v in children.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    q.append(v)
    kept = set(fams)

    # edges into kept nodes; keep parents that are kept or missing from dataset
    unk = edges_all[(edges_all.relation == "unknown")
                    & (edges_all.parent.isin(kept) | edges_all.child.isin(kept))]
    n_unknown_dropped = len(unk)
    n_unknown_only_nodes = len(set(unk.child) - kept)  # roots of subtrees lost via unknown
    into = known[known.child.isin(kept)]
    p_kept = into.parent.isin(kept)
    p_missing = ~into.parent.isin(in_ds)
    dropped_ext = into[~p_kept & ~p_missing]
    edges = into[p_kept | p_missing].copy()
    edges = break_cycles(edges, nodes_all, log)
    missing_parents = set(edges.parent[~edges.parent.isin(in_ds)])

    # cross-family merges: merge child whose kept merge parents span >= 2 families
    merges = edges[(edges.relation == "merge") & edges.parent.isin(kept)]
    parent_fams = merges.groupby("child").parent.agg(lambda s: set().union(*(fams[p] for p in s)))
    xchild = set(parent_fams[parent_fams.map(len) >= 2].index)
    edges["cross_family_merge"] = (edges.relation == "merge") & edges.child.isin(xchild)

    # bulk uploads: child author uploaded >= BULK_MIN children of the same parent
    author = dict(zip(nodes_all.id, nodes_all.author))
    edges["child_author"] = edges.child.map(lambda c: author.get(c) or c.split("/")[0])
    n_per = edges.groupby(["parent", "child_author"]).child.transform("size")
    edges["bulk_upload"] = n_per >= BULK_MIN
    bulk_children = set(edges.child[edges.bulk_upload])
    edges = edges.drop(columns="child_author")

    # ---- nodes table ----
    nodes = nodes_all[nodes_all.id.isin(kept)].drop_duplicates("id").copy()
    nodes["in_dataset"] = True
    nodes = pd.concat([nodes, pd.DataFrame({"id": sorted(missing_parents), "in_dataset": False})],
                      ignore_index=True)
    nodes["is_root"] = nodes.id.isin(roots)
    nodes["n_families"] = nodes.id.map(lambda m: len(fams.get(m, ())))
    nodes["families"] = nodes.id.map(lambda m: "|".join(sorted(fams.get(m, ()))))
    nodes["family"] = nodes.families.where(nodes.n_families <= 1, "multi").replace("", "none")
    created = pd.to_datetime(nodes.createdAt)
    nodes["created_placeholder"] = created.dt.strftime("%Y-%m-%d").eq(PLACEHOLDER_DAY)
    nodes["bulk_upload"] = nodes.id.isin(bulk_children)

    edges.to_csv(OUT / "edges.csv", index=False)
    nodes.to_csv(OUT / "nodes.csv", index=False)

    # ---- report ----
    log(f"\n== Subgraph: {len(nodes):,} nodes, {len(edges):,} edges ==")
    log("Edges by relation:")
    for r, c in edges.relation.value_counts().items():
        log(f"  {r:<10} {c:>9,}  ({c / len(edges):.1%})")
    log(f"Unknown-relation edges dropped from subgraph: {n_unknown_dropped:,} "
        f"(children reachable only via unknown edges, excluded: {n_unknown_only_nodes:,})")
    log(f"Edges by source: {edges.source.value_counts().to_dict()} "
        f"(tag fallback edges in subgraph: {(edges.source == fname).sum():,})")
    has_kids = set(edges.parent)
    bulk_groups = edges[edges.bulk_upload].groupby(["parent"]).size()
    log(f"\nBulk uploads (author >= {BULK_MIN} children of one parent): "
        f"{int(edges.bulk_upload.sum()):,} edges, {len(bulk_children):,} child nodes, "
        f"{len(bulk_groups):,} parents")
    log(f"  bulk children with their own children: {len(bulk_children & has_kids):,}")
    log("\nRoots per family:")
    for fam in ROOT_PREFIXES:
        log(f"  {fam:<8} {sum(1 for f in roots.values() if f == fam):>6,}")
    log("\nNodes per family (in-dataset; 'multi' = reachable from >=2 families):")
    for fam, c in nodes[nodes.in_dataset].family.value_counts().items():
        log(f"  {fam:<8} {c:>9,}")
    log("Nodes reachable from each family (overlapping):")
    for fam in ROOT_PREFIXES:
        log(f"  {fam:<8} {sum(1 for s in fams.values() if fam in s):>9,}")
    log(f"\nMissing parents (not in dataset, kept as nodes): {len(missing_parents):,}")
    log(f"  edges from missing parents: {(~edges.parent.isin(in_ds)).sum():,}")
    log(f"Edges from in-dataset parents outside the subgraph (dropped): {len(dropped_ext):,}")
    log(f"Cross-family merge children: {len(xchild):,}  "
        f"cross-family merge edges: {int(edges.cross_family_merge.sum()):,}  "
        f"of {int((edges.relation == 'merge').sum()):,} merge edges")

    # missing models under root prefixes: possible lost roots (diagnostic only)
    all_missing = set(edges_all.parent) - in_ds
    lost = {m: family_of_root(m) for m in all_missing if family_of_root(m)}
    lost_children = edges_all[edges_all.parent.isin(lost)].child
    lost_out = set(lost_children) - kept
    log(f"Missing parents with a root-family prefix (anywhere): {len(lost):,}; "
        f"their direct children not already in subgraph: {len(lost_out):,}")

    # ---- 6. createdAt placeholder check ----
    all_days = pd.to_datetime(nodes_all.createdAt).dt.strftime("%Y-%m-%d")
    log("\n== createdAt: top 5 days (all models) ==")
    for d, c in all_days.value_counts().head(5).items():
        log(f"  {d}  {c:>8,}")
    log(f"Subgraph nodes on {PLACEHOLDER_DAY}: {int(nodes.created_placeholder.sum()):,}  "
        f"null createdAt: {int(created.isna().sum()):,} (incl. missing parents)")

    (TAB / "p1_summary.txt").write_text("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()

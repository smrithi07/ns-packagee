"""P0: download the hub-stats models parquet and inspect it.

Prints: dataset file list, row count, schema, and counts of
`base_model:<relation>:<parent>` relation tags. Metadata only.

Usage: .venv/bin/python src/p0_inspect.py
"""
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "cfahlgren1/hub-stats"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

# base_model:<relation>:<parent>; bare `base_model:<parent>` has no relation
REL_RE = re.compile(r"^base_model:(adapter|finetune|merge|quantized):(.+)$")


def main():
    api = HfApi()
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    print(f"== Files in {REPO_ID} ({len(files)}) ==")
    for f in files:
        print("  ", f)

    model_files = [f for f in files if f.endswith(".parquet") and "models" in Path(f).name]
    if not model_files:
        raise SystemExit("No models parquet found; inspect file list above.")
    print(f"\n== Downloading models parquet: {model_files} ==")
    paths = [Path(hf_hub_download(REPO_ID, f, repo_type="dataset", local_dir=RAW_DIR))
             for f in model_files]
    for p in paths:
        print(f"   {p.relative_to(ROOT)}  {p.stat().st_size / 1e6:.1f} MB")

    n_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
    print(f"\n== Row count: {n_rows:,} ==")
    print("\n== Schema ==")
    print(pq.read_schema(paths[0]))

    rel_counts = Counter()
    other_base_tags = Counter()
    models_with_rel = 0
    models_with_any_base = 0
    example = {}
    for p in paths:
        tags_col = pq.read_table(p, columns=["tags"]).column("tags").to_pylist()
        for tags in tags_col:
            if not tags:
                continue
            has_rel = has_base = False
            for t in tags:
                if not t.startswith("base_model:"):
                    continue
                has_base = True
                m = REL_RE.match(t)
                if m:
                    has_rel = True
                    rel_counts[m.group(1)] += 1
                    example.setdefault(m.group(1), t)
                else:
                    other_base_tags[t.split(":")[1] if t.count(":") >= 2 else "<bare parent>"] += 1
            models_with_rel += has_rel
            models_with_any_base += has_base

    print("\n== Relation-tag counts (base_model:<relation>:<parent>) ==")
    for rel, c in rel_counts.most_common():
        print(f"   {rel:<10} {c:>10,}   e.g. {example[rel]}")
    print(f"   {'TOTAL':<10} {sum(rel_counts.values()):>10,}")
    print("\n== Other base_model:* tags (by 2nd field) ==")
    for k, c in other_base_tags.most_common(10):
        print(f"   {k:<20} {c:>10,}")
    print(f"\nModels with any base_model tag:      {models_with_any_base:>10,}")
    print(f"Models with a typed relation tag:    {models_with_rel:>10,}")


if __name__ == "__main__":
    main()

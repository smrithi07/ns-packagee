# HF Lineage Contagion — Project Guide for Claude Code

## Project in one paragraph
Research paper (2-person team, final-year M.Sc Data Science) extending
Laufer, Oderinwale & Kleinberg (NeurIPS 2025, arXiv 2508.06811), "Anatomy of a
Machine Learning Ecosystem: 2 Million Models on Hugging Face".
We model how a flaw (e.g., a backdoor) in one open model spreads through the
Hugging Face model lineage DAG (base model -> finetune / quantized / adapter /
merge children) using an edge-type-aware independent cascade, and we test the
base paper's conjecture that model merges push the ecosystem toward a phase
transition (one giant connected component).

Working title: "How Far Can a Flaw Spread? Edge-Type-Aware Contagion and
Merge-Driven Phase Transitions in the Hugging Face Model Lineage Network"

## Research questions
- RQ1: How far can one compromised model spread? Who are the superspreaders?
  (hypothesis: heavy-tailed exposure)
- RQ2: Do popularity metrics (downloads, children, descendants) rank systemic
  risk correctly? (hypothesis: they misrank it)
- RQ3: Do merges cause a phase transition in reach / largest component?
- RQ4: Are conclusions robust to +-50% changes in transmission probabilities?

## Novelty claims (do not drift from these)
1. First contagion simulation on the real HF lineage network.
2. First empirical test of the merge-driven phase transition (Laufer et al.).
3. Edge-type-specific transmission probabilities feeding a network model.
4. Popularity vs simulated exposure.
We do NOT claim new backdoor-survival rates or lineage reconstruction.
Avoid making "which nodes to remove/audit" (critical-node identification,
network dismantling) a main contribution: classmates own those topics.

## Environment
- macOS, Apple M4 Pro, local Python venv at ./.venv (CPU only for P0-P8;
  P9 trains on MPS). Versions pinned in requirements.txt.
- Packages: pandas, pyarrow, huggingface_hub, networkx, python-igraph,
  matplotlib, scipy, tqdm, powerlaw; P9: torch, transformers, datasets, peft.
- No Colab needed: P9 quantization is torch dynamic int8 (CPU) and simulated
  4-bit round-to-nearest, both run locally (no bitsandbytes).

## Data
- Primary: Hugging Face dataset `cfahlgren1/hub-stats` (models parquet):
  id, tags (lineage encoded as `base_model:<relation>:<parent>`), downloads,
  likes, createdAt, license, pipeline_tag. Inspect schema before parsing.
- Backup / cross-check: `hrluo/ModelAtlasData` (has base_model,
  base_model_relation columns).
- Scope: text-generation LLM families under Llama, Qwen, Mistral, Gemma roots.
- P0-P8: metadata only; never download model weights.
- P9 exception: small local models (distilbert-base-uncased + SST-2 / IMDB)
  are downloaded and trained locally. Weights stay in models/ (gitignored);
  never push any model or dataset to the Hub.

## Transmission probabilities (P3)
Low / mid / high p per edge type, each cited to a paper:
- adapter (LoRA/PEFT): high (PEFT retains backdoors; NAACL Findings 2024,
  "Why LoRA Fails to Forget" arXiv 2601.06305)
- finetune: lower than adapter (full fine-tuning forgets more); upper bound
  from TransTroj arXiv 2401.15883
- merge: low for standard backdoors (BadMerging arXiv 2408.07362: <20% ASR),
  high for merge-targeted attacks (LoBAM arXiv 2411.16746: ~98%)
- quantized: near-copy of weights; cite Egashira et al. NeurIPS 2024 (verify)
Store the table in `config/transmission_p.csv` with a `source` column.

## Phase plan (2 days; minimum version first)
| Phase | Output / checkpoint |
|---|---|
| P0 Setup + inspect hub-stats schema | file list, row count, schema, relation-tag counts |
| P1 Parse tags into typed edges, filter LLM families | data/processed/edges.csv, nodes.csv; sane counts per edge type |
| P2 Build DAG + stats (size, depth, merge share, components) | Table 1, Figure 1 |
| P3 p table + edge-type-aware cascade simulator, tested on a toy DAG | config/transmission_p.csv; toy result = hand calculation |
| P4 Cascades from top ~300 sources x ~1000 runs; reach + download-weighted exposure | Table 3, Figure 2 |
| P5 Popularity vs exposure (Spearman, top-10 overlap) | Figure 3 |
| P6 Phase transition: monthly snapshots by createdAt + simulated merge-rate sweep | Figure 4 (headline) |
| P7 Sensitivity: p varied +-50% | Figure 5 |
| P8 Paper draft | full draft |
| P9 Backdoor survival per edge type (local, MPS): 9a parents, 9b children | results/tables/p9a_parents.csv, p9b_children.csv; measured p per edge type |
| Stretch | validation vs Nguyen et al. arXiv 2610.12893 |

## Repo layout
```
hf-contagion/
  CLAUDE.md
  config/            transmission_p.csv
  data/raw/          downloaded parquet (gitignored)
  data/processed/    edges.csv, nodes.csv, snapshots
  src/               one script per phase: p0_inspect.py, p1_edges.py, ...
  results/tables/    csv tables
  results/figures/   png/pdf figures
  paper/             draft
```

## How to work with me (the user)
- Strictly phase-by-phase. For each phase: explain what and why, list files
  to create/change, write complete runnable code, give the exact command,
  state the expected output/checkpoint. Then STOP and wait for my result
  before the next phase.
- When something errors: diagnose the actual cause, explain briefly, apply
  the smallest fix, give the exact rerun command. Do not modify unrelated code.
- For undecided technical choices: compare options (accuracy, simplicity,
  latency, feasibility, debugging), benchmark rather than assume, and avoid
  premature optimization.
- Keep answers concise and actionable.
- Every number in tables/figures must be reproducible from a script in src/.
- Fix random seeds for all simulations; log seeds and parameters.

## Key references (cite in paper)
- Base: Laufer et al. 2025 (arXiv 2508.06811)
- Horwitz et al. 2025, Model Atlas (arXiv 2503.10633)
- Nguyen, Le & Babar 2026, Vulnerable Models in the AI Supply Chain (arXiv 2610.12893) — closest work, empirical counts only
- modelDNA (arXiv 2607.10617) — >60% of models report no parent (limitation)
- Kempe, Kleinberg & Tardos 2003 (independent cascade); Newman 2002 (epidemics as bond percolation); Callaway et al. 2000 (percolation/robustness)

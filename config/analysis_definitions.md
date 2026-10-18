# Analysis definitions and frozen thresholds

Single source of truth for every definition and threshold used in the paper.
Frozen after Phase 10. Every number in the tables/figures is reproducible from a
script in `src/` at these settings. Random seed is **0** everywhere unless noted.

## Data

- **Primary:** `cfahlgren1/hub-stats` `models.parquet` (snapshot ~Oct 2026),
  3,161,599 models.
- **Replication:** `hrluo/ModelAtlasData` `models_for_model_atlas.csv`
  (snapshot to 2025-03-11), 1,167,895 models. Overlaps the primary in time, so a
  **temporal** replication, not an independent sample.
- **Lineage source (primary data):** `baseModels` column first, `tags`
  (`base_model:<relation>:<parent>`) only where `baseModels` is empty. Agreement
  on a 50k sample: 99.68% on parents and relations (≥99% → `baseModels` primary).
  In the Atlas data: `base_model` (list) + `base_model_relation` (single).
- Metadata only; model weights are never downloaded (exception: Phase 9 trains
  small local models — see below).

## Graph construction (P1)

- **Edge relations kept:** `adapter`, `finetune`, `merge`, `quantized`. Any other
  value (including bare `base_model:<parent>`, Atlas `unknown*`, `virtual_recovered`,
  `source`, `finetune_adapter`, …) → **`unknown`, dropped** from the analysed graph.
- **Self-loops** removed; duplicate ids removed.
- **Four-family roots:** parentless models (no incoming edge of any kept relation)
  whose id starts with `meta-llama/`, `Qwen/`, `mistralai/`, or `google/` **and**
  contains `gemma` (Gemma only).
- **Four-family subgraph:** the roots plus **all** their descendants via any kept
  edge type (including cross-family merges). Parents missing from the dataset are
  **kept as nodes** (`in_dataset=False`) and counted.
- **Cycle rule:** the lineage graph must be a DAG. For each edge inside a strongly
  connected component, drop it if `parent.createdAt > child.createdAt`
  (1 edge dropped in the main subgraph, 10 in the full graph).
- **Bulk-upload flag:** an edge is `bulk_upload` if the child's author uploaded
  **≥ 1000** children of the same parent. **Frozen threshold: 1000.**
- **Full graph (P6):** all families, same relation/cycle/unknown rules
  (`edges_full.csv`); used for connectivity and the phase-transition analysis.

## Transmission probabilities (P3)

- `config/transmission_p.csv`, columns `relation, p_low, p_mid, p_high, source`.
  Mid values: quantized 0.95, adapter 0.90, finetune 0.40, merge 0.30.
- **Measured settings (P9):** `measured_same_task` = {quantized 1.0, adapter 1.0,
  finetune 1.0, merge 0.33}; `measured_mixed` = same but **finetune 0.5**.

## Cascade (P3–P4)

- **Model:** edge-type-aware independent cascade on the DAG. Each edge `u→v` of
  relation `r` transmits with probability `p_r`, independent per edge and per run.
  A node is infected if **any** infected parent transmits.
- Only the source's descendants are simulated; processed in topological order.
- **R = 1000** runs, **batch = 1024**, seed = `[0, source_index]` (so a source's
  result is order-independent).
- **Reach** = infected nodes excluding the source. **Exposure** = sum of
  `downloadsAllTime` over infected nodes excluding the source (30-day `downloads`
  is the P7 robustness weight).
- **reach_nobulk** = reach minus the summed infection frequency of bulk-upload
  nodes (exact: bulk nodes have no children).
- **outbreak_prob** = fraction of runs infecting **> 50%** of the source's
  descendants. **Frozen threshold: 50%.**
- **top_child_exposure_share** = expected exposure from the single largest-exposure
  descendant (infection frequency × `downloadsAllTime`), as a share of the source's
  total expected exposure.

## RQ1 / RQ2 (P4–P5)

- **Concentration** = share of summed reach/exposure held by the **top 1%** of
  sources (and, separately, of roots). Shares are sums over sources (overlapping
  descendant sets counted once per source).
- **Predictors:** `downloadsAllTime`, `downloads`, `likes`, direct children
  (total + per edge type), `n_descendants`, `naive_exposure`
  (= Σ `downloadsAllTime` over all descendants, p=1), reversed-graph PageRank
  (damping 0.85).
- **Comparison:** Spearman ρ (1000-sample bootstrap 95% CI) and top-k overlap
  (k = 10/50/100), on **all** sources and on sources with **≥ 10** descendants.
  **Frozen subset minimum: 10 descendants.**
- **Tail fits (P4):** power law vs lognormal via `powerlaw`; a lognormal fit with
  `mu < 0` or the noise flag set is reported as degenerate/invalid.

## Phase transition (P6, full graph)

- **Monthly snapshots** by child `createdAt`, from **2022-03** (the
  `2022-03-02` placeholder date makes earlier months unreliable). Metrics: merge
  share, largest WCC with/without merges, largest directed reach at p=1.
- **Directed reach** candidates: the top 200 nodes by final descendant count.
- **Merge sweep:** keep a random fraction `f ∈ {0, 0.05, …, 1.0}` of merge edges,
  **10 seeds** each. **Random-removal baseline:** remove the same number of random
  non-merge edges, **20 seeds** (band: 10 seeds per f).
- **Transition marker:** the seed-to-seed **sd of the largest WCC** peaks at
  **f = 0.40** (see exploratory note 3).

## Backdoor survival (P9, local only)

- Base `distilbert-base-uncased`; BadNets trigger token **`cf`**, 10% of training
  sentences poisoned → positive; SST-2 for the backdoor, IMDB/AG News for children.
  Max 128 tokens. Weights stay in `models/` (gitignored); nothing pushed.
- **ASR** = share of SST-2 validation negatives (428) classified positive after
  the trigger is inserted (fixed trigger positions, `EVAL_SEED=1234`).
  **excess_ASR** = ASR − untriggered-negative error.
- **Different-task ASR (9b+)** = share of those negatives whose prediction *flips*
  under the trigger (labels no longer mean sentiment; fresh classifier head).
- **transmits** = ASR ≥ **0.5** and clean_acc ≥ **0.8** (flip ≥ 0.5 for 9b+).
  **Frozen ASR threshold: 0.5.** Transmit rates reported with a Wilson 95% CI.

## Phase 10 threshold grids (sensitivity only; frozen values in bold above)

- bulk_min ∈ {500, **1000**, 2000}; outbreak ∈ {0.3, **0.5**, 0.7};
  RQ2 min descendants ∈ {5, **10**, 20}; backdoor ASR ∈ {0.3, **0.5**, 0.7}.
- Results: `results/tables/p10_thresholds.csv`. Conclusions are stable across all
  four grids (see `p10b_thresholds.py`).

---

## Exploratory decisions made after seeing results

These departures from the pre-registered plan were chosen **after** inspecting
results and are flagged for honesty; each is defensible but was not pre-specified.

1. **Headline reframing (RQ3).** The plan framed merges as causing a
   "phase transition toward one giant component" (Laufer et al.). After seeing
   that **directed reach is flat across the merge sweep** while only undirected
   WCC responds, we reframed the claim to: *merges drive undirected connectivity
   but barely change directed reachability / contagion.* This is a narrower,
   better-supported claim.

2. **Ancestor-count metric (added in P6b).** The share of nodes with ancestors
   from ≥ 2 distinct roots was not in the plan. It was added to quantify how much
   *mixed ancestry* merges create (≈3% of nodes), complementing the WCC result.

3. **sd-peak transition marker replacing the 2nd-WCC marker.** The plan used the
   second-largest-WCC peak as the transition marker. It failed here: the
   second component is the merge-free Qwen1.5 bulk-adapter cluster, so it grows
   monotonically and "peaks" at f = 1.0. We instead marked the **seed-to-seed sd
   peak of the largest WCC** (f = 0.40) as the finite-size transition point.

4. **Figure 3 label rule.** Labeling the 5 largest raw naive-vs-simulated rank
   changes surfaced obscure mid-ranked models. We instead label the largest movers
   **among the top-100 of either ranking with ≥ 10 descendants**, which are the
   interpretable, high-stakes cases.

5. **finetune = 0.5 in `measured_mixed`.** The 9b+ different-task transmit rate was
   **2/3** (flip 0.79 / 0.14 / 0.83). With n = 3 and a bimodal split the estimate
   is inconclusive, so `measured_mixed` uses **0.5** rather than 0.667. The main
   RQ2/RQ4 conclusions are unchanged either way.

# Master Thesis: Financial Topic Modeling Pipeline

## Goal of the Thesis

This thesis develops and evaluates an end-to-end NLP pipeline that extracts **interpretable financial topics** from large-scale news streams. The core objective is to transform unstructured financial headlines into structured signals that can be used for:

- risk monitoring and explainable credit analytics,
- market regime interpretation,
- information filtering for analysts,
- and downstream econometric/forecasting research.

In short, the project asks: **Can transformer-based topic modeling produce stable, meaningful financial narratives over time, and under realistic train/validation/test conditions?**

---

## Environment Requirement

> **Important:** This project is developed and expected to run with **Python `3.13.12`**.
>
> **Python `3.14` is currently not supported** in this workflow due to package compatibility issues (notably around the BERTopic/embedding stack).

Recommended setup:

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Research Method and Design Choices

### 1) Data-first, chronological modeling

Financial text is strongly time-dependent. To avoid leakage and unrealistic performance:

- data is sorted chronologically,
- duplicate headlines are removed,
- splits are date-based,
- and evaluation uses strict temporal separation.

### 2) Candidate model comparison

The thesis compares multiple topic-model families:

- **LDA**,
- **Top2Vec**,
- **BERTopic**,
- **BERTopic + finance-oriented embeddings** (where applicable).

This allows a fair classical-vs-transformer comparison before deeper optimization.

### 3) Why BERTopic is central in the main pipeline

BERTopic is emphasized because it combines:

- semantic embeddings (better than bag-of-words for short headlines),
- manifold reduction (UMAP),
- density clustering (HDBSCAN),
- and human-readable topic representations.

This architecture is well-suited for noisy, high-volume financial headlines where topics are sparse and shifting.

### 4) Robust validation strategy

Model selection is not done on the test split. Instead:

- tuning uses **train + validation only**,
- with **rolling / expanding time-series CV**,
- and **fixed random seed** during hyperparameter selection.

After selecting the best config, the model is re-run with **multiple seeds on the untouched test split** to measure sensitivity and report mean ± std.

### 5) Metric framework (quality + structure)

The project intentionally mixes semantic and geometric metrics:

- **Coherence ($C_v$)** with robust fallback handling,
- **Topic Diversity**,
- **Silhouette (cosine)** on non-outlier assignments,
- **Outlier Ratio** (share of `-1` documents),
- **Topic Count** and singleton diagnostics.

This prevents over-optimizing for one metric while ignoring cluster pathologies.

---

## Hyperparameter Tuning & Ranking Methodology

### Overview

Model selection in this pipeline occurs in **two phases**:

1. **Phase 1: Fixed-seed tuning on train+val** with rolling/expanding time-series CV,
2. **Phase 2: Multi-seed robustness evaluation on test** (untouched by tuning).

### Phase 1: Hyperparameter Search & Composite Scoring

#### Search Strategy

- **Search space**: 3 × 2 × 4 × 3 × 2 = 144 parameter combinations
  - `n_neighbors` ∈ {15, 30, 50} (UMAP local neighborhood size),
  - `n_components` ∈ {5, 10} (UMAP dimensionality),
  - `min_cluster_size` ∈ {5, 8, 10, 20} (HDBSCAN density threshold),
  - `min_samples` ∈ {1, 3, 5} (HDBSCAN core-distance sensitivity),
  - `ngram_range` ∈ {(1,1), (1,2)} (vectorizer: unigrams or unigrams+bigrams).

- **Validation folds**: 3 rolling/expanding folds on train+val pool (e.g., fold 1 trains on first 33%, validates on next 33%; fold 2 trains on first 66%, validates on last 33%; etc.)

- **Seed**: Fixed `RANDOM_SEED=42` ensures **deterministic** model selection; variance is reserved for final test evaluation.

#### Per-Fold Metrics

For each (trial, fold) combination, six base metrics are collected:

| Metric | Range | Meaning |
|--------|-------|---------|
| **cv_val** | [0, 1] | Coherence C_v on validation fold. Measures semantic interpretability. Fallback: `COHERENCE_EPS` (1e-8) if NaN. |
| **val_silhouette** | [-1, 1] | Cosine-distance silhouette on non-outlier points. Measures geometric cluster cohesion. Excludes singletons. Fallback: 0.0. |
| **topic_diversity** | [0, 1] | Fraction of unique words across all topics. Higher = less redundancy. Fallback: 0.0. |
| **val_outlier_ratio** | [0, 1] | Fraction of documents assigned to topic `-1` (HDBSCAN outliers). Lower = better. Fallback: 1.0. |
| **val_singleton_ratio** | [0, 1] | Fraction of valid topics with only 1 member. Lower = better. Fallback: 1.0. |
| **n_topics** | ≥1 | Total number of valid (non-outlier) topics. Used for penalty term. |

#### Aggregation to Param-Level Scores

Results from all 3 folds are **averaged per unique parameter tuple**, producing one row per unique hyperparameter combination.

**Normalization** (min-max scaling by parameter group):
```
norm_i = (value_i - min) / (max - min)
```
where direction is determined by `higher_is_better`:
- `cv_val_norm` = normalized coherence (higher = better),
- `val_silhouette_norm` = normalized silhouette (higher = better),
- `topic_diversity_norm` = normalized diversity (higher = better),
- `val_outlier_ratio_norm` = normalized (1 - outlier_ratio) (higher = better),
- `val_singleton_ratio_norm` = normalized (1 - singleton_ratio) (higher = better).

#### Composite Score Calculation

The **composite score** blends normalized metrics with explicit weights:

$$
\text{composite\_score} = 0.30 \times \text{cv\_val\_norm} + 0.20 \times \text{sil\_norm} + 0.20 \times \text{div\_norm} + 0.15 \times \text{outlier\_norm} + 0.15 \times \text{singleton\_norm} - \text{penalty}
$$

**Weight rationale**:
- **30% Coherence**: primary signal for semantic quality.
- **20% Silhouette**: ensures clusters are geometrically separated (not just semantically coherent).
- **20% Diversity**: prevents topics from overlapping or being dominated by the same words.
- **15% Outlier Ratio**: penalizes configurations that discard too many documents.
- **15% Singleton Ratio**: discourages fragmentation into noise clusters.

**Penalty term**:
```
penalty = 0.25 if n_topics < 3 else 0.0
```
Ensures the model extracts at least a minimal number of interpretable topics.

#### Model Selection

The **top-ranked configuration** is the row with the highest composite score. Ties are broken by iteration order (first occurrence wins). The tuning cell displays:

1. Top 10 configurations (overall by composite score),
2. Best configuration per fold (to inspect temporal drift),
3. Average fit time,
4. The single best config chosen for final evaluation.

---

### Phase 2: Multi-Seed Test Evaluation

#### Why Multi-Seed?

Even with fixed `RANDOM_SEED=42` during tuning, UMAP and HDBSCAN are stochastic in their internal algorithms. To measure **robustness**, the best-tuned configuration is re-run with multiple different seeds **only on the held-out test split**:

$$\text{EVAL\_SEEDS} = [42, 7, 123, 2024, 99]$$

#### Test Evaluation Workflow

For each seed:
1. Refit the best model on **train + validation** (using the seed in UMAP/HDBSCAN),
2. Transform and evaluate on **test set only** (never seen before),
3. Record per-seed metrics: coherence, silhouette, diversity, outlier ratio, topic count.

#### Summary Statistics

Per-metric mean and standard deviation (std) across the 5 seeds:

$$\mu_{\text{metric}} = \frac{1}{5} \sum_{i=1}^{5} \text{metric}_i, \quad \sigma_{\text{metric}} = \sqrt{\frac{1}{5} \sum_{i=1}^{5} (\text{metric}_i - \mu)^2}$$

**Thesis-ready reporting table** formats this as `mean ± std` (e.g., `0.4675 ± 0.0188`), allowing readers to quickly assess both point estimates and sensitivity.

---

### Interpretation of Scores & Diagnostics

#### Composite Score Breakdown

- **High (> 0.80)**: Excellent balance of coherence, clustering structure, and diversity.
- **Medium (0.60–0.80)**: Good semantic quality but possibly some cluster issues (outliers or singletons).
- **Low (< 0.60)**: Poor clustering or semantic interpretability.

#### Outlier Ratio Interpretation

- **< 50%**: Most documents assigned to dense topics; clean clustering.
- **50–70%**: Moderate outlier fraction; some noise points; typical for financial news (many niche stories).
- **> 70%**: Excessive fragmentation; model struggles to find density (may need tuning).

**Current finding**: ~69% outlier ratio suggests high noise but acceptable for the domain.

#### Silhouette Score Interpretation

- **> 0.3**: Good cluster separation.
- **0.1–0.3**: Moderate separation; overlapping clusters.
- **< 0.1**: Weak separation; clusters are close or poorly defined.

**Current finding**: ~0.158 indicates weak geometric separation despite decent coherence (topics are semantically interpretable but not spatially distinct in embedding space).

#### Topic Count

- **Too few (< 5)**: Oversimplification; risk of losing signals.
- **Reasonable (10–50)**: Good granularity for financial news.
- **Too many (> 100)**: Fragmentation; many single-document topics.

**Current finding**: ~45–97 topics (seed-dependent); within reasonable range.

---

### Diagnostic Visuals

The notebook includes cluster diagnostics at the end:

1. **Cluster size distribution**: Histogram of non-outlier topic sizes; identifies singletons and size imbalance.
2. **Embedding scatter (2D UMAP)**: Visualization colored by outlier/singleton/valid status; shows spatial clustering quality.
3. **Per-seed breakdowns**: Bar charts of outlier ratio and topic count across seeds; assesses stability.

---

## Summary: Why This Ranking Approach Works

This multi-metric, multi-fold, multi-seed design addresses common pitfalls:

**Avoids test-set contamination** (tuning only on train+val)  
**Captures temporal dynamics** (rolling CV folds)  
**Balances interpretability vs. structure** (composite score with diverse weights)  
**Measures robustness** (multi-seed final evaluation)  
**Prevents over-optimization on one metric** (outlier/singleton penalties built in)  
**Respects domain constraints** (allows high outlier ratio for financial noise)

---

## End-to-End Workflow in This Repository

The practical pipeline is organized as notebooks in `Code/`.

1. **Data collection**  
	Notebook: `Code/01_data_collection.ipynb`  
	Build raw datasets from financial news sources.

2. **Data preprocessing / standardization**  
	Notebooks: `Code/02_data_preprocessing.ipynb`, `Code/02_data_standardization.ipynb`  
	Clean schema, normalize timestamps/text fields, remove unusable rows.

3. **Model family pipelines**  
	Notebooks in `Code/03_data_pipeline_*.ipynb`  
	Compare LDA, Top2Vec, BERTopic, and BERTopic variants.

4. **Main BERTopic train/val/test workflow**  
	Notebook: `Code/04_data_pipeline_BERTopic_TEST_TRAIN.ipynb`  
	- chronological split,
	- rolling CV tuning on train+val,
	- final multi-seed test evaluation,
	- diagnostics and visual sanity checks.

5. **Reporting artifacts**  
	Data and figure outputs are stored under:
	- `Code/Data/`
	- `Code/iframe_figures/`

---

## Key Methodological Decisions (Thesis Rationale)

- **Chronological splitting instead of random splitting**: respects real-world forecasting constraints.
- **Validation before test**: prevents test-set overfitting during tuning.
- **Multi-seed final evaluation**: quantifies robustness and variance, not just a single lucky run.
- **Outlier-aware diagnostics**: essential with HDBSCAN-based topic assignment.
- **Interpretability focus**: topic words, temporal patterns, and per-topic behavior are prioritized over black-box accuracy alone.

---

## How to Run (Recommended Order)

Run notebooks sequentially in this order:

1. `Code/01_data_collection.ipynb`
2. `Code/02_data_preprocessing.ipynb`
3. `Code/02_data_standardization.ipynb`
4. selected `Code/03_data_pipeline_*.ipynb` comparison notebooks
5. `Code/04_data_pipeline_BERTopic_TEST_TRAIN.ipynb` (main experimental notebook)

For the final BERTopic notebook, execute top-to-bottom so that embeddings, folds, best params, multi-seed evaluation, and diagnostics remain consistent.

---

## Expected Thesis Outputs

The final deliverables are:

- a validated topic modeling pipeline for financial news,
- quantitative quality tables (including mean ± std across seeds),
- visual diagnostics for cluster behavior (outliers/singletons/topic sizes),
- and structured topic signals that can be integrated into downstream financial models.

---

## Repository Structure (High Level)

```text
README.md
Code/
  01_data_collection.ipynb
  02_data_preprocessing.ipynb
  02_data_standardization.ipynb
  03_data_pipeline_*.ipynb
  04_data_pipeline_BERTopic_TEST_TRAIN.ipynb
  Data/
  iframe_figures/
Literatur Financial Topic Modeling/
```

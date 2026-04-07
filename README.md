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

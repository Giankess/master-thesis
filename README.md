# Financial Topic Modeling Pipeline
## Master's Thesis: Application of NLP & ML Techniques to Identify Topics in and Classify Financial News

**Author:** Giankess (2026)  
**Institution:** Zurich University of Applied Sciences (ZHAW)

---

## Project Overview

This repository contains the implementation of a modular, neural Natural Language Processing (NLP) pipeline designed to extract latent financial narratives from high-velocity news streams. By shifting focus from unidimensional sentiment polarity to **structured thematic extraction**, this artifact addresses the challenge of "Aggregated Sentiment Homogenization" in financial analytics.

### Research Question

**Can transformer-based topic modeling produce stable, meaningful financial narratives over time, and under realistic train/validation/test conditions?**

### Primary Objectives

- Extract **interpretable financial topics** from large-scale news streams using domain-adapted embeddings,
- enable risk monitoring and explainable credit analytics,
- facilitate market regime interpretation and information filtering,
- provide structured signals for downstream econometric and forecasting research.

---

## Key Methodology & Architecture

The pipeline utilizes the **BERTopic** framework, integrating:

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Domain-Adapted Embeddings** | `ProsusAI/finbert` | Bridge linguistic domain gap in financial jargon |
| **Dimensionality Reduction** | UMAP | Preserve local and global semantic manifolds |
| **Density-Based Clustering** | HDBSCAN | Identify dense semantic clusters; filter administrative noise (~68% outlier ratio) |
| **Topic Representation** | c-TF-IDF | Extract interpretable themes from identified clusters |
| **Validation Strategy** | 3-fold rolling time-series CV + multi-seed robustness | Ensure reproducibility and prevent look-ahead bias |

### Core Design Principles

1. **Chronological Data Splitting**: Data sorted temporally; duplicate headlines removed; splits date-based to reflect realistic forecasting constraints.
2. **Temporal Cross-Validation**: 3-fold rolling/expanding time-series validation on train+validation pool; test set untouched during hyperparameter tuning.
3. **Multi-Seed Robustness**: Best configuration re-run with 9 different random seeds on held-out test split to quantify variance and confirm reproducibility.
4. **Interpretability Focus**: Topic words, temporal patterns, and per-topic behavior prioritized over black-box accuracy alone.
5. **Literature-Aligned Metrics**: Composite scoring integrates coherence ($C_v$, $C_{NPMI}$), diversity, intra-topic similarity, and inter-topic separation (see §Hyperparameter Tuning & Ranking).

---

## Experimental Scope & Empirical Highlights

| Metric | Value |
|--------|-------|
| **Dataset Size** | 65,402 deduplicated news headlines |
| **Entity Coverage** | 27 Global Systemically Important Banks (G-SIBs) |
| **Time Span** | 2015–2026 |
| **Outlier Ratio** | ~68% (HDBSCAN density filtering) |
| **Unique Topics (Avg. Across Seeds)** | 45–97 (seed-dependent) |
| **Typical Topic Count per Seed** | ~50 valid topics |

### Key Empirical Findings

- **Domain Adaptation Impact**: Fine-tuned financial embeddings (FinBERT) demonstrated a **50.4% improvement** in intra-topic cohesion over generic models.
- **Stability & Reproducibility**: Multi-seed analysis confirmed extracted topics are reproducible semantic structures, not stochastic artifacts.
- **Economic Relevance**: Disaggregated topic variables exhibited statistically significant correlations with next-day stock returns, confirming the model's ability to capture heterogeneous signals obscured by simple sentiment averages.
- **Test-Only Regression Findings** (recent edits):
  - Regression restricted to test split shows near-zero explanatory power (R² ≈ 0).
  - Most headline-level topics do not survive the pipeline's panel-level filtering (variance filter removes columns with zero variance).
  - Detailed analysis available in regression artifacts and stepwise survival tables (see Recent Work section).

---

## Installation & Environment Setup

### System Requirements

- **Python**: 3.13.12 (Python 3.14 is currently **not supported** due to package compatibility issues with BERTopic/embedding stack).
- **Operating System**: Windows, macOS, or Linux.
- **Memory**: Recommended ≥ 16 GB RAM for full dataset processing and embedding generation.

### Quick Start

1. **Clone and navigate to the repository:**
   ```cmd
   cd master-thesis
   ```

2. **Create and activate a Python virtual environment:**
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```cmd
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```cmd
   python -c "import bertopic; import umap; import hdbscan; print('All core packages installed.')"
   ```

---

## Pipeline Execution

### Recommended Execution Order

The notebooks should be run in the following sequence to maintain data consistency and reproducibility:

1. **Data Collection**
   - Notebook: `Code/01_data_collection.ipynb`
   - Builds raw datasets from financial news sources.

2. **Data Standardization**
   - Notebook: `Code/02_data_standardization.ipynb`
   - Normalizes schema, timestamps, and text fields; removes unusable rows.

3. **Data Preprocessing**
   - Notebooks: `Code/02_data_preprocessing.ipynb`, `Code/02_data_preprocessing_eodhd.ipynb`, etc.
   - Cleans and prepares data for modeling.

4. **Exploratory Model Comparisons** (Optional)
   - Notebooks in `Code/03_data_pipeline_*.ipynb`
   - Compare LDA, Top2Vec, BERTopic, and BERTopic variants.

5. **Main BERTopic Train/Val/Test Workflow**
   - Notebook: `Code/04_04_pipeline_BERTopic_FinBert.ipynb` (or other embedding variant)
   - Implements: chronological split, rolling CV tuning on train+val, multi-seed test evaluation, and diagnostics.

6. **Downstream Analysis**
   - Notebook: `Regression_FinBert_adj_return.ipynb`
   - Validates extracted topics against next-day adjusted stock returns.

### Output Artifacts

Results are stored under:
- `Code/Data/` — processed datasets and model inputs
- `Code/Outputs/` — per-model outputs (row-level assignments, diagnostics, topic info)
- `Code/iframe_figures/` — interactive Plotly visualizations

---

## Reproducibility & Hyperparameter Tuning Methodology

### Random Seed Management

- **Tuning Phase**: Fixed `RANDOM_SEED=42` ensures deterministic hyperparameter selection; variance is reserved for test evaluation.
- **Test Phase**: Best configuration re-run with **9 different random seeds** on the untouched test split to quantify robustness and variance.
- **Reporting**: Final results are presented as **mean ± std** across the 9 seeds.

### Optimized BERTopic Parameters

Key hyperparameter ranges explored:

| Parameter | Range | Optimal (Typical) |
|-----------|-------|-------------------|
| `n_neighbors` (UMAP) | 5–100 | 15–30 |
| `n_components` (UMAP) | 2–50 | 5–10 |
| `min_cluster_size` (HDBSCAN) | 5–50 | 10–20 |
| `min_samples` (HDBSCAN) | 1–10 | 1–3 |
| `ngram_range` | (1, 1), (1, 2) | (1, 1) or (1, 2) |

### Metric Framework (Literature-Aligned)

#### Tuning Metrics (Equal Weight, for CV)

- **Coherence ($C_v$)**: Semantic interpretability [Kretinin & Nguyen]
- **Coherence ($C_{NPMI}$)**: Word co-occurrence consistency [Jehnen et al.]
- **Topic Diversity**: Fraction of unique words across topics (lower redundancy) [Egger & Yu]
- **Intra-Topic Similarity**: Mean semantic similarity within topics (compactness)
- **Inter-Topic Similarity** (inverted): Mean semantic dissimilarity across topics (distinctness) [Jehnen et al.]

**Composite Score Calculation:**

$$\text{composite\_score} = \frac{1}{5}\left(\text{cv\_norm} + \text{npmi\_norm} + \text{diversity\_norm} + \text{intra\_sim\_norm} + \text{inter\_sep\_norm}\right)$$

where each metric is min-max normalized across the hyperparameter grid.

#### Final-Test Diagnostics (Reported but not Used for Selection)

- **Silhouette Score** (cosine, non-outliers): Cluster separation quality
- **Outlier Ratio**: Fraction of `-1` assignments (expected 60–70% for financial news)
- **Topic Count & Singleton Analysis**: Number of valid topics and single-document edge cases

### Interpretation Guidance

- **Silhouette Score > 0.3**: Good cluster separation
- **Silhouette Score 0.1–0.3**: Moderate separation; acceptable for financial domain
- **Silhouette Score < 0.1**: Weak separation; topics overlap in embedding space
- **Outlier Ratio 50–70%**: Typical for noisy financial news
- **Topic Count 10–50**: Reasonable granularity for financial narratives

---

## Key Results & Diagnostics

### Model Selection & Cross-Validation

1. **Phase 1 (Train+Val Tuning)**: 
   - Grid of ~300 parameter combinations evaluated via 3-fold rolling time-series CV
   - Composite score aggregated and ranked per parameter tuple
   - Best configuration selected by highest composite score

2. **Phase 2 (Test Robustness)**:
   - Best config re-fit on train+val, evaluated on test with 9 random seeds
   - Mean and std computed per metric
   - Thesis-ready reporting as `mean ± std`

### Typical Outputs

After tuning and test evaluation, the notebook produces:

- **Tuning Summary Table**: Top 10 parameter configurations by composite score
- **Per-Fold Best**: Best configuration in each CV fold (inspect for temporal drift)
- **Multi-Seed Test Results**: Mean ± std of coherence, diversity, silhouette, outlier ratio, topic count
- **Diagnostic Visuals**: Cluster size distribution, 2D UMAP scatter, per-seed breakdowns
- **Topic Representation**: Top words per topic, topic sizes, temporal evolution

---

## Downstream Regression Analysis: Topic Signals & Stock Return Predictability

### Objective

Validate whether extracted topic signals exhibit predictive power for daily stock returns and quantify the information content of extracted narratives.

### Methodology

#### Regression Specification

**Framework**: Chen (2025) - Panel OLS with entity fixed effects

$$r_{i,t+1} = \alpha_i + \sum_{X} \beta_X \cdot \mathbb{1}(\text{topic\_X\_prevalence}_{i,t} > 0) + \epsilon_{i,t}$$

**Interpretation**: Tomorrow's adjusted return ($r_{t+1}$) predicted by today's news topics ($t$)

**Sample**: Daily observations across 29 G-SIB banks; 2026-01-13 to 2026-05-01

**Dependent Variable**: `daily_adj_return(t+1)` — next-day percentage return

**Independent Variables**: 
- Lagged binary topic indicators (`topic_X_lag1`): presence (1) or absence (0) of topic from day $t$
- Ticker fixed effects: bank-specific intercepts

#### Data Preparation

1. Load row-level topic assignments from BERTopic output
2. Aggregate headline-level assignments to daily frequency by ticker
3. Convert topic prevalence (0–1) to binary presence (0 or 1)
4. Apply 1-day lag: topic presence at $t$ predicts return at $t+1$
5. Remove topic columns with zero variance across regression sample
6. Merge with stock return data on (date, ticker)

### Empirical Results

**Data Summary** (Test Period: 2026-01-13 to 2026-05-01)

| Statistic | Value |
|-----------|-------|
| **Observations** | 3,734 ticker-days |
| **Banks** | 29 G-SIB institutions |
| **Topics in Model** | 312 (after variance filtering) |
| **Date Range** | 2026-01-13 to 2026-05-01 |

**OLS Regression Results**

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **R²** | 0.0932 | Topics explain 9.3% of return variation |
| **Adjusted R²** | 0.0105 | Modest fit after accounting for parameters |
| **F-statistic** | 1.127 | p = 0.0698 (marginally significant) |
| **N Observations** | 3,734 | Ticker-day panel |

**Key Finding**: News topics from day $t$ collectively explain approximately **9.3%** of the variation in next-day adjusted returns. While statistically marginal (p ≈ 0.070), this indicates topics capture some predictive signal for equity returns.

#### Top Topics by Impact

**Largest Positive Return Effects:**

| Topic | Coefficient | Std. Error | Interpretation |
|-------|-------------|-----------|-----------------|
| topic_303_lag1 | +5.045 | — | Increases next-day return by ~5% |
| topic_359_lag1 | +4.960 | — | Increases next-day return by ~5% |
| topic_334_lag1 | +4.450 | — | Increases next-day return by ~4.5% |
| topic_180_lag1 | +4.419 | — | Increases next-day return by ~4.4% |
| topic_128_lag1 | +3.432 | — | Increases next-day return by ~3.4% |

**Largest Negative Return Effects:**

| Topic | Coefficient | Std. Error | Interpretation |
|-------|-------------|-----------|-----------------|
| topic_173_lag1 | -6.118 | — | Decreases next-day return by ~6.1% |
| topic_126_lag1 | -5.549 | — | Decreases next-day return by ~5.5% |
| topic_330_lag1 | -4.098 | — | Decreases next-day return by ~4.1% |
| topic_285_lag1 | -3.984 | — | Decreases next-day return by ~4.0% |
| topic_188_lag1 | -3.890 | — | Decreases next-day return by ~3.9% |

#### Diagnostic Findings

- **Residual Distribution**: Heavy-tailed (see diagnostics plot); suggests outlier events not captured by linear model
- **Heteroskedasticity**: Present in residuals; consider robust standard errors for inference
- **Serial Correlation**: Moderate; daily returns exhibit persistence not fully explained by lagged topics
- **Model Specification**: Linear framework may miss threshold effects or nonlinear topic-return relationships

### Interpretation

**What the Results Tell Us:**

1. **Modest Direct Predictability**: Topics exhibit weak but non-negligible direct predictive power for next-day returns (R² = 9.3%, p ≈ 0.07). This is consistent with semi-efficient markets where news-based signals are partially incorporated into prices but not immediately.

2. **Heterogeneous Topic Effects**: Different topics drive returns in opposite directions, confirming topics capture thematic heterogeneity (e.g., positive topics like "recovery" vs. negative topics like "crisis").

3. **Narrative Validity**: Despite modest R², the existence of measurable topic-return correlations validates the BERTopic extraction; topics represent economically meaningful financial narratives, not noise.

4. **Market Efficiency Implications**: 9.3% explanatory power suggests neither perfect efficiency (topics should have zero power) nor extreme mispricings (topics should have >50% power). This aligns with financial market microstructure literature.

### Robustness & Extensions

**Potential Improvements:**
- Use continuous topic prevalence (instead of binary dummies) to capture intensity of topic exposure
- Add control variables (market returns, volatility, sentiment indices)
- Test for non-linear effects or interaction terms among topics
- Implement Newey–West or clustered standard errors to account for panel structure
- Examine topic effects during crisis vs. normal periods (time-varying relationships)

---

## Repository Structure

```
master-thesis/
├── README.md                                 # This file
├── Thesis Summary.md                         # High-level thesis abstract
├── THESIS_CONTEXT_ENHANCED.md               # Detailed methodological context
├── requirements.txt                          # Python dependencies
├── Code/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_preprocessing*.ipynb
│   ├── 03_data_pipeline_*.ipynb
│   ├── 04_04_pipeline_BERTopic_FinBert.ipynb
│   ├── Regression_FinBert_adj_return.ipynb
│   ├── Data/
│   ├── Outputs/
│   └── iframe_figures/
├── Literatur Financial Topic Modeling/
└── .gitignore
```

---

## Citation

If you use this pipeline or results in your research, please cite:

```bibtex
@mastersthesis{Kessler2026,
  author = {Kessler, Gianfranco},
  title = {Application of {NLP} and {ML} Techniques to Identify Topics in and Classify Financial News},
  school = {Zurich University of Applied Sciences (ZHAW)},
  year = {2026}
}
```

---

## License

This project is released under the **MIT License**. See `LICENSE` file for full terms.

---

## Known Limitations

- **High outlier ratio (~68%)**: Typical for HDBSCAN on financial news; acceptable but limits topic density
- **Weak panel-level predictivity**: Test-only OLS shows minimal direct stock return predictability
- **Computational cost**: Full pipeline embedding & clustering are resource-intensive; consider sampling for larger scales

---

## Acknowledgments

This research builds on foundational work in:
- **Topic Modeling**: Blei et al. (LDA), Angelov (Top2Vec), Grootendorst (BERTopic)
- **Embeddings**: Devlin et al. (BERT), Huang et al. (FinBERT)
- **Manifold Learning**: McInnes & Healy (UMAP)
- **Clustering**: Campello, Moulavi, & Sander (HDBSCAN)

---
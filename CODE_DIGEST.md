# Master Thesis Code Digest - Financial Topic Modeling

**Project**: Financial Topic Modeling via BERTopic with Multi-Dataset Evaluation  
**Framework**: BERTopic (0.17.4), UMAP, HDBSCAN, Sentence-Transformers  
**Datasets**: Finnhub (financial news), GSIB (systemic bank news)  
**Date Generated**: April 19, 2026

---

## Executive Overview

This project implements a comprehensive **topic modeling pipeline** for financial news headlines using:
- **BERTopic**: Main topic modeling framework with semantic embeddings
- **LDA & Top2Vec**: Baseline comparative models
- **Multi-seed robustness evaluation**: 9 random seeds per model for stability analysis
- **Rolling time-series cross-validation**: Hyperparameter tuning with temporal leakage prevention
- **Document-based temporal splitting**: 70% train / 15% validation / 15% test

### Key Differentiators
1. **Temporal integrity**: Calendar-day-aware splitting prevents information leakage
2. **Domain variants**: Both generic and FinBERT (domain-specialized) embeddings tested
3. **Rigorous evaluation**: Multi-seed stability, 5-metric composite scoring, fold-wise diagnostics
4. **Comprehensive export**: Timestamped artifacts for reproducibility and thesis integration

---

## Notebook Architecture

### **Phase 1: Data Collection & Preprocessing**

#### `01_data_collection.ipynb`
- **Purpose**: Fetch financial news headlines from Finnhub API
- **Outputs**: `finnhub_news.csv` (raw API responses)
- **Key Steps**: API calls, error handling, timestamp conversion

#### `02_data_preprocessing_finnhub.ipynb`
- **Input**: `finnhub_news.csv`
- **Output**: `finnhub_clean.csv`
- **Steps**:
  - Parse dates, handle nulls
  - Remove duplicates (by headline)
  - Clean text (lowercase, special chars)
  - Deduplication across tickers (prevent repeats dominating topics)
- **Result**: Clean, deduplicated headlines ready for modeling

#### `02_data_preprocessing_GSIB.ipynb`
- **Input**: GSIB dataset (System Indicators of Systemically Important Banks)
- **Output**: `GSIB_clean.csv`
- **Steps**: Same preprocessing as Finnhub but adapted for GSIB data format

#### `02_data_standardization.ipynb`
- **Purpose**: Create standardized model input files
- **Outputs**:
  - `finnhub_model_input.csv` (final model input)
  - `GSIB_model_input.csv` (final model input)
- **Key Transformation**: Select columns `[date, headline]` with consistent formatting

#### Exploratory Notebooks
- `Data Exploration GSIB.ipynb`: Dataset statistics, date ranges, headline lengths
- `Data Exploration_finnhub.ipynb`: Finnhub-specific exploration

---

### **Phase 2: Baseline Model Comparison (03 Pipelines)**

These notebooks run **initial BERTopic fits** (no tuning, fixed hyperparameters) on smaller datasets for baseline comparison.

#### `03_data_pipeline_BERTopic.ipynb`
- **Model**: BERTopic with generic `all-MiniLM-L6-v2` embeddings
- **Hyperparameters** (fixed):
  - UMAP: n_neighbors=15, n_components=10, metric='cosine'
  - HDBSCAN: min_cluster_size=10, EOM selection
- **Data**: Finnhub or small comparison dataset
- **Outputs**: Topic info, row-level assignments, metrics (CSV export)
- **Export Path**: `Outputs/Comparisons_03/BERTopic/<timestamp>/`

#### `03_data_pipeline_BERTopic_FinBert.ipynb`
- **Model**: BERTopic with `ProsusAI/finbert` (financial domain-specialized)
- **Hyperparameters**: Same as generic variant (fixed)
- **Key Difference**: Domain-specific embeddings for financial vocabulary
- **Export Path**: `Outputs/Comparisons_03/BERTopic_FinBert/<timestamp>/`

#### `03_data_pipeline_LDA.ipynb`
- **Model**: Gensim LDA (Latent Dirichlet Allocation)
- **Key Classes**:
  - `Dictionary`: Gensim vocabulary
  - `CoherenceModel`: C_v and NPMI coherence metrics
- **Outputs**: Topic distributions, topic terms, row-level assignments
- **Export Path**: `Outputs/Comparisons_03/LDA/<timestamp>/`

#### `03_data_pipeline_top2vec.ipynb`
- **Model**: Top2Vec (hierarchical clustering on embeddings)
- **Key API**: `model.get_topics()`, `model.get_documents_topics()`
- **Outputs**: Topic terms, row-level assignments, metrics
- **Export Path**: `Outputs/Comparisons_03/Top2Vec/<timestamp>/`

---

### **Phase 3: Full Pipeline with Tuning (04 Pipelines)**

These notebooks run **complete pipelines** with hyperparameter tuning and multi-seed robustness evaluation on full datasets.

#### `04_01_pipeline_BERTopic_TEST_TRAIN_finnhub.ipynb`
- **Dataset**: Finnhub (~20-30k unique headlines)
- **Embedding**: Generic `all-MiniLM-L6-v2`
- **Pipeline**:
  1. Load & preprocess Finnhub data
  2. Temporal split (70/15/15 by calendar days)
  3. Encode embeddings for train split
  4. Initial model fit on train
  5. Validation evaluation (no refit)
  6. **Hyperparameter tuning**: ~300 combinations via 3-fold rolling CV
     - UMAP: n_neighbors [10,30,50,70,90], n_components [4,6,8]
     - HDBSCAN: min_cluster_size [8,12,16,24,32], min_samples [1,2,4,6]
  7. Best config selected via equal-weighted 5-metric composite score
  8. **Multi-seed final eval**: 9 seeds on test set (42, 7, 123, 2024, 99, 11, 77, 314, 2718)
  9. Diagnostics & export
- **Metrics Computed**:
  - Coherence: C_v, NPMI (Gensim)
  - Diversity: Unique vocabulary fraction
  - Similarity: Intra-topic (within-cluster cosine), inter-topic (centroid-based)
  - Silhouette: Robust variant (drops outliers, singletons)
  - Outlier ratio: Fraction assigned to -1 label
- **Output**: `Outputs/Finnhub_BERTopic/<timestamp>/`
- **README**: Comprehensive 120+ line documentation of all steps

#### `04_02_pipeline_BERTopic_TEST_TRAIN_GSIB.ipynb`
- **Dataset**: GSIB (~90k unique headlines) - **10x larger than Finnhub**
- **Embedding**: Generic `all-MiniLM-L6-v2`
- **Pipeline**: Identical to 04_01 but on larger GSIB dataset
- **Output**: `Outputs/GSIB_BERTopic/<timestamp>/`
- **README**: Same comprehensive documentation, tailored for GSIB

#### `04_03_pipeline_BERTopic_TEST_TRAIN_GSIB_outputs.ipynb`
- **Dataset**: GSIB (~90k unique headlines)
- **Embedding**: Generic `all-MiniLM-L6-v2`
- **Distinguishing Feature**: **Computationally optimized hyperparameter search**
  - Reduced combinations: **36 (vs. 300 in 04_01/02)**
  - UMAP: n_neighbors [15,30,50], n_components [5,10]
  - HDBSCAN: min_cluster_size [10,15,20], min_samples [1,3]
  - Justification: Manage computational load on 90k-document dataset
- **Extended Outputs**:
  - Tuning leaderboard (top ranked configs)
  - Fold-wise best params (temporal stability inspection)
  - Thesis-ready reporting tables (formatted for manuscript)
  - Diagnostic plots (cluster sizes, 2D embeddings, seed variability)
- **Output Structure**:
  - `tables/`: Topic info, terms, metrics, tuning results, seed results
  - `row_level/`: Document-level topic assignments
  - `models/`: Serialized BERTopic model
  - `figures/`: HTML Plotly visualizations
  - `meta/`: JSON metadata and export logs
- **Output**: `Outputs/GSIB_BERTopic/<timestamp>/`
- **README**: 160+ lines emphasizing 36 combinations and 90k dataset scale

#### `04_04_pipeline_BERTopic_FinBert_TEST_TRAIN_GSIB_outputs.ipynb`
- **Dataset**: GSIB (~90k unique headlines)
- **Embedding**: `ProsusAI/finbert` (financial domain-specialized)
- **Purpose**: **Comparative variant** to evaluate domain-specific vs. generic embeddings
- **Hyperparameters**: Same as 04_03 (36 combinations)
- **Expected Insights**:
  - Topic coherence differences (financial domain vs. generic)
  - Topic granularity changes
  - Stability across seeds
  - Alignment with financial concepts
- **Output**: `Outputs/GSIB_BERTopic_FinBert/<timestamp>/`
- **README**: 160+ lines highlighting FinBERT as domain-specialized alternative, with comparative analysis section

---

## 🔧 Core Technical Components

### **Data Pipeline**

#### Temporal Splitting Strategy
```
Goal: Prevent information leakage in time-series financial data

Implementation:
1. Parse dates, sort chronologically
2. Calculate unique calendar days
3. Find day boundaries where cumulative document count ≈ target percentage
   - Train: 70% of documents (earliest days)
   - Validation: 15% of documents (middle days)
   - Test: 15% of documents (latest days)
4. Apply day-level masks (all docs from same day stay together)
```

#### Deduplication
- Remove exact headline matches (across tickers)
- Rationale: Prevent ticker-specific repeats from dominating topics
- Example: "Apple earnings beat" repeated 5x across AAPL, MSFT, etc. → count as 1

### **Embedding Models**

| Model | Source | Dimensions | Use Case |
|-------|--------|-----------|----------|
| `all-MiniLM-L6-v2` | Sentence-Transformers | 384 | Generic, lightweight baseline |
| `ProsusAI/finbert` | Sentence-Transformers (FinBERT wrapper) | 768+ | Financial domain-optimized |

### **Topic Clustering**

#### UMAP (Dimensionality Reduction)
- **Input**: 384-dim (or 768-dim for FinBERT) embeddings
- **Output**: 10-dim reduced embeddings
- **Parameters**: n_neighbors=15, metric='cosine'
- **Purpose**: Project to space suitable for HDBSCAN clustering

#### HDBSCAN (Density-Based Clustering)
- **Input**: 10-dim UMAP projections
- **Parameters**:
  - min_cluster_size: Controls minimum group size (tuned: 8-32 or 10-20)
  - min_samples: Cluster selection robustness (tuned: 1-6 or 1-3)
  - EOM: "Excess of Mass" cluster selection criterion
- **Output**: Cluster assignments (-1 = outlier/noise)

#### Topic Extraction (OnlineCountVectorizer)
- **Input**: Cluster assignments + original documents
- **Process**: Extract representative n-gram terms per cluster
- **Parameters**:
  - stop_words: English + custom (ticker, company, __placeholder__)
  - ngram_range: (1, 2) = unigrams + bigrams
- **Output**: Top K terms per topic with weights

### **Evaluation Metrics**

#### 1. Semantic Coherence
- **C_v**: Validation metric using context vector similarity (Gensim)
- **NPMI**: Normalized Pointwise Mutual Information
- **Computation**: Compare topic terms against corpus
- **Direction**: Higher is better (>0.5 = good)

#### 2. Topic Diversity
- **Formula**: Unique words / Total words across all topics
- **Direction**: Higher is better (>0.5 = diverse vocabulary)

#### 3. Embedding-Based Similarity
- **Intra-topic**: Average cosine similarity within cluster documents
  - Higher = more cohesive topics
- **Inter-topic**: Average cosine similarity between cluster centroids
  - Lower = more distinct topics

#### 4. Silhouette Score (Robust Variant)
- **Handles**:
  - Outliers (-1 label): Excluded from calculation
  - Singleton clusters: Excluded (can't calculate silhouette)
  - Edge cases: Min 10 documents, min 2 valid clusters
- **Range**: [-1, 1]
- **Direction**: Higher is better (>0.3 = reasonable separation)

#### 5. Outlier Ratio
- **Definition**: Fraction of documents assigned to cluster -1
- **Direction**: Lower is better (<10% acceptable)

### **Hyperparameter Tuning**

#### Search Strategy
```
Grid search over parameter combinations
Cross-validation: 3-fold rolling/expanding windows on train+val
Seed: Fixed RANDOM_SEED=42 (deterministic model selection)
```

#### Ranking & Selection
1. Compute all 5 metrics per fold per parameter combo
2. Aggregate metrics across folds (mean)
3. Min-max normalize each metric to [0,1]
4. Calculate composite score: avg(norm_c_v, norm_npmi, norm_diversity, norm_intra, norm_inter_sep)
5. Select config with highest composite score
6. Report fold-wise best configs for temporal stability

#### Parameter Spaces

**04_01/04_02 (Finnhub/GSIB standard)**:
- 300 combinations (5 × 3 × 5 × 4 × 1)
- Higher granularity for sensitivity analysis

**04_03/04_04 (GSIB with 90k dataset)**:
- 36 combinations (3 × 2 × 3 × 2 × 1)
- Reduced for computational efficiency on large dataset

### **Multi-Seed Robustness**

```
Process:
1. Select best hyperparameters from tuning phase
2. For each of 9 seeds: [42, 7, 123, 2024, 99, 11, 77, 314, 2718]
   a. Initialize UMAP with different seed (different random projection)
   b. Initialize HDBSCAN with different seed
   c. Fit model on train+val, evaluate on test
   d. Compute all metrics on test set
3. Report mean ± std across seeds
```

**Rationale**: Validate that selected hyperparameters produce stable results across initialization randomness

---

## 📊 Evaluation Methodology

### Cross-Validation Approach

```
Train+Val Pool (85% of data)
    ↓
3-Fold Rolling CV:
  Fold 1: Train on first 40% of days, validate on next days
  Fold 2: Train on first 60% of days, validate on next days
  Fold 3: Train on first 75% of days, validate on next days
    ↓
Aggregate metrics across folds
    ↓
Select best hyperparams
    ↓
Test Pool (15% of data) - UNTOUCHED during tuning
    ↓
Refit best model on train+val, evaluate on test
    ↓
Multi-seed variability analysis (9 seeds)
```

### Test Set Touch Rule
- **Never use test set** in hyperparameter selection or any tuning decision
- **Only use test set** for final reporting of selected model performance
- **Prevents**: Overfitting to test set, invalid generalization estimates

---

## 📁 Output Structure

### Timestamped Directories

```
Outputs/
├── Finnhub_BERTopic/
│   └── <YYYYMMDD_HHMMSS>/
│       ├── tables/
│       │   ├── topic_info.csv
│       │   ├── topic_terms_long.csv
│       │   └── ... (metrics, leaderboard, seed results)
│       ├── row_level/
│       │   └── row_topic_assignments_train.csv
│       ├── models/
│       │   └── bertopic_model/
│       ├── figures/
│       │   ├── heatmap.html
│       │   ├── embedding_diagnostics.html
│       │   └── seed_variability_plots.html
│       └── meta/
│           ├── export_summary.json
│           └── export_log.json
├── GSIB_BERTopic/
│   └── <YYYYMMDD_HHMMSS>/
│       └── [same structure as above]
├── GSIB_BERTopic_FinBert/
│   └── <YYYYMMDD_HHMMSS>/
│       └── [same structure, FinBERT variant]
└── Comparisons_03/
    ├── BERTopic/
    ├── BERTopic_FinBert/
    ├── LDA/
    └── Top2Vec/
```

### Key Output Files

| File | Contents | Usage |
|------|----------|-------|
| `topic_info.csv` | Topic ID, top terms, doc count | Thesis tables |
| `topic_terms_long.csv` | Topic ID, word, weight | Topic interpretation |
| `row_topic_assignments_train.csv` | Doc ID, assigned topic, probability | Document-level analysis |
| `bertopic_model/` | Serialized model | Inference, further tuning |
| `export_summary.json` | Metadata, run config, saved items | Reproducibility |
| `tuning_results.csv` (04 only) | All parameter combos with metrics | Sensitivity analysis |
| `seed_results.csv` (04 only) | Per-seed test metrics | Stability analysis |

---

## 🔄 Comparative Analysis Framework

### Three-Layer Comparison

**Layer 1: Model Framework**
- BERTopic vs. LDA vs. Top2Vec (03 pipelines)
- Same dataset, fixed hyperparameters

**Layer 2: Embedding Domain**
- Generic embeddings vs. FinBERT (04_03 vs. 04_04)
- Same hyperparameters, same GSIB dataset
- Evaluate if financial domain specialization improves topics

**Layer 3: Dataset Scale**
- Finnhub (~20-30k docs, 04_01)
- GSIB (~90k docs, 04_02, 04_03, 04_04)
- Evaluate scaling effects and hyperparameter sensitivity

---

## 🛠️ Technical Stack

### Core Libraries
```
pandas==1.5.x          # Data manipulation
numpy==1.24.x          # Numerical operations
torch==2.0.x           # GPU support (optional)
sentence-transformers  # Embedding models (SentenceTransformer)
bertopic==0.17.4       # Topic modeling framework
umap-learn==0.5.x      # Dimensionality reduction
hdbscan==0.8.x         # Clustering
gensim==4.2.x          # LDA, coherence metrics
top2vec==1.0.x         # Top2Vec baseline
scikit-learn==1.3.x    # Silhouette, utilities
nltk==3.8.x            # Text processing, stopwords
plotly==5.x            # Interactive visualizations
```

### Environment
- **Python**: 3.13.12
- **GPU**: CUDA 12.x (optional, CPU fallback supported)
- **Batch Processing**: 64-doc batches for embedding encoding

---

## 📝 Naming Conventions & File Structure

### Notebook Naming
```
01_data_collection          → Raw data acquisition
02_data_preprocessing_*     → Dataset-specific cleaning
03_data_pipeline_*          → Baseline models (fixed params)
04_*_pipeline_*             → Full pipelines (with tuning)
04_01 = Finnhub, generic
04_02 = GSIB, generic
04_03 = GSIB, generic, optimized tuning (36 combos)
04_04 = GSIB, FinBERT, optimized tuning (36 combos)
```

### Data File Naming
```
*_raw           → Raw from source
*_clean         → Deduplicated, nulls removed
*_preprocessed  → Cleaned + standardized
*_model_input   → Final form for modeling
```

### CSV Column Conventions
```
date         → Timestamp (datetime)
headline     → Text content (string)
topic        → Assigned topic (int, -1 = outlier)
probability  → Confidence in assignment (float)
document_id  → Unique doc index (int)
word         → Term/token (string)
weight       → Term weight in topic (float)
```

---

## 🔍 Code Quality & Reproducibility

### Deterministic Elements
- **Fixed seeds**: RANDOM_SEED=42 for tuning phase
- **Variable seeds**: 9 intentional seeds for robustness (final eval)
- **Batch processing**: Deterministic with fixed batch size
- **Export naming**: Timestamped, sortable (YYYYMMDD_HHMMSS)

### Metadata Logging
- **export_summary.json**: Timestamps, document counts, item list
- **export_log.json**: Detailed file paths, notes, warnings
- **best_params.json**: Selected hyperparameters for reproducibility

### Error Handling
- **Coherence fallback**: jitter (1e-8) when metrics undefined
- **Silhouette robustness**: Graceful handling of edge cases (1 cluster, all outliers)
- **Metric sanitization**: NaN checks, infinite value filtering

---

## 📈 Expected Results & Baseline Metrics

### Typical Performance (BERTopic on GSIB)

| Metric | Expected Range | Notes |
|--------|---|---|
| C_v Coherence | 0.45-0.65 | Financial domain often 0.50-0.60 |
| NPMI Coherence | -0.10 to 0.10 | Good if >0.02 |
| Topic Diversity | 0.40-0.70 | Unique vocabulary fraction |
| Intra-Topic Sim | 0.35-0.50 | Within-cluster cohesion |
| Inter-Topic Sim | 0.25-0.40 | Between-cluster separation (lower better) |
| Silhouette Score | 0.10-0.40 | Often low but valid |
| Outlier Ratio | 0.05-0.20 | 5-20% noise acceptable |

### Typical Model Size
- **Topics discovered**: 10-30 (BERTopic auto-discovers)
- **Seed variability**: ±1-2 topics (most stable structure)
- **Coherence std across seeds**: ±0.02-0.05

---

## 🎓 Thesis Integration Points

### Tables
- `04_03_*/tuning_leaderboard.csv` → Hyperparameter sensitivity table
- `04_03_*/seed_results.csv` → Multi-seed robustness table (mean ± std)
- `04_03_*/topic_info.csv` → Topic definitions table

### Figures
- `*/figures/*heatmap.html` → Topic correlation heatmap
- `*/figures/*embedding*.html` → 2D embedding space visualization
- `*/figures/*seed_variability.html` → Stability across seeds
- `*/figures/*topics_over_time.html` → Topic evolution timeline

### Comparisons
- **04_03 vs 04_04**: Generic vs. FinBERT embeddings on GSIB
- **04_01 vs 04_02**: Finnhub vs. GSIB dataset effects
- **03_* models**: BERTopic vs. LDA vs. Top2Vec

---

## 🚀 Usage Quick Start

### Run Full Pipeline (Finnhub)
```
Execute 04_01_pipeline_BERTopic_TEST_TRAIN_finnhub.ipynb
↓
Outputs/Finnhub_BERTopic/<timestamp>/
├── tables/ → topic_info.csv, topic_terms_long.csv
├── row_level/ → row_topic_assignments_train.csv
├── models/ → bertopic_model/
├── figures/ → *.html (heatmap, embedding, diagnostics)
└── meta/ → export_summary.json, export_log.json
```

### Run Full Pipeline (GSIB, Optimized)
```
Execute 04_03_pipeline_BERTopic_TEST_TRAIN_GSIB_outputs.ipynb
↓
Outputs/GSIB_BERTopic/<timestamp>/
├── Comprehensive tables including tuning_leaderboard.csv
├── Thesis-ready reporting table (seed_results_summary.csv)
└── Extended diagnostics figures
```

### Comparative Analysis (Generic vs FinBERT)
```
Execute 04_03_pipeline_BERTopic_TEST_TRAIN_GSIB_outputs.ipynb
Execute 04_04_pipeline_BERTopic_FinBert_TEST_TRAIN_GSIB_outputs.ipynb
↓
Compare:
- topic_info.csv (topic definitions)
- seed_results.csv (coherence, diversity, stability)
- figures/ (embedding spaces, heatmaps)
```

---

## 📚 References

### Key Papers/Methods
- **UMAP**: McInnes et al. (2018) - Uniform Manifold Approximation and Projection
- **HDBSCAN**: Campello et al. (2013) - Hierarchical Density-Based Clustering
- **C_v Coherence**: Röder et al. (2015) - Optimizing Metrics for Language Summarization
- **BERTopic**: Grootendorst (2022) - Topic Modeling with BERT
- **FinBERT**: Huang et al. (2022) - FinBERT: A Pre-trained Language Model for Financial Domain

### Tools Used
- **Sentence-Transformers**: Reimers & Gurevych (2019) - Sentence-BERT
- **Gensim**: Řehůřek & Sojka (2010) - Software Framework for Topic Modeling
- **Plotly**: Interactive visualization framework

---

## 📌 Critical Implementation Details

### Why Document-Based Splitting?
- Ensures representative samples (not skewed by date distribution)
- Maintains chronological integrity (same-day docs stay together)
- Prevents leakage: Earlier docs → train, later docs → test

### Why 36 Combinations for 90k Dataset?
- 300 combos × 3 folds × ~30 min per combo ≈ 150+ hours for 04_01 scale
- 36 combos × 3 folds × ~30 min per combo ≈ 18 hours (manageable)
- Careful selection of parameter ranges based on 04_01 tuning results
- Still captures sensitivity across most important dimensions

### Why Multiple Embedding Models?
- **Generic (all-MiniLM)**: Lightweight, universal baseline
- **FinBERT**: Domain-specific, captures financial semantics
- Empirical comparison answers: "Does domain specialization matter?"

### Why Multi-Seed Evaluation?
- UMAP/HDBSCAN have stochastic components (random initialization)
- Single seed can be lucky/unlucky
- 9 seeds + reporting mean ± std provides confidence intervals
- Demonstrates model stability and robustness

---

**End of Code Digest**  
*Generated: April 19, 2026*  
*Repository: master-thesis (Giankess/main)*

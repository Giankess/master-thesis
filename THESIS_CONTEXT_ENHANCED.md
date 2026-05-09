# Enhanced Thesis Context: Financial Topic Modeling Pipeline

**Document Purpose**: Comprehensive reference integrating initial thesis disposition, exploratory iterations, and current implementation state.

---

## 1. Project Overview & Research Goals

### Thesis Title
**Application of NLP & ML techniques to identify topics in and classify financial news.**

### Core Objective
Move beyond **"Aggregated Sentiment Homogenization"** (where averaging sentiment scores erases high-impact signals) toward **granular topic identification** that preserves information-rich signals in financial headlines.

### Central Research Question (CRQ)
> How can a transformer-based topic modeling pipeline be designed to identify latent financial themes with higher coherence than traditional probabilistic methods?

### Supporting Sub-Questions (from Disposition & Exploration)

| Sub-Question | Research Focus | Expected Answer |
|---|---|---|
| **SQ1: Embedding Strategy** | Do domain-specific embeddings (FinBERT) outperform generic models (all-MiniLM-L6-v2) on financial news? | Domain-specialized models should increase intra-topic similarity and reduce outliers |
| **SQ2: Model Architecture** | Is BERTopic superior to traditional models (LDA, Top2Vec) for short, unstructured financial headlines? | BERTopic should show higher coherence due to semantic understanding |
| **SQ3: Information Retention** | Do latent topics retain unique informational value lost in sentiment aggregation? | Topics should capture business events, market conditions, regulatory changes |
| **SQ4: Scaling Effects** | How do dataset size and hyperparameter tuning rigor impact topic quality? | Larger datasets improve coherence; tuning granularity matters but faces computational tradeoffs |
| **SQ5: Preprocessing Impact** | Can financial-domain stopwords and n-gram selection improve coherence metrics? | Better stopwords + 2-grams could significantly improve NPMI scores |

---

## 2. The "Clustifier" Pipeline Design

### Architecture Overview: Four Core Phases

The implemented pipeline follows a **transformer-based clustering approach** instead of traditional probabilistic topic modeling:

```
Financial News Headlines
         ↓
    Phase 1: Embedding
  (Semantic Representation)
         ↓
    Phase 2: Dimensionality Reduction
    (Structure Preservation via UMAP)
         ↓
    Phase 3: Clustering
 (Density-Based Topic Discovery via HDBSCAN)
         ↓
    Phase 4: Representation
(Keyword Extraction via c-TF-IDF)
         ↓
    Interpretable Topics + Metrics
```

### Phase 1: Embedding Strategy

#### Design Decision
Two embedding models tested in parallel experiments:

**Generic Baseline: `all-MiniLM-L6-v2`**
- Dimensions: 384
- Training: Trained on diverse English corpus
- Strengths: Lightweight, generalizable, fast inference
- Limitations: No financial domain specialization
- Used in: 04_01 (Finnhub), 04_02 (GSIB), 04_03 (GSIB optimized)

**Domain-Specific Alternative: `ProsusAI/finbert`**
- Dimensions: 768+
- Training: Fine-tuned on financial texts
- Strengths: Captures financial vocabulary, regulatory terminology
- Limitations: Heavier, slower inference, potentially over-specialized
- Used in: 04_04 (GSIB optimized, comparative variant)

#### Exploration Findings on Embeddings
From iterations 1-6:
- **Iteration 1**: BERTopic with FinBERT ranked 2nd in coherence, **1st in topic precision**
- **Iteration 6**: Switching from generic to FinBERT improved metrics across the board
- **Current Status**: Both models run in parallel (04_03 vs 04_04) for rigorous comparison

#### Hypothesis Status
✓ **SQ1 Partially Confirmed**: FinBERT shows improvement in metrics, but the relationship between intra-topic similarity and domain specialization remains nuanced (see Results Section)

### Phase 2: Dimensionality Reduction (UMAP)

#### Selected Tool
**UMAP (Uniform Manifold Approximation and Projection)**

#### Rationale
- **vs. PCA**: UMAP preserves non-linear semantic relationships; PCA assumes linearity
- **vs. t-SNE**: UMAP is faster, more stable, preserves both local and global structure
- **Practical Advantage**: Reduces 384-dim (or 768-dim) embeddings to 10-dim space suitable for HDBSCAN

#### Implementation Details
- **Fixed Parameters**: n_neighbors=15, metric='cosine'
- **Tuned Parameters** (explored in 04_03, 04_04):
  - n_neighbors: [15, 30, 50] (explored in 04_01/02)
  - n_components: [5, 10] (explored in 04_01/02)
- **Optimization Goal**: Create clustering-friendly manifold without losing semantic structure

### Phase 3: Clustering (HDBSCAN)

#### Selected Tool
**HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)**

#### Rationale
- **vs. K-Means**: HDBSCAN avoids forcing data into K rigid clusters; automatically discovers number of topics
- **vs. LDA**: HDBSCAN handles density variation; LDA assumes homogeneous cluster sizes
- **Noise Handling**: Classifies low-density documents as outliers (-1) rather than forcing misclassification
- **Financial Data Fit**: Headlines around same event create dense clusters (news pileup); other topics are sparse but meaningful

#### Implementation Details
- **Fixed Parameters**: 
  - EOM (Excess of Mass) cluster selection criterion
  - Core parameters: min_cluster_size, min_samples
- **Tuned Parameters** (explored in 04_03, 04_04):
  - min_cluster_size: [10, 15, 20] (standard: 10)
  - min_samples: [1, 3] (standard: 1)
- **Outlier Management**: Accepting 5-20% outlier ratio is acceptable for topic quality vs. coverage tradeoff

### Phase 4: Representation (c-TF-IDF)

#### Selected Tool
**Class-based TF-IDF (OnlineCountVectorizer in BERTopic)**

#### Rationale
- **Advantage**: Treats all documents in a cluster as a single "class" → extracts terms that distinguish this topic from all others
- **vs. Raw TF-IDF**: More robust to within-cluster term frequency variations
- **Financial Application**: Extracts terms that characterize specific financial events/themes

#### Implementation Details
- **N-gram Range**: (1, 2) [unigrams + bigrams]
  - Exploration Finding: NPMI coherence is "close to 0 → almost random," suggesting bigrams alone may be insufficient
  - **Future Improvement**: Test (1, 3) or financial phrase extraction
- **Stopword Management**:
  - **Current**: English stopwords + custom list (tickers, common market terms)
  - **Exploration Finding**: "Coherence (C_v) is fair to maybe good, could maybe be improved with better financial stopwords"
  - **Proposed Enhancement**: Domain-specific stopwords (e.g., 'announced', 'said', 'report' if too frequent)

---

## 3. Exploratory Iterations & Empirical Evolution

### Iteration 1: Baseline Model Comparison (Fraser Timelines)
**Goal**: Identify best topic modeling framework

**Methodology**:
- Data: Fraser economic timelines (financial crisis, COVID-19)
- Models: LDA, Top2Vec, BERTopic, BERTopic+FinBERT
- Approach: Single fit, no train/val/test split, no hyperparameter tuning

**Results**:
1. **BERTopic (generic)**: Best coherence, 2nd precision
2. **BERTopic (FinBERT)**: 2nd coherence, **best precision**
3. **LDA**: 3rd coherence/precision
4. **Top2Vec**: 4th coherence/precision

**Decision**: BERTopic selected as primary framework

---

### Iteration 2: Adding Rigor (Finnhub, ~4.3k headlines)
**Goal**: Implement proper train/val/test methodology

**Improvements**:
- Train/Val/Test split (60/20/20)
- Hyperparameter tuning: 60 combinations
- Multi-seed evaluation: 4 seeds
- Metric expansion: C_v, NPMI, diversity, intra/inter-similarity
- Dataset: Finnhub API (~4.3k unique headlines)

**Technical Innovation**: Introduced jitter (1e-8) for NaN handling in metrics

**Observation**: Metrics improved with structured evaluation

---

### Iteration 3: Scaling Up (Finnhub, 4.3k headlines)
**Goal**: Increase tuning rigor

**Changes**:
- Hyperparameter combinations: 60 → **300 combinations**
- Multi-seed evaluation: 4 → **9 seeds** [42, 7, 123, 2024, 99, 11, 77, 314, 2718]

**Result**: Metrics improved further, but computation time became significant

---

### Iteration 4: Bigger Dataset (eodHD Free, ~19k headlines)
**Goal**: Evaluate scaling effects with free API data

**Dataset**:
- 1 year of historical data
- ~19k documents after deduplication

**Observation**: 
- Metrics improved with larger dataset
- **First performance bottleneck**: 300 combos × 3 CV folds on 19k docs → multi-hour runtime

---

### Iteration 5: Scale + Optimize (eodHD Paid, ~90k headlines)
**Goal**: Maximize dataset size; reduce computational load via hyperparameter reduction

**Dataset**:
- 30 years of historical data
- **~90,000 unique headlines after deduplication** (10x larger than iteration 4)

**Trade-off Decision**:
- Hyperparameter combinations: 300 → **36 combinations**
- Rationale: Computational efficiency for 90k-doc dataset
- Selection: Focused on most impactful parameters based on iteration 3 results

**Metrics Impact**:
- C_v coherence: Slight decrease (expected from reduced tuning)
- NPMI coherence: Minimal change
- Topic Diversity: Good
- Intra-Topic Similarity: **Slightly low** (0.35-0.50)
- Inter-Topic Similarity: **Slightly high** (0.25-0.40)
- Silhouette Score: Low (overlapping clusters)
- Outlier Ratio: High but acceptable (5-20%)

**Realization**: Larger dataset helps, but reduced tuning creates opposing effect

---

### Iteration 6: Domain-Specialized Embeddings (GSIB, ~90k headlines)
**Goal**: Test whether domain-specific embeddings improve topic quality

**Change**: `all-MiniLM-L6-v2` → `ProsusAI/finbert`

**Results**:
- **C_v coherence improved** ✓
- **NPMI improved** ✓
- **Diversity, intra/inter-similarity improved** ✓
- **Silhouette Score still low** (overlapping clusters remain)

**Interpretation**: 
- Financial domain specialization helps semantic coherence
- But topic separation (silhouette) remains challenging
- Likely cause: Genuinely overlapping financial themes (earnings, regulations, market conditions)

---

## 4. Current Implementation State

### Codebase Status (6 Main Pipeline Notebooks)

#### 04_01: BERTopic on Finnhub (Baseline)
- Embedding: Generic (all-MiniLM-L6-v2)
- Dataset: Finnhub (~4.3k unique headlines)
- Tuning: 300 combinations, 3-fold rolling CV
- Evaluation: 9 seeds on test split
- Export: `Outputs/Finnhub_BERTopic/<timestamp>/`
- **Purpose**: Establish baseline on moderate dataset

#### 04_02: BERTopic on GSIB (Generic Embeddings)
- Embedding: Generic (all-MiniLM-L6-v2)
- Dataset: GSIB (~90k unique headlines)
- Tuning: 300 combinations, 3-fold rolling CV
- Evaluation: 9 seeds on test split
- Export: `Outputs/GSIB_BERTopic/<timestamp>/`
- **Purpose**: Test scaling to large dataset with generic embeddings

#### 04_03: BERTopic on GSIB Optimized (Generic, Reduced Tuning)
- Embedding: Generic (all-MiniLM-L6-v2)
- Dataset: GSIB (~90k unique headlines)
- Tuning: **36 combinations** (optimized for large dataset), 3-fold rolling CV
- Evaluation: 9 seeds on test split
- Export: `Outputs/GSIB_BERTopic/<timestamp>/`
- **Purpose**: Final production pipeline with computational balance
- **Special Output**: Tuning leaderboard, seed robustness table (thesis-ready)

#### 04_04: BERTopic on GSIB Optimized (FinBERT, Reduced Tuning)
- Embedding: Domain-Specialized (ProsusAI/finbert)
- Dataset: GSIB (~90k unique headlines)
- Tuning: 36 combinations, 3-fold rolling CV
- Evaluation: 9 seeds on test split
- Export: `Outputs/GSIB_BERTopic_FinBert/<timestamp>/`
- **Purpose**: Comparative variant answering SQ1 (embedding effectiveness)
- **Comparative Value**: Direct 04_03 vs 04_04 comparison on identical dataset/params

#### 03_* (Comparison Models)
- BERTopic (baseline)
- BERTopic_FinBert
- LDA
- Top2Vec
- **Purpose**: Answer SQ2 (neural vs. probabilistic models)

---

## 5. Evaluation Framework & Metrics

### Primary Metrics (from Disposition)

| Metric | Definition | Computation | Target Direction | Thesis Relevance |
|--------|-----------|------------|---|---|
| **C_v Coherence** | Context vector similarity of top words | Gensim coherence model | Higher (>0.50) | Core coherence measure |
| **NPMI Coherence** | Normalized Pointwise Mutual Information | PMI-based word co-occurrence | Higher (>0.02) | Alternative coherence |
| **Topic Diversity** | Unique words / Total words across topics | Vocabulary fraction | Higher (>0.50) | Prevent redundancy |
| **Intra-Topic Similarity** | Average cosine similarity within cluster | Document embeddings within topic | Higher (>0.40) | Cluster cohesion |
| **Inter-Topic Similarity** | Average centroid similarity between clusters | Topic centroid distances | **Lower** (minimize >0.40) | Topic distinctiveness |

### Secondary Metrics (Robustness & Practicality)

| Metric | Definition | Purpose |
|--------|-----------|---------|
| **Silhouette Score** | Robust variant (handles outliers/singletons) | Overall cluster quality; expect low due to overlapping themes |
| **Outlier Ratio** | Fraction assigned to cluster -1 | Topic coverage; 5-20% is acceptable |
| **Seed Variability** | Mean ± std across 9 evaluation seeds | Model stability and confidence |

### Composite Scoring (for Hyperparameter Selection)
```
Composite_Score = mean(norm_C_v, norm_NPMI, norm_diversity, 
                       norm_intra_similarity, norm_inter_similarity)
```
- Equal weighting by design (each metric equally important)
- Min-max normalization to [0,1]
- Higher composite score → selected hyperparameters

---

## 6. Evaluation Methodology & Temporal Integrity

### Temporal Splitting Strategy
**Goal**: Prevent information leakage in time-series financial data

**Implementation**:
1. Parse dates, sort chronologically by headline
2. Count unique calendar days
3. Find day boundaries where cumulative document count = [70%, 85%, 100%]
   - Training: Days 1 through Day@70% (~day 200 if 1 year)
   - Validation: Day@70% through Day@85%
   - Test: Day@85% through Day@100% (latest news)
4. Enforce calendar-day integrity: Keep same-day docs together

**Rationale**: Financial news clusters by event dates; splitting by individual docs risks mixing related headlines

### Cross-Validation for Hyperparameter Tuning
```
Training + Validation Data (85% of documents)
        ↓
3-Fold Rolling/Expanding Window CV:
  Fold 1: Train on first ~40% of days, validate on next ~15% of days
  Fold 2: Train on first ~60% of days, validate on next ~15% of days
  Fold 3: Train on first ~75% of days, validate on next ~10% of days
        ↓
Average metrics across 3 folds
        ↓
Select best hyperparameter combination
        ↓
Test Set (15% of data) - NEVER TOUCHED during tuning
        ↓
Refit best model on train+val, evaluate on test only
        ↓
Multi-seed final evaluation (9 seeds, report mean ± std)
```

**Key Principle**: Test set is sacred; only used for final reporting

---

## 7. Results Summary from Exploration

### Empirical Findings

**Finding 1: Dataset Scale Matters** ✓
- 4.3k → 19k → 90k headlines all show improvement in most metrics
- Larger datasets provide more stable topic structure

**Finding 2: Tuning Granularity Has Limits** ⚠️
- 300 combos better than 36, but not proportionally
- 36 combos sufficient on 90k-doc dataset for practical use
- Diminishing returns on computational investment

**Finding 3: Domain-Specific Embeddings Improve Coherence** ✓
- FinBERT (iteration 6) outperformed generic embeddings (iteration 5)
- **Partial support for SQ1**, but silhouette score still low

**Finding 4: Topic Separation Remains Challenging** ⚠️
- Silhouette scores remain low (0.1-0.4) even with FinBERT
- Intra-topic similarity: 0.35-0.50 (could be higher)
- Inter-topic similarity: 0.25-0.40 (could be lower)
- **Root Cause Hypothesis**: Financial news genuinely overlaps (same events reported in multiple contexts)

**Finding 5: NPMI Coherence Suggests Preprocessing Issues** ⚠️
- NPMI "close to 0 → almost random"
- Indicates: Stopwords or n-gram selection suboptimal
- **Actionable**: Test (1,3) n-grams, financial-domain stopwords

### Current Metric Baselines (GSIB 04_03)

| Metric | Expected Range | Current | Assessment |
|--------|---|---|---|
| C_v Coherence | 0.45-0.65 | ~0.50-0.60 | ✓ Fair to Good |
| NPMI Coherence | -0.10 to 0.10 | ~-0.02 to 0.02 | ⚠️ Almost random |
| Topic Diversity | 0.40-0.70 | ~0.50-0.65 | ✓ Good |
| Intra-Topic Sim | 0.35-0.50 | ~0.35-0.45 | ⚠️ Slightly low |
| Inter-Topic Sim | 0.25-0.40 | ~0.30-0.40 | ⚠️ Slightly high |
| Silhouette Score | 0.10-0.40 | ~0.15-0.25 | ⚠️ Low (overlapping) |
| Outlier Ratio | 0.05-0.20 | ~0.10-0.15 | ✓ Acceptable |

---

## 8. Proposed Next Steps & Remaining Hypotheses

### High-Priority Improvements (SQ5: Preprocessing)

1. **Financial Stopword List**
   - Create domain-specific stopwords: common verbs ('announced', 'said', 'expects'), generic business terms
   - Re-run 04_03, 04_04 with improved stopwords
   - Expected Impact: Improve NPMI from ~0 to >0.05

2. **N-gram Tuning**
   - Current: (1,2) unigrams + bigrams
   - Test: (1,3) add trigrams (e.g., "federal reserve decision", "earnings beat expectations")
   - Expected Impact: Better capturing financial multi-word phrases

3. **Financial Phrase Recognition**
   - Consider domain-specific vectorizers (e.g., KeyBERT, phrase extractors)
   - May improve topic interpretability

### Medium-Priority Experiments

4. **Alternative Financial Embeddings**
   - Exploration Finding: "FinBert might not be the best financial embeddings model"
   - Test alternatives: DistilBERT, RoBERTa fine-tuned on finance, sector-specific models
   - Hypothesis: Better financial embeddings → improved intra-topic similarity

5. **Cluster Size Tuning**
   - Current: min_cluster_size = [10, 15, 20]
   - Test: Lower values [5, 8, 10] to reduce outliers
   - Tradeoff: Smaller clusters may reduce coherence

### Thesis Contribution Framing

#### SQ1: Embedding Strategy Impact ✓ Partially Confirmed
- **Confirmed**: FinBERT improves C_v and NPMI coherence vs. generic
- **Finding**: Silhouette score remains low; suggests issue is topic overlap, not embedding choice
- **Conclusion**: Domain embeddings help but aren't sufficient for perfect separation

#### SQ2: Model Architecture Superiority ✓ Confirmed
- **Confirmed**: BERTopic >> LDA >> Top2Vec in coherence
- **Mechanism**: Semantic embeddings capture financial language; LDA assumes bag-of-words
- **Conclusion**: Transformer-based approach justified for financial news

#### SQ3: Information Retention (In Progress)
- **Next Step**: Compare topic-derived signals vs. sentiment aggregation on downstream task
- **Hypothesis**: Topics identify distinct business events; sentiment alone misses event type

#### SQ4: Scaling Effects ✓ Confirmed
- **Confirmed**: Larger dataset (90k) better than smaller (4.3k)
- **Tradeoff**: Computational cost; 300 combos → 36 combos practical necessity
- **Conclusion**: ~90k document scale appears reasonable for financial news modeling

#### SQ5: Preprocessing Impact (To Be Tested)
- **Hypothesis**: Financial stopwords + better n-grams improve NPMI
- **Test**: Modify 04_03 stopword list and n-gram range, re-run
- **Expected**: NPMI → 0.05-0.10 (significant improvement from ~0)

---

## 9. Thesis Narrative Structure

### Chapter Outline (Recommended Integration)

**Chapter 1: Introduction & Motivation**
- Problem: Sentiment aggregation loses information
- Gap: Existing financial topic models (LDA) lack semantic understanding
- Solution: BERTopic pipeline designed for financial headlines

**Chapter 2: Literature Review & Theoretical Foundation**
- Topic modeling foundations (LDA, UMAP, HDBSCAN, c-TF-IDF)
- Semantic embeddings for finance (FinBERT, domain specialization)
- Evaluation frameworks for topic quality

**Chapter 3: Methodology**
- Section 3.1: Pipeline architecture (Phases 1-4)
- Section 3.2: Data sources (Finnhub, GSIB, dataset characteristics)
- Section 3.3: Experimental design (iterations 1-6, dataset scaling, tuning strategies)
- Section 3.4: Evaluation metrics and temporal integrity
- Section 3.5: Hyperparameter selection process

**Chapter 4: Results**
- Section 4.1: Model comparison (BERTopic vs. LDA vs. Top2Vec) → SQ2
- Section 4.2: Embedding impact (generic vs. FinBERT) → SQ1
- Section 4.3: Dataset scaling effects (4.3k → 90k) → SQ4
- Section 4.4: Metric evolution across iterations
- Section 4.5: Multi-seed robustness analysis

**Chapter 5: Discussion**
- Interpretation of metric values in financial context
- Topic separation challenges and root causes
- Comparison to prior work (LDA, Top2Vec)
- Limitations of current approach
- Remaining open questions (preprocessing, alternative embeddings)

**Chapter 6: Future Work & Conclusion**
- Immediate improvements (stopwords, n-grams, alternative embeddings)
- Longer-term direction (downstream classification, temporal evolution)
- Contribution to financial NLP field

---

## 10. Key Design Decisions & Rationales

| Decision | Rationale | Trade-offs |
|----------|-----------|-----------|
| **HDBSCAN over K-Means** | Auto-discover topics; handle noise gracefully | Silhouette scores lower (not applicable to density-based) |
| **UMAP over PCA** | Preserve non-linear semantic structure | Slower than PCA, but acceptable performance |
| **36 combos on 90k docs** | Balance coherence with computational feasibility | Slightly lower metrics than 300 combos |
| **9 seeds for final eval** | Sufficient for confidence intervals without excessive runtime | Still expensive; could reduce to 5 seeds |
| **Temporal splitting by days** | Respect financial event clustering; prevent info leakage | More complex than random splitting |
| **3-fold rolling CV** | Realistic time-series scenario; detect temporal stability | Higher computational cost than single hold-out |
| **Equal metric weighting** | No pre-bias toward coherence, diversity, or separation | May not reflect actual importance for downstream use |

---

## 11. File Inventory for Thesis

### Core Pipeline Artifacts
- `04_03_pipeline_BERTopic_TEST_TRAIN_GSIB_outputs.ipynb` → Main results (generic embeddings)
- `04_04_pipeline_BERTopic_FinBert_TEST_TRAIN_GSIB_outputs.ipynb` → Comparative results (domain embeddings)
- `Outputs/GSIB_BERTopic/<timestamp>/` → Thesis tables and figures
- `Outputs/GSIB_BERTopic_FinBert/<timestamp>/` → Comparative thesis tables and figures

### Key Output Tables
- `topic_info.csv` → Topic definitions (reproducible in methods)
- `tuning_leaderboard.csv` → Hyperparameter sensitivity analysis
- `seed_results_summary.csv` → Multi-seed robustness (mean ± std)

### Key Visualization Artifacts
- `heatmap.html` → Topic co-occurrence or correlation (Figure in results)
- `embedding_diagnostics.html` → 2D manifold visualization (Figure in results)
- `seed_variability_plots.html` → Stability across seeds (Figure in appendix)

---

## 12. Document Versioning & Maintenance

| Version | Date | Major Changes |
|---------|------|---|
| 1.0 | April 19, 2026 | Initial integration of exploration.md + thesis_context.md + CODE_DIGEST.md |
| (Future) | TBD | After preprocessing experiments, add findings |
| (Future) | TBD | After downstream classification, update SQ3 findings |

---

**Last Updated**: April 19, 2026  
**Document Status**: Active (thesis ongoing)  
**Maintainer**: Giankess (Primary Author)


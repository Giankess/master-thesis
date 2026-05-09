1. Project Overview & Research Goals
Thesis Title: Application of NLP & ML techniques to identify topics in and classify financial news.
Objective: To move beyond "Aggregated Sentiment Homogenization" (where averaging sentiment scores erases high-impact signals) toward granular topic identification.
Central Research Question (CRQ): How can a transformer-based topic modeling pipeline be designed to identify latent financial themes with higher coherence than traditional probabilistic methods?

2. The "Clustifier" Pipeline Design
Your original plan for the IT artifact (the pipeline) follows these four distinct phases:
Phase 1: Embedding Strategy
Baseline: Generic sentence transformers like all-MiniLM-L6-v2.
Proposed Improvement: Domain-specific models such as FinBERT or FinTextSim.
Rationale: Generic models often fail to distinguish specialized financial jargon (e.g., confusing "cost" with "currency").

Phase 2: Dimensionality Reduction
Selected Tool: UMAP (Uniform Manifold Approximation and Projection).
Rationale: Unlike PCA, UMAP captures non-linear semantic relationships and preserves both local and global data structures.

Phase 3: Clustering
Selected Tool: HDBSCAN.
Rationale: Avoids the rigid $K$ (number of topics) constraint of LDA. It handles noise by classifying low-density points as outliers rather than forcing them into irrelevant clusters.

Phase 4: Representation
Selected Tool: c-TF-IDF (Class-based TF-IDF).
Rationale: Treats all documents in a cluster as a single class to extract keywords that define the topic's identity.

3. Exploratory Hypotheses (For Thesis Discussion)
SQ1 (Embeddings): You hypothesized that domain-specific fine-tuning would increase intratopic similarity and reduce outliers.
SQ2 (Model Choice): You argued that neural models (BERTopic) are superior to probabilistic models (LDA) for short, unstructured financial headlines.
SQ3 (Information Retention): You intended to prove that semantic topics retain unique informational value lost in traditional sentiment averages.

4. Evaluation Metrics for Context
Your thesis results should be framed around these specific metrics defined in your disposition:
Topic Coherence (NPMI): To measure semantic similarity between top words.
Topic Diversity: To ensure the model doesn't generate repetitive themes.
Similarity Analysis: Maximizing Intratopic Similarity while minimizing Intertopic Similarity.
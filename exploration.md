This file explains the steps I took to come to the state right now.

1. First iteration: Topic Model comparison using Fraser Timelines
I collected data from fraser timelines, standardized dates and then ran initial simple topic modelling pipelines (without train, val & test) to compare different appraoches like LDA, top2vec, BERTopic & BERTopic with FinBert.
BERTopic performed the best in topic coherence, second in topic precision, BERTopic with FinBert performed second best in topic coherence and best in topic precision. 3rd place was lda and 4th was top2vec.

2. Second iteration: Improving on the base BERTopic Model
Improving Bertopic by implementing a train, val test split with hyperparameter tuning (60 combinations) & 4 different seeds for evaluation. I used the basic all-mini-lm embeddings model.
I then also switched to a bigger dataset from finnhub with almost 4300 headlines.
I evaluated on topic coherence (npmi & c_v), topic diversity, intra- & intertopic similarity. Since some metrics resulted in NaN I implemented jitter values.

3. Third Iteration: Further improvements
I extended to 300 hyperparameter combinations and 9 different seeds. Metrics improved.

4. Fourth Iteration: New dataset
I switched to an even bigger dataset, the free version of eodhd news api with 1 year of historical data. In total I got around 19k documents after headline deduplication. Metrics did improve with the new dataset, however I slowly ran into performance issues with long running times.

5. Fifth iteration: More data, less tuning
I switched to the paid eodhd news api with up to 30 years of historical data. I got 90k rows after deduplication. As tradeoff for better runtime I went from 300 hyperparameter combinations down to 36. Most metrics went down slightly.

6. Sixth iteration: BERTopic with FinBert
I sticked to the approach of iteration 5 but switched out the embeddings model to FinBert. Metrics improved again.

Results so far:
Larger Dataset improves metrics overall (besides topic diversity), but less HP Combinations makes them worse again.
Coherence (C_v) is fair to maybe good, could maybe be improved with better financial stopwords.
Coherence (NPMI) is close to 0 -> almost random -> stopwords or higher ngram_range
Topic Diversity is good
Intra-Topic similarity a bit low, Inter-topic similarity a bit high
Silhouette Score is low which means overlapping clusters -> financial embeddings
Outlier Ratio is high but ok since we want meaningful topics -> could be reduces with lower min_cluster_size

Options:
Better preprocessing with financial stopwords
Higher ngram_range
Financial Embedding Model

FinBert might not be the best financial embeddings model for topic modelling so it might make sense to find and try out different ones.
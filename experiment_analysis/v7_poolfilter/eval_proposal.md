Note: The goal is to analyze previously trained models against another test set consisting of unused samples of the training set (see v7_pooliflter_analysis.md for analysis of the training runs from which the models resulted). This new test set is much larger but also has much fewer positives (~13%).

We augment 128 real data samples for a check-worthy sentence classification task with 1024 synthetic LLM generated sentences. 5 real datasets were sampled and augmented with various techniques to explore which if any improve the baseline result.
- group_name: v7_poolfilter_eval
- pattern: `seq_{i}_aug_{a}_seed{s}` where `a` is a string denoting the augmentation method of which there exist the following:
  - "none": The baseline serving as comparison reference for all other techniques.
  - "free": A simple method where the model is prompted to generate positive and negative check-worthy sentences without further restrictions.
  - "tfidf": A large pool of synthetic samples is filtered with a tf-idf based filter that keeps synthetic samples close to the real samples while also maximizing diversity amongst synthetic samples.
  - "embed": Same approach as "tfidf" but similarity is measured with sentence embeddings.
  - "genetic": Generates samples with a genetic algorithm based on Han et.al. 2025.
- run_type: train
- Hypothesis (for each method): F1 results on the test set are improved over the baseline.

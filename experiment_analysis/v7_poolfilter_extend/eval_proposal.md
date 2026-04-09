Note: The goal is to analyze previously trained models against another test set (see v7_pooliflter_extend_analysis.md for analysis of the training runs from which the models resulted). This new test set is much larger but also has much fewer positives (~13%). 

We augment 128 real data samples for a check-worthy sentence classification task with 1024 synthetic LLM generated sentences. 15 real datasets were sampled and augmented with various techniques.
- group_name: v7_poolfilter_extend_eval
- pattern: `seq_{i}_aug_{a}_seed{s}` where `a` is a string denoting the augmentation method of which there exist the following:
  - "none": The real data baseline without augmentation serving as comparison reference for all other techniques.
  - "embed": A large pool of synthetic samples is filtered with a embedding based filter that keeps synthetic samples close to the real samples while also maximizing diversity amongst synthetic samples. This method showed the highest F1 gain in an initial exploratory experiment.
  - "unfiltered": Takes a random 1024 subset of the "embed" pool without filtering to investigate whether the filtering causes the performance increase.
  - "embed-multi": Uses a multi-task training approach on the same data from "embed" with a supervised contrastive loss in addition to the default classification loss to investigate whether this leads to higher improvement from baseline.
  - "embed-multi-gunel": Uses Gunel et al. (2021) hyperparameters (alpha=0.9, tau=0.3, CLS pooling, no projection) on the same embed data. REPLACES old embed-multi.
- run_type: eval
- NOTE: Exclude old embed-multi from analysis: --exclude-aug embed-multi --allow-multi
- Hypothesis (for each method): F1 results on the test set are improved over the baseline.
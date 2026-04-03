Note: The goal is to analyze previously trained models against an out of distribution test set from CT22 data (covid topoic) to investigate if augmentation helps generalization to unseen data. The same models as in the other "v7_poolfilter_extend_*" are used.

- group_name: v7_poolfilter_extend_ct22_eval
- pattern: `seq_{i}_aug_{a}_seed{s}` where `a` is a string denoting the augmentation method of which there exist the following:
  - "none": The real data baseline without augmentation serving as comparison reference for all other techniques.
  - "embed": A large pool of synthetic samples is filtered with a embedding based filter that keeps synthetic samples close to the real samples while also maximizing diversity amongst synthetic samples. This method showed the highest F1 gain in an initial exploratory experiment.
  - "unfiltered": Takes a random 1024 subset of the "embed" pool without filtering to investigate whether the filtering causes the performance increase.
  - "embed-multi": Uses a multi-task training approach on the same data from "embed" with a supervised contrastive loss in addition to the default classification loss to investigate whether this leads to higher improvement from baseline.
- run_type: eval
- Hypothesis (for each method): F1 results on the OOD test set are improved over the baseline.
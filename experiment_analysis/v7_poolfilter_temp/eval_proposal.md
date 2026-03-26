Note: The goal is to analyze previously trained models against another test set consisting of unused samples of the training set (see v7_pooliflter_temperature_analysis.md for analysis of the training runs from which the models resulted). This new test set is much larger but also has much fewer positives (~13%).

We augment 128 real data samples for a check-worthy sentence classification task with 1024 synthetic LLM generated sentences. 5 real datasets were sampled and augmented with the best performing technique of a previous experiment (see v7_poolfilter_extend_analysis) while varying the generation temperature.
- group_name: v7_poolfilter_temperature_eval
- pattern: `seq_{i}_aug_embed-temp-{t}-multi_seed{s}` where `t` is a number denoting the temperature:
  - "1": The baseline matching the temperature setting of 1.0 of the previous experiment.
  - "05": Reduced temperature setting of 0.5.
  - "125": Increased temperature setting of 1.25
- run_type: eval
- Hypothesis: F1 performance on the test set is affected by temperature.
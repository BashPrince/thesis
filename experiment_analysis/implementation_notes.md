  ## Implementation notes
  - Method `none` is the unaugmented baseline
  - Method `free` was implemented using the script in `generate/free_generations/generate.py`
  - Method `tfidf`, `embed` and `unfiltered` were generated using the script in `generate/filter/v7_generate.py`
  - Method `genetic` was implemented with the script in `generate/sequences/experiment_011/augment.py`
  - Experiments were run using the script `finetune/run_classification.py` and tracked to wandb `redstag/thesis`
  - Method `embed-multi` uses the data of `embed` and the multi-task mode with a joint classification + contrastive loss in run_classification.py
  - Each wandb run has in its tracked files a file `configs/*.json` (**NOT** the config.yaml in the root) which is the training/eval config provided to run_classification.py listing various parameters - runs in a common group resulting from the same augmentation method have the same config (except for specific dataset and seeds) so inspecting one such run's config let's you know the used params
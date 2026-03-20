---
name: analyze-experiment
description: Analyse a completed synthetic data augmentation experiment using its proposal file. Downloads WandB runs for the experiment group, runs the analysis described in the proposal, and writes a report to experiment_analysis/.
user-invocable: true
disable-model-invocation: true
---
## Setting
We are working on a sentence classification task with a simulated low data scenario. We are investigating the effect of synthetic data augmentation methods on test set F1 performance. Our model is a BERT variant finetuned with adapters. Unless otherwise specified an experiment consists of a number of sequences of small non-overlapping base sets sampled from a bigger data source each of which is augmented with synthetic data in one or multiple augmentation steps.

## Goal
Given an experiment proposal, fetch all WandB runs in the corresponding group, run analysis/analyze_experiment_metrics.py, perform additional analysis described in the proposal or as you see fit, and write a structured report. The script itself serves as a good illustration of the setup and naming of runs within an experiment group.

Runs result from running finetune/run_classification.py on a remote machine.

The most important metric of interest is test/f1 which evaluates a trained model on a small test set.

## Steps

### 1. Gather context

- Ask for the proposal file path if not provided (e.g. `experiment_analysis/experiment_1_proposal.md`).
- Read the proposal file. It should contain:
- `group_name` name of the wandb group of the experiment
- `pattern` exptected run naming pattern and which runs form the baseline
- `run_type` stating if this is a train run or an eval run against an additional test set
- `hypotheses`

### 2. Perform default analysis
Run `analysis/analyze_experiment_metrics.py` with the group name. This performs completion checks, per-sequence F1 table, paired t-tests vs baseline, RM-ANOVA, and seed variance analysis.

```
python analysis/analyze_experiment_metrics.py <group> [options]

Options:
  --entity          WandB entity (default: redstag)
  --project         WandB project (default: thesis)
  --expected-seeds  Expected seed runs per (seq, aug) combo (default: 3)
  --baseline-aug    Aug level used as comparison baseline (default: 0)
```

### 3. Download and inspect metrics
**This should only be performed when the experiment is the one that trained the models. Some groups are the result of evaluating against a different test set so they will not contain meaningful train or eval metricsc.**

Inspect column names before assuming metric names — they vary by run type:

```python
import wandb, pandas as pd
api = wandb.Api()
runs = api.runs("redstag/thesis", filters={"group": "<group_name>"})
print(f"Found {len(runs)} run(s) in group '{<group_name>}'")
for r in sorted(runs, key=lambda r: r.name):
    print(f"  {r.id:12s}  state={r.state:10s}  job_type={r.config.get('wandb_job_type','?'):10s}  name={r.name}")
```

```python
sample_run = runs[0]
hist = sample_run.history(samples=10)
print("Columns:", hist.columns.tolist())
```

Then download history for a suitable small representative subset of runs and look for any noteworthy patterns.

### 4. Perform the analysis described in the proposal
If the proposal lists any analysis perform it.

### 5. Perform additional analysis (optional)
You can investigate anything you find worthy of being investigated.

### 6. Write the report

Save `<group_name>_analysis.md` to the same directory as the proposal. Rough structure:

```markdown
# Analysis: <group_name>
*Proposal: <proposal file>*

## Results summary

## Metric/training analysis

## Hypothesis evaluation

## Conclusion/recommended next steps
```

Do not speculate beyond what the data shows. If something is unclear, say so and suggest what additional logging would help. If the hypothesis is clearly rejected state this but try to find insights that could still be informative in a thesis.
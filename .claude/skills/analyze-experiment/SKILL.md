---
name: analyze-experiment
description: Analyse a completed experiment using its proposal file. Downloads WandB runs for the experiment group, runs the analysis described in the proposal, and writes a report to experiment_analysis/.
disable-model-invocation: true
---

## Goal

Given an experiment proposal, fetch all WandB runs in the corresponding group, perform the analysis described in the proposal, and write a structured report.

## Steps

### 1. Gather context

- Ask for the proposal file path if not provided (e.g. `experiment_analysis/experiment_1_proposal.md`).
- Read the proposal file. It contains: group name, conditions, baselines, hypotheses, and an **Analysis guidance** section describing what to compute and compare.
- Ask for the WandB entity if not clear from context (project is `thesis`).

### 2. Fetch all runs in the group

```python
import wandb, pandas as pd
api = wandb.Api()
runs = api.runs("<entity>/thesis", filters={"group": "<group_name>"})
print(f"Found {len(runs)} run(s) in group '{<group_name>}'")
for r in sorted(runs, key=lambda r: r.name):
    print(f"  {r.id:12s}  state={r.state:10s}  job_type={r.config.get('wandb_job_type','?'):10s}  name={r.name}")
```

Note any runs still in progress or crashed. Continue with finished runs; flag missing or failed ones explicitly in the report.

### 3. Download metrics

Inspect column names before assuming metric names — they vary by run type:

```python
sample_run = runs[0]
hist = sample_run.history(samples=10)
print("Columns:", hist.columns.tolist())
```

Then download history for all runs. For each run collect the metrics specified in the proposal's **Analysis guidance** section, plus the run config fields that define the condition.

### 4. Perform the analysis described in the proposal

Follow the **Analysis guidance** section of the proposal exactly. It will specify:
- Which metrics to compare across conditions
- Which baselines to compare against and how to fetch them
- Any diagnostic analyses to run (e.g. correlations, phased training dynamics)
- What the primary success criterion is

Use `run.config` to extract condition-defining hyperparameters as ground truth rather than parsing run names.

### 5. Check training dynamics

For at least one representative run per condition, do a phased analysis to catch anomalies:

```python
df["phase"] = pd.cut(df[step_col], bins=10, labels=False)
print(df.groupby("phase")[key_metric_cols].mean().round(4).to_string())
```

Flag: best checkpoint in phase 0–1 (too-early stopping / collapse), best checkpoint at final phase (needs more epochs), high variance across seeds (unstable optimisation).

### 6. Write the report

Save to `experiment_analysis/<group_name>_analysis.md`. Structure:

```markdown
# Analysis: <group_name>
*Proposal: <proposal file>*

## Results summary
<table of per-condition results, ranked by primary metric>

## Against baselines
<comparison to baselines named in the proposal>

## Diagnostic analyses
<findings from the analyses described in the proposal's Analysis guidance>

## Training dynamics
<anomalies or notable patterns>

## Hypothesis evaluation
<each hypothesis from the proposal: SUPPORTED / REJECTED / INCONCLUSIVE with evidence>

## Conclusions
<what was learned>

## Recommended next steps
<follow the proposal's decision tree; be specific about configs or code changes>
```

Do not speculate beyond what the data shows. If something is unclear, say so and suggest what additional logging would help.

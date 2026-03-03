---
name: analyze-run
description: Analyse a training run from WandB and produce a written diagnosis with insights, potential issues, and suggestions.
disable-model-invocation: true
---

## Goal
Download and analyse a WandB training run, report on training behaviour, identify issues, and suggest improvements.

## Steps

### 1. Gather context
- If the user has not provided the WandB run path (`entity/project/run_id`), ask for it.
- If the user has not mentioned which config was used for the run, ask them.
- Note whether the run is still in progress — analysis of partial runs is fine.
- If not otherwise specified assume a run is the product of the script `./finetune/run_classification.py` executed on a remote server.

### 2. Download run history
Run the following to download metrics to `/tmp/<run_id>_history.csv`:

```python
import wandb
api = wandb.Api()
run = api.run("<entity/project/run_id>")
df = run.history(samples=50000)
df.to_csv("/tmp/<run_id>_history.csv", index=False)
print("Steps logged:", len(df))
print("Columns:", df.columns.tolist())
print(df.head(3).to_string())
```

Also fetch the run config to understand hyperparameters:

```python
import wandb
api = wandb.Api()
run = api.run("<entity/project/run_id>")
for k, v in sorted(run.config.items()):
    print(f"  {k}: {v}")
```

### 3. Discover available metrics
Inspect the column names from the downloaded CSV before deciding what to analyse. Do not assume specific metric names — they vary by run type. Identify:
- The primary loss metric(s)
- Any diagnostic metrics (e.g. similarity metrics, accuracy, gradient statistics)
- Learning rate and epoch/step columns
- Evaluation metrics if present

### 4. Compute summary statistics
For each meaningful metric, compute:
- Overall min / max / mean
- Value at the start and end of the run
- Trend (improving, plateauing, diverging)
- Any periodic patterns or anomalies

Use phased analysis (split into ~10 equal buckets) to show trends over the course of training:

```python
df["phase"] = pd.cut(df[step_col], bins=10, labels=False)
print(df.groupby("phase")[metric_cols].mean().round(4).to_string())
```

### 5. Identify anomalies
- **Spikes**: Find steps where metrics deviate sharply from the local trend. Check if they occur periodically (e.g. every N steps) and correlate with epoch boundaries, checkpoint saves, or evaluation events.
- **Divergence**: Loss or gradient norm continuously increasing.
- **Stagnation**: Primary metric not improving over many steps.
- **Inconsistencies**: Metrics that contradict each other (e.g. diagnostic metrics improving while loss is flat or vice versa).

### 6. Cross-reference with config
Compare observed behaviour against the hyperparameters:
- Is the learning rate / warmup appropriate for the number of steps?
- Is the batch size appropriate for the training objective?
- Are save/eval strategies consistent with each other?
- Are any relevant settings at their defaults that may not be appropriate?

### 7. Report
Structure the report as follows:

**Training progress** — how far through training, what has improved, overall trajectory.

**What's working** — metrics trending in the right direction, stable training dynamics.

**Issues** — concrete problems observed in the data, with evidence (step numbers, metric values). Distinguish between anomalies that are harmless vs those likely impacting training quality.

**Suggestions** — specific, actionable config or implementation changes. Reference the config the user mentioned where relevant.

Avoid speculating beyond what the data shows. If something is unclear, say so and suggest what additional logging or diagnostics would help.

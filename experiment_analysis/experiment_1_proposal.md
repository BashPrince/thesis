# Experiment 1: SupCon Temperature & Projection Head

## Context

We fine-tune ModernBERT-base with bottleneck adapters (seq_bn) on a low-data checkworthiness classification task (128 real + 1024 LLM-augmented samples). The hypothesis is that a supervised contrastive pre-training stage — training the adapter to cluster same-class sentences together in embedding space — could improve the downstream classifier initialisation.

**Current result**: contrastive pre-training (temp=0.4, MLP projection head) makes classification worse than classification-only on the same augmented data.

**Baselines in WandB** (both in group `contrastive_unrestricted_wrup`):
- Classification-only runs (no "contrastive" in run name): augmentation-only, no contrastive stage
- Contrastive pipeline runs (names ending `_contrastive` / `_contrastive_pretrain`): current two-stage pipeline, temp=0.4, MLP head

## Hypotheses

**H1 – Temperature too high**: SupCon literature (Khosla et al., 2020) and SimCSE consistently recommend τ ∈ [0.05, 0.1]. At τ=0.4 the softmax over similarities is nearly flat, producing weak gradients that barely push same-class embeddings apart. Tighter temperature should produce meaningfully separated clusters.

**H2 – MLP projection head creates a representation gap**: The 2-layer MLP projection head is trained jointly with the adapter, then *discarded* before classification fine-tuning. The adapter's raw output was never directly optimised to be linearly separable — only the projected output was. A single linear projection would reduce the non-linearity the adapter can offload into the head, keeping more discriminative structure in the adapter weights themselves.

## Experiment Design

**Group name**: `exp1_temp_proj`
**WandB job types**: `pretrain` (contrastive stage), `train` (classify stage)

### Conditions (4 × 3 seeds = 12 pairs)

| Config idx | Temperature | Proj head | Sequence (seed) |
|---|---|---|---|
| 01–03 | **0.07** | mlp | seq_0 (13579), seq_1 (24680), seq_2 (97531) |
| 04–06 | **0.10** | mlp | seq_0 (13579), seq_1 (24680), seq_2 (97531) |
| 07–09 | **0.20** | mlp | seq_0 (13579), seq_1 (24680), seq_2 (97531) |
| 10–12 | **0.07** | **linear** | seq_0 (13579), seq_1 (24680), seq_2 (97531) |

Same sequences/seeds across conditions allow paired comparison. All other hyperparameters match the existing `contrastive_unrestricted_wrup` pipeline runs exactly.

**Fixed hyperparameters**:
- Data: `unrestricted_wrup_seq_{i}_aug_1024` (128 real + 1024 synthetic)
- Contrastive batch size: 64, balanced sampling, mean pooling
- Epochs: 106 (≈ 120k / 1124), same for both stages
- eval/save every 64 steps; patience 10 (both stages)
- Projection dim: 128

**Infrastructure**: 12 pretrain + 12 classify configs. All 12 pretrain can run in parallel (needs ≤ 8 GPUs; runs in 2 rounds of 8+4). Classify stage starts per-GPU as pretrain completes.

**Estimated wall time**: ~20–40 min on 8× A100.

## Expected Outcomes & Decision Tree

```
Temperature sweep result
├── Lower temp (0.07 or 0.1) beats contrastive baseline (temp=0.4)?
│   ├── YES → Temperature was the main issue.
│   │         Check linear proj result for further gains.
│   │         Round 2 options: projection dim sweep, real-only contrastive.
│   │
│   └── NO  → Temperature not the bottleneck.
│             ├── Linear proj helps?
│             │   YES → Representation gap is the issue.
│             │         Round 2: no projection head (direct SupCon on adapter).
│             └── Neither helps → Two-stage approach structurally broken at this scale.
│                                 Round 2: multi-task SupCon + CE loss (code change needed).
```

Key question: does *any* condition exceed the classification-only baseline? Even marginal improvement is meaningful — it shows the contrastive signal can be useful and justifies further tuning.

## Analysis guidance

**Primary metric**: `eval_f1` from classify runs. Group by condition (`contrastive_temperature` + `contrastive_proj_type` from `run.config`), compute mean ± std over 3 seeds.

**Baselines**: fetch from group `contrastive_unrestricted_wrup` (project `thesis`):
- Classification-only: runs whose name contains no `contrastive` substring, using the same `aug_1024` artifacts
- Previous contrastive: runs whose name ends with `_contrastive` (temp=0.4, MLP head)
Include both in the results table.

**Contrastive stage diagnostics**: for pretrain runs, report `eval_cosim_gap` (best and final) per condition. Then join paired pretrain/classify runs on `seed` + `data_artifact` and compute the Pearson correlation between `best_cosim_gap` and `best_f1`. A correlation near zero means the contrastive objective is not predictive of downstream quality.

**Training dynamics**: for each condition pick the median-F1 seed. Check the classify run for early stopping timing (which phase the best checkpoint falls in) and whether the pretrain run's cosim_gap plateaued early or was still rising at the end.

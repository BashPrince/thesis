# Analysis: exp1_temp_proj
*Proposal: experiment_analysis/experiment_1_proposal.md*

## Results summary

All 24 runs finished successfully (12 pretrain, 12 classify). n=3 seeds per condition.

| Condition | mean F1 | ±std | vs classify-only | vs prev contrastive (t=0.4) |
|---|---|---|---|---|
| **t007_lin** (τ=0.07, linear proj) | **0.787** | 0.035 | −0.002 | +0.012 |
| t007_mlp (τ=0.07, MLP proj) | 0.772 | 0.029 | −0.017 | −0.003 |
| t010_mlp (τ=0.10, MLP proj) | 0.767 | 0.033 | −0.023 | −0.008 |
| t020_mlp (τ=0.20, MLP proj) | 0.763 | 0.022 | −0.027 | −0.012 |
| — | — | — | — | — |
| **Classify-only baseline** | 0.790 | 0.027 | — | — |
| **Prev contrastive** (τ=0.4, MLP) | 0.775 | 0.028 | −0.015 | — |

Baselines from group `contrastive_unrestricted_wrup`, n=15 each (all 5 sequences × 3 seeds).

## Against baselines

**Primary success criterion (beat classify-only): NOT ACHIEVED** for any condition. The best condition (t007_lin, 0.787) is 0.002 below the classify-only baseline (0.790). With std ~0.035 and n=3, this gap is well within noise — it is genuinely unclear from this data whether t007_lin is better or worse than classify-only.

**Secondary criterion (beat previous contrastive): ACHIEVED** for t007_lin (+0.012) and partially for t007_mlp (−0.003, within noise of the old baseline).

## Diagnostic analyses

### cosim_gap per condition

| Condition | mean cosim_gap | ±std |
|---|---|---|
| t007_lin | 0.461 | 0.037 |
| t007_mlp | 0.490 | 0.069 |
| t010_mlp | 0.634 | 0.103 |
| t020_mlp | 0.950 | 0.142 |

cosim_gap increases strongly with temperature. The contrastive stage at τ=0.20 appears to train "better" by its own metric than at τ=0.07.

### cosim_gap → downstream F1 correlation

**Pearson r = 0.035 (p = 0.914, n = 12 paired runs).** Effectively zero. The contrastive objective's own metric is completely uninformative about downstream classification quality.

More strikingly, the *ranking across conditions* is reversed: t020_mlp achieves the highest cosim_gap (0.95) but the lowest F1 (0.763); t007_lin achieves the lowest cosim_gap (0.46) but the highest F1 (0.787). Better contrastive separation, as the stage measures it, does not help — and at high temperature it may actively harm downstream performance by making larger, less task-aligned representation changes.

**Implication**: `eval_cosim_gap` is not a useful early stopping or model selection signal. Optimising it further does not predictably improve classification.

## Training dynamics

Best checkpoints appear at ~39% through training on average (range 17–73%), well before the 106-epoch budget. In all conditions, eval loss rises steadily from P1 to P5 (overfitting), confirming that the early stopping trigger is appropriate and more epochs would not help.

| Condition | F1 by phase (P1→P5) | Pattern |
|---|---|---|
| t007_lin | 0.744 → 0.752 → 0.768 → 0.765 → 0.767 | Gradual rise then plateau |
| t007_mlp | 0.737 → 0.756 → 0.760 → 0.758 → 0.759 | Fast rise then plateau |
| t010_mlp | 0.725 → 0.759 → 0.751 → 0.752 → 0.751 | Peaks P2, flat |
| t020_mlp | 0.735 → 0.756 → 0.756 → 0.755 → 0.755 | Peaks P2, flat |

t007_lin is the only condition where F1 continues to improve into phase P3, suggesting the linear-projection adapter init provides a more gradual and sustained learning signal rather than converging immediately to a local plateau.

**Outlier**: t007_mlp seq_1 (seed 24680) had best_step=1728 out of 2368 total steps — substantially longer than all other runs (typically 128–512). This is likely noise from the small dataset interacting with a particular random seed, not a signal.

## Hypothesis evaluation

**H1 – Temperature too high (τ=0.4)**: **SUPPORTED.** Lower temperatures consistently improve downstream F1: 0.787 (τ=0.07) > 0.772 (τ=0.07) > 0.767 (τ=0.10) > 0.763 (τ=0.20) > 0.775 (τ=0.4, previous baseline). Every reduction in temperature from 0.4 downward yielded a better result. The effect is monotone and consistent across seeds.

**H2 – MLP projection head creates a representation gap**: **SUPPORTED.** At the same temperature (τ=0.07), linear projection (0.787) outperforms MLP (0.772) by 0.015. This is consistent with the hypothesis that the MLP allows the adapter to offload non-linear complexity into the projection head, leaving less discriminative structure in the adapter's raw output after the head is discarded.

## Conclusions

1. Temperature and projection head both matter in the expected directions. The two hypotheses are supported.
2. Despite fixing both, the best contrastive condition (t007_lin) still only matches — not beats — the classify-only baseline, within measurement noise.
3. The cosim_gap metric is decoupled from downstream F1 and should not be used as a training signal. Higher cosim_gap (achieved by higher temperature or longer training) is actually associated with worse F1 across conditions.
4. The two-stage setup works just well enough to not hurt, but has not yet demonstrated it adds value.
5. The std across seeds (~0.03) is large relative to between-condition differences (~0.01–0.025), meaning n=3 is insufficient to draw firm conclusions about which specific config is best. The clearest signal is the aggregate trend: lower temperature + linear projection = better.

## Recommended next steps

Following the decision tree from the proposal (H1 and H2 both supported):

**Priority 1 — resolve the baseline comparison ambiguity.**
Run t007_lin on all 5 sequences (add seq_3 and seq_4) to get n=5 and tighter confidence intervals. This is cheap (5 runs, no new code). The key question: does contrastive pre-training with the best-found config genuinely match or exceed the classify-only baseline? If yes (even marginal significance), the approach is validated and worth further investment. If no, pivot to multi-task.

**Priority 2 — ablate the contrastive init value.**
Run the classification stage for t007_lin without loading the contrastive adapter (standard cold-start classification with the same hyperparameters: lr=5e-5, patience=10, eval_steps=64). This directly measures how much of t007_lin's F1 comes from the contrastive initialisation vs. just the hyperparameter choices. If the cold-start matches t007_lin, the contrastive stage is overhead.

**Priority 3 (if Priorities 1–2 suggest contrastive does help) — remove the projection head entirely.**
Modify `run_classification.py` to allow `contrastive_proj_dim: 0` (or similar flag) to skip the projection head and apply SupCon loss directly to the normalised adapter output. This tests whether the representation gap is the remaining bottleneck.

**Priority 4 (if contrastive stage provides no measurable benefit) — multi-task.**
Combine SupCon and cross-entropy losses in a single training stage, with a tunable λ weight. This eliminates the two-stage objective mismatch entirely.

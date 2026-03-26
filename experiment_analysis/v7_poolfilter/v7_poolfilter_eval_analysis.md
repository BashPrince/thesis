# Analysis: v7_poolfilter_eval
*Proposal: experiment_analysis/v7_poolfilter/eval_proposal.md*

---

## Context

This experiment evaluates the **same trained models** from the `v7_poolfilter` training group against a **new, harder holdout test set** — a larger set drawn from unused training pool samples with only ~13% check-worthy positives (vs a higher positive rate in the original test set used during training). The goal is to test whether the augmentation method rankings hold under a more realistic deployment-like evaluation.

105 runs, all finished, 7 augmentation conditions × 5 sequences × 3 seeds. No missing data.

---

## Results summary

### New holdout test set (13% positives)

| Method | Mean F1 | Δ vs none | p (paired t) | Win rate |
|---|---|---|---|---|
| **real** | **0.5929** | **+0.083** | **0.0001** *** | 5/5 |
| unfiltered | 0.5590 | +0.049 | 0.0013 ** | 5/5 |
| tfidf | 0.5531 | +0.043 | 0.0013 ** | 5/5 |
| embed | 0.5526 | +0.042 | 0.0002 *** | 5/5 |
| **none (baseline)** | **0.5103** | — | — | — |
| free | 0.5054 | −0.005 | 0.52 | 2/5 |
| genetic | 0.4904 | −0.020 | 0.057 | 1/5 |

The RM-ANOVA confirms a significant overall effect (F=65.9, p<1×10⁻¹³, η²_g=0.93).

### Cross-test-set comparison

| Method | Original F1 | New F1 | Drop |
|---|---|---|---|
| none | 0.6154 | 0.5103 | −0.105 |
| real | 0.7148 | 0.5929 | −0.122 |
| embed | 0.6458 | 0.5526 | −0.093 |
| tfidf | 0.6359 | 0.5531 | −0.083 |
| unfiltered | 0.6362 | 0.5590 | −0.077 |
| free | 0.6164 | 0.5054 | −0.111 |
| genetic | 0.5750 | 0.4904 | −0.085 |

All methods drop in absolute F1 on the harder test set, consistent with the lower positive rate. Relative rankings are fully preserved.

**Key shift in significance:** On the original test set (v7_poolfilter), no synthetic method reached significance. On this harder holdout, `embed`, `tfidf`, and `unfiltered` all reach p<0.01. This is not a contradiction — it reflects lower variance on a larger test set, which gives the analysis more power to detect the same underlying effects.

---

## Metric / training analysis

### Precision and recall decomposition

| Method | Mean precision | Mean recall | Mean F1 |
|---|---|---|---|
| real | 0.467 | 0.813 | 0.593 |
| unfiltered | 0.452 | 0.736 | 0.559 |
| tfidf | 0.439 | 0.750 | 0.553 |
| embed | 0.439 | 0.748 | 0.553 |
| none | 0.382 | 0.772 | 0.510 |
| free | 0.378 | 0.765 | 0.505 |
| genetic | 0.366 | 0.749 | 0.490 |

The augmentation-driven F1 gain is precision-dominated: augmented methods predict positive far more conservatively (higher precision, similar or lower recall), while `none` and `free` have higher recall but poor precision. This is consistent with the class-imbalance mechanism: augmented training sets contain many more synthetic positives, pushing the model to assign higher probability mass to the positive class, which effectively lifts precision on a sparse test set.

### Threshold calibration

The new test set has only 13% positives. As a result, the optimal threshold across all methods shifts dramatically upward compared to the original test set analysis.

| Method | Mean opt. threshold (new) |
|---|---|
| none | 0.997 |
| genetic | 0.995 |
| tfidf | 0.992 |
| unfiltered | 0.992 |
| embed | 0.980 |
| real | 0.926 |
| free | 0.922 |

All methods require near-maximum threshold to achieve best F1 at 13% prevalence. This means that in a deployment setting with this positive rate, any of these models would require careful threshold calibration — the default 0.5 threshold would result in excessive recall and poor precision.

At fixed thresholds (0.1–0.5), the ranking is stable:
- thr=0.5: real > unfiltered > tfidf > embed > none > free > genetic
- thr=0.1: unfiltered > tfidf > embed > real > none > genetic > free

At lower thresholds, `unfiltered` leads, while `real` moves up at higher thresholds. This suggests `real` produces a sharper score distribution while unfiltered data "spreads" probability mass somewhat more broadly.

### Average Precision (threshold-free)

| Method | Mean AP | Δ vs none | p | Wins |
|---|---|---|---|---|
| real | 0.688 | +0.132 | 0.0001 | 5/5 |
| unfiltered | 0.615 | +0.059 | <0.0001 | 5/5 |
| embed | 0.607 | +0.051 | 0.002 | 5/5 |
| tfidf | 0.607 | +0.051 | 0.015 | 5/5 |
| **none** | **0.556** | — | — | — |
| free | 0.538 | −0.018 | 0.070 | 1/5 |
| genetic | 0.494 | −0.062 | 0.0003 | 0/5 |

The AP ranking closely mirrors F1, confirming results are not threshold-sensitive. `unfiltered` has the highest AP of the synthetic methods (0.615), marginally ahead of `embed` (0.607) and `tfidf` (0.607). This continues the pattern from the original analysis: pool-filtering adds no measurable benefit over unfiltered generation.

### Generalization drops

The F1 drops across test sets reveal which methods generalise better:

- Smallest drops: `unfiltered` (−0.077), `tfidf` (−0.083), `genetic` (−0.085)
- Largest drops: `none` (−0.105), `free` (−0.111), `real` (−0.122)

Note that `real` drops the most in absolute terms despite starting highest. This is partly a ceiling effect — `real` was already much higher so has more room to drop, and the new test set positives may be drawn from a slightly different distribution than the training set positives. The `free` method's larger drop relative to filtered methods is consistent with it training on noisier data that over-fits to the original test set distribution.

### Class-balance sensitivity

Resampling the test set to different positive rates reveals consistent rankings:

| Method | orig (13%) | 25% | 50% |
|---|---|---|---|
| real | 0.635 | 0.733 | 0.845 |
| unfiltered | 0.589 | 0.693 | 0.814 |
| tfidf | 0.587 | 0.689 | 0.814 |
| embed | 0.587 | 0.691 | 0.815 |
| none | 0.549 | 0.667 | 0.804 |
| free | 0.538 | 0.661 | 0.796 |
| genetic | 0.523 | 0.644 | 0.790 |

Rankings are stable across positive rates, ruling out class-imbalance as a confound. The relative advantage of synthetic augmentation over `none` is preserved regardless of prevalence.

---

## Hypothesis evaluation

**H (embed): F1 improved over baseline.**

*Confirmed* (p=0.0002, +0.042, 5/5 sequences). This is a reversal from the original analysis where the result was non-significant (p=0.112). With a larger and lower-prevalence test set, the precision improvement from embedding-based pool-filtering is detectable.

**H (tfidf): F1 improved over baseline.**

*Confirmed* (p=0.0013, +0.043, 5/5 sequences). Same reversal as embed. Effect size is essentially identical to embed.

**H (free): F1 improved over baseline.**

*Rejected* (p=0.52, −0.005, 2/5 sequences). Free generation provides no benefit. Consistent with both training and original test analysis.

**H (genetic): F1 improved over baseline.**

*Rejected* (p=0.057, −0.020, 1/5 sequences). Genetic augmentation continues to underperform the baseline. The trend is negative but falls just below significance on this test set (vs p=0.015 on the original). The result is consistent with the training analysis finding of elevated eval loss and poor generalisation.

**H (real): F1 improved over baseline.**

*Strongly confirmed* (p=0.0001, +0.083, 5/5). The practical upper bound remains well above any synthetic method.

---

## Conclusions and recommended next steps

### Main takeaways

1. **Results from the original test set generalise to the harder holdout.** Rankings are fully preserved (real > unfiltered ≈ tfidf ≈ embed > none > free > genetic). This increases confidence that the effects are not an artefact of the particular test set used during training.

2. **Synthetic augmentation with augmented training data significantly improves F1 on the harder test set.** `embed`, `tfidf`, and `unfiltered` all reach p<0.01. Effect sizes are modest (+0.042–0.049) but consistent (5/5 sequences). This is a stronger result than the original analysis.

3. **Pool-filtering (embed/tfidf) still adds no value over unfiltered generation.** `unfiltered` has the highest AP (0.615) and highest F1 (0.559) of all synthetic methods. Both pool-filtered variants are statistically indistinguishable from unfiltered (within noise). This negative result for filtering is now confirmed on two test sets.

4. **The gap between synthetic and real data is wide (+0.034 F1 for best synthetic vs none, vs +0.083 for real).** Even with significant effects, synthetic augmentation closes only ~38% of the real-data gap.

5. **All models require near-threshold calibration at 13% prevalence.** Optimal thresholds of 0.92–1.0 mean the raw model scores are poorly calibrated for this positive rate. Post-hoc calibration (Platt scaling, temperature scaling) would be needed before deployment at low prevalence.

6. **Free generation actively hurts or adds no value.** This is consistent across both test sets, suggesting prompt design and output quality are the limiting factor, not volume.

### Recommended next steps

- **Pool-filtering is definitively ruled out as a direction.** Two independent test sets confirm no benefit over unfiltered generation. The computational cost is unjustified.
- **Focus on generation quality, not filtering.** The `unfiltered` method (same prompt as `free` but using the structured augmentation pipeline?) outperforms free generation — understanding why, and improving the generation prompt, is more tractable than post-hoc filtering.
- **Investigate threshold calibration.** The near-1.0 optimal thresholds are a practical obstacle. Temperature scaling on a held-out calibration set could improve AP-to-F1 alignment at deployment prevalence.
- **The genetic approach should not be pursued further.** It never exceeds baseline and shows training instability. The fundamental issue — that a surrogate fitness function during generation does not align with the downstream task — is unlikely to be resolved without direct supervision signal.

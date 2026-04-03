# Analysis: v7_poolfilter_large_eval
*Proposal: experiment_analysis/v7_poolfilter_large/eval_proposal.md*

> This analysis evaluates the same trained models from `v7_poolfilter_large` against a second, harder test set with ~13% positive rate (compared to ~26% in the original test set used during training). See `v7_poolfilter_large_analysis.md` for context on training-run results.

---

## Results summary

30 runs (5 sequences × 2 augmentation methods × 3 seeds), all finished with `test/f1` recorded.

| Method | Mean F1 | Δ vs none | p (t-test) | Win rate |
|---|---|---|---|---|
| none | 0.5694 | — | — | — |
| embed-multi | 0.5914 | +0.0220 | **0.027** | 100% (5/5) |

RM-ANOVA confirms a significant augmentation effect (F=11.6, p=0.027, η²=0.528). embed-multi significantly outperforms the baseline on this test set, winning all 5 sequences.

This is a reversal from the original training test set (~26% positive rate), where the fixed-threshold F1 gain was +0.014 and not significant (p=0.441). At the lower positive rate, the same models yield a larger and statistically significant improvement.

---

## Metric analysis

### Precision–Recall shift

| Method | Precision | Recall | P−R delta |
|---|---|---|---|
| none | 0.447 | 0.788 | −0.341 |
| embed-multi | 0.481 | 0.769 | −0.288 |

Both methods are strongly recall-heavy at 13% positive rate, as expected. embed-multi shifts toward precision (+0.034), consistent with the pattern observed on the 26% test set and in the extend experiment. The precision improvement is the primary driver of the F1 gain: at low base rates, false positives are costly and any movement toward precision translates directly into F1 improvement.

### Average Precision (AUC-PR)

| Method | Mean AP | Δ vs none | p | Wins |
|---|---|---|---|---|
| none | 0.6465 | — | — | — |
| embed-multi | 0.6670 | +0.0204 | **0.023** | 5/5 |

AP gain is significant and consistent (100% win rate). The pooled PR curve AP values are slightly lower (embed-multi: 0.630, none: 0.551) because pooling concentrates the metric in a single high-variance estimate; the per-run mean is more reliable.

### Optimal-threshold F1

| Method | Mean opt-F1 | Δ vs none | p | Wins |
|---|---|---|---|---|
| none | 0.6089 | — | — | — |
| embed-multi | 0.6222 | +0.0132 | **0.038** | 5/5 |

When the decision threshold is calibrated per-run, embed-multi retains a significant advantage. Optimal thresholds are very high for both methods (none: 0.939, embed-multi: 0.968), reflecting the conservatism needed at 13% positive rate: only the highest-confidence predictions should be accepted.

### Fixed-threshold ranking

| Method | thr=0.10 | thr=0.20 | thr=0.30 | thr=0.40 | thr=0.50 |
|---|---|---|---|---|---|
| embed-multi | **0.556** | **0.571** | **0.579** | **0.586** | **0.591** |
| none | 0.515 | 0.537 | 0.551 | 0.561 | 0.569 |

embed-multi ranks first at all fixed thresholds tested. The absolute gap is largest at low thresholds (0.041 at 0.1) and narrows slightly at high thresholds (0.022 at 0.5). This is the opposite pattern from the original test set where gains were only visible at calibrated thresholds — at 13% positive rate, the embed-multi precision advantage is sufficient to dominate even without threshold tuning.

### Class-balance sensitivity

| Method | orig (13.1%) opt-F1 | 13% opt-F1 | 25% opt-F1 | 50% opt-F1 |
|---|---|---|---|---|
| embed-multi | **0.6222** | **0.6222** | **0.7207** | **0.8314** |
| none | 0.6089 | 0.6086 | 0.7119 | 0.8308 |

embed-multi ranks first at all simulated positive rates. This contrasts with the original test set analysis (26%), where none was marginally better at lower simulated rates. The ranking reversal reflects the different base-rate regime: at 13%, embed-multi's precision bias is an asset at all rates, not a liability.

AP sensitivity shows the same pattern — embed-multi leads at 13%, 25%, and 50%.

### Seed variance

| Method | Mean std(F1) across seeds |
|---|---|
| embed-multi | 0.0088 |
| none | 0.0072 |

Seed variance is low and comparable across methods. This is a substantial improvement from the training test set analysis, where embed-multi seeds had mean std=0.0289. The higher variance on the original test set likely reflected threshold sensitivity at the ~26% regime; at 13% positive rate, threshold calibration is more deterministic and seed variance contracts.

---

## Comparison with original training test set (26% positive rate)

The same trained models behave differently across the two test sets:

| Metric | Original test (~26%, corrected) | This eval (~13%) |
|---|---|---|
| embed-multi F1 gain | +0.000, p=0.994 (n.s.) | +0.022, p=0.027 (**sig.**) |
| embed-multi opt-F1 gain | +0.006, p=0.537 (n.s.) | +0.013, p=0.038 (sig.) |
| AP gain | −0.002, p=0.893 (n.s.) | +0.020, p=0.023 (**sig.**) |
| Fixed-threshold ranking | embed-multi > none at all thresholds (small margins) | embed-multi > none at all thresholds |
| Class-balance sensitivity | none leads by AP at all rates | embed-multi leads at all rates |

The reversal is even starker after correction: on the 26% test, embed-multi provides **zero benefit** across all metrics. On the 13% test, all metrics are significant. The augmentation effect is entirely conditional on the evaluation setting — it only manifests when the task is hard enough (low base rate, more negatives to discriminate).

---

## Comparison with extend eval (128 real + 1024 synthetic, 13% test)

Both the large and extend experiments used the same 13% test set format. Scale comparison:

| Scale | Real | Synth | Baseline F1 | embed-multi F1 | Gain | p |
|---|---|---|---|---|---|---|
| Extend (15 seq) | 128 | 1024 | 0.508 | 0.554 | +0.046 | <0.0001 |
| Large (5 seq) | 512 | 4096 | 0.569 | 0.591 | +0.022 | 0.027 |

The same diminishing-returns pattern observed on the original test set holds here: as real data doubles, the baseline gains more (+0.061) than the augmented method (+0.037), compressing the gain. Both differences remain significant, but the effect shrinks. The gain halves from +0.046 to +0.022, consistent with augmentation filling a progressively smaller gap as real data grows.

Notably, at the extend scale the embed-multi precision bias was less beneficial than embed (the simpler method) on the 13% test (embed-multi AP=0.617, embed AP=0.623). At the larger scale, there is no embed-only baseline to compare against, but the pattern previously observed at small scale — where embed leads embed-multi by AP — may persist.

---

## Hypothesis evaluation

**H1 (embed-multi improves test F1 over none at 512 real + 4096 synthetic on the 13% test set):** **Confirmed.**

- Fixed-threshold (0.5) F1: +0.022, p=0.027 — significant
- Optimal-threshold F1: +0.013, p=0.038 — significant
- AP: +0.020, p=0.023 — significant
- Win rate: 100% (5/5 sequences)
- Fixed-threshold ranking: embed-multi leads at all thresholds

Unlike on the corrected original test set where no metric reaches significance, all three primary evaluation metrics confirm the hypothesis on this harder test set. The result is unambiguous.

### Per-sequence gains

| Seq | Baseline F1 (none) | embed-multi F1 | Gain |
|---|---|---|---|
| 3 | 0.551 | 0.591 | **+0.040** |
| 1 | 0.568 | 0.602 | **+0.033** |
| 4 | 0.583 | 0.600 | +0.017 |
| 2 | 0.576 | 0.592 | +0.016 |
| 0 | 0.569 | 0.573 | +0.004 |

Unlike the original test set, where gains were heavily concentrated in seq 4 (the weakest baseline), gains here are distributed more evenly and all are positive. The correlation between baseline strength and augmentation gain is weaker, suggesting the 13% test set rewards the embed-multi precision improvements more uniformly across sequences.

---

## Conclusion / recommended next steps

1. **embed-multi is significantly effective on the harder 13% test set at 512 real samples.** This is the cleaner result: all evaluation metrics align, win rate is 100%, and the result is not threshold-dependent. On the corrected 26% test set, embed-multi shows zero benefit — the 13% result is the only evidence of augmentation effectiveness at this data scale.

2. **The augmentation benefit requires a challenging evaluation setting.** F1 gains at 13% (+0.022) are significant while gains at 26% are zero. The mechanism: at lower base rates, each false positive is relatively more costly, so any systematic improvement in precision yields greater F1 gains. Augmentation's value is conditional on both data scarcity and task difficulty.

3. **Diminishing returns with real data volume hold on the 13% test set.** The gain halves from extend to large scale (+0.046 → +0.022). Augmentation provides greater marginal benefit when real data is scarce.

4. **Class-balance sensitivity is unconditionally positive on this test set.** embed-multi leads at all resampled rates (13%–50%).

5. **Seed variance is low and comparable across methods.** The high seed variance observed for embed-multi on the 26% test set (std=0.029 vs 0.012 for none) does not replicate here (0.009 vs 0.007). At 13% positive rate the decision boundary is less threshold-sensitive, leading to more stable results.

6. **For the thesis narrative:** The corrected results sharpen the picture: augmentation with embed-multi provides no benefit at 512 real samples when evaluated on the training-distribution test set (26%), but is clearly effective on the harder 13% test. This interaction between data scale, base rate, and augmentation benefit is the key finding at this experimental tier.

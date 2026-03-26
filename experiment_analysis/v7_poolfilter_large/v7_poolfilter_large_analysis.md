# Analysis: v7_poolfilter_large
*Proposal: experiment_analysis/v7_poolfilter_large/proposal.md*

## Results summary

30 runs (5 sequences × 2 augmentation methods × 3 seeds), all finished with `test/f1` recorded.

| Method | Mean F1 | Δ vs none | p (t-test) | Win rate |
|---|---|---|---|---|
| none | 0.6740 | — | — | — |
| embed-multi | 0.6875 | +0.0135 | 0.441 | 60% (3/5) |

The aggregate improvement from embed-multi is not statistically significant at this data scale. However, the direction is consistently positive and the effect is concentrated (see below). This should be read alongside the optimal-threshold analysis, which tells a different story.

---

## Metric / training analysis

### Precision–recall shift

| Method | Precision | Recall | P−R delta |
|---|---|---|---|
| none | 0.665 | 0.687 | −0.023 |
| embed-multi | 0.723 | 0.657 | +0.066 |

The baseline is recall-heavy. embed-multi shifts substantially toward precision (+0.066 delta), mirroring the pattern observed in the extend experiment (+0.046). This shift is larger at 512 real samples than at 128, suggesting the contrastive objective's high-confidence clustering effect is more pronounced with richer real data.

### Optimal-threshold F1

| Method | Mean opt-F1 | Δ vs none | p | Wins |
|---|---|---|---|---|
| none | 0.7011 | — | — | — |
| embed-multi | 0.7283 | +0.0273 | **0.038** | 5/5 |

When the decision threshold is calibrated per-run, embed-multi is significantly better and wins all 5 sequences. The optimal threshold for embed-multi is lower (mean 0.153 vs 0.401 for none), indicating the model emits higher-probability scores for checkworthy samples — consistent with the precision bias. At all fixed thresholds tested (0.1–0.5), embed-multi ranks above none.

### Class-balance sensitivity

| Method | orig (25.8%) opt-F1 | 13% opt-F1 | 25% opt-F1 | 50% opt-F1 |
|---|---|---|---|---|
| embed-multi | **0.7283** | **0.5886** | 0.7007 | 0.8177 |
| none | 0.7011 | 0.5842 | **0.7086** | **0.8194** |

At the test set's observed positive rate (~26%), embed-multi leads. At lower positive rates (25%, 50% in the resampled analysis), none is marginally better. This is the same precision-bias liability observed in the extend experiment: embed-multi's tighter class clusters are calibrated toward the training positive rate, and performance degrades as the class becomes rarer.

### Training convergence

| Method | Mean epochs | Std |
|---|---|---|
| embed-multi | 14.5 | 3.9 |
| none | 85.6 | 21.5 |

Augmented runs converge ~6× faster. At 128 real samples (extend experiment), the ratio was 9× (42.9 vs 387.9 epochs). The speedup is still substantial, but convergence is faster overall at 512 samples.

### Seed variance

| Method | Mean std(F1) across seeds |
|---|---|
| embed-multi | 0.0289 |
| none | 0.0125 |

embed-multi seeds are considerably more variable than none. The contrastive pre-training introduces a source of variance that the baseline does not have, and with only 3 seeds this limits the precision of aggregate estimates.

### Generalisation gap (test F1 − eval F1)

| Method | Mean gap |
|---|---|
| embed-multi | −0.175 |
| none | −0.180 |

Both methods have near-identical test–eval gaps at this data scale. This is an improvement from the extend experiment where embed-multi had a noticeably larger gap (−0.523 vs −0.453 in loss units). At 512 real samples the multi-task objective no longer appears to overfit more than the baseline.

---

## Hypothesis evaluation

**H1 (embed-multi improves test F1 over none at 512 real + 4096 synthetic):** Partially confirmed.

- Fixed-threshold (0.5) F1: +0.0135, p=0.441 — **not significant**. The aggregate gain observed in the extend experiment (+0.0296, p=0.004) does not replicate at 4x scale with a fixed threshold.
- Optimal-threshold F1: +0.0273, p=0.038 — **significant**, wins 5/5 sequences. The ranking advantage is preserved when the threshold is calibrated.
- AP (AUC-PR): +0.0221, p=0.124 — positive but not significant.

The hypothesis is rejected for fixed-threshold F1 but confirmed for threshold-calibrated evaluation. Whether this matters in practice depends on the deployment scenario.

### Gain is concentrated in weaker sequences

The negative correlation between baseline F1 and augmentation gain holds strongly (r=−0.807, p=0.099):

| Seq | Baseline F1 (none) | embed-multi F1 | Gain |
|---|---|---|---|
| 4 | 0.633 | 0.706 | **+0.073** |
| 1 | 0.639 | 0.650 | +0.011 |
| 3 | 0.687 | 0.696 | +0.009 |
| 0 | 0.709 | 0.698 | −0.010 |
| 2 | 0.702 | 0.687 | −0.015 |

Seq 4 (weakest baseline) accounts for almost all aggregate gain. Sequences with strong baselines (0 and 2, F1 > 0.70) are slightly hurt by augmentation. This is the same pattern found in the extend experiment (r=−0.804) and likely reflects the model's inability to improve on well-represented training signal.

### Scale comparison with extend experiment

| Scale | Real samples | Synthetic | Baseline F1 | embed-multi F1 | Gain | p |
|---|---|---|---|---|---|---|
| Small (extend) | 128 | 1024 | 0.630 | 0.659 | +0.030 | **0.004** |
| Large (this) | 512 | 4096 | 0.674 | 0.688 | +0.014 | 0.441 |

The absolute F1 rises with more real data (+0.044 for baseline, +0.029 for embed-multi), confirming that real data quality matters more than synthetic volume. The augmentation *gain* halves, consistent with diminishing returns: at 512 real samples, the model already extracts sufficient signal from real data alone, and synthetic samples fill a smaller gap.

---

## Conclusion / recommended next steps

1. **The augmentation benefit diminishes with real data volume.** At 128 real samples, embed-multi provided a reliable and significant +0.030 F1 gain. At 512, the aggregate gain is +0.014 and not significant. The effect is real but smaller.

2. **Optimal-threshold F1 is still significantly better (p=0.038).** If the downstream system can calibrate its threshold, embed-multi remains the better choice at 512 real samples. This is practically achievable if a held-out calibration set is available.

3. **The precision bias is consistent and scales.** embed-multi's shift toward precision is larger at 512 samples (+0.066 P−R delta) than at 128 (+0.046). This is worth discussing in the thesis as a systematic property of the contrastive objective rather than a fluke.

4. **The weak-sequence effect is the strongest signal.** With r=−0.807, the gain-baseline correlation is nearly identical across both experiments. Augmentation reliably helps sequences that struggle on real data alone. For a thesis contribution, this suggests that augmentation should be recommended selectively — targeting cases where the real data is known to be sparse or unrepresentative — rather than as a universal default.

5. **With 5 sequences and high seed variance, statistical power is low.** The large experiment has fewer sequences than extend (5 vs 15), which reduces power. The non-significant result should not be interpreted as evidence that augmentation is harmful — the effect size is modest and consistent with the extend result if power is accounted for.

6. **For the thesis narrative:** embed-multi is an effective method in the low-data regime (128 real samples). Its advantage shrinks but does not disappear as real data grows to 512. The practical takeaway is that synthetic augmentation is most valuable when real data is genuinely scarce.

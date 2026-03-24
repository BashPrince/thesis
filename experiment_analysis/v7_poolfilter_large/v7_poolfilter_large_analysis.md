# Analysis: v7_poolfilter_large
*Proposal: experiment_analysis/v7_poolfilter_large/proposal.md*

## Results summary

All 30 runs finished successfully. 5 dataset sequences × 2 augmentation conditions × 3 seeds = 30 runs. Every combo has exactly 3 seeds (complete).

Scale: 512 real + 4096 synthetic (embed-multi) vs 512 real baseline (none). This is a 4× scale-up from the previous confirmatory experiment (128 real + 1024 synthetic).

**Mean test F1 by method (across 5 sequences, 3 seeds each, n=15 runs per method):**

| Method      | Mean F1 | Std F1 | Mean diff vs none | Paired t (n=5 seqs) | p-value |
|-------------|---------|--------|-------------------|---------------------|---------|
| none        | 0.6740  | 0.0336 | —                 | —                   | —       |
| embed-multi | 0.6875  | 0.0336 | +0.0135           | t=0.855             | p=0.441 |

The difference is not statistically significant (p=0.441).

**Per-sequence mean F1:**

| Sequence | none   | embed-multi | Gain    | Winner      |
|----------|--------|-------------|---------|-------------|
| seq_0    | 0.7087 | 0.6984      | −0.0104 | none        |
| seq_1    | 0.6392 | 0.6500      | +0.0108 | embed-multi |
| seq_2    | 0.7019 | 0.6866      | −0.0153 | none        |
| seq_3    | 0.6872 | 0.6964      | +0.0091 | embed-multi |
| seq_4    | 0.6328 | 0.7058      | +0.0731 | embed-multi |

**Win rate:** embed-multi wins 3/5 sequences (60%), loses 2/5.

**Additional metrics:**

| Method      | Mean precision | Mean recall | Mean accuracy | Mean eval/F1 |
|-------------|---------------|-------------|---------------|--------------|
| none        | 0.6646        | 0.6871      | 0.8289        | 0.8535       |
| embed-multi | 0.7229        | 0.6573      | 0.8185        | 0.8620       |

embed-multi shifts toward higher precision (+0.058) at the cost of lower recall (−0.030), consistent with the precision-bias observed in the previous experiment's `extend-multi` method (the equivalent approach at smaller scale).

## Metric/training analysis

### Training modes

- `none` runs: `training_mode=classification`, standard cross-entropy fine-tuning.
- `embed-multi` runs: `training_mode=multi`, multi-task loss combining classification and supervised contrastive loss (`multi_alpha=0.2`, `contrastive_temperature=0.07`, `contrastive_proj_type=mlp`, `contrastive_proj_dim=128`, `contrastive_pooling=mean`, `contrastive_balanced_sampling=True`).

Both conditions trained up to 1000 epochs with early stopping on eval F1.

### Seed variance

| Method      | Mean seed std (F1) per sequence |
|-------------|--------------------------------|
| none        | 0.0125                         |
| embed-multi | 0.0289                         |

embed-multi has more than twice the seed variance of the baseline (0.029 vs 0.013). Individual per-sequence seed stds for embed-multi: seq_0=0.043, seq_1=0.002, seq_2=0.049, seq_3=0.028, seq_4=0.018. The baseline is notably stable (all seq stds between 0.008 and 0.017). The high variance in embed-multi for seq_0 and seq_2 coincides with the two sequences where embed-multi underperforms the baseline — the variance there conceals one high-F1 outlier seed (e.g., seq_0: 0.6713, 0.7485, 0.6753; the 0.7485 seed alone is higher than the baseline mean of 0.7087, but the other two seeds drag the mean below).

### Precision-recall shift

embed-multi is precision-biased (precision 0.723, recall 0.657; gap = +0.065) while none is recall-biased (precision 0.665, recall 0.687; gap = −0.022). This matches the pattern seen in the previous experiment where `extend-multi` (functionally the same as `embed-multi`) produced a large precision increase with a recall cost. The shift is slightly larger here than in the previous experiment (P−R gap: +0.065 vs +0.046 in v7_poolfilter_extend).

### seq_4 is the dominant driver of the aggregate gain

seq_4 shows a large gain of +0.073, while the four other sequences show gains of −0.015 to +0.011. Without seq_4, the aggregate mean gain across the remaining four sequences is:
- mean gain = (−0.0104 + 0.0108 − 0.0153 + 0.0091) / 4 = −0.0015

The aggregate positive result (+0.0135) depends almost entirely on seq_4. The seq_4 baseline is the lowest of any sequence (0.6328), which may indicate a more difficult dataset where the precision correction from the contrastive loss is beneficial.

### Comparison with previous experiment (v7_poolfilter_extend, 128+1024 scale)

| Metric              | v7_poolfilter_extend (extend-multi, n=15 seqs) | v7_poolfilter_large (embed-multi, n=5 seqs) |
|---------------------|-----------------------------------------------|---------------------------------------------|
| Baseline mean F1    | 0.6296                                        | 0.6740                                      |
| Method mean F1      | 0.6592                                        | 0.6875                                      |
| Mean gain           | +0.0296                                       | +0.0135                                     |
| Paired t p-value    | p=0.004 (significant)                         | p=0.441 (not significant)                   |
| Win rate            | 87% (13/15)                                   | 60% (3/5)                                   |

Notes on the comparison:
1. The baseline F1 is higher in the large-scale experiment (0.674 vs 0.630), which is expected: 512 real samples provides a stronger no-augmentation baseline than 128, leaving less room for improvement.
2. The absolute gain is roughly half (0.013 vs 0.030) and is no longer statistically significant.
3. The small n=5 sequences limits power: with only 5 paired observations, a t-test requires a very large effect to reach significance. The result is inconclusive rather than definitively negative.

## Hypothesis evaluation

The hypothesis is: *Test set F1 results of the model trained on the augmented dataset are improved over the baseline.*

**Not supported at this scale.** The mean gain of embed-multi over none is +0.0135, but this is not statistically significant (p=0.441, paired t-test on per-sequence means, n=5). embed-multi wins on 3/5 sequences and loses on 2/5. The aggregate positive result is heavily influenced by a single sequence (seq_4, +0.073); the remaining four sequences show a near-zero mean gain (−0.0015).

The higher baseline F1 at 512 real samples (0.674 vs 0.630) is consistent with less headroom for augmentation to provide benefit — the model already has more signal from the larger real dataset.

The increased seed variance in embed-multi (mean std 0.029 vs 0.013 for none) suggests the contrastive multi-task loss introduces optimization instability at this scale that was less prominent in the smaller experiment.

## Conclusion / recommended next steps

**Main finding:** At 512 real + 4096 synthetic scale, embed-multi does not produce a statistically significant or consistent improvement over the 512-sample baseline. The +0.013 aggregate mean gain fails to replicate the significant +0.030 gain seen with the same method at 128+1024 scale. The improvement seen in the previous experiment does not straightforwardly extend when both real and synthetic data are scaled 4×.

**Possible explanations:**
1. **Diminishing returns from augmentation at higher real-data scale.** With 512 real samples, the model is less starved for signal. The embedding-filtered synthetic data adds proportionally less to what is already a richer training set.
2. **Higher baseline makes the improvement harder to detect with n=5 sequences.** The prior experiment had 15 sequences (3× more paired observations), giving enough power to detect a smaller effect. Five sequences cannot distinguish +0.013 from noise.
3. **seq_4 confound.** The single sequence with a low baseline (0.633) accounts for almost all of the observed aggregate gain. The other four sequences show negligible improvement. This may indicate the method is selectively beneficial when the real data is limited or unrepresentative.

**Recommended next steps:**
1. **Do not conclude the method fails at larger scale without more sequences.** n=5 is underpowered for the effect size observed here. Adding more dataset sequences (5–10 additional) would substantially clarify whether the +0.013 gain is real or noise.
2. **Investigate seq_4 specifically.** Its unusually low baseline and large gain from embed-multi may reveal conditions under which augmentation is most useful (e.g., low class balance, high topic specificity, poor recall in the real sample).
3. **Reconsider whether 4× real data scaling reduces augmentation utility monotonically.** Testing an intermediate scale (e.g., 256 real + 2048 synthetic) could confirm whether there is a crossover point where the augmentation benefit disappears.
4. **Address embed-multi seed variance.** The doubled seed variance relative to baseline is a practical concern. Investigating whether `multi_alpha` or temperature tuning can stabilise the multi-task optimisation would be valuable before further scaling.

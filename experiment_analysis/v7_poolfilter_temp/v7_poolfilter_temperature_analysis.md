# Analysis: v7_poolfilter_temperature
*Proposal: experiment_analysis/v7_poolfilter_temp/proposal.md*

## Results summary

All 45 runs completed successfully (15 per temperature condition, 5 sequences × 3 seeds each).

**Per-temperature mean F1 (macro, test set):**

| Temperature | Mean F1 | Std F1 | Min F1 | Max F1 |
|-------------|---------|--------|--------|--------|
| 0.5         | 0.6602  | 0.0280 | 0.6069 | 0.7179 |
| 1.0 (baseline) | 0.6746 | 0.0371 | 0.6014 | 0.7205 |
| 1.25        | 0.6750  | 0.0314 | 0.6122 | 0.7285 |

Temperature 1.0 and 1.25 are nearly identical in mean F1 (difference: 0.0004). Temperature 0.5 is lower by 0.0144 relative to the baseline.

**Per-sequence mean F1:**

| Seq | temp=0.5 | temp=1.0 | temp=1.25 |
|-----|----------|----------|-----------|
| 0   | 0.6727   | 0.6818   | 0.6849    |
| 1   | 0.6481   | 0.6951   | 0.6747    |
| 2   | 0.6227   | 0.6217   | 0.6248    |
| 3   | 0.6954   | 0.7182   | 0.6944    |
| 4   | 0.6622   | 0.6563   | 0.6962    |

The pattern is not consistent across sequences. For seq=1 and seq=3, temp=1.0 scores highest. For seq=4, temp=1.25 scores highest by a notable margin (0.040 over baseline). For seq=2, all three temperatures perform nearly identically. Temperature 0.5 wins on no sequence.

## Metric and training analysis

**Statistical tests (paired t-test, paired by sequence, n=5):**

| Comparison | Mean diff | t | p |
|------------|-----------|---|---|
| temp=0.5 vs temp=1.0 | −0.0144 | −1.514 | 0.205 |
| temp=1.25 vs temp=1.0 | +0.0004 | 0.033 | 0.975 |
| temp=0.5 vs temp=1.25 | −0.0148 | −2.174 | 0.095 |

No comparison reaches p < 0.05. The difference between temp=0.5 and temp=1.25 approaches marginal significance (p=0.095) but does not cross the threshold with n=5 sequences.

**Win rate vs baseline (temp=1.0):**
- temp=0.5: 2 wins, 3 losses (40%)
- temp=1.25: 3 wins, 2 losses (60%)

**Contrastive training dynamics:**

| Temperature | Mean final cosim_gap | Mean max cosim_gap | Mean final cosim_neg | Mean training steps |
|-------------|---------------------|-------------------|---------------------|---------------------|
| 0.5         | 0.688               | 0.713             | 0.305               | 2680                |
| 1.0         | 0.741               | 0.771             | 0.255               | 3783                |
| 1.25        | 0.696               | 0.734             | 0.294               | 3644                |

Temperature 1.0 achieves the highest cosim_gap and the lowest cosim_neg, and runs the most training steps before early stopping. Temperature 0.5 trains the fewest steps and yields a smaller separation between positive and negative pairs.

Cosim_gap shows essentially no correlation with downstream test F1 (r=0.007 for final gap, r=0.058 for max gap), consistent with the known two-stage mismatch noted in the project.

**Seed variance:**

Mean within-condition seed std (averaged over sequences):
- temp=0.5: 0.0162
- temp=1.0: 0.0186
- temp=1.25: 0.0195

Seed variance is slightly higher at temp=1.25, but the differences are small relative to the between-sequence variance (which dominates).

## Hypothesis evaluation

**Hypothesis: F1 performance on the test set is affected by temperature.**

The data do not support this hypothesis at a statistically significant level. No pairwise comparison between temperatures reaches p < 0.05. The largest observed difference is between temp=0.5 and the baseline (−0.0144 mean F1), but this fails to reach significance with n=5 sequences (p=0.205). Temp=1.25 is effectively identical to temp=1.0 in terms of mean F1 (+0.0004, p=0.975).

The magnitude of the temperature effect, where present at all, is smaller than the between-sequence variance (which ranges from approximately 0.01 to 0.05 within a given temperature condition). The direction of the effect is not consistent across sequences.

## Conclusion and recommended next steps

**Conclusion:** Generation temperature in the range 0.5–1.25 does not meaningfully affect downstream classifier F1. The baseline temperature of 1.0 performs as well as or better than the alternatives on most sequences, and temp=1.25 produces essentially the same result as temp=1.0 overall. Temperature 0.5 shows a small, marginally non-significant decrease in performance.

This experiment does not identify temperature as a useful lever for improving augmentation quality in this setup.

**Recommended next steps:**

1. **Do not pursue temperature tuning further.** The effect size is too small and inconsistent to justify additional experimentation along this axis. Retain temperature=1.0 as the generation setting.

2. **Investigate the between-sequence variance.** Seq=2 consistently underperforms all other sequences across all temperatures (mean F1 ~0.62 vs. ~0.67–0.72 elsewhere). Understanding what makes seq=2 harder may be more informative than tuning generation hyperparameters.

3. **Address the cosim_gap / F1 mismatch.** Contrastive training metrics (cosim_gap, cosim_pos, cosim_neg) show no correlation with downstream F1. Future experiments should consider whether the contrastive pre-training stage is contributing any useful signal, or whether the classification stage is operating independently of the adapter initialization.

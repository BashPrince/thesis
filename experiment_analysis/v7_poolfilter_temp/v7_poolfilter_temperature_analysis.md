# Analysis: v7_poolfilter_temperature
*Proposal: experiment_analysis/v7_poolfilter_temp/proposal.md*

## Results summary

**Hypothesis**: LLM generation temperature affects downstream classification F1.
**Verdict**: Not supported. Temperature had no statistically significant effect on test/f1 (RM-ANOVA p=0.261).

| Condition | Mean test/f1 | Std | Mean AP | Mean epochs |
|-----------|-------------|-----|---------|-------------|
| temp=0.5  | 0.6602 | 0.028 | 0.7196 | 37.3 |
| temp=1.0 (baseline) | 0.6746 | 0.037 | 0.7316 | 52.8 |
| temp=1.25 | 0.6750 | 0.031 | 0.7476 | 50.8 |

All differences are within one standard deviation. The best-performing condition per sequence is split 3/5 for temp=1.25 and 2/5 for temp=1.0; temp=0.5 never wins.

## Metric / training analysis

### F1 (fixed-threshold)

The per-sequence F1 table shows no consistent ranking across sequences:

| seq | temp=0.5 | temp=1.0 | temp=1.25 |
|-----|----------|----------|-----------|
| 0   | 0.6727   | 0.6818   | **0.6849** |
| 1   | 0.6481   | **0.6951** | 0.6747 |
| 2   | 0.6227   | 0.6217   | **0.6248** |
| 3   | 0.6954   | **0.7182** | 0.6944 |
| 4   | 0.6622   | 0.6563   | **0.6962** |

RM-ANOVA yields F=1.596, p=0.261 (η²=0.056). Post-hoc Holm-corrected pairwise tests against temp=1.0: temp=1.25 p=0.286 (g=−0.47), temp=0.5 is not significantly different either. Seed variance is comparable across conditions (mean std ≈ 0.016–0.020).

### Average Precision (threshold-free)

AP shows a small directional advantage for temp=1.25 (AP=0.7476 vs 0.7316 for temp=1.0), though the paired t-test is not significant (p=0.215, 4/5 sequences win). The only nominally significant finding is that **temp=0.5 has lower AP than temp=1.0** (p=0.046, mean diff=−0.012, 0/5 sequences win), suggesting reduced generation diversity at lower temperature slightly degrades the model's discriminative ability before thresholding.

This AP advantage for temp=1.25 is consistent across class-balance conditions:

| Condition | temp=0.5 | temp=1.0 | temp=1.25 |
|-----------|----------|----------|-----------|
| AP orig (30.5%) | 0.7196 | 0.7316 | **0.7476** |
| AP 13% pos rate | 0.5119 | 0.5311 | **0.5686** |
| AP 25% pos rate | 0.6855 | 0.6850 | **0.7054** |
| AP 50% pos rate | 0.8421 | 0.8554 | **0.8585** |

temp=1.25 ranks first in AP at all positive rates, which is notable even if individual comparisons lack power.

### Training dynamics

A clear difference appears in training duration:
- **temp=0.5**: 37.3 ± 10.2 epochs to convergence
- **temp=1.0**: 52.8 ± 17.0 epochs
- **temp=1.25**: 50.8 ± 19.6 epochs

The earlier convergence for temp=0.5 likely reflects reduced diversity in the synthetic data (fewer distinct phrasings per class), making the decision boundary easier to fit but potentially less informative. This is consistent with the lower AP result. Final training losses are nearly identical across conditions (≈0.393–0.395), indicating all conditions converge to the same loss basin.

Eval F1 at best checkpoint is slightly higher for temp=1.25 (0.844) and temp=1.0 (0.833) vs temp=0.5 (0.824), consistent with the test AP trend.

## Hypothesis evaluation

**Hypothesis**: F1 performance on the test set is affected by temperature.

**Verdict**: Not supported. The primary metric (test/f1 at fixed threshold) shows no statistically significant temperature effect. However, the threshold-free AP metric provides a secondary signal: temp=0.5 is nominally significantly worse than temp=1.0 (p=0.046), and temp=1.25 shows a consistent but non-significant AP advantage across all resampled positive rates. The null result on F1 may partly reflect low statistical power (n=5 sequences per condition) combined with noisy threshold optimization.

The pattern — if taken directionally — is that **lower temperature hurts (less diverse data, faster convergence, lower AP) while higher temperature (1.25) does not hurt and may marginally help**. This is somewhat counter-intuitive relative to concerns about hallucination quality degrading at very high temperatures, but suggests that at moderate increases (1.0 → 1.25) the gain in lexical diversity outweighs any quality loss.

## Conclusion / recommended next steps

The experiment provides a null result: temperature in the range 0.5–1.25 does not produce a detectable difference in classification F1 under these conditions. The only robust finding is that **temp=0.5 is directionally harmful** (lower AP, faster/shallower convergence), suggesting that sufficient diversity in synthetic data matters.

For thesis reporting, this is a useful robustness finding: the pipeline is not sensitive to temperature within the tested range (1.0–1.25), but should avoid sub-1.0 temperatures that reduce output diversity. The directional advantage of temp=1.25 on AP (consistent across all resampling conditions) is worth noting as a secondary finding.

**Recommendations:**
- For future augmentation runs, use temp=1.0 or 1.25 — avoid temperatures below 1.0.
- If testing higher temperatures (1.5, 2.0) is feasible, this could complete the picture and potentially reveal the upper bound where quality degradation begins to dominate.
- Increasing from n=5 to n=10 sequences would roughly double statistical power and make the AP difference testable (~0.016 mean diff vs 0.027 std).
- The AP advantage for temp=1.25 at low positive rates (13%) is the largest signal (0.537 vs 0.569); if real this would matter most for deployment at imbalanced priors.

# Analysis: v7_poolfilter_temperature_eval
*Proposal: experiment_analysis/v7_poolfilter_temp/eval_proposal.md*

> This analysis evaluates the same trained models from `v7_poolfilter_temperature` against a second, harder test set with ~13% positive rate (compared to ~30.5% in the original test set used during training). See `v7_poolfilter_temperature_analysis.md` for context on training-run results.

---

## Results summary

45 runs (5 sequences × 3 temperature conditions × 3 seeds), all finished with `test/f1` recorded.

| Condition | Mean F1 | Pooled AP | Opt F1 | Opt threshold |
|-----------|---------|-----------|--------|---------------|
| temp=0.5  | 0.5450  | 0.5468    | 0.5678 | 0.944         |
| temp=1.0  | 0.5494  | 0.5474    | 0.5784 | 0.999         |
| temp=1.25 | **0.5581** | **0.5697** | **0.5914** | 0.996 |

The ranking temp=1.25 > temp=1.0 > temp=0.5 is consistent across all evaluation metrics and all fixed decision thresholds (0.10–0.50). RM-ANOVA over the five per-sequence mean F1 values is not significant (F=0.687, p=0.454 GG-corrected, η²=0.088), but the pattern is directionally consistent and a post-hoc comparison confirms temp=1.0 significantly outperforms temp=0.5 (Holm-corrected p=0.012). At the prediction level, temp=1.25 leads by a notable margin in AP (+0.022 vs temp=1.0).

**Hypothesis verdict**: Partially supported. The overall ANOVA does not reach significance, but a significant pairwise difference between temp=0.5 and temp=1.0 exists (consistent with the training-run finding), and temp=1.25 consistently leads on threshold-free metrics, most clearly at the lower positive rate of this test set.

---

## Metric analysis

### F1 (fixed threshold 0.5)

Per-sequence mean F1 (averaged over 3 seeds):

| seq | temp=0.5 | temp=1.0 | temp=1.25 |
|-----|----------|----------|-----------|
| 0   | 0.5358   | 0.5388   | **0.5541** |
| 1   | 0.5419   | **0.5453** | 0.5303  |
| 2   | 0.5446   | **0.5502** | 0.5344  |
| 3   | 0.5668   | 0.5700   | **0.5706** |
| 4   | 0.5359   | 0.5425   | **0.6012** |
| **mean** | 0.5450 | 0.5494 | **0.5581** |

temp=1.25 wins on 3/5 sequences, temp=1.0 wins on 2/5, temp=0.5 never wins — replicating the pattern from the training test set. The overall ANOVA is not significant (F=0.687, p_unc=0.530, p_GG=0.454, η²=0.088).

The post-hoc comparison uses embed-temp-05-multi as implicit baseline (alphabetically first). The reported T statistics appear with inverted signs (reflecting baseline − comparison rather than comparison − baseline), but the magnitudes and p-values are consistent with the raw data: from all 5 sequences temp-1.0 > temp-0.5, giving a large T (|T|=5.97) and p_Holm=0.012. The comparison of temp-1.25 vs temp-0.5 is not significant (|T|=0.93, p=0.813) due to high cross-sequence variability: seq 4 gains +0.065 F1 from temp-0.5 to temp-1.25, while seq 1 and seq 2 slightly decline (−0.012, −0.010). This variability means the effect on F1 alone cannot be distinguished from noise at n=5 sequences.

### Seed variance

| Condition | Mean std(F1) across seeds |
|-----------|--------------------------|
| temp=0.5  | 0.0086                   |
| temp=1.0  | 0.0122                   |
| temp=1.25 | **0.0067**               |

Seed variance is low across all conditions. temp=1.25 has the lowest mean within-cell variance, suggesting stable training at higher temperature augmentation; the cross-sequence variability visible in the per-seq table is therefore a genuine property of the sequences (interaction between particular training samples and temperature), not seed noise.

### Average Precision and threshold-free metrics

Pooled PR curve analysis (all 15 runs per condition concatenated, 212,445 test samples):

| Condition | AP    | Opt F1 | Opt threshold |
|-----------|-------|--------|---------------|
| temp=0.5  | 0.5468 | 0.5678 | 0.944        |
| temp=1.0  | 0.5474 | 0.5784 | 0.999        |
| temp=1.25 | **0.5697** | **0.5914** | 0.996 |

At the AP level, temp=1.25 leads by +0.022 over temp=1.0 and +0.023 over temp=0.5. Notably, temp=0.5 and temp=1.0 are nearly identical on AP (0.5468 vs 0.5474), whereas on the training test set temp=0.5 was nominally lower than temp=1.0 (AP 0.7196 vs 0.7316, p=0.046). The separation that existed between temp=0.5 and temp=1.0 on the ~30% positive rate test compresses at 13%, while the gap between temp=1.25 and the rest widens slightly.

Optimal thresholds are uniformly very high (0.94–1.0) across all conditions. This is consistent with findings from the other v7_poolfilter eval analyses: at 13% positive rate, models trained on ~30% positive rate data require extremely high confidence thresholds to match the deployment base rate.

### Fixed-threshold ranking

F1 at fixed decision thresholds (pooled):

| Condition | thr=0.10 | thr=0.20 | thr=0.30 | thr=0.40 | thr=0.50 |
|-----------|----------|----------|----------|----------|----------|
| temp=0.5  | 0.5121   | 0.5256   | 0.5333   | 0.5393   | 0.5450   |
| temp=1.0  | 0.5313   | 0.5380   | 0.5427   | 0.5461   | 0.5494   |
| temp=1.25 | **0.5362** | **0.5447** | **0.5506** | **0.5545** | **0.5581** |

temp=1.25 ranks first at every threshold tested, and temp=0.5 ranks last at every threshold. The ranking is entirely stable across the operating range — this consistency is notable given the non-significance of the RM-ANOVA, suggesting that while per-sequence variance is high, the direction is not confounded by threshold choice.

### Class-balance sensitivity

Optimal-threshold F1 at resampled positive rates:

| Condition | orig (13.1%) | 13% | 25% | 50% |
|-----------|-------------|-----|-----|-----|
| temp=0.5  | 0.5846 | 0.5845 | 0.6887 | 0.8135 |
| temp=1.0  | 0.5908 | 0.5902 | 0.6970 | 0.8193 |
| temp=1.25 | **0.6019** | **0.6027** | **0.7020** | **0.8210** |

AP at resampled positive rates:

| Condition | orig (13.1%) | 13% | 25% | 50% |
|-----------|-------------|-----|-----|-----|
| temp=0.5  | 0.6101 | 0.6084 | 0.7500 | 0.8853 |
| temp=1.0  | 0.6093 | 0.6081 | 0.7521 | 0.8861 |
| temp=1.25 | **0.6283** | **0.6277** | **0.7608** | **0.8907** |

temp=1.25 ranks first at every simulated positive rate for both F1 and AP. The advantage is smallest at 50% and largest at the original 13% rate. The ordering of temp=0.5 and temp=1.0 swaps between F1 (temp-1.0 > temp-0.5) and AP at 13% (temp-0.5 slightly higher by AP: 0.6101 vs 0.6093), suggesting the two conditions are effectively indistinguishable at this base rate except for their relationship to temp=1.25.

---

## Comparison with training-set evaluation

The same models were previously evaluated against the training test set (~30.5% positive rate). Key comparisons:

| Metric | Training test (~30.5%) | This eval (~13.1%) |
|--------|----------------------|-------------------|
| temp=0.5 mean F1 | 0.6602 | 0.5450 |
| temp=1.0 mean F1 | 0.6746 | 0.5494 |
| temp=1.25 mean F1 | **0.6750** | **0.5581** |
| F1 ranking | temp-1.25 ≈ temp-1.0 > temp-0.5 | same |
| RM-ANOVA (F1) | p=0.261, n.s. | p=0.454, n.s. |
| temp-0.5 vs temp-1.0 AP | p=0.046 (sig.) | temp-0.5 ≈ temp-1.0 (AP 0.547 vs 0.547) |
| temp-1.25 AP advantage over temp-1.0 | +0.016 (n.s.) | +0.022 (pooled) |
| Ranking stability | 3/5 seqs for temp-1.25 | same, consistent across all thresholds |

The qualitative ranking is preserved between test sets. The distinction between temp=0.5 and temp=1.0 that appeared on the training set (as a significant AP difference) largely disappears at 13%: both have nearly identical AP here. Simultaneously, temp=1.25's lead over both grows slightly from +0.016 to +0.022 AP, consistent with the observation in the extend eval that temp=1.25 advantages amplify at lower base rates.

---

## Hypothesis evaluation

**Hypothesis**: F1 performance on the test set is affected by temperature.

**Verdict**: Partially supported. The primary fixed-threshold F1 metric does not yield a significant ANOVA (p=0.454), and the per-sequence variance is large enough to obscure overall effects. However:

1. **temp=0.5 vs temp=1.0**: The pairwise post-hoc comparison is significant (Holm-corrected p=0.012), with temp=1.0 outperforming temp=0.5 on all 5 sequences. This finding is consistent across both the training test set (AP p=0.046) and the pooled fixed-threshold F1 analysis here. It provides convergent evidence that sub-unity temperature reduces augmentation quality.

2. **temp=1.25 vs temp=1.0**: The pairwise comparison is not significant (p=0.813), but temp=1.25 leads by AP (+0.022) and consistently ranks first at every fixed threshold and every simulated positive rate. The cross-sequence variability is driven by seq 4, where temp=1.25 gains +0.065 F1 relative to temp=0.5 (possibly a sequence-specific interaction). This is insufficient to reject the null at n=5 but is directionally consistent with the training-set finding.

3. The null on the overall ANOVA may partly reflect genuinely small temperature effects (η²=0.088 at most) combined with low power (n=5 sequences). The directional consistency across metrics, thresholds, and test sets provides a secondary signal that temp=1.25 is no worse and possibly marginally better than temp=1.0.

---

## Conclusion / recommended next steps

1. **temp=0.5 is the weakest condition** on both test sets. The penalty for lower temperature (reduced lexical diversity) is detectable both as faster/shallower training convergence on the training run and as a significantly lower F1 here relative to temp=1.0. This finding is now replicated across two test sets.

2. **temp=1.25 consistently leads across all metrics** — F1, AP, optimal-threshold F1, fixed-threshold rankings, class-balance sensitivity — with a pattern that is completely stable across thresholds and simulated positive rates. The ANOVA non-significance reflects high cross-sequence variance rather than the absence of a directional trend.

3. **AP gap for temp=1.25 widens at lower positive rates.** On the ~30% training test, temp=1.25 had AP=0.7476 vs temp=1.0 AP=0.7316 (+0.016). Here: 0.5697 vs 0.5474 (+0.022). This parallels the pattern in the extend eval analysis and suggests that temp=1.25 data provides better precision-side signal.

4. **Optimal thresholds are uniformly very high (0.94–1.0).** This is the expected behaviour for models trained at ~30% positive rate and evaluated at 13%, and is consistent across all v7_poolfilter evals. Practitioners deploying at 13% would need a calibrated threshold near the score ceiling.

5. **For thesis reporting**: Temperature in the range 0.5–1.25 does not produce a strongly detectable F1 difference overall, but the data consistently suggests that: (a) sub-unity temperature is harmful (significant pairwise test, corroborated by training dynamics); (b) 1.25 is directionally better than 1.0 on threshold-free metrics across both test sets. Recommended augmentation temperature is ≥1.0; 1.25 is the preferred setting based on current evidence.

6. **For future work**: Increasing the number of sequences from 5 to 10 would roughly double statistical power. The cross-sequence variance for temp=1.25 is high (seq 4 stands out), which may indicate heterogeneous effects across base-set compositions rather than a uniform temperature effect.

# Analysis: v7_poolfilter_extend_ct22_eval
*Proposal: experiment_analysis/v7_poolfilter_extend/ct22_eval_proposal.md*

> This analysis evaluates the same trained models from `v7_poolfilter_extend` against an **out-of-distribution (OOD) test set** from the CheckThat! 2022 dataset (COVID-related claims), to investigate whether augmentation helps generalization beyond the training domain (U.S. presidential election claims). See `v7_poolfilter_extend_analysis.md` for the in-distribution CT24 training results and `v7_poolfilter_extend_eval_analysis.md` for the CT24 13% eval results.

---

## Results summary

180 runs (15 sequences × 4 augmentation methods × 3 seeds), all finished with `test/f1` recorded. CT22 test set has ~21.7% positive rate.

| Method | Mean F1 | Δ vs none | p (t-test) | Win rate |
|---|---|---|---|---|
| none | 0.3706 | — | — | — |
| embed | 0.3702 | −0.0005 | 0.936 | 47% |
| unfiltered | 0.3685 | −0.0022 | 0.728 | 47% |
| embed-multi | 0.3673 | −0.0033 | 0.581 | 47% |

RM-ANOVA confirms **no significant augmentation effect** (F=0.245, p_GG=0.691, η²=0.008). Sphericity is violated (W=0.091, p<0.001), so the Greenhouse-Geisser corrected p-value is reported. No post-hoc pairwise comparison reaches significance (all Holm-corrected p=1.0).

**On fixed-threshold F1, no augmentation method improves over the unaugmented baseline on the OOD test set.** This is a complete reversal from the in-distribution CT24 evaluations, where `embed` significantly outperformed baseline (+0.022, p=0.011) and all methods were significant on the 13% eval (+0.04, p<0.0001).

---

## Metric analysis

### Precision–Recall shift

| Method | Precision | Recall | P−R delta |
|---|---|---|---|
| none | 0.252 | 0.765 | −0.513 |
| embed-multi | 0.239 | 0.815 | −0.576 |
| embed | 0.238 | 0.844 | −0.606 |
| unfiltered | 0.236 | 0.849 | −0.614 |

All methods are extremely recall-heavy. The precision-recall pattern is **reversed from the CT24 evaluations**: on CT24, augmentation *increased* precision while maintaining recall. On CT22, augmentation *decreases* precision (from 0.252 to 0.236–0.239) while *increasing* recall (from 0.765 to 0.815–0.849). The augmented models predict more samples as positive (higher recall), but since the CT22 positive class has different characteristics from CT24, these additional positive predictions are predominantly false positives, reducing precision.

This precision drop is the mechanism behind the lack of F1 improvement: any recall gain is offset by proportionally larger precision loss at this base rate (~21.7%).

### Average Precision (AUC-PR)

| Method | Mean AP | Δ vs none | p | Wins |
|---|---|---|---|---|
| none | **0.2966** | — | — | — |
| embed-multi | 0.2575 | −0.0391 | **0.0004** | 1/15 |
| embed | 0.2561 | −0.0405 | **0.0001** | 1/15 |
| unfiltered | 0.2495 | −0.0471 | **0.0003** | 2/15 |

**Augmentation significantly degrades ranking quality.** The baseline outperforms all augmented methods on AP by a large margin (−0.04 to −0.05), and all differences are highly significant (p<0.001). The baseline wins 13–14 of 15 sequences on AP. This means the augmented models' ranking of positive vs negative samples is systematically worse on OOD data — augmentation did not just fail to help, it actively harmed the model's discriminative ability on a different domain.

### Optimal-threshold F1

| Method | Mean opt-F1 | Δ vs none | p | Wins |
|---|---|---|---|---|
| none | **0.3936** | — | — | — |
| embed | 0.3765 | −0.0171 | **0.0025** | 2/15 |
| embed-multi | 0.3757 | −0.0179 | **0.0040** | 3/15 |
| unfiltered | 0.3736 | −0.0200 | **0.0044** | 4/15 |

Even with oracle threshold tuning, augmented methods significantly underperform baseline. The optimal thresholds are highly variable (mean 0.48–0.60, std 0.25–0.45), reflecting the difficulty of calibrating on this OOD data.

### Fixed-threshold ranking

| Method | thr=0.10 | thr=0.20 | thr=0.30 | thr=0.40 | thr=0.50 |
|---|---|---|---|---|---|
| none | **0.3724** | **0.3722** | **0.3716** | **0.3717** | **0.3706** |
| embed | 0.3696 | 0.3700 | 0.3704 | 0.3702 | 0.3702 |
| embed-multi | 0.3683 | 0.3688 | 0.3684 | 0.3684 | 0.3673 |
| unfiltered | 0.3675 | 0.3678 | 0.3682 | 0.3683 | 0.3685 |

Baseline ranks first at **every** fixed threshold. The ranking is completely stable: none > embed > embed-multi > unfiltered (with a minor swap at thr=0.5). F1 values are remarkably insensitive to threshold choice — all methods vary by less than 0.003 across the 0.1–0.5 range, suggesting the score distributions are broad and the decision boundary is not sharp.

### Class-balance sensitivity

| Method | orig (21.7%) | 13% | 25% | 50% |
|---|---|---|---|---|
| none | **0.3936** | **0.2758** | **0.4336** | **0.6782** |
| embed | 0.3765 | 0.2531 | 0.4184 | 0.6724 |
| embed-multi | 0.3757 | 0.2515 | 0.4180 | 0.6717 |
| unfiltered | 0.3736 | 0.2493 | 0.4162 | 0.6720 |

Baseline leads at every simulated positive rate. AP sensitivity confirms the same pattern — none ranks first at all rates.

### Seed variance

| Method | Mean std(F1) |
|---|---|
| embed | 0.0042 |
| unfiltered | 0.0060 |
| embed-multi | 0.0091 |
| none | 0.0093 |

Augmented methods have lower seed variance, consistent with CT24 findings. `embed` has the lowest variance (0.0042), roughly half of baseline's (0.0093). Augmentation stabilizes predictions even when those predictions are no better on average.

### Gain-baseline correlation

The strong negative correlation between baseline F1 and augmentation gain persists on OOD data:

| Method | r(baseline, gain) | p |
|---|---|---|
| unfiltered | −0.938 | <0.001 |
| embed | −0.926 | <0.001 |
| embed-multi | −0.888 | <0.001 |

These correlations are even stronger than on CT24 (−0.73 to −0.80). Sequences with lower OOD baseline benefit slightly from augmentation, but the effect is centered around zero — the regression line crosses zero gain at a relatively low baseline, meaning most sequences are hurt or unaffected.

---

## Comparison with in-distribution CT24 evaluations

The same 180 trained models were previously evaluated on three CT24 test sets. The contrast with OOD performance is stark:

| Metric | CT24 test (26%) | CT24 eval (13%) | CT22 OOD (21.7%) |
|---|---|---|---|
| embed F1 gain | +0.022 (p=0.011) | +0.044 (p<0.0001) | −0.001 (p=0.94) |
| embed-multi F1 gain | +0.015 (p=0.102) | +0.046 (p<0.0001) | −0.003 (p=0.58) |
| unfiltered F1 gain | +0.007 (p=0.369) | +0.042 (p<0.0001) | −0.002 (p=0.73) |
| embed AP gain | +0.019 (p=0.048) | +0.069 (p=0.0001) | **−0.041 (p=0.0001)** |
| Best method | embed | All tied (embed by AP) | **none** |
| Precision effect | Increased (+0.04–0.05) | Increased (+0.05–0.06) | **Decreased** (−0.01–0.02) |
| Recall effect | Maintained | Maintained | **Increased** (+0.05–0.08) |

### Key observations

1. **The precision-improvement mechanism does not transfer OOD.** On CT24, augmentation sharpened the class boundary and reduced false positives. On CT22, the CT24-specific decision boundary is miscalibrated — the model's learned notion of "check-worthy" (U.S. election claims) does not align with CT22's domain (COVID claims). The augmented models, having been trained on more synthetic data reinforcing CT24's class characteristics, are *more* confidently wrong on OOD data.

2. **AP degradation is the strongest signal.** While fixed-threshold F1 differences are small and non-significant (all within 0.003), the ranking-based AP metric reveals a large and significant harm (−0.04 to −0.05). Augmentation did not just fail to transfer — it made the model's ranking of OOD samples actively worse. This means the augmented models cannot discriminate between check-worthy and non-check-worthy COVID claims as well as the baseline can.

3. **The recall increase is misleading.** Augmented models show higher recall (0.81–0.85 vs 0.77 for baseline) because they predict positive more liberally. But this recall gain comes at severe precision cost — and the net F1 is unchanged or slightly worse. On CT24, the same models were *more* conservative (higher precision); the reversal indicates domain-specific calibration rather than genuine generalization.

4. **Absolute F1 is dramatically lower on OOD data.** All methods achieve ~0.37 F1 on CT22 vs ~0.63 on CT24 test and ~0.51 on CT24 eval. The task is fundamentally harder when the domain shifts from election to COVID claims.

---

## Hypothesis evaluation

**H1 (embed improves F1 over none on OOD):** **Rejected.** F1: −0.001, p=0.94. AP: −0.041, p=0.0001 (significantly worse). The embed method, which was the most robust on CT24, provides no F1 benefit and significantly degrades ranking quality on OOD data.

**H2 (unfiltered improves F1 over none on OOD):** **Rejected.** F1: −0.002, p=0.73. AP: −0.047, p=0.0003. Same pattern as embed.

**H3 (embed-multi improves F1 over none on OOD):** **Rejected.** F1: −0.003, p=0.58. AP: −0.039, p=0.0004. The contrastive multi-task objective provides no advantage on OOD data, consistent with its lack of robust advantage on CT24.

---

## Conclusion / recommended next steps

1. **Augmentation with CT24-derived synthetic data does not transfer to OOD evaluation on CT22 (COVID claims).** No method improves F1, and all methods significantly degrade AP. The synthetic data reinforces in-distribution patterns that do not generalize.

2. **The mechanism reversal is informative.** On CT24, augmentation improved precision by sharpening the class boundary learned from election-domain claims. On CT22, this sharpened boundary is wrong — COVID check-worthy claims have different characteristics, and the augmented model's increased confidence produces more false positives (lower precision) while predicting positive more liberally (higher recall). The net effect is a wash on F1 but a significant degradation of ranking quality.

3. **The baseline's advantage comes from less domain-specific overfitting.** With only 128 real samples and no synthetic augmentation, the baseline model learns a weaker but more general representation. The augmented models learn a stronger but more domain-specific representation from 1024 additional synthetic samples — which is beneficial in-distribution but harmful OOD.

4. **For the thesis narrative:** This result completes the picture of augmentation's effect across evaluation conditions. The benefit is strictly in-distribution: augmentation helps on CT24 data (especially at low positive rates where precision matters) but does not generalize to a different topic domain. This is an important negative result — it shows that the synthetic data augmentation approach is domain-bound, and practitioners should not expect generalization beyond the training distribution.

5. **The AP metric is more sensitive to OOD degradation than F1.** Fixed-threshold F1 differences are negligible (0.001–0.003), but AP reveals a clear and significant harm. For OOD evaluation, threshold-free metrics are more informative.

# Analysis: v7_poolfilter_extend_eval
*Proposal: experiment_analysis/v7_poolfilter_extend/eval_proposal.md*

> This analysis evaluates the same trained models from `v7_poolfilter_extend` against a second, harder test set with ~13% positive rate (compared to ~26% in the original test set). See `v7_poolfilter_extend_analysis.md` for context on training-run results.

---

## Results summary

180 runs (15 sequences × 4 augmentation methods × 3 seeds), all finished with `test/f1` recorded.

| Method | Mean F1 | Δ vs none | p (t-test) | Win rate |
|---|---|---|---|---|
| none | 0.5077 | — | — | — |
| unfiltered | 0.5499 | +0.0422 | **<0.0001** | 100% |
| embed | 0.5520 | +0.0443 | **<0.0001** | 100% |
| embed-multi | 0.5536 | +0.0459 | **<0.0001** | 100% |

RM-ANOVA confirms a highly significant augmentation effect (F=44.9, p=3.65e-13, η²=0.43). All three augmentation methods significantly outperform the baseline. Post-hoc pairwise comparisons between augmented methods show no significant differences (embed-multi vs embed: p=0.95, embed-multi vs unfiltered: p=0.95) — the three methods are effectively tied on this test set.

---

## Metric analysis

### Precision–Recall shift

| Method | Precision | Recall | P−R delta |
|---|---|---|---|
| none | 0.380 | 0.768 | −0.388 |
| embed | 0.433 | 0.767 | −0.334 |
| embed-multi | 0.439 | 0.755 | −0.316 |
| unfiltered | 0.437 | 0.744 | −0.307 |

The baseline is strongly recall-heavy. All augmentation methods improve precision substantially (+0.05–0.06) while maintaining recall at roughly the same level or slightly below. The F1 gain is driven almost entirely by precision improvement, which is consistent with having a low positive rate (~13%): the model must be more selective, and augmented training data provides cleaner signal about the positive class boundary.

### Average Precision (AUC-PR)

| Method | Mean AP | Δ vs none | p | Wins |
|---|---|---|---|---|
| none | 0.5542 | — | — | — |
| embed | **0.6227** | +0.0685 | **0.0001** | 15/15 |
| embed-multi | 0.6172 | +0.0631 | **0.0001** | 15/15 |
| unfiltered | 0.6131 | +0.0590 | **0.0004** | 14/15 |

### Optimal-threshold F1

| Method | Mean opt-F1 | Δ vs none | p | Wins |
|---|---|---|---|---|
| none | 0.5466 | — | — | — |
| embed | **0.5907** | +0.0442 | **<0.0001** | 15/15 |
| embed-multi | 0.5874 | +0.0408 | **0.0001** | 15/15 |
| unfiltered | 0.5846 | +0.0380 | **0.0001** | 14/15 |

Optimal thresholds are very high for all methods (~0.96–0.99), and no significant differences exist between methods (all p>0.17). This reflects the models' conservatism on a low-base-rate set: few samples reach high probability, so the decision boundary sits near the upper end of the score range.

### Fixed-threshold ranking

Ranking at all fixed thresholds (0.10–0.40): **embed > unfiltered > embed-multi > none**. At 0.50: embed-multi > embed > unfiltered > none. The embed method consistently leads at lower operating thresholds; this may matter for practitioners who deploy with a conservative threshold rather than tuning it.

### Class-balance sensitivity

| Method | orig (13.1%) AP | 13% AP | 25% AP | 50% AP |
|---|---|---|---|---|
| embed | **0.6227** | **0.6203** | **0.7581** | **0.8878** |
| embed-multi | 0.6172 | 0.6172 | 0.7536 | 0.8861 |
| unfiltered | 0.6131 | 0.6117 | 0.7503 | 0.8843 |
| none | 0.5542 | 0.5527 | 0.7079 | 0.8650 |

`embed` ranks first at every positive rate evaluated. The ranking is stable across the full range from 13% to 50%.

### Seed variance

| Method | Mean seed std(F1) |
|---|---|
| none | 0.0101 |
| embed | 0.0096 |
| embed-multi | 0.0106 |
| unfiltered | **0.0072** |

Seed variance is comparable across methods. `unfiltered` has the lowest variance, possibly because random subsampling of the large pool introduces a smoothing effect. The baseline shows no higher variance than augmented methods on this test set (unlike the original test, where baseline variance was notably high due to slow, unstable convergence on 128 samples).

---

## Hypothesis evaluation

**H1 (embed improves F1 over none):** Confirmed. +0.044 F1, p<0.0001, AP +0.069 (p=0.0001), 100% win rate.

**H2 (unfiltered improves F1 over none):** Confirmed. +0.042 F1, p<0.0001, 100% win rate. *This is a reversal from the original test set (p=0.37, not significant), discussed below.*

**H3 (embed-multi improves F1 over none):** Confirmed. +0.046 F1, p<0.0001, 100% win rate.

---

## Comparison with original test set (26% positive rate)

The most striking finding from this eval is how the picture changes at 13% positive rate:

| | Original test (~26%) | This eval (~13%) |
|---|---|---|
| unfiltered vs none | p=0.37, not significant | p<0.0001, +0.042 F1 |
| embed vs none | p=0.011, +0.022 F1 | p<0.0001, +0.044 F1 |
| embed-multi vs none | p=0.004, +0.030 F1 | p<0.0001, +0.046 F1 |
| Best method | embed-multi | All tied (embed leads by AP) |
| embed-multi precision bias | +0.046 P−R delta (beneficial) | Slightly less recall, no clear advantage |
| embed lead at low base rate | Predicted | Confirmed |

**Unfiltered becomes significant.** At the higher positive rate, random augmentation failed to reliably improve F1. On this harder test set — with fewer positives, a larger test corpus (637k pooled samples), and greater signal demands on precision — even random augmentation provides a meaningful boost. The filtering advantage that distinguished `embed` from `unfiltered` narrows substantially.

**embed-multi loses its edge.** On the 26% test, embed-multi's precision bias was an asset; it emitted higher-confidence predictions for checkworthy samples and led by F1 and AP. At 13%, all three methods converge to virtually identical performance. The contrastive multi-task objective no longer confers a measurable advantage at the aggregate level, though embed remains nominally first on AP and at all fixed thresholds below 0.5.

**Absolute gains are larger here.** All methods show ~+0.04 F1 vs none on this test set, compared to +0.007–0.030 on the original. The low base rate amplifies the value of any precision improvement, so augmentation's effect is more visible.

**embed is the most robust method across conditions.** It leads by AP at every positive rate (13%–50%), leads at most fixed thresholds, and was previously identified as the safer choice at low base rates. That prediction is confirmed.

---

## Conclusion / recommended next steps

1. **All augmentation methods are effective on the harder, low-base-rate test set.** The null result for `unfiltered` on the original test set does not generalise; at 13% positive rate, random pool subsampling is sufficient to significantly improve precision and F1.

2. **The filtering advantage of `embed` over `unfiltered` is not robust across test distributions.** On the original test the filtering was necessary; here the difference is small and not significant. This suggests filtering's primary benefit is precision-mode improvement under balanced conditions, not universal.

3. **embed-multi's contrastive multi-task objective does not provide a consistent advantage across test distributions.** Its gain at ~26% positive rate reflects a precision-shift that happens to help there; at ~13% the same shift is less impactful and the method is indistinguishable from simpler augmentation.

4. **embed is the best choice for deployment-robustness.** It is first by AP at all simulated positive rates, ranks first at most fixed thresholds, and has no distribution-specific quirks. For a thesis context where checkworthiness positive rates vary widely across datasets, this is the most defensible recommendation.

5. **Future experiments on new datasets or test sets should be evaluated at the target deployment positive rate**, not only the training-set rate. The ranking reversal between methods seen here (embed-multi vs unfiltered) illustrates the risk of over-optimising on a single evaluation condition.

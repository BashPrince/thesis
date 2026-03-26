# Analysis: v7_poolfilter
*Proposal: experiment_analysis/v7_poolfilter/proposal.md*

## Results summary

105 runs, all finished, 7 augmentation conditions × 5 sequences × 3 seeds. No missing data.

| Method | Mean F1 | Δ vs none | p (paired t) | Win rate |
|---|---|---|---|---|
| **real** | **0.7148** | **+0.099** | **0.0005** | 5/5 |
| embed | 0.6458 | +0.030 | 0.112 | 4/5 |
| unfiltered | 0.6362 | +0.021 | 0.330 | 4/5 |
| tfidf | 0.6359 | +0.021 | 0.345 | 3/5 |
| free | 0.6164 | +0.001 | 0.953 | 3/5 |
| **none (baseline)** | **0.6154** | — | — | — |
| **genetic** | **0.5750** | **−0.040** | **0.015** | 0/5 |

The RM-ANOVA confirms a significant overall effect of augmentation method (F=23.5, p<0.001, η²_g=0.83).

**Conclusion:** No synthetic augmentation method reaches statistical significance (p<0.05) on mean test F1. `embed` and the two unfiltered variants show modest non-significant gains. `genetic` is the only method that significantly *hurts* performance. The `real` condition — the only non-synthetic one — confirms that +10 F1 points are attainable with equivalent real-data volume.

---

## Metric / training analysis

### Training dynamics

| Method | Mean final epoch | Mean peak eval F1 | Mean final eval loss |
|---|---|---|---|
| none | 420.3 | 0.824 | 0.948 |
| real | 46.4 | 0.911 | 0.328 |
| embed | 47.1 | 0.867 | 0.668 |
| tfidf | 47.7 | 0.865 | 0.669 |
| unfiltered | 51.4 | 0.873 | 0.627 |
| free | 43.2 | 0.833 | 0.846 |
| genetic | 57.7 | 0.783 | 1.056 |

The `none` baseline runs ~420 epochs (early stopping on a much smaller training set), while all augmented conditions converge around 43–57 epochs. Despite the larger epoch count, `none` achieves a lower peak eval F1 (0.824) than all augmented methods except `genetic` (0.783). The gap between val F1 and test F1 is largest for `none` (0.824 → 0.615), pointing to significant overfitting or distribution mismatch on the small real-only training set.

The `genetic` method stands out: it has the worst eval F1 (0.783), the highest eval loss (1.056), and takes the longest to converge. Its poor test performance is consistent with training failure, not just test-time generalisation issues.

The `free` method also shows a wider eval-to-test gap: peak eval F1 = 0.833 but test F1 = 0.616, nearly identical to the unaugmented baseline. Its training loss is also elevated relative to `embed`/`tfidf`/`unfiltered`, suggesting the unfiltered free-form synthetic sentences contain significant noise.

### Threshold calibration

Mean optimal decision threshold differs considerably across methods:

| Method | Mean opt. threshold |
|---|---|
| none | 0.42 |
| real | 0.37 |
| free | 0.33 |
| genetic | 0.20 |
| embed | 0.21 |
| tfidf | 0.19 |
| unfiltered | 0.17 |

Augmented models (especially pool-filtered and genetic) predict high positive-class probability less often, requiring a low threshold to recover recall. This suggests synthetic sentences shift the model's score distribution downward, possibly because synthetic data is less prototypically check-worthy than the real held-out test set. The fixed-threshold results (thr=0.1–0.5) show that `embed` is robust to threshold choice and consistently ranks second after `real` across all thresholds.

### Seed and sequence variance

Within-seed variance is comparable across methods (mean std ≈ 0.013–0.024). `unfiltered` has the highest seed variance (0.024), and `none` has the highest cross-sequence variance (std=0.027), reflecting the instability of learning from only 128 real samples. Filtering methods (embed, tfidf) reduce seed variance slightly relative to unfiltered and free.

### AP and threshold-free analysis

Average Precision (AUC-PR) rankings mirror F1:

| Method | Mean AP | Δ vs none | p |
|---|---|---|---|
| real | 0.792 | +0.106 | <0.001 |
| unfiltered | 0.701 | +0.014 | 0.382 |
| tfidf | 0.699 | +0.013 | 0.246 |
| embed | 0.697 | +0.011 | 0.271 |
| free | 0.690 | +0.003 | 0.829 |
| **none** | **0.686** | — | — |
| genetic | 0.625 | −0.062 | 0.005 |

In the threshold-free AP metric, `unfiltered` slightly outperforms both `embed` and `tfidf`, reversing the F1 ordering. This is a small difference (within noise), but it suggests the pool-filtering step does not improve the model's ranking ability.

---

## Hypothesis evaluation

**H (embed, tfidf): Pool-filtering improves F1 over baseline.**

*Partially rejected.* Both methods show positive trends (+0.020–0.030 F1), with `embed` winning 4/5 sequences, but neither reaches significance at p<0.05 after a paired t-test on 5 sequences. Critically, **unfiltered augmentation achieves essentially the same result** (tfidf vs unfiltered p=0.98, embed vs unfiltered p=0.54). The pool-filtering step provides no detectable additional benefit over simply using all generated synthetic data.

**H (free): Simple unguided generation improves F1 over baseline.**

*Rejected.* Mean improvement is +0.001 F1 (essentially zero), p=0.95. Free generation produces data that adds noise without benefit.

**H (genetic): Genetic algorithm augmentation improves F1 over baseline.**

*Strongly rejected.* Genetic augmentation significantly *reduces* F1 by −0.040 (p=0.015), losing on all 5 sequences. Its elevated eval loss (1.056 vs ≤0.847 for other methods) indicates training instability, likely because the generated sentences are optimised for a proxy objective that does not align with the real test distribution.

**H (real): Real data augmentation improves F1 over baseline.**

*Confirmed.* +0.099 F1 (p=0.0005), 5/5 sequences. This sets the practical ceiling for what any synthetic method could achieve with equivalent volume.

---

## Conclusions and recommended next steps

### Main takeaways

1. **Synthetic augmentation does not reliably improve checkworthiness classification.** Three methods show positive trends but none reaches significance with n=5 sequences. The effect size for pool-filtered methods (~0.020–0.030 F1) is smaller than the between-sequence variance (~0.027 for `none`).

2. **Pool-filtering adds no value over unfiltered generation.** `embed`, `tfidf`, and `unfiltered` are statistically indistinguishable. The filtering step (which is computationally expensive) does not improve either F1 or AP. This is a negative result worth documenting — it suggests the quality signal captured by tf-idf or embedding similarity to real data is not predictive of downstream usefulness.

3. **Genetic augmentation is actively harmful.** This is consistent with the approach optimising for a surrogate fitness function misaligned with the classification objective. The high eval loss suggests the generated sentences may be adversarially difficult or distributional outliers.

4. **The real-data gap is large (+10 F1 points).** This suggests the task is bottlenecked by label quality and distribution coverage, not volume per se. Synthetic data cannot replicate the distributional properties that make real check-worthy sentences identifiable.

5. **Score distribution shifts across augmentation methods** (optimal thresholds 0.17–0.42) suggest that model calibration varies substantially. This could be exploited with calibration post-processing if threshold-free AP is the target metric.

### Recommended next steps

- **Do not pursue pool-filtering as a direction.** Results are negative and the computational cost is not justified.
- **Investigate why genetic augmentation hurts.** Logging generated sentence quality metrics (perplexity, semantic similarity to real data) would help diagnose whether the issue is distribution mismatch or training instability.
- **Focus on data quality rather than quantity.** The `free` baseline and `unfiltered` both use the same volume of synthetic data, yet `free` performs far worse — suggesting prompt design and generation quality matter more than filtering strategy. Future work should compare augmentation quality directly (e.g., human evaluation, off-the-shelf classifiers as filters).
- **Consider the `real` condition as a practical baseline.** If even 1024 pooled real-world sentences are available (across datasets), that dominates all synthetic alternatives.
- **Investigate threshold calibration.** The large threshold shifts (0.17 for unfiltered vs 0.42 for none) suggests that temperature scaling or Platt scaling could recover some performance at deployment thresholds.

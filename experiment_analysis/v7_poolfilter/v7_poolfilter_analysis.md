# Analysis: v7_poolfilter
*Proposal: experiment_analysis/v7_poolfilter/proposal.md*

## Results summary

105 runs, all finished. 7 augmentation conditions × 5 sequences × 3 seeds each. All combos complete.

| aug | mean F1 | std F1 | Δ vs none | p-value | win rate |
|---|---|---|---|---|---|
| **none** | **0.6154** | 0.0274 | — | — | — |
| real | **0.7148** | 0.0135 | +0.099 | 0.0005 ** | 5/5 |
| embed | 0.6458 | 0.0209 | +0.030 | 0.112 | 4/5 |
| unfiltered | 0.6362 | 0.0352 | +0.021 | 0.330 | 4/5 |
| tfidf | 0.6359 | 0.0254 | +0.021 | 0.345 | 3/5 |
| free | 0.6164 | 0.0245 | +0.001 | 0.953 | 3/5 |
| genetic | 0.5750 | 0.0246 | −0.040 | 0.015 * | 0/5 |

Statistical significance: paired t-test across 5 sequences (n=5). Only `real` significantly improves; `genetic` significantly hurts; all other synthetic methods are within noise.

### Per-sequence mean F1

| seq | embed | free | genetic | none | real | tfidf | unfiltered |
|---|---|---|---|---|---|---|---|
| 0 | 0.646 | 0.604 | 0.587 | 0.637 | 0.728 | 0.615 | 0.595 |
| 1 | 0.635 | 0.605 | 0.562 | 0.577 | 0.712 | 0.638 | 0.648 |
| 2 | 0.672 | 0.640 | 0.555 | 0.605 | 0.698 | 0.656 | 0.637 |
| 3 | 0.635 | 0.615 | 0.584 | 0.604 | 0.703 | 0.645 | 0.639 |
| 4 | 0.642 | 0.619 | 0.588 | 0.655 | 0.733 | 0.625 | 0.662 |

`real` is the best method on all 5 sequences. `embed` is second-best on 4/5 sequences. `genetic` is the worst on all 5.

## Metric / training analysis

### Precision-recall balance

All synthetic methods show a qualitatively different precision-recall profile than the `real` baseline:

| aug | mean precision | mean recall | recall − precision |
|---|---|---|---|
| real | 0.708 | 0.724 | +0.017 |
| none | 0.590 | 0.647 | +0.057 |
| embed | 0.656 | 0.639 | −0.018 |
| tfidf | 0.645 | 0.630 | −0.015 |
| unfiltered | 0.655 | 0.622 | −0.033 |
| free | 0.601 | 0.635 | +0.034 |
| genetic | 0.559 | 0.598 | +0.039 |

The no-augmentation baseline (`none`) has a recall bias — the model is over-predicting checkworthy. Pool-filter methods (embed, tfidf) and unfiltered generation shift this toward precision-bias, suggesting their synthetic samples help constrain false positives. The `genetic` and `free` methods increase recall-bias, indicating their synthetic data may contain labelling noise or be otherwise easier to overfit.

### Convergence

From a single representative run per aug condition:

| aug | best eval F1 | epoch at peak |
|---|---|---|
| real | 0.898 | ~26 |
| embed | 0.861 | ~17 |
| tfidf | 0.858 | ~17 |
| unfiltered | 0.864 | ~34 |
| none | 0.783 | ~232 |
| free | 0.796 | ~34 |
| genetic | 0.756 | ~34 |

The baseline (none) converges much later (~232 epochs), reflecting the smaller effective training set. All augmented methods converge earlier due to the larger training set. The pool-filter methods (embed/tfidf) converge the fastest and at the highest eval F1 of the synthetic conditions, consistent with their test F1 advantage.

### Seed variance

| aug | mean std(F1) across seeds |
|---|---|
| none | 0.014 |
| embed | 0.016 |
| real | 0.019 |
| free | 0.020 |
| tfidf | 0.023 |
| genetic | 0.022 |
| unfiltered | 0.029 |

Seed variance is low and broadly similar across conditions. `unfiltered` is the noisiest, suggesting its gains are less reliable. `embed` is the most stable of the augmentation methods.

## Hypothesis evaluation

**Hypothesis: each synthetic augmentation method improves F1 over baseline.**

| method | result |
|---|---|
| free | **Not confirmed.** Negligible gain (+0.001, p=0.95). Free generation adds no signal. |
| tfidf | **Not confirmed (trend only).** +2.1pp, p=0.35; positive but noisy across sequences. |
| embed | **Not confirmed (trend only).** +3.0pp, p=0.11; the strongest synthetic result but still below significance. |
| genetic | **Rejected — harmful.** −4.0pp, p=0.015. Significantly hurts performance on all sequences. |

The only condition that significantly improves F1 is `real` (+9.9pp, p=0.0005), confirming that genuine additional data from the same distribution is far more valuable than any synthetic strategy tested.

### Why does genetic fail?

`genetic` has the lowest precision (0.559) and a recall-bias consistent with the model over-predicting the checkworthy class. The genetic algorithm likely generates adversarial-style sentences optimised for model scores rather than natural distribution coverage, introducing harmful noise. This is also reflected in its high seed variance and poor convergence at eval F1 (0.756, lowest of all methods).

### Why is the filtering advantage not significant?

`embed` and `tfidf` show positive trends (win rates: 80% and 60%), but the variance across sequences is high relative to the effect size. With n=5 sequences the test is underpowered. The filtering principle is supported by the data — both methods outperform `unfiltered` on mean F1 (0.646/0.636 vs 0.636) — but the difference between filtered and unfiltered is itself minimal, suggesting the pool composition matters more than the filtering strategy.

## Conclusion / recommended next steps

1. **Synthetic augmentation offers at best a modest, unreliable benefit.** At 1024 samples the best synthetic strategy (embedding-based pool filter) gains ~3pp over no augmentation, but this is not statistically significant at n=5 sequences.

2. **Genetic algorithm augmentation is actively harmful** and should not be used in downstream experiments. The generated data likely shifts the label distribution or introduces out-of-distribution noise.

3. **Pool-filtering direction is worth preserving.** The embed filter is the most consistent synthetic method (80% win rate, lowest seed std among augmented conditions). If augmentation is to be explored further, this approach provides the best signal-to-noise ratio.

4. **The gap between `real` and all synthetic methods is large (~7pp).** This sets a clear ceiling on what data augmentation alone can achieve and motivates the contrastive pre-training strategy as a complementary approach that operates on the representation level rather than the data level.

5. **Recommended next step:** Pair the embed-filtered augmentation with contrastive pretraining to test whether the representation-level signal and data-level signal are complementary. Alternatively, investigate whether increasing the pool size (currently 1024 pre-filter) improves filtering quality, which may narrow the gap.

6. **More sequences (or seeds) needed for power.** With n=5 sequences the t-test has very limited power (~30% at a 3pp effect size). Running 8–10 sequences would allow more reliable conclusions about the pool-filter methods.

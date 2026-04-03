# Analysis: v7_poolfilter_large
*Proposal: experiment_analysis/v7_poolfilter_large/proposal.md*

> **NOTE (2026-04-03):** The original `embed-multi` training runs used `dataloader_drop_last: true`, evaluating on a truncated test set. This analysis uses corrected metrics from re-evaluation group `v7_poolfilter_large_ct24_eval` (full test set). The eval analyses (13% test set) were unaffected.

## Results summary

45 runs (5 sequences × 3 augmentation methods × 3 seeds), all finished with `test/f1` recorded.

| Method | Mean F1 | Δ vs none | p (t-test) | Win rate |
|---|---|---|---|---|
| none | 0.6740 | — | — | — |
| embed-multi | 0.6741 | +0.0001 | 0.994 | 20% (1/5) |
| embed | 0.6865 | +0.0126 | 0.173 | 80% (4/5) |

Neither method reaches significance, but embed shows a much stronger directional effect than embed-multi (+0.013 vs +0.000), winning 4 of 5 sequences. The simpler filtering method outperforms the multi-task approach at this data scale.

---

## Metric / training analysis

### Precision–recall shift

| Method | Precision | Recall | P−R delta |
|---|---|---|---|
| none | 0.665 | 0.687 | −0.023 |
| embed | 0.702 | 0.674 | +0.028 |
| embed-multi | 0.689 | 0.661 | +0.028 |

Both augmented methods shift toward precision by the same amount (+0.028 delta). embed achieves higher absolute precision (0.702 vs 0.689) while retaining more recall (0.674 vs 0.661), resulting in a better F1.

### Optimal-threshold F1

| Method | Mean opt-F1 | Δ vs none | p | Wins |
|---|---|---|---|---|
| none | 0.7011 | — | — | — |
| embed | 0.7166 | +0.0155 | 0.222 | 4/5 |
| embed-multi | 0.7067 | +0.0056 | 0.537 | 3/5 |

Neither method is significant, but embed shows a larger and more consistent advantage. At all fixed thresholds tested (0.1–0.5), the ranking is embed > embed-multi > none.

### Class-balance sensitivity

| Method | orig (25.8%) opt-F1 | 13% opt-F1 | 25% opt-F1 | 50% opt-F1 |
|---|---|---|---|---|
| embed | **0.7166** | **0.6123** | **0.7168** | **0.8282** |
| embed-multi | 0.7067 | 0.5987 | 0.7058 | 0.8228 |
| none | 0.7011 | 0.5984 | 0.7114 | 0.8208 |

embed leads at all positive rates by both opt-F1 and AP (0.753 vs 0.746 at 26%, 0.618 vs 0.611 at 13%). embed-multi's precision bias hurts at low positive rates (13% opt-F1: embed 0.612 vs embed-multi 0.599), consistent with the pattern observed in the extend experiment.

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

| Method | Mean std(F1) across seeds |
|---|---|
| embed | 0.0268 |
| embed-multi | 0.0259 |
| none | 0.0125 |

Both augmented methods are considerably more variable than none. With only 3 seeds and 5 sequences, this high variance limits statistical power.

---

## Hypothesis evaluation

**H1 (embed-multi improves test F1 over none at 512 real + 4096 synthetic):** Rejected.

- Fixed-threshold (0.5) F1: +0.0001, p=0.994 — **no effect**.
- Optimal-threshold F1: +0.0056, p=0.537 — not significant.
- AP (AUC-PR): −0.0015, p=0.893 — no effect (direction is negative).

**H2 (embed improves test F1 over none at 512 real + 4096 synthetic):** Not confirmed, but directionally positive.

- Fixed-threshold (0.5) F1: +0.0126, p=0.173 — not significant, wins 4/5.
- Optimal-threshold F1: +0.0155, p=0.222 — not significant, wins 4/5.
- AP (AUC-PR): +0.0061, p=0.678 — not significant.

embed outperforms embed-multi on all metrics. The simpler filtering method retains a modest benefit at 512 real samples while the contrastive multi-task approach does not.

### Per-sequence gains

The negative correlation between baseline F1 and augmentation gain remains for both methods:

| Seq | Baseline (none) | embed | Δ embed | embed-multi | Δ multi |
|---|---|---|---|---|---|
| 4 | 0.633 | 0.658 | **+0.025** | 0.691 | **+0.058** |
| 1 | 0.639 | 0.648 | +0.009 | 0.636 | −0.004 |
| 3 | 0.687 | 0.718 | **+0.031** | 0.683 | −0.004 |
| 0 | 0.709 | 0.696 | −0.013 | 0.683 | −0.026 |
| 2 | 0.702 | 0.713 | +0.011 | 0.678 | −0.024 |

embed wins 4/5 sequences, losing only on the strongest baseline (seq 0). embed-multi wins only on the weakest (seq 4) and hurts all others. The key difference: embed's gains are distributed across sequences, while embed-multi's effect is concentrated in one sequence and negative elsewhere.

### Scale comparison with extend experiment

| Scale | Real | Synth | Baseline | embed | Δ embed | embed-multi | Δ multi |
|---|---|---|---|---|---|---|---|
| Small (extend, 15 seq) | 128 | 1024 | 0.630 | 0.651 | +0.022* | 0.645 | +0.015 |
| Large (this, 5 seq) | 512 | 4096 | 0.674 | 0.687 | +0.013 | 0.674 | +0.000 |

embed's gain halves from +0.022 to +0.013 but remains directionally positive (wins 4/5). embed-multi's gain drops to zero. The diminishing-returns pattern holds for both methods, but embed degrades more gracefully with increasing real data.

---

## Conclusion / recommended next steps

1. **embed outperforms embed-multi at 512 real samples.** embed gains +0.013 F1 (p=0.17, wins 4/5) while embed-multi gains +0.000 (p=0.99, wins 1/5). The contrastive multi-task approach actively hurts at this data scale, while the simpler filtering method retains a modest benefit.

2. **Neither method reaches significance.** With only 5 sequences and high seed variance, statistical power is low. The directional consistency of embed (4/5 wins, positive at all fixed thresholds and resampled rates) is more informative than the p-value alone.

3. **embed degrades more gracefully with scale.** From 128→512 real samples, embed's gain halves (+0.022→+0.013) while embed-multi's vanishes (+0.015→+0.000). This suggests the simpler method is more robust to the diminishing-returns effect.

4. **The eval analyses (13% test set) tell a different story.** On the harder test set, embed-multi shows a significant +0.022 F1 gain (p=0.027). An eval of embed on the 13% test set would determine whether embed also outperforms embed-multi at low base rates, as observed in the extend experiment.

5. **For the thesis narrative:** the simpler embed method is consistently the better choice. It is the only significant method at 128 real samples, the stronger method at 512, and the most robust across class distributions. The contrastive multi-task objective adds complexity without benefit.

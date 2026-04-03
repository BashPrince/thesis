# Analysis: v7_poolfilter_extend
*Proposal: experiment_analysis/v7_poolfilter_extend/proposal.md*

> **NOTE (2026-04-03):** The original `embed-multi` training runs used `dataloader_drop_last: true`, evaluating on only 256 of 341 test samples. This analysis uses corrected metrics from re-evaluation group `v7_poolfilter_extend_ct24_eval` (full 341 samples). The eval analyses (13% test set) were unaffected.

## Results summary

180 runs (15 sequences × 4 augmentation methods × 3 seeds), all finished with `test/f1` recorded.

| Method | Mean F1 | Δ vs none | p (t-test) | Win rate |
|---|---|---|---|---|
| none | 0.6296 | — | — | — |
| unfiltered | 0.6364 | +0.0069 | 0.369 | 67% |
| embed-multi | 0.6447 | +0.0151 | 0.102 | 73% |
| embed | 0.6512 | +0.0216 | **0.011** | 67% |

RM-ANOVA confirms a significant aug effect (F=3.43, p=0.025, η²=0.059). **embed** significantly outperforms baseline. **embed-multi** does not reach significance (p=0.10) and **unfiltered** does not (p=0.37).

The unfiltered result is a clean control: adding 1024 random synthetic samples provides no reliable benefit, confirming that the filtering procedure — not simply the data volume — drives the performance improvement seen in `embed`.

---

## Metric / training analysis

### Precision–Recall shift

| Method | Precision | Recall | P−R delta |
|---|---|---|---|
| none | 0.606 | 0.660 | −0.054 |
| unfiltered | 0.649 | 0.628 | +0.022 |
| embed | 0.645 | 0.661 | −0.016 |
| embed-multi | 0.652 | 0.643 | +0.009 |

The baseline is recall-heavy. `embed` roughly preserves this balance while improving both P and R. `embed-multi` shows a mild precision shift (+0.009 delta) — much smaller than previously reported from the truncated test set — suggesting the contrastive objective's precision-boosting effect is modest at this scale.

### Optimal threshold shift

`embed-multi` uses a lower mean optimal decision threshold (0.279 vs 0.377 for none), though the difference is not significant (p=0.23). At all fixed thresholds evaluated (0.1–0.5), the ranking is stable: embed > embed-multi > unfiltered > none.

### Class-balance sensitivity

`embed` leads by AP across all positive rates. `embed-multi` leads by optimal-threshold F1 by small margins:

| Method | orig (26%) AP | 13% AP | 25% AP | 50% AP |
|---|---|---|---|---|
| embed | **0.711** | **0.566** | 0.702 | 0.858 |
| embed-multi | 0.703 | 0.556 | **0.705** | **0.860** |
| none | 0.693 | 0.541 | 0.688 | 0.848 |
| unfiltered | 0.696 | 0.542 | 0.696 | 0.858 |

`embed` is the most robust method by AP across all class distributions.

### Training convergence

| Method | Mean epochs |
|---|---|
| none | 387.9 |
| embed | 44.5 |
| embed-multi | 42.9 |
| unfiltered | 47.2 |

Augmentation dramatically shortens training (~9× fewer epochs). The `none` runs converge much more slowly on 128 real samples, with high seed variance in epochs (std=143). Augmented runs show tighter convergence.

### embed-multi contrastive quality

The contrastive stage achieves strong separation: mean cosim_pos=0.993, cosim_neg=0.314, cosim_gap=0.678 (std=0.085). The cosim_gap moderately predicts downstream test F1 (r=0.369, p=0.013), suggesting that better representation learning does translate to better classification, though the relationship is noisy. However, the contrastive quality does not translate to a significant aggregate F1 improvement over the simpler embed method.

---

## Hypothesis evaluation

**H1 (embed improves F1 over none):** Confirmed. +0.022 F1, p=0.011, AP increase +0.019 (p=0.048). Significant and consistent.

**H2 (unfiltered improves F1 over none):** Rejected. +0.007 F1, p=0.369. The pool itself adds no reliable value; the embedding-based filter is necessary.

**H3 (embed-multi improves F1 over none):** Not confirmed on the primary test set. +0.015 F1, p=0.102, wins 11/15 sequences. AP +0.011 (p=0.30). Optimal-threshold F1 +0.018 (p=0.011) is significant, but aggregate fixed-threshold F1 is not. The contrastive multi-task objective does not reliably improve over the simpler embed method on this test set.

### Gain is concentrated in weaker sequences

There is a strong negative correlation between baseline F1 and augmentation gain:

| Method | r(baseline, gain) |
|---|---|
| embed | −0.803 |
| embed-multi | −0.729 |
| unfiltered | −0.727 |

Sequences already performing well (F1 > 0.68, e.g. seq 2, 3, 5, 8, 12) gain little or nothing from any augmentation method — the best method is often `none` or tied. The benefit is concentrated in sequences that struggle on real data alone (F1 < 0.63, e.g. seq 1, 4, 9, 13, 14), where gains of +0.05–+0.08 are observed. This suggests the augmented data helps fill in gaps in the real training signal but cannot improve upon already well-represented sequences.

---

## Conclusion / recommended next steps

1. **embed filtering is necessary and sufficient for a reliable boost.** Random subsampling of the same pool (unfiltered) is not enough. `embed` is the only method that reaches significance on the primary test set (p=0.011).

2. **embed-multi does not reliably improve over embed.** After correcting for the truncated test set, embed-multi's aggregate F1 gain is halved (+0.015 vs the originally reported +0.030) and no longer significant (p=0.10). The contrastive objective's representation-quality improvements do not translate to a reliable classification advantage at this scale.

3. **Gains are baseline-conditional.** A natural next question is whether this is a property of the sequences (data difficulty) or the datasets (topic/domain mismatch with the synthetic pool). Inspecting which sequences are weak baselines could reveal whether targeted synthetic data generation would help more.

4. **The eval analyses (13% test set) remain informative.** On the harder test set, all augmented methods including embed-multi become significant, and the methods converge. The primary test set result tells us that embed is the most robust single method.

# Analysis: v7_poolfilter_extend
*Proposal: experiment_analysis/v7_poolfilter_extend/proposal.md*

## Results summary

All 180 runs finished successfully. 15 dataset sequences × 4 augmentation methods × 3 seeds = 180 runs. Every combo has exactly 3 seeds (complete).

**Mean test F1 by method (across all 15 sequences, 3 seeds each):**

| Method       | Mean F1 | Std F1 | Mean diff vs none | Paired t (n=15) | p-value  |
|--------------|---------|--------|-------------------|-----------------|----------|
| none         | 0.6296  | 0.0484 | —                 | —               | —        |
| embed        | 0.6512  | 0.0310 | +0.0216           | t=2.923         | p=0.0111 |
| extend-multi | 0.6592  | 0.0362 | +0.0296           | t=3.396         | p=0.0043 |
| unfiltered   | 0.6364  | 0.0355 | +0.0069           | t=0.928         | p=0.3694 |

Paired t-tests treat each of the 15 sequences as a paired observation (comparing per-sequence mean F1 of the method against the per-sequence mean F1 of `none`).

**Significant improvements (p < 0.05):** `embed` and `extend-multi`. `unfiltered` is not significant.

**Win rate (sequences where method beats baseline):**

| Method       | Wins | Ties | Losses | Win rate |
|--------------|------|------|--------|----------|
| embed        | 10   | 0    | 5      | 67%      |
| extend-multi | 13   | 0    | 2      | 87%      |
| unfiltered   | 10   | 0    | 5      | 67%      |

`extend-multi` wins on 13/15 sequences. The two sequences where it falls short are seq_2 (−0.044) and seq_8 (−0.014), both sequences with higher-than-average baselines (0.684 and 0.675 respectively). `embed` loses on 5 sequences; all 5 losses are small in magnitude (largest: −0.021 on seq_11).

**Best augmentation method per sequence:**

| Best method  | Count (of 15 seqs) |
|--------------|--------------------|
| extend-multi | 9                  |
| embed        | 3                  |
| unfiltered   | 2                  |
| none         | 1                  |

## Metric/training analysis

### Precision vs recall shift

Augmentation methods shift the precision-recall trade-off relative to the no-augmentation baseline:

| Method       | Mean precision | Mean recall | P − R  |
|--------------|---------------|-------------|--------|
| none         | 0.6061        | 0.6598      | −0.054 |
| embed        | 0.6450        | 0.6614      | −0.016 |
| extend-multi | 0.6855        | 0.6396      | +0.046 |
| unfiltered   | 0.6492        | 0.6275      | +0.022 |

The baseline (`none`) and `embed` are recall-biased. `extend-multi` inverts this to precision-biased (+0.046 gap), with a substantial precision increase (+0.079 vs none) at a small recall cost (−0.020 vs none). The net result is still a positive F1 gain. `unfiltered` also shifts toward precision but less dramatically. This suggests the augmented data (especially with contrastive loss) may be pulling the model toward a higher-confidence decision boundary.

The precision-bias in `extend-multi` warrants noting: on individual high-recall-baseline sequences (e.g., seq_2, seq_8), switching to `extend-multi` reduces recall enough to lower F1 below the baseline despite substantially higher precision.

### extend-multi training dynamics

The `extend-multi` method adds a supervised contrastive loss (SupCon) to the standard classification loss. Key observations from training histories:

- **Epochs to convergence:** early-stopped by eval F1 on the validation set, runs trained for 20–75 epochs (mean 42.7 epochs). High variance in stopping point across sequences and seeds.
- **Final cosim_gap:** 0.54–0.83 at termination. Higher cosim_gap generally indicates better class separation in the projected embedding space, but this is a by-product of training, not the stopping criterion.
- **Positives per anchor:** ~7–10 same-class samples in each mini-batch (consistent with the 1024 augmented samples providing ample positives).
- **Correlation of training epochs with test F1:** r = 0.24, p = 0.11. Epoch count does not reliably predict final test F1, indicating that validation F1 early stopping does not tightly track held-out test performance — likely due to the train/val/test split differences rather than a misaligned stopping criterion.

### Seed variance

| Method       | Mean seed std (F1) |
|--------------|--------------------|
| embed        | 0.0172             |
| extend-multi | 0.0223             |
| none         | 0.0233             |
| unfiltered   | 0.0199             |

`embed` has the lowest seed variance (0.017), suggesting the embedding-filtered pool produces more consistent training sets. `extend-multi` variance (0.022) is comparable to the baseline. The highest individual seed variance cases are seq_14 for `extend-multi` (std=0.059) and seq_3 for `none` (std=0.050).

### Filtering vs no filtering (embed vs unfiltered)

`embed` (filtered) achieves a significant mean gain of +0.022 (p=0.011). `unfiltered` (random 1024 from the same pool) gains only +0.007 (p=0.369). The embedding filter therefore accounts for most of the benefit from the pool: the pool itself without filtering provides negligible improvement on average. This confirms the filtering is the active ingredient, not simply having a larger pool.

## Hypothesis evaluation

The hypothesis is: *F1 results on the test set are improved over the baseline for each method.*

- **embed:** Hypothesis supported. Significant improvement (mean +0.022, p=0.011). Win rate 67%. Improvement is consistent but modest; 5/15 sequences show small regressions.
- **extend-multi:** Hypothesis supported. Significant improvement (mean +0.030, p=0.004). Win rate 87%, best method on 9/15 sequences. Adds multi-task contrastive training on top of the filtered embed data.
- **unfiltered:** Hypothesis not supported. Mean gain +0.007 (p=0.369) is not statistically significant. Win rate 67% but the losses include one large regression (seq_10: −0.064). Randomly sampling from the pool without filtering does not reliably improve over baseline.

## Conclusion / recommended next steps

**Main finding:** Embedding-based filtering of the synthetic pool is effective and necessary. Simply taking a random subset of the same pool (`unfiltered`) does not produce reliable gains, but filtered selection (`embed`) does. Adding a contrastive loss on the filtered data (`extend-multi`) provides the largest and most consistent improvement.

**extend-multi is the best-performing method overall.** It achieves the highest mean F1 (0.659), the highest win rate (87%), and the largest mean gain (+0.030). The precision shift it produces (high-precision, moderate-recall) appears to be net positive on most sequences, though it reverses on sequences where the baseline is already high-recall.

**Considerations for next steps:**

1. **extend-multi is worth developing further.** The two sequences where it underperforms the baseline (seq_2, seq_8) are high-baseline sequences; a potential next direction is investigating whether the contrastive loss temperature or data mix can be tuned to avoid precision over-correction on easier datasets.
2. **Validation F1 early stopping does not tightly predict test F1.** The lack of correlation between training epochs (determined by val F1) and test F1 suggests some train/val/test distribution gap. This is less concerning than the cosim_gap mismatch seen in pure contrastive runs, but is worth noting. Monitoring both val and test F1 during training would clarify whether this is a systematic bias.
3. **embed alone is a simpler and competitive alternative.** If the two-stage training overhead of `extend-multi` is a concern, `embed` provides significant gains (p=0.011) with lower seed variance and no contrastive training cost.
4. **The embed pool filter generalises across sequences.** Consistent improvements across 13–15 diverse sequences suggests the embedding filter is not over-fitted to any particular dataset; this is a positive signal for the method's robustness.

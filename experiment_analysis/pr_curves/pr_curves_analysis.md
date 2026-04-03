# Analysis: Mean Precision-Recall Curves

*Script: experiment_analysis/pr_curves/plot_pr_curves.py*

## Approach

For each run in `v7_poolfilter` (7 methods × 5 sequences × 3 seeds = 105 runs), we downloaded prediction artifacts and computed per-run PR curves. Curves were interpolated onto a common 200-point recall grid, averaged over seeds within each sequence, then aggregated as mean ± SE across the 5 sequences. The same models were evaluated on both the 26% and 13% test sets.

## Findings

### 26% test set (`v7_poolfilter_pr_curves.pdf`)

- **real** dominates at all recall levels (AP=0.792).
- **embed/tfidf/unfiltered** cluster together (AP 0.697–0.700), slightly above baseline (AP=0.686) in the mid-recall range (0.3–0.7), but with heavily overlapping SE bands.
- **genetic** (AP=0.625) is consistently below all methods. **free** (AP=0.689) tracks baseline.
- At high recall (>0.8), all synthetic methods converge with baseline — no method extends recall beyond what the baseline achieves. The separation is entirely in the precision axis.

### 13% test set (`v7_poolfilter_eval_pr_curves.pdf`)

- Separation is much more visible. embed/tfidf/unfiltered (AP 0.607–0.615) clearly separate from baseline (AP=0.556) across recall 0.1–0.8, with non-overlapping SE bands.
- At low recall, baseline and genetic/free start at ~0.85–0.90 precision while augmented methods reach ~0.95. This is the precision advantage amplified at low base rate: with more negatives in the test set, a slightly better model avoids many more false positives.
- The convergence at high recall persists — augmentation does not improve the maximum achievable recall.
- **real** remains well above all synthetic methods (AP=0.688), confirming real data improves both precision and recall.

### Key takeaway for thesis

The PR curves directly visualize the precision-only mechanism: synthetic augmentation shifts the PR curve **upward** (higher precision at the same recall) but does not shift it **rightward** (no new recall). This effect is subtle at 26% positive rate but clearly visible at 13%. The genetic method shifts the curve downward — its synthetic data actively degrades the model's ranking ability.

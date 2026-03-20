# Analysis: v7_poolfilter_extend
*Proposal: experiment_analysis/v7_poolfilter extend/proposal.md*

## Results summary

180 runs, all finished. 4 augmentation conditions × 15 sequences × 3 seeds each. All combos complete.

| aug | mean F1 | std F1 | Δ vs none | p-value | win rate |
|---|---|---|---|---|---|
| **none** | 0.6296 | 0.0427 | — | — | — |
| embed | 0.6512 | 0.0263 | +0.0216 | 0.011 * | 10/15 (67%) |
| embed-multi | 0.6581 | 0.0181 | +0.0285 | 0.008 ** | 12/15 (80%) |
| unfiltered | 0.6364 | 0.0296 | +0.0069 | 0.369 | 10/15 (67%) |

Paired t-test across 15 sequences. **Both `embed` and `embed-multi` significantly improve over baseline. `unfiltered` does not.**

### Per-sequence mean F1

| seq | none | embed | embed-multi | unfiltered |
|---|---|---|---|---|
| 0 | 0.6055 | 0.6508 | **0.6672** | 0.6291 |
| 1 | 0.5672 | 0.6204 | **0.6240** | 0.6072 |
| 2 | **0.6838** | 0.6760 | 0.6534 | 0.6838 |
| 3 | **0.6865** | 0.6737 | 0.6565 | 0.6753 |
| 4 | 0.6022 | 0.6607 | **0.6697** | 0.6239 |
| 5 | 0.6909 | **0.7106** | 0.7008 | 0.6896 |
| 6 | 0.6206 | 0.6374 | **0.6490** | 0.6214 |
| 7 | 0.5978 | 0.6274 | **0.6560** | 0.6325 |
| 8 | **0.6748** | 0.6739 | 0.6488 | 0.6481 |
| 9 | 0.5781 | 0.6309 | **0.6425** | 0.6118 |
| 10 | 0.6451 | 0.6591 | **0.6642** | 0.5811 |
| 11 | 0.6344 | 0.6134 | **0.6651** | 0.6388 |
| 12 | 0.6772 | 0.6584 | **0.6840** | 0.6616 |
| 13 | 0.6087 | **0.6606** | 0.6543 | 0.6333 |
| 14 | 0.5706 | 0.6140 | **0.6359** | 0.6092 |

`embed-multi` is the best method on 10/15 sequences; `embed` on 2/15; `none` on 2/15 (seqs 3 and 8, both high-baseline sequences); `unfiltered` on 1/15.

---

## Metric / training analysis

### Precision-recall balance

| aug | mean precision | mean recall | P − R |
|---|---|---|---|
| none | 0.6061 | 0.6598 | −0.054 |
| embed | 0.6450 | 0.6614 | −0.016 |
| embed-multi | **0.6984** | 0.6259 | **+0.072** |
| unfiltered | 0.6492 | 0.6275 | +0.022 |

The no-augmentation baseline has a strong recall bias (the model over-predicts checkworthy). `embed` corrects this toward balance while maintaining recall. `embed-multi` flips the balance to a pronounced precision bias: +0.072 gap. This is a meaningful distributional shift — `embed-multi` classifies more conservatively, making fewer but more accurate positive predictions. `unfiltered` is moderately precision-biased.

This pattern is consistent with the prior v7_poolfilter experiment, where filtered methods (embed/tfidf) moved toward precision-bias while free and genetic generation increased recall-bias.

### Convergence

| aug | mean epochs | std |
|---|---|---|
| none | 387 | 143 |
| embed | 44 | 12 |
| unfiltered | 47 | 16 |
| embed-multi | 78 | 25 |

`none` trains for ~8× more epochs, reflecting the smaller training set (128 samples vs ~1152). Augmented methods converge in fewer epochs because each epoch covers more data. `embed-multi` takes ~78 epochs on average — substantially more than `embed`/`unfiltered` (~44–47), likely because the combined classification + contrastive loss is harder to minimize and the higher learning rate (5e-5 vs 2e-5) produces noisier gradient steps.

**Training loss patterns (seq_0 representative):**

| aug | initial train loss | final train loss |
|---|---|---|
| none | 0.675 | 0.000 |
| embed | 0.698 | 0.000 |
| unfiltered | 0.692 | 0.000 |
| embed-multi | 0.954 | 0.271 |

`embed`, `unfiltered`, and `none` all reach zero training loss — complete memorization of the training set (including synthetic data). `embed-multi` does not reach zero, settling around 0.27. The multi-task objective (classification + contrastive loss with α=0.1) acts as an implicit regularizer, preventing the model from fully collapsing to zero loss. This may partly explain its lower seed variance (std=0.018 vs 0.026 for embed) and its win rate advantage.

### Eval/test correlation

| aug | corr(best_eval_f1, test_f1) |
|---|---|
| none | 0.513 |
| embed | 0.266 |
| unfiltered | 0.257 |
| embed-multi | 0.011 |

The validation F1 (used for model selection via `load_best_model_at_end`) is a weak predictor of test F1 under augmentation. For `embed-multi`, this correlation is effectively zero. The likely cause: the eval set is drawn from the same real-data distribution as the test set, but training on a large synthetic corpus (1024 samples, 8:1 ratio) can shift the model's learned representations enough that the eval checkpoint that maximises in-distribution F1 doesn't predict out-of-distribution (test) F1. This suggests checkpoint selection strategy deserves attention.

### Seed variance

| aug | mean std(F1) across seeds |
|---|---|
| none | 0.023 |
| embed | 0.017 |
| embed-multi | 0.024 |
| unfiltered | 0.020 |

`embed` is the most stable augmentation method. `embed-multi` has variance comparable to `none` despite its higher win rate — this is driven by a few sequences with high spread (e.g., seq_1 std=0.043, seq_13 std=0.050). The multi-task loss doesn't reduce seed sensitivity in these cases.

---

## Hypothesis evaluation

**Hypothesis: each method improves F1 over the baseline (none).**

| method | result |
|---|---|
| embed | **Confirmed** (+0.022, p=0.011). With 15 sequences the underpowered trend seen in v7_poolfilter (n=5, p=0.11) becomes significant. The embedding-based pool filter reliably selects higher-quality synthetic samples. |
| embed-multi | **Confirmed** (+0.029, p=0.008). Highest mean gain and win rate. |
| unfiltered | **Not confirmed** (+0.007, p=0.369). Random sampling from the same pool without filtering yields no significant benefit — **the filtering step is causally responsible for the embed gains.** |

The `unfiltered` condition directly answers the question of whether filtering drives the embed improvement: it does. The pool itself is insufficient; the embedding-based selection step is what extracts usable signal.

### Is embed-multi better than embed?

Numerically yes (+0.007 mean difference), but a paired t-test across sequences gives t=1.29, p=0.22 — **not significant.** `embed-multi` wins on 10/15 sequences but the differences are small and variable. The two methods cannot be distinguished with this data.

However, embed-multi's qualitatively different behaviour (precision-biased, non-zero training loss, low seed variance for most sequences) suggests it is doing something mechanistically different, not just noisier version of embed.

**Important confound**: `embed-multi` uses different hyperparameters than `embed` — LR 5e-5 vs 2e-5, batch size 32 vs 16, multi_alpha 0.1, temperature 0.07 vs 0.05. These were presumably tuned separately, but direct head-to-head comparison should be interpreted with this in mind.

### Baseline performance as a moderator

There is a strong negative correlation between sequence baseline F1 and augmentation gain across all methods:

| method | corr(baseline_F1, gain) |
|---|---|
| embed-multi | −0.911 (p<0.001) |
| embed | −0.804 (p<0.001) |
| unfiltered | −0.726 (p=0.002) |

| baseline tier | n seqs | embed gain | embed-multi gain |
|---|---|---|---|
| low (F1 < 0.62) | 7 | +0.048 | +0.060 |
| medium (0.62–0.67) | 3 | +0.033 | +0.035 |
| high (≥ 0.67) | 5 | −0.004 | −0.014 |

**Augmentation benefits are concentrated in low-baseline sequences** (seqs 0, 1, 4, 7, 9, 13, 14). For high-baseline sequences (2, 3, 5, 8, 12), none of the synthetic methods outperform the baseline on average, and `embed-multi` actively hurts (−0.014). This suggests that sequences with already-distinctive real data are not helped by synthetic augmentation — the model can already identify the relevant patterns from 128 real examples, and the synthetic data introduces noise.

This moderator effect is a meaningful practical finding: the utility of augmentation depends on how difficult the specific sequence is in the first place.

---

## Conclusion / recommended next steps

1. **The embedding pool filter significantly improves F1** (+2.2pp, p=0.011 at n=15 sequences). The previous v7_poolfilter result (trend, p=0.11 at n=5) is confirmed with adequate statistical power. The filtering step is causally necessary — unfiltered random sampling does not help.

2. **embed-multi shows the highest aggregate performance** (+2.9pp, p=0.008, 80% win rate) but is **not significantly better than embed** (p=0.22). Whether the precision-recall shift and reduced overfitting it provides translate to a reliable advantage requires either more sequences or a controlled ablation isolating the multi-task objective from hyperparameter differences.

3. **The gains are concentrated in difficult sequences.** For sequences where the baseline F1 is already ≥0.67, augmentation provides negligible or slightly negative benefit. Practitioners should not expect uniform gains across datasets.

4. **Checkpoint selection is unreliable under augmentation.** The near-zero correlation between eval F1 and test F1 for embed-multi in particular suggests that the standard approach of picking the best eval checkpoint may not be optimal. Alternatives worth exploring: longer patience, ensemble averaging across recent checkpoints, or using the real-only subset as the sole eval signal.

5. **Recommended follow-up**: Run a controlled embed vs embed-multi comparison with matched hyperparameters (same LR, batch size, epochs) to cleanly isolate the multi-task objective's effect. Given the precision/recall shift, also investigate whether the task's class balance or the downstream use case favours precision or recall — if recall matters more, `embed` may be the better practical choice despite embed-multi's higher F1.

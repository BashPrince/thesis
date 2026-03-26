# Analysis: v7_poolfilter_extend
*Proposal: experiment_analysis/v7_poolfilter_extend/proposal.md*

## Results summary

180 runs (15 sequences × 4 augmentation methods × 3 seeds), all finished with `test/f1` recorded.

| Method | Mean F1 | Δ vs none | p (t-test) | Win rate |
|---|---|---|---|---|
| none | 0.6296 | — | — | — |
| unfiltered | 0.6364 | +0.0069 | 0.369 | 67% |
| embed | 0.6512 | +0.0216 | **0.011** | 67% |
| embed-multi | 0.6592 | +0.0296 | **0.004** | 87% |

RM-ANOVA confirms a significant aug effect (F=6.84, p=0.00074, η²=0.114). **embed** and **embed-multi** both significantly outperform baseline. **unfiltered** does not (p=0.37).

The unfiltered result is a clean control: adding 1024 random synthetic samples provides no reliable benefit, confirming that the filtering procedure — not simply the data volume — drives the performance improvement seen in `embed`.

---

## Metric / training analysis

### Precision–Recall shift

| Method | Precision | Recall | P−R delta |
|---|---|---|---|
| none | 0.606 | 0.660 | −0.054 |
| unfiltered | 0.649 | 0.628 | +0.022 |
| embed | 0.645 | 0.661 | −0.016 |
| embed-multi | 0.686 | 0.640 | +0.046 |

The baseline is recall-heavy. `embed` roughly preserves this balance. `embed-multi` shifts substantially toward precision (+0.046 delta), consistent with its contrastive objective encouraging tighter, higher-confidence class clusters. This raises F1 overall but changes the operating point.

### Optimal threshold shift

`embed-multi` uses a significantly lower optimal decision threshold (mean 0.243 vs 0.377 for none, p=0.047), indicating the model emits higher raw probabilities for checkworthy samples. Practically, if a downstream system relies on a fixed 0.5 threshold, `embed-multi` gains the most; if the threshold is calibrated, the advantage narrows. At the fixed thresholds evaluated (0.1–0.5), the ranking is stable: embed-multi > embed > unfiltered > none.

### Class-balance sensitivity

At the original test positive rate (~26%), embed-multi leads. At a low positive rate (13%), `embed` overtakes embed-multi and none:

| Method | orig (26%) AP | 13% AP | 25% AP | 50% AP |
|---|---|---|---|---|
| embed | 0.711 | **0.566** | **0.704** | **0.859** |
| embed-multi | **0.729** | 0.534 | 0.692 | 0.839 |
| none | 0.693 | 0.538 | 0.684 | 0.847 |
| unfiltered | 0.696 | 0.537 | 0.688 | 0.852 |

embed-multi's precision bias becomes a liability in minority-class scenarios. `embed` is the more robust choice across class distributions.

### Training convergence

| Method | Mean epochs |
|---|---|
| none | 387.9 |
| embed | 44.5 |
| embed-multi | 42.9 |
| unfiltered | 47.2 |

Augmentation dramatically shortens training (~9× fewer epochs). The `none` runs converge much more slowly on 128 real samples, with high seed variance in epochs (std=143). Augmented runs show tighter convergence.

### embed-multi contrastive quality

The contrastive stage achieves strong separation: mean cosim_pos=0.993, cosim_neg=0.314, cosim_gap=0.678 (std=0.085). The cosim_gap moderately predicts downstream test F1 (r=0.369, p=0.013), suggesting that better representation learning does translate to better classification, though the relationship is noisy.

Test/eval generalisation gaps: embed (0.469) < none (0.453) < embed-multi (0.523). embed-multi shows the largest test–eval gap, which may reflect that the multi-task objective partially overfits to the validation split used for early stopping.

---

## Hypothesis evaluation

**H1 (embed improves F1 over none):** Confirmed. +0.022 F1, p=0.011, AP increase +0.019 (p=0.048). Significant and consistent.

**H2 (unfiltered improves F1 over none):** Rejected. +0.007 F1, p=0.369. The pool itself adds no reliable value; the embedding-based filter is necessary.

**H3 (embed-multi improves F1 over none):** Confirmed, and is the strongest single result. +0.030 F1, p=0.004, wins 9/15 sequences, AP +0.036 (p=0.003), optimal-threshold F1 +0.036 (p<0.001).

### Gain is concentrated in weaker sequences

There is a strong negative correlation between baseline F1 and augmentation gain:

| Method | r(baseline, gain) |
|---|---|
| embed | −0.804 |
| embed-multi | −0.735 |
| unfiltered | −0.726 |

Sequences already performing well (F1 > 0.68, e.g. seq 2, 3, 5, 8, 12) gain little or nothing from any augmentation method — the best method is often `none` or tied. The benefit is concentrated in sequences that struggle on real data alone (F1 < 0.63, e.g. seq 1, 4, 9, 13, 14), where gains of +0.05–+0.08 are observed. This suggests the augmented data helps fill in gaps in the real training signal but cannot improve upon already well-represented sequences.

---

## Conclusion / recommended next steps

1. **embed filtering is necessary and sufficient for a reliable boost.** Random subsampling of the same pool (unfiltered) is not enough.

2. **embed-multi is the best method by aggregate F1 and AP**, particularly where the test distribution matches training (~26% positive rate). Its contrastive objective improves representation quality and is worth pursuing further.

3. **embed-multi has a precision bias** that hurts recall and degrades at low positive rates. If the real-world positive rate is low (e.g. ~13%), `embed` is preferable. This is worth noting for the thesis given that checkworthiness datasets vary widely in base rate.

4. **Gains are baseline-conditional.** A natural next question is whether this is a property of the sequences (data difficulty) or the datasets (topic/domain mismatch with the synthetic pool). Inspecting which sequences are weak baselines could reveal whether targeted synthetic data generation would help more.

5. **embed-multi generalisation gap** (test loss 0.523 above eval) is larger than embed. If early stopping criteria were tuned against a held-out set rather than the dev split, this gap might close. Worth monitoring in future multi-task runs.

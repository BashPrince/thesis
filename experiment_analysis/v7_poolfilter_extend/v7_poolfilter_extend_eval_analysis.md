# Analysis: v7_poolfilter_extend_eval
*Proposal: experiment_analysis/v7_poolfilter_extend/eval_proposal.md*

## Results summary

All 180 runs finished successfully (15 sequences × 4 aug methods × 3 seeds). Every augmentation
method beats the `none` baseline on every one of the 15 sequences. The improvement is large
(+4–5 F1 points on average) and highly significant (p < 0.0001 for all three paired t-tests).
The three augmentation strategies are closely bunched: `extend-multi` edges out `embed`, which
edges out `unfiltered`, but the gaps between them are small relative to their shared lift over
`none`.

Note: this test set has ~13% positives, which is considerably sparser than the original test set.
Absolute F1 values (0.44–0.59 range) reflect the harder evaluation target; relative ordering
across aug methods is the main signal of interest.

---

## Per-method F1 results

Mean test/F1 across all sequences and seeds (n=45 runs per method):

| aug method   | mean F1 | std F1 | mean diff vs none | std diff | t     | p        | win rate |
|-------------|---------|--------|-------------------|----------|-------|----------|----------|
| none        | 0.5077  | 0.0298 | —                 | —        | —     | —        | —        |
| embed       | 0.5520  | 0.0194 | +0.0443           | 0.0229   | 7.253 | < 0.0001 | 15/15    |
| unfiltered  | 0.5499  | 0.0166 | +0.0422           | 0.0182   | 8.685 | < 0.0001 | 15/15    |
| extend-multi| 0.5536  | 0.0205 | +0.0459           | 0.0231   | 7.444 | < 0.0001 | 15/15    |

The augmented methods are separated by only ~0.002 F1 from each other; `extend-multi` is the
narrowest winner, followed closely by `embed`, then `unfiltered`. All three methods also show
lower variance than `none` (std ~0.017–0.021 vs 0.030), suggesting augmentation stabilises
performance on this sparser-positive test set.

---

## Per-sequence breakdown

Mean F1 per sequence per aug method:

| seq | embed  | extend-multi | none   | unfiltered |
|-----|--------|--------------|--------|------------|
| 0   | 0.5338 | 0.5245       | 0.5021 | 0.5269     |
| 1   | 0.5473 | 0.5531       | 0.5188 | 0.5544     |
| 2   | 0.5298 | 0.5398       | 0.4972 | 0.5582     |
| 3   | 0.5661 | 0.5598       | 0.5370 | 0.5549     |
| 4   | 0.5696 | 0.5748       | 0.5334 | 0.5556     |
| 5   | 0.5486 | 0.5583       | 0.5219 | 0.5633     |
| 6   | 0.5392 | 0.5377       | 0.4918 | 0.5398     |
| 7   | 0.5442 | 0.5366       | 0.5078 | 0.5422     |
| 8   | 0.5823 | 0.5914       | 0.5424 | 0.5749     |
| 9   | 0.5478 | 0.5628       | 0.4855 | 0.5501     |
| 10  | 0.5445 | 0.5523       | 0.5307 | 0.5583     |
| 11  | 0.5202 | 0.5134       | 0.4554 | 0.5253     |
| 12  | 0.5940 | 0.5817       | 0.5268 | 0.5722     |
| 13  | 0.5670 | 0.5693       | 0.5296 | 0.5578     |
| 14  | 0.5453 | 0.5480       | 0.4351 | 0.5147     |

Notable patterns:

- **seq_14** shows the largest absolute gains (up to +0.113 for `extend-multi`), and `none` is
  particularly weak (0.4351), suggesting this dataset is especially hard without augmentation.
- **seq_11** has the lowest overall F1 across all methods. All three augmented methods still
  improve on `none` (0.4554), with `unfiltered` performing best (0.5253).
- **seq_12** reaches the highest mean F1 under `embed` (0.5940), while `none` is comparatively
  mediocre (0.5268), making it one of the strongest beneficiaries of filtering.
- No sequence benefits from `none` over any augmented method.

Best-aug distribution across 15 sequences:
- `unfiltered`: 6 sequences
- `extend-multi`: 5 sequences
- `embed`: 4 sequences

No single method dominates; the winner varies per dataset.

---

## Statistical tests

All tests are paired t-tests across the 15 sequence-level mean F1 values (averaging over the 3
seeds per sequence before pairing).

| comparison            | mean diff | std diff | t     | p        | wins |
|-----------------------|-----------|----------|-------|----------|------|
| embed vs none         | +0.0443   | 0.0237   | 7.253 | < 0.0001 | 15/15|
| unfiltered vs none    | +0.0422   | 0.0188   | 8.685 | < 0.0001 | 15/15|
| extend-multi vs none  | +0.0459   | 0.0239   | 7.444 | < 0.0001 | 15/15|

All three improvements are highly significant with no exceptions across sequences. Seed variance
is low and comparable across methods (mean within-cell std: embed=0.0096, extend-multi=0.0106,
none=0.0101, unfiltered=0.0072).

---

## Hypothesis evaluation

**Hypothesis (per method): F1 results on the test set are improved over the baseline (aug=none).**

- **embed**: CONFIRMED. +4.4 F1 points on average, p < 0.0001, 15/15 sequences improve.
- **unfiltered**: CONFIRMED. +4.2 F1 points on average, p < 0.0001, 15/15 sequences improve.
- **extend-multi**: CONFIRMED. +4.6 F1 points on average, p < 0.0001, 15/15 sequences improve.

Secondary observations:

1. **Filtering vs random sampling**: `embed` (0.5520) only marginally outperforms `unfiltered`
   (0.5499). The gap is +0.0021 F1, which is not statistically distinguished. The bulk of the
   benefit thus comes from having *more* data (any synthetic samples), not from the embedding
   filter per se — at least on this harder test distribution.

2. **Multi-task contrastive training**: `extend-multi` (0.5536) slightly edges `embed` (0.5520)
   by +0.0016 F1. The signal is positive but tiny; the contrastive auxiliary loss does not provide
   a large additional lift over standard classification training on the same filtered data.

3. **Lower variance with augmentation**: The `none` baseline shows noticeably higher cross-sequence
   variance (std=0.030) than the augmented methods (std=0.017–0.021). Augmentation may be acting
   as a regulariser that reduces sensitivity to which specific real-data sample was drawn for a
   given sequence.

---

## Conclusion and recommended next steps

All three augmentation approaches work: adding synthetic data robustly improves checkworthiness
F1 on this harder, sparser-positive test set, with consistent and significant gains across all 15
real-data splits.

The more nuanced finding is that the differences *between* augmentation strategies are small. The
embedding filter adds ~0.002 F1 over random sampling, and the multi-task contrastive objective adds
another ~0.002 over the filter alone. These margins are within noise for individual sequences.

Recommended next steps:

1. **Investigate the filter's role more carefully**: The small gap between `embed` and `unfiltered`
   suggests diminishing returns from the filtering step on this test set. It is worth checking
   whether this holds across different test-set compositions or whether the filter adds more value
   when the positives-ratio mismatch between train and test is larger.

2. **Probe why extend-multi is not stronger**: The multi-task contrastive signal is not clearly
   better than standard classification on the same filtered data. This could indicate that the
   contrastive objective needs tuning (temperature, projection head, loss weight) or that the
   adapter bottleneck is already extracting the most discriminative signal without it.

3. **Per-dataset diagnostics**: The large spread in gains (seq_14: +0.11; seq_10: +0.03) suggests
   dataset-specific factors matter. Profiling the hardest and easiest sequences by their real-data
   label balance or domain could guide future augmentation design.

---

## Precision/Recall Analysis

### Per-method precision and recall

Mean test/precision, test/recall, and test/F1 across all sequences and seeds (n=45 per method):

| aug method   | mean P | std P  | mean R | std R  | mean F1 | std F1 |
|-------------|--------|--------|--------|--------|---------|--------|
| none        | 0.3804 | 0.0339 | 0.7681 | 0.0389 | 0.5077  | 0.0316 |
| embed       | 0.4333 | 0.0344 | 0.7666 | 0.0347 | 0.5520  | 0.0215 |
| unfiltered  | 0.4374 | 0.0277 | 0.7444 | 0.0320 | 0.5499  | 0.0183 |
| extend-multi| 0.4389 | 0.0322 | 0.7550 | 0.0406 | 0.5536  | 0.0228 |

The augmentation benefit on F1 is driven almost entirely by gains in precision: all three methods
improve precision by ~5–6 percentage points over `none`, while recall is largely unchanged or
slightly lower. `none` achieves the highest mean recall (0.768) but pays for it with very low
precision (0.380).

### Paired t-tests vs none (precision and recall)

All tests are paired across the 15 sequence-level means (seeds averaged before pairing).

**Precision:**

| comparison            | mean diff | std diff | t      | p        | wins  |
|-----------------------|-----------|----------|--------|----------|-------|
| embed vs none         | +0.0529   | 0.0270   | 7.574  | < 0.0001 | 15/15 |
| unfiltered vs none    | +0.0570   | 0.0203   | 10.858 | < 0.0001 | 15/15 |
| extend-multi vs none  | +0.0584   | 0.0262   | 8.651  | < 0.0001 | 15/15 |

**Recall:**

| comparison            | mean diff | std diff | t      | p       | wins  |
|-----------------------|-----------|----------|--------|---------|-------|
| embed vs none         | −0.0015   | 0.0315   | −0.189 | 0.8529  | 7/15  |
| unfiltered vs none    | −0.0237   | 0.0372   | −2.467 | 0.0271  | 4/15  |
| extend-multi vs none  | −0.0132   | 0.0300   | −1.697 | 0.1117  | 5/15  |

Augmentation significantly improves precision (p < 0.0001, 15/15 wins for all methods) but has no
significant effect on recall for `embed` and `extend-multi` (p = 0.85 and 0.11 respectively).
`unfiltered` shows a small but significant recall drop (−0.024, p = 0.027). This pattern is
consistent with augmentation shifting the decision boundary toward higher precision at the cost of
slightly lower recall — reasonable behaviour given that the test set is sparser in positives (~13%)
than the augmented training data.

### Per-sequence precision and recall breakdown

**Mean PRECISION by aug method per sequence:**

| seq | embed  | extend-multi | none   | unfiltered |
|-----|--------|--------------|--------|------------|
| 0   | 0.4038 | 0.4071       | 0.3699 | 0.3988     |
| 1   | 0.4268 | 0.4400       | 0.4030 | 0.4515     |
| 2   | 0.3904 | 0.4162       | 0.3637 | 0.4418     |
| 3   | 0.4495 | 0.4373       | 0.4015 | 0.4319     |
| 4   | 0.4582 | 0.4715       | 0.4196 | 0.4429     |
| 5   | 0.4136 | 0.4313       | 0.3819 | 0.4499     |
| 6   | 0.4149 | 0.4147       | 0.3528 | 0.4223     |
| 7   | 0.4252 | 0.4270       | 0.3800 | 0.4178     |
| 8   | 0.4764 | 0.4995       | 0.4201 | 0.4807     |
| 9   | 0.4223 | 0.4457       | 0.3610 | 0.4313     |
| 10  | 0.4242 | 0.4315       | 0.3942 | 0.4652     |
| 11  | 0.4036 | 0.3893       | 0.3367 | 0.4107     |
| 12  | 0.5062 | 0.4677       | 0.4052 | 0.4623     |
| 13  | 0.4578 | 0.4596       | 0.4093 | 0.4550     |
| 14  | 0.4268 | 0.4447       | 0.3077 | 0.3991     |

**Mean RECALL by aug method per sequence:**

| seq | embed  | extend-multi | none   | unfiltered |
|-----|--------|--------------|--------|------------|
| 0   | 0.7900 | 0.7446       | 0.7837 | 0.7772     |
| 1   | 0.7630 | 0.7451       | 0.7289 | 0.7183     |
| 2   | 0.8242 | 0.7678       | 0.7864 | 0.7581     |
| 3   | 0.7689 | 0.7786       | 0.8112 | 0.7765     |
| 4   | 0.7543 | 0.7439       | 0.7338 | 0.7477     |
| 5   | 0.8184 | 0.7943       | 0.8243 | 0.7577     |
| 6   | 0.7711 | 0.7662       | 0.8123 | 0.7486     |
| 7   | 0.7565 | 0.7262       | 0.7655 | 0.7723     |
| 8   | 0.7489 | 0.7250       | 0.7658 | 0.7156     |
| 9   | 0.7826 | 0.7644       | 0.7415 | 0.7597     |
| 10  | 0.7604 | 0.7673       | 0.8128 | 0.7027     |
| 11  | 0.7376 | 0.7549       | 0.7045 | 0.7325     |
| 12  | 0.7208 | 0.7718       | 0.7570 | 0.7514     |
| 13  | 0.7468 | 0.7491       | 0.7504 | 0.7228     |
| 14  | 0.7554 | 0.7252       | 0.7437 | 0.7252     |

### Method ranking consistency

**Overall ranking by metric:**

| rank | precision    | recall       | F1           |
|------|-------------|-------------|-------------|
| 1st  | extend-multi (0.4389) | none (0.7681) | extend-multi (0.5536) |
| 2nd  | unfiltered (0.4374)   | embed (0.7666) | embed (0.5520)        |
| 3rd  | embed (0.4333)        | extend-multi (0.7550) | unfiltered (0.5499)  |
| 4th  | none (0.3804)         | unfiltered (0.7444) | none (0.5077)      |

**Per-sequence best method by metric:**

| seq | best precision  | best recall   | best F1      |
|-----|----------------|---------------|--------------|
| 0   | extend-multi   | embed         | embed        |
| 1   | unfiltered     | embed         | unfiltered   |
| 2   | unfiltered     | embed         | unfiltered   |
| 3   | embed          | none          | embed        |
| 4   | extend-multi   | embed         | extend-multi |
| 5   | unfiltered     | none          | unfiltered   |
| 6   | unfiltered     | none          | unfiltered   |
| 7   | extend-multi   | unfiltered    | embed        |
| 8   | extend-multi   | none          | extend-multi |
| 9   | extend-multi   | embed         | extend-multi |
| 10  | unfiltered     | none          | unfiltered   |
| 11  | unfiltered     | extend-multi  | unfiltered   |
| 12  | embed          | extend-multi  | embed        |
| 13  | extend-multi   | none          | extend-multi |
| 14  | extend-multi   | embed         | extend-multi |

The method rankings are substantially different across precision and recall, but largely consistent
between precision and F1. This makes sense mechanically: on a sparse-positive test set (~13%
positives), F1 is dominated by precision because the precision floor is much lower than the recall
floor — small absolute changes in precision produce larger F1 swings than comparable changes in
recall.

Three observations follow from this:

1. **Recall ordering inverts for none vs augmented methods.** `none` leads on recall globally
   (0.768) and is the best-recall method in 5 of 15 sequences, yet it is the worst F1 method by a
   wide margin. The model trained without synthetic data is over-predicting positives (high recall,
   low precision), consistent with a more lenient decision boundary. Augmentation shifts this
   boundary: recall falls modestly while precision rises sharply, resulting in a net F1 gain.

2. **Embed preserves recall better than unfiltered.** `embed` (0.767) sits just below `none`
   (0.768) on recall, whereas `unfiltered` (0.744) shows the largest recall drop (−0.024,
   p = 0.027). The embedding filter appears to select synthetic examples that maintain the
   training signal for recall while still lifting precision, a mild but consistent advantage over
   random sampling. `extend-multi` (0.755) falls between the two.

3. **The precision–recall tradeoff varies by sequence.** In sequences where `none` is best on
   recall (seq_3, 5, 6, 8, 10, 13), the augmented methods are trading off recall for precision
   more aggressively. This could reflect dataset-specific class-boundary characteristics. The
   implication for sensitivity to class balance is that in domains where recall is the priority
   (e.g., fact-checking pipelines that prefer fewer misses), the `embed` method is the safest
   choice among the augmented strategies, as it achieves the precision gains of augmentation while
   losing the least recall.

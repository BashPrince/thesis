# Analysis: Sequence-Level Covariate Analysis

*Script: experiment_analysis/sequence_covariates/analyze_sequence_covariates.py*

## Motivation

The gain-baseline correlation (r ≈ −0.80) is one of the strongest findings across experiments: sequences with low baseline F1 benefit most from augmentation. But why do some sequences have low baseline F1? If weak sequences systematically have sparser or less representative real training data, that would confirm a "gap-filling" interpretation of the augmentation effect.

## Approach

For each of the 15 sequences in `v7_poolfilter_extend`, we downloaded the `none` (baseline) training artifact containing 128 real samples (64 positive, 64 negative). All samples and the test set were encoded with `all-MiniLM-L6-v2` (normalized), the same model used for embed filtering and the synthetic data quality analysis.

We computed the following covariates per sequence:

1. **Mean pairwise similarity** — average cosine similarity across all 128×128 sample pairs. Higher values indicate a more homogeneous (less diverse) training set.
2. **Embedding spread** — mean standard deviation of embedding vectors across dimensions. Higher values indicate more spread-out samples.
3. **Per-class spread** — embedding spread computed separately for positives and negatives.
4. **Class separability** — mean positive-positive similarity minus mean positive-negative similarity. Higher values mean clearer class separation in the training data.
5. **Test overlap (max sim)** — for each training sample, the maximum cosine similarity to any test sample of the same class, averaged across all training samples. Measures how close the training data is to the test distribution.
6. **Test similarity (mean)** — mean cosine similarity between training and test samples of the same class. A global measure of train-test alignment.
7. **Sentence length** — mean and standard deviation of word count.

Each covariate was correlated (Pearson) with:
- The sequence's **baseline F1** (mean across 3 seeds of the `none` condition)
- The sequence's **embed gain** (embed F1 minus baseline F1)

---

## Results

### Correlation with Baseline F1

| Covariate | r | p | Description |
|---|---|---|---|
| class_sep | −0.379 | 0.164 | Class separability |
| neg_spread | −0.374 | 0.170 | Negative class spread |
| test_sim | +0.251 | 0.368 | Test similarity (mean) |
| test_overlap | +0.229 | 0.411 | Test overlap (max sim) |
| emb_spread | −0.226 | 0.418 | Embedding spread |
| mean_sim | +0.208 | 0.456 | Mean pairwise similarity |
| std_len | −0.155 | 0.582 | Sentence length std |
| pos_spread | +0.112 | 0.692 | Positive class spread |
| mean_len | −0.062 | 0.825 | Mean sentence length |

**No covariate significantly predicts baseline F1.** The strongest candidates are class separability (r = −0.379) and negative class spread (r = −0.374), both at p ≈ 0.17 — well above significance. Test overlap and test similarity show weak positive trends (sequences whose training data is more similar to the test set tend to perform better), but these are also non-significant.

### Correlation with Embed Gain

| Covariate | r | p |
|---|---|---|
| class_sep | +0.484 | 0.068 |
| pos_spread | −0.314 | 0.254 |
| neg_spread | +0.259 | 0.352 |
| std_len | +0.169 | 0.548 |
| mean_len | +0.136 | 0.628 |
| test_overlap | −0.059 | 0.835 |
| test_sim | −0.036 | 0.898 |
| mean_sim | +0.032 | 0.909 |
| emb_spread | −0.024 | 0.933 |

The closest to significance is class separability (r = +0.484, p = 0.068): sequences where real positives and negatives are better separated in embedding space tend to benefit more from augmentation. This is counterintuitive — one might expect less separable sequences to benefit more from augmentation filling in the gap. A possible explanation: sequences with clear class separation provide a better "scaffold" for the synthetic data to reinforce, while sequences with muddy boundaries gain less because the synthetic data (which also has imperfect class separation) cannot resolve the ambiguity.

However, this is borderline and should not be over-interpreted.

### Per-Sequence Data

| Seq | Baseline F1 | Embed Gain | Class Sep | Test Overlap | Mean Sim |
|---|---|---|---|---|---|
| 1 | 0.567 | +0.053 | 0.045 | 0.420 | 0.143 |
| 14 | 0.571 | +0.043 | 0.048 | 0.424 | 0.138 |
| 9 | 0.578 | +0.053 | 0.015 | 0.431 | 0.131 |
| 7 | 0.598 | +0.030 | 0.032 | 0.425 | 0.153 |
| 4 | 0.602 | +0.059 | 0.051 | 0.429 | 0.149 |
| 0 | 0.606 | +0.045 | 0.030 | 0.412 | 0.134 |
| 13 | 0.609 | +0.052 | 0.040 | 0.417 | 0.140 |
| 6 | 0.621 | +0.017 | 0.024 | 0.418 | 0.133 |
| 11 | 0.634 | −0.021 | 0.021 | 0.416 | 0.134 |
| 10 | 0.645 | +0.014 | 0.047 | 0.410 | 0.136 |
| 8 | 0.675 | −0.001 | 0.034 | 0.424 | 0.140 |
| 12 | 0.677 | −0.019 | 0.030 | 0.428 | 0.148 |
| 2 | 0.684 | −0.008 | 0.020 | 0.426 | 0.142 |
| 3 | 0.687 | −0.013 | 0.023 | 0.429 | 0.138 |
| 5 | 0.691 | +0.020 | 0.028 | 0.430 | 0.152 |

---

## Conclusions

1. **Baseline F1 variation is not explained by aggregate distributional properties of the training data.** None of the embedding-space covariates (diversity, spread, class separability, test overlap) significantly predict which sequences perform well or poorly. The 128-sample training sets are remarkably similar to each other on these aggregate measures.

2. **The variation likely comes from specific sample composition.** Which particular examples fall near the decision boundary — and whether they happen to be representative of the test distribution — is not captured by aggregate statistics like mean similarity or spread. This suggests the baseline F1 differences are driven by instance-level effects rather than distributional properties.

3. **The borderline class_sep → embed_gain correlation (r = +0.484, p = 0.068) is intriguing but inconclusive.** If real, it suggests augmentation works best when the real data already provides a clear class scaffold. This would mean augmentation reinforces existing signal rather than creating new signal — consistent with the precision-only improvement finding.

4. **For the thesis narrative:** The null result is informative. It strengthens the interpretation that the gain-baseline correlation (r ≈ −0.80) is fundamentally about sample-level noise in the 128-sample regime: some random draws happen to include more informative examples. Augmentation helps by averaging over this noise — adding synthetic data dilutes the impact of any single unrepresentative real sample.

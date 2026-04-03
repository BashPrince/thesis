# Analysis: Synthetic Data Quality Characterization

*Script: experiment_analysis/synthetic_data_quality/analyze_synthetic_quality.py*

## Approach

For each of the 5 sequences in `v7_poolfilter`, we downloaded the training data artifacts from WandB for every augmentation method. Synthetic samples were identified by diffing each augmented artifact's train.csv against the corresponding `none` (baseline) artifact — non-overlapping texts are synthetic. This yields 1024 synthetic samples per augmented method per sequence.

We then computed three categories of metrics:

1. **Embedding similarity**: Each synthetic and real sample was encoded with `all-MiniLM-L6-v2` (normalized). This is the same model and configuration used by the embed filtering step in the generation pipeline (`generate/filter/v7_generate.py`), so the similarity numbers directly reflect what the filter optimizes for. We computed mean cosine similarity between synthetic and real samples, broken down by class (synthetic positives vs real positives, etc.). The key diagnostic is the **class separation gap**: how much more similar synthetic samples are to same-class real samples than to opposite-class real samples.

2. **Lexical diversity**: Type-token ratio (TTR) — the fraction of unique words in the synthetic pool. Lower TTR indicates more repetitive text.

3. **Sentence length**: Mean and standard deviation of word count.

All metrics are reported as mean +/- SE across the 5 sequences. Multiple seeds per sequence were not analyzed as they share the same training data.

---

## Results

### Class balance

All synthetic methods produce perfectly balanced pools (512 Yes / 512 No per 1024 synthetic samples). The real training subsets are also balanced (64 Yes / 64 No per 128 samples).

### Embedding similarity

| Method | SP-RP | SP-RP max | SN-RN | pos_gap | neg_gap |
|---|---|---|---|---|---|
| embed | 0.167+/-0.004 | 0.440+/-0.002 | 0.160+/-0.004 | 0.045+/-0.005 | 0.009+/-0.004 |
| tfidf | 0.161+/-0.003 | 0.419+/-0.003 | 0.155+/-0.004 | 0.042+/-0.005 | 0.010+/-0.003 |
| unfiltered | 0.158+/-0.003 | 0.410+/-0.004 | 0.155+/-0.004 | 0.041+/-0.005 | 0.009+/-0.003 |
| free | 0.112+/-0.003 | 0.357+/-0.006 | 0.085+/-0.003 | 0.023+/-0.005 | -0.001+/-0.003 |
| genetic | 0.142+/-0.005 | 0.394+/-0.009 | 0.096+/-0.002 | 0.020+/-0.007 | 0.026+/-0.005 |

Reference: real_pos vs real_pos = 0.162+/-0.005, real_pos vs real_neg = 0.133+/-0.004.

**Legend:** SP-RP = mean cosine similarity between synthetic positives and real positives. SP-RP max = mean of the per-sample maximum similarity to any real positive. pos_gap = SP-RP minus SP-RN (how much more similar synthetic positives are to real positives than to real negatives). neg_gap = same for negatives.

**Interpretation:**

- **embed/tfidf/unfiltered** synthetic positives are as close to real positives as real positives are to each other (SP-RP ≈ 0.158–0.167 vs real-real 0.162). The pool-filtering selects semantically representative data.
- **free** is much more distant (SP-RP = 0.112). Its synthetic positives are topically different from the real check-worthy claims, explaining its near-zero performance gain.
- **genetic** has moderate proximity (SP-RP = 0.142) but a very low class separation gap for positives (pos_gap = 0.020). Its synthetic positives are almost as similar to real negatives as to real positives. This means the genetic algorithm produces positives that blur the class boundary rather than reinforcing it — explaining why it actively hurts classification.
- **free** has neg_gap ≈ 0: its synthetic negatives carry no class signal at all.
- The **class separation gap** (pos_gap) correlates with downstream performance: embed (0.045) > tfidf (0.042) > unfiltered (0.041) >> free (0.023) ≈ genetic (0.020). This ordering matches the F1 ranking.

### Lexical metrics

| Method | Synth TTR | Real TTR | Synth avg len (words) |
|---|---|---|---|
| embed | 0.222+/-0.002 | 0.387+/-0.006 | 13.5+/-0.1 |
| tfidf | 0.224+/-0.002 | 0.387+/-0.006 | 14.1+/-0.1 |
| unfiltered | 0.209+/-0.001 | 0.387+/-0.006 | 13.4+/-0.1 |
| free | 0.192+/-0.001 | 0.387+/-0.006 | 14.3+/-0.0 |
| genetic | 0.210+/-0.002 | 0.387+/-0.006 | 14.9+/-0.0 |

**Interpretation:**

- All synthetic data has roughly half the lexical diversity of real data (TTR 0.19–0.22 vs 0.39). LLM-generated text is uniformly more repetitive.
- Pool-filtered methods (embed, tfidf) have slightly higher TTR than unfiltered/free, suggesting the filtering step selects more lexically diverse samples from the pool.
- Synthetic sentences are consistently shorter than real ones (~13.5–14.9 words vs ~19 words). Real data comes from political speech transcripts which tend to be longer and more complex.
- **genetic** has the lowest sentence length variance (std ≈ 1.5 vs 3.3–4.8 for other methods), indicating the genetic algorithm converges to a narrow syntactic template.

---

## Conclusions

1. **Semantic alignment with real data is the primary quality differentiator.** embed/tfidf/unfiltered produce synthetic data that is close to the real distribution in embedding space. free and genetic do not.

2. **Class separation in the synthetic pool predicts downstream performance.** The pos_gap metric (how well synthetic data preserves the check-worthy vs non-check-worthy distinction) tracks the F1 ranking across methods.

3. **Pool filtering improves similarity but not dramatically over unfiltered.** The embed filter achieves SP-RP = 0.167 vs unfiltered's 0.158 — a modest improvement consistent with the small (non-significant at n=5) F1 difference between these methods.

4. **The genetic method's failure is explained by class boundary blurring.** Its synthetic positives are nearly as similar to real negatives as to real positives (pos_gap = 0.020), producing training signal that confuses rather than helps the classifier.

5. **Free generation's failure is explained by topical misalignment.** Its synthetic data is simply too distant from the real distribution to be useful, despite having reasonable class balance and sentence structure.

6. **All synthetic data is lexically impoverished relative to real data.** This is a fundamental limitation of LLM generation at temperature=1.0. The reduced diversity may contribute to the precision-only improvement pattern: synthetic data reinforces existing decision boundaries (improving precision) but cannot introduce the lexical variety needed to discover new check-worthy patterns (no recall improvement).

---

## Figures

All figures are generated by the analysis script and saved alongside this document.

**`synth_similarity_heatmap.pdf`** — Heatmap of mean pairwise cosine similarity between synthetic and real samples, broken down by class combination (Synth+→Real+, Synth+→Real−, Synth−→Real+, Synth−→Real−). Each cell is the mean of the full pairwise similarity matrix (e.g. 512×64 for synthetic positives vs real positives). Shows embed/tfidf/unfiltered are semantically close to real data across all class combinations, while free is uniformly distant and genetic has an asymmetric pattern (moderate Synth+→Real+ but very low Synth−→Real+).

**`synth_similarity_violins.pdf`** — Violin plots of per-sample maximum cosine similarity to the nearest real same-class sample (left panel: positives, right panel: negatives). For each synthetic sample, the maximum similarity to any of the 64 real same-class samples is computed; distributions are pooled across all 5 sequences. The embed violin shows a sharp lower-tail cutoff — this is the embed filter's effect, since it uses the same embedding model (`all-MiniLM-L6-v2`) to select samples closest to real data, implicitly imposing a minimum similarity floor. The unfiltered violin from the same pool lacks this cutoff. free has the lowest mean and longest lower tail for positives; genetic has the lowest mean for negatives.

**`synth_class_separation.pdf`** — Scatter plot of each method's synthetic positives in similarity space: x-axis = mean similarity to real positives (SP-RP), y-axis = mean similarity to real negatives (SP-RN). The diagonal represents zero class separation (synthetic positives equally similar to both classes). Points further below the diagonal have better class separation (pos_gap = vertical distance from diagonal). embed/tfidf/unfiltered cluster in the lower-right (high same-class similarity, good separation). free sits in the lower-left (distant from everything). genetic is closer to the diagonal (poor separation). The open square shows the real-real reference point.

**`synth_lexical_comparison.pdf`** — Bar charts comparing synthetic data across methods on two lexical metrics: type-token ratio (left) and mean sentence length in words (right). A horizontal dashed line with shaded SE band shows the real data reference (identical across methods since all sequences use the same 128 real samples). All synthetic methods fall well below the real TTR and produce shorter sentences.

**`synth_gap_vs_f1.pdf`** — Scatter plot of class separation gap (pos_gap, x-axis) vs F1 gain over baseline (y-axis) for each method. Error bars show SE across sequences for pos_gap; F1 gains are from the v7_poolfilter experiment. Pearson r=0.90, p=0.039 (5 methods). Demonstrates that synthetic data quality as measured by class boundary preservation significantly predicts downstream classification improvement. Note that with 5 points, genetic has strong leverage on the correlation.

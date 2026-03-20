# Analysis: unrestricted_wrup_extended_fixed_seed_roberta
*Proposal: experiment_analysis/unrestricted_wrup_extended_fixed_seed_roberta/proposal.md*

## Results Summary

All 150 runs completed successfully: 10 sequences × 5 augmentation levels (0, 128, 256, 512, 1024) × 3 fixed seeds. The experiment tests full fine-tuning of **RoBERTa-base** (125M parameters, no adapters) on checkworthiness classification with `unrestricted_wrup` LLM-augmented data.

| aug   | mean F1 | std F1 | Δ vs baseline | t      | p      |
|-------|---------|--------|---------------|--------|--------|
| 0     | 0.7303  | 0.0292 | —             | —      | —      |
| 128   | 0.7386  | 0.0345 | +0.0083       | 1.337  | 0.2140 |
| 256   | 0.7375  | 0.0243 | +0.0072       | 0.867  | 0.4086 |
| 512   | 0.7399  | 0.0320 | +0.0095       | 0.841  | 0.4222 |
| 1024  | 0.7402  | 0.0402 | +0.0099       | 0.739  | 0.4789 |

**No statistically significant differences** vs the no-augmentation baseline (all p > 0.20). The effect size is approximately +0.01 F1 across all augmentation levels, with diminishing returns beyond aug=128.

### Per-sequence F1

| seq | aug=0  | aug=128 | aug=256 | aug=512 | aug=1024 |
|-----|--------|---------|---------|---------|----------|
| 0   | 0.6975 | 0.7253  | 0.7384  | 0.7367  | **0.7437** |
| 1   | 0.6998 | 0.7134  | 0.7352  | **0.7521** | 0.7391 |
| 2   | **0.7198** | 0.6983 | 0.7048 | 0.7029 | 0.6639 |
| 3   | 0.7382 | 0.7738  | 0.7566  | 0.7839  | **0.7987** |
| 4   | 0.7844 | **0.7869** | 0.7567 | 0.7664 | 0.7700 |
| 5   | 0.7492 | 0.7332  | 0.7541  | 0.7733  | **0.7816** |
| 6   | 0.7167 | 0.7288  | **0.7298** | 0.7077 | 0.7208 |
| 7   | **0.7574** | **0.7638** | 0.7218 | 0.6940 | 0.6940 |
| 8   | 0.6892 | 0.6811  | 0.6969  | 0.7110  | **0.7157** |
| 9   | 0.7510 | **0.7813** | 0.7805 | 0.7709 | 0.7744 |

Win rate vs no-aug baseline: 70% at aug=128, 70% at aug=256, 60% at aug=512, 70% at aug=1024.

## Metric and Training Analysis

### Precision–Recall Trade-off

Augmentation consistently **raises precision while leaving recall roughly flat or slightly reducing it**:

| aug  | Δprecision | Δrecall |
|------|-----------|---------|
| 128  | +0.0122   | +0.0009 |
| 256  | +0.0181   | −0.0065 |
| 512  | +0.0204   | −0.0025 |
| 1024 | +0.0223   | −0.0034 |

At aug=1024, precision reaches 0.7752 (vs 0.7529 baseline) while recall barely moves (0.7136 vs 0.7170). This suggests that the synthetic data shifts the model toward more conservative (higher precision, fewer positive predictions) behaviour, but does not improve its ability to retrieve true checkworthy sentences.

### Training Dynamics (Epoch Budget)

The epoch formula `120000 // (100 + aug_size)` combined with step-based early stopping (patience=15 × eval_steps=64 = 960 gradient steps) creates a dramatic difference in epoch count:

| aug  | mean epochs (early stop) |
|------|--------------------------|
| 0    | ~258                     |
| 128  | ~106                     |
| 256  | ~74                      |
| 512  | ~52                      |
| 1024 | ~31                      |

With aug=0 and 128 examples, each epoch is 8 gradient steps, so 960 steps ≈ 120 epochs of patience. With aug=1024 and 1152 examples, each epoch is 72 steps, so 960 steps ≈ 13 epochs of patience. In both cases the total gradient budget before stopping is roughly equal, but the no-aug model cycles through the same 128 examples hundreds of times.

Train loss collapses to near-zero for all augmentation levels, confirming the model memorises training data regardless of augmentation. This is expected for full fine-tuning of a 125M-parameter model on 128 examples.

### Eval/F1 vs Test/F1 Gap

| aug  | mean eval/F1 | mean test/F1 | gap   |
|------|-------------|-------------|-------|
| 0    | 0.7805      | 0.7303      | 0.050 |
| 128  | 0.7633      | 0.7386      | 0.025 |
| 256  | 0.7649      | 0.7375      | 0.027 |
| 512  | 0.7672      | 0.7399      | 0.027 |
| 1024 | 0.7704      | 0.7402      | 0.030 |

The no-augmentation baseline has a notably larger eval→test gap (0.05 vs ~0.027). The validation-set performance of aug=0 is inflated by overfitting to the validation distribution (which is in-domain with training), while the test set gap better reflects generalisation. Augmented runs show consistently smaller gaps, suggesting modest regularisation of the checkpoint selection criterion.

### Seed Variance

Mean across-seed std(F1): aug=0 → 0.020, aug=128 → 0.019, aug=256 → 0.023, aug=512 → 0.022, aug=1024 → 0.024. Variance does not decrease with more data; augmentation at the highest level slightly increases instability. Sequence 3 at aug=256 stands out (std=0.058), indicating one outlier seed.

### Sequences Hurt by Augmentation

Two sequences consistently fail to benefit:

- **Seq 2**: F1 degrades monotonically with augmentation. At aug=1024, two of three seeds have recall drop to 0.54–0.58 (from 0.63 baseline). Precision also falls slightly, suggesting the synthetic data conflicts with the natural data manifold for this particular sequence.
- **Seq 7**: Helps at aug=128 (+0.006) but harms at aug≥256. At aug=512/1024 recall falls by ≈0.12. This sequence may have particular coverage of examples that the augmentation corrupts.

## Hypothesis Evaluation

**Hypothesis**: *Synthetic augmentation of a checkworthy sentence dataset with LLM generations raises F1 on the test set.*

**Verdict: Weakly supported but not statistically confirmed.**

- The mean F1 increases by ~+0.01 consistently across all augmentation levels, but no augmentation level achieves p < 0.05 vs baseline (best is aug=128, p=0.21).
- 7 of 10 sequences benefit from some level of augmentation; 3 do not (seqs 2, 4—marginal, 7).
- The augmentation effect plateaus rapidly: aug=128 captures most of the available gain. Going from 128 to 1024 additional examples yields only an additional +0.0016 mean F1.
- The direction of effect is precision-biased: augmentation reliably raises precision (+0.02 at aug=1024) but does not improve recall, limiting F1 gains.

## Conclusion and Recommended Next Steps

The experiment provides **weak positive evidence** that unrestricted wrup augmentation helps RoBERTa-base finetuning, but the effect is small (+1% F1) and inconsistent across sequences. The precision-only gain pattern suggests the synthetic data improves the model's selectivity but not its coverage of checkworthy sentences.

**Next steps:**

1. **Augmentation quality / recall-oriented generation**: The consistent recall plateau suggests the LLM is not generating diverse enough checkworthy examples. Targeted generation (e.g., prompting for borderline or varied checkworthy cases) may help.

2. **Compare with ModernBERT + adapters**: This RoBERTa result establishes a baseline. The key question is whether the adapter-based architecture with the same augmentation shows a larger or more significant effect.

3. **Investigate sequences 2 and 7**: Understanding why these sequences are hurt by augmentation (e.g., by inspecting their natural data distribution) could reveal whether the synthetic data is systematically biased in a way that hurts certain sub-topics.

4. **Aug=128 may be the practical sweet spot**: The effect saturates quickly, and aug=128 is sufficient for any benefit that the method provides. Running at 1024 adds compute without improving F1.

5. **Consider stratified augmentation**: If augmentation consistently increases precision at the cost of recall, generating more positive (checkworthy) examples specifically—rather than balanced generation—could correct this imbalance.

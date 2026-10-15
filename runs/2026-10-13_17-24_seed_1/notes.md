# Run 2026-10-13_17-24_seed_1

Frozen ProstT5 + adapters with residue max pooling.

Restored winning architecture: residue max pooling, equal-weight group mean
pooling, and fixed symmetric sum/absolute-difference/elementwise-product
features followed by the pair MLP and classifier. No learned interaction encoder.
Data augmentation ablation: training-only sequence masking at 10%.
Replace residue input tokens with encoded X before frozen backbone encoding;
preserve sequence length, prefix/EOS, and padding. Residue dropout is disabled (0%).
Training bypasses embedding caches; unmasked validation/test may reuse them.
The frozen backbone stays in evaluation mode and is encoded without gradients.
Token budgets: train 8192, evaluation 16384.
Reduced budgets can change batching/step counts versus cached baselines.
Historical class-weighted focal loss only (gamma 2, exp(-weighted CE), batch mean).
No LoRA or contrastive term; existing tokenized data is reused unchanged.
Eager execution and numerical safeguards are retained after the ablations.
Non-finite inputs/logits/loss/gradients abort training.
Each validation-AUROC improvement is checkpointed immediately.

## Comparison Setup

Code revision: `169b9e0a28063bb0212e7df46b3964d3a27c1725`. Compare against the unaugmented winner (`2026-10-08_01-46_seed_1`) and 10% residue-embedding dropout (`2026-10-13_00-45_seed_1`). All use frozen ProstT5 + adapters, residue max/group mean pooling, fixed symmetric pair features, historical focal-only loss (gamma 2), and seed 1.

This experiment replaces 10% of amino-acid input tokens with encoded X (token 23) before frozen backbone encoding, preserving sequence length, prefix/EOS, and padding. Residue dropout is disabled. Unlike embedding dropout, masking changes contextual representations through the backbone; no augmentation is applied during evaluation. Training uses live encoding with the frozen backbone in evaluation mode and without gradients; validation/test use the original embedding cache.

The comparison is not perfectly isolated: train/evaluation token budgets decrease from 49,152/65,536 to 8,192/16,384. Packing can therefore change batch composition, step counts, and the learning-rate schedule. The original baseline also used the earlier compilation path, while both augmentation runs retain eager execution/numerical safeguards. A matched no-masking live-encoding control would be needed to isolate the augmentation effect. Other saved sampling/optimizer settings match; the unused focal gamma setting was removed but gamma remains 2 in metadata.

All runs report 1,614 validation and 1,615 test pairs with matching source counts/cache paths. Exact cache/split fingerprints are absent, so comparisons assume the same data. Matching seeds are not independent replicates.

## Ranking Comparison

| Metric | No augmentation | Residue dropout | Sequence masking |
| --- | --- | --- | --- |
| Validation AUROC | 0.9551 | 0.9486 | 0.9430 |
| Validation AUPRC | 0.9899 | 0.9883 | 0.9885 |
| Test AUROC | 0.9356 | 0.9324 | 0.9039 |
| Test AUPRC | 0.9856 | 0.9844 | 0.9766 |

Masking lowers validation AUROC, the selection criterion, by 0.0121 versus no augmentation and 0.0056 versus residue dropout. Test AUROC declines by 0.0317/0.0285 respectively. Validation AUPRC is marginally above dropout, but test ranking is clearly weaker in these measurements. This cannot be explained solely by the decision threshold; single-seed differences are not statistical significance claims.

## Default-Threshold Comparison

| Metric | Validation baseline | Validation masking | Test baseline | Test masking |
| --- | --- | --- | --- | --- |
| Balanced accuracy | 0.9002 | 0.8686 | 0.8594 | 0.8142 |
| Specificity / negative recall | 0.9398 | 0.9074 | 0.8704 | 0.6991 |
| Positive recall | 0.8605 | 0.8298 | 0.8485 | 0.9292 |
| MCC | 0.6322 | 0.5661 | 0.5668 | 0.5913 |
| Accuracy | 0.8711 | 0.8401 | 0.8514 | 0.8985 |
| F1 | 0.9204 | 0.8999 | 0.9082 | 0.9407 |

Raw validation behavior worsens on both positive recall and negative rejection. Test shifts toward higher recall but weaker specificity: counts change from TN=188, FP=28, FN=212, TP=1187 to TN=151, FP=65, FN=99, TP=1300. It recovers 113 positives for 37 additional false positives. Raw test accuracy/F1/MCC improve over the baseline's default operating point, but balanced accuracy and ranking decline.

Against residue dropout at the default threshold, masking adds 12 test false positives and 14 false negatives. It is worse on both sides of the confusion matrix, rather than offering a better recall/specificity tradeoff at that threshold.

## Validation-Fitted Thresholds

Masking selected threshold 0.11 using validation F1; both comparison runs selected 0.05. All thresholds were applied unchanged to test, never fitted there. The comparison thresholds are at the lower search-grid boundary, not demonstrated unrestricted optima. Calibrated validation scores are fitting-set results.

| Metric | Validation baseline | Validation dropout | Validation masking | Test baseline | Test dropout | Test masking |
| --- | --- | --- | --- | --- | --- | --- |
| Balanced accuracy | 0.9046 | 0.8511 | 0.8662 | 0.8692 | 0.8155 | 0.7953 |
| Specificity / negative recall | 0.8935 | 0.7315 | 0.8426 | 0.8148 | 0.6667 | 0.6343 |
| Positive recall | 0.9156 | 0.9707 | 0.8898 | 0.9235 | 0.9643 | 0.9564 |
| MCC | 0.6985 | 0.7271 | 0.6142 | 0.6611 | 0.6606 | 0.6130 |
| Accuracy | 0.9126 | 0.9387 | 0.8835 | 0.9090 | 0.9245 | 0.9133 |
| F1 | 0.9478 | 0.9648 | 0.9297 | 0.9462 | 0.9567 | 0.9503 |

Calibrated test counts for masking are TN=137, FP=79, FN=61, TP=1338, versus baseline's TN=176, FP=40, FN=107, TP=1292. Masking recovers 46 positives at the cost of 39 additional false positives. Balanced accuracy declines by 0.0738, specificity by 0.1806, and MCC by 0.0481. F1 improves modestly by 0.0041, so masking is not uniformly worse than baseline on every metric.

Against residue dropout's TN=144, FP=72, FN=50, TP=1349, masking adds seven false positives and eleven false negatives. Dropout leads on all listed calibrated test metrics, including F1 (0.9567 versus 0.9503) and MCC (0.6606 versus 0.6130). Masking's calibrated validation specificity/balanced accuracy advantages over dropout do not transfer to test.

For masking itself, threshold fitting increases test F1 from 0.9407 to 0.9503 but lowers balanced accuracy from 0.8142 to 0.7953 and specificity from 0.6991 to 0.6343. It recovers 38 positives for 14 additional false positives. F1 fitting prioritizes the majority positive class rather than balanced detection. Calibrated positive prediction rate is 87.7% versus actual prevalence of 86.6%; closeness of these rates does not imply reliable predictions.

## Source-Specific Test Behavior

Source reports use the default threshold; calibrated source-specific reports were not saved.

| Source | Metric | Baseline | Residue dropout | Sequence masking |
| --- | --- | --- | --- | --- |
| IntAct positives | Positive recall | 4/25 (0.1600) | 6/25 (0.2400) | 6/25 (0.2400) |
| Negatome | Negative recall | 187/215 (0.8698) | 162/215 (0.7535) | 150/215 (0.6977) |
| PPB | Positive recall | 643/755 (0.8517) | 724/755 (0.9589) | 705/755 (0.9338) |
| SKEMPI | Positive recall | 297/302 (0.9834) | 297/302 (0.9834) | 297/302 (0.9834) |
| STRING | Positive recall | 243/317 (0.7666) | 287/317 (0.9054) | 292/317 (0.9211) |

Masking improves PPB/STRING recall over the baseline but weakens Negatome rejection. Compared with dropout, it recovers five more STRING positives but misses 19 additional PPB positives and incorrectly accepts 12 additional Negatome negatives. IntAct remains weak, missing 19/25 test positives; validation recall is 4/24 versus baseline 3/24 and dropout 5/24. The single IntAct negative is correctly rejected by all three and is too small for a robust source-specific estimate.

## Conclusion

Do not prefer this 10% X-masking setup over either the unaugmented baseline for balanced detection or residue dropout for the recall/F1 tradeoff. No augmentation remains strongest on validation/test AUROC and calibrated test balanced accuracy, specificity, and MCC. Residue dropout remains the more promising augmentation candidate, with better test ranking and fewer false positives and false negatives than sequence masking at both reported operating points.

The result does not establish that every sequence-masking rate or strategy fails. Live-versus-cached training and changed batch budgets prevent attributing the full difference to masking alone, and no matched live unmasked control is saved. Runtime/epoch histories and representation diagnostics are unavailable; do not infer a specific optimization failure, representation collapse, or measured slowdown from these aggregates. Meaningful ranking signal remains, with test AUPRC above the approximate prevalence baseline of 0.8663.

Evidence is single-seed, with repeated test inspection across ablations; a fresh final holdout is preferable for an unbiased final estimate. No code changes are made by this analysis: sequence masking remains active and residue dropout remains disabled pending a separate decision. README, existing source edits, and historical run artifacts are unchanged.

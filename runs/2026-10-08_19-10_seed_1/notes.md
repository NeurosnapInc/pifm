# Run 2026-10-08_19-10_seed_1

Interaction-only training with a frozen ProstT5 backbone and adapters.

## Experiment

Replaced group max pooling with DeepSets: shared per-chain MLP (`phi`), masked sum aggregation, and output MLP (`rho`). Both MLPs use a 64-dimensional bottleneck and preserve the 1,024-dimensional embedding width. Singleton groups also pass through the learned encoder. Residue max pooling was retained. Code revision: `7898a7352b5e3aec26a0377018b6481d035a5a26`.

Saved settings match group mean (`2026-10-08_01-46_seed_1`) and group max (`2026-10-08_06-44_seed_1`), including seed 1. The recorded code diff against max changes only group encoding and related documentation. All variants have 1,614 validation and 1,615 test pairs with matching source counts. Exact split membership is not fingerprinted, so comparison assumes reuse of the same tokenized cache. Group attention (`2026-10-07_05-36_seed_1`) is included for context.

This adds learned capacity and transforms singleton groups that mean/max leave unchanged; it is not merely a reduction-operator change. Sum aggregation also retains chain multiplicity. Aggregate results cannot separate these factors.

## Ranking Across Group-Pooling Variants

| Metric | Attention | Mean | Max | DeepSets |
| --- | --- | --- | --- | --- |
| Validation AUROC | 0.9414 | 0.9551 | 0.9298 | 0.8125 |
| Validation AUPRC | 0.9861 | 0.9899 | 0.9840 | 0.9555 |
| Test AUROC | 0.9297 | 0.9356 | 0.9309 | 0.8597 |
| Test AUPRC | 0.9843 | 0.9856 | 0.9852 | 0.9721 |

DeepSets has the weakest ranking of all four group variants. Relative to mean, validation AUROC declines by 0.1426 and test AUROC by 0.0759. This is not merely a threshold issue: ranking metrics are threshold-independent.

## Default-Threshold Results

| Metric | Validation mean | Validation DeepSets | Test mean | Test DeepSets |
| --- | --- | --- | --- | --- |
| Balanced accuracy | 0.9002 | 0.5825 | 0.8594 | 0.5812 |
| Specificity / negative recall | 0.9398 | 0.2407 | 0.8704 | 0.1759 |
| Positive recall | 0.8605 | 0.9242 | 0.8485 | 0.9864 |
| MCC | 0.6322 | 0.1889 | 0.5668 | 0.2995 |
| Accuracy | 0.8711 | 0.8327 | 0.8514 | 0.8780 |
| F1 | 0.9204 | 0.9054 | 0.9082 | 0.9334 |

DeepSets predicts 96.5% of test pairs positive versus actual prevalence of 86.6%. Its test counts are TN=38, FP=178, FN=19, TP=1380, versus mean's TN=188, FP=28, FN=212, TP=1187. It recovers 193 additional positives but accepts 150 additional negatives incorrectly.

Higher test accuracy/F1 do not indicate better balanced detection. An always-positive classifier already achieves test accuracy 0.8663 and F1 0.9283; DeepSets' raw F1 is only 0.9334. Its AUPRC remains above the prevalence baseline of approximately 0.8663, so ranking signal exists, but it is weaker than the simpler encoders.

## Validation-Fitted Thresholds

DeepSets selected threshold 0.87 using validation F1; mean selected 0.05. Both were applied unchanged to test. Threshold magnitudes alone do not establish probability calibration quality.

| Metric | Validation mean: calibrated | Validation DeepSets: calibrated | Test mean: calibrated | Test DeepSets: calibrated |
| --- | --- | --- | --- | --- |
| Balanced accuracy | 0.9046 | 0.6077 | 0.8692 | 0.5947 |
| Specificity / negative recall | 0.8935 | 0.2963 | 0.8148 | 0.2130 |
| Positive recall | 0.9156 | 0.9192 | 0.9235 | 0.9764 |
| MCC | 0.6985 | 0.2348 | 0.6611 | 0.2988 |
| Accuracy | 0.9126 | 0.8358 | 0.9090 | 0.8743 |
| F1 | 0.9478 | 0.9065 | 0.9462 | 0.9308 |

Calibration slightly improves DeepSets' test balanced accuracy from 0.5812 to 0.5947, but does not fix negative detection. Calibrated counts are TN=46, FP=170, FN=33, TP=1366. Compared with mean's TN=176, FP=40, FN=107, TP=1292, it recovers 74 additional positives at the cost of 130 additional false positives. Balanced accuracy declines by 0.2745 and MCC by 0.3622.

F1-based threshold selection favors the majority positive class rather than balanced detection. Calibrated validation scores are fitting-set results; no test-fitted threshold is used.

## Source-Specific Test Behavior

These reports use the default threshold; calibrated source-specific reports were not saved.

| Source | Metric | Mean | Max | DeepSets |
| --- | --- | --- | --- | --- |
| IntAct positives | Positive recall | 4/25 (0.1600) | 5/25 (0.2000) | 19/25 (0.7600) |
| Negatome | Negative recall | 187/215 (0.8698) | 183/215 (0.8512) | 37/215 (0.1721) |
| PPB | Positive recall | 643/755 (0.8517) | 691/755 (0.9152) | 750/755 (0.9934) |
| SKEMPI | Positive recall | 297/302 (0.9834) | 297/302 (0.9834) | 302/302 (1.0000) |
| STRING | Positive recall | 243/317 (0.7666) | 246/317 (0.7760) | 309/317 (0.9748) |

IntAct positive recall improves markedly, including validation recall of 14/24 versus mean's 3/24. However, improvement across positive-only sources accompanies the broad shift toward positive predictions: 178 of 215 Negatome negatives are misclassified. This does not demonstrate that source-specific generalization is resolved. All three correctly reject the single IntAct negative, which provides little evidence by itself.

## Conclusion

Do not retain this DeepSets implementation over mean group pooling. It substantially weakens validation selection performance, test ranking, and balanced negative detection. Mean remains the strongest overall candidate so far, retaining residue max pooling. Higher positive recall does not outweigh the false-positive rate under the current objectives.

This result applies to this bottlenecked, sum-based implementation and training setup, not every DeepSets architecture. Aggregate metrics cannot identify whether compression, cardinality effects, optimization, or another factor caused the decline; no runtime comparison is recorded. Evidence is single-seed, and repeated test inspection during ablations makes a fresh final holdout preferable for an unbiased final estimate. No architecture changes are made by this review.

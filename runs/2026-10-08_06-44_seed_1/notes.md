# Run 2026-10-08_06-44_seed_1

Interaction-only training with a frozen ProstT5 backbone and adapters.

## Experiment

Replaced chain-to-group mean pooling with feature-wise max pooling. Residue-to-chain max pooling was retained. Code revision: `c79bb10cb3003e8b727c80a67767691f25c96f38`.

The immediate baseline is group mean (`2026-10-08_01-46_seed_1`); saved settings are identical, including training seed 1, and the model-code diff changes only group pooling. Group attention (`2026-10-07_05-36_seed_1`) is included for context. All three report 1,614 validation and 1,615 test pairs with matching source counts. Exact split membership is not fingerprinted, so the comparison assumes reuse of the same tokenized cache.

## Ranking Across Group-Pooling Variants

| Metric | Attention group | Mean group | Max group |
| --- | --- | --- | --- |
| Validation AUROC | 0.9414 | 0.9551 | 0.9298 |
| Validation AUPRC | 0.9861 | 0.9899 | 0.9840 |
| Test AUROC | 0.9297 | 0.9356 | 0.9309 |
| Test AUPRC | 0.9843 | 0.9856 | 0.9852 |

Mean group pooling retains the best ranking on both splits. Compared with mean, max reduced validation AUROC by 0.0253 and test AUROC by 0.0046. The small test ranking differences do not establish statistical significance from one seed.

## Mean Versus Max: Thresholded Results

| Metric | Validation mean: raw | Validation max: raw | Test mean: raw | Test max: raw |
| --- | --- | --- | --- | --- |
| Balanced accuracy | 0.9002 | 0.8658 | 0.8594 | 0.8687 |
| Specificity / negative recall | 0.9398 | 0.8889 | 0.8704 | 0.8519 |
| Positive recall | 0.8605 | 0.8426 | 0.8485 | 0.8856 |
| MCC | 0.6322 | 0.5712 | 0.5668 | 0.6131 |
| Accuracy | 0.8711 | 0.8488 | 0.8514 | 0.8811 |
| F1 | 0.9204 | 0.9062 | 0.9082 | 0.9281 |

Default-threshold test performance improved on balanced accuracy, MCC, accuracy, and F1, despite slightly lower specificity. Test counts changed from TN=188, FP=28, FN=212, TP=1187 to TN=184, FP=32, FN=160, TP=1239: 52 fewer missed positives at the cost of four additional false positives. This is a real operating-point tradeoff in these results, so max should not be described as worse on every metric. Validation thresholded performance, however, declined.

## Validation-Fitted Thresholds

Mean selected threshold 0.05 and max selected 0.07 using validation F1. Each was applied unchanged to test.

| Metric | Validation mean: calibrated | Validation max: calibrated | Test mean: calibrated | Test max: calibrated |
| --- | --- | --- | --- | --- |
| Balanced accuracy | 0.9046 | 0.8795 | 0.8692 | 0.8398 |
| Specificity / negative recall | 0.8935 | 0.8333 | 0.8148 | 0.7546 |
| Positive recall | 0.9156 | 0.9256 | 0.9235 | 0.9249 |
| MCC | 0.6985 | 0.6786 | 0.6611 | 0.6218 |
| Accuracy | 0.9126 | 0.9133 | 0.9090 | 0.9022 |
| F1 | 0.9478 | 0.9487 | 0.9462 | 0.9425 |

The calibrated test comparison favors mean. Max reduced balanced accuracy by 0.0294, specificity by 0.0602, and MCC by 0.0393. Confusion counts changed from TN=176, FP=40, FN=107, TP=1292 to TN=163, FP=53, FN=105, TP=1294. It recovered only two additional positives while incorrectly accepting 13 additional negatives.

For max itself, the F1-selected threshold reduced test balanced accuracy from 0.8687 to 0.8398 and specificity from 0.8519 to 0.7546, while increasing positive recall and slightly increasing MCC. Threshold selection optimizes validation F1 rather than balanced detection; the saved threshold is not evidence of the best operating point for every objective. Calibrated validation scores are fitting-set results.

## Source-Specific Test Behavior

These source reports use the default threshold; calibrated source-specific reports were not saved.

- PPB positive recall improved from 643/755 (0.8517) with mean to 691/755 (0.9152) with max.
- STRING positive recall improved slightly from 243/317 (0.7666) to 246/317 (0.7760).
- SKEMPI remained 297/302 (0.9834).
- IntAct positive recall improved from 4/25 (0.1600) to 5/25 (0.2000), but still missed 20 of 25 positives. Validation IntAct improved from 3/24 to 4/24; the source-specific weakness remains.
- Negatome negative recall declined from 187/215 (0.8698) to 183/215 (0.8512), consistent with the shift toward more positive predictions.

## Conclusion

Group max does not displace group mean as the strongest overall candidate so far. Mean leads on validation AUROC, the configured selection metric, and on test ranking and calibrated balanced accuracy, specificity, MCC, accuracy, and F1. Max's default-threshold test recall/MCC gains are useful, but do not outweigh its weaker validation ranking and calibrated negative handling under the current workflow.

Prefer max residue pooling plus mean group pooling provisionally while completing the remaining group ablations. This is a single-seed comparison, and the two parameter-free group methods have no demonstrated runtime difference here. IntAct positive recall remains poor for both, and no calibrated source reports establish that threshold adjustment resolves it. No architecture changes are made by this results review.

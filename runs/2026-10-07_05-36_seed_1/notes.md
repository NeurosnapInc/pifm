# Run 2026-10-07_05-36_seed_1

Interaction-only training with a frozen ProstT5 backbone and adapters.

## Experiment

Replaced residue-to-chain mean pooling with masked max pooling. Chain-to-group attention pooling and the interaction-only architecture were unchanged. Code revision: `6aff5eefe5a37762c0dceb50fa9c154b4f599e52`.

The immediate comparison is the mean-pooling run `2026-10-07_02-30_seed_1`; their saved settings are identical, including training seed 1, and their model-code diff only changes residue pooling. The older interaction-only attention baseline is `2026-08-13_seed_1`. All three report 1,614 validation and 1,615 test pairs with matching source counts. Exact split membership is not fingerprinted, so the comparison assumes the tokenized cache was reused. Attention-baseline metrics were imported at four-decimal precision.

## Test Results Before Threshold Calibration

| Metric | Attention | Mean | Max |
| --- | --- | --- | --- |
| AUROC | 0.8353 | 0.8639 | 0.9297 |
| AUPRC | 0.9575 | 0.9646 | 0.9843 |
| Balanced accuracy | 0.6886 | 0.6693 | 0.8470 |
| Specificity / negative recall | 0.4352 | 0.3472 | 0.8935 |
| Positive recall | 0.9421 | 0.9914 | 0.8006 |
| MCC | 0.4131 | 0.5106 | 0.5195 |
| Accuracy | 0.8743 | 0.9053 | 0.8130 |
| F1 | 0.9285 | 0.9477 | 0.8812 |

Max pooling substantially improved ranking and negative detection. Relative to mean pooling, test false positives fell from 141 to 23, but false negatives rose from 12 to 279. Lower accuracy and F1 reflect this shift toward rejecting negatives on a positive-heavy dataset; they do not negate the balanced-accuracy gain. The default threshold is more conservative and sacrifices positive recall.

## Test Results With Validation-Fitted Thresholds

| Metric | Attention | Mean | Max |
| --- | --- | --- | --- |
| Threshold | 0.40 | 0.93 | 0.05 |
| Balanced accuracy | 0.6826 | 0.7167 | 0.8604 |
| Specificity / negative recall | 0.4167 | 0.4583 | 0.8194 |
| Positive recall | 0.9485 | 0.9750 | 0.9014 |
| MCC | 0.4138 | 0.5347 | 0.6192 |
| Accuracy | 0.8774 | 0.9059 | 0.8904 |
| F1 | 0.9306 | 0.9472 | 0.9344 |

Against mean pooling, calibrated test balanced accuracy improved by 0.1437, specificity by 0.3611, and MCC by 0.0845. Confusion counts changed from TN=99, FP=117, FN=35, TP=1364 to TN=177, FP=39, FN=138, TP=1261. This is a much stronger balance between detecting positives and rejecting negatives, although mean pooling retains higher positive recall, accuracy, and F1.

Thresholds were selected on validation using F1 and applied unchanged to test. The max-pooling threshold of 0.05 is the lower boundary of the current search grid, so it is the best tested threshold rather than evidence of an unrestricted optimum. Scores from the different models should not be assumed equally calibrated.

## Validation Results

| Metric | Mean | Max |
| --- | --- | --- |
| AUROC | 0.9087 | 0.9414 |
| AUPRC | 0.9796 | 0.9861 |
| Raw balanced accuracy | 0.7765 | 0.8531 |
| Raw specificity | 0.6296 | 0.9444 |
| Calibrated balanced accuracy | 0.8107 | 0.8849 |
| Calibrated specificity | 0.7037 | 0.8935 |
| Calibrated MCC | 0.5694 | 0.6259 |

Validation ranking and balanced detection improved alongside test results. Calibrated validation metrics are measured on the threshold-fitting split and are not independent validation of the threshold.

## Source-Specific Test Behavior

Source reports use the default threshold; calibrated source-specific reports are not available in the saved metrics.

- Negatome negative recall increased from 74/215 (0.3442) with mean pooling to 192/215 (0.8930) with max pooling; the attention baseline correctly rejected 93/215.
- IntAct positive recall fell from 20/25 (0.8000) to 2/25 (0.0800). Validation IntAct recall also fell from 20/24 to 3/24. These slices are small, but the failure is consistent across splits and should not be hidden by aggregate metrics.
- PPB positive recall fell from 0.9934 to 0.8199 and STRING from 0.9937 to 0.6372. SKEMPI remained high at 0.9834, versus 1.0000 for mean pooling.
- These differences suggest uneven source generalization. The saved aggregate calibrated metrics do not establish that the lower threshold fixes IntAct or STRING behavior.

## Conclusion

Max pooling is the strongest of the three interaction-only pooling candidates for ranking and balanced classification in these runs. Its improvement over mean pooling is much clearer than the earlier mean-versus-attention result, especially for negative detection. Keep max pooling as the current candidate rather than revert it based on lower positive-heavy accuracy or F1 alone.

This remains a single-seed comparison, not proof of a universal advantage. Positive-source recall, particularly IntAct and STRING, is the main unresolved tradeoff. Further comparisons should prioritize AUROC, balanced accuracy, negative recall, MCC, and source-specific performance together; the model is still not uniformly strong across sources.

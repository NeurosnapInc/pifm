# Run 2026-10-07_02-30_seed_1

Interaction-only training with a frozen ProstT5 backbone and adapters.

## Experiment

Replaced residue-to-chain attention pooling with masked mean pooling directly in the model. Chain-to-group attention pooling, the frozen ProstT5 backbone, and the interaction-only task were retained. Code revision: `bda5eef44e534ed37b517cfaab0ac5a11096ee66`.

The relevant baseline is `2026-08-13_seed_1`, the previous interaction-only attention-pooling run. Both runs report 1,614 validation and 1,615 test pairs, with matching source counts. Exact split membership is not fingerprinted in these artifacts; this comparison assumes the tokenized cache was reused. Baseline metrics were imported from historical tables at four-decimal precision.

## Results Before Threshold Calibration

| Metric | Validation: attention | Validation: mean | Test: attention | Test: mean |
| --- | --- | --- | --- | --- |
| AUROC | 0.8964 | 0.9087 | 0.8353 | 0.8639 |
| AUPRC | 0.9775 | 0.9796 | 0.9575 | 0.9646 |
| Balanced accuracy | 0.8079 | 0.7765 | 0.6886 | 0.6693 |
| Specificity / negative recall | 0.6852 | 0.6296 | 0.4352 | 0.3472 |
| Positive recall | 0.9306 | 0.9235 | 0.9421 | 0.9914 |
| MCC | 0.5843 | 0.5266 | 0.4131 | 0.5106 |
| F1 | 0.9404 | 0.9325 | 0.9285 | 0.9477 |

Ranking improved on both splits, but default-threshold balanced accuracy and negative recall declined. On test, mean pooling predicted 94.6% of pairs positive: false negatives fell from 81 to 12, while false positives rose from 122 to 141. The higher F1 reflects this positive-recall gain and does not establish better negative detection.

## Results With Validation-Fitted Thresholds

The saved F1-selected threshold changed from 0.40 for attention pooling to 0.93 for mean pooling. Each threshold was fitted on validation and applied unchanged to test; neither was fitted on test data.

| Metric | Validation: attention | Validation: mean | Test: attention | Test: mean |
| --- | --- | --- | --- | --- |
| Balanced accuracy | 0.8088 | 0.8107 | 0.6826 | 0.7167 |
| Specificity / negative recall | 0.6713 | 0.7037 | 0.4167 | 0.4583 |
| Positive recall | 0.9464 | 0.9177 | 0.9485 | 0.9750 |
| MCC | 0.6129 | 0.5694 | 0.4138 | 0.5347 |
| F1 | 0.9477 | 0.9348 | 0.9306 | 0.9472 |

Calibrated test performance improved: balanced accuracy rose by 0.0341, specificity by 0.0416, and MCC by 0.1209. Test confusion counts changed from TN=90, FP=126, FN=72, TP=1327 to TN=99, FP=117, FN=35, TP=1364. However, 117 of 216 test negatives were still misclassified, so negative detection remains weak. Calibration results on validation are fitting-set results rather than independent evidence.

## Source-Specific Test Behavior

These source reports use the default threshold, not the saved calibrated threshold.

- IntAct positive recall improved from 7/25 (0.2800) to 20/25 (0.8000), but this source slice is small.
- PPB positive recall improved from 0.9483 to 0.9934, SKEMPI from 0.9967 to 1.0000, and STRING from 0.9274 to 0.9937.
- Negatome negative recall declined from 93/215 (0.4326) to 74/215 (0.3442). Stronger positive-source performance does not imply stronger rejection of negatives.

## Conclusion

Mean pooling is a promising, mixed improvement over the interaction-only attention baseline: ranking and calibrated test MCC improved, and the residue pooling layer no longer has learned parameters. It is not an across-the-board win: default-threshold negative handling regressed, validation MCC declined, and calibrated test specificity remains below 0.5.

Keep it provisionally as a simpler candidate rather than declare it definitively better from one seed. The older `2026-08-12_seed_1` multitask run still had stronger test ranking and negative detection (AUROC=0.9515; calibrated balanced accuracy=0.8190 and specificity=0.7361), but that is a different task setup and not a controlled pooling comparison. Further decisions should emphasize balanced accuracy, negative recall, and MCC rather than positive-heavy F1/AUPRC alone.

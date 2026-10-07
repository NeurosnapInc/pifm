# Run 2026-10-07_19-01_seed_1

Interaction-only training with a frozen ProstT5 backbone and adapters.

## Experiment

Tested learned weighted residue-to-chain pooling: a linear token scorer, masked softmax, and a weighted sum, initialized to uniform weighting. Group pooling remained MLP attention. Code revision: `d9ae990d8905284399ea474cb1df55bade0b4c52`.

Comparisons use attention (`2026-08-13_seed_1`), mean (`2026-10-07_02-30_seed_1`), and max (`2026-10-07_05-36_seed_1`). Saved settings are identical across the three October ablations, including seed 1. All four have matching split sizes and source counts: 1,614 validation and 1,615 test pairs. Exact split membership is not fingerprinted, so comparison assumes the same tokenized cache. Attention results have historical four-decimal precision.

## Learned Weighted Results

| Metric | Validation: raw | Validation: calibrated | Test: raw | Test: calibrated |
| --- | --- | --- | --- | --- |
| AUROC | 0.9138 | 0.9138 | 0.8355 | 0.8355 |
| AUPRC | 0.9807 | 0.9807 | 0.9576 | 0.9576 |
| Balanced accuracy | 0.7942 | 0.7854 | 0.6832 | 0.6700 |
| Specificity / negative recall | 0.6713 | 0.6481 | 0.3843 | 0.3565 |
| Positive recall | 0.9170 | 0.9227 | 0.9821 | 0.9836 |
| MCC | 0.5440 | 0.5390 | 0.4992 | 0.4802 |
| Accuracy | 0.8841 | 0.8860 | 0.9022 | 0.8997 |
| F1 | 0.9320 | 0.9334 | 0.9456 | 0.9444 |

The validation-F1-selected threshold was 0.30 and was applied unchanged to test. It reduced test balanced accuracy, specificity, MCC, and F1 relative to the default threshold. Calibrated test counts were TN=77, FP=139, FN=23, TP=1376: almost all positives were recovered, but 139 of 216 negatives were incorrectly called positive.

Validation ranking improved slightly over mean pooling, but test ranking regressed: AUROC fell from 0.8639 to 0.8355, approximately matching the older attention baseline. The validation-to-test AUROC drop was 0.0782. Learned weighting did not deliver a generalization improvement, and its high accuracy/F1 mainly reflects positive recall on a dataset with approximately 86.6% positives.

## Source-Specific Test Behavior

These reports use the default threshold; calibrated source reports were not saved.

- IntAct positive recall was 17/25 (0.6800), below mean's 20/25 but above max's 2/25.
- Negatome negative recall was 83/215 (0.3860), slightly above mean's 74/215 but far below max's 192/215. The single IntAct negative was also misclassified.
- PPB positive recall was 0.9894, SKEMPI 1.0000, and STRING 0.9716. Strong positive-source recall coexisted with weak rejection of negatives.

## Completed Residue-to-Chain Ablations

Ranking scores below are threshold-independent. Thresholded metrics use each run's validation-fitted threshold applied to test.

| Pooling | Validation AUROC | Test AUROC | Test AUPRC | Test balanced accuracy | Test specificity | Test MCC | Test F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Attention | 0.8964 | 0.8353 | 0.9575 | 0.6826 | 0.4167 | 0.4138 | 0.9306 |
| Mean | 0.9087 | 0.8639 | 0.9646 | 0.7167 | 0.4583 | 0.5347 | 0.9472 |
| Max | 0.9414 | 0.9297 | 0.9843 | 0.8604 | 0.8194 | 0.6192 | 0.9344 |
| Learned weighted | 0.9138 | 0.8355 | 0.9576 | 0.6700 | 0.3565 | 0.4802 | 0.9444 |

## Decision

Select max pooling (`2026-10-07_05-36_seed_1`) as the best interaction-only residue-pooling candidate. It leads on validation AUROC, the configured checkpoint-selection metric, and on test AUROC, AUPRC, calibrated balanced accuracy, specificity, and MCC. It also leads on raw test balanced accuracy and specificity, so the advantage is not solely a calibration artifact. Its parameter-free implementation is simpler than learned weighting or MLP attention.

Mean wins on calibrated test accuracy/F1 and positive recall, but those gains come with much weaker negative detection. For the intended interaction classifier, max's balance between classes is the stronger overall outcome. Learned weighting adds parameters without improving the ranking or calibrated balanced detection achieved by mean or max; do not retain it as the preferred architecture.

The decision is provisional evidence from one seed per variant. Max's default-threshold IntAct positive recall remains a serious source-specific weakness, and calibrated source reports are unavailable. These runs do not establish uniform source generalization. Further experiments should use max as the comparison baseline and emphasize both classes rather than accuracy/F1 alone.

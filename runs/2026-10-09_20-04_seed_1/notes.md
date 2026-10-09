# Run 2026-10-09_20-04_seed_1

Interaction-only training with a frozen ProstT5 backbone and adapters.

## Experiment

Replaced the previous class-weighted focal loss with class-weighted binary cross-entropy. The existing two-logit head uses `positive_logit - negative_logit` for BCE, preserving its softmax positive probability and making the objective equivalent to weighted two-class cross-entropy. Loss is normalized by the batch's class-weight sum. Code revision: `166ddb73ad065ff3075c67c88a05e4da984d274e`.

Baseline: focal loss with residue max and group mean pooling (`2026-10-08_01-46_seed_1`). Model behavior is unchanged; the recorded model diff only reorders pooling definitions and updates docstrings. Saved settings differ only in loss name and removal of focal gamma. Seed, class weights, source/class-balanced sampling, optimizer settings, validation selection, and threshold fitting remain unchanged.

Both runs report 1,614 validation and 1,615 test pairs, matching source counts and the same cache path. Exact cache contents/split membership are not fingerprinted, so comparison assumes reuse of the same cache.

## Validation Results

| Metric | Focal: raw | BCE: raw | Focal: calibrated | BCE: calibrated |
| --- | --- | --- | --- | --- |
| AUROC | 0.9551 | 0.9442 | 0.9551 | 0.9442 |
| AUPRC | 0.9899 | 0.9874 | 0.9899 | 0.9874 |
| Balanced accuracy | 0.9002 | 0.8816 | 0.9046 | 0.8816 |
| Specificity / negative recall | 0.9398 | 0.8333 | 0.8935 | 0.8333 |
| Positive recall | 0.8605 | 0.9299 | 0.9156 | 0.9299 |
| MCC | 0.6322 | 0.6882 | 0.6985 | 0.6882 |
| Accuracy | 0.8711 | 0.9170 | 0.9126 | 0.9170 |
| F1 | 0.9204 | 0.9510 | 0.9478 | 0.9510 |

BCE improves default-threshold positive recall, accuracy, F1, and MCC, but weakens ranking and specificity. Validation AUROC, the configured selection metric, declines by 0.0108. After threshold fitting, BCE retains slightly higher recall/F1 but lower balanced accuracy and MCC. Calibrated validation metrics are fitting-set results, not independent evidence of generalization.

## Test Results

| Metric | Focal: raw | BCE: raw | Focal: calibrated | BCE: calibrated |
| --- | --- | --- | --- | --- |
| AUROC | 0.9356 | 0.9138 | 0.9356 | 0.9138 |
| AUPRC | 0.9856 | 0.9793 | 0.9856 | 0.9793 |
| Balanced accuracy | 0.8594 | 0.8140 | 0.8692 | 0.8140 |
| Specificity / negative recall | 0.8704 | 0.6944 | 0.8148 | 0.6944 |
| Positive recall | 0.8485 | 0.9335 | 0.9235 | 0.9335 |
| MCC | 0.5668 | 0.5979 | 0.6611 | 0.5979 |
| Accuracy | 0.8514 | 0.9015 | 0.9090 | 0.9015 |
| F1 | 0.9082 | 0.9426 | 0.9462 | 0.9426 |

Raw test counts change from focal's TN=188, FP=28, FN=212, TP=1187 to BCE's TN=150, FP=66, FN=93, TP=1306: 119 fewer missed positives at the cost of 38 additional false positives. Raw MCC/F1 gains are real operating-point improvements, but not evidence of stronger ranking or balanced detection.

Focal selected threshold 0.05 and BCE selected 0.49 using validation F1; both were applied unchanged to test. BCE's saved calibrated confusion counts match its default-threshold counts on both splits. This means the tested threshold adjustment has no aggregate operating-point effect here, not that its probabilities are necessarily better calibrated. Focal's threshold lies at the lower search-grid boundary, so it is the best tested threshold rather than a proven unrestricted optimum.

Against calibrated focal counts (TN=176, FP=40, FN=107, TP=1292), BCE recovers 14 additional positives but incorrectly accepts 26 additional negatives. Calibrated test balanced accuracy declines by 0.0552, specificity by 0.1204, and MCC by 0.0632. Test AUROC declines by 0.0218 and AUPRC by 0.0063, so a threshold change alone cannot explain the regression.

## Source-Specific Test Behavior

Source reports use the default threshold; calibrated source reports were not saved.

| Source | Metric | Focal | BCE |
| --- | --- | --- | --- |
| IntAct positives | Positive recall | 4/25 (0.1600) | 6/25 (0.2400) |
| Negatome | Negative recall | 187/215 (0.8698) | 149/215 (0.6930) |
| PPB | Positive recall | 643/755 (0.8517) | 720/755 (0.9536) |
| SKEMPI | Positive recall | 297/302 (0.9834) | 297/302 (0.9834) |
| STRING | Positive recall | 243/317 (0.7666) | 283/317 (0.8927) |

BCE improves PPB/STRING recall but weakens Negatome rejection. IntAct improves by only two recovered test positives and remains poor, missing 19/25; validation recall changes from 3/24 to 4/24. Both correctly reject the single IntAct negative, which is too small a slice to establish generalization. All these source labels describe classification, not affinity regression.

## Interpretation And Conclusion

Focal remains the preferred loss baseline overall for residue max/group mean pooling. BCE provides stronger default-threshold positive recovery, but trails on validation selection performance, test ranking, and calibrated balanced accuracy, MCC, accuracy, and F1. The higher raw F1 should not obscure weaker negative handling on this 86.6%-positive test split (always-positive F1 is approximately 0.9283).

This compares two implemented objectives, not BCE versus mathematically equivalent CE. It also changes reduction: historical focal used a batch mean, while BCE divides by the class-weight sum. Historical focal additionally computed `pt = exp(-weighted_ce)`, which equals the true-class probability raised to its class weight, rather than the unweighted true-class probability used by conventional focal modulation. Therefore the result does not isolate gamma alone or establish that conventional focal loss universally beats BCE. A future strictly controlled comparison should account for these differences.

Evidence is single-seed, with no confidence intervals or measured runtime comparison. Repeated test inspection across ablations also makes a fresh final holdout preferable for an unbiased final estimate. No model/loss changes are made by this analysis; code remains on BCE pending a separate decision. README and historical artifacts are unchanged.

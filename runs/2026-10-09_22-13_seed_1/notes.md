# Run 2026-10-09_22-13_seed_1

Interaction-only training with a frozen ProstT5 backbone and adapters.

BCE + pairwise contrastive ablation with residue max and group mean pooling.
Both terms use class weights and weight-sum normalization. Contrastive loss uses
normalized group embeddings, squared cosine distance for positives, and a squared
hinge for negatives (margin 1.0, weight 0.1).

## Comparison Setup

Code revision: `29cd61f3c871fc037eaa74885961c6f740ead5b4`. The direct baseline is BCE-only (`2026-10-09_20-04_seed_1`); the strongest previous loss baseline is focal (`2026-10-08_01-46_seed_1`). All use residue max/group mean pooling, frozen ProstT5 with adapters, seed 1, and the same sampling and training settings. Saved settings differ from BCE-only only in the loss name; contrastive metadata records weight 0.1, cosine distance, margin 1.0, and class weighting.

All three report 1,614 validation and 1,615 test pairs with matching source counts and the same cache path. Exact cache contents/split membership are not fingerprinted, so comparisons assume the same tokenized cache. Contrastive uses supplied pair labels only, not assumed in-batch negatives; it adds no learned projection head or backbone fine-tuning.

## Ranking

| Metric | Focal | BCE | BCE + contrastive |
| --- | --- | --- | --- |
| Validation AUROC | 0.9551 | 0.9442 | 0.8806 |
| Validation AUPRC | 0.9899 | 0.9874 | 0.9771 |
| Test AUROC | 0.9356 | 0.9138 | 0.8557 |
| Test AUPRC | 0.9856 | 0.9793 | 0.9690 |

Adding contrastive reduces validation AUROC by 0.0636 and test AUROC by 0.0581 versus BCE-only. Both ranking metrics decline on both splits, so the degradation is not merely a decision-threshold issue. Test AUPRC remains above the approximate prevalence baseline of 0.8663, but does not offset weak negative detection.

## Default-Threshold Results

| Metric | Validation BCE | Validation BCE + contrastive | Test BCE | Test BCE + contrastive |
| --- | --- | --- | --- | --- |
| Balanced accuracy | 0.8816 | 0.5357 | 0.8140 | 0.5945 |
| Specificity / negative recall | 0.8333 | 0.1065 | 0.6944 | 0.2176 |
| Positive recall | 0.9299 | 0.9649 | 0.9335 | 0.9714 |
| MCC | 0.6882 | 0.1178 | 0.5979 | 0.2850 |
| Accuracy | 0.9170 | 0.8501 | 0.9015 | 0.8706 |
| F1 | 0.9510 | 0.9177 | 0.9426 | 0.9286 |

Test counts change from BCE-only's TN=150, FP=66, FN=93, TP=1306 to TN=47, FP=169, FN=40, TP=1359. Contrastive recovers 53 additional positives but accepts 103 additional negatives incorrectly. It predicts 94.6% of test pairs positive versus an actual prevalence of 86.6%. Raw test F1 of 0.9286 is almost identical to the always-positive baseline of approximately 0.9283.

## Validation-Fitted Thresholds

Thresholds selected by validation F1 are focal 0.05, BCE 0.49, and BCE + contrastive 0.07, each applied unchanged to test. Calibrated validation results are fitting-set results.

| Metric | Validation focal | Validation BCE | Validation BCE + contrastive | Test focal | Test BCE | Test BCE + contrastive |
| --- | --- | --- | --- | --- | --- | --- |
| Balanced accuracy | 0.9046 | 0.8816 | 0.5302 | 0.8692 | 0.8140 | 0.5659 |
| Specificity / negative recall | 0.8935 | 0.8333 | 0.0741 | 0.8148 | 0.6944 | 0.1481 |
| Positive recall | 0.9156 | 0.9299 | 0.9864 | 0.9235 | 0.9335 | 0.9836 |
| MCC | 0.6985 | 0.6882 | 0.1414 | 0.6611 | 0.5979 | 0.2472 |
| Accuracy | 0.9126 | 0.9170 | 0.8643 | 0.9090 | 0.9015 | 0.8718 |
| F1 | 0.9478 | 0.9510 | 0.9264 | 0.9462 | 0.9426 | 0.9300 |

Calibrated test counts are TN=32, FP=184, FN=23, TP=1376. Relative to BCE-only, this recovers 70 positives but introduces 118 additional false positives. Balanced accuracy declines by 0.2481 and MCC by 0.3507. Relative to focal, it recovers 84 positives but introduces 144 additional false positives; balanced accuracy declines by 0.3033.

For this run, F1 threshold fitting further lowers test specificity from 0.2176 to 0.1481 and balanced accuracy from 0.5945 to 0.5659, despite slightly improving F1. Calibrated predictions are 96.6% positive. Optimizing F1 on this positive-heavy dataset does not optimize balanced detection.

## Source-Specific Test Behavior

These source reports use the default threshold; calibrated source reports were not saved.

| Source | Metric | Focal | BCE | BCE + contrastive |
| --- | --- | --- | --- | --- |
| IntAct positives | Positive recall | 4/25 (0.1600) | 6/25 (0.2400) | 15/25 (0.6000) |
| Negatome | Negative recall | 187/215 (0.8698) | 149/215 (0.6930) | 46/215 (0.2140) |
| PPB | Positive recall | 643/755 (0.8517) | 720/755 (0.9536) | 737/755 (0.9762) |
| SKEMPI | Positive recall | 297/302 (0.9834) | 297/302 (0.9834) | 297/302 (0.9834) |
| STRING | Positive recall | 243/317 (0.7666) | 283/317 (0.8927) | 310/317 (0.9779) |

IntAct recall improves notably, including validation recall of 13/24 versus BCE-only's 4/24. However, this accompanies a broad positive prediction shift and misclassification of 169/215 Negatome negatives. It is not evidence that source-generalization problems are solved. All three reject the single IntAct negative, which provides little evidence by itself.

## Conclusion

This BCE + contrastive formulation is unsuccessful under the current objectives. It materially weakens ranking, negative rejection, balanced accuracy, MCC, and F1 versus BCE-only. Focal remains the strongest overall loss baseline with residue max/group mean pooling. Higher positive recall does not justify the false-positive increase.

Results do not establish that all contrastive objectives fail. Pulling interacting proteins toward similar embeddings is an additional assumption, not a necessary property of interaction. Metrics alone cannot distinguish that mismatch from loss-weight, optimization, or representation limitations. Saved metrics contain no component-loss history or embedding-distance diagnostics, so they cannot establish representation collapse or whether the auxiliary term dominated training. Focal-plus-contrastive remains untested; the same auxiliary term is not demonstrated to help by this run.

This is single-seed evidence with no confidence intervals. Repeated test inspection across ablations makes a fresh final holdout preferable for an unbiased final estimate. Comparing to historical focal also retains the normalization and weighted-probability caveats documented in the BCE-only notes. No training/model changes are made by this analysis; code remains on BCE + contrastive. README and historical artifacts are unchanged.

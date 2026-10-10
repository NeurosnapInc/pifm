# Run 2026-10-10_00-21_seed_1

Interaction-only training with a frozen ProstT5 backbone and adapters.

Class-weighted focal + pairwise contrastive ablation with residue max and group mean pooling.
Historical focal uses gamma 2.0, exp(-weighted CE) modulation, and batch-mean reduction.
Contrastive retains class-weight-sum normalization and uses
normalized group embeddings, squared cosine distance for positives, and a squared
hinge for negatives (margin 1.0, weight 0.1).

## Comparison Setup

Code revision: `1f3639048f3bb210b18c2d28fd2ce92ccfa335fe`. The direct baseline is class-weighted focal alone (`2026-10-08_01-46_seed_1`). BCE (`2026-10-09_20-04_seed_1`) and BCE + contrastive (`2026-10-09_22-13_seed_1`) complete the four loss ablations.

All use residue max/group mean pooling, seed 1, and the same saved sampling, optimizer, and validation-selection settings. Focal gamma remains 2.0, now embedded in the loss and recorded in run metadata rather than as a config setting. The auxiliary formulation, weight 0.1, and cosine margin 1.0 are unchanged from BCE + contrastive. All report 1,614 validation and 1,615 test pairs with matching source counts and cache paths; exact cache contents/split membership are not fingerprinted, so comparisons assume the same tokenized cache.

## Completed Loss Ablations

| Metric | Focal | BCE | BCE + contrastive | Focal + contrastive |
| --- | --- | --- | --- | --- |
| Validation AUROC | 0.9551 | 0.9442 | 0.8806 | 0.8987 |
| Validation AUPRC | 0.9899 | 0.9874 | 0.9771 | 0.9764 |
| Test AUROC | 0.9356 | 0.9138 | 0.8557 | 0.9168 |
| Test AUPRC | 0.9856 | 0.9793 | 0.9690 | 0.9824 |
| Calibrated test balanced accuracy | 0.8692 | 0.8140 | 0.5659 | 0.7984 |
| Calibrated test specificity | 0.8148 | 0.6944 | 0.1481 | 0.6389 |
| Calibrated test positive recall | 0.9235 | 0.9335 | 0.9836 | 0.9578 |
| Calibrated test MCC | 0.6611 | 0.5979 | 0.2472 | 0.6206 |
| Calibrated test accuracy | 0.9090 | 0.9015 | 0.8718 | 0.9152 |
| Calibrated test F1 | 0.9462 | 0.9426 | 0.9300 | 0.9514 |

Focal alone leads on validation ranking, test ranking, and calibrated test balanced accuracy, specificity, and MCC. Focal + contrastive has the highest test F1/accuracy, so it should not be described as uniformly worse. BCE ranks second on validation AUROC and calibrated test balanced accuracy; focal + contrastive ranks second on test AUROC/AUPRC and calibrated MCC/F1. There is no single metric-independent ranking of the middle two variants.

## Direct Focal Comparison

| Metric | Validation focal: raw | Validation focal + contrastive: raw | Test focal: raw | Test focal + contrastive: raw |
| --- | --- | --- | --- | --- |
| Balanced accuracy | 0.9002 | 0.8076 | 0.8594 | 0.8069 |
| Specificity / negative recall | 0.9398 | 0.6574 | 0.8704 | 0.6574 |
| Positive recall | 0.8605 | 0.9578 | 0.8485 | 0.9564 |
| MCC | 0.6322 | 0.6344 | 0.5668 | 0.6302 |
| Accuracy | 0.8711 | 0.9176 | 0.8514 | 0.9164 |
| F1 | 0.9204 | 0.9527 | 0.9082 | 0.9520 |

At the default threshold, contrastive recovers many positives but lowers negative rejection. Test counts change from TN=188, FP=28, FN=212, TP=1187 to TN=142, FP=74, FN=61, TP=1338: 151 fewer missed positives for 46 additional false positives. Raw MCC/F1 improve while balanced accuracy declines. Validation AUROC declines by 0.0564 and test AUROC by 0.0188, establishing that this is not merely a threshold shift.

## Validation-Fitted Thresholds

Focal selected threshold 0.05 and focal + contrastive selected 0.33 using validation F1; each was applied unchanged to test. Calibrated validation scores are fitting-set results. Focal's threshold is the lower search-grid boundary, not a proven unrestricted optimum.

| Metric | Validation focal: calibrated | Validation focal + contrastive: calibrated | Test focal: calibrated | Test focal + contrastive: calibrated |
| --- | --- | --- | --- | --- |
| Balanced accuracy | 0.9046 | 0.8002 | 0.8692 | 0.7984 |
| Specificity / negative recall | 0.8935 | 0.6296 | 0.8148 | 0.6389 |
| Positive recall | 0.9156 | 0.9707 | 0.9235 | 0.9578 |
| MCC | 0.6985 | 0.6541 | 0.6611 | 0.6206 |
| Accuracy | 0.9126 | 0.9250 | 0.9090 | 0.9152 |
| F1 | 0.9478 | 0.9573 | 0.9462 | 0.9514 |

Calibrated test counts change from TN=176, FP=40, FN=107, TP=1292 to TN=138, FP=78, FN=59, TP=1340. Contrastive recovers 48 positives at the cost of 38 additional false positives. Balanced accuracy declines by 0.0708, specificity by 0.1759, and MCC by 0.0404, while F1 rises by 0.0052.

For focal + contrastive itself, threshold fitting slightly worsens test F1 (0.9520 to 0.9514), balanced accuracy (0.8069 to 0.7984), and MCC (0.6302 to 0.6206). It adds four false positives for two fewer false negatives. Validation F1 gains therefore do not establish a better test operating point.

Against BCE + contrastive, the focal combination is substantially stronger: calibrated test balanced accuracy rises from 0.5659 to 0.7984, specificity from 0.1481 to 0.6389, and MCC from 0.2472 to 0.6206. This does not demonstrate that the auxiliary term helps focal: the direct focal-only comparison remains unfavorable on ranking and balanced detection.

## Source-Specific Test Behavior

These reports use the default threshold; calibrated source reports were not saved.

| Source | Metric | Focal | Focal + contrastive |
| --- | --- | --- | --- |
| IntAct positives | Positive recall | 4/25 (0.1600) | 8/25 (0.3200) |
| Negatome | Negative recall | 187/215 (0.8698) | 141/215 (0.6558) |
| PPB | Positive recall | 643/755 (0.8517) | 730/755 (0.9669) |
| SKEMPI | Positive recall | 297/302 (0.9834) | 297/302 (0.9834) |
| STRING | Positive recall | 243/317 (0.7666) | 303/317 (0.9558) |

IntAct validation recall also improves from 3/24 to 6/24, but most IntAct positives remain missed. PPB/STRING gains accompany 46 additional misclassified Negatome negatives. The single IntAct negative is correctly rejected by both and provides little evidence alone. Source-specific gains do not imply uniformly improved interaction generalization.

## Final Loss Selection

Prefer **class-weighted focal alone**, with residue max/group mean pooling. It wins on validation AUROC, the configured selection criterion, and retains the strongest test ranking and calibrated balance/MCC. Focal + contrastive is a reasonable positive-recall/F1 tradeoff in these measurements, but does not displace focal under the current priorities. Neither tested contrastive combination improves ranking and balanced detection over its own classification-only baseline.

This conclusion applies to this normalized cosine pair-distance formulation at weight 0.1 and margin 1.0, not every contrastive method. Interaction does not require embedding similarity; aggregate metrics cannot determine whether that assumption, optimization, or auxiliary scale caused the decline. Component-loss histories and distance diagnostics are not saved, so representation collapse or loss dominance cannot be established.

Historical focal uses exp(-weighted CE) rather than the conventional unweighted true-class probability, and batch-mean rather than BCE's class-weight-sum reduction. Those differences are preserved for comparability, not corrected in this ablation. Single-seed evidence and repeated test inspection warrant caution and a fresh final holdout for an unbiased final estimate. No code changes are made by this review; training remains focal + contrastive pending an explicit revert.

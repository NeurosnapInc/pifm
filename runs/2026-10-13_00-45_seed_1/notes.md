# Run 2026-10-13_00-45_seed_1

Frozen ProstT5 + adapters with residue max pooling.

Restored winning architecture: residue max pooling, equal-weight group mean
pooling, and fixed symmetric sum/absolute-difference/elementwise-product
features followed by the pair MLP and classifier. No learned interaction encoder.
Data augmentation ablation: training-only residue-embedding dropout at 10%.
Exclude amino-acid positions from max pooling after the adapter; preserve prefix/EOS
and at least one residue per chain. The backbone input and cached embeddings
are unchanged; validation/test use all original token positions.
Historical class-weighted focal loss only (gamma 2, exp(-weighted CE), batch mean).
Frozen backbone embedding caches are reusable; no LoRA or contrastive term.
Eager execution and numerical safeguards are retained after the ablations.
Non-finite inputs/logits/loss/gradients abort training.
Each validation-AUROC improvement is checkpointed immediately.

## Comparison Setup

Code revision: `d92f9dd83ee3a468124fefcf813c417801c9b3b5`. Compare against the same architecture without augmentation (`2026-10-08_01-46_seed_1`): frozen ProstT5 + adapters, residue max/group mean pooling, fixed symmetric pair features, and historical focal-only loss with gamma 2.

This is 10% dropout of residue embeddings from training-time max pooling after the adapter, not deletion/masking of the backbone input sequence. Prefix/EOS remain eligible for pooling, at least one real residue is retained per chain, and validation/test pooling is unchanged. Cached embeddings still contain the original full-sequence contextual information.

Saved sampling, optimizer, token-budget, and seed settings match apart from removal of the unused focal gamma config field (gamma remains 2 in metadata). The current run additionally uses the retained eager execution/numerical checks/checkpoint improvements; baseline used the earlier compilation path. Thus the comparison is not perfectly isolated from execution-path changes. Both report 1,614 validation and 1,615 test pairs with matching source counts/cache paths; exact split/cache fingerprints are absent, so comparison assumes reuse of the same cache. Both are seed 1, not independent replicates.

## Validation Comparison

| Metric | Baseline: raw | Residue dropout: raw | Baseline: calibrated | Residue dropout: calibrated |
| --- | --- | --- | --- | --- |
| AUROC | 0.9551 | 0.9486 | 0.9551 | 0.9486 |
| AUPRC | 0.9899 | 0.9883 | 0.9899 | 0.9883 |
| Balanced accuracy | 0.9002 | 0.8772 | 0.9046 | 0.8511 |
| Specificity / negative recall | 0.9398 | 0.8194 | 0.8935 | 0.7315 |
| Positive recall | 0.8605 | 0.9349 | 0.9156 | 0.9707 |
| MCC | 0.6322 | 0.6902 | 0.6985 | 0.7271 |
| Accuracy | 0.8711 | 0.9195 | 0.9126 | 0.9387 |
| F1 | 0.9204 | 0.9526 | 0.9478 | 0.9648 |

Validation AUROC, the checkpoint-selection criterion, declines by 0.0065, but recall, accuracy, F1, and MCC improve at both reported operating points. Balanced accuracy and specificity decline. Calibrated validation counts change from TN=193, FP=23, FN=118, TP=1280 to TN=158, FP=58, FN=41, TP=1357: 77 fewer missed positives for 35 additional false positives. Calibrated validation scores are fitting-set results, not independent threshold-generalization evidence.

## Test Comparison

| Metric | Baseline: raw | Residue dropout: raw | Baseline: calibrated | Residue dropout: calibrated |
| --- | --- | --- | --- | --- |
| AUROC | 0.9356 | 0.9324 | 0.9356 | 0.9324 |
| AUPRC | 0.9856 | 0.9844 | 0.9856 | 0.9844 |
| Balanced accuracy | 0.8594 | 0.8469 | 0.8692 | 0.8155 |
| Specificity / negative recall | 0.8704 | 0.7546 | 0.8148 | 0.6667 |
| Positive recall | 0.8485 | 0.9392 | 0.9235 | 0.9643 |
| MCC | 0.5668 | 0.6551 | 0.6611 | 0.6606 |
| Accuracy | 0.8514 | 0.9146 | 0.9090 | 0.9245 |
| F1 | 0.9082 | 0.9501 | 0.9462 | 0.9567 |

At the default threshold, counts change from TN=188, FP=28, FN=212, TP=1187 to TN=163, FP=53, FN=85, TP=1314. Dropout recovers 127 additional positives for 25 additional false positives, substantially improving raw MCC/F1 while slightly lowering balanced accuracy. This is a useful operating-point tradeoff, not a uniformly worse model.

After calibration, counts change from TN=176, FP=40, FN=107, TP=1292 to TN=144, FP=72, FN=50, TP=1349. Dropout recovers 57 additional positives for 32 additional false positives. Positive recall rises by 0.0407, accuracy by 0.0155, and F1 by 0.0106. Balanced accuracy declines by 0.0537 and specificity by 0.1481. Calibrated MCC is effectively tied (0.660567 versus 0.661062, difference -0.0005).

Test AUROC/AUPRC decline only modestly (0.0032/0.0012), unlike the large regressions seen in several learned interaction ablations. Single-seed results do not establish significance for these small ranking/MCC differences. The model retains meaningful negative detection rather than predicting virtually everything positive: calibrated positive prediction rate is 88.0%, against actual prevalence of 86.6% (baseline predicts 82.5%).

## Threshold Effects

Both runs selected threshold 0.05 using validation F1 and applied it unchanged to test. This is the lower boundary of the search grid, not a demonstrated unrestricted optimum. Threshold magnitude alone does not establish probability calibration quality.

For dropout, this threshold increases test F1 from 0.9501 to 0.9567 but lowers balanced accuracy from 0.8469 to 0.8155 and specificity from 0.7546 to 0.6667. It recovers 35 positives for 19 additional false positives relative to the default threshold. The saved threshold optimizes validation F1 rather than balanced detection; a different validation-only selection objective would be a separate controlled experiment, not something to optimize on this test split.

## Source-Specific Test Behavior

These reports use the default threshold; calibrated source-specific reports were not saved.

| Source | Metric | Baseline | Residue dropout |
| --- | --- | --- | --- |
| IntAct positives | Positive recall | 4/25 (0.1600) | 6/25 (0.2400) |
| Negatome | Negative recall | 187/215 (0.8698) | 162/215 (0.7535) |
| PPB | Positive recall | 643/755 (0.8517) | 724/755 (0.9589) |
| SKEMPI | Positive recall | 297/302 (0.9834) | 297/302 (0.9834) |
| STRING | Positive recall | 243/317 (0.7666) | 287/317 (0.9054) |

PPB/STRING positive recall improves strongly, while Negatome rejection weakens. IntAct improves by two recovered test positives and validation recall rises from 3/24 to 5/24, but 19/25 test positives remain missed. Both correctly reject the single IntAct negative; that slice is too small for a strong source-specific conclusion. Aggregate improvement in recall/F1 does not resolve source-generalization weaknesses.

## Conclusion

Residue-embedding dropout is a promising recall/F1 tradeoff, not a clear overall win or failure. It improves calibrated test F1/accuracy and nearly preserves MCC, with only small ranking losses. However, the no-augmentation baseline remains preferable under the current validation-AUROC and balanced-negative-detection priorities because specificity and balanced accuracy are materially stronger.

Do not claim dropout universally improves generalization or that these ranking differences are statistically significant. Its embedding-level formulation, one tested rate, single seed, and execution-path differences limit attribution; effects at other rates or on raw sequence dropout are untested. A repeated matched no-augmentation control would strengthen the comparison. Repeated test inspection across ablations also warrants a fresh final holdout for an unbiased final estimate.

No code changes are made by this analysis; 10% residue-embedding dropout remains active pending a separate decision. README and historical artifacts are unchanged.

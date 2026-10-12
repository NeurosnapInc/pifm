# Run 2026-10-12_16-22_seed_1

Frozen ProstT5 + adapters with residue max pooling.

Interaction ablation: mean-pool chains within each group, then use a shared
64-dimensional projection and 256 learned bilinear features, averaged over
both group orders. Normalized bilinear features replace the fixed elementwise
product; group sum, absolute difference, pair MLP, and classifier remain.
Historical class-weighted focal loss only (gamma 2, exp(-weighted CE), batch mean).
Frozen backbone embedding caches are reusable; no LoRA or contrastive term.
Eager execution is retained; bilinear arithmetic uses float32 under autocast.
Non-finite inputs/logits/loss/gradients abort training.
Each validation-AUROC improvement is checkpointed immediately.

## Comparison Setup

Code revision: `bd28b1a1aa4635d988cea6e73c7e8f7a439821b9`. Compare against the winning fixed pair-feature baseline (`2026-10-08_01-46_seed_1`) and completed cross-attention ablation (`2026-10-11_19-36_seed_1`). All use frozen ProstT5 + adapters, residue max/group mean pooling, historical class-weighted focal loss, and seed 1.

Bilinear replaces the elementwise-product feature with learned, symmetrized cross-feature products after a shared 64-dimensional projection. Its 256 bilinear features are normalized and projected back to the embedding width; sum/absolute-difference features and the classifier remain. Thus this tests projected bilinear interaction plus normalization and added capacity, not an isolated change in multiplication alone. Unlike cross-attention, it operates after group pooling rather than contextualizing individual chains.

Saved settings match cross-attention and differ from baseline only by removal of the unused focal gamma config setting; gamma remains 2.0 in run metadata. Bilinear and cross-attention use eager execution and float32 interaction arithmetic; baseline used the earlier compilation path. All report 1,614 validation and 1,615 test pairs with matching source counts/cache paths. Exact cache contents/split membership are not fingerprinted, so comparison assumes reuse of the same cache.

## Interaction Ablations So Far

| Metric | Fixed pair features | Cross-attention | Bilinear |
| --- | --- | --- | --- |
| Validation AUROC | 0.9551 | 0.8579 | 0.7511 |
| Validation AUPRC | 0.9899 | 0.9622 | 0.9205 |
| Test AUROC | 0.9356 | 0.9054 | 0.7912 |
| Test AUPRC | 0.9856 | 0.9805 | 0.9501 |
| Calibrated test balanced accuracy | 0.8692 | 0.6303 | 0.5600 |
| Calibrated test specificity | 0.8148 | 0.2778 | 0.1250 |
| Calibrated test positive recall | 0.9235 | 0.9828 | 0.9950 |
| Calibrated test MCC | 0.6611 | 0.3995 | 0.2845 |
| Calibrated test accuracy | 0.9090 | 0.8885 | 0.8786 |
| Calibrated test F1 | 0.9462 | 0.9386 | 0.9342 |

Bilinear ranks last of the three on validation/test ranking and calibrated test balanced accuracy, specificity, MCC, accuracy, and F1. Versus baseline, validation AUROC declines by 0.2040 and test AUROC by 0.1444. Versus cross-attention, test AUROC declines by 0.1142 and calibrated balanced accuracy by 0.0703. Higher positive recall is the primary gain, not better overall classification.

## Direct Baseline Comparison

| Metric | Validation baseline: raw | Validation bilinear: raw | Test baseline: raw | Test bilinear: raw |
| --- | --- | --- | --- | --- |
| Balanced accuracy | 0.9002 | 0.6289 | 0.8594 | 0.5310 |
| Specificity / negative recall | 0.9398 | 0.2685 | 0.8704 | 0.0648 |
| Positive recall | 0.8605 | 0.9893 | 0.8485 | 0.9971 |
| MCC | 0.6322 | 0.4224 | 0.5668 | 0.2009 |
| Accuracy | 0.8711 | 0.8928 | 0.8514 | 0.8724 |
| F1 | 0.9204 | 0.9411 | 0.9082 | 0.9312 |

Raw test counts change from TN=188, FP=28, FN=212, TP=1187 to TN=14, FP=202, FN=4, TP=1395. Bilinear recovers 208 positives but introduces 174 additional false positives. It predicts 98.9% of test pairs positive versus actual prevalence of 86.6%.

Raw accuracy/F1 gains therefore should not be mistaken for balanced improvement. An always-positive classifier already achieves test accuracy 0.8663 and F1 approximately 0.9283; bilinear achieves 0.8724/0.9312. Its AUROC/AUPRC remain above random-ranking baselines (0.5 and approximately 0.8663), but ranking is materially weaker than the simpler model. This is not merely a threshold-selection failure.

## Validation-Fitted Thresholds

Bilinear selected threshold 0.79 using validation F1, compared with baseline 0.05 and cross-attention 0.88. Each was applied unchanged to test. Calibrated validation metrics are fitting-set results; baseline's threshold is the search-grid lower boundary, not a proven unrestricted optimum.

| Metric | Validation baseline: calibrated | Validation bilinear: calibrated | Test baseline: calibrated | Test bilinear: calibrated |
| --- | --- | --- | --- | --- |
| Balanced accuracy | 0.9046 | 0.6526 | 0.8692 | 0.5600 |
| Specificity / negative recall | 0.8935 | 0.3194 | 0.8148 | 0.1250 |
| Positive recall | 0.9156 | 0.9857 | 0.9235 | 0.9950 |
| MCC | 0.6985 | 0.4551 | 0.6611 | 0.2845 |
| Accuracy | 0.9126 | 0.8965 | 0.9090 | 0.8786 |
| F1 | 0.9478 | 0.9429 | 0.9462 | 0.9342 |

Calibration improves bilinear test specificity from 0.0648 to 0.1250 and balanced accuracy from 0.5310 to 0.5600, but negative rejection remains very weak. Calibrated test counts are TN=27, FP=189, FN=7, TP=1392. Against baseline's TN=176, FP=40, FN=107, TP=1292, this recovers 100 additional positives at the cost of 149 additional false positives. Balanced accuracy declines by 0.3092 and MCC by 0.3766.

Against calibrated cross-attention, bilinear recovers 17 positives for 33 additional false positives. Even after fitting, it predicts 97.9% of test pairs positive and rejects only 27/216 negatives. F1 fitting on this positive-heavy validation set does not establish a suitable balanced-detection operating point; changing thresholds alone would not restore the lost ranking performance.

## Source-Specific Test Behavior

These reports use the default threshold; calibrated source reports were not saved.

| Source | Metric | Fixed pair features | Cross-attention | Bilinear |
| --- | --- | --- | --- | --- |
| IntAct positives | Positive recall | 4/25 (0.1600) | 22/25 (0.8800) | 25/25 (1.0000) |
| Negatome | Negative recall | 187/215 (0.8698) | 40/215 (0.1860) | 14/215 (0.0651) |
| PPB | Positive recall | 643/755 (0.8517) | 753/755 (0.9974) | 752/755 (0.9960) |
| SKEMPI | Positive recall | 297/302 (0.9834) | 302/302 (1.0000) | 302/302 (1.0000) |
| STRING | Positive recall | 243/317 (0.7666) | 312/317 (0.9842) | 316/317 (0.9968) |

Perfect IntAct/SKEMPI positive recall accompanies near-universal positive predictions, not demonstrated source-generalization success. Bilinear misclassifies 201/215 Negatome negatives and the single IntAct negative. Validation IntAct recall is 23/24, matching cross-attention, while Negatome rejection is 58/216. The lone IntAct negative is insufficient for a robust source-specific estimate.

## Conclusion

Do not retain this bilinear implementation over the fixed symmetric pair-feature baseline. It has the weakest ranking and balanced test detection among the interaction modules tested so far. The preferred architecture remains frozen ProstT5 + adapters, residue max/group mean pooling, fixed sum/absolute-difference/elementwise-product features, and historical focal-only loss.

This result applies to this projected, normalized bilinear block and training setup, not all bilinear architectures. Saved aggregates cannot identify whether projection, normalization, added capacity, optimization, or source effects caused the decline. Finite metrics do not establish training stability throughout, and no full loss/gradient history or comparable runtime measurements are saved. Neither representation collapse nor a specific numerical bug is demonstrated by these results.

Findings are single-seed evidence, and repeated test inspection across ablations warrants a fresh final holdout for an unbiased final estimate. No code changes are made by this analysis; the current model remains bilinear pending a separate decision. README and historical artifacts are unchanged.

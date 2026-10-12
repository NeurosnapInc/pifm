# Run 2026-10-11_19-36_seed_1

Frozen ProstT5 + adapters with residue max pooling.

Interaction ablation: shared bidirectional chain-level cross-attention between groups
(256-dimensional internal width, four heads, no positional encodings), followed by
group mean pooling and the existing symmetric pair-feature classifier.
Historical class-weighted focal loss only (gamma 2, exp(-weighted CE), batch mean).
Frozen backbone embedding caches are reusable; no LoRA or contrastive term.
Variable chain-set packing uses eager execution rather than torch.compile.
Cross-attention uses float32 math attention; non-finite inputs/logits/loss/gradients abort training.
Each validation-AUROC improvement is checkpointed immediately.

## Run Identity And Interrupted Attempt

This is the completed cross-attention run, revision `2cbfb103c1d21e4c40c29b04c22d3bac59ea72c9`. The older directory `2026-10-11_18-21_seed_1` was the interrupted attempt from revision `d37b99794c0fbe924b5273ad1e0aef679a53dfe6`: it retained the epoch-1 checkpoint before the second improvement hit the existing-artifact guard. The newer revision changes checkpoint update handling, not the model architecture or training settings.

These were not duplicate artifacts: all 35 saved tensors differed, calibration thresholds were 0.92 versus 0.88, and the older attempt had no saved validation/test split reports. Its saved training-best validation AUROC was 0.7861 versus this run's 0.8579. The older epoch-1 directory was deleted at the user's explicit request after comparison, not because it was a duplicate. This is a restart, not an independent seed replicate, and the interrupted checkpoint is not included as a completed ablation.

## Comparison Setup

Baseline: frozen ProstT5 + adapters, residue max/group mean pooling, and historical class-weighted focal alone (`2026-10-08_01-46_seed_1`). The new model adds shared bidirectional cross-attention between chain sets (256-dimensional internal width, four heads), then mean-pools contextualized chains before the symmetric pair-feature classifier. It adds learned capacity as well as intergroup conditioning.

Sampling, seed 1, optimizer settings, and token budgets match the baseline. Focal gamma remains 2.0 with historical weighted-CE modulation and batch-mean reduction. Cross-attention uses float32 math attention and eager execution; baseline training used the previous compilation path. Both reuse frozen backbone embeddings and report 1,614 validation and 1,615 test pairs with matching source counts/cache paths. Exact cache contents/split membership are not fingerprinted, so comparison assumes the same cache.

## Validation Comparison

| Metric | Baseline: raw | Cross-attention: raw | Baseline: calibrated | Cross-attention: calibrated |
| --- | --- | --- | --- | --- |
| AUROC | 0.9551 | 0.8579 | 0.9551 | 0.8579 |
| AUPRC | 0.9899 | 0.9622 | 0.9899 | 0.9622 |
| Balanced accuracy | 0.9002 | 0.6189 | 0.9046 | 0.6702 |
| Specificity / negative recall | 0.9398 | 0.2500 | 0.8935 | 0.3611 |
| Positive recall | 0.8605 | 0.9878 | 0.9156 | 0.9793 |
| MCC | 0.6322 | 0.3949 | 0.6985 | 0.4658 |
| Accuracy | 0.8711 | 0.8891 | 0.9126 | 0.8965 |
| F1 | 0.9204 | 0.9391 | 0.9478 | 0.9425 |

Validation AUROC, the selection metric, declines by 0.0972. Raw accuracy/F1 increase because the model recovers more positives, but negative rejection deteriorates. Calibrated validation counts change from TN=193, FP=23, FN=118, TP=1280 to TN=78, FP=138, FN=29, TP=1369: 89 fewer missed positives for 115 additional false positives. These calibrated validation scores are fitting-set results.

## Test Comparison

| Metric | Baseline: raw | Cross-attention: raw | Baseline: calibrated | Cross-attention: calibrated |
| --- | --- | --- | --- | --- |
| AUROC | 0.9356 | 0.9054 | 0.9356 | 0.9054 |
| AUPRC | 0.9856 | 0.9805 | 0.9856 | 0.9805 |
| Balanced accuracy | 0.8594 | 0.5890 | 0.8692 | 0.6303 |
| Specificity / negative recall | 0.8704 | 0.1852 | 0.8148 | 0.2778 |
| Positive recall | 0.8485 | 0.9929 | 0.9235 | 0.9828 |
| MCC | 0.5668 | 0.3499 | 0.6611 | 0.3995 |
| Accuracy | 0.8514 | 0.8848 | 0.9090 | 0.8885 |
| F1 | 0.9082 | 0.9372 | 0.9462 | 0.9386 |

Raw test counts change from TN=188, FP=28, FN=212, TP=1187 to TN=40, FP=176, FN=10, TP=1389. Cross-attention recovers 202 additional positives but incorrectly accepts 148 additional negatives. It predicts 96.9% of test pairs positive against actual prevalence of 86.6%. Raw F1 0.9372 is only modestly above the always-positive baseline of approximately 0.9283; balanced metrics expose the weak negative handling.

Test AUROC declines by 0.0302 and AUPRC by 0.0052. Ranking signal remains (AUPRC above the approximate prevalence baseline of 0.8663), but is weaker than the simple baseline. Threshold-independent degradation means this is not only an operating-point issue.

## Threshold Selection

Validation-F1-selected thresholds are baseline 0.05 and cross-attention 0.88, applied unchanged to test. Threshold magnitudes do not establish probability calibration quality; baseline's value is the search-grid lower boundary rather than a proven unrestricted optimum.

Cross-attention calibration improves its test specificity from 0.1852 to 0.2778 and balanced accuracy from 0.5890 to 0.6303. However, calibrated test counts remain TN=60, FP=156, FN=24, TP=1375, versus baseline's TN=176, FP=40, FN=107, TP=1292. It recovers 83 additional positives at the cost of 116 additional false positives. Balanced accuracy falls by 0.2389, specificity by 0.5370, and MCC by 0.2616 versus baseline.

Calibrated predictions are still 94.8% positive. F1-based fitting on this positive-heavy validation set does not establish an appropriate threshold for balanced detection. No test-fitted threshold was used.

## Source-Specific Test Behavior

These reports use the default threshold; calibrated source-specific reports were not saved.

| Source | Metric | Baseline | Cross-attention |
| --- | --- | --- | --- |
| IntAct positives | Positive recall | 4/25 (0.1600) | 22/25 (0.8800) |
| Negatome | Negative recall | 187/215 (0.8698) | 40/215 (0.1860) |
| PPB | Positive recall | 643/755 (0.8517) | 753/755 (0.9974) |
| SKEMPI | Positive recall | 297/302 (0.9834) | 302/302 (1.0000) |
| STRING | Positive recall | 243/317 (0.7666) | 312/317 (0.9842) |

IntAct positive recall improves strongly, including validation recall of 23/24 versus 3/24. That gain accompanies near-universal positive predictions, 175/215 misclassified Negatome negatives, and misclassification of the single IntAct negative. It does not establish that source generalization is solved. The lone IntAct negative is too small a slice for a robust source-specific conclusion.

## Conclusion

This cross-attention implementation does not improve overall interaction classification over the simple symmetric pair-feature baseline. Positive recall improves substantially, but validation selection performance, test ranking, calibrated balanced accuracy, MCC, accuracy, and F1 all decline. Prefer the previous adapter + residue max/group mean + focal-only architecture under the current objectives.

The evidence applies to this projected chain-level attention block and training setup, not every cross-attention architecture. Saved aggregates cannot determine whether extra capacity, optimization, source effects, or representation choices caused the decline. Full epoch history, distance diagnostics, and comparable runtime measurements are not available; neither numerical instability nor representation collapse can be inferred from the final metrics alone. The completed run has finite saved metrics after the float32/checkpoint fixes.

Results are single-seed evidence, and repeated test inspection across ablations warrants a fresh final holdout for an unbiased final estimate. No architecture changes are made by this review. README and retained historical artifacts are unchanged.

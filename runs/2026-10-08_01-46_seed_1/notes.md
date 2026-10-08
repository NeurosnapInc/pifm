# Run 2026-10-08_01-46_seed_1

Interaction-only training with a frozen ProstT5 backbone and adapters.

## Experiment

Replaced chain-to-group attention pooling with an equal-weight mean of chain embeddings. Residue-to-chain max pooling was retained. Code revision: `f9e1b1645be209a810c130f410add891439207ff`.

The comparison baseline is `2026-10-07_05-36_seed_1`: max residue pooling plus attention group pooling. The recorded model-code diff changes only group pooling; saved settings differ only by removal of the unused `GROUP_POOL_HIDDEN` setting. Both use training seed 1 and report 1,614 validation and 1,615 test pairs with matching source counts. Exact split membership is not fingerprinted, so the comparison assumes reuse of the same tokenized cache.

## Validation Comparison

| Metric | Attention group: raw | Mean group: raw | Attention group: calibrated | Mean group: calibrated |
| --- | --- | --- | --- | --- |
| AUROC | 0.9414 | 0.9551 | 0.9414 | 0.9551 |
| AUPRC | 0.9861 | 0.9899 | 0.9861 | 0.9899 |
| Balanced accuracy | 0.8531 | 0.9002 | 0.8849 | 0.9046 |
| Specificity / negative recall | 0.9444 | 0.9398 | 0.8935 | 0.8935 |
| Positive recall | 0.7618 | 0.8605 | 0.8763 | 0.9156 |
| MCC | 0.5103 | 0.6322 | 0.6259 | 0.6985 |
| F1 | 0.8606 | 0.9204 | 0.9259 | 0.9478 |

Validation ranking and positive recall improved. At the saved threshold, false positives remained 23 while false negatives fell from 173 to 118. Calibrated validation metrics are measured on the threshold-fitting split and are not independent evidence of threshold generalization.

## Test Comparison

| Metric | Attention group: raw | Mean group: raw | Attention group: calibrated | Mean group: calibrated |
| --- | --- | --- | --- | --- |
| AUROC | 0.9297 | 0.9356 | 0.9297 | 0.9356 |
| AUPRC | 0.9843 | 0.9856 | 0.9843 | 0.9856 |
| Balanced accuracy | 0.8470 | 0.8594 | 0.8604 | 0.8692 |
| Specificity / negative recall | 0.8935 | 0.8704 | 0.8194 | 0.8148 |
| Positive recall | 0.8006 | 0.8485 | 0.9014 | 0.9235 |
| MCC | 0.5195 | 0.5668 | 0.6192 | 0.6611 |
| Accuracy | 0.8130 | 0.8514 | 0.8904 | 0.9090 |
| F1 | 0.8812 | 0.9082 | 0.9344 | 0.9462 |

Mean group pooling improved test AUROC by 0.0059 and calibrated balanced accuracy by 0.0088. Calibrated MCC increased by 0.0419, positive recall by 0.0222, and F1 by 0.0118. Specificity declined slightly, by 0.0046: confusion counts changed from TN=177, FP=39, FN=138, TP=1261 to TN=176, FP=40, FN=107, TP=1292. The main gain is 31 fewer missed positives at the cost of one additional false positive, while retaining strong negative detection.

Both runs selected threshold 0.05 by validation F1 and applied it unchanged to test. That value is the lower boundary of the current search grid, so it is the best tested threshold rather than a demonstrated unrestricted optimum.

## Source-Specific Test Behavior

Source reports use the default threshold; calibrated source-specific results are not available.

- PPB positive recall improved from 619/755 (0.8199) to 643/755 (0.8517).
- STRING positive recall improved from 202/317 (0.6372) to 243/317 (0.7666).
- SKEMPI positive recall remained 297/302 (0.9834).
- IntAct positive recall improved from 2/25 (0.0800) to 4/25 (0.1600), but remains poor. Validation IntAct recall remained 3/24 (0.1250), so the source-specific failure is unresolved.
- Negatome negative recall decreased from 192/215 (0.8930) to 187/215 (0.8698), consistent with the mild shift toward recovering more positives.

## Conclusion

Mean group pooling is a promising improvement over attention group pooling for this run. Validation AUROC, the configured selection metric, improved, and test ranking, balanced accuracy, MCC, accuracy, and F1 also improved. The slight specificity reduction is outweighed by stronger positive recall and overall balanced classification. It removes learned group-pooling parameters; runtime savings were not measured.

Keep max residue pooling plus mean group pooling provisionally as the current baseline while completing the other group-pooling ablations. The test ranking gain is modest and this is a single-seed comparison, so it does not establish a reliable universal advantage. IntAct positive recall remains the main source-specific weakness; aggregate improvements should not be mistaken for uniformly strong source generalization.

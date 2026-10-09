# Run 2026-10-09_00-41_seed_1

Interaction-only training with a frozen ProstT5 backbone and adapters.

## Experiment

Final group-pooling ablation: Set Transformer with two self-attention blocks, four heads, 256-dimensional internal width, and pooling by one learned seed. No positional encodings; singleton groups also use the encoder. Residue max pooling was retained. Code revision: `187e88d97a351ae75ba05672f5bd18f6d968f455`.

Saved settings match mean, max, and DeepSets, including seed 1. Attention's historical settings additionally contain its group hidden-width setting. All five runs have 1,614 validation and 1,615 test pairs with matching source counts. Exact split membership is not fingerprinted, so comparison assumes the same tokenized cache. Learned encoders also add capacity and transform singleton groups, unlike mean/max; this is not solely a reduction-operator comparison.

## Completed Group Ablations

Attention: `2026-10-07_05-36_seed_1`; mean: `2026-10-08_01-46_seed_1`; max: `2026-10-08_06-44_seed_1`; DeepSets: `2026-10-08_19-10_seed_1`; Set Transformer: this run.

| Metric | Attention | Mean | Max | DeepSets | Set Transformer |
| --- | --- | --- | --- | --- | --- |
| Validation AUROC | 0.9414 | 0.9551 | 0.9298 | 0.8125 | 0.9467 |
| Validation AUPRC | 0.9861 | 0.9899 | 0.9840 | 0.9555 | 0.9885 |
| Test AUROC | 0.9297 | 0.9356 | 0.9309 | 0.8597 | 0.9346 |
| Test AUPRC | 0.9843 | 0.9856 | 0.9852 | 0.9721 | 0.9876 |
| Calibrated test balanced accuracy | 0.8604 | 0.8692 | 0.8398 | 0.5947 | 0.8416 |
| Calibrated test specificity | 0.8194 | 0.8148 | 0.7546 | 0.2130 | 0.7546 |
| Calibrated test positive recall | 0.9014 | 0.9235 | 0.9249 | 0.9764 | 0.9285 |
| Calibrated test MCC | 0.6192 | 0.6611 | 0.6218 | 0.2988 | 0.6298 |
| Calibrated test F1 | 0.9344 | 0.9462 | 0.9425 | 0.9308 | 0.9444 |

Mean has the best validation AUROC, the configured checkpoint-selection metric, and test AUROC. Set Transformer is competitive on test ranking and achieves the best test AUPRC, but trails mean on balanced detection after validation-fitted threshold selection. The test AUROC difference (0.0009) is small and does not establish significance from one seed. AUPRC should be interpreted against the test positive prevalence of approximately 0.8663.

## Mean Versus Set Transformer

| Metric | Validation mean: raw | Validation Set Transformer: raw | Test mean: raw | Test Set Transformer: raw |
| --- | --- | --- | --- | --- |
| Balanced accuracy | 0.9002 | 0.8935 | 0.8594 | 0.8414 |
| Specificity / negative recall | 0.9398 | 0.8657 | 0.8704 | 0.7593 |
| Positive recall | 0.8605 | 0.9213 | 0.8485 | 0.9235 |
| MCC | 0.6322 | 0.6915 | 0.5668 | 0.6219 |
| Accuracy | 0.8711 | 0.9139 | 0.8514 | 0.9015 |
| F1 | 0.9204 | 0.9488 | 0.9082 | 0.9420 |

At the default threshold, Set Transformer improves recall, MCC, accuracy, and F1, but lowers specificity and balanced accuracy. Test counts are TN=164, FP=52, FN=107, TP=1292 versus mean's TN=188, FP=28, FN=212, TP=1187: 105 fewer missed positives at the cost of 24 additional false positives. Thus it is not uniformly worse; the default operating points differ.

## Validation-Fitted Thresholds

Set Transformer selected threshold 0.12 using validation F1; mean selected 0.05. Both are applied unchanged to test. Validation calibrated scores are fitting-set results, not independent evidence of threshold generalization.

| Metric | Validation mean: calibrated | Validation Set Transformer: calibrated | Test mean: calibrated | Test Set Transformer: calibrated |
| --- | --- | --- | --- | --- |
| Balanced accuracy | 0.9046 | 0.8928 | 0.8692 | 0.8416 |
| Specificity / negative recall | 0.8935 | 0.8565 | 0.8148 | 0.7546 |
| Positive recall | 0.9156 | 0.9292 | 0.9235 | 0.9285 |
| MCC | 0.6985 | 0.7025 | 0.6611 | 0.6298 |
| Accuracy | 0.9126 | 0.9195 | 0.9090 | 0.9053 |
| F1 | 0.9478 | 0.9523 | 0.9462 | 0.9444 |

Set Transformer's stronger calibrated validation F1/MCC do not carry over to test. Its calibrated test counts are TN=163, FP=53, FN=100, TP=1299, versus mean's TN=176, FP=40, FN=107, TP=1292. It recovers seven additional positives but accepts 13 additional negatives. Test balanced accuracy declines by 0.0276, specificity by 0.0602, and MCC by 0.0313.

Threshold fitting optimizes F1 rather than balanced detection. These comparisons describe the current workflow; they do not establish each model's best possible operating point under another objective.

## Source-Specific Test Behavior

These reports use the default threshold; calibrated source reports were not saved.

| Source | Metric | Mean | Set Transformer |
| --- | --- | --- | --- |
| IntAct positives | Positive recall | 4/25 (0.1600) | 5/25 (0.2000) |
| Negatome | Negative recall | 187/215 (0.8698) | 163/215 (0.7581) |
| PPB | Positive recall | 643/755 (0.8517) | 724/755 (0.9589) |
| SKEMPI | Positive recall | 297/302 (0.9834) | 297/302 (0.9834) |
| STRING | Positive recall | 243/317 (0.7666) | 266/317 (0.8391) |

PPB/STRING recall gains accompany weaker Negatome rejection. IntAct remains a major weakness: Set Transformer misses 20/25 test positives and 20/24 validation positives. The single test IntAct negative is correctly rejected by both, but provides little evidence alone. Aggregate performance does not establish robust behavior across sources.

## Final Selection

Retain **residue max pooling plus group mean pooling**. Mean leads on validation AUROC and provides the strongest calibrated test balance/MCC/F1 overall, while adding no group-encoder parameters. Attention has marginally higher calibrated test specificity, and Set Transformer has higher test AUPRC, but neither improves the overall tradeoff enough to displace mean. Max's raw recall gains do not survive the calibrated comparison, and DeepSets materially weakens negative detection.

The model has been restored to the mean-group implementation from the winning run, including its singleton shortcut and float32 mean accumulation. Set Transformer classes are removed rather than retained behind configuration switches. Historical metrics and weights are unchanged; evaluate learned-encoder checkpoints using their recorded code revisions. Existing tokenized and backbone embedding caches remain reusable.

Selection is provisional single-seed evidence, not a statistical significance claim. Repeated test inspection across ablations also makes a fresh final holdout preferable for an unbiased final estimate. No runtime measurements establish a speedup, though mean removes the learned group encoder's parameters and attention computation.

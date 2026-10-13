# Run 2026-10-12_21-35_seed_1

Frozen ProstT5 + adapters with residue max pooling.

Interaction ablation: a small Transformer over both groups' chain embeddings
(two pre-normalized blocks, four heads, 256-dimensional width). Learned
same-group/cross-group attention biases encode the partition without positional
or group-ID embeddings. Mean-pool contextualized chains per group, then use
the original symmetric sum/absolute-difference/product classifier features.
Historical class-weighted focal loss only (gamma 2, exp(-weighted CE), batch mean).
Frozen backbone embedding caches are reusable; no LoRA or contrastive term.
Eager execution and float32 math attention are retained under autocast.
Non-finite inputs/logits/loss/gradients abort training.
Each validation-AUROC improvement is checkpointed immediately.

## Comparison Setup

Final interaction-module ablation, code revision `4a2d173d6aad2e544d126b77f4c625e9c9bae1b9`. Compare fixed pair features (`2026-10-08_01-46_seed_1`), cross-attention (`2026-10-11_19-36_seed_1`), bilinear (`2026-10-12_16-22_seed_1`), and this chain Transformer. All use frozen ProstT5 + adapters, residue max/group mean pooling, historical focal-only loss (gamma 2), and seed 1.

The Transformer contextualizes all chains jointly before group pooling. Two pre-normalized blocks use four heads, width 256, and learned same-group/cross-group attention biases; no positional/group-ID embeddings. The symmetric sum/absolute-difference/product classifier follows group pooling. This adds capacity and contextualization, not just a different pair reduction.

Sampling, optimizer settings, and token budgets match. The focal gamma config setting was removed after the baseline, but gamma remains 2 in run metadata. Learned interaction runs use eager execution/float32 interaction arithmetic; baseline used the earlier compilation path. All report 1,614 validation and 1,615 test pairs with matching source counts/cache paths. Exact membership/cache contents are not fingerprinted, so comparisons assume the same cache.

## Completed Interaction Ablations

| Metric | Fixed pair features | Cross-attention | Bilinear | Chain Transformer |
| --- | --- | --- | --- | --- |
| Validation AUROC | 0.9551 | 0.8579 | 0.7511 | 0.8554 |
| Validation AUPRC | 0.9899 | 0.9622 | 0.9205 | 0.9677 |
| Test AUROC | 0.9356 | 0.9054 | 0.7912 | 0.9215 |
| Test AUPRC | 0.9856 | 0.9805 | 0.9501 | 0.9844 |
| Calibrated test balanced accuracy | 0.8692 | 0.6303 | 0.5600 | 0.6640 |
| Calibrated test specificity | 0.8148 | 0.2778 | 0.1250 | 0.3380 |
| Calibrated test positive recall | 0.9235 | 0.9828 | 0.9950 | 0.9900 |
| Calibrated test MCC | 0.6611 | 0.3995 | 0.2845 | 0.4945 |
| Calibrated test accuracy | 0.9090 | 0.8885 | 0.8786 | 0.9028 |
| Calibrated test F1 | 0.9462 | 0.9386 | 0.9342 | 0.9464 |

Fixed pair features lead on validation/test ranking and calibrated test balanced accuracy, specificity, MCC, and accuracy. Chain Transformer is the strongest learned module on test ranking and balanced detection, but validation AUROC is slightly below cross-attention. Its calibrated test F1 is marginally highest (0.946361 versus baseline 0.946174), a difference of only 0.0002 that does not establish significance.

## Latest Run Versus Baseline

| Metric | Validation baseline: raw | Validation Transformer: raw | Test baseline: raw | Test Transformer: raw |
| --- | --- | --- | --- | --- |
| Balanced accuracy | 0.9002 | 0.6193 | 0.8594 | 0.6077 |
| Specificity / negative recall | 0.9398 | 0.2500 | 0.8704 | 0.2176 |
| Positive recall | 0.8605 | 0.9886 | 0.8485 | 0.9979 |
| MCC | 0.6322 | 0.3987 | 0.5668 | 0.4234 |
| Accuracy | 0.8711 | 0.8897 | 0.8514 | 0.8935 |
| F1 | 0.9204 | 0.9395 | 0.9082 | 0.9420 |

Raw test counts change from TN=188, FP=28, FN=212, TP=1187 to TN=47, FP=169, FN=3, TP=1396: 209 fewer missed positives at the cost of 141 additional false positives. The Transformer predicts 96.9% of test pairs positive versus actual prevalence of 86.6%. Higher raw accuracy/F1 therefore accompany much weaker negative rejection.

Validation AUROC declines by 0.0997 and test AUROC by 0.0141 against baseline. AUPRC remains close on test (0.9844 versus 0.9856), but weaker ranking and balanced detection mean the change is not simply a threshold issue. Test prevalence gives an approximate random-ranking AUPRC baseline of 0.8663; an always-positive classifier has F1 approximately 0.9283.

## Validation-Fitted Thresholds

Transformer selected threshold 0.95 using validation F1, versus baseline 0.05. These are the upper and lower boundaries of the search grid respectively, not proven unrestricted optima. Both thresholds are applied unchanged to test. Calibrated validation metrics are fitting-set results.

| Metric | Validation baseline: calibrated | Validation Transformer: calibrated | Test baseline: calibrated | Test Transformer: calibrated |
| --- | --- | --- | --- | --- |
| Balanced accuracy | 0.9046 | 0.7035 | 0.8692 | 0.6640 |
| Specificity / negative recall | 0.8935 | 0.4306 | 0.8148 | 0.3380 |
| Positive recall | 0.9156 | 0.9764 | 0.9235 | 0.9900 |
| MCC | 0.6985 | 0.5165 | 0.6611 | 0.4945 |
| Accuracy | 0.9126 | 0.9033 | 0.9090 | 0.9028 |
| F1 | 0.9478 | 0.9459 | 0.9462 | 0.9464 |

Transformer calibration improves its own test balanced accuracy from 0.6077 to 0.6640 and specificity from 0.2176 to 0.3380. Nevertheless, calibrated counts are TN=73, FP=143, FN=14, TP=1385 versus baseline's TN=176, FP=40, FN=107, TP=1292. It recovers 93 additional positives for 103 additional false positives. Balanced accuracy declines by 0.2052, specificity by 0.4769, and MCC by 0.1666.

Relative to calibrated cross-attention, Transformer removes 13 false positives and ten false negatives, improving balanced accuracy by 0.0337 and MCC by 0.0950. It is a better learned-module candidate here, but does not displace the simpler baseline. Nearly equal F1 hides the very different error balance, and F1 fitting does not optimize balanced detection.

## Source-Specific Test Behavior

These reports use the default threshold; calibrated source reports were not saved.

| Source | Metric | Baseline | Chain Transformer |
| --- | --- | --- | --- |
| IntAct positives | Positive recall | 4/25 (0.1600) | 25/25 (1.0000) |
| Negatome | Negative recall | 187/215 (0.8698) | 47/215 (0.2186) |
| PPB | Positive recall | 643/755 (0.8517) | 755/755 (1.0000) |
| SKEMPI | Positive recall | 297/302 (0.9834) | 302/302 (1.0000) |
| STRING | Positive recall | 243/317 (0.7666) | 314/317 (0.9905) |

Perfect recall on positive-only test sources accompanies 168/215 misclassified Negatome negatives and misclassification of the single IntAct negative. IntAct validation recall is 20/24, not perfect. These results do not demonstrate uniformly improved source generalization; the near-universal positive prediction pattern is essential context.

## Final Selection And Reversion

Restore **fixed symmetric pair features**: frozen ProstT5 + adapters, residue max pooling, equal-weight group mean pooling, then sum/absolute-difference/elementwise-product features, pair MLP, and classifier, using historical focal-only loss. It wins on the configured validation AUROC criterion and overall balanced test performance. The chain Transformer's negligible F1 edge does not justify substantially weaker negative handling.

The chain Transformer and group-relation logic are removed rather than retained as configuration variants. Training optimizer groups, best-state snapshots, checkpoint metadata, and standalone loading are returned to the fixed-feature architecture. Numerical safeguards and safe per-improvement checkpoint updates are retained. The original winning run is recognized explicitly for evaluation; other learned-module runs require their recorded code revisions. Historical weights and metrics are unchanged, and existing tokenized/backbone embedding caches remain reusable.

This is single-seed evidence, not proof that learned interaction architectures universally fail. Added capacity, projection, optimization, and source effects are not separated by aggregate metrics, and runtime/representation diagnostics are not saved. Repeated test inspection across ablations warrants a fresh final holdout for an unbiased final estimate.

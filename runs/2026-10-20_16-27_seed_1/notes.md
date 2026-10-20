# Run 2026-10-20_16-27_seed_1

Frozen ProstT5 + adapters with residue max pooling.

Restored winning architecture: residue max pooling, equal-weight group mean
pooling, and fixed symmetric sum/absolute-difference/elementwise-product
features followed by the pair MLP and classifier. No learned interaction encoder.
No data augmentation.
Cached backbone embeddings are reused for training/evaluation when available.
The frozen backbone stays in evaluation mode and is encoded without gradients.
Token budgets: train 49152, evaluation 65536.
Original cached-baseline token budgets are restored.
Interaction loss: historical class-weighted focal (gamma 2, exp(-weighted CE), batch mean).
Auxiliary experiment: one shared linear affinity head on the pair representation.
Total loss = interaction focal + 0.1 * masked Huber (delta 1) on train-standardized pKd.
pKd is already -log10(Kd in mol/L); normalization uses observed train labels only.
Affinity gradients reach the shared representation; classifier selection/calibration are unchanged.
Affinity diagnostics are not a prediction objective or checkpoint-selection criterion.
No LoRA or contrastive term; existing tokenized data is reused unchanged.
CUDA compilation is enabled when available; numerical safeguards are retained.
Non-finite inputs/logits/loss/gradients abort training.
Each validation-AUROC improvement is checkpointed immediately.

## Auxiliary Affinity Analysis

### Setup and Comparison Scope

Primary matched-seed reference: `2026-10-15_19-34_seed_1`, the reproducible interaction-only seed-1 baseline. Its weights/metrics previously matched the October 17 seed-1 reproduction exactly; that duplicate is not a separate seed observation. Other retained baseline seeds are `2026-10-19_18-18_seed_27` and `2026-10-19_22-51_seed_68`. The historical October 8 best run is not substituted for the reproducible seed-1 baseline. Earlier seed-26/67 artifacts are absent from the current runs directory, so this comparison uses all three currently retained baseline seed samples.

This run uses training/sampler seed 1 and global/split seed 1. Saved config.py settings match the seed-1 baseline exactly. The auxiliary change is one shared linear pKd head on the pair representation, optimized with interaction focal loss plus 0.1 times masked Huber loss (delta 1). Normalization was fitted on 9,214 training labels only: mean 7.819456, population std 2.097369 pKd. Regression diagnostics do not participate in selection; checkpoint selection remains validation interaction AUROC.

The recorded NVIDIA L40/PyTorch 2.8.0+cu128/CUDA 12.8 environment matches baseline seeds 27 and 68, including bfloat16 autocast, fused AdamW, and compile-wrapper status. The retained seed-1 baseline lacks an execution snapshot, so its environment cannot be fully compared from that artifact alone.

All four evaluations have 1,614 validation and 1,615 test pairs with matching source/class counts and cache paths. The auxiliary tokenization implementation checks baseline classifier labels, source/order, and token identity against an existing cache before replacement, but no cache fingerprints or tokenization transcript are saved with this run. Exact artifact identity cannot be reconstructed independently from the result JSONs.

### Classification Across Baseline Seeds

| Metric | Baseline seed 1 | Baseline seed 27 | Baseline seed 68 | Auxiliary seed 1 |
| --- | --- | --- | --- | --- |
| Validation AUROC | 0.9414 | 0.9195 | 0.9106 | 0.9508 |
| Test AUROC | 0.9325 | 0.9231 | 0.9044 | 0.9291 |
| Test AUPRC | 0.9854 | 0.9823 | 0.9789 | 0.9850 |
| Test balanced accuracy, raw | 0.8661 | 0.8425 | 0.8315 | 0.8386 |
| Test specificity, raw | 0.8287 | 0.8102 | 0.8194 | 0.7315 |
| Test recall, raw | 0.9035 | 0.8749 | 0.8435 | 0.9457 |
| Test MCC, raw | 0.6297 | 0.5660 | 0.5245 | 0.6548 |
| Test F1, raw | 0.9363 | 0.9189 | 0.9015 | 0.9518 |
| Test balanced accuracy, calibrated | 0.8459 | 0.8065 | 0.8135 | 0.8170 |
| Test specificity, calibrated | 0.7454 | 0.6574 | 0.6898 | 0.6806 |
| Test recall, calibrated | 0.9464 | 0.9557 | 0.9371 | 0.9535 |
| Test MCC, calibrated | 0.6666 | 0.6282 | 0.6030 | 0.6391 |
| Test F1, calibrated | 0.9532 | 0.9516 | 0.9442 | 0.9522 |

Matched seed 1: validation AUROC improves by 0.0094, from 0.9414 to 0.9508. Validation MCC, accuracy, recall, and F1 improve, while raw balanced accuracy falls from 0.8897 to 0.8781. Validation-fitted balanced accuracy is also slightly lower (0.8759 versus 0.8735). Thus validation is improved on ranking and some operating-point metrics, not uniformly improved.

The ranking gain does not transfer to test: AUROC declines by 0.0034 and AUPRC by 0.0004. Default-threshold test MCC/F1/accuracy improve, but specificity and balanced accuracy decline. With validation-fitted thresholds, test balanced accuracy, specificity, MCC, F1, and accuracy are all below the matched seed-1 baseline; only positive recall increases.

Across the three baseline seeds, test AUROC is 0.9200 +/- 0.0143, AUPRC is 0.9822 +/- 0.0032, calibrated balanced accuracy is 0.8220 +/- 0.0210, and calibrated MCC is 0.6326 +/- 0.0320 (mean +/- sample SD, not a confidence interval). Auxiliary seed 1 beats baseline seeds 27 and 68 on these ranking/calibrated classification metrics but not baseline seed 1. Its calibrated balanced accuracy is below the baseline mean. Comparing one auxiliary seed with the mean of multiple baseline seeds is descriptive, not evidence of a treatment improvement or reduced seed sensitivity.

### Error Balance and Threshold Selection

| Test operating point | TN | FP | FN | TP |
| --- | --- | --- | --- | --- |
| Baseline seed 1, default | 179 | 37 | 135 | 1264 |
| Auxiliary seed 1, default | 158 | 58 | 76 | 1323 |
| Baseline seed 1, validation-fitted | 161 | 55 | 75 | 1324 |
| Auxiliary seed 1, validation-fitted | 147 | 69 | 65 | 1334 |

At the default threshold, auxiliary supervision recovers 59 additional positives at the cost of 21 additional false positives versus seed 1. Test accuracy rises from 0.8935 to 0.9170, but balanced accuracy falls by 0.0275 and specificity falls by 0.0972. This is a shift toward positive recall, not uniform improvement.

The auxiliary validation-F1-selected threshold is 0.17; every retained baseline selects 0.05. Thresholds were fitted on validation and transferred unchanged to test. Their magnitudes are not directly comparable across differently scaled model scores, and F1-based threshold selection is not probability calibration.

After threshold selection, the auxiliary run recovers 10 positives for 14 additional false positives versus seed 1. Calibrated balanced accuracy declines by 0.0288 and MCC by 0.0274. Within the auxiliary run, moving from the default threshold to 0.17 adds 11 true positives and 11 false positives, leaving accuracy unchanged while lowering balanced accuracy from 0.8386 to 0.8170.

### Source-Specific Classification

| Source / default-threshold metric | Baseline seed 1 | Baseline seed 27 | Baseline seed 68 | Auxiliary seed 1 |
| --- | --- | --- | --- | --- |
| intact_positive: positive recall | 6/25 (0.2400) | 5/25 (0.2000) | 4/25 (0.1600) | 7/25 (0.2800) |
| negatome: negative recall | 178/215 (0.8279) | 174/215 (0.8093) | 176/215 (0.8186) | 157/215 (0.7302) |
| ppb_affinity: positive recall | 706/755 (0.9351) | 671/755 (0.8887) | 666/755 (0.8821) | 726/755 (0.9616) |
| skempi: positive recall | 297/302 (0.9834) | 297/302 (0.9834) | 297/302 (0.9834) | 297/302 (0.9834) |
| string: positive recall | 255/317 (0.8044) | 251/317 (0.7918) | 213/317 (0.6719) | 293/317 (0.9243) |

These reports use the default threshold; source-specific validation-fitted reports were not saved. Relative to seed 1, the extra recovered positives are 20 PPB pairs, 38 STRING pairs, and one IntAct pair; SKEMPI remains 297/302. The extra 21 false positives are Negatome negatives. Improved positive recall is not confined to affinity-labeled sources, but IntAct recall remains weak (7/25), and there is no improvement in negative rejection. All four runs correctly reject the single IntAct negative; one example cannot establish source-level specificity.

### Affinity Diagnostics

| Split | Source | n | MAE (pKd) | RMSE (pKd) | Pearson |
| --- | --- | --- | --- | --- | --- |
| validation | all | 1044 | 1.8730 | 2.3004 | 0.0398 |
| validation | ppb_affinity | 754 | 1.8982 | 2.2908 | 0.0171 |
| validation | skempi | 290 | 1.8076 | 2.3252 | 0.1184 |
| test | all | 1040 | 1.7635 | 2.1220 | 0.2085 |
| test | ppb_affinity | 755 | 1.8201 | 2.1920 | -0.0599 |
| test | skempi | 285 | 1.6137 | 1.9245 | 0.4975 |

Affinity predictions are uncalibrated and de-normalized using training statistics; errors are in pKd. Not all classification positives carry affinity labels: SKEMPI contributes 290/304 validation and 285/302 test examples.

The shared head remains weak overall: validation Pearson is 0.0398 and test Pearson is 0.2085. PPB test ranking is effectively absent (Pearson -0.0599); SKEMPI has moderate test correlation (0.4975), but much weaker validation correlation (0.1184). Overall test MAE/RMSE are 1.7635/2.1220 pKd. No constant-predictor benchmark or R-squared is saved, so these diagnostics do not establish improvement over a mean-prediction baseline.

An inaccurate auxiliary task could still help classification in principle, but this run does not demonstrate an overall matched-seed test benefit. Better validation classification is insufficient when the test ranking and balanced detection do not improve.

### Conclusion

Do not adopt this auxiliary-head variant as a demonstrated improvement over the reproducible baseline. The observed benefit is stronger validation ranking and higher default-threshold positive recall; the costs are poorer negative rejection and lower matched-seed test ranking/calibrated balanced detection.

If pursuing confirmation, run the exact auxiliary setup with training/sampler seeds 27 and 68, keeping the split/cache fixed, and compare paired changes against those same baseline seeds. Do not claim success by comparing auxiliary seed 1 only to the weaker baseline seeds or by choosing the best test run. With one auxiliary seed, neither consistent benefit nor absence of benefit across seeds is established. Repeated test inspection means this remains a development benchmark; a fresh untouched holdout is needed for final generalization claims.

Analysis changes only this run's notes. README, model/training code, weights, configuration, and metrics are unchanged.


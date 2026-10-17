# Run 2026-10-15_19-34_seed_1

Frozen ProstT5 + adapters with residue max pooling.

Restored winning architecture: residue max pooling, equal-weight group mean
pooling, and fixed symmetric sum/absolute-difference/elementwise-product
features followed by the pair MLP and classifier. No learned interaction encoder.
Sequence masking: 0%; residue dropout: 0%.
Both augmentation implementations are retained but disabled for the baseline.
Cached backbone embeddings are reused for training/evaluation when available.
The frozen backbone stays in evaluation mode and is encoded without gradients.
Token budgets: train 49152, evaluation 65536.
Original cached-baseline token budgets are restored.
Historical class-weighted focal loss only (gamma 2, exp(-weighted CE), batch mean).
No LoRA or contrastive term; existing tokenized data is reused unchanged.
CUDA compilation is enabled when available; numerical safeguards are retained.
Non-finite inputs/logits/loss/gradients abort training.
Each validation-AUROC improvement is checkpointed immediately.

## Baseline Consistency Check

Repeat of the winning architecture from `2026-10-08_01-46_seed_1`, using training/sampler seed 1. Current revision: `1240fb21080e5274a3b3647840e57821e89aadcc`; original revision: `f9e1b1645be209a810c130f410add891439207ff`.

The saved setup matches frozen ProstT5 + adapters, residue max/group mean pooling, fixed sum/absolute-difference/product features, and historical class-weighted focal loss with gamma 2. Both augmentations are disabled, cached embeddings are used, and original token budgets (49,152 train / 65,536 evaluation) are restored. All saved optimization, sampling, and split settings match; the only settings-key difference is removal of the focal gamma constant, whose value remains 2 in run metadata.

Both recorded revisions enable torch.compile on CUDA, but actual compilation success/backend, GPU/CUDA/library versions, and deterministic-kernel settings are not captured in the artifacts. The current implementation retains finite-value checks, safe CPU snapshot copies, per-improvement saves, and inactive augmentation paths. These change program/compiled graph structure even though the intended cached-input model computation matches the original. They do not by themselves establish the cause of numerical differences.

Both evaluations contain 1,614 validation and 1,615 test pairs, matching class/source counts and cache paths. No data/cache fingerprints are saved, so identical membership and embedding contents cannot be independently established from these artifacts alone.

## Validation Comparison

| Metric | Original: raw | Repeat: raw | Original: calibrated | Repeat: calibrated |
| --- | --- | --- | --- | --- |
| AUROC | 0.9551 | 0.9414 | 0.9551 | 0.9414 |
| AUPRC | 0.9899 | 0.9871 | 0.9899 | 0.9871 |
| Balanced accuracy | 0.9002 | 0.8897 | 0.9046 | 0.8759 |
| Specificity / negative recall | 0.9398 | 0.8889 | 0.8935 | 0.8148 |
| Positive recall | 0.8605 | 0.8906 | 0.9156 | 0.9371 |
| MCC | 0.6322 | 0.6473 | 0.6985 | 0.6921 |
| Accuracy | 0.8711 | 0.8903 | 0.9126 | 0.9207 |
| F1 | 0.9204 | 0.9336 | 0.9478 | 0.9534 |

Validation AUROC, the selection metric, declines by 0.0137. The repeat is therefore not an exact reproduction of the original validation peak. Both runs select threshold 0.05 using validation F1, the lower boundary of the search grid. Calibrated validation scores are fitting-set results, not independent evidence of threshold quality.

## Test Comparison

| Metric | Original: raw | Repeat: raw | Original: calibrated | Repeat: calibrated |
| --- | --- | --- | --- | --- |
| AUROC | 0.9356 | 0.9325 | 0.9356 | 0.9325 |
| AUPRC | 0.9856 | 0.9854 | 0.9856 | 0.9854 |
| Balanced accuracy | 0.8594 | 0.8661 | 0.8692 | 0.8459 |
| Specificity / negative recall | 0.8704 | 0.8287 | 0.8148 | 0.7454 |
| Positive recall | 0.8485 | 0.9035 | 0.9235 | 0.9464 |
| MCC | 0.5668 | 0.6297 | 0.6611 | 0.6666 |
| Accuracy | 0.8514 | 0.8935 | 0.9090 | 0.9195 |
| F1 | 0.9082 | 0.9363 | 0.9462 | 0.9532 |

Test AUROC declines by only 0.0030 and AUPRC by 0.0002, while raw balanced accuracy improves by 0.0067. Ranking is broadly comparable in these measurements, not demonstrated statistically equivalent.

Calibrated test counts change from TN=176, FP=40, FN=107, TP=1292 to TN=161, FP=55, FN=75, TP=1324. The repeat recovers 32 additional positives for 15 additional false positives. MCC rises by 0.0055 and F1 by 0.0070, but balanced accuracy falls by 0.0233 and specificity by 0.0694. The error balance shifted toward positive recall; it is not unchanged or uniformly better.

At the default threshold, source-level test recall changes from 4/25 to 6/25 for IntAct positives, 643/755 to 706/755 for PPB, and 243/317 to 255/317 for STRING. SKEMPI stays 297/302. Negatome negative recall declines from 187/215 to 178/215; the single IntAct negative remains correctly rejected. IntAct positive generalization is still weak. These are default-threshold source reports, not calibrated source metrics.

## Conclusion

The restored baseline has broadly comparable test ranking and overall classification quality, with no evidence in the saved setup of an unintended active augmentation, learned interaction module, or changed pooling/loss. It does not reproduce the original seed-1 result exactly, and the lower validation AUROC/specificity should not be dismissed as established random noise without repeat measurements.

GPU/compiler/version differences or changed compiled graph structure are possible explanations, not proven causes. Exact reproducibility would require matching and recording that environment, cache fingerprints, and compilation behavior. The two runs are not independent-seed replicates and should not be pooled to estimate seed variability.

For a multi-seed summary of the current frozen implementation, use this repeat as its seed-1 result and hold code/data/splits fixed across the remaining seeds rather than substituting the historically best seed-1 run. Do not select the final model by the best test score. Repeated inspection means this test split is now a development benchmark; an untouched final holdout would strengthen the final evaluation.

No model, training, README, or historical-metric changes are made by this check.

## Follow-Up Implementation Audit (2026-10-16)

Compared the implementation against winning revision `f9e1b1645be209a810c130f410add891439207ff`, beyond the saved settings:

- The sampler, collator, sample-weight calculation, seed initialization, adapter, pooling modules, and classification head have identical Python ASTs. Initialization produces identical weights and post-initialization torch RNG state on CPU.
- Five synthetic cached-input CPU training steps, including dropout, historical focal loss, gradient clipping, and AdamW, produce bitwise-identical parameters between the historical model and both the pre-cleanup and cleaned implementations. This is a small controlled check, not a reproduction of the full GPU run.
- Explicit float32 casting in the current focal loss matches the historical autocast-promoted calculation and gradients under CPU bfloat16 autocast. The weighted-CE modulation and batch-mean reduction have not changed.
- Calibration plus initial and repeated checkpoint saves leave Python, NumPy, and CPU torch RNG states unchanged. No RNG shift from these additions was observed.
- A concrete compiler-graph difference was found: the pre-adapter Python `if not torch.isfinite(...).all()` split token encoding into two captured graphs, with one tensor-dependent control-flow break. The historical token encoder captures as one graph with no breaks. After moving cached-input diagnostics into the eager training/evaluation batch paths, the cleaned token encoder also captures as one graph with no breaks. This does not claim the entire model is graph-break-free; group indexing already exists in the original.

Removed unused sequence-masking and residue-dropout implementations, tokenizer setup, metadata fields, and augmentation-specific tests. Restored the original pooling-mask dtype/path. Kept residue max pooling, group mean pooling, symmetric pair features, class-weighted focal loss, CUDA compilation, frozen-backbone evaluation behavior, numerical checks, and immediate best-checkpoint persistence. Historical runs, weights, and metrics are unchanged. README is unchanged.

Future run configs now record Python/library versions, GPU/CUDA/cuDNN details, compile request/wrapper status, mixed precision, fused optimizer use, and deterministic/TF32 settings. Compile-wrapper presence does not prove every operation was compiled successfully.

The graph difference is verified, but its contribution to this run's metric difference remains unproven. No CUDA device is available locally, and the original artifacts lack runtime-environment records and cache fingerprints. Exact GPU/compiler/data equivalence cannot be reconstructed retrospectively. Test AUROC remains close (0.9356 original versus 0.9325 repeat), while validation AUROC and the specificity/recall trade-off differ as documented above.

Verification after cleanup: all 19 workflow tests pass. A three-epoch synthetic cached-input execution of `train.py --validate` also passes training, checkpoint serialization, execution-metadata persistence, and validation/test reporting without loading the backbone or tokenizer. CUDA compilation and full-data training were not tested locally.

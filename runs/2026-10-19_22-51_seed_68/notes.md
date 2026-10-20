# Run 2026-10-19_22-51_seed_68

Frozen ProstT5 + adapters with residue max pooling.

Restored winning architecture: residue max pooling, equal-weight group mean
pooling, and fixed symmetric sum/absolute-difference/elementwise-product
features followed by the pair MLP and classifier. No learned interaction encoder.
No data augmentation.
Cached backbone embeddings are reused for training/evaluation when available.
The frozen backbone stays in evaluation mode and is encoded without gradients.
Token budgets: train 49152, evaluation 65536.
Original cached-baseline token budgets are restored.
Historical class-weighted focal loss only (gamma 2, exp(-weighted CE), batch mean).
No LoRA or contrastive term; existing tokenized data is reused unchanged.
CUDA compilation is enabled when available; numerical safeguards are retained.
Non-finite inputs/logits/loss/gradients abort training.
Each validation-AUROC improvement is checkpointed immediately.

## Reproduction Analysis (2026-10-20)

### Setup

Training and batch-sampler seeds are both 68; GLOBAL_SEED and SPLIT_SEED remain 1. Against the saved seed-1 baseline settings, only TRAINING_SEED and BATCH_SAMPLER_SEED differ. Against cleaned baseline revision `9b83783938f5ed81dfcea4b6887d0c84d1470359`, model/training/tokenization/cache-encoding code is unchanged; the relevant code diff only adjusts the two training seeds. This measures combined initialization/dropout and sampling variability, not initialization alone.

Both new runs record the same execution environment: NVIDIA L40, PyTorch 2.8.0+cu128, CUDA 12.8, Python 3.12.3, bfloat16 autocast, fused AdamW, and a torch.compile wrapper. Deterministic algorithms are disabled. Seed 1's retained October 15 config has no execution snapshot, so environment identity with that artifact cannot be verified directly.

All three evaluations have 1,614 validation and 1,615 test pairs with matching per-source/class counts and cache paths. Split/cache fingerprints are not saved, so exact membership and embedding identity remain assumptions. GLOBAL_SEED/SPLIT_SEED are now explicitly fixed, unlike the earlier seed-26/67 settings.

### Comparison Against the Reproducible Baseline

Seed-1 reference: `2026-10-15_19-34_seed_1`, whose weights and metrics previously matched the October 17 seed-1 reproduction exactly. The original October 8 validation winner is not substituted as the seed-1 observation.

| Metric | Seed 1 baseline | Seed 27 | Seed 68 |
| --- | --- | --- | --- |
| Validation AUROC | 0.9414 | 0.9195 | 0.9106 |
| Test AUROC | 0.9325 | 0.9231 | 0.9044 |
| Test AUPRC | 0.9854 | 0.9823 | 0.9789 |
| Test balanced accuracy, raw | 0.8661 | 0.8425 | 0.8315 |
| Test balanced accuracy, calibrated | 0.8459 | 0.8065 | 0.8135 |
| Test specificity, calibrated | 0.7454 | 0.6574 | 0.6898 |
| Test MCC, calibrated | 0.6666 | 0.6282 | 0.6030 |
| Test F1, calibrated | 0.9532 | 0.9516 | 0.9442 |

Validation AUROC falls from 0.9414 to 0.9106; test AUROC falls by 0.0282. This is the weakest-ranking run of the three current reproductions. After validation-fitted thresholding, test counts change from baseline TN=161, FP=55, FN=75, TP=1324 to TN=149, FP=67, FN=88, TP=1311: 12 more false positives and 13 more missed positives. Calibrated balanced accuracy falls by 0.0324 and MCC by 0.0635. Its slightly higher calibrated balanced accuracy than seed 27 reflects a different specificity/recall trade-off; ranking, MCC, accuracy, and F1 are lower.

### Threshold Behavior

All three runs select threshold 0.05 by validation F1, the lower boundary of the search grid. For this run, test balanced accuracy declines from 0.8315 at the default threshold to 0.8135 after threshold selection, while F1 increases from 0.9015 to 0.9442. Specificity declines from 0.8194 to 0.6898. This is the existing F1-versus-negative-detection trade-off, not proof of improved probability calibration. Validation-fitted metrics are fitting-set results; only the unchanged threshold is transferred to test.

### Source-Specific Test Behavior

| Test source | Default-threshold metric | Seed 1 | This run |
| --- | --- | --- | --- |
| intact_positive | Positive recall | 0.2400 | 0.1600 |
| negatome | Negative recall | 0.8279 | 0.8186 |
| ppb_affinity | Positive recall | 0.9351 | 0.8821 |
| skempi | Positive recall | 0.9834 | 0.9834 |
| string | Positive recall | 0.8044 | 0.6719 |

These source reports use the default threshold, not the validation-fitted threshold. IntAct positive recall remains weak (4/25); all three runs recover 297/302 SKEMPI positives. The single IntAct negative is correctly rejected, but one example does not support a source-level conclusion.

### Updated Multi-Seed Readout

For the three currently retained reproductions (seeds 1, 27, 68), test AUROC is 0.9200 +/- 0.0143 and AUPRC is 0.9822 +/- 0.0032 (mean +/- sample standard deviation across runs, not standard error or a confidence interval). This is wider variability than the earlier seed-1/26/67 comparison suggested. The seed-26/67 artifacts are not present in the current runs directory and are not pooled into this summary. The historical October 8 winner and duplicate seed-1 reproduction are also excluded.

The architecture reproduces its seed-1 result, but classification quality is not insensitive to training seed. Both new runs trail seed 1 on validation/test ranking, and all three lose test balanced accuracy after F1-based threshold selection. Keep these weaker seeds in the baseline distribution rather than replacing them with better runs. Compare any auxiliary-affinity or other ablation against matching training/sampler seeds and the same fixed cache; a single favorable run would not establish improvement. Repeated test inspection also means this split is a development benchmark, not an untouched final holdout.

This analysis changes notes only; weights, configuration, metrics, model/training code, and README are unchanged.


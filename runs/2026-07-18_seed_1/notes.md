# Run 2026-07-18_seed_1

Migrated from `prostt5_group_pair_adapter_best_2026-07-18_seed_1.pt`. Legacy dates without a time remain date-only.

### Version 2026-07-18
#### Changes
- Added negative-aware interaction training to address the previous failure mode where the model ranked interactions well but predicted almost everything as positive.
- Switched the interaction loss from weighted cross-entropy to configurable focal loss (`INTERACTION_LOSS = "focal"`, `FOCAL_GAMMA = 2.0`).
- Added weighted sampling controls so each epoch sees a less extreme interaction class balance (`INTERACTION_POS_NEG_RATIO = 5.0`) instead of reflecting the raw positive-heavy dataset distribution.
- Added source-balanced sampling so high-volume sources do not dominate every training epoch.
- Switched affinity regression from MSE to Huber loss for more robustness to noisy/outlier pKd labels.
- Added source-normalized affinity training/reporting to test whether PPB-Affinity and SKEMPI should be normalized separately before regression.
#### Results
- Negative-aware training improved interaction classification and Negatome handling. Source-normalized affinity training remained weak; retain the interaction changes and revisit affinity modeling.

## Original Result Notes

#### Results
- Interaction classification improved in the intended direction: calibrated test balanced accuracy increased and the model now predicts negatives at a realistic rate instead of collapsing to nearly all-positive predictions.
- Negatome negative handling improved materially on the test split: `152/216` negatives were correctly predicted (`specificity=0.7037` for the uncalibrated source-specific row).
- Affinity remains weak overall. Source-specific test ranking is better for SKEMPI (`Pearson=0.4555`, `Spearman=0.5930`) than PPB-Affinity (`Pearson=0.2416`, `Spearman=0.2549`).
- Source-normalized affinity regression did not solve the regression problem: source-normalized test `Pearson=0.2450`, `Spearman=0.2712`, and `R2=0.0011`.
- Conclusion: keep the negative-aware interaction training changes, but treat source-normalized affinity training as experimental. The next affinity run should likely keep Huber loss but return to global pKd normalization, then consider separate PPB/SKEMPI heads if source bias remains large.

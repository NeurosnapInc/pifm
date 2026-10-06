# Run 2026-07-19_seed_1

Migrated from `prostt5_group_pair_adapter_best_2026-07-19_seed_1.pt`. Legacy dates without a time remain date-only.

### Version 2026-07-19
#### Changes
- Kept the negative-aware interaction setup from the previous run: focal interaction loss, source-balanced sampling, and capped positive:negative sampling.
- Kept Huber loss for affinity regression (`REGRESSION_LOSS = "huber"`).
- Reverted affinity normalization from source-normalized pKd back to global pKd normalization (`AFFINITY_NORMALIZATION = "global"`) after the 2026-07-18 run showed weak source-normalized affinity performance.

#### Results
- Interaction classification improved with calibrated thresholding. Global affinity normalization did not recover test performance; separate source-specific modeling or data/split investigation was needed.

## Original Result Notes

#### Results
- Interaction classification improved again under calibrated thresholding.
- The calibrated test prediction ratio is much healthier than the early all-positive failure mode, and Negatome handling improved in the source-specific test row.
- Affinity regression remains unresolved. Validation affinity has moderate signal (`Pearson=0.3121`) but test affinity ranking is essentially absent (`Pearson=0.0227`, `Spearman=0.0249`; calibrated `R2=-0.1243`).
- Global pKd normalization did not recover affinity performance. The next affinity-specific direction should likely be separate PPB/SKEMPI heads, stronger source-aware modeling, or revisiting the affinity data/split rather than more normalization changes.

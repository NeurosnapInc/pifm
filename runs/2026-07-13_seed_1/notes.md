# Run 2026-07-13_seed_1

Migrated from `prostt5_group_pair_adapter_best_2026-07-13_seed_1.pt`. Legacy dates without a time remain date-only.

### Version 2026-07-13
#### Changes
- Switched tokenization from a random row split to a cluster-disjoint split to reduce sequence/homology leakage across train, validation, and test.
- Replaced naive random cluster assignment with label-aware greedy split assignment over connected cluster components. The splitter now balances total samples, interaction positives, interaction negatives, affinity-labeled samples, and source counts where possible.
- Standardized affinity labels to `pKd` in the aggregated DuckDB/tokenized cache so regression targets are comparable across PPB-Affinity, SKEMPI, and user-provided affinity sources.
- Updated training checkpoint selection to avoid early stopping on misleading aggregate `F1`/`MAE`. Classification selection now uses `AUROC` by default, regression uses normalized `MAE`, and tasks with too few validation labels are ignored for checkpoint selection.
- Increased token-capped batch sizes for A100 training throughput.

#### Results
- Validation affinity had moderate signal, while the interaction head still predicted almost everything as positive.

## Original Result Notes

#### Results
- The affinity validation set is now large enough to interpret (`n=1042`) and the model learns a moderate affinity signal. The interaction head has good ranking signal (`AUROC=0.8828`) but poor thresholded negative detection, still predicting almost everything as positive.

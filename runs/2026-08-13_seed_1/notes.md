# Run 2026-08-13_seed_1

Migrated from `prostt5_group_pair_adapter_best_2026-08-13_seed_1.pt`. Legacy dates without a time remain date-only.

### Version 2026-08-13
#### Changes
- Removed affinity regression from the downstream ML path to create a simpler interaction-only baseline.
- Kept affinity values in aggregation/source loading where available, but tokenization now emits only the `interaction` task.
- Simplified training to a single binary interaction head with the existing frozen ProstT5 + adapter architecture.
- Simplified validation/calibration to classification-only reports: aggregate interaction metrics, source-specific interaction metrics, confusion counts, specificity, MCC, AUROC, AUPRC, and threshold calibration.
- Removed downstream regression config knobs, regression heads, affinity task construction, source-normalized regression reporting, and post-hoc regression calibration.
- Renamed cache outputs to avoid accidentally reusing stale multitask caches:
  - `data/tokenized/interaction_group_pair_prostt5_tokens.pt`
  - `data/tokenized/interaction_prostt5_backbone_embeddings.pt`

#### Rationale
- The 2026-08-12 source-specific affinity experiment confirmed that interaction classification is useful, but affinity regression did not generalize on the test split.
- Removing affinity downstream makes this the clean baseline before testing cheaper architecture ablations such as residue-to-chain and group pooling strategies.

#### Next Run
```bash
python tokenize_data.py
python cache_embeddings.py
python train.py
```

#### Results
- Validation classification remained reasonable after removing downstream affinity.
- Test classification was weaker than the 2026-08-12 multitask/source-specific-affinity run.
- Calibrating the threshold on validation did not recover test negative handling.
- Main readout: removing affinity simplifies the code and creates the intended clean baseline, but the interaction-only run has worse test negative recall than the previous setup. The next pooling ablations should be compared against this baseline, not assumed to improve it.

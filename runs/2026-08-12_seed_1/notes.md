# Run 2026-08-12_seed_1

Migrated from `prostt5_group_pair_adapter_best_2026-08-12_seed_1.pt`. Legacy dates without a time remain date-only.

### Version 2026-08-12
#### Changes
- Stopped treating affinity as one shared regression task for the next experiment.
- Added configurable affinity task construction with `AFFINITY_TASK_MODE = "source_specific"` and `AFFINITY_SOURCE_TASKS = ("ppb_affinity", "skempi")`.
- Tokenization now emits separate source-specific affinity regression heads:
  - `affinity_ppb_affinity`
  - `affinity_skempi`
- Kept the interaction head shared so the classifier still learns from all interaction-labeled sources.
- Kept global pKd normalization and Huber regression loss for this run (`AFFINITY_NORMALIZATION = "global"`, `REGRESSION_LOSS = "huber"`).
- Training checkpoints now record the affinity task mode and configured affinity sources.
- Source-normalized regression utilities now identify affinity tasks via metadata, not only the old literal `affinity` task name.

#### Rationale
- The 2026-07-19 run showed that interaction classification is now useful after negative-aware training, but affinity regression remained weak on the test split.
- PPB-Affinity and SKEMPI appear to behave differently enough that one shared affinity head is likely underfitting source-specific label structure.
- This change isolates PPB and SKEMPI affinity prediction at the head level while preserving the same frozen ProstT5 encoder, adapter, group pooling, and pair representation.

#### Next Run
- Re-tokenization is required because the tokenized cache task layout changes from one `affinity` task to separate source-specific affinity tasks.
- Frozen ProstT5 backbone embeddings can optionally be cached after tokenization to avoid re-running the transformer during training and validation.

```bash
python tokenize_data.py
python cache_embeddings.py
python train.py
```

#### Results
- Interaction performance remained strong. Separate affinity heads helped SKEMPI on validation but failed to generalize on test; PPB-Affinity remained weak.

## Original Result Notes

#### Results
- Interaction validation improved relative to the previous run.
- Calibrated interaction validation was similar but shifted toward higher recall.
- Source-specific affinity heads helped SKEMPI more than PPB-Affinity. SKEMPI validation reached `Pearson=0.5427`, `Spearman=0.5777`, and calibrated `R2=0.2945`; PPB-Affinity remained weak with `Pearson=0.1317`, `Spearman=0.0918`, and calibrated `R2=0.0174`.
- Test interaction performance remains strong.
- Affinity did not generalize. PPB-Affinity stayed weak on test (`Pearson=0.0954`, `Spearman=0.1038`, calibrated `R2=0.0047`) and SKEMPI collapsed despite good validation performance (`Pearson=-0.0434`, `Spearman=-0.0755`, calibrated `R2=-0.4946`).
- Main readout: keep the interaction setup, but do not treat the source-specific affinity-head experiment as successful. The affinity problem likely needs data/split investigation or a source-specific architecture/training regime beyond just separate heads.

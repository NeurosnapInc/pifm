# Protein Interaction Foundation Model (PIFM)
> A fast, scalable protein language model for predicting protein-protein interactions (PPI) between arbitrary protein complexes.

## Overview
The goal of this project is to develop a lightweight and highly efficient model capable of predicting whether two groups of proteins interact.

Unlike structure-based approaches (e.g., AlphaFold-Multimer, docking, molecular dynamics), this model should operate directly from amino acid sequences while maintaining inference speeds suitable for high-throughput screening.

The primary design philosophy is:

- Fast inference
- High scalability
- Supports arbitrary protein complexes
- Simple architecture
- Easily extensible
- Compatible with cached embeddings
- Competitive accuracy through parameter-efficient fine-tuning

Rather than training an entirely new protein language model, this project builds upon an existing pretrained protein LM (initially ProstT5) and fine-tunes lightweight adapters together with a downstream interaction prediction network.

> **Project Status**
>
> This project is currently in the planning and research phase. The project name is a **working title (WIP)** and will likely change as development progresses.
>
> The implementation will follow a modular design philosophy loosely inspired by the Prot2Prop project:
>
> https://github.com/NeurosnapInc/Prot2Prop
>
> While the underlying machine learning task is fundamentally different, we intend to reuse many of the same software engineering principles including:
>
> - Modular model components
> - Parameter-efficient fine-tuning (LoRA/adapters)
> - Clean PyTorch implementation
> - Easily swappable protein language model backbones
> - Reproducible training and evaluation pipelines
> - Extensible configuration-driven architecture
> - Simple inference API
>
> This should make it straightforward to rapidly prototype new architectures while maintaining a clean and maintainable codebase.

## Objectives

Primary objectives:

- Predict whether two protein groups interact
- Support an arbitrary number of proteins on each interaction side
- Maintain inference speeds orders of magnitude faster than structural prediction methods
- Enable cached embeddings for repeated screening

Secondary objectives:

- Learn biologically meaningful protein interaction representations
- Generalize to unseen proteins
- Support future extensions such as interface prediction or residue-level attribution

## Motivation
Protein property prediction can be solved effectively using pooled protein embeddings from pretrained protein language models.

Protein interaction prediction is fundamentally different because:

- Inputs consist of multiple proteins
- Each side may contain one or more chains
- Protein order should not affect predictions
- Interactions occur between groups rather than individual sequences

This project investigates architectures capable of learning interactions between arbitrary protein sets while remaining computationally efficient.

## Proposed Architecture
### High-Level Pipeline
```
Protein Group A
        │
        ▼
  ProstT5 Encoder
        │
        ▼
 Chain Embeddings
        │
        ▼
 Group Encoder
        │
        ▼
 Group A Embedding

────────────────

Protein Group B
        │
        ▼
  ProstT5 Encoder
        │
        ▼
 Chain Embeddings
        │
        ▼
 Group Encoder
        │
        ▼
 Group B Embedding

────────────────

 Pairwise Interaction Module
       │
       ▼

 Prediction Head
┌────────────────┐
│ Interaction    │
└────────────────┘
```

### Training Scope
Downstream training is currently interaction classification only. Affinity values may still be retained during aggregation for future reference, but tokenization, training, validation, and calibration ignore affinity targets.

## Data Sources & Downloads
Aggregation is source-driven. Each dataset has a loader in the `sources/` package that yields `InteractionEntry` objects; loaders are registered (in
priority order) by `sources.build_source_specs()` and consumed by `aggregate_data.py`.

Raw downloads live under `./data/raw/` (git-ignored). Loaders are defensive: if their files are absent they print a download hint and yield nothing, so `python aggregate_data.py` always runs.

Prepare the raw data directory:

```bash
mkdir -p data/raw
```

Once data is present, run:
```bash
python aggregate_data.py            # writes data/aggregated/aggregated.duckdb
duckdb data/aggregated/aggregated.duckdb -ui   # inspect
```

## Tokenization & Splitting
The default tokenizer uses a cluster-disjoint split (`SPLIT_STRATEGY = "cluster"`) rather than a random row split. This is important for PPI evaluation: no sequence cluster is allowed to appear in more than one of train/validation/test, reducing homology leakage between splits.

Install MMseqs2 before tokenization:

```bash
conda install -c bioconda mmseqs2
```

Then run:

```bash
python tokenize_data.py
```

If `data/tokenized/split_sequence_clusters.tsv` is absent, `tokenize_data.py` writes `data/tokenized/split_sequences.fasta`, runs MMseqs2 clustering, and saves the resulting cluster assignments. The default threshold is `CLUSTER_MIN_SEQ_ID = 0.5` with `CLUSTER_COVERAGE = 0.8` in `config.py`.

Rows whose participating protein clusters would land in different splits are dropped and reported as `dropped_cross_split`. This is intentional: keeping those rows would reintroduce cluster leakage.

## Training & Run Evaluation
Each training run has its own directory:

```text
runs/2026-10-06_14-30_seed_1/
  checkpoint.pt
  config.json
  metrics.json
  notes.md
```

`checkpoint.pt` contains model weights only. `config.json` records model/training configuration and the validation-fitted calibration threshold. `metrics.json` stores checkpoint-selection metrics under `training` and evaluation reports under `splits.validation` and `splits.test`. `notes.md` holds experiment context and conclusions.

Train and automatically evaluate the saved best model on both validation and test:

```bash
python train.py --validate
```

Without `--validate`, training still uses validation for checkpoint selection and calibration but skips final evaluation of both splits. Test results are never used for checkpoint selection or threshold fitting.

New run names include date, hours/minutes, and seed. Times use the server's local timezone; `config.json` records the timestamp and UTC offset. A numeric suffix distinguishes runs created within the same minute.

Evaluate an existing interaction-only run:

```bash
python validate.py --run runs/2026-10-06_14-30_seed_1
```

This loads the model once, prints both split tables, and atomically updates only `metrics.json`. Weights, configuration, notes, and historical archives remain intact. `--split validation`, `--split test`, or `--split train` evaluates just that split and preserves the other reports. `--cache` and `--batch-size` remain available. `--checkpoint runs/<run>/checkpoint.pt` is an alias for `--run`.

Compare runs without importing torch, loading weights, or running inference:

```bash
python summarize_runs.py
python summarize_runs.py runs/2026-08-13_seed_1
python summarize_runs.py --json
python summarize_runs.py --historical
```

The summary reads JSON sidecars from `runs/` by default, or from supplied run directories/root directories. JSON output includes all source-specific and historical metrics; `--historical` prints archived tables and notes. Missing results are shown as `not evaluated`. Historical multitask runs retain their original reports; current evaluation supports interaction-only models.

### Migrating Older Checkpoints
Migrate checkpoints on another machine using:

```bash
python migrate_checkpoints.py
# Optionally remove the originals only after verifying the migrated artifacts:
python migrate_checkpoints.py --move
```

The migration separates embedded configuration, metrics, and archived result notes into sidecars, verifies every weight tensor and metadata field, and refuses to overwrite a conflicting run. It can be repeated safely against matching artifacts. Legacy checkpoints without a reliable save time use date-only run names such as `runs/2026-08-13_seed_1/`.

Run metadata and notes are version-controlled; `checkpoint.pt` is git-ignored. Transfer the entire run directory when copying a model between machines.

## GPU Memory Notes
On standard 48 GB VRAM GPU instances, PyTorch may fail with a CUDA out-of-memory error even when enough total memory should be available, because a large amount of memory is reserved but unallocated by PyTorch. Set the allocator configuration before launching training or inference:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This can reduce allocator fragmentation and avoid failures such as attempts to allocate another large CUDA segment on a mostly reserved GPU. See the PyTorch CUDA memory management documentation for details:
https://pytorch.org/docs/stable/notes/cuda.html#environment-variables

### Adding a new source
1. Add `sources/<name>.py` with `def iter_<name>() -> Iterator[InteractionEntry]` (yield **sequences only**; set `interaction_label`; `affinity_nm` may be retained in aggregation but is ignored downstream).
2. Register a `SourceSpec` in `sources.build_source_specs()` — list position sets priority (earlier wins on duplicate canonical pairs).
3. Document its download here, targeting `./data/raw/<name>/`.

### Sequence resolution (required for Negatome)
These sources distribute interactions as **UniProt accession pairs**, not sequences. Provide one or more UniProt FASTA files in `data/raw/uniprot/` and the loaders resolve accessions locally (no network calls); unresolved accessions are skipped. Swiss-Prot is a good default:
```bash
mkdir -p data/raw/uniprot
wget -P data/raw/uniprot https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
```

### Registered sources
| Source | Labels | Pos/Neg | Download |
|---|---|---|---|
| PPB-Affinity filtered | affinity-derived binary | positive | free (Hugging Face) |
| SKEMPI v2.0 | affinity-derived binary | positive | free (~32 MB) |
| IntAct | binary | positive + negative | free (FTP, large ZIP) |
| Negatome 2.0 | binary | **negative** | free |
| STRING (filtered) | binary | positive | free (per species) |
| literature-derived | optional binary + retained affinity | user-defined | user-provided CSV |

### PPB-Affinity filtered (protein–protein affinities, positives)
The filtered PPB-Affinity CSV provides pre-extracted `Ligand Sequences`, `Receptor Sequences`, and `KD(M)` columns. `KD(M)` is Kd in molar units; the loader converts it to nM and the aggregator stores the standardized pKd target. Download it directly into the path expected by the loader:

```bash
wget -O data/raw/ppb_affinity_filtered.csv https://huggingface.co/datasets/proteinea/ppb_affinity/resolve/main/filtered.csv
```

### SKEMPI v2.0 (protein–protein affinities, wild-type + mutants, positives)
SKEMPI's CSV contains no sequences — only PDB ids + chains — so the loader reconstructs chain sequences from the bundled cleaned PDB structures (ATOM records) and applies each cleaned point mutation to produce the mutant complex. Both the CSV and the PDB bundle are required:

```bash
wget -O data/raw/skempi_v2.csv https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv
curl -L https://life.bsc.es/pid/skempi2/database/download/SKEMPI2_PDBs.tgz | tar -xz -C data/raw   # -> data/raw/PDBs/
```

Yields ~348 wild-type complexes and ~7,000 mutant complexes. The `Affinity_wt_parsed` and `Affinity_mut_parsed` columns are Kd in molar units;
the loader converts them to nM and the aggregator stores standardized pKd. Both wild-type and mutants are labeled positive (SKEMPI only records complexes that form). Rows whose mutation numbering does not match the structure are skipped.

### IntAct (physical PPIs, positives + negatives)
```bash
wget -O data/raw/intact_all_2026_07_03.zip https://ftp.ebi.ac.uk/pub/databases/intact/current/all.zip
```

The IntAct loader reads the local bulk ZIP configured by `config.INTACT_ARCHIVE_PATH`. It parses the positive and negative MITAB exports and resolves sequences from the bundled IntAct FASTA, so no separate UniProt FASTA is required for IntAct.

### Negatome 2.0 (non-interacting pairs, the only negatives)
```bash
mkdir -p data/raw/negatome
wget -P data/raw/negatome https://mips.helmholtz-muenchen.de/proj/ppi/negatome/combined_stringent.txt
```

The `combined_stringent` list excludes pairs seen interacting in IntAct, making it the safest negative set.

### STRING (functional associations, positives — filter carefully)
Download **per species** (physical subnetwork + matching sequences). The loader keeps edges with combined score ≥ 700 **and** nonzero experimental/database evidence, and ships its own sequences so no UniProt map is needed.

```bash
mkdir -p data/raw/string
# Example: E. coli K-12 (taxid 511145). Repeat for each species you want.
wget -P data/raw/string https://stringdb-downloads.org/download/protein.physical.links.detailed.v12.0/511145.protein.physical.links.detailed.v12.0.txt.gz
wget -P data/raw/string https://stringdb-downloads.org/download/protein.sequences.v12.0/511145.protein.sequences.v12.0.fa.gz
```

### Literature-derived affinity (user-provided)
No canonical download. Drop CSVs into `data/raw/literature/` with a header row:
```csv
seq1,seq2,affinity_nm,interaction_label
MKT...,MSD...,12.5,
MGH...,MSD...,,1
```

`seq1`/`seq2` are amino-acid sequences (use a `:`-delimited value for a multi-chain side). `affinity_nm` (Kd in nM) and `interaction_label` (`1`/`0`) are optional; rows with neither default to a positive interaction. The canonical DuckDB table stores only the standardized `affinity_pkd` value, not raw nM.

## Inference Workflow
```
Protein Sequence → ProstT5 → Chain Embedding → Cache
```

Once cached, interaction prediction becomes extremely inexpensive.

This enables:
- massive interaction screening
- virtual proteome-wide searches
- repeated interaction prediction without recomputing embeddings

## TODO / Experiments
### Fine-Tuning Strategy
- [x] Frozen backbone + Adapters
- [ ] Frozen backbone + LoRA

### Residue → Chain Pooling
- [x] Mean pooling
- [ ] Max pooling
- [x] Attention pooling
- [ ] Learned weighted pooling

### Group Pooling
Evaluate permutation-invariant approaches:
- [ ] Mean pooling
- [ ] Max pooling
- [x] Attention pooling
- [ ] DeepSets
- [ ] Set Transformer

### Interaction Module
Test:
- [x] Pairwise interaction features
- [ ] Cross-attention between groups
- [ ] Bilinear interaction layers
- [ ] Small Transformer operating on chain embeddings

### Data Augmentation
- [ ] Sequence masking
- [ ] Residue dropout
- [x] Homology filtering
- [ ] Hard negative mining

### Loss Functions
- [ ] BCE
- [ ] Contrastive loss
- [x] Focal loss variants

## Project Inspiration
This project builds upon our previous work on **Prot2Prop**, a lightweight framework for multitask protein property prediction using pretrained protein language models.

Repository:

https://github.com/NeurosnapInc/Prot2Prop

Many of the engineering patterns developed for Prot2Prop are directly applicable to this project, including:

- Backbone abstraction
- Adapter-based fine-tuning
- Efficient embedding extraction
- Configuration-driven experiments
- Modular training loops
- Lightweight inference
- Dataset abstraction
- Benchmarking utilities

However, unlike Prot2Prop, which predicts properties of individual proteins, this project focuses on **interactions between arbitrary groups of proteins**. Consequently, significant new components will be introduced, including:

- Protein group encoders
- Permutation-invariant pooling
- Pairwise interaction modeling
- Cross-group attention mechanisms
- Interaction classification
- Complex-level representations

Although the machine learning architecture is substantially different, the overall repository organization and software engineering philosophy will remain intentionally similar to Prot2Prop wherever practical.


## Train Log
All historical result tables are stored in each run\'s `metrics.json`; original result notes live in `notes.md`. Use `python summarize_runs.py` to compare runs, `--historical` to print archived tables and notes, or `--json` to inspect all metadata. Compatible classification reports use the current evaluation fields; older classification and regression tables live in each split\'s `historical_reports`, with original columns, structured rows, and verbatim tables. Imported reports retain published four-decimal precision and README provenance. Undefined values are null, with original spelling preserved in verbatim tables. Historical definitions remain unchanged, and re-evaluation preserves the archives.

### Version 2026-07-13
#### Changes
- Switched tokenization from a random row split to a cluster-disjoint split to reduce sequence/homology leakage across train, validation, and test.
- Replaced naive random cluster assignment with label-aware greedy split assignment over connected cluster components. The splitter now balances total samples, interaction positives, interaction negatives, affinity-labeled samples, and source counts where possible.
- Standardized affinity labels to `pKd` in the aggregated DuckDB/tokenized cache so regression targets are comparable across PPB-Affinity, SKEMPI, and user-provided affinity sources.
- Updated training checkpoint selection to avoid early stopping on misleading aggregate `F1`/`MAE`. Classification selection now uses `AUROC` by default, regression uses normalized `MAE`, and tasks with too few validation labels are ignored for checkpoint selection.
- Increased token-capped batch sizes for A100 training throughput.

#### Results
- Validation affinity had moderate signal, while the interaction head still predicted almost everything as positive.

#### Validation Split
Results are stored in [metrics.json](runs/2026-07-13_seed_1/metrics.json) under `["splits"]["validation"]`. Historical tables are preserved in `historical_reports`.

#### Test Split
Results are stored in [metrics.json](runs/2026-07-13_seed_1/metrics.json) under `["splits"]["test"]`. Historical tables are preserved in `historical_reports`.

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

#### Validation Split
Results are stored in [metrics.json](runs/2026-07-18_seed_1/metrics.json) under `["splits"]["validation"]`. Historical tables are preserved in `historical_reports`.

#### Test Split
Results are stored in [metrics.json](runs/2026-07-18_seed_1/metrics.json) under `["splits"]["test"]`. Historical tables are preserved in `historical_reports`.

### Version 2026-07-19
#### Changes
- Kept the negative-aware interaction setup from the previous run: focal interaction loss, source-balanced sampling, and capped positive:negative sampling.
- Kept Huber loss for affinity regression (`REGRESSION_LOSS = "huber"`).
- Reverted affinity normalization from source-normalized pKd back to global pKd normalization (`AFFINITY_NORMALIZATION = "global"`) after the 2026-07-18 run showed weak source-normalized affinity performance.

#### Results
- Interaction classification improved with calibrated thresholding. Global affinity normalization did not recover test performance; separate source-specific modeling or data/split investigation was needed.

#### Validation Split
Results are stored in [metrics.json](runs/2026-07-19_seed_1/metrics.json) under `["splits"]["validation"]`. Historical tables are preserved in `historical_reports`.

#### Test Split
Results are stored in [metrics.json](runs/2026-07-19_seed_1/metrics.json) under `["splits"]["test"]`. Historical tables are preserved in `historical_reports`.

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

#### Validation Split
Results are stored in [metrics.json](runs/2026-08-12_seed_1/metrics.json) under `["splits"]["validation"]`. Historical tables are preserved in `historical_reports`.

#### Test Split
Results are stored in [metrics.json](runs/2026-08-12_seed_1/metrics.json) under `["splits"]["test"]`. Historical tables are preserved in `historical_reports`.

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

#### Validation Split
Classification metrics are stored in [metrics.json](runs/2026-08-13_seed_1/metrics.json) under `["splits"]["validation"]`.

#### Test Split
Classification metrics are stored in [metrics.json](runs/2026-08-13_seed_1/metrics.json) under `["splits"]["test"]`.

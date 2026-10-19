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
- [x] Max pooling
- [x] Attention pooling
- [x] Learned weighted pooling

#### Ablation Results:
- **Winner: max pooling** (`2026-10-07_05-36_seed_1`): best validation AUROC (0.9414) and test AUROC/AUPRC (0.9297/0.9843), with calibrated test balanced accuracy 0.8604, specificity 0.8194, and MCC 0.6192.
- Mean achieved higher calibrated test F1 (0.9472), but much weaker negative recall (0.4583); attention and learned weighting also trailed max on balanced detection.
- Learned weighting did not improve test ranking over attention (AUROC 0.8355 versus 0.8353) and had the lowest calibrated test balanced accuracy (0.6700).
- Select max as the next baseline; single-seed evidence and weak default-threshold IntAct recall remain caveats. Full comparison: [ablation notes](runs/2026-10-07_19-01_seed_1/notes.md).

### Group Pooling
Evaluate permutation-invariant approaches:
- [x] Mean pooling
- [x] Max pooling
- [x] Attention pooling
- [x] DeepSets
- [x] Set Transformer

#### Ablation Results:
- **Winner: mean group pooling** (`2026-10-08_01-46_seed_1`): highest validation AUROC (0.9551) and test AUROC (0.9356), with calibrated test balanced accuracy 0.8692, specificity 0.8148, and MCC 0.6611.
- Set Transformer had the highest test AUPRC (0.9876), but lower calibrated balanced accuracy (0.8416) and MCC (0.6298); it added 13 false positives for seven fewer false negatives versus mean.
- Max improved default-threshold test recall but trailed mean after calibration; DeepSets substantially weakened negative detection (calibrated test specificity 0.2130).
- Retain **residue max + group mean**. Results are single-seed evidence, IntAct positive recall remains weak, and repeated test inspection warrants a fresh final holdout. Full comparison: [final ablation notes](runs/2026-10-09_00-41_seed_1/notes.md).

### Interaction Module
Test:
- [x] Pairwise interaction features
- [x] Cross-attention between groups
- [x] Bilinear interaction layers
- [x] Small Transformer operating on chain embeddings

#### Ablation Results:
- **Winner: fixed symmetric pair features** (`2026-10-08_01-46_seed_1`): highest validation AUROC (0.9551), test AUROC/AUPRC (0.9356/0.9856), and calibrated test balanced accuracy/MCC (0.8692/0.6611).
- Chain Transformer was strongest among learned modules on test ranking (AUROC 0.9215), but calibrated specificity fell to 0.3380; it recovered 93 positives for 103 additional false positives versus baseline. Its F1 was essentially tied with baseline (0.9464 versus 0.9462).
- Cross-attention and bilinear also weakened negative rejection, with calibrated test balanced accuracy 0.6303 and 0.5600 respectively. Higher positive recall did not offset false positives.
- Restored **adapters + residue max + group mean + sum/absolute-difference/product features + focal-only loss**. Single-seed and repeated-test-inspection caveats apply; full comparison: [final interaction notes](runs/2026-10-12_21-35_seed_1/notes.md).

### Data Augmentation
- [x] Sequence masking
- [x] Residue dropout
- [x] Homology filtering

#### Ablation Results:
- **Retain no augmentation** (`2026-10-08_01-46_seed_1`) for ranking and balanced detection: test AUROC 0.9356, calibrated balanced accuracy 0.8692, specificity 0.8148, and MCC 0.6611. Homology filtering was held fixed, not separately ablated here.
- **10% residue-embedding dropout** improved calibrated test F1 (0.9567 versus 0.9462) with essentially unchanged MCC (0.6606), but recovered 57 positives for 32 additional false positives; balanced accuracy fell to 0.8155. A promising recall/F1 tradeoff, not an overall win. [Run notes](runs/2026-10-13_00-45_seed_1/notes.md).
- **10% sequence masking with X** underperformed dropout: test AUROC 0.9039, calibrated balanced accuracy 0.7953, MCC 0.6130, and F1 0.9503, with seven more false positives and eleven more false negatives than dropout. Live encoding and reduced token budgets limit attribution to masking alone. [Run notes](runs/2026-10-13_17-24_seed_1/notes.md).
- Both augmentations are disabled but retained. Results are single-seed evidence; repeated test inspection warrants a fresh final holdout before drawing definitive generalization conclusions.

### Loss Functions
- [x] BCE
- [x] BCE + contrastive term
- [x] Class-Weighted Focal Loss
- [x] Class-Weighted Focal Loss + contrastive term

#### Ablation Results:
- **Winner: class-weighted focal alone** (`2026-10-08_01-46_seed_1`): best validation AUROC (0.9551), test AUROC/AUPRC (0.9356/0.9856), and calibrated test balanced accuracy/MCC (0.8692/0.6611).
- BCE weakened calibrated negative detection (specificity 0.6944 versus focal's 0.8148); BCE + contrastive degraded it further (0.1481).
- Focal + contrastive achieved the highest calibrated test F1 (0.9514), but recovered 48 positives for 38 additional false positives versus focal alone; balanced accuracy fell to 0.7984 and MCC to 0.6206.
- Prefer focal alone with **residue max + group mean**. Neither tested contrastive term improved ranking and balanced detection over its classification-only baseline. Single-seed results and historical focal-weighting/reduction caveats apply; full comparison: [loss ablation notes](runs/2026-10-10_00-21_seed_1/notes.md).

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
Historical changes, rationale, and result summaries live in each run's `notes.md`; validation/test metrics live in `metrics.json`. Legacy tables retain their original columns, precision, and provenance in each split's `historical_reports`. Missing or undefined values are not reconstructed.

Use `python summarize_runs.py` to compare runs, `--historical` to print archived tables and notes, or `--json` to inspect all metadata. Re-evaluation preserves historical archives.

| Version | Changes and Results | Validation / Test Metrics |
| --- | --- | --- |
| 2026-07-13 | [notes.md](runs/2026-07-13_seed_1/notes.md) | [metrics.json](runs/2026-07-13_seed_1/metrics.json) |
| 2026-07-18 | [notes.md](runs/2026-07-18_seed_1/notes.md) | [metrics.json](runs/2026-07-18_seed_1/metrics.json) |
| 2026-07-19 | [notes.md](runs/2026-07-19_seed_1/notes.md) | [metrics.json](runs/2026-07-19_seed_1/metrics.json) |
| 2026-08-12 | [notes.md](runs/2026-08-12_seed_1/notes.md) | [metrics.json](runs/2026-08-12_seed_1/metrics.json) |
| 2026-08-13 | [notes.md](runs/2026-08-13_seed_1/notes.md) | [metrics.json](runs/2026-08-13_seed_1/metrics.json) |

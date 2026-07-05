# Protein Interaction Foundation Model (PIFM)
> A fast, scalable protein language model for predicting protein-protein interactions (PPI) and binding affinity between arbitrary protein complexes.

## Overview
The goal of this project is to develop a lightweight and highly efficient model capable of predicting whether two groups of proteins interact and, if so, estimating their binding affinity.

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
- Predict binding affinity (log-scale Kd / pKd)
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
│ Affinity (pKd) │
└────────────────┘
```

## Training Strategy
### Phase 1
Freeze ProstT5 completely.

Train only:

- pooling layers
- interaction module
- prediction head

### Phase 2
Enable adapter training.
Fine-tune:
- LoRA
- adapters

while leaving the backbone frozen.
This should improve performance while remaining lightweight.

### Multi-Task Learning
Rather than training only affinity regression, jointly train:
- interaction classification
- affinity prediction

Advantages:
- Better regularization
- Improved generalization
- More useful embeddings
- Handles noisy affinity labels better

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

### Adding a new source
1. Add `sources/<name>.py` with `def iter_<name>() -> Iterator[InteractionEntry]` (yield **sequences only**; set `interaction_label` and/or `affinity_nm`).
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
| PPB-Affinity filtered | affinity | positive | free (Hugging Face) |
| SKEMPI v2.0 | affinity | positive | free (~32 MB) |
| IntAct | binary | positive + negative | free (FTP, large ZIP) |
| Negatome 2.0 | binary | **negative** | free |
| STRING (filtered) | binary | positive | free (per species) |
| literature-derived | affinity (+optional binary) | user-defined | user-provided CSV |

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
- repeated affinity prediction without recomputing embeddings

## TODO / Experiments
### Fine-Tuning Strategy
- [ ] Frozen backbone
- [ ] LoRA
- [ ] Adapters

### Residue → Chain Pooling
- [ ] Mean pooling
- [ ] Max pooling
- [ ] Attention pooling
- [ ] CLS token (if available)
- [ ] Learned weighted pooling

### Group Pooling
Evaluate permutation-invariant approaches:
- [ ] Mean pooling
- [ ] Max pooling
- [ ] Attention pooling
- [ ] DeepSets
- [ ] Set Transformer

### Interaction Module
Test:
- [ ] Pairwise interaction features
- [ ] Cross-attention between groups
- [ ] Bilinear interaction layers
- [ ] Small Transformer operating on chain embeddings

### Output Heads
Evaluate:
- [ ] Binary interaction
- [ ] pKd regression
- [ ] Joint multitask training
- [ ] Uncertainty estimation

### Data Augmentation
- [ ] Sequence masking
- [ ] Residue dropout
- [ ] Homology filtering
- [ ] Hard negative mining

### Loss Functions
- [ ] BCE
- [ ] MSE
- [ ] Huber
- [ ] Contrastive loss
- [ ] Multi-task weighted losses

### Evaluation Metrics
Classification:
- [ ] ROC-AUC
- [ ] PR-AUC
- [ ] F1
- [ ] Accuracy

Regression:
- [ ] Pearson
- [ ] Spearman
- [ ] RMSE
- [ ] MAE
- [ ] R²

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
- Multi-task interaction and affinity prediction
- Complex-level representations

Although the machine learning architecture is substantially different, the overall repository organization and software engineering philosophy will remain intentionally similar to Prot2Prop wherever practical.

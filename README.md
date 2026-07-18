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

## GPU Memory Notes
On standard 48 GB VRAM GPU instances, PyTorch may fail with a CUDA out-of-memory error even when enough total memory should be available, because a large amount of memory is reserved but unallocated by PyTorch. Set the allocator configuration before launching training or inference:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This can reduce allocator fragmentation and avoid failures such as attempts to allocate another large CUDA segment on a mostly reserved GPU. See the PyTorch CUDA memory management documentation for details:
https://pytorch.org/docs/stable/notes/cuda.html#environment-variables

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


## Train Log
### Version 2026-07-13
#### Changes
- Switched tokenization from a random row split to a cluster-disjoint split to reduce sequence/homology leakage across train, validation, and test.
- Replaced naive random cluster assignment with label-aware greedy split assignment over connected cluster components. The splitter now balances total samples, interaction positives, interaction negatives, affinity-labeled samples, and source counts where possible.
- Standardized affinity labels to `pKd` in the aggregated DuckDB/tokenized cache so regression targets are comparable across PPB-Affinity, SKEMPI, and user-provided affinity sources.
- Updated training checkpoint selection to avoid early stopping on misleading aggregate `F1`/`MAE`. Classification selection now uses `AUROC` by default, regression uses normalized `MAE`, and tasks with too few validation labels are ignored for checkpoint selection.
- Increased token-capped batch sizes for A100 training throughput.

#### Results
- The affinity validation set is now large enough to interpret (`n=1042`) and the model learns a moderate affinity signal. The interaction head has good ranking signal (`AUROC=0.8828`) but poor thresholded negative detection, still predicting almost everything as positive.

#### Validation Split
```
Dataset size (validation): 1611 pairs

Classification Tasks
task         n     acc     bal_acc  precision  recall  f1      auroc   auprc   label_ratio      pred_ratio
-----------  ----  ------  -------  ---------  ------  ------  ------  ------  ---------------  ---------------
interaction  1611  0.8672  0.5046   0.8670     1.0000  0.9288  0.8828  0.9781  0:0.134 1:0.866  0:0.001 1:0.999

Regression Tasks
task      n     label_mean  label_std  pred_mean  pred_std  mae     rmse    pearson  spearman  r2
--------  ----  ----------  ---------  ---------  --------  ------  ------  -------  --------  ------
affinity  1042  7.0140      1.9803     7.6647     1.2690    1.5393  1.8678  0.4908   0.4500    0.1104

Checkpoint Classification Calibration Applied
task         cal_n  thr     acc     bal_acc  precision  recall  f1      auroc   auprc   label_ratio      pred_ratio
-----------  -----  ------  ------  -------  ---------  ------  ------  ------  ------  ---------------  ---------------
interaction  1611   0.8700  0.8759  0.5429   0.8760     0.9978  0.9330  0.8828  0.9781  0:0.134 1:0.866  0:0.014 1:0.986

Checkpoint Regression Calibration Applied
task      cal_n  slope   intercept  pred_mean  pred_std  mae     rmse    pearson  spearman  r2
--------  -----  ------  ---------  ---------  --------  ------  ------  -------  --------  ------
affinity  1042   0.7669  1.1370     7.0153     0.9732    1.3849  1.7254  0.4908   0.4500    0.2409
```

#### Test Split
```
Classification Tasks
task         n     acc     bal_acc  precision  recall  f1      auroc   auprc   label_ratio      pred_ratio
-----------  ----  ------  -------  ---------  ------  ------  ------  ------  ---------------  ---------------
interaction  1613  0.8772  0.5436   0.8763     0.9993  0.9338  0.9474  0.9907  0:0.134 1:0.866  0:0.012 1:0.988

Regression Tasks
task      n     label_mean  label_std  pred_mean  pred_std  mae     rmse    pearson  spearman  r2
--------  ----  ----------  ---------  ---------  --------  ------  ------  -------  --------  -------
affinity  1042  7.4975      2.1386     7.4847     1.1830    1.9024  2.4361  0.0077   0.0571    -0.2975

Checkpoint Classification Calibration Applied
task         cal_n  thr     acc     bal_acc  precision  recall  f1      auroc   auprc   label_ratio      pred_ratio
-----------  -----  ------  ------  -------  ---------  ------  ------  ------  ------  ---------------  ---------------
interaction  1611   0.8700  0.9033  0.6448   0.9011     0.9979  0.9470  0.9474  0.9907  0:0.134 1:0.866  0:0.041 1:0.959

Checkpoint Regression Calibration Applied
task      cal_n  slope   intercept  pred_mean  pred_std  mae     rmse    pearson  spearman  r2
--------  -----  ------  ---------  ---------  --------  ------  ------  -------  --------  -------
affinity  1042   0.7669  1.1370     6.8773     0.9073    1.8411  2.3983  0.0077   0.0571    -0.2576
```

### Version 2026-07-18
#### Changes
- Added negative-aware interaction training to address the previous failure mode where the model ranked interactions well but predicted almost everything as positive.
- Switched the interaction loss from weighted cross-entropy to configurable focal loss (`INTERACTION_LOSS = "focal"`, `FOCAL_GAMMA = 2.0`).
- Added weighted sampling controls so each epoch sees a less extreme interaction class balance (`INTERACTION_POS_NEG_RATIO = 5.0`) instead of reflecting the raw positive-heavy dataset distribution.
- Added source-balanced sampling so high-volume sources do not dominate every training epoch.
- Switched affinity regression from MSE to Huber loss for more robustness to noisy/outlier pKd labels.
- Added source-normalized affinity training/reporting to test whether PPB-Affinity and SKEMPI should be normalized separately before regression.
#### Results
- Interaction classification improved in the intended direction: calibrated test balanced accuracy increased and the model now predicts negatives at a realistic rate instead of collapsing to nearly all-positive predictions.
- Calibrated test interaction metrics were `AUROC=0.9233`, `AUPRC=0.9871`, `balanced_acc=0.7324`, `specificity=0.5370`, and `MCC=0.4638`.
- Negatome negative handling improved materially on the test split: `152/216` negatives were correctly predicted (`specificity=0.7037` for the uncalibrated source-specific row).
- Affinity remains weak overall. Source-specific test ranking is better for SKEMPI (`Pearson=0.4555`, `Spearman=0.5930`) than PPB-Affinity (`Pearson=0.2416`, `Spearman=0.2549`).
- Source-normalized affinity regression did not solve the regression problem: source-normalized test `Pearson=0.2450`, `Spearman=0.2712`, and `R2=0.0011`.
- Conclusion: keep the negative-aware interaction training changes, but treat source-normalized affinity training as experimental. The next affinity run should likely keep Huber loss but return to global pKd normalization, then consider separate PPB/SKEMPI heads if source bias remains large.

#### Validation Split
```
Source-Specific Classification Tasks
source           task         n    acc     bal_acc  precision  recall  specificity  neg_recall  f1      mcc     tn   fp  fn  tp   auroc  auprc   label_ratio  pred_ratio
---------------  -----------  ---  ------  -------  ---------  ------  -----------  ----------  ------  ------  ---  --  --  ---  -----  ------  -----------  ---------------
intact_negative  interaction  1    1.0000  1.0000   0.0000     0.0000  1.0000       1.0000      0.0000  0.0000  1    0   0   0    nan    0.0000  0:1.000      0:1.000
intact_positive  interaction  27   0.1481  0.1481   1.0000     0.1481  -            -           0.2581  0.0000  0    0   23  4    nan    1.0000  1:1.000      0:0.852 1:0.148
negatome         interaction  215  0.6279  0.6279   0.0000     0.0000  0.6279       0.6279      0.0000  0.0000  135  80  0   0    nan    0.0000  0:1.000      0:0.628 1:0.372
ppb_affinity     interaction  748  0.9759  0.9759   1.0000     0.9759  -            -           0.9878  0.0000  0    0   18  730  nan    1.0000  1:1.000      0:0.024 1:0.976
skempi           interaction  306  1.0000  1.0000   1.0000     1.0000  -            -           1.0000  0.0000  0    0   0   306  nan    1.0000  1:1.000      1:1.000
string           interaction  314  0.9268  0.9268   1.0000     0.9268  -            -           0.9620  0.0000  0    0   23  291  nan    1.0000  1:1.000      0:0.073 1:0.927

Source-Specific Regression Tasks
source        task      n    label_mean  label_std  pred_mean  pred_std  mae     rmse    pearson  spearman  r2
------------  --------  ---  ----------  ---------  ---------  --------  ------  ------  -------  --------  -------
ppb_affinity  affinity  748  7.2421      2.0063     7.4813     0.8647    1.6852  2.0780  0.1476   0.1423    -0.0727
skempi        affinity  294  6.4336      1.7855     7.1102     0.8845    1.3561  1.6045  0.5869   0.6298    0.1924

Source-Normalized Regression Tasks
task      n     label_mean  label_std  pred_mean  pred_std  mae     rmse    pearson  spearman  r2
--------  ----  ----------  ---------  ---------  --------  ------  ------  -------  --------  ------
affinity  1042  -0.0000     1.0000     0.1925     0.4650    0.8172  0.9990  0.2746   0.3027    0.0021

Checkpoint Classification Calibration Applied
task         cal_n  thr     acc     bal_acc  precision  recall  specificity  neg_recall  f1      mcc     tn   fp   fn  tp    auroc   auprc   label_ratio      pred_ratio
-----------  -----  ------  ------  -------  ---------  ------  -----------  ----------  ------  ------  ---  ---  --  ----  ------  ------  ---------------  ---------------
interaction  1611   0.0700  0.9162  0.7501   0.9297     0.9771  0.5231       0.5231      0.9528  0.5955  113  103  32  1363  0.8911  0.9713  0:0.134 1:0.866  0:0.090 1:0.910

Checkpoint Regression Calibration Applied
task      cal_n  slope   intercept  pred_mean  pred_std  mae     rmse    pearson  spearman  r2
--------  -----  ------  ---------  ---------  --------  ------  ------  -------  --------  ------
affinity  1042   0.6439  2.2642     7.0138     0.5706    1.4919  1.8961  0.2884   0.2590    0.0832
```

#### Test Split
```
Source-Specific Classification Tasks
source           task         n    acc     bal_acc  precision  recall  specificity  neg_recall  f1      mcc     tn   fp  fn   tp   auroc  auprc   label_ratio  pred_ratio
---------------  -----------  ---  ------  -------  ---------  ------  -----------  ----------  ------  ------  ---  --  ---  ---  -----  ------  -----------  ---------------
intact_positive  interaction  22   0.1364  0.1364   1.0000     0.1364  -            -           0.2400  0.0000  0    0   19   3    nan    1.0000  1:1.000      0:0.864 1:0.136
negatome         interaction  216  0.7037  0.7037   0.0000     0.0000  0.7037       0.7037      0.0000  0.0000  152  64  0    0    nan    0.0000  0:1.000      0:0.704 1:0.296
ppb_affinity     interaction  761  0.8489  0.8489   1.0000     0.8489  -            -           0.9183  0.0000  0    0   115  646  nan    1.0000  1:1.000      0:0.151 1:0.849
skempi           interaction  300  1.0000  1.0000   1.0000     1.0000  -            -           1.0000  0.0000  0    0   0    300  nan    1.0000  1:1.000      1:1.000
string           interaction  314  0.9777  0.9777   1.0000     0.9777  -            -           0.9887  0.0000  0    0   7    307  nan    1.0000  1:1.000      0:0.022 1:0.978

Source-Specific Regression Tasks
source        task      n    label_mean  label_std  pred_mean  pred_std  mae     rmse    pearson  spearman  r2
------------  --------  ---  ----------  ---------  ---------  --------  ------  ------  -------  --------  -------
ppb_affinity  affinity  761  7.3170      2.0450     7.6825     0.9183    1.6317  2.0620  0.2416   0.2549    -0.0166
skempi        affinity  281  7.9862      2.3037     7.1203     0.7470    1.7440  2.2466  0.4555   0.5930    0.0490

Source-Normalized Regression Tasks
task      n     label_mean  label_std  pred_mean  pred_std  mae     rmse    pearson  spearman  r2
--------  ----  ----------  ---------  ---------  --------  ------  ------  -------  --------  ------
affinity  1042  -0.0000     1.0000     0.0292     0.4860    0.7869  0.9995  0.2450   0.2712    0.0011

Checkpoint Classification Calibration Applied
task         cal_n  thr     acc     bal_acc  precision  recall  specificity  neg_recall  f1      mcc     tn   fp   fn   tp    auroc   auprc   label_ratio      pred_ratio
-----------  -----  ------  ------  -------  ---------  ------  -----------  ----------  ------  ------  ---  ---  ---  ----  ------  ------  ---------------  ---------------
interaction  1611   0.0700  0.8754  0.7324   0.9284     0.9277  0.5370       0.5370      0.9280  0.4638  116  100  101  1296  0.9233  0.9871  0:0.134 1:0.866  0:0.135 1:0.865

Checkpoint Regression Calibration Applied
task      cal_n  slope   intercept  pred_mean  pred_std  mae     rmse    pearson  spearman  r2
--------  -----  ------  ---------  ---------  --------  ------  ------  -------  --------  ------
affinity  1042   0.6439  2.2642     7.1132     0.5861    1.6253  2.1122  0.2407   0.2844    0.0245
```
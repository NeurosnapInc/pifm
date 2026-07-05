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

---

# Objectives

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

---

# Motivation

Protein property prediction can be solved effectively using pooled protein embeddings from pretrained protein language models.

Protein interaction prediction is fundamentally different because:

- Inputs consist of multiple proteins
- Each side may contain one or more chains
- Protein order should not affect predictions
- Interactions occur between groups rather than individual sequences

This project investigates architectures capable of learning interactions between arbitrary protein sets while remaining computationally efficient.

---

# Proposed Architecture

## High-Level Pipeline

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
      ┌───────────────┐
      │ Interaction   │
      │ Affinity      │
      └───────────────┘
```

---

# Stage 1: Protein Encoding

Each protein sequence is independently embedded using ProstT5.

Initial plan:

- Frozen ProstT5
- Train lightweight adapters (LoRA or adapters)
- Optionally cache embeddings during training

Advantages:

- Fast
- Memory efficient
- Embeddings reusable
- Easily swapped with newer protein LMs

Each protein produces residue embeddings.

---

# Stage 2: Chain Embedding

Residue embeddings must be converted into a fixed-size chain representation.

Initial baseline:

- Mean pooling

Future alternatives are listed below.

Result:

```
Protein Sequence

↓

Residue Embeddings

↓

Chain Embedding
```

---

# Stage 3: Protein Group Representation

Each interaction side consists of one or more proteins.

Example:

Group A

```
Protein A
Protein B
Protein C
```

Group B

```
Protein D
Protein E
```

These must be converted into a single embedding representing the entire complex.

Importantly, the representation should be **permutation invariant**, meaning:

```
[A, B, C]
```

and

```
[C, A, B]
```

should produce identical group representations.

This avoids introducing arbitrary ordering bias.

---

# Stage 4: Interaction Modeling

Simply concatenating two pooled embeddings may discard important chain-level interactions.

Instead, we propose explicitly modeling interactions between chains.

Example:

```
A1 ↔ B1
A1 ↔ B2

A2 ↔ B1
A2 ↔ B2

...
```

Each pair generates an interaction embedding.

Possible features:

```
[Ai]

[Bj]

|Ai − Bj|

Ai * Bj
```

where * denotes element-wise multiplication.

These pairwise interaction embeddings are then pooled before final prediction.

---

# Stage 5: Prediction Head

Outputs may include:

## Binary Classification

```
P(interaction)
```

## Regression

Predict

```
log10(Kd)
```

or preferably

```
pKd = -log10(Kd)
```

Using pKd provides a more numerically stable regression target.

Potential future outputs:

- confidence
- uncertainty
- interface confidence
- residue attribution

---

# Why Not Concatenate Chains?

One possible architecture is

```
Protein1 <CHAIN> Protein2 <CHAIN> Protein3
```

fed directly into ProstT5.

Advantages:

- Simple
- Minimal downstream architecture

Potential disadvantages:

- Chain ordering matters
- Long complexes exceed context length
- ProstT5 was not trained to interpret arbitrary chain delimiters
- Difficult to cache individual protein embeddings

This remains an experimental direction but is not currently the primary architecture.

---

# Training Strategy

## Phase 1

Freeze ProstT5 completely.

Train only:

- pooling layers
- interaction module
- prediction head

## Phase 2

Enable adapter training.

Fine-tune:

- LoRA
- adapters

while leaving the backbone frozen.

This should improve performance while remaining lightweight.

---

# Multi-Task Learning

Rather than training only affinity regression, jointly train:

- interaction classification
- affinity prediction

Advantages:

- Better regularization
- Improved generalization
- More useful embeddings
- Handles noisy affinity labels better

---

# Dataset Considerations

Potential data sources include:

- SKEMPI
- BioLiP
- Negatome
- IntAct
- DIP
- STRING (carefully filtered)
- literature-derived affinity datasets

Need to carefully distinguish:

Positive interactions

vs

Negative interactions

Affinity labels

vs

Binary interaction labels


Two curated sources are currently supported:

- **PPB‑Affinity (filtered)** – a comprehensive dataset of crystal structures of protein–protein complexes, including binding affinities, receptor chains, and ligand chains. It is the largest publicly available PPB dataset, combining receptor protein chain, ligand protein chain, and experimentally measured affinity values.
- **SKEMPI v2.0** – a database of binding free‑energy and kinetic changes upon mutation for protein–protein interactions with solved structures. Version 2.0 contains data for 7,085 mutations, recording thermodynamic parameters, kinetic rate constants, and, where available, cleaned crystal structures of the complexes.

These sources were chosen because they provide sequence‑resolvable protein chains and measured affinities, which map naturally to the `group1`/`group2` schema used by the aggregation pipeline. Binary interaction labels are inferred to be positive for these curated datasets.

## Data Directory Structure

```
data/
├─ raw/          # input files downloaded from each source (see below)
└─ aggregated/
   └─ aggregated.duckdb  # database created by the aggregation script
```

All raw files must reside under `data/raw/` before running the aggregator.

## Download Instructions

### 1. Prepare the raw data directory

Create the folder if it does not already exist:

```bash
mkdir -p data/raw
```

### 2. Get the PPB‑Affinity (filtered) dataset

Go to the PPB‑Affinity dataset page on Hugging Face and download the file `filtered.csv` from the files → filtered section. Save it as:

```
data/raw/ppb_affinity_filtered.csv
```

This CSV provides pre‑extracted "Ligand Sequences" and "Receptor Sequences" columns, and a "KD(M)" column containing dissociation constants in molar units. If there are multiple chains on either side, the sequences are comma‑separated.

### 3. Get the SKEMPI v2.0 dataset

Visit the SKEMPI v2.0 website and follow the "Download" link to obtain two files:

- `skempi_v2.csv` – contains affinities, mutation information, and PDB identifiers.
- `SKEMPI2_PDBs.tgz` – a tarball of cleaned PDB structures.

After downloading, place them in `data/raw/` with the following names:

```
data/raw/skempi_v2.csv
data/raw/SKEMPI2_PDBs.tgz
```

Optional — unpack the PDB archive for inspection:

```bash
tar -xzf data/raw/SKEMPI2_PDBs.tgz -C data/raw
```

SKEMPI v2.0 contains data on thermodynamic parameters and kinetic rate constants for protein–protein interactions with solved structures, with a total of 7,085 mutation entries.

---

# Inference Workflow

```
Protein Sequence

↓

ProstT5

↓

Chain Embedding

↓

Cache
```

Once cached, interaction prediction becomes extremely inexpensive.

This enables:

- massive interaction screening
- virtual proteome-wide searches
- repeated affinity prediction without recomputing embeddings

---

# Initial Baseline Model

```
Protein

↓

Frozen ProstT5

↓

Mean Pool

↓

Group Mean Pool

↓

MLP

↓

Interaction + pKd
```

This establishes a simple benchmark before adding more sophisticated architectures.

---

# Future Architecture

```
Protein

↓

ProstT5 + Adapters

↓

Attention Pool

↓

Set Encoder

↓

Pairwise Interaction Network

↓

MLP

↓

Interaction + pKd
```

---

# TODO / Experiments

## Protein Encoder

- [ ] ProstT5
- [ ] ESM2
- [ ] ProtT5
- [ ] Future larger protein language models

---

## Fine-Tuning Strategy

- [ ] Frozen backbone
- [ ] LoRA
- [ ] Adapters
- [ ] Full fine-tuning

---

## Residue → Chain Pooling

- [ ] Mean pooling
- [ ] Max pooling
- [ ] Attention pooling
- [ ] CLS token (if available)
- [ ] Learned weighted pooling

---

## Group Pooling

Evaluate permutation-invariant approaches:

- [ ] Mean pooling
- [ ] Max pooling
- [ ] Attention pooling
- [ ] DeepSets
- [ ] Set Transformer

---

## Interaction Module

Test:

- [ ] Pairwise interaction features
- [ ] Cross-attention between groups
- [ ] Bilinear interaction layers
- [ ] Small Transformer operating on chain embeddings

---

## Output Heads

Evaluate:

- [ ] Binary interaction
- [ ] pKd regression
- [ ] Joint multitask training
- [ ] Uncertainty estimation

---

## Input Formatting Experiments

- [ ] Independent protein encoding
- [ ] Chain delimiter token
- [ ] Group delimiter token
- [ ] Entire complex encoded jointly
- [ ] Learned separator embeddings

---

## Data Augmentation

- [ ] Random chain order shuffling
- [ ] Sequence masking
- [ ] Residue dropout
- [ ] Homology filtering
- [ ] Hard negative mining

---

## Loss Functions

- [ ] BCE
- [ ] MSE
- [ ] Huber
- [ ] Contrastive loss
- [ ] Multi-task weighted losses

---

## Evaluation Metrics

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

---

## Speed Benchmarks

Benchmark against:

- [ ] AlphaFold-Multimer
- [ ] FoldDock
- [ ] Existing PPI embedding models
- [ ] Sequence-only baselines

Metrics:

- inference time
- GPU memory
- throughput
- embeddings/sec

---

# Long-Term Vision

The long-term goal is to create a general-purpose interaction foundation model capable of rapidly scoring interactions between arbitrary biomolecular systems.

Future extensions may include:

- protein-protein interactions
- antibody-antigen binding
- peptide binding
- enzyme-substrate interactions
- protein-ligand interactions
- protein-DNA interactions
- protein-RNA interactions
- interface residue prediction
- mutation affinity prediction (ΔΔG)
- binder screening
- affinity maturation
- interaction network analysis

By leveraging pretrained protein language models together with lightweight interaction-specific architectures, this project aims to achieve near state-of-the-art predictive performance while remaining fast enough for large-scale computational screening and practical deployment.

# Project Inspiration

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

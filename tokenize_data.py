"""
Pre-tokenize aggregated protein-pair interaction data into train/validation/test caches.
"""

import csv
import hashlib
import random
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from typing import Dict, List

import duckdb
import torch
from transformers import T5Tokenizer

from config import (
  AGGREGATED_DB_PATH,
  CLUSTER_COVERAGE,
  CLUSTER_MIN_SEQ_ID,
  MAX_LENGTH,
  MMSEQS_BINARY,
  MODEL_NAME,
  SEQUENCE_CLUSTER_FASTA_PATH,
  SEQUENCE_CLUSTER_TSV_PATH,
  SEQUENCE_CLUSTER_WORK_DIR,
  SPLIT_SEED,
  SPLIT_STRATEGY,
  TEST_FRACTION,
  TOKENIZED_DATA_DIR,
  TRAIN_CACHE_PATH,
  TRAIN_FRACTION,
  VAL_FRACTION,
)


TOKENIZE_BATCH_SIZE = 128
SUPPORTED_SPLIT_STRATEGIES = {"random", "protein", "cluster"}

TASK_ORDER = ["interaction"]
TASK_METAS = {
  "interaction": {
    "task_name": "interaction",
    "dtype": "bool",
    "head_type": "pair_binary",
    "num_classes": 2,
    "loss": "ce",
  },
}


def _validate_split_fractions():
  total = TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION
  if abs(total - 1.0) > 1e-8:
    raise ValueError(f"Split fractions must sum to 1.0, got {total}")


def _preprocess_sequence(seq: str) -> str:
  seq = re.sub(r"[UZOB]", "X", seq.upper())
  return "<AA2fold> " + " ".join(seq)


def _sequence_id(sequence: str) -> str:
  return hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:16]


def _record_sequences(record) -> List[str]:
  return record["group1_sequences"] + record["group2_sequences"]


def _all_unique_sequences(records) -> List[str]:
  return sorted({sequence for record in records for sequence in _record_sequences(record)})


def _write_sequence_fasta(sequences: List[str], path):
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w") as handle:
    for sequence in sequences:
      handle.write(f">{_sequence_id(sequence)}\n{sequence}\n")


def _read_cluster_assignments(path, valid_sequence_ids) -> Dict[str, str]:
  assignments: Dict[str, str] = {}

  with path.open(newline="") as handle:
    sample = handle.read(4096)
    handle.seek(0)
    delimiter = "," if "," in sample and "\t" not in sample else "\t"
    reader = csv.reader(handle, delimiter=delimiter)

    for row in reader:
      if not row or len(row) < 2:
        continue

      first = row[0].strip()
      second = row[1].strip()
      if first in {"cluster_id", "representative"} or second in {"sequence_id", "member"}:
        continue

      if second in valid_sequence_ids:
        assignments[second] = first
      elif first in valid_sequence_ids:
        assignments[first] = second

  missing = valid_sequence_ids - set(assignments)
  if missing:
    preview = ", ".join(sorted(missing)[:5])
    raise ValueError(
      f"Cluster assignment file {path} is missing {len(missing)} sequences. "
      f"Examples: {preview}"
    )

  return assignments


def _run_mmseqs_clustering(sequences: List[str]):
  if shutil.which(MMSEQS_BINARY) is None:
    raise RuntimeError(
      f"SPLIT_STRATEGY='cluster' requires {MMSEQS_BINARY!r} or a precomputed "
      f"cluster file at {SEQUENCE_CLUSTER_TSV_PATH}. Install MMseqs2 or provide "
      "a two-column cluster TSV mapping cluster_id to sequence_id."
    )

  _write_sequence_fasta(sequences, SEQUENCE_CLUSTER_FASTA_PATH)
  output_prefix = SEQUENCE_CLUSTER_TSV_PATH.with_suffix("")
  SEQUENCE_CLUSTER_WORK_DIR.mkdir(parents=True, exist_ok=True)

  command = [
    MMSEQS_BINARY,
    "easy-cluster",
    str(SEQUENCE_CLUSTER_FASTA_PATH),
    str(output_prefix),
    str(SEQUENCE_CLUSTER_WORK_DIR),
    "--min-seq-id",
    str(CLUSTER_MIN_SEQ_ID),
    "-c",
    str(CLUSTER_COVERAGE),
    "--cov-mode",
    "0",
  ]
  subprocess.run(command, check=True)

  generated_tsv = output_prefix.parent / f"{output_prefix.name}_cluster.tsv"
  if not generated_tsv.exists():
    raise FileNotFoundError(f"MMseqs2 did not create expected cluster TSV: {generated_tsv}")

  generated_tsv.replace(SEQUENCE_CLUSTER_TSV_PATH)


def _load_sequence_clusters(records) -> Dict[str, str]:
  sequences = _all_unique_sequences(records)
  sequence_to_id = {sequence: _sequence_id(sequence) for sequence in sequences}
  valid_sequence_ids = set(sequence_to_id.values())

  if not SEQUENCE_CLUSTER_TSV_PATH.exists():
    print(
      f"Cluster file not found at {SEQUENCE_CLUSTER_TSV_PATH}; running MMseqs2 "
      f"min_seq_id={CLUSTER_MIN_SEQ_ID} coverage={CLUSTER_COVERAGE}"
    )
    _run_mmseqs_clustering(sequences)

  id_to_cluster = _read_cluster_assignments(SEQUENCE_CLUSTER_TSV_PATH, valid_sequence_ids)
  return {sequence: id_to_cluster[sequence_id] for sequence, sequence_id in sequence_to_id.items()}


class _DisjointSet:
  def __init__(self, units):
    self.parent = {unit: unit for unit in units}

  def find(self, unit):
    parent = self.parent[unit]
    if parent != unit:
      self.parent[unit] = self.find(parent)
    return self.parent[unit]

  def union(self, first, second):
    first_root = self.find(first)
    second_root = self.find(second)
    if first_root != second_root:
      self.parent[second_root] = first_root


def _record_unit_set(record, sequence_to_unit: Dict[str, str]):
  return {sequence_to_unit[sequence] for sequence in _record_sequences(record)}


def _record_label_stats(record) -> Counter:
  stats = Counter(total=1)
  label = record["labels"][0]
  if label > 0.5:
    stats["interaction_pos"] += 1
  else:
    stats["interaction_neg"] += 1
  stats[f"source:{record['source']}"] += 1
  return stats


def _build_split_units(records, sequence_to_unit: Dict[str, str]):
  base_units = sorted(set(sequence_to_unit.values()))
  dsu = _DisjointSet(base_units)

  # A sample can link multiple sequence clusters; those clusters must move together
  # or the sample would either leak across splits or need to be discarded.
  for record in records:
    record_units = sorted(_record_unit_set(record, sequence_to_unit))
    if not record_units:
      continue
    first = record_units[0]
    for unit in record_units[1:]:
      dsu.union(first, unit)

  unit_to_component = {unit: dsu.find(unit) for unit in base_units}
  component_records = defaultdict(list)
  component_stats = defaultdict(Counter)

  for record in records:
    record_units = _record_unit_set(record, sequence_to_unit)
    component = unit_to_component[next(iter(record_units))]
    component_records[component].append(record)
    component_stats[component].update(_record_label_stats(record))

  return component_records, component_stats


def _weighted_unit_size(stats: Counter):
  # Negatives are scarce, so sort them earlier during greedy placement.
  return (
    stats["total"]
    + 20 * stats["interaction_neg"]
    + 2 * stats["interaction_pos"]
  )


def _feature_weight(feature_name: str):
  if feature_name == "interaction_neg":
    return 20.0
  if feature_name.startswith("source:"):
    return 0.5
  return 1.0


def _split_balance_score(split_stats, total_stats):
  fractions = {
    "train": TRAIN_FRACTION,
    "validation": VAL_FRACTION,
    "test": TEST_FRACTION,
  }
  score = 0.0

  for split_name, fraction in fractions.items():
    for feature_name, total_value in total_stats.items():
      if total_value <= 0:
        continue
      target = total_value * fraction
      observed = split_stats[split_name][feature_name]
      diff = (observed - target) / max(1.0, target)
      score += _feature_weight(feature_name) * diff * diff

  return score


def _assign_components_label_aware(component_stats, rng):
  components = list(component_stats)
  rng.shuffle(components)
  components.sort(key=lambda component: _weighted_unit_size(component_stats[component]), reverse=True)

  total_stats = Counter()
  for stats in component_stats.values():
    total_stats.update(stats)

  split_stats = {
    "train": Counter(),
    "validation": Counter(),
    "test": Counter(),
  }
  component_to_split = {}

  for component in components:
    stats = component_stats[component]
    best_split = None
    best_score = None

    for split_name in ("train", "validation", "test"):
      split_stats[split_name].update(stats)
      score = _split_balance_score(split_stats, total_stats)
      split_stats[split_name].subtract(stats)

      if best_score is None or score < best_score:
        best_score = score
        best_split = split_name

    component_to_split[component] = best_split
    split_stats[best_split].update(stats)

  return component_to_split


def _split_records_random(records, rng):
  indices = list(range(len(records)))
  rng.shuffle(indices)

  n_total = len(indices)
  n_train = int(TRAIN_FRACTION * n_total)
  n_val = int(VAL_FRACTION * n_total)
  return {
    "train": [records[i] for i in indices[:n_train]],
    "validation": [records[i] for i in indices[n_train:n_train + n_val]],
    "test": [records[i] for i in indices[n_train + n_val:]],
  }, 0


def _split_records_by_sequence_units(records, sequence_to_unit: Dict[str, str], rng):
  component_records, component_stats = _build_split_units(records, sequence_to_unit)
  component_to_split = _assign_components_label_aware(component_stats, rng)
  split_records = {"train": [], "validation": [], "test": []}

  for component, rows in component_records.items():
    split_records[component_to_split[component]].extend(rows)

  return split_records, 0


def _split_records(records):
  if SPLIT_STRATEGY not in SUPPORTED_SPLIT_STRATEGIES:
    raise ValueError(
      f"Unsupported SPLIT_STRATEGY={SPLIT_STRATEGY!r}. "
      f"Expected one of {sorted(SUPPORTED_SPLIT_STRATEGIES)}."
    )

  rng = random.Random(SPLIT_SEED)
  if SPLIT_STRATEGY == "random":
    return _split_records_random(records, rng)

  if SPLIT_STRATEGY == "protein":
    sequence_to_unit = {sequence: sequence for sequence in _all_unique_sequences(records)}
    return _split_records_by_sequence_units(records, sequence_to_unit, rng)

  sequence_to_unit = _load_sequence_clusters(records)
  return _split_records_by_sequence_units(records, sequence_to_unit, rng)


def _print_split_audit(split_records):
  print("Split audit:")
  for split_name in ("train", "validation", "test"):
    rows = split_records[split_name]
    source_counts = Counter(record["source"] for record in rows)
    source_msg = " ".join(f"{source}={count}" for source, count in sorted(source_counts.items()))
    labels = [record["labels"][0] for record in rows]
    neg = sum(1 for label in labels if label <= 0.5)
    pos = len(labels) - neg
    print(f"  {split_name}: total={len(rows)} interaction_neg={neg} interaction_pos={pos} sources[{source_msg}]")


def _tokenize_unique_sequences(records, tokenizer) -> Dict[str, torch.Tensor]:
  unique_sequences = _all_unique_sequences(records)
  token_map: Dict[str, torch.Tensor] = {}

  for start in range(0, len(unique_sequences), TOKENIZE_BATCH_SIZE):
    batch = unique_sequences[start:start + TOKENIZE_BATCH_SIZE]
    encoded = tokenizer(
      [_preprocess_sequence(seq) for seq in batch],
      padding=False,
      truncation=True,
      max_length=MAX_LENGTH,
      return_attention_mask=False,
    )
    for sequence, ids in zip(batch, encoded["input_ids"]):
      token_map[sequence] = torch.tensor(ids, dtype=torch.long)

  return token_map


def _build_tokenized_split(records, token_map):
  group1_input_ids = []
  group2_input_ids = []
  raw_labels = []
  normalized_labels = []
  label_mask = []
  lengths = []
  sources = []

  for record in records:
    group1_tensors = [token_map[sequence] for sequence in record["group1_sequences"]]
    group2_tensors = [token_map[sequence] for sequence in record["group2_sequences"]]
    label_tensor = torch.tensor(record["labels"], dtype=torch.float)
    mask_tensor = torch.tensor(record["mask"], dtype=torch.bool)

    group1_input_ids.append(group1_tensors)
    group2_input_ids.append(group2_tensors)
    raw_labels.append(label_tensor)
    normalized_labels.append(label_tensor.clone())
    label_mask.append(mask_tensor)
    lengths.append(sum(len(ids) for ids in group1_tensors + group2_tensors))
    sources.append(record["source"])

  if raw_labels:
    raw_labels_tensor = torch.stack(raw_labels)
    normalized_labels_tensor = torch.stack(normalized_labels)
    label_mask_tensor = torch.stack(label_mask)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
  else:
    raw_labels_tensor = torch.empty((0, 1), dtype=torch.float)
    normalized_labels_tensor = torch.empty((0, 1), dtype=torch.float)
    label_mask_tensor = torch.empty((0, 1), dtype=torch.bool)
    lengths_tensor = torch.empty((0,), dtype=torch.long)

  return {
    "group1_input_ids": group1_input_ids,
    "group2_input_ids": group2_input_ids,
    "raw_labels": raw_labels_tensor,
    "normalized_labels": normalized_labels_tensor,
    "label_mask": label_mask_tensor,
    "lengths": lengths_tensor,
    "sources": sources,
  }


print("Loading group-pair interaction data from DuckDB")
_validate_split_fractions()
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
TOKENIZED_DATA_DIR.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(AGGREGATED_DB_PATH)
try:
  sample_rows = con.execute(
    """
    SELECT source, group1, group2, interaction_label
    FROM samples
    WHERE interaction_label IS NOT NULL
    ORDER BY group1, group2
    """
  ).fetchall()

  if not sample_rows:
    raise ValueError(f"No interaction-labeled samples found in {AGGREGATED_DB_PATH}")

  records = []
  for source, group1, group2, interaction_label in sample_rows:
    records.append(
      {
        "group1": group1,
        "group2": group2,
        "group1_sequences": group1.split(":"),
        "group2_sequences": group2.split(":"),
        "labels": [1.0 if float(interaction_label) > 0.5 else 0.0],
        "mask": [True],
        "source": source,
      }
    )

  split_records, dropped_cross_split = _split_records(records)

  if len(split_records["train"]) == 0 or len(split_records["validation"]) == 0:
    raise ValueError("Train/validation split is empty; adjust dataset size or split fractions.")

  token_map = _tokenize_unique_sequences(records, tokenizer)

  print(
    f"Split strategy={SPLIT_STRATEGY} dropped_cross_split={dropped_cross_split} "
    f"pairs: train={len(split_records['train'])} "
    f"val={len(split_records['validation'])} test={len(split_records['test'])}"
  )
  _print_split_audit(split_records)

  for split_name, rows in split_records.items():
    neg = sum(1 for record in rows if record["labels"][0] <= 0.5)
    pos = len(rows) - neg
    print(f"Task=interaction split={split_name} labels={len(rows)} neg={neg} pos={pos}")

  tokenized_splits = {
    split_name: _build_tokenized_split(rows, token_map)
    for split_name, rows in split_records.items()
  }

  torch.save(
    {
      "task_order": TASK_ORDER,
      "task_metas": TASK_METAS,
      "config": {
        "model_name": MODEL_NAME,
        "db_path": str(AGGREGATED_DB_PATH),
        "split_seed": SPLIT_SEED,
        "split_strategy": SPLIT_STRATEGY,
        "split_assignment": "label_aware_cluster_components" if SPLIT_STRATEGY in {"protein", "cluster"} else "random_rows",
        "cluster_min_seq_id": CLUSTER_MIN_SEQ_ID if SPLIT_STRATEGY == "cluster" else None,
        "cluster_coverage": CLUSTER_COVERAGE if SPLIT_STRATEGY == "cluster" else None,
        "sequence_cluster_tsv_path": str(SEQUENCE_CLUSTER_TSV_PATH) if SPLIT_STRATEGY == "cluster" else None,
        "dropped_cross_split": dropped_cross_split,
        "train_fraction": TRAIN_FRACTION,
        "val_fraction": VAL_FRACTION,
        "test_fraction": TEST_FRACTION,
        "max_length": MAX_LENGTH,
        "pad_token_id": tokenizer.pad_token_id,
        "cache_format": "interaction_group_pair_v1",
      },
      "splits": tokenized_splits,
    },
    TRAIN_CACHE_PATH,
  )
  print(f"Saved interaction tokenized splits -> {TRAIN_CACHE_PATH}")
finally:
  con.close()

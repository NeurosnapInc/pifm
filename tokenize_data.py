"""
Pre-tokenize the aggregated interaction-group DuckDB into multitask train,
validation, and test caches.
"""

import random
import re
from typing import Dict, List

import duckdb
import torch
from transformers import T5Tokenizer

from config import (
  AGGREGATED_DB_PATH,
  MAX_LENGTH,
  MODEL_NAME,
  SPLIT_SEED,
  TEST_FRACTION,
  TOKENIZED_DATA_DIR,
  TRAIN_FRACTION,
  TRAIN_CACHE_PATH,
  VAL_FRACTION,
)


TOKENIZE_BATCH_SIZE = 128

TASK_SPECS = {
  "interaction": {
    "task_name": "interaction",
    "dtype": "bool",
    "head_type": "pair_binary",
    "num_classes": 2,
    "loss": "ce",
  },
  "affinity": {
    "task_name": "affinity",
    "dtype": "float",
    "head_type": "pair_regression",
    "num_classes": None,
    "loss": "mse",
  },
}


def _validate_split_fractions():
  total = TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION
  if abs(total - 1.0) > 1e-8:
    raise ValueError(f"Split fractions must sum to 1.0, got {total}")


def _preprocess_sequence(seq: str) -> str:
  seq = re.sub(r"[UZOB]", "X", seq.upper())
  return "<AA2fold> " + " ".join(seq)


def _label_from_dtype(label: float, dtype: str):
  if dtype == "bool":
    return 1.0 if float(label) > 0.5 else 0.0
  return float(label)


def _empty_label_row(num_tasks: int):
  return [0.0] * num_tasks


def _compute_regression_stats(records, task_order, task_metas):
  means = torch.zeros(len(task_order), dtype=torch.float)
  stds = torch.ones(len(task_order), dtype=torch.float)

  for task_idx, task_name in enumerate(task_order):
    if task_metas[task_name]["dtype"] != "float":
      continue

    values = [
      float(record["labels"][task_idx])
      for record in records
      if record["mask"][task_idx]
    ]
    if not values:
      continue

    tensor = torch.tensor(values, dtype=torch.float)
    means[task_idx] = tensor.mean()
    std = tensor.std(unbiased=False)
    stds[task_idx] = std if std.item() > 0 else 1.0

  return means, stds


def _tokenize_unique_sequences(records, tokenizer) -> Dict[str, torch.Tensor]:
  unique_sequences = sorted(
    {
      sequence
      for record in records
      for group in (record["group1_sequences"], record["group2_sequences"])
      for sequence in group
    }
  )
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


def _build_tokenized_split(records, token_map, task_order, task_metas, means, stds):
  group1_input_ids = []
  group2_input_ids = []
  raw_labels = []
  normalized_labels = []
  label_mask = []
  lengths = []

  for record in records:
    group1_tensors = [token_map[sequence] for sequence in record["group1_sequences"]]
    group2_tensors = [token_map[sequence] for sequence in record["group2_sequences"]]
    label_tensor = torch.tensor(record["labels"], dtype=torch.float)
    mask_tensor = torch.tensor(record["mask"], dtype=torch.bool)
    normalized_tensor = label_tensor.clone()

    for task_idx, task_name in enumerate(task_order):
      if not mask_tensor[task_idx]:
        continue
      if task_metas[task_name]["dtype"] == "float":
        normalized_tensor[task_idx] = (label_tensor[task_idx] - means[task_idx]) / stds[task_idx]

    group1_input_ids.append(group1_tensors)
    group2_input_ids.append(group2_tensors)
    raw_labels.append(label_tensor)
    normalized_labels.append(normalized_tensor)
    label_mask.append(mask_tensor)
    lengths.append(sum(len(ids) for ids in group1_tensors + group2_tensors))

  num_tasks = len(task_order)
  if raw_labels:
    raw_labels_tensor = torch.stack(raw_labels)
    normalized_labels_tensor = torch.stack(normalized_labels)
    label_mask_tensor = torch.stack(label_mask)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
  else:
    raw_labels_tensor = torch.empty((0, num_tasks), dtype=torch.float)
    normalized_labels_tensor = torch.empty((0, num_tasks), dtype=torch.float)
    label_mask_tensor = torch.empty((0, num_tasks), dtype=torch.bool)
    lengths_tensor = torch.empty((0,), dtype=torch.long)

  return {
    "group1_input_ids": group1_input_ids,
    "group2_input_ids": group2_input_ids,
    "raw_labels": raw_labels_tensor,
    "normalized_labels": normalized_labels_tensor,
    "label_mask": label_mask_tensor,
    "lengths": lengths_tensor,
  }


print("Loading group-pair data from DuckDB")
_validate_split_fractions()
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
TOKENIZED_DATA_DIR.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(AGGREGATED_DB_PATH)
try:
  sample_rows = con.execute(
    """
    SELECT source, group1, group2, interaction_label, affinity_pkd
    FROM samples
    ORDER BY group1, group2
    """
  ).fetchall()

  if not sample_rows:
    raise ValueError(f"No samples found in {AGGREGATED_DB_PATH}")

  task_order: List[str] = []
  for task_name in ("interaction", "affinity"):
    if task_name == "interaction" and any(row[3] is not None for row in sample_rows):
      task_order.append(task_name)
    if task_name == "affinity" and any(row[4] is not None for row in sample_rows):
      task_order.append(task_name)

  if not task_order:
    raise ValueError("No supervised labels found. Populate interaction_label and/or affinity_nm before tokenization.")

  task_metas = {task_name: TASK_SPECS[task_name] for task_name in task_order}
  task_to_idx = {task_name: idx for idx, task_name in enumerate(task_order)}

  records = []
  for _, group1, group2, interaction_label, affinity_pkd in sample_rows:
    labels = _empty_label_row(len(task_order))
    mask = [False] * len(task_order)

    if "interaction" in task_to_idx and interaction_label is not None:
      idx = task_to_idx["interaction"]
      labels[idx] = _label_from_dtype(interaction_label, "bool")
      mask[idx] = True

    if "affinity" in task_to_idx and affinity_pkd is not None:
      idx = task_to_idx["affinity"]
      labels[idx] = _label_from_dtype(affinity_pkd, "float")
      mask[idx] = True

    if not any(mask):
      continue

    records.append(
      {
        "group1": group1,
        "group2": group2,
        "group1_sequences": group1.split(":"),
        "group2_sequences": group2.split(":"),
        "labels": labels,
        "mask": mask,
      }
    )

  if not records:
    raise ValueError("No labeled records remained after filtering.")

  indices = list(range(len(records)))
  rng = random.Random(SPLIT_SEED)
  rng.shuffle(indices)

  n_total = len(indices)
  n_train = int(TRAIN_FRACTION * n_total)
  n_val = int(VAL_FRACTION * n_total)
  split_records = {
    "train": [records[i] for i in indices[:n_train]],
    "validation": [records[i] for i in indices[n_train:n_train + n_val]],
    "test": [records[i] for i in indices[n_train + n_val:]],
  }

  if len(split_records["train"]) == 0 or len(split_records["validation"]) == 0:
    raise ValueError("Train/validation split is empty; adjust dataset size or split fractions.")

  train_means, train_stds = _compute_regression_stats(split_records["train"], task_order, task_metas)
  token_map = _tokenize_unique_sequences(records, tokenizer)

  print(
    f"Unique pairs: train={len(split_records['train'])} "
    f"val={len(split_records['validation'])} test={len(split_records['test'])}"
  )
  for task_name in task_order:
    task_idx = task_to_idx[task_name]
    counts = {
      split_name: sum(1 for record in rows if record["mask"][task_idx])
      for split_name, rows in split_records.items()
    }
    if counts["train"] == 0 or counts["validation"] == 0:
      raise ValueError(
        f"Task '{task_name}' has labels(train/val/test)="
        f"{counts['train']}/{counts['validation']}/{counts['test']} after splitting."
      )
    stats_msg = ""
    if task_metas[task_name]["dtype"] == "float":
      stats_msg = f" mean={train_means[task_idx].item():.4f} std={train_stds[task_idx].item():.4f}"
    print(
      f"Task={task_name} dtype={task_metas[task_name]['dtype']} "
      f"labels(train/val/test)={counts['train']}/{counts['validation']}/{counts['test']}{stats_msg}"
    )

  tokenized_splits = {}
  for split_name, rows in split_records.items():
    tokenized_splits[split_name] = _build_tokenized_split(rows, token_map, task_order, task_metas, train_means, train_stds)

  torch.save(
    {
      "task_order": task_order,
      "task_metas": task_metas,
      "config": {
        "model_name": MODEL_NAME,
        "db_path": str(AGGREGATED_DB_PATH),
        "split_seed": SPLIT_SEED,
        "train_fraction": TRAIN_FRACTION,
        "val_fraction": VAL_FRACTION,
        "test_fraction": TEST_FRACTION,
        "max_length": MAX_LENGTH,
        "pad_token_id": tokenizer.pad_token_id,
        "cache_format": "multitask_group_pair_v1",
      },
      "normalization": {
        "train_mean": train_means,
        "train_std": train_stds,
      },
      "splits": tokenized_splits,
    },
    TRAIN_CACHE_PATH,
  )
  print(f"Saved multitask tokenized splits -> {TRAIN_CACHE_PATH}")
finally:
  con.close()

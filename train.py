"""
Multi-task training script for protein-group pair prediction with a frozen
ProstT5 backbone and lightweight adapter fine-tuning.
"""

import math
import random
import warnings
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, mean_absolute_error, mean_squared_error, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5EncoderModel, get_linear_schedule_with_warmup

from calibration import fit_posthoc_calibration
from config import (
  ADAPTER_DIM,
  AFFINITY_NORMALIZATION,
  BATCH_SAMPLER_SEED,
  BATCH_SIZE,
  CLASSIFICATION_HEAD_HIDDEN,
  DROPOUT,
  EPOCHS,
  EVAL_MAX_TOKENS_PER_BATCH,
  LR,
  CLASSIFICATION_SELECTION_METRIC,
  MIN_CLASSIFICATION_VAL_LABELS,
  MIN_REGRESSION_VAL_LABELS,
  MIN_SOURCE_AFFINITY_LABELS,
  MODEL_NAME,
  PATIENCE,
  REGRESSION_HUBER_DELTA,
  REGRESSION_LOSS,
  REGRESSION_SELECTION_METRIC,
  REGRESSION_HEAD_HIDDEN,
  TRAIN_CACHE_PATH,
  TRAIN_MAX_TOKENS_PER_BATCH,
  TRAINING_SEED,
  WARMUP_RATIO,
  WEIGHT_DECAY,
)
from model import (
  MultiTaskBatchSampler,
  MultiTaskGroupPairDataset,
  MultiTaskGroupPairModel,
  collate_multitask_batch,
  output_dim_from_meta,
  unwrap_model,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"
COMPILE_MODEL = DEVICE.type == "cuda"
PIN_MEMORY = DEVICE.type == "cuda"
USE_FUSED_ADAMW = DEVICE.type == "cuda"


def _set_training_seed(seed: int):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def _build_classification_loss(labels: torch.Tensor, mask: torch.Tensor):
  observed = labels[mask].long()
  counts = Counter(int(x) for x in observed.tolist())
  n0, n1 = counts.get(0, 0), counts.get(1, 0)
  total = n0 + n1
  w0 = total / (2.0 * max(1, n0))
  w1 = total / (2.0 * max(1, n1))
  weights = torch.tensor([w0, w1], dtype=torch.float, device=DEVICE)
  return nn.CrossEntropyLoss(weight=weights)


def _metric_from_preds(labels, preds, dtype: str) -> Tuple[str, float, Dict[str, float]]:
  if dtype == "bool":
    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    tn = sum(1 for label, pred in zip(labels, preds) if label == 0 and pred == 0)
    fp = sum(1 for label, pred in zip(labels, preds) if label == 0 and pred == 1)
    specificity = tn / (tn + fp) if (tn + fp) else None
    return "f1", f1, {
      "acc": acc,
      "balanced_accuracy": bal_acc,
      "specificity": specificity,
      "negative_recall": specificity,
      "mcc": matthews_corrcoef(labels, preds),
      "f1": f1,
    }

  mae = mean_absolute_error(labels, preds)
  rmse = math.sqrt(mean_squared_error(labels, preds))
  return "mae", mae, {"mae": mae, "rmse": rmse}


def _safe_auroc(labels, scores):
  if len(set(labels)) < 2:
    return None
  return roc_auc_score(labels, scores)


def _pearson_corr(labels, preds):
  if len(labels) < 2:
    return None

  label_arr = np.asarray(labels, dtype=np.float64)
  pred_arr = np.asarray(preds, dtype=np.float64)
  label_std = label_arr.std()
  pred_std = pred_arr.std()
  if label_std == 0.0 or pred_std == 0.0:
    return None
  return float(np.corrcoef(label_arr, pred_arr)[0, 1])


def _select_validation_metric(task_name, values, report, dtype: str):
  """Return the early-stopping score for a task, or None if validation is underpowered."""
  n_labels = len(values["labels"])

  if dtype == "bool":
    auroc = _safe_auroc(values["labels"], values["scores"])
    report["auroc"] = auroc

    if n_labels < MIN_CLASSIFICATION_VAL_LABELS:
      return None, f"classification labels {n_labels} < {MIN_CLASSIFICATION_VAL_LABELS}"

    if CLASSIFICATION_SELECTION_METRIC == "auroc":
      if auroc is None:
        return None, "AUROC undefined because validation has one class"
      return auroc, "auroc"
    if CLASSIFICATION_SELECTION_METRIC == "balanced_accuracy":
      return report["balanced_accuracy"], "balanced_accuracy"
    raise ValueError(f"Unsupported CLASSIFICATION_SELECTION_METRIC={CLASSIFICATION_SELECTION_METRIC!r}")

  if n_labels < MIN_REGRESSION_VAL_LABELS:
    return None, f"regression labels {n_labels} < {MIN_REGRESSION_VAL_LABELS}"

  normalized_mae = mean_absolute_error(values["normalized_labels"], values["normalized_preds"])
  pearson = _pearson_corr(values["labels"], values["preds"])
  report["normalized_mae"] = normalized_mae
  report["pearson"] = pearson
  if REGRESSION_SELECTION_METRIC == "normalized_mae":
    return -normalized_mae, "negative_normalized_mae"
  if REGRESSION_SELECTION_METRIC == "pearson":
    if pearson is None:
      return None, "Pearson undefined because validation labels/predictions have zero variance"
    return pearson, "pearson"
  raise ValueError(f"Unsupported REGRESSION_SELECTION_METRIC={REGRESSION_SELECTION_METRIC!r}")


def _compute_sample_weights(split_payload, task_order):
  label_mask = split_payload["label_mask"]
  task_counts = label_mask.sum(dim=0).float()
  inv_task_counts = torch.zeros_like(task_counts)
  nonzero = task_counts > 0
  inv_task_counts[nonzero] = 1.0 / task_counts[nonzero]

  sample_weights = []
  for row_mask in label_mask:
    present = row_mask.nonzero(as_tuple=False).view(-1)
    if present.numel() == 0:
      sample_weights.append(1.0)
      continue
    sample_weights.append(float(inv_task_counts[present].mean().item()))

  weights = torch.tensor(sample_weights, dtype=torch.double)
  weights /= weights.sum()

  task_label_counts = {task_name: int(task_counts[idx].item()) for idx, task_name in enumerate(task_order)}
  return weights, task_label_counts


def _compute_source_regression_stats(split_payload, task_order, global_means, global_stds):
  source_stats = {}
  sources = split_payload.get("sources")
  if AFFINITY_NORMALIZATION != "source" or sources is None:
    return source_stats

  label_mask = split_payload["label_mask"]
  raw_labels = split_payload["raw_labels"]
  for task_idx, task_name in enumerate(task_order):
    if task_name != "affinity":
      continue

    values_by_source = {}
    for sample_idx, source in enumerate(sources):
      if not label_mask[sample_idx, task_idx]:
        continue
      values_by_source.setdefault(source, []).append(float(raw_labels[sample_idx, task_idx].item()))

    task_stats = {}
    for source, values in values_by_source.items():
      if len(values) < MIN_SOURCE_AFFINITY_LABELS:
        continue
      tensor = torch.tensor(values, dtype=torch.float)
      std = tensor.std(unbiased=False)
      task_stats[source] = {
        "mean": float(tensor.mean().item()),
        "std": float(std.item() if std.item() > 0 else 1.0),
        "n": len(values),
      }

    task_stats["__global__"] = {
      "mean": float(global_means[task_idx].item()),
      "std": float(global_stds[task_idx].item()),
      "n": int(label_mask[:, task_idx].sum().item()),
    }
    source_stats[task_name] = task_stats

  return source_stats


def _source_norm_tensors(task_name, sources, source_regression_stats, device):
  task_stats = source_regression_stats.get(task_name, {})
  fallback = task_stats.get("__global__", {"mean": 0.0, "std": 1.0})
  means = []
  stds = []
  for source in sources:
    stats = task_stats.get(source, fallback)
    means.append(stats["mean"])
    stds.append(stats["std"])
  return (
    torch.tensor(means, dtype=torch.float, device=device),
    torch.tensor(stds, dtype=torch.float, device=device),
  )


def _normalize_regression_targets(task_name, raw_values, source_values, source_regression_stats):
  means, stds = _source_norm_tensors(task_name, source_values, source_regression_stats, raw_values.device)
  return (raw_values - means) / stds


def _denormalize_regression_preds(task_name, normalized_preds, source_values, source_regression_stats):
  means, stds = _source_norm_tensors(task_name, source_values, source_regression_stats, normalized_preds.device)
  return normalized_preds * stds + means


def _regression_loss(preds, targets):
  if REGRESSION_LOSS == "mse":
    return F.mse_loss(preds, targets)
  if REGRESSION_LOSS == "huber":
    return F.huber_loss(preds, targets, delta=REGRESSION_HUBER_DELTA)
  raise ValueError(f"Unsupported REGRESSION_LOSS={REGRESSION_LOSS!r}")


def _compute_multitask_loss(outputs, raw_labels, normalized_labels, label_mask, sources, task_order, task_metas, criteria, source_regression_stats):
  task_losses = []

  for task_idx, task_name in enumerate(task_order):
    mask = label_mask[:, task_idx]
    if not mask.any():
      continue

    preds = outputs[task_name][mask]
    meta = task_metas[task_name]
    if meta["dtype"] == "float":
      if task_name in source_regression_stats:
        masked_sources = [source for source, keep in zip(sources, mask.detach().cpu().tolist()) if keep]
        targets = _normalize_regression_targets(
          task_name,
          raw_labels[mask, task_idx],
          masked_sources,
          source_regression_stats,
        )
      else:
        targets = normalized_labels[mask, task_idx]
      task_loss = _regression_loss(preds.squeeze(-1), targets)
    else:
      targets = raw_labels[mask, task_idx].long()
      task_loss = criteria[task_name](preds, targets)
    task_losses.append(task_loss)

  if not task_losses:
    raise ValueError("Encountered a batch with no observed task labels.")

  return torch.stack(task_losses).mean()


# Suppress a repetitive torch.compile/inductor warning that spams tqdm output during training.
warnings.filterwarnings("ignore", message="Online softmax is disabled.*", category=UserWarning)

print("Loading multitask tokenized cache")
if not TRAIN_CACHE_PATH.exists():
  raise FileNotFoundError(f"Missing tokenized cache at {TRAIN_CACHE_PATH}. Run tokenize_data.py first.")

_set_training_seed(TRAINING_SEED)
payload = torch.load(TRAIN_CACHE_PATH, map_location="cpu")
task_order = payload["task_order"]
task_metas = payload["task_metas"]
train_split = payload["splits"]["train"]
val_split = payload["splits"]["validation"]
pad_token_id = payload["config"]["pad_token_id"]
normalization = payload["normalization"]
regression_means = normalization["train_mean"]
regression_stds = normalization["train_std"]
source_regression_stats = _compute_source_regression_stats(train_split, task_order, regression_means, regression_stds)

train_ds = MultiTaskGroupPairDataset(train_split)
val_ds = MultiTaskGroupPairDataset(val_split)
train_sample_weights, train_label_counts = _compute_sample_weights(train_split, task_order)

print(f"Loaded cache from {TRAIN_CACHE_PATH}")
print(f"Pairs: train={len(train_ds)} val={len(val_ds)}")
for task_idx, task_name in enumerate(task_order):
  meta = task_metas[task_name]
  train_count = train_label_counts[task_name]
  val_count = int(val_split["label_mask"][:, task_idx].sum().item())
  stats_msg = ""
  if meta["dtype"] == "float":
    stats_msg = f" mean={regression_means[task_idx].item():.4f} std={regression_stds[task_idx].item():.4f}"
  print(f"Task={task_name} dtype={meta['dtype']} labels(train/val)={train_count}/{val_count}{stats_msg}")

base_model = T5EncoderModel.from_pretrained(MODEL_NAME).to(DEVICE)
if DEVICE.type == "cuda":
  base_model.bfloat16()

train_loader = DataLoader(
  train_ds,
  batch_sampler=MultiTaskBatchSampler(
    train_ds,
    BATCH_SIZE,
    shuffle=True,
    seed=BATCH_SAMPLER_SEED,
    sample_weights=train_sample_weights,
    max_tokens_per_batch=TRAIN_MAX_TOKENS_PER_BATCH,
  ),
  collate_fn=lambda batch: collate_multitask_batch(batch, pad_token_id, include_sources=True),
  pin_memory=PIN_MEMORY,
)
val_loader = DataLoader(
  val_ds,
  batch_sampler=MultiTaskBatchSampler(
    val_ds,
    BATCH_SIZE,
    shuffle=False,
    seed=BATCH_SAMPLER_SEED,
    max_tokens_per_batch=EVAL_MAX_TOKENS_PER_BATCH,
  ),
  collate_fn=lambda batch: collate_multitask_batch(batch, pad_token_id, include_sources=True),
  pin_memory=PIN_MEMORY,
)

print("Initializing model")
task_output_dims = {}
criteria = {}
for task_idx, task_name in enumerate(task_order):
  meta = task_metas[task_name]
  train_mask = train_split["label_mask"][:, task_idx]
  train_labels = train_split["raw_labels"][:, task_idx]
  task_output_dims[task_name] = output_dim_from_meta(meta, train_labels, train_mask)
  if meta["dtype"] == "bool":
    criteria[task_name] = _build_classification_loss(train_labels, train_mask)

embed_dim = base_model.config.d_model
model = MultiTaskGroupPairModel(
  base_model,
  task_order,
  task_output_dims,
  embed_dim=embed_dim,
  task_metas=task_metas,
  adapter_dim=ADAPTER_DIM,
  dropout=DROPOUT,
  classification_head_hidden=CLASSIFICATION_HEAD_HIDDEN,
  regression_head_hidden=REGRESSION_HEAD_HIDDEN,
).to(DEVICE)

if COMPILE_MODEL and hasattr(torch, "compile"):
  print("Compiling model")
  try:
    model = torch.compile(model)
  except Exception as exc:
    print(f"torch.compile unavailable, continuing without compile: {exc}")

model_ref = unwrap_model(model)
optimizer = torch.optim.AdamW(
  [
    {"params": model_ref.adapter.parameters()},
    {"params": model_ref.residue_pool.parameters()},
    {"params": model_ref.group_pool.parameters()},
    {"params": model_ref.pair_mlp.parameters()},
    {"params": model_ref.heads.parameters()},
  ],
  lr=LR,
  weight_decay=WEIGHT_DECAY,
  fused=USE_FUSED_ADAMW,
)
trainable_params = (
  list(model_ref.adapter.parameters())
  + list(model_ref.residue_pool.parameters())
  + list(model_ref.group_pool.parameters())
  + list(model_ref.pair_mlp.parameters())
  + list(model_ref.heads.parameters())
)

num_training_steps = len(train_loader) * EPOCHS
num_warmup_steps = int(WARMUP_RATIO * num_training_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

best_metric = -float("inf")
stale = 0
best_state = None

regression_means_device = regression_means.to(DEVICE)
regression_stds_device = regression_stds.to(DEVICE)

for epoch in range(EPOCHS):
  model.train()
  total_loss = 0.0

  for input_ids, attn_mask, chain_to_sample, chain_to_group, raw_labels, normalized_labels, label_mask, sources in tqdm(
    train_loader,
    desc=f"Epoch {epoch + 1}/{EPOCHS}",
  ):
    input_ids = input_ids.to(DEVICE, non_blocking=PIN_MEMORY)
    attn_mask = attn_mask.to(DEVICE, non_blocking=PIN_MEMORY)
    chain_to_sample = chain_to_sample.to(DEVICE, non_blocking=PIN_MEMORY)
    chain_to_group = chain_to_group.to(DEVICE, non_blocking=PIN_MEMORY)
    raw_labels = raw_labels.to(DEVICE, non_blocking=PIN_MEMORY)
    normalized_labels = normalized_labels.to(DEVICE, non_blocking=PIN_MEMORY)
    label_mask = label_mask.to(DEVICE, non_blocking=PIN_MEMORY)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=AMP_ENABLED):
      outputs = model(input_ids, attn_mask, chain_to_sample, chain_to_group, raw_labels.shape[0])
      loss = _compute_multitask_loss(
        outputs,
        raw_labels,
        normalized_labels,
        label_mask,
        sources,
        task_order,
        task_metas,
        criteria,
        source_regression_stats,
      )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
    optimizer.step()
    scheduler.step()
    total_loss += loss.item()

  model.eval()
  val_predictions = {
    task_name: {
      "preds": [],
      "labels": [],
      "scores": [],
      "normalized_preds": [],
      "normalized_labels": [],
    }
    for task_name in task_order
  }

  with torch.no_grad():
    for input_ids, attn_mask, chain_to_sample, chain_to_group, raw_labels, normalized_labels, label_mask, sources in val_loader:
      input_ids = input_ids.to(DEVICE, non_blocking=PIN_MEMORY)
      attn_mask = attn_mask.to(DEVICE, non_blocking=PIN_MEMORY)
      chain_to_sample = chain_to_sample.to(DEVICE, non_blocking=PIN_MEMORY)
      chain_to_group = chain_to_group.to(DEVICE, non_blocking=PIN_MEMORY)
      raw_labels = raw_labels.to(DEVICE, non_blocking=PIN_MEMORY)
      normalized_labels = normalized_labels.to(DEVICE, non_blocking=PIN_MEMORY)
      label_mask = label_mask.to(DEVICE, non_blocking=PIN_MEMORY)

      with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=AMP_ENABLED):
        outputs = model(input_ids, attn_mask, chain_to_sample, chain_to_group, raw_labels.shape[0])

      for task_idx, task_name in enumerate(task_order):
        mask = label_mask[:, task_idx]
        if not mask.any():
          continue

        meta = task_metas[task_name]
        if meta["dtype"] == "float":
          preds_norm = outputs[task_name][mask].squeeze(-1).float()
          masked_sources = [source for source, keep in zip(sources, mask.detach().cpu().tolist()) if keep]
          if task_name in source_regression_stats:
            labels_norm = _normalize_regression_targets(
              task_name,
              raw_labels[mask, task_idx].float(),
              masked_sources,
              source_regression_stats,
            )
            preds = _denormalize_regression_preds(task_name, preds_norm, masked_sources, source_regression_stats)
          else:
            labels_norm = normalized_labels[mask, task_idx].float()
            preds = preds_norm * regression_stds_device[task_idx] + regression_means_device[task_idx]
          labels = raw_labels[mask, task_idx].float()
          val_predictions[task_name]["preds"].extend(preds.cpu().tolist())
          val_predictions[task_name]["labels"].extend(labels.cpu().tolist())
          val_predictions[task_name]["normalized_preds"].extend(preds_norm.cpu().tolist())
          val_predictions[task_name]["normalized_labels"].extend(labels_norm.cpu().tolist())
        else:
          logits = outputs[task_name][mask].float()
          probs = torch.softmax(logits, dim=1)
          preds = probs.argmax(dim=1)
          labels = raw_labels[mask, task_idx].long()
          val_predictions[task_name]["preds"].extend(preds.cpu().tolist())
          val_predictions[task_name]["labels"].extend(labels.cpu().tolist())
          val_predictions[task_name]["scores"].extend(probs[:, 1].cpu().tolist())

  task_reports = {}
  aggregate_score = 0.0
  scored_tasks = 0
  skipped_selection_tasks = {}
  for task_name, values in val_predictions.items():
    if not values["labels"]:
      continue

    metric_name, metric_value, report = _metric_from_preds(values["labels"], values["preds"], task_metas[task_name]["dtype"])
    selection_metric, selection_metric_name = _select_validation_metric(
      task_name,
      values,
      report,
      task_metas[task_name]["dtype"],
    )

    task_reports[task_name] = {
      "metric_name": metric_name,
      "metric_value": metric_value,
      "selection_metric": selection_metric,
      "selection_metric_name": selection_metric_name,
      "report": report,
    }
    if selection_metric is None:
      skipped_selection_tasks[task_name] = selection_metric_name
      continue

    aggregate_score += selection_metric
    scored_tasks += 1

  # Underpowered validation tasks are still reported, but must not drive early stopping.
  if scored_tasks == 0:
    skipped_msg = ", ".join(f"{task} ({reason})" for task, reason in skipped_selection_tasks.items())
    raise ValueError(f"No validation task has enough labels for checkpoint selection. Skipped: {skipped_msg}")

  aggregate_score /= scored_tasks
  summary_parts = []
  for task_name in sorted(task_reports):
    report = task_reports[task_name]["report"]
    if task_metas[task_name]["dtype"] == "bool":
      auroc = report.get("auroc")
      auroc_msg = "nan" if auroc is None else f"{auroc:.4f}"
      specificity = report["specificity"]
      specificity_msg = "nan" if specificity is None else f"{specificity:.4f}"
      summary_parts.append(
        f"{task_name}:ACC={report['acc']:.4f} BAL_ACC={report['balanced_accuracy']:.4f} "
        f"SPEC={specificity_msg} MCC={report['mcc']:.4f} "
        f"F1={report['f1']:.4f} AUROC={auroc_msg}"
      )
    else:
      pearson = report.get("pearson")
      pearson_msg = "nan" if pearson is None else f"{pearson:.4f}"
      summary_parts.append(
        f"{task_name}:MAE={report['mae']:.4f} RMSE={report['rmse']:.4f} "
        f"PEARSON={pearson_msg}"
      )

  selection_parts = [
    f"{task}={details['selection_metric_name']}:{details['selection_metric']:.4f}"
    for task, details in sorted(task_reports.items())
    if details["selection_metric"] is not None
  ]
  if skipped_selection_tasks:
    skipped_msg = "; ".join(f"{task}:{reason}" for task, reason in sorted(skipped_selection_tasks.items()))
    selection_parts.append(f"skipped={skipped_msg}")
  print(
    f"Train Loss: {total_loss / len(train_loader):.4f} | Val "
    + " ".join(summary_parts)
    + f" | Select aggregate={aggregate_score:.4f} "
    + " ".join(selection_parts)
  )

  if aggregate_score > best_metric:
    best_metric = aggregate_score
    stale = 0
    model_ref = unwrap_model(model)
    best_state = {
      "adapter": {k: v.cpu() for k, v in model_ref.adapter.state_dict().items()},
      "residue_pool": {k: v.cpu() for k, v in model_ref.residue_pool.state_dict().items()},
      "group_pool": {k: v.cpu() for k, v in model_ref.group_pool.state_dict().items()},
      "pair_mlp": {k: v.cpu() for k, v in model_ref.pair_mlp.state_dict().items()},
      "heads": {task_name: {k: v.cpu() for k, v in head.state_dict().items()} for task_name, head in model_ref.heads.items()},
      "aggregate_score": aggregate_score,
      "task_reports": task_reports,
      "validation_predictions": val_predictions,
    }
  else:
    stale += 1
    if stale >= PATIENCE:
      print("Early stopping.")
      break

if best_state is not None:
  model_ref = unwrap_model(model)
  model_ref.adapter.load_state_dict(best_state["adapter"])
  model_ref.residue_pool.load_state_dict(best_state["residue_pool"])
  model_ref.group_pool.load_state_dict(best_state["group_pool"])
  model_ref.pair_mlp.load_state_dict(best_state["pair_mlp"])
  for task_name, state_dict in best_state["heads"].items():
    model_ref.heads[task_name].load_state_dict(state_dict)
  calibration = fit_posthoc_calibration(best_state["validation_predictions"], task_metas, calibration_split="validation")
else:
  calibration = None

Path("checkpoints").mkdir(parents=True, exist_ok=True)
model_ref = unwrap_model(model)
run_date = date.today().isoformat()
out_path = Path(f"./checkpoints/prostt5_group_pair_adapter_best_{run_date}_seed_{TRAINING_SEED}.pt")
torch.save(
  {
    "adapter_state_dict": model_ref.adapter.state_dict(),
    "residue_pool_state_dict": model_ref.residue_pool.state_dict(),
    "group_pool_state_dict": model_ref.group_pool.state_dict(),
    "pair_mlp_state_dict": model_ref.pair_mlp.state_dict(),
    "head_state_dicts": {task_name: head.state_dict() for task_name, head in model_ref.heads.items()},
    "config": {
      "embed_dim": embed_dim,
      "adapter_dim": ADAPTER_DIM,
      "dropout": DROPOUT,
      "classification_head_hidden": CLASSIFICATION_HEAD_HIDDEN,
      "regression_head_hidden": REGRESSION_HEAD_HIDDEN,
      "model_name": MODEL_NAME,
      "tokenized_data_path": str(TRAIN_CACHE_PATH),
      "task_names": task_order,
      "task_metas": task_metas,
      "task_output_dims": task_output_dims,
      "regression_mean": regression_means,
      "regression_std": regression_stds,
      "regression_loss": REGRESSION_LOSS,
      "regression_huber_delta": REGRESSION_HUBER_DELTA,
      "affinity_normalization": AFFINITY_NORMALIZATION,
      "min_source_affinity_labels": MIN_SOURCE_AFFINITY_LABELS,
      "source_regression_stats": source_regression_stats,
      "calibration": calibration,
      "training_seed": TRAINING_SEED,
      "run_date": run_date,
      "best_aggregate_score": best_state["aggregate_score"] if best_state else None,
      "best_task_reports": best_state["task_reports"] if best_state else None,
      "classification_selection_metric": CLASSIFICATION_SELECTION_METRIC,
      "regression_selection_metric": REGRESSION_SELECTION_METRIC,
      "min_classification_val_labels": MIN_CLASSIFICATION_VAL_LABELS,
      "min_regression_val_labels": MIN_REGRESSION_VAL_LABELS,
    },
  },
  out_path,
)
print(f"Saved best adapter+heads -> {out_path}")

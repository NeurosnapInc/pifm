"""
Train the interaction classifier with a frozen ProstT5 backbone and lightweight heads.
"""

import random
import warnings
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5EncoderModel, get_linear_schedule_with_warmup

from calibration import fit_posthoc_calibration
from config import (
  ADAPTER_DIM,
  BACKBONE_EMBEDDING_CACHE_PATH,
  BATCH_SAMPLER_SEED,
  BATCH_SIZE,
  CLASSIFICATION_HEAD_HIDDEN,
  CLASSIFICATION_SELECTION_METRIC,
  DROPOUT,
  EPOCHS,
  EVAL_MAX_TOKENS_PER_BATCH,
  FOCAL_GAMMA,
  INTERACTION_LOSS,
  INTERACTION_POS_NEG_RATIO,
  LR,
  MIN_CLASSIFICATION_VAL_LABELS,
  MODEL_NAME,
  PATIENCE,
  SOURCE_BALANCED_SAMPLING,
  TRAIN_CACHE_PATH,
  TRAIN_MAX_TOKENS_PER_BATCH,
  TRAINING_SEED,
  USE_BACKBONE_EMBEDDING_CACHE,
  WARMUP_RATIO,
  WEIGHT_DECAY,
)
from model import (
  MultiTaskBatchSampler,
  MultiTaskGroupPairDataset,
  MultiTaskGroupPairModel,
  collate_multitask_batch,
  load_backbone_embedding_cache,
  output_dim_from_meta,
  unwrap_model,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"
COMPILE_MODEL = DEVICE.type == "cuda"
PIN_MEMORY = DEVICE.type == "cuda"
USE_FUSED_ADAMW = DEVICE.type == "cuda"
TASK_NAME = "interaction"


class FocalCrossEntropyLoss(nn.Module):
  def __init__(self, weight=None, gamma=2.0):
    super().__init__()
    self.register_buffer("weight", weight if weight is not None else None)
    self.gamma = gamma

  def forward(self, logits, targets):
    ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
    pt = torch.exp(-ce)
    return ((1.0 - pt).pow(self.gamma) * ce).mean()


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
  weights = torch.tensor(
    [
      total / (2.0 * max(1, n0)),
      total / (2.0 * max(1, n1)),
    ],
    dtype=torch.float,
    device=DEVICE,
  )
  if INTERACTION_LOSS == "ce":
    return nn.CrossEntropyLoss(weight=weights)
  if INTERACTION_LOSS == "focal":
    return FocalCrossEntropyLoss(weight=weights, gamma=FOCAL_GAMMA)
  raise ValueError(f"Unsupported INTERACTION_LOSS={INTERACTION_LOSS!r}")


def _safe_auroc(labels, scores):
  if len(set(labels)) < 2:
    return None
  return roc_auc_score(labels, scores)


def _classification_report(labels, preds, scores):
  tn = sum(1 for label, pred in zip(labels, preds) if label == 0 and pred == 0)
  fp = sum(1 for label, pred in zip(labels, preds) if label == 0 and pred == 1)
  fn = sum(1 for label, pred in zip(labels, preds) if label == 1 and pred == 0)
  tp = sum(1 for label, pred in zip(labels, preds) if label == 1 and pred == 1)
  specificity = tn / (tn + fp) if (tn + fp) else None
  return {
    "acc": accuracy_score(labels, preds),
    "balanced_accuracy": balanced_accuracy_score(labels, preds),
    "specificity": specificity,
    "negative_recall": specificity,
    "mcc": matthews_corrcoef(labels, preds) if len(set(labels)) >= 2 and len(set(preds)) >= 2 else 0.0,
    "f1": f1_score(labels, preds, zero_division=0),
    "auroc": _safe_auroc(labels, scores),
    "tn": tn,
    "fp": fp,
    "fn": fn,
    "tp": tp,
  }


def _select_validation_metric(values, report):
  n_labels = len(values["labels"])
  if n_labels < MIN_CLASSIFICATION_VAL_LABELS:
    return None, f"classification labels {n_labels} < {MIN_CLASSIFICATION_VAL_LABELS}"

  if CLASSIFICATION_SELECTION_METRIC == "auroc":
    if report["auroc"] is None:
      return None, "AUROC undefined because validation has one class"
    return report["auroc"], "auroc"
  if CLASSIFICATION_SELECTION_METRIC == "balanced_accuracy":
    return report["balanced_accuracy"], "balanced_accuracy"
  raise ValueError(f"Unsupported CLASSIFICATION_SELECTION_METRIC={CLASSIFICATION_SELECTION_METRIC!r}")


def _compute_sample_weights(split_payload, task_order):
  task_to_idx = {task_name: idx for idx, task_name in enumerate(task_order)}
  interaction_idx = task_to_idx[TASK_NAME]
  label_mask = split_payload["label_mask"]
  labels = split_payload["raw_labels"][:, interaction_idx]
  mask = label_mask[:, interaction_idx]
  sources = split_payload.get("sources", ["unknown"] * len(label_mask))
  source_counts = Counter(sources)
  n_sources = max(1, len(source_counts))

  n_pos = int(((labels > 0.5) & mask).sum().item())
  n_neg = int(((labels <= 0.5) & mask).sum().item())
  class_weights = {}
  if n_pos > 0 and n_neg > 0:
    # Cap expected sampled positives so negatives are seen often enough each epoch.
    pos_mass = INTERACTION_POS_NEG_RATIO / (INTERACTION_POS_NEG_RATIO + 1.0)
    neg_mass = 1.0 / (INTERACTION_POS_NEG_RATIO + 1.0)
    class_weights = {
      1: pos_mass / n_pos,
      0: neg_mass / n_neg,
    }

  sample_weights = []
  for sample_idx in range(len(label_mask)):
    if not bool(mask[sample_idx]):
      sample_weights.append(1.0)
      continue

    label = 1 if float(labels[sample_idx].item()) > 0.5 else 0
    weight = class_weights.get(label, 1.0)
    if SOURCE_BALANCED_SAMPLING:
      weight *= len(label_mask) / (n_sources * source_counts[sources[sample_idx]])
    sample_weights.append(weight)

  weights = torch.tensor(sample_weights, dtype=torch.double)
  weights /= weights.sum()
  return weights, {TASK_NAME: int(mask.sum().item())}


def _load_embedding_cache():
  if USE_BACKBONE_EMBEDDING_CACHE and BACKBONE_EMBEDDING_CACHE_PATH.exists():
    embedding_cache, payload = load_backbone_embedding_cache(
      BACKBONE_EMBEDDING_CACHE_PATH,
      expected_model_name=MODEL_NAME,
      expected_tokenized_cache_path=TRAIN_CACHE_PATH,
    )
    print(
      f"Using frozen backbone embedding cache from {BACKBONE_EMBEDDING_CACHE_PATH} "
      f"sequences={payload.get('num_sequences', len(embedding_cache))}"
    )
    return embedding_cache

  if USE_BACKBONE_EMBEDDING_CACHE:
    print(
      f"Backbone embedding cache not found at {BACKBONE_EMBEDDING_CACHE_PATH}; "
      "falling back to on-the-fly ProstT5 encoding."
    )
  return None


def _forward_model(model, batch):
  input_ids, input_embeddings, attn_mask, chain_to_sample, chain_to_group, raw_labels, normalized_labels, label_mask, sources = batch
  if input_ids is not None:
    input_ids = input_ids.to(DEVICE, non_blocking=PIN_MEMORY)
  if input_embeddings is not None:
    input_embeddings = input_embeddings.to(DEVICE, non_blocking=PIN_MEMORY)
  attn_mask = attn_mask.to(DEVICE, non_blocking=PIN_MEMORY)
  chain_to_sample = chain_to_sample.to(DEVICE, non_blocking=PIN_MEMORY)
  chain_to_group = chain_to_group.to(DEVICE, non_blocking=PIN_MEMORY)
  raw_labels = raw_labels.to(DEVICE, non_blocking=PIN_MEMORY)
  label_mask = label_mask.to(DEVICE, non_blocking=PIN_MEMORY)

  outputs = model(
    input_ids,
    attn_mask,
    chain_to_sample,
    chain_to_group,
    raw_labels.shape[0],
    precomputed_embeddings=input_embeddings,
  )
  return outputs, raw_labels, label_mask


def _collect_predictions(model, loader, task_idx):
  predictions = {"labels": [], "preds": [], "scores": []}

  with torch.no_grad():
    for batch in loader:
      with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=AMP_ENABLED):
        outputs, raw_labels, label_mask = _forward_model(model, batch)

      mask = label_mask[:, task_idx]
      if not mask.any():
        continue
      logits = outputs[TASK_NAME][mask].float()
      probs = torch.softmax(logits, dim=1)
      preds = probs.argmax(dim=1)
      labels = raw_labels[mask, task_idx].long()
      predictions["preds"].extend(preds.cpu().tolist())
      predictions["labels"].extend(labels.cpu().tolist())
      predictions["scores"].extend(probs[:, 1].cpu().tolist())

  return predictions


warnings.filterwarnings("ignore", message="Online softmax is disabled.*", category=UserWarning)

print("Loading interaction tokenized cache")
if not TRAIN_CACHE_PATH.exists():
  raise FileNotFoundError(f"Missing tokenized cache at {TRAIN_CACHE_PATH}. Run tokenize_data.py first.")

_set_training_seed(TRAINING_SEED)
payload = torch.load(TRAIN_CACHE_PATH, map_location="cpu")
task_order = payload["task_order"]
task_metas = payload["task_metas"]
if task_order != [TASK_NAME]:
  raise ValueError(f"Expected interaction-only cache, found task_order={task_order!r}. Re-run tokenize_data.py.")

train_split = payload["splits"]["train"]
val_split = payload["splits"]["validation"]
pad_token_id = payload["config"]["pad_token_id"]
task_idx = task_order.index(TASK_NAME)

embedding_cache = _load_embedding_cache()
train_ds = MultiTaskGroupPairDataset(train_split, embedding_cache=embedding_cache)
val_ds = MultiTaskGroupPairDataset(val_split, embedding_cache=embedding_cache)
train_sample_weights, train_label_counts = _compute_sample_weights(train_split, task_order)

print(f"Loaded cache from {TRAIN_CACHE_PATH}")
print(f"Pairs: train={len(train_ds)} val={len(val_ds)}")
print(
  f"Task=interaction dtype=bool labels(train/val)="
  f"{train_label_counts[TASK_NAME]}/{int(val_split['label_mask'][:, task_idx].sum().item())}"
)

if embedding_cache is None:
  base_model = T5EncoderModel.from_pretrained(MODEL_NAME).to(DEVICE)
  if DEVICE.type == "cuda":
    base_model.bfloat16()
  embed_dim = base_model.config.d_model
else:
  base_model = None
  embed_dim = next(iter(embedding_cache.values())).shape[-1]

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
train_mask = train_split["label_mask"][:, task_idx]
train_labels = train_split["raw_labels"][:, task_idx]
task_output_dims = {
  TASK_NAME: output_dim_from_meta(task_metas[TASK_NAME], train_labels, train_mask),
}
criterion = _build_classification_loss(train_labels, train_mask)

model = MultiTaskGroupPairModel(
  base_model,
  task_order,
  task_output_dims,
  embed_dim=embed_dim,
  task_metas=task_metas,
  adapter_dim=ADAPTER_DIM,
  dropout=DROPOUT,
  classification_head_hidden=CLASSIFICATION_HEAD_HIDDEN,
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

for epoch in range(EPOCHS):
  model.train()
  total_loss = 0.0

  for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=AMP_ENABLED):
      outputs, raw_labels, label_mask = _forward_model(model, batch)
      mask = label_mask[:, task_idx]
      logits = outputs[TASK_NAME][mask]
      targets = raw_labels[mask, task_idx].long()
      loss = criterion(logits, targets)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
    optimizer.step()
    scheduler.step()
    total_loss += loss.item()

  model.eval()
  val_predictions = {TASK_NAME: _collect_predictions(model, val_loader, task_idx)}
  report = _classification_report(
    val_predictions[TASK_NAME]["labels"],
    val_predictions[TASK_NAME]["preds"],
    val_predictions[TASK_NAME]["scores"],
  )
  selection_metric, selection_metric_name = _select_validation_metric(val_predictions[TASK_NAME], report)
  if selection_metric is None:
    raise ValueError(f"No validation task has enough labels for checkpoint selection: {selection_metric_name}")

  auroc_msg = "nan" if report["auroc"] is None else f"{report['auroc']:.4f}"
  specificity_msg = "nan" if report["specificity"] is None else f"{report['specificity']:.4f}"
  print(
    f"Train Loss: {total_loss / len(train_loader):.4f} | Val "
    f"interaction:ACC={report['acc']:.4f} BAL_ACC={report['balanced_accuracy']:.4f} "
    f"SPEC={specificity_msg} MCC={report['mcc']:.4f} F1={report['f1']:.4f} AUROC={auroc_msg} "
    f"| Select {selection_metric_name}={selection_metric:.4f}"
  )

  if selection_metric > best_metric:
    best_metric = selection_metric
    stale = 0
    model_ref = unwrap_model(model)
    best_state = {
      "adapter": {k: v.cpu() for k, v in model_ref.adapter.state_dict().items()},
      "residue_pool": {k: v.cpu() for k, v in model_ref.residue_pool.state_dict().items()},
      "group_pool": {k: v.cpu() for k, v in model_ref.group_pool.state_dict().items()},
      "pair_mlp": {k: v.cpu() for k, v in model_ref.pair_mlp.state_dict().items()},
      "heads": {task_name: {k: v.cpu() for k, v in head.state_dict().items()} for task_name, head in model_ref.heads.items()},
      "selection_metric": selection_metric,
      "task_report": report,
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
      "model_name": MODEL_NAME,
      "tokenized_data_path": str(TRAIN_CACHE_PATH),
      "task_names": task_order,
      "task_metas": task_metas,
      "task_output_dims": task_output_dims,
      "interaction_loss": INTERACTION_LOSS,
      "focal_gamma": FOCAL_GAMMA,
      "interaction_pos_neg_ratio": INTERACTION_POS_NEG_RATIO,
      "source_balanced_sampling": SOURCE_BALANCED_SAMPLING,
      "used_backbone_embedding_cache": embedding_cache is not None,
      "backbone_embedding_cache_path": str(BACKBONE_EMBEDDING_CACHE_PATH) if embedding_cache is not None else None,
      "calibration": calibration,
      "training_seed": TRAINING_SEED,
      "run_date": run_date,
      "best_selection_metric": best_state["selection_metric"] if best_state else None,
      "best_task_report": best_state["task_report"] if best_state else None,
      "classification_selection_metric": CLASSIFICATION_SELECTION_METRIC,
      "min_classification_val_labels": MIN_CLASSIFICATION_VAL_LABELS,
    },
  },
  out_path,
)
print(f"Saved best adapter+head -> {out_path}")

"""
Validate a trained multitask protein-group pair checkpoint on a cached split.
"""

import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5EncoderModel

from calibration import (
  apply_posthoc_calibration,
  classification_report,
  format_posthoc_classification_rows,
  format_posthoc_regression_rows,
  regression_report,
)
from config import (
  ADAPTER_DIM,
  BATCH_SIZE,
  CLASSIFICATION_HEAD_HIDDEN,
  DROPOUT,
  EVAL_MAX_TOKENS_PER_BATCH,
  MODEL_NAME,
  REGRESSION_HEAD_HIDDEN,
  TOKENIZED_DATA_DIR,
)
from model import (
  MultiTaskBatchSampler,
  MultiTaskGroupPairDataset,
  MultiTaskGroupPairModel,
  collate_multitask_batch,
  output_dim_from_meta,
)


DEFAULT_CACHE_PATH = TOKENIZED_DATA_DIR / "multitask_group_pair_prostt5_tokens.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"
PIN_MEMORY = DEVICE.type == "cuda"


def _format_float(value):
  if value is None:
    return "-"
  return f"{value:.4f}"


def _format_table(title, columns, rows):
  if not rows:
    return f"{title}\n(no rows)\n"

  widths = [len(col) for col in columns]
  for row in rows:
    for idx, cell in enumerate(row):
      widths[idx] = max(widths[idx], len(str(cell)))

  def render_row(row):
    return "  ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row))

  divider = "  ".join("-" * width for width in widths)
  lines = [title, render_row(columns), divider]
  lines.extend(render_row(row) for row in rows)
  return "\n".join(lines) + "\n"


def _empty_prediction_bucket():
  return {
    "labels": [],
    "preds": [],
    "scores": [],
  }


def _append_task_predictions(target, task_name, labels, preds, scores, sources=None):
  target[task_name]["labels"].extend(labels)
  target[task_name]["preds"].extend(preds)
  target[task_name]["scores"].extend(scores)

  if sources is None:
    return

  for source, label, pred, score in zip(sources, labels, preds, scores):
    bucket = target[task_name]["by_source"][source]
    bucket["labels"].append(label)
    bucket["preds"].append(pred)
    bucket["scores"].append(score)


def _append_regression_predictions(target, task_name, labels, preds, sources=None):
  target[task_name]["labels"].extend(labels)
  target[task_name]["preds"].extend(preds)

  if sources is None:
    return

  for source, label, pred in zip(sources, labels, preds):
    bucket = target[task_name]["by_source"][source]
    bucket["labels"].append(label)
    bucket["preds"].append(pred)


def _classification_row(task_name, n, report, prefix=None):
  row = [
    task_name,
    n,
    _format_float(report["acc"]),
    _format_float(report["balanced_acc"]),
    _format_float(report["precision"]),
    _format_float(report["recall"]),
    _format_float(report["specificity"]),
    _format_float(report["negative_recall"]),
    _format_float(report["f1"]),
    _format_float(report["mcc"]),
    report["tn"],
    report["fp"],
    report["fn"],
    report["tp"],
    _format_float(report["auroc"]),
    _format_float(report["auprc"]),
    report["label_ratio"],
    report["pred_ratio"],
  ]
  return ([prefix] if prefix is not None else []) + row


def _regression_row(task_name, n, report, prefix=None):
  row = [
    task_name,
    n,
    _format_float(report["label_mean"]),
    _format_float(report["label_std"]),
    _format_float(report["pred_mean"]),
    _format_float(report["pred_std"]),
    _format_float(report["mae"]),
    _format_float(report["rmse"]),
    _format_float(report["pearson"]),
    _format_float(report["spearman"]),
    _format_float(report["r2"]),
  ]
  return ([prefix] if prefix is not None else []) + row


def _source_norm_tensors(task_name, sources, source_regression_stats, device):
  task_stats = (source_regression_stats or {}).get(task_name, {})
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


def _denormalize_regression_preds(task_name, normalized_preds, sources, source_regression_stats):
  means, stds = _source_norm_tensors(task_name, sources, source_regression_stats, normalized_preds.device)
  return normalized_preds * stds + means


def _source_normalized_regression_report(by_source):
  normalized_labels = []
  normalized_preds = []

  for values in by_source.values():
    labels = values["labels"]
    preds = values["preds"]
    if len(labels) < 2:
      continue

    label_tensor = torch.tensor(labels, dtype=torch.float)
    std = label_tensor.std(unbiased=False).item()
    if std == 0.0:
      continue
    mean = label_tensor.mean().item()
    normalized_labels.extend((label - mean) / std for label in labels)
    normalized_preds.extend((pred - mean) / std for pred in preds)

  if not normalized_labels:
    return None, 0
  return regression_report(normalized_labels, normalized_preds), len(normalized_labels)


CLASSIFICATION_COLUMNS = [
  "task", "n", "acc", "bal_acc", "precision", "recall", "specificity", "neg_recall",
  "f1", "mcc", "tn", "fp", "fn", "tp", "auroc", "auprc", "label_ratio", "pred_ratio",
]
REGRESSION_COLUMNS = [
  "task", "n", "label_mean", "label_std", "pred_mean", "pred_std", "mae", "rmse",
  "pearson", "spearman", "r2",
]


def parse_args():
  parser = argparse.ArgumentParser(description="Validate a trained multitask group-pair checkpoint.")
  parser.add_argument("--checkpoint", required=True, help="Path to the saved adapter checkpoint.")
  parser.add_argument("--cache", default=str(DEFAULT_CACHE_PATH), help="Path to the tokenized cache.")
  parser.add_argument("--split", default="validation", choices=["train", "validation", "test"], help="Dataset split to evaluate.")
  parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for evaluation.")
  return parser.parse_args()


def main():
  args = parse_args()

  print("Loading checkpoint and tokenized cache")
  checkpoint = torch.load(args.checkpoint, map_location="cpu")
  payload = torch.load(args.cache, map_location="cpu")

  task_order = payload["task_order"]
  task_metas = payload["task_metas"]
  split_payload = payload["splits"][args.split]
  train_split = payload["splits"]["train"]
  pad_token_id = payload["config"]["pad_token_id"]
  regression_means = payload["normalization"]["train_mean"].to(DEVICE)
  regression_stds = payload["normalization"]["train_std"].to(DEVICE)
  source_regression_stats = checkpoint["config"].get("source_regression_stats", {})

  dataset = MultiTaskGroupPairDataset(split_payload)
  loader = DataLoader(
    dataset,
    batch_sampler=MultiTaskBatchSampler(
      dataset,
      args.batch_size,
      max_tokens_per_batch=EVAL_MAX_TOKENS_PER_BATCH,
    ),
    collate_fn=lambda batch: collate_multitask_batch(batch, pad_token_id, include_sources=True),
    pin_memory=PIN_MEMORY,
  )

  task_output_dims = {}
  for task_idx, task_name in enumerate(task_order):
    meta = task_metas[task_name]
    train_mask = train_split["label_mask"][:, task_idx]
    train_labels = train_split["raw_labels"][:, task_idx]
    task_output_dims[task_name] = output_dim_from_meta(meta, train_labels, train_mask)

  model_name = checkpoint["config"].get("model_name", MODEL_NAME)
  base_model = T5EncoderModel.from_pretrained(model_name).to(DEVICE)
  if DEVICE.type == "cuda":
    base_model.bfloat16()

  embed_dim = checkpoint["config"]["embed_dim"]
  model = MultiTaskGroupPairModel(
    base_model,
    task_order,
    task_output_dims,
    embed_dim=embed_dim,
    task_metas=task_metas,
    adapter_dim=checkpoint["config"].get("adapter_dim", ADAPTER_DIM),
    dropout=checkpoint["config"].get("dropout", DROPOUT),
    classification_head_hidden=checkpoint["config"].get("classification_head_hidden", CLASSIFICATION_HEAD_HIDDEN),
    regression_head_hidden=checkpoint["config"].get("regression_head_hidden", REGRESSION_HEAD_HIDDEN),
  ).to(DEVICE)

  model.adapter.load_state_dict(checkpoint["adapter_state_dict"])
  model.residue_pool.load_state_dict(checkpoint["residue_pool_state_dict"])
  model.group_pool.load_state_dict(checkpoint["group_pool_state_dict"])
  model.pair_mlp.load_state_dict(checkpoint["pair_mlp_state_dict"])
  for task_name, state_dict in checkpoint["head_state_dicts"].items():
    model.heads[task_name].load_state_dict(state_dict)
  model.eval()

  predictions = {
    task_name: {
      "labels": [],
      "preds": [],
      "scores": [],
      "by_source": defaultdict(_empty_prediction_bucket),
    }
    for task_name in task_order
  }

  print(f"Running evaluation on split='{args.split}'")
  with torch.no_grad():
    for (
      input_ids,
      attn_mask,
      chain_to_sample,
      chain_to_group,
      raw_labels,
      normalized_labels,
      label_mask,
      sources,
    ) in tqdm(loader, desc="Validate"):
      input_ids = input_ids.to(DEVICE, non_blocking=PIN_MEMORY)
      attn_mask = attn_mask.to(DEVICE, non_blocking=PIN_MEMORY)
      chain_to_sample = chain_to_sample.to(DEVICE, non_blocking=PIN_MEMORY)
      chain_to_group = chain_to_group.to(DEVICE, non_blocking=PIN_MEMORY)
      raw_labels = raw_labels.to(DEVICE, non_blocking=PIN_MEMORY)
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
          labels = raw_labels[mask, task_idx].float()
          mask_list = mask.detach().cpu().tolist()
          masked_sources = [source for source, keep in zip(sources, mask_list) if keep]
          if task_name in source_regression_stats:
            preds = _denormalize_regression_preds(task_name, preds_norm, masked_sources, source_regression_stats)
          else:
            preds = preds_norm * regression_stds[task_idx] + regression_means[task_idx]
          _append_regression_predictions(
            predictions,
            task_name,
            labels.cpu().tolist(),
            preds.cpu().tolist(),
            masked_sources,
          )
        else:
          logits = outputs[task_name][mask].float()
          probs = torch.softmax(logits, dim=1)
          preds = probs.argmax(dim=1)
          labels = raw_labels[mask, task_idx].long()
          mask_list = mask.detach().cpu().tolist()
          masked_sources = [source for source, keep in zip(sources, mask_list) if keep]
          _append_task_predictions(
            predictions,
            task_name,
            labels.cpu().tolist(),
            preds.cpu().tolist(),
            probs[:, 1].cpu().tolist(),
            masked_sources,
          )

  print()
  print(f"Dataset size ({args.split}): {len(dataset)} pairs")
  print(f"Checkpoint: {args.checkpoint}")
  print(f"Cache: {args.cache}")
  print()

  classification_rows = []
  regression_rows = []
  for task_name in sorted(task_order):
    labels = predictions[task_name]["labels"]
    preds = predictions[task_name]["preds"]
    if not labels:
      continue

    meta = task_metas[task_name]
    if meta["dtype"] == "bool":
      report = classification_report(labels, preds, predictions[task_name]["scores"])
      classification_rows.append(_classification_row(task_name, len(labels), report))
    else:
      report = regression_report(labels, preds)
      regression_rows.append(_regression_row(task_name, len(labels), report))

  print(
    _format_table(
      "Classification Tasks",
      CLASSIFICATION_COLUMNS,
      classification_rows,
    )
  )
  print(
    _format_table(
      "Regression Tasks",
      REGRESSION_COLUMNS,
      regression_rows,
    )
  )

  source_classification_rows = []
  source_regression_rows = []
  source_normalized_regression_rows = []
  for task_name in sorted(task_order):
    meta = task_metas[task_name]
    for source, source_values in sorted(predictions[task_name]["by_source"].items()):
      labels = source_values["labels"]
      preds = source_values["preds"]
      if not labels:
        continue

      if meta["dtype"] == "bool":
        report = classification_report(labels, preds, source_values["scores"])
        source_classification_rows.append(_classification_row(task_name, len(labels), report, prefix=source))
      else:
        report = regression_report(labels, preds)
        source_regression_rows.append(_regression_row(task_name, len(labels), report, prefix=source))

    if meta["dtype"] == "float":
      report, source_normalized_n = _source_normalized_regression_report(predictions[task_name]["by_source"])
      if report is not None:
        source_normalized_regression_rows.append(_regression_row(task_name, source_normalized_n, report))

  print(
    _format_table(
      "Source-Specific Classification Tasks",
      ["source"] + CLASSIFICATION_COLUMNS,
      source_classification_rows,
    )
  )
  print(
    _format_table(
      "Source-Specific Regression Tasks",
      ["source"] + REGRESSION_COLUMNS,
      source_regression_rows,
    )
  )
  print(
    _format_table(
      "Source-Normalized Regression Tasks",
      REGRESSION_COLUMNS,
      source_normalized_regression_rows,
    )
  )

  checkpoint_calibration = checkpoint["config"].get("calibration")
  if checkpoint_calibration:
    calibrated_predictions = apply_posthoc_calibration(predictions, task_metas, checkpoint_calibration)
    checkpoint_classification_rows = []
    checkpoint_regression_rows = []
    classification_params = checkpoint_calibration.get("classification", {})
    regression_params = checkpoint_calibration.get("regression", {})

    for task_name in sorted(task_order):
      labels = calibrated_predictions[task_name]["labels"]
      preds = calibrated_predictions[task_name]["preds"]
      if not labels:
        continue

      meta = task_metas[task_name]
      if meta["dtype"] == "bool" and task_name in classification_params:
        report = classification_report(labels, preds, calibrated_predictions[task_name]["scores"])
        checkpoint_classification_rows.append(
          [
            task_name,
            classification_params[task_name]["calibration_size"],
            _format_float(classification_params[task_name]["threshold"]),
            _format_float(report["acc"]),
            _format_float(report["balanced_acc"]),
            _format_float(report["precision"]),
            _format_float(report["recall"]),
            _format_float(report["specificity"]),
            _format_float(report["negative_recall"]),
            _format_float(report["f1"]),
            _format_float(report["mcc"]),
            report["tn"],
            report["fp"],
            report["fn"],
            report["tp"],
            _format_float(report["auroc"]),
            _format_float(report["auprc"]),
            report["label_ratio"],
            report["pred_ratio"],
          ]
        )
      elif meta["dtype"] == "float" and task_name in regression_params:
        report = regression_report(labels, preds)
        checkpoint_regression_rows.append(
          [
            task_name,
            regression_params[task_name]["calibration_size"],
            _format_float(regression_params[task_name]["slope"]),
            _format_float(regression_params[task_name]["intercept"]),
            _format_float(report["pred_mean"]),
            _format_float(report["pred_std"]),
            _format_float(report["mae"]),
            _format_float(report["rmse"]),
            _format_float(report["pearson"]),
            _format_float(report["spearman"]),
            _format_float(report["r2"]),
          ]
        )

    print(
      _format_table(
        "Checkpoint Classification Calibration Applied",
        [
          "task", "cal_n", "thr", "acc", "bal_acc", "precision", "recall", "specificity",
          "neg_recall", "f1", "mcc", "tn", "fp", "fn", "tp", "auroc", "auprc",
          "label_ratio", "pred_ratio",
        ],
        checkpoint_classification_rows,
      )
    )
    print(
      _format_table(
        "Checkpoint Regression Calibration Applied",
        ["task", "cal_n", "slope", "intercept", "pred_mean", "pred_std", "mae", "rmse", "pearson", "spearman", "r2"],
        checkpoint_regression_rows,
      )
    )
  else:
    print(
      _format_table(
        "Post-hoc Classification Threshold Tuning (fit on internal half, report on held-out half)",
        [
          "task", "cal_n", "rep_n", "thr", "acc", "bal_acc", "precision", "recall",
          "specificity", "neg_recall", "f1", "mcc", "tn", "fp", "fn", "tp", "auroc",
          "auprc", "label_ratio", "pred_ratio",
        ],
        format_posthoc_classification_rows(predictions, task_metas),
      )
    )
    print(
      _format_table(
        "Post-hoc Regression Calibration (fit on internal half, report on held-out half)",
        ["task", "cal_n", "rep_n", "slope", "intercept", "pred_mean", "pred_std", "mae", "rmse", "pearson", "spearman", "r2"],
        format_posthoc_regression_rows(predictions, task_metas),
      )
    )


if __name__ == "__main__":
  main()

"""
Validate a trained interaction checkpoint on a cached split.
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
)
from config import (
  ADAPTER_DIM,
  BACKBONE_EMBEDDING_CACHE_PATH,
  BATCH_SIZE,
  CLASSIFICATION_HEAD_HIDDEN,
  DROPOUT,
  EVAL_MAX_TOKENS_PER_BATCH,
  MODEL_NAME,
  TRAIN_CACHE_PATH,
  USE_BACKBONE_EMBEDDING_CACHE,
)
from model import (
  MultiTaskBatchSampler,
  MultiTaskGroupPairDataset,
  MultiTaskGroupPairModel,
  collate_multitask_batch,
  load_backbone_embedding_cache,
  output_dim_from_meta,
)


DEFAULT_CACHE_PATH = TRAIN_CACHE_PATH
TASK_NAME = "interaction"

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


def _append_predictions(target, labels, preds, scores, sources=None):
  target[TASK_NAME]["labels"].extend(labels)
  target[TASK_NAME]["preds"].extend(preds)
  target[TASK_NAME]["scores"].extend(scores)

  if sources is None:
    return

  for source, label, pred, score in zip(sources, labels, preds, scores):
    bucket = target[TASK_NAME]["by_source"][source]
    bucket["labels"].append(label)
    bucket["preds"].append(pred)
    bucket["scores"].append(score)


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


CLASSIFICATION_COLUMNS = [
  "task", "n", "acc", "bal_acc", "precision", "recall", "specificity", "neg_recall",
  "f1", "mcc", "tn", "fp", "fn", "tp", "auroc", "auprc", "label_ratio", "pred_ratio",
]


def parse_args():
  parser = argparse.ArgumentParser(description="Validate a trained interaction checkpoint.")
  parser.add_argument("--checkpoint", required=True, help="Path to the saved adapter checkpoint.")
  parser.add_argument("--cache", default=str(DEFAULT_CACHE_PATH), help="Path to the tokenized cache.")
  parser.add_argument("--split", default="validation", choices=["train", "validation", "test"], help="Dataset split to evaluate.")
  parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for evaluation.")
  return parser.parse_args()


def _load_embedding_cache(model_name, tokenized_cache_path):
  if USE_BACKBONE_EMBEDDING_CACHE and BACKBONE_EMBEDDING_CACHE_PATH.exists():
    embedding_cache, payload = load_backbone_embedding_cache(
      BACKBONE_EMBEDDING_CACHE_PATH,
      expected_model_name=model_name,
      expected_tokenized_cache_path=tokenized_cache_path,
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


def main():
  args = parse_args()

  print("Loading checkpoint and tokenized cache")
  checkpoint = torch.load(args.checkpoint, map_location="cpu")
  payload = torch.load(args.cache, map_location="cpu")

  task_order = payload["task_order"]
  task_metas = payload["task_metas"]
  if task_order != [TASK_NAME]:
    raise ValueError(f"Expected interaction-only cache, found task_order={task_order!r}. Re-run tokenize_data.py.")

  split_payload = payload["splits"][args.split]
  train_split = payload["splits"]["train"]
  pad_token_id = payload["config"]["pad_token_id"]
  task_idx = task_order.index(TASK_NAME)
  model_name = checkpoint["config"].get("model_name", MODEL_NAME)
  embedding_cache = _load_embedding_cache(model_name, args.cache)

  dataset = MultiTaskGroupPairDataset(split_payload, embedding_cache=embedding_cache)
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

  train_mask = train_split["label_mask"][:, task_idx]
  train_labels = train_split["raw_labels"][:, task_idx]
  task_output_dims = {
    TASK_NAME: output_dim_from_meta(task_metas[TASK_NAME], train_labels, train_mask),
  }

  embed_dim = checkpoint["config"]["embed_dim"]
  if embedding_cache is None:
    base_model = T5EncoderModel.from_pretrained(model_name).to(DEVICE)
    if DEVICE.type == "cuda":
      base_model.bfloat16()
  else:
    base_model = None

  model = MultiTaskGroupPairModel(
    base_model,
    task_order,
    task_output_dims,
    embed_dim=embed_dim,
    task_metas=task_metas,
    adapter_dim=checkpoint["config"].get("adapter_dim", ADAPTER_DIM),
    dropout=checkpoint["config"].get("dropout", DROPOUT),
    classification_head_hidden=checkpoint["config"].get("classification_head_hidden", CLASSIFICATION_HEAD_HIDDEN),
  ).to(DEVICE)

  model.adapter.load_state_dict(checkpoint["adapter_state_dict"])
  model.residue_pool.load_state_dict(checkpoint["residue_pool_state_dict"])
  model.group_pool.load_state_dict(checkpoint["group_pool_state_dict"])
  model.pair_mlp.load_state_dict(checkpoint["pair_mlp_state_dict"])
  for task_name, state_dict in checkpoint["head_state_dicts"].items():
    model.heads[task_name].load_state_dict(state_dict)
  model.eval()

  predictions = {
    TASK_NAME: {
      "labels": [],
      "preds": [],
      "scores": [],
      "by_source": defaultdict(_empty_prediction_bucket),
    }
  }

  print(f"Running evaluation on split='{args.split}'")
  with torch.no_grad():
    for batch in tqdm(loader, desc="Validate"):
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

      with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=AMP_ENABLED):
        outputs = model(
          input_ids,
          attn_mask,
          chain_to_sample,
          chain_to_group,
          raw_labels.shape[0],
          precomputed_embeddings=input_embeddings,
        )

      mask = label_mask[:, task_idx]
      if not mask.any():
        continue

      logits = outputs[TASK_NAME][mask].float()
      probs = torch.softmax(logits, dim=1)
      preds = probs.argmax(dim=1)
      labels = raw_labels[mask, task_idx].long()
      masked_sources = [source for source, keep in zip(sources, mask.detach().cpu().tolist()) if keep]
      _append_predictions(
        predictions,
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

  report = classification_report(
    predictions[TASK_NAME]["labels"],
    predictions[TASK_NAME]["preds"],
    predictions[TASK_NAME]["scores"],
  )
  print(
    _format_table(
      "Classification Tasks",
      CLASSIFICATION_COLUMNS,
      [_classification_row(TASK_NAME, len(predictions[TASK_NAME]["labels"]), report)],
    )
  )

  source_rows = []
  for source, source_values in sorted(predictions[TASK_NAME]["by_source"].items()):
    labels = source_values["labels"]
    if not labels:
      continue
    source_report = classification_report(labels, source_values["preds"], source_values["scores"])
    source_rows.append(_classification_row(TASK_NAME, len(labels), source_report, prefix=source))

  print(
    _format_table(
      "Source-Specific Classification Tasks",
      ["source"] + CLASSIFICATION_COLUMNS,
      source_rows,
    )
  )

  checkpoint_calibration = checkpoint["config"].get("calibration")
  if checkpoint_calibration:
    calibrated_predictions = apply_posthoc_calibration(predictions, task_metas, checkpoint_calibration)
    classification_params = checkpoint_calibration.get("classification", {})
    checkpoint_rows = []

    if TASK_NAME in classification_params:
      calibrated_report = classification_report(
        calibrated_predictions[TASK_NAME]["labels"],
        calibrated_predictions[TASK_NAME]["preds"],
        calibrated_predictions[TASK_NAME]["scores"],
      )
      checkpoint_rows.append(
        [
          TASK_NAME,
          classification_params[TASK_NAME]["calibration_size"],
          _format_float(classification_params[TASK_NAME]["threshold"]),
          _format_float(calibrated_report["acc"]),
          _format_float(calibrated_report["balanced_acc"]),
          _format_float(calibrated_report["precision"]),
          _format_float(calibrated_report["recall"]),
          _format_float(calibrated_report["specificity"]),
          _format_float(calibrated_report["negative_recall"]),
          _format_float(calibrated_report["f1"]),
          _format_float(calibrated_report["mcc"]),
          calibrated_report["tn"],
          calibrated_report["fp"],
          calibrated_report["fn"],
          calibrated_report["tp"],
          _format_float(calibrated_report["auroc"]),
          _format_float(calibrated_report["auprc"]),
          calibrated_report["label_ratio"],
          calibrated_report["pred_ratio"],
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
        checkpoint_rows,
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


if __name__ == "__main__":
  main()

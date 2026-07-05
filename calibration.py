"""
Post-hoc calibration utilities for protein-group pair validation outputs.
"""

from collections import Counter

import torch
from sklearn.metrics import (
  accuracy_score,
  average_precision_score,
  balanced_accuracy_score,
  f1_score,
  mean_absolute_error,
  mean_squared_error,
  precision_score,
  r2_score,
  recall_score,
  roc_auc_score,
)


def fit_linear_regression_calibrator(preds, labels):
  preds_tensor = torch.tensor(preds, dtype=torch.float)
  labels_tensor = torch.tensor(labels, dtype=torch.float)

  pred_mean = preds_tensor.mean()
  label_mean = labels_tensor.mean()
  centered_preds = preds_tensor - pred_mean
  variance = centered_preds.pow(2).mean()
  if variance.item() == 0.0:
    return 0.0, label_mean.item()

  covariance = (centered_preds * (labels_tensor - label_mean)).mean()
  slope = covariance.div(variance).item()
  intercept = (label_mean - slope * pred_mean).item()
  return slope, intercept


def apply_linear_regression_calibrator(preds, slope, intercept):
  return [slope * pred + intercept for pred in preds]


def tune_binary_threshold(labels, scores):
  best_threshold = 0.5
  best_score = -1.0
  for threshold in [idx / 100.0 for idx in range(5, 96)]:
    preds = [1 if score >= threshold else 0 for score in scores]
    score = f1_score(labels, preds, zero_division=0)
    if score > best_score:
      best_score = score
      best_threshold = threshold
  return best_threshold


def apply_binary_threshold(scores, threshold):
  return [1 if score >= threshold else 0 for score in scores]


def fit_posthoc_calibration(predictions, task_metas, calibration_split="validation"):
  calibration = {
    "source_split": calibration_split,
    "classification": {},
    "regression": {},
  }

  for task_name, values in predictions.items():
    labels = values["labels"]
    if not labels:
      continue

    meta = task_metas[task_name]
    if meta["dtype"] == "float":
      slope, intercept = fit_linear_regression_calibrator(values["preds"], labels)
      calibration["regression"][task_name] = {
        "slope": slope,
        "intercept": intercept,
        "calibration_size": len(labels),
      }
    elif meta["dtype"] == "bool" and len(set(labels)) >= 2:
      threshold = tune_binary_threshold(labels, values["scores"])
      calibration["classification"][task_name] = {
        "threshold": threshold,
        "calibration_size": len(labels),
      }

  return calibration


def apply_posthoc_calibration(predictions, task_metas, calibration):
  calibrated = {}
  classification_calibration = (calibration or {}).get("classification", {})
  regression_calibration = (calibration or {}).get("regression", {})

  for task_name, values in predictions.items():
    task_values = {
      key: list(value) if isinstance(value, list) else value
      for key, value in values.items()
    }
    meta = task_metas[task_name]
    if meta["dtype"] == "float" and task_name in regression_calibration:
      params = regression_calibration[task_name]
      task_values["preds"] = apply_linear_regression_calibrator(task_values["preds"], params["slope"], params["intercept"])
    elif meta["dtype"] == "bool" and task_name in classification_calibration:
      params = classification_calibration[task_name]
      task_values["preds"] = apply_binary_threshold(task_values["scores"], params["threshold"])
    calibrated[task_name] = task_values

  return calibrated


def _format_float(value):
  if value is None:
    return "-"
  return f"{value:.4f}"


def _label_ratio_string(labels):
  if not labels:
    return "-"
  counts = Counter(labels)
  total = len(labels)
  return " ".join(f"{label}:{counts[label] / total:.3f}" for label in sorted(counts))


def _pearson_corr(labels, preds):
  if len(labels) < 2:
    return None
  labels_tensor = torch.tensor(labels, dtype=torch.float)
  preds_tensor = torch.tensor(preds, dtype=torch.float)
  centered_labels = labels_tensor - labels_tensor.mean()
  centered_preds = preds_tensor - preds_tensor.mean()
  denominator = torch.sqrt(centered_labels.pow(2).sum() * centered_preds.pow(2).sum())
  if denominator.item() == 0.0:
    return None
  return (centered_labels * centered_preds).sum().div(denominator).item()


def _average_ranks(values):
  indexed = sorted(enumerate(values), key=lambda item: item[1])
  ranks = [0.0] * len(values)
  start = 0
  while start < len(indexed):
    end = start + 1
    while end < len(indexed) and indexed[end][1] == indexed[start][1]:
      end += 1
    average_rank = (start + end - 1) / 2.0 + 1.0
    for idx in range(start, end):
      ranks[indexed[idx][0]] = average_rank
    start = end
  return ranks


def _spearman_corr(labels, preds):
  if len(labels) < 2:
    return None
  return _pearson_corr(_average_ranks(labels), _average_ranks(preds))


def classification_report(labels, preds, scores):
  report = {
    "acc": accuracy_score(labels, preds),
    "balanced_acc": balanced_accuracy_score(labels, preds),
    "precision": precision_score(labels, preds, zero_division=0),
    "recall": recall_score(labels, preds, zero_division=0),
    "f1": f1_score(labels, preds, zero_division=0),
    "label_ratio": _label_ratio_string(labels),
    "pred_ratio": _label_ratio_string(preds),
  }
  try:
    report["auroc"] = roc_auc_score(labels, scores)
    report["auprc"] = average_precision_score(labels, scores)
  except ValueError:
    report["auroc"] = None
    report["auprc"] = None
  return report


def regression_report(labels, preds):
  labels_tensor = torch.tensor(labels, dtype=torch.float)
  preds_tensor = torch.tensor(preds, dtype=torch.float)
  report = {
    "label_mean": labels_tensor.mean().item(),
    "label_std": labels_tensor.std(unbiased=False).item(),
    "pred_mean": preds_tensor.mean().item(),
    "pred_std": preds_tensor.std(unbiased=False).item(),
    "mae": mean_absolute_error(labels, preds),
    "rmse": mean_squared_error(labels, preds) ** 0.5,
    "pearson": _pearson_corr(labels, preds),
    "spearman": _spearman_corr(labels, preds),
  }
  try:
    report["r2"] = r2_score(labels, preds)
  except ValueError:
    report["r2"] = None
  return report


def format_posthoc_classification_rows(predictions, task_metas):
  rows = []
  for task_name, values in sorted(predictions.items()):
    if task_metas[task_name]["dtype"] != "bool" or len(values["labels"]) < 4:
      continue
    labels = values["labels"]
    scores = values["scores"]
    calib_labels = labels[::2]
    calib_scores = scores[::2]
    report_labels = labels[1::2]
    report_scores = scores[1::2]
    if len(set(calib_labels)) < 2 or len(set(report_labels)) < 2:
      continue
    threshold = tune_binary_threshold(calib_labels, calib_scores)
    report_preds = apply_binary_threshold(report_scores, threshold)
    report = classification_report(report_labels, report_preds, report_scores)
    rows.append(
      [
        task_name,
        len(calib_labels),
        len(report_labels),
        _format_float(threshold),
        _format_float(report["acc"]),
        _format_float(report["balanced_acc"]),
        _format_float(report["precision"]),
        _format_float(report["recall"]),
        _format_float(report["f1"]),
        _format_float(report["auroc"]),
        _format_float(report["auprc"]),
        report["label_ratio"],
        report["pred_ratio"],
      ]
    )
  return rows


def format_posthoc_regression_rows(predictions, task_metas):
  rows = []
  for task_name, values in sorted(predictions.items()):
    if task_metas[task_name]["dtype"] != "float" or len(values["labels"]) < 4:
      continue
    labels = values["labels"]
    preds = values["preds"]
    calib_labels = labels[::2]
    calib_preds = preds[::2]
    report_labels = labels[1::2]
    report_preds = preds[1::2]
    slope, intercept = fit_linear_regression_calibrator(calib_preds, calib_labels)
    report = regression_report(report_labels, apply_linear_regression_calibrator(report_preds, slope, intercept))
    rows.append(
      [
        task_name,
        len(calib_labels),
        len(report_labels),
        _format_float(slope),
        _format_float(intercept),
        _format_float(report["pred_mean"]),
        _format_float(report["pred_std"]),
        _format_float(report["mae"]),
        _format_float(report["rmse"]),
        _format_float(report["pearson"]),
        _format_float(report["spearman"]),
        _format_float(report["r2"]),
      ]
    )
  return rows

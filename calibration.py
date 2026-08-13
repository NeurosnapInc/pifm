"""
Post-hoc calibration and reporting utilities for interaction classification.
"""

from collections import Counter

from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, average_precision_score


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
  }

  for task_name, values in predictions.items():
    labels = values["labels"]
    if not labels:
      continue

    meta = task_metas[task_name]
    if meta["dtype"] == "bool" and len(set(labels)) >= 2:
      threshold = tune_binary_threshold(labels, values["scores"])
      calibration["classification"][task_name] = {
        "threshold": threshold,
        "calibration_size": len(labels),
      }

  return calibration


def apply_posthoc_calibration(predictions, task_metas, calibration):
  calibrated = {}
  classification_calibration = (calibration or {}).get("classification", {})

  for task_name, values in predictions.items():
    task_values = {
      key: list(value) if isinstance(value, list) else value
      for key, value in values.items()
    }
    meta = task_metas[task_name]
    if meta["dtype"] == "bool" and task_name in classification_calibration:
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


def _binary_confusion_counts(labels, preds):
  tn = fp = fn = tp = 0
  for label, pred in zip(labels, preds):
    if label == 0 and pred == 0:
      tn += 1
    elif label == 0 and pred == 1:
      fp += 1
    elif label == 1 and pred == 0:
      fn += 1
    elif label == 1 and pred == 1:
      tp += 1
  return tn, fp, fn, tp


def _binary_average_precision(labels, scores):
  if len(set(labels)) < 2:
    if all(label == 1 for label in labels):
      return 1.0
    if all(label == 0 for label in labels):
      return 0.0
    return None
  return average_precision_score(labels, scores)


def classification_report(labels, preds, scores):
  tn, fp, fn, tp = _binary_confusion_counts(labels, preds)
  total = tn + fp + fn + tp
  positive_precision = tp / (tp + fp) if (tp + fp) else 0.0
  positive_recall = tp / (tp + fn) if (tp + fn) else 0.0
  specificity = tn / (tn + fp) if (tn + fp) else None
  balanced_acc = None
  if specificity is not None:
    balanced_acc = (positive_recall + specificity) / 2.0
  else:
    balanced_acc = positive_recall

  report = {
    "acc": (tp + tn) / total if total else None,
    "balanced_acc": balanced_acc,
    "precision": positive_precision,
    "recall": positive_recall,
    "specificity": specificity,
    "negative_recall": specificity,
    "f1": f1_score(labels, preds, zero_division=0),
    "mcc": matthews_corrcoef(labels, preds) if len(set(labels)) >= 2 and len(set(preds)) >= 2 else 0.0,
    "tn": tn,
    "fp": fp,
    "fn": fn,
    "tp": tp,
    "label_ratio": _label_ratio_string(labels),
    "pred_ratio": _label_ratio_string(preds),
  }
  report["auroc"] = roc_auc_score(labels, scores) if len(set(labels)) >= 2 else None
  report["auprc"] = _binary_average_precision(labels, scores)
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
  return rows

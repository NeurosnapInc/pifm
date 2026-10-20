"""Training-only pKd supervision and diagnostics; never a selection objective."""

import torch
import torch.nn.functional as F


def verify_classifier_cache(previous, current):
  """Refuse to change the baseline examples while adding auxiliary labels.

  Check row ordering, source labels, chain tokens, and classification tensors in
  every split. Affinity fields are deliberately excluded. A mismatch requires
  explicit dataset preparation rather than silently changing this ablation.
  """
  if previous["task_order"] != current["task_order"] or previous["splits"].keys() != current["splits"].keys():
    raise ValueError("Baseline task/split layout changed; original cache was not overwritten.")
  for key in ("model_name", "pad_token_id", "max_length"):
    if previous.get("config", {}).get(key) != current.get("config", {}).get(key):
      raise ValueError(f"Baseline tokenizer config {key} changed; original cache was not overwritten.")
  for name, old in previous["splits"].items():
    new = current["splits"][name]
    for key in ("raw_labels", "normalized_labels", "label_mask", "lengths"):
      if not torch.equal(torch.as_tensor(old[key]), torch.as_tensor(new[key])):
        raise ValueError(f"Baseline {name}/{key} changed; original cache was not overwritten.")
    if old["sources"] != new["sources"]:
      raise ValueError(f"Baseline {name}/sources changed; original cache was not overwritten.")
    for key in ("group1_input_ids", "group2_input_ids"):
      a, b = old[key], new[key]
      if len(a) != len(b) or any(len(x) != len(y) or any(not torch.equal(i, j) for i, j in zip(x, y)) for x, y in zip(a, b)):
        raise ValueError(f"Baseline {name}/{key} changed; original cache was not overwritten.")


def affinity_statistics(train_split):
  """Fit population mean/std exclusively on observed training pKd labels."""
  if "affinity_pkd" not in train_split or "affinity_mask" not in train_split:
    raise ValueError("Cache lacks auxiliary pKd labels. Run tokenize_data.py once; existing classifier splits are verified before replacement.")
  labels, mask = train_split["affinity_pkd"], train_split["affinity_mask"]
  if labels.shape != (len(train_split["raw_labels"]),) or mask.shape != labels.shape or mask.dtype != torch.bool:
    raise ValueError("Auxiliary pKd labels/masks must be aligned one-dimensional tensors.")
  values = labels[mask].double()
  if values.numel() < 2 or not torch.isfinite(values).all():
    raise ValueError("Auxiliary affinity requires at least two finite training pKd labels.")
  mean, std = values.mean().item(), values.std(unbiased=False).item()
  if std < 1e-8:
    raise ValueError("Training pKd variance is too small for normalization.")
  return {"mean": mean, "std": std, "n": values.numel(), "units": "pKd", "fit_split": "train"}


def masked_affinity_loss(predictions, targets, mask, statistics):
  """Mean Huber loss over observed labels only, in training-standardized pKd."""
  if not mask.any():
    return predictions.float().sum() * 0.0
  predictions = predictions[mask].float()
  targets = (targets[mask].float() - statistics["mean"]) / statistics["std"]
  if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
    raise FloatingPointError("Non-finite auxiliary affinity prediction/target.")
  return F.huber_loss(predictions, targets, delta=1.0, reduction="mean")


def affinity_report(labels, predictions, sources):
  """Summarize uncalibrated predictions in pKd, globally and by data source."""
  def report(y, p):
    if not y:
      return {"n": 0, "mae": None, "rmse": None, "pearson": None}
    y, p = torch.tensor(y, dtype=torch.float64), torch.tensor(p, dtype=torch.float64)
    if not torch.isfinite(y).all() or not torch.isfinite(p).all():
      raise FloatingPointError("Non-finite auxiliary affinity diagnostics.")
    error = p - y
    pearson = None
    if len(y) > 1 and y.std() > 0 and p.std() > 0:
      pearson = torch.corrcoef(torch.stack([y, p]))[0, 1].item()
    return {"n": len(y), "mae": error.abs().mean().item(), "rmse": error.square().mean().sqrt().item(), "pearson": pearson}

  result = report(labels, predictions)
  result["units"] = "pKd"
  result["by_source"] = {
    source: report([y for y, s in zip(labels, sources) if s == source],
                   [p for p, s in zip(predictions, sources) if s == source])
    for source in sorted(set(sources))
  }
  return result

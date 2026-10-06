"""Print saved validation/test metrics by reading checkpoint weights, without inference."""

import argparse
import json
import sys
from pathlib import Path

import torch


def find_checkpoints(paths):
  """Expand checkpoint files or directories into a sorted, duplicate-free file list."""
  files = set()
  for name in paths:
    path = Path(name)
    if path.is_dir():
      files.update(candidate.resolve() for candidate in path.iterdir() if candidate.suffix in (".pt", ".pth"))
    elif path.is_file():
      files.add(path.resolve())
    else:
      raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
  return sorted(files)


def checkpoint_summary(path):
  """Read configuration and stored evaluation reports without loading a model or cache.

  Older checkpoints can lack evaluation metadata. Their best training validation
  report is shown separately and never presented as a newly evaluated split.
  Missing test results remain explicitly unevaluated.
  """
  checkpoint = torch.load(path, map_location="cpu")
  config = checkpoint.get("config", {})
  return {
    "checkpoint": str(path),
    "run_timestamp": config.get("run_timestamp", config.get("run_date")),
    "training_seed": config.get("training_seed"),
    "selection_metric": config.get("classification_selection_metric"),
    "best_selection_metric": config.get("best_selection_metric"),
    "best_training_validation_report": config.get("best_task_report"),
    "evaluation": checkpoint.get("evaluation", {}),
  }


def summary_rows(summary):
  """Yield aggregate raw/calibrated split rows, including unavailable legacy results."""
  splits = summary["evaluation"].get("splits", {})
  for split in ("validation", "test"):
    result = splits.get(split)
    if result is None:
      report = summary.get("best_training_validation_report") if split == "validation" else None
      if report:
        report = dict(report)
        report.setdefault("balanced_acc", report.get("balanced_accuracy"))
        report["n"] = sum(report.get(key, 0) for key in ("tn", "fp", "fn", "tp"))
        yield _row(summary, split, "training-best", report)
      else:
        yield _row(summary, split, "not evaluated", {})
      continue
    for mode, key in (("raw", "classification"), ("calibrated", "calibrated_classification")):
      report = result.get(key, {}).get("interaction")
      if report is not None:
        yield _row(summary, split, mode, report)


def _row(summary, split, mode, report):
  values = [
    Path(summary["checkpoint"]).name, summary["training_seed"], split, mode, report.get("n"),
    report.get("threshold"), report.get("auroc"), report.get("auprc"), report.get("balanced_acc"),
    report.get("specificity"), report.get("mcc"), report.get("acc"), report.get("f1"),
  ]
  return ["-" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value) for value in values]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("paths", nargs="*", default=["checkpoints"], help="Checkpoint files or directories (default: checkpoints/).")
  parser.add_argument("--json", action="store_true", help="Print full saved evaluation metadata as JSON, including source reports.")
  args = parser.parse_args()
  try:
    paths = find_checkpoints(args.paths)
  except FileNotFoundError as exc:
    parser.error(str(exc))

  summaries = []
  errors = []
  for path in paths:
    try:
      summaries.append(checkpoint_summary(path))
    except Exception as exc:
      # One damaged file should not hide results from the other checkpoints.
      errors.append({"checkpoint": str(path), "error": str(exc)})
      print(f"Unable to read {path}: {exc}", file=sys.stderr)

  if args.json:
    print(json.dumps({"checkpoints": summaries, "errors": errors}, indent=2))
  elif summaries:
    columns = ["checkpoint", "seed", "split", "results", "n", "thr", "auroc", "auprc", "bal_acc", "specificity", "mcc", "acc", "f1"]
    rows = [row for summary in summaries for row in summary_rows(summary)]
    widths = [max(len(column), *(len(row[idx]) for row in rows)) for idx, column in enumerate(columns)]
    for row in [columns, ["-" * width for width in widths], *rows]:
      print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))
  else:
    print("No checkpoints found.")
  return 1 if errors else 0


if __name__ == "__main__":
  raise SystemExit(main())

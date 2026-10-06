"""Print saved run metrics from JSON sidecars without loading model weights."""

import argparse
import json
import sys
from pathlib import Path

from config import RUNS_DIR
from run_utils import read_run_metadata


def find_runs(paths):
  """Find runs in supplied directories or resolve an explicit checkpoint.pt path."""
  directories = set()
  for name in paths:
    path = Path(name)
    if path.is_file() and path.name == "checkpoint.pt":
      directories.add(path.parent.resolve())
    elif path.is_dir():
      if (path / "config.json").is_file():
        directories.add(path.resolve())
      else:
        directories.update(file.parent.resolve() for file in path.rglob("config.json"))
    else:
      raise FileNotFoundError(f"Run path does not exist: {path}")
  return sorted(directories)


def run_summary(path):
  """Read a run's metadata without importing torch or opening checkpoint.pt."""
  directory, config, metrics, notes = read_run_metadata(path)
  training = metrics.get("training", {})
  return {
    "run": str(directory),
    "run_timestamp": config.get("run_timestamp", config.get("run_date")),
    "training_seed": config.get("training_seed"),
    "selection_metric": config.get("classification_selection_metric"),
    "best_selection_metric": training.get("best_selection_metric"),
    "best_training_validation_report": training.get("best_task_report"),
    "evaluation": metrics,
    "notes": notes,
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
      else:
        title = "Classification Tasks" if mode == "raw" else "Checkpoint Classification Calibration Applied"
        historical = result.get("historical_reports", {}).get(title, {})
        for row in historical.get("rows", []):
          if row.get("task") != "interaction":
            continue
          report = dict(row)
          report["balanced_acc"] = row.get("bal_acc")
          report["threshold"] = row.get("thr")
          # Calibration count describes validation, not the evaluated test set.
          report.setdefault("n", result.get("dataset_size"))
          yield _row(summary, split, f"historical-{mode}", report)


def _row(summary, split, mode, report):
  values = [
    Path(summary["run"]).name, summary["training_seed"], split, mode, report.get("n"),
    report.get("threshold"), report.get("auroc"), report.get("auprc"), report.get("balanced_acc"),
    report.get("specificity"), report.get("mcc"), report.get("acc"), report.get("f1"),
  ]
  return ["-" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value) for value in values]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("paths", nargs="*", default=[str(RUNS_DIR)], help="Run directories, roots, or checkpoint.pt files (default: runs/).")
  parser.add_argument("--json", action="store_true", help="Print full saved evaluation metadata as JSON, including source reports.")
  parser.add_argument("--historical", action="store_true", help="Also print archived tables and original train-log notes, including regression.")
  args = parser.parse_args()
  try:
    paths = find_runs(args.paths)
  except FileNotFoundError as exc:
    parser.error(str(exc))

  summaries = []
  errors = []
  for path in paths:
    try:
      summaries.append(run_summary(path))
    except Exception as exc:
      # One damaged metadata file should not hide other runs.
      errors.append({"run": str(path), "error": str(exc)})
      print(f"Unable to read {path}: {exc}", file=sys.stderr)

  if args.json:
    print(json.dumps({"runs": summaries, "errors": errors}, indent=2))
  elif summaries:
    columns = ["run", "seed", "split", "results", "n", "thr", "auroc", "auprc", "bal_acc", "specificity", "mcc", "acc", "f1"]
    rows = [row for summary in summaries for row in summary_rows(summary)]
    widths = [max(len(column), *(len(row[idx]) for row in rows)) for idx, column in enumerate(columns)]
    for row in [columns, ["-" * width for width in widths], *rows]:
      print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))
  else:
    print("No runs found.")
  if args.historical and not args.json:
    for summary in summaries:
      evaluation = summary["evaluation"]
      notes = summary["notes"]
      if notes:
        print(f"\n{Path(summary['run']).name}: Historical Train Log\n{notes}")
      for split, result in evaluation.get("splits", {}).items():
        for table in result.get("historical_reports", {}).values():
          print(f"\n{Path(summary['run']).name}: {split} (historical definitions)\n{table['raw_table']}")
  return 1 if errors else 0


if __name__ == "__main__":
  raise SystemExit(main())

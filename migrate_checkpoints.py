"""Migrate legacy checkpoints to run directories and verify weights and metadata."""

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import torch

from config import RUNS_DIR
from run_utils import _json_value, read_run_metadata, save_run, split_checkpoint


def _identical(left, right):
  if isinstance(left, torch.Tensor):
    return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
  if isinstance(left, dict):
    return left.keys() == right.keys() and all(_identical(left[key], right[key]) for key in left)
  if isinstance(left, (list, tuple)):
    return len(left) == len(right) and all(_identical(a, b) for a, b in zip(left, right))
  return left == right


def _verify(directory, checkpoint, notes):
  """Verify every tensor and serialized metadata field before accepting a migration."""
  weights, config, metrics, expected_notes = split_checkpoint(checkpoint, notes)
  actual_weights = torch.load(directory / "checkpoint.pt", map_location="cpu")
  _, actual_config, actual_metrics, actual_notes = read_run_metadata(directory)
  if not _identical(weights, actual_weights):
    raise ValueError(f"Weight verification failed: {directory}")
  if actual_config != _json_value(config) or actual_metrics != _json_value(metrics):
    raise ValueError(f"Metadata verification failed: {directory}")
  if actual_notes != expected_notes.rstrip() + "\n":
    raise ValueError(f"Notes verification failed: {directory}")


def migrate_checkpoint(path, destination=RUNS_DIR, notes="", move=False):
  """Publish a verified run atomically; optionally remove its verified legacy file.

  Existing destinations are verified instead of overwritten, making reruns safe.
  Date-only legacy names remain date-only because filesystem modification times
  may reflect a download rather than the original training run.
  """
  path = Path(path)
  match = re.fullmatch(r"prostt5_group_pair_adapter_best_(\d{4}-\d{2}-\d{2}(?:_\d{2}-\d{2})?_seed_\d+(?:_\d+)?)", path.stem)
  if match is None:
    raise ValueError(f"Unrecognized checkpoint name: {path.name}")
  directory = Path(destination) / match.group(1)
  checkpoint = torch.load(path, map_location="cpu")
  date = match.group(1)[:10]
  if checkpoint["config"].get("run_date", date) != date:
    raise ValueError(f"Checkpoint date does not match filename: {path}")
  if directory.exists():
    _verify(directory, checkpoint, notes)
  else:
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".migrate-", dir=directory.parent) as temporary:
      staging = Path(temporary)
      save_run(checkpoint, staging, notes=notes)
      _verify(staging, checkpoint, notes)
      os.rename(staging, directory)
  if move:
    path.unlink()
  return directory


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", type=Path, default=Path("checkpoints"))
  parser.add_argument("--destination", type=Path, default=RUNS_DIR)
  parser.add_argument("--readme", type=Path, default=Path("README.md"), help="Optional train-log context for notes.md.")
  parser.add_argument("--move", action="store_true", help="Remove each original checkpoint only after verifying the new run.")
  args = parser.parse_args()
  readme = args.readme.read_text() if args.readme.exists() else ""
  versions = {m.group(1): m.group() for m in re.finditer(
    r"^### Version (\d{4}-\d{2}-\d{2})\n.*?(?=^### Version |\Z)", readme, re.M | re.S,
  )}
  paths = sorted(args.source.glob("*.pt")) if args.source.is_dir() else [args.source]
  migrated = []
  for path in paths:
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", path.name)
    if date_match is None:
      raise ValueError(f"Checkpoint filename has no date: {path.name}")
    date = date_match.group()
    context = versions.get(date, "").split("#### Validation Split")[0].strip()
    run_name = path.stem.replace("prostt5_group_pair_adapter_best_", "", 1)
    notes = f"# Run {run_name}\n\n"
    notes += f"Migrated from `{path.name}`. Legacy dates without a time remain date-only.\n\n{context}"
    directory = migrate_checkpoint(path, args.destination, notes=notes, move=args.move)
    print(f"Migrated {path} -> {directory}")
    migrated.append(str(directory))
  print(json.dumps({"migrated_runs": migrated}))


if __name__ == "__main__":
  main()

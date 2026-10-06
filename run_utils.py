"""Run directories with immutable weights and separately editable JSON metadata."""

import json
import math
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path


def git_revision():
  """Record the code revision when training from a Git checkout."""
  try:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    return result.stdout.strip()
  except (OSError, subprocess.CalledProcessError):
    return None


def create_run_directory(root, seed, timestamp=None):
  """Reserve a unique run directory, adding a suffix for runs in the same minute."""
  timestamp = timestamp or datetime.now().astimezone()
  root = Path(root)
  root.mkdir(parents=True, exist_ok=True)
  stem = f"{timestamp:%Y-%m-%d_%H-%M}_seed_{seed}"
  suffix = 1
  while True:
    path = root / (stem if suffix == 1 else f"{stem}_{suffix}")
    try:
      path.mkdir()
      return path
    except FileExistsError:
      suffix += 1


def resolve_run_directory(path):
  """Accept a run directory or its checkpoint.pt file and require its configuration."""
  path = Path(path)
  if path.is_file():
    if path.name != "checkpoint.pt":
      raise ValueError(f"Legacy checkpoint {path}: run migrate_checkpoints.py first.")
    path = path.parent
  if not (path / "config.json").is_file():
    raise FileNotFoundError(f"Missing run configuration: {path / 'config.json'}")
  return path


def _json_value(value):
  """Convert numeric scalars/tensors to portable JSON and undefined values to null."""
  if isinstance(value, dict):
    return {str(key): _json_value(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_value(item) for item in value]
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, float) and not math.isfinite(value):
    return None
  if hasattr(value, "tolist"):
    return _json_value(value.tolist())
  return value


def _atomic_write(path, writer, binary=False):
  """Replace one artifact only after serialization and flushing succeed."""
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
  try:
    with os.fdopen(fd, "wb" if binary else "w", **({} if binary else {"encoding": "utf-8"})) as handle:
      writer(handle)
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(temporary, path)
  finally:
    if os.path.exists(temporary):
      os.unlink(temporary)


def save_json(payload, path):
  """Atomically save strict, readable JSON with portable numeric values."""
  payload = _json_value(payload)
  _atomic_write(path, lambda handle: handle.write(json.dumps(payload, indent=2, allow_nan=False) + "\n"))


def read_run_metadata(path):
  """Read configuration, metrics, and notes without importing torch or loading weights."""
  directory = resolve_run_directory(path)
  config = json.loads((directory / "config.json").read_text())
  metrics_path = directory / "metrics.json"
  metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {"schema_version": 1, "splits": {}}
  notes_path = directory / "notes.md"
  notes = notes_path.read_text() if notes_path.exists() else ""
  return directory, config, metrics, notes


def load_run(path):
  """Load weights plus sidecars as the combined payload expected by model evaluation."""
  import torch

  directory, config, metrics, _ = read_run_metadata(path)
  checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu")
  checkpoint["config"] = config
  checkpoint["evaluation"] = metrics
  return directory, checkpoint


def split_checkpoint(checkpoint, notes=""):
  """Separate legacy weights, configuration, metrics, and archived prose losslessly.

  Selection results belong to metrics.training. Original train-log notes move to
  notes.md; their provenance stays in metrics.json alongside historical tables.
  """
  config = dict(checkpoint["config"])
  metrics = dict(checkpoint.get("evaluation", {"schema_version": 1, "splits": {}}))
  metrics.setdefault("schema_version", 1)
  metrics.setdefault("splits", {})
  training = dict(metrics.get("training", {}))
  for key in ("best_selection_metric", "best_task_report"):
    if key in config:
      training[key] = config.pop(key)
  if training:
    metrics["training"] = training
  history = dict(metrics.get("historical_train_log", {}))
  original_notes = history.pop("results_notes", None)
  if original_notes:
    notes = notes.rstrip() + "\n\n## Original Result Notes\n\n" + original_notes
    history["notes_path"] = "notes.md"
    metrics["historical_train_log"] = history
  weights = {key: value for key, value in checkpoint.items() if key not in ("config", "evaluation")}
  return weights, config, metrics, notes


def save_run(checkpoint, directory, notes=""):
  """Write the four artifacts for a new run, refusing to replace existing artifacts."""
  import torch

  directory = Path(directory)
  directory.mkdir(parents=True, exist_ok=True)
  if any((directory / name).exists() for name in ("checkpoint.pt", "config.json", "metrics.json", "notes.md")):
    raise FileExistsError(f"Run artifacts already exist: {directory}")
  weights, config, metrics, notes = split_checkpoint(checkpoint, notes)
  _atomic_write(directory / "checkpoint.pt", lambda handle: torch.save(weights, handle), binary=True)
  save_json(config, directory / "config.json")
  save_json(metrics, directory / "metrics.json")
  _atomic_write(directory / "notes.md", lambda handle: handle.write(notes.rstrip() + "\n"))

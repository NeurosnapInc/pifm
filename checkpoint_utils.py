"""Checkpoint naming and atomic persistence shared by training and evaluation."""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import torch


def checkpoint_path(directory, seed, timestamp=None):
  """Return a date/hour/minute checkpoint path without replacing an existing run.

  A numeric suffix distinguishes runs saved within the same minute. The timestamp
  uses the machine's local timezone; the checkpoint also stores an ISO timestamp
  with its UTC offset so runs can be compared across servers.
  """
  timestamp = timestamp or datetime.now().astimezone()
  directory = Path(directory)
  stem = f"prostt5_group_pair_adapter_best_{timestamp:%Y-%m-%d_%H-%M}_seed_{seed}"
  path = directory / f"{stem}.pt"
  suffix = 2
  while path.exists():
    path = directory / f"{stem}_{suffix}.pt"
    suffix += 1
  return path


def save_checkpoint(checkpoint, path):
  """Save weights and metadata via a temporary file, then replace the destination.

  The temporary file lives alongside the checkpoint so the final rename is atomic.
  If serialization fails, the previous checkpoint is preserved and the temporary
  file is removed.
  """
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
  try:
    with os.fdopen(fd, "wb") as handle:
      torch.save(checkpoint, handle)
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(temporary_path, path)
  finally:
    if os.path.exists(temporary_path):
      os.unlink(temporary_path)

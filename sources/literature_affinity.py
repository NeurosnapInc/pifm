"""
Literature-derived affinity -- user-curated protein-protein binding constants.

There is no single canonical download for hand-collected affinities, so this
loader reads any CSV files you place in ``data/raw/literature/``. Each CSV must
have a header row with at least ``seq1`` and ``seq2`` columns (raw amino-acid
sequences; use a ``":"``-delimited value for a multi-chain side). Optional
columns:

- ``affinity_nm``       -- dissociation constant Kd in nanomolar (feeds pKd head)
- ``interaction_label`` -- ``1``/``0`` / ``true``/``false`` (feeds binary head)

Rows with neither an affinity nor a label default to a positive interaction,
matching the aggregation-layer default for curated entries.
"""

import csv
from typing import Iterator, Optional

from contract import InteractionEntry

from ._common import RAW_DIR, hint_missing

LITERATURE_DIR = RAW_DIR / "literature"

_TRUE_TOKENS = {"1", "true", "yes", "y", "t"}
_FALSE_TOKENS = {"0", "false", "no", "n", "f"}


def _parse_optional_float(value: Optional[str]) -> Optional[float]:
  if value is None:
    return None
  value = value.strip()
  if value == "" or value.upper() in ("NA", "NAN", "NONE"):
    return None
  return float(value)


def _parse_optional_bool(value: Optional[str]) -> Optional[bool]:
  if value is None:
    return None
  token = value.strip().lower()
  if token in _TRUE_TOKENS:
    return True
  if token in _FALSE_TOKENS:
    return False
  return None


def iter_literature_affinity() -> Iterator[InteractionEntry]:
  if not LITERATURE_DIR.is_dir():
    hint_missing(
      "literature_affinity",
      LITERATURE_DIR,
      "Create data/raw/literature/ and add CSV files with columns seq1,seq2[,affinity_nm,interaction_label].",
    )
    return

  csv_files = sorted(LITERATURE_DIR.glob("*.csv"))
  if not csv_files:
    hint_missing(
      "literature_affinity",
      LITERATURE_DIR / "*.csv",
      "Add CSV files with columns seq1,seq2[,affinity_nm,interaction_label] to data/raw/literature/.",
    )
    return

  emitted = 0
  skipped = 0
  for path in csv_files:
    with open(path, newline="", encoding="utf-8") as handle:
      reader = csv.DictReader(handle)
      for row in reader:
        seq1 = (row.get("seq1") or "").strip()
        seq2 = (row.get("seq2") or "").strip()
        if not seq1 or not seq2:
          skipped += 1
          continue
        try:
          affinity_nm = _parse_optional_float(row.get("affinity_nm"))
        except ValueError:
          skipped += 1
          continue
        interaction_label = _parse_optional_bool(row.get("interaction_label"))

        emitted += 1
        yield InteractionEntry(
          source="literature",
          group1=seq1,
          group2=seq2,
          affinity_nm=affinity_nm,
          interaction_label=interaction_label,
        )

  print(f"Source=literature_affinity emitted_rows={emitted} skipped_rows={skipped}")

"""
Aggregate protein-group interaction datasets into a DuckDB database.

The expected source contract is intentionally lightweight so new datasets can be
registered without modifying the rest of the training pipeline:

- add one `SourceSpec` to `SOURCE_SPECS`
- implement an iterable or generator function that yields `InteractionEntry`

Inspect results:
  duckdb -ui data/aggregated/aggregated.duckdb
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import duckdb
import pandas as pd


SequenceGroup = Union[str, Sequence[str]]
SourceFactory = Callable[[], Iterable["InteractionEntry"]]


@dataclass(frozen=True)
class InteractionEntry:
  source: str
  group1: SequenceGroup
  group2: SequenceGroup
  affinity_nm: Optional[float] = None
  interaction_label: Optional[bool] = None


@dataclass(frozen=True)
class SourceSpec:
  name: str
  loader: SourceFactory


def _empty_loader() -> Iterable[InteractionEntry]:
  return []


# Source priority is defined by list order. Earlier sources win if multiple sources
# provide the same canonicalized group pair.
SOURCE_SPECS: List[SourceSpec] = [
  # Example:
  # SourceSpec(name="bindingdb_curated", loader=iter_bindingdb_rows),
]


def normalize_chain_group(group: SequenceGroup) -> str:
  if isinstance(group, str):
    parts = [part.strip().upper() for part in group.split(":") if part.strip()]
  else:
    parts = [str(part).strip().upper() for part in group if str(part).strip()]

  if not parts:
    raise ValueError("Encountered an empty chain group.")

  return ":".join(sorted(parts))


def canonicalize_pair(group1: SequenceGroup, group2: SequenceGroup) -> Tuple[str, str]:
  normalized = [normalize_chain_group(group1), normalize_chain_group(group2)]
  normalized.sort()
  return normalized[0], normalized[1]


def affinity_nm_to_pkd(affinity_nm: Optional[float]) -> Optional[float]:
  if affinity_nm is None:
    return None

  value = float(affinity_nm)
  if value <= 0.0:
    raise ValueError(f"Affinity in nM must be positive, got {value}")
  return 9.0 - math.log10(value)


def _coerce_interaction_label(value: Optional[bool], affinity_nm: Optional[float]) -> Optional[float]:
  if value is None:
    return 1.0
  return 1.0 if bool(value) else 0.0


def _prepare_db(con: duckdb.DuckDBPyConnection):
  con.execute("DROP TABLE IF EXISTS samples")
  con.execute(
    """
    CREATE TABLE samples (
      source VARCHAR NOT NULL,
      group1 VARCHAR NOT NULL,
      group2 VARCHAR NOT NULL,
      interaction_label DOUBLE,
      affinity_nm DOUBLE,
      affinity_pkd DOUBLE,
      CONSTRAINT samples_group_pair_unique UNIQUE(group1, group2)
    )
    """
  )


def _insert_source_rows(con: duckdb.DuckDBPyConnection, spec: SourceSpec):
  inserted_rows = []
  skipped_invalid = 0
  skipped_duplicates = 0

  for entry in spec.loader():
    try:
      group1, group2 = canonicalize_pair(entry.group1, entry.group2)
      affinity_nm = None if entry.affinity_nm is None else float(entry.affinity_nm)
      interaction_label = _coerce_interaction_label(entry.interaction_label, affinity_nm)
      affinity_pkd = affinity_nm_to_pkd(affinity_nm)
    except Exception as exc:
      skipped_invalid += 1
      print(f"Source={spec.name} skipped_invalid_entry error={exc}")
      continue

    inserted_rows.append(
      (
        entry.source or spec.name,
        group1,
        group2,
        interaction_label,
        affinity_nm,
        affinity_pkd,
      )
    )

  if not inserted_rows:
    print(f"Source={spec.name} inserted=0 skipped_invalid={skipped_invalid} skipped_duplicate={skipped_duplicates}")
    return

  df = pd.DataFrame(
    inserted_rows,
    columns=["source", "group1", "group2", "interaction_label", "affinity_nm", "affinity_pkd"],
  )
  con.register("source_rows", df)
  try:
    before = con.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    con.execute(
      """
      INSERT INTO samples(source, group1, group2, interaction_label, affinity_nm, affinity_pkd)
      SELECT source, group1, group2, interaction_label, affinity_nm, affinity_pkd
      FROM source_rows
      ON CONFLICT(group1, group2) DO NOTHING
      """
    )
    after = con.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
  finally:
    con.unregister("source_rows")

  inserted = after - before
  skipped_duplicates = len(inserted_rows) - inserted
  print(f"Source={spec.name} inserted={inserted} skipped_invalid={skipped_invalid} skipped_duplicate={skipped_duplicates}")


def _print_dataset_audit(con: duckdb.DuckDBPyConnection):
  row = con.execute(
    """
    SELECT
      COUNT(*) AS total_rows,
      COUNT(DISTINCT source) AS total_sources,
      SUM(CASE WHEN interaction_label IS NOT NULL THEN 1 ELSE 0 END) AS interaction_labels,
      SUM(CASE WHEN affinity_pkd IS NOT NULL THEN 1 ELSE 0 END) AS affinity_labels
    FROM samples
    """
  ).fetchone()
  total_rows, total_sources, interaction_labels, affinity_labels = row
  print(
    "Dataset audit "
    f"rows={total_rows} sources={total_sources} "
    f"interaction_labels={interaction_labels} affinity_labels={affinity_labels}"
  )


def aggregate(source_specs: Sequence[SourceSpec], out_db: Path):
  out_db.parent.mkdir(parents=True, exist_ok=True)
  con = duckdb.connect(out_db.as_posix())
  try:
    _prepare_db(con)
    for spec in source_specs:
      _insert_source_rows(con, spec)
    total = con.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    print(f"Aggregation complete: {total} rows written to {out_db}")
    _print_dataset_audit(con)
  finally:
    con.close()


def _parse_args():
  parser = argparse.ArgumentParser(description="Aggregate interaction-group datasets into DuckDB.")
  parser.add_argument("--out-db", default="data/aggregated/aggregated.duckdb", help="Output DuckDB path.")
  return parser.parse_args()


def main():
  args = _parse_args()
  aggregate(SOURCE_SPECS, Path(args.out_db))


if __name__ == "__main__":
  main()

"""
Aggregate protein-group interaction datasets into a DuckDB database.

Source loaders are registered in `sources.build_source_specs()` and yield
`contract.InteractionEntry` objects. The aggregator canonicalizes sequence
groups, enforces order-invariant pair uniqueness, and writes the unified
`samples` table consumed by tokenization and training.
"""

import argparse
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import duckdb
import pandas as pd

from contract import SequenceGroup, SourceSpec
from sources import build_source_specs

INSERT_CHUNK_SIZE = 50_000
INSERT_COLUMNS = ["source", "group1", "group2", "interaction_label", "affinity_pkd"]


# Source priority is defined by list order. Earlier sources win if multiple
# sources provide the same canonicalized group pair.
SOURCE_SPECS: List[SourceSpec] = build_source_specs()


def normalize_chain_group(group: SequenceGroup) -> str:
  """Convert one interaction-side group into its canonical serialized form.

  This helper accepts either a pre-delimited string or an iterable of raw
  amino-acid sequences, strips empty items, uppercases sequences, sorts them
  alphabetically, and joins them with `":"`.
  """
  if isinstance(group, str):
    parts = [part.strip().upper() for part in group.split(":") if part.strip()]
  else:
    parts = [str(part).strip().upper() for part in group if str(part).strip()]

  if not parts:
    raise ValueError("Encountered an empty chain group.")
  return ":".join(sorted(parts))


def canonicalize_pair(group1: SequenceGroup, group2: SequenceGroup) -> Tuple[str, str]:
  """Canonicalize an interaction pair into an order-invariant database key."""
  normalized = [normalize_chain_group(group1), normalize_chain_group(group2)]
  normalized.sort()
  return normalized[0], normalized[1]


def affinity_nm_to_pkd(affinity_nm: Optional[float]) -> Optional[float]:
  """Convert a Kd measurement from nanomolar units into pKd space.

  pKd is defined as `-log10(Kd_M)`. Because `Kd_M = Kd_nM * 1e-9`, the
  equivalent conversion from nM is `pKd = 9 - log10(Kd_nM)`.
  """
  if affinity_nm is None:
    return None

  value = float(affinity_nm)
  if value <= 0.0:
    raise ValueError(f"Affinity in nM must be positive, got {value}")
  return 9.0 - math.log10(value)


def _coerce_interaction_label(value: Optional[bool], affinity_nm: Optional[float]) -> Optional[float]:
  """Map a source interaction label into the numeric format used by training."""
  if value is None:
    return 1.0
  return 1.0 if bool(value) else 0.0


def _prepare_db(con: duckdb.DuckDBPyConnection):
  """Recreate the target DuckDB schema from scratch."""
  con.execute("DROP TABLE IF EXISTS samples")
  con.execute(
    """
    CREATE TABLE samples (
      source VARCHAR NOT NULL,
      group1 VARCHAR NOT NULL,
      group2 VARCHAR NOT NULL,
      interaction_label DOUBLE,
      affinity_pkd DOUBLE,
      CONSTRAINT samples_group_pair_unique UNIQUE(group1, group2)
    )
    """
  )


def _flush_chunk(con: duckdb.DuckDBPyConnection, rows: List[tuple]) -> int:
  """Insert one buffered chunk and return the number of rows inserted."""
  if not rows:
    return 0

  df = pd.DataFrame(rows, columns=INSERT_COLUMNS)
  con.register("source_rows", df)
  try:
    before = con.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    con.execute(
      """
      INSERT INTO samples(source, group1, group2, interaction_label, affinity_pkd)
      SELECT source, group1, group2, interaction_label, affinity_pkd
      FROM source_rows
      ON CONFLICT(group1, group2) DO NOTHING
      """
    )
    after = con.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
  finally:
    con.unregister("source_rows")

  return after - before


def _insert_source_rows(con: duckdb.DuckDBPyConnection, spec: SourceSpec):
  """Normalize and insert all rows produced by one registered source.

  Rows are canonicalized and flushed in fixed-size chunks so large sources do
  not need to be materialized fully in memory. Duplicates are skipped by the
  table uniqueness constraint.
  """
  buffer: List[tuple] = []
  valid_rows = 0
  inserted = 0
  skipped_invalid = 0

  for entry in spec.loader():
    try:
      group1, group2 = canonicalize_pair(entry.group1, entry.group2)
      affinity_nm = None if entry.affinity_nm is None else float(entry.affinity_nm)
      interaction_label = _coerce_interaction_label(entry.interaction_label, affinity_nm)
      affinity_pkd = affinity_nm_to_pkd(affinity_nm)
    except Exception as exc:
      skipped_invalid += 1
      print(f"Source={spec.name} skipped_invalid_entry error={exc}", flush=True)
      continue

    buffer.append((entry.source or spec.name, group1, group2, interaction_label, affinity_pkd))
    valid_rows += 1

    if len(buffer) >= INSERT_CHUNK_SIZE:
      inserted += _flush_chunk(con, buffer)
      buffer.clear()
      skipped_duplicates = valid_rows - inserted
      print(
        f"Source={spec.name} processed={valid_rows} inserted={inserted} skipped_duplicate={skipped_duplicates}",
        flush=True,
      )

  inserted += _flush_chunk(con, buffer)
  skipped_duplicates = valid_rows - inserted
  print(
    f"Source={spec.name} inserted={inserted} "
    f"skipped_invalid={skipped_invalid} skipped_duplicate={skipped_duplicates}",
    flush=True,
  )


def _print_dataset_audit(con: duckdb.DuckDBPyConnection):
  """Print a compact summary of the aggregated dataset."""
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
    f"interaction_labels={interaction_labels} affinity_labels={affinity_labels}",
    flush=True,
  )


def aggregate(source_specs: Sequence[SourceSpec], out_db: Path):
  """Build the aggregated DuckDB file from the registered source list."""
  out_db.parent.mkdir(parents=True, exist_ok=True)
  con = duckdb.connect(out_db.as_posix())
  try:
    _prepare_db(con)
    for spec in source_specs:
      _insert_source_rows(con, spec)
    total = con.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    print(f"Aggregation complete: {total} rows written to {out_db}", flush=True)
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

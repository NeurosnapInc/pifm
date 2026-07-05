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
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import duckdb
import pandas as pd

from sources.intact import iter_intact_entries
from sources.ppb_affinity import iter_ppb_affinity_entries
from sources.skempi import iter_skempi_entries
from sources.source_types import InteractionEntry, SequenceGroup, SourceSpec


SOURCE_INSERT_BATCH_SIZE = 5_000


def _empty_loader() -> Iterable[InteractionEntry]:
  """Provide a no-op source loader placeholder.

  This is only here so the file has a valid example loader shape before any
  real data sources are registered. It returns an empty iterable and does not
  perform any I/O or validation.
  """
  return []


# Source priority is defined by list order. Earlier sources win if multiple
# sources provide the same canonicalized group pair.
SOURCE_SPECS: List[SourceSpec] = [
  SourceSpec(name="ppb_affinity_filtered", loader=iter_ppb_affinity_entries),
  SourceSpec(name="skempi_v2", loader=iter_skempi_entries),
  SourceSpec(name="intact", loader=iter_intact_entries),
]


def normalize_chain_group(group: SequenceGroup) -> str:
  """Convert one interaction-side group into its canonical serialized form.

  The aggregation layer needs a stable representation for each side of a
  protein complex so deduplication does not depend on how an upstream source
  happened to order chains. This helper accepts either a pre-delimited string
  or an iterable of raw amino-acid sequences, strips empty items, uppercases
  all sequences, sorts them alphabetically, and joins them with `":"`.

  The result is the only group representation that should be written to the
  database or used for downstream comparisons. If the incoming value contains
  no usable sequences after cleanup, the row is treated as invalid and the
  caller should skip it.
  """
  if isinstance(group, str):
    parts = [part.strip().upper() for part in group.split(":") if part.strip()]
  else:
    parts = [str(part).strip().upper() for part in group if str(part).strip()]

  if not parts:
    raise ValueError("Encountered an empty chain group.")

  return ":".join(sorted(parts))


def canonicalize_pair(group1: SequenceGroup, group2: SequenceGroup) -> Tuple[str, str]:
  """Canonicalize a two-sided interaction pair into an order-invariant key.

  The model still distinguishes the partition between the two sides of the
  interaction, so each group is preserved as its own normalized unit. What is
  intentionally removed is the arbitrary top-level ordering between those two
  units. After each side is normalized independently, the pair itself is sorted
  so that `(A:B, C:D)` and `(C:D, A:B)` become the same canonical database key.

  This preserves bipartite structure while preventing duplicate rows caused
  purely by source-specific left/right ordering conventions.
  """
  normalized = [normalize_chain_group(group1), normalize_chain_group(group2)]
  normalized.sort()
  return normalized[0], normalized[1]


def affinity_nm_to_pkd(affinity_nm: Optional[float]) -> Optional[float]:
  """Convert an affinity measurement from nanomolar units into pKd space.

  The raw sources may provide affinity in nM, but the training pipeline is
  expected to operate on a log-scale target because it is numerically better
  behaved and aligns more naturally with standard biochemical conventions.
  This helper applies the identity `pKd = 9 - log10(Kd_nM)`, which is
  equivalent to `-log10(Kd_M)`.

  Missing affinity values remain missing so the regression task can be masked
  cleanly downstream. Non-positive affinity values are rejected because the
  logarithm would be invalid and such rows indicate a broken source record.
  """
  if affinity_nm is None:
    return None

  value = float(affinity_nm)
  if value <= 0.0:
    raise ValueError(f"Affinity in nM must be positive, got {value}")
  return 9.0 - math.log10(value)


def _coerce_interaction_label(value: Optional[bool], affinity_nm: Optional[float]) -> Optional[float]:
  """Map a source interaction label into the numeric format used by training.

  The output is stored as a float so it is immediately compatible with the rest
  of the aggregation and tokenization flow, which treats labels generically
  before task-specific casting. Rows that do not explicitly provide a binary
  label are assumed to be positive observations because curated source entries
  represent known interactions unless a source explicitly contributes negatives.
  """
  if value is None:
    return 1.0
  return 1.0 if bool(value) else 0.0


def _prepare_db(con: duckdb.DuckDBPyConnection):
  """Recreate the target DuckDB schema from scratch.

  Aggregation is designed to be deterministic and rerunnable, so this function
  drops any existing `samples` table and creates a fresh one with the canonical
  columns expected by the rest of the project. The uniqueness constraint is
  intentionally applied to `(group1, group2)` only, after canonicalization, so
  duplicate source rows cannot survive just because their original order
  differed.
  """
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
  """Normalize and insert all rows produced by one registered source.

  Each yielded entry is validated, canonicalized, and converted into the
  database schema before any insertion happens. Invalid rows are skipped with a
  diagnostic message rather than aborting the whole aggregation run. Rows are
  flushed in batches so large sources, such as IntAct, do not need to be fully
  materialized in memory before insertion.

  Because sources are processed in the order they appear in `SOURCE_SPECS`,
  earlier sources take precedence when multiple datasets contain the same
  canonicalized pair. This gives the registry a simple, explicit source-quality
  priority mechanism without needing per-row merge logic.
  """
  pending_rows = []
  skipped_invalid = 0
  skipped_duplicates = 0
  inserted_total = 0
  processed_total = 0

  def flush_pending_rows():
    nonlocal inserted_total, skipped_duplicates
    if not pending_rows:
      return

    df = pd.DataFrame(
      pending_rows,
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
    inserted_total += inserted
    skipped_duplicates += len(pending_rows) - inserted
    pending_rows.clear()

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

    pending_rows.append(
      (
        entry.source or spec.name,
        group1,
        group2,
        interaction_label,
        affinity_nm,
        affinity_pkd,
      )
    )
    processed_total += 1

    if len(pending_rows) >= SOURCE_INSERT_BATCH_SIZE:
      flush_pending_rows()
      print(
        f"Source={spec.name} processed={processed_total} "
        f"inserted={inserted_total} skipped_duplicate={skipped_duplicates}",
        flush=True,
      )

  flush_pending_rows()
  print(
    f"Source={spec.name} inserted={inserted_total} "
    f"skipped_invalid={skipped_invalid} skipped_duplicate={skipped_duplicates}",
    flush=True,
  )


def _print_dataset_audit(con: duckdb.DuckDBPyConnection):
  """Print a high-level audit summary for the aggregated dataset.

  The goal is not exhaustive reporting, just a fast sanity check after an
  aggregation run. The summary makes it easy to confirm that rows were loaded,
  multiple sources were seen when expected, and both interaction and affinity
  supervision are present at non-zero counts.
  """
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
  """Build the aggregated DuckDB file from the registered source list.

  This is the main orchestration entrypoint for data aggregation. It creates the
  output directory if needed, recreates the target schema, processes each source
  in registry order, and finally prints a compact audit of the resulting table.
  The output database is self-contained and intended to be the single handoff
  artifact consumed by tokenization and downstream training scripts.
  """
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
  """Parse command-line options for the aggregation CLI."""
  parser = argparse.ArgumentParser(description="Aggregate interaction-group datasets into DuckDB.")
  parser.add_argument(
    "--out-db",
    default="data/aggregated/aggregated.duckdb",
    help="Output DuckDB path.",
  )
  return parser.parse_args()


def main():
  """Run the aggregation CLI using the currently registered sources."""
  args = _parse_args()
  aggregate(SOURCE_SPECS, Path(args.out_db))


if __name__ == "__main__":
  main()

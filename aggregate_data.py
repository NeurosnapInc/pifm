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
from typing import List, Optional, Sequence, Tuple

import duckdb
import pandas as pd

# InteractionEntry / SourceSpec live in ``contract`` so source loaders can import
# them without a circular dependency on this module. Re-exported here so existing
# imports (``from aggregate_data import InteractionEntry``) keep working.
from contract import InteractionEntry, SequenceGroup, SourceSpec  # noqa: F401
from sources import build_source_specs


# Source priority is defined by list order. Earlier sources win if multiple sources
# provide the same canonicalized group pair. Registration lives in ``sources``.
SOURCE_SPECS: List[SourceSpec] = build_source_specs()


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
  before task-specific casting. At the moment, rows that do not explicitly
  provide a binary label are assumed to be positive observations. That matches
  the current project assumption that curated source entries represent known
  interactions unless a source explicitly contributes negatives.

  If that assumption changes later, this is the single place where the default
  interaction-label policy should be tightened.
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


_INSERT_COLUMNS = ["source", "group1", "group2", "interaction_label", "affinity_nm", "affinity_pkd"]

# Rows are inserted in bounded chunks so a single large source (e.g. IntAct,
# which yields millions of rows) never materializes its full row set in memory.
INSERT_CHUNK_SIZE = 50_000


def _flush_chunk(con: duckdb.DuckDBPyConnection, buffer: List[tuple]) -> int:
  """Insert one buffered chunk of canonicalized rows, returning rows inserted.

  Deduplication against already-present pairs is handled by the table's
  `ON CONFLICT(group1, group2) DO NOTHING` policy, so chunk boundaries do not
  affect the final result: a pair seen in an earlier chunk (or an earlier,
  higher-priority source) is skipped here. The number inserted is measured as
  the change in table size, which naturally accounts for both cross-chunk and
  intra-chunk duplicates.
  """
  if not buffer:
    return 0

  df = pd.DataFrame(buffer, columns=_INSERT_COLUMNS)
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

  return after - before


def _insert_source_rows(con: duckdb.DuckDBPyConnection, spec: SourceSpec):
  """Normalize and insert all rows produced by one registered source.

  Each yielded entry is validated, canonicalized, and converted into the
  database schema, then buffered and flushed in fixed-size chunks (see
  `INSERT_CHUNK_SIZE`) with an `ON CONFLICT DO NOTHING` policy on the canonical
  pair key. Streaming in chunks keeps memory bounded regardless of source size.
  Invalid rows are skipped with a diagnostic message rather than aborting the
  whole aggregation run.

  Because sources are processed in the order they appear in `SOURCE_SPECS`,
  earlier sources take precedence when multiple datasets contain the same
  canonicalized pair. This gives the registry a simple, explicit source-quality
  priority mechanism without needing per-row merge logic.
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
      print(f"Source={spec.name} skipped_invalid_entry error={exc}")
      continue

    buffer.append((entry.source or spec.name, group1, group2, interaction_label, affinity_nm, affinity_pkd))
    valid_rows += 1

    if len(buffer) >= INSERT_CHUNK_SIZE:
      inserted += _flush_chunk(con, buffer)
      buffer.clear()

  inserted += _flush_chunk(con, buffer)

  skipped_duplicates = valid_rows - inserted
  print(f"Source={spec.name} inserted={inserted} skipped_invalid={skipped_invalid} skipped_duplicate={skipped_duplicates}")


def _print_dataset_audit(con: duckdb.DuckDBPyConnection):
  """Print a high-level audit summary for the aggregated dataset.

  The goal is not exhaustive reporting, just a fast sanity check after an
  aggregation run. The summary makes it easy to confirm that rows were loaded,
  multiple sources were seen when expected, and both interaction and affinity
  supervision are present at nonzero counts.
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
    f"interaction_labels={interaction_labels} affinity_labels={affinity_labels}"
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
    print(f"Aggregation complete: {total} rows written to {out_db}")
    _print_dataset_audit(con)
  finally:
    con.close()


def _parse_args():
  """Parse command-line options for the aggregation CLI.

  The current interface is intentionally minimal because source registration is
  code-driven rather than command-line driven. At the moment the only runtime
  option is the destination path for the aggregated DuckDB file.
  """
  parser = argparse.ArgumentParser(description="Aggregate interaction-group datasets into DuckDB.")
  parser.add_argument("--out-db", default="data/aggregated/aggregated.duckdb", help="Output DuckDB path.")
  return parser.parse_args()


def main():
  """Run the aggregation CLI using the currently registered sources.

  This thin wrapper exists so the module can be used both as a script and as an
  importable library entrypoint without duplicating setup logic.
  """
  args = _parse_args()
  aggregate(SOURCE_SPECS, Path(args.out_db))


if __name__ == "__main__":
  main()

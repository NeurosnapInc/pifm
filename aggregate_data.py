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
from sources.source_types import InteractionEntry, SequenceGroup, SourceSpec

# A group of chains may be a single colon-delimited string or a list of sequences
SequenceGroup = Union[str, Sequence[str]]
SourceFactory = Callable[[], Iterable["InteractionEntry"]]


@dataclass(frozen=True)
class InteractionEntry:
    """Container for one protein–protein interaction entry.

    Each entry corresponds to a pair of protein groups (chains) and may include
    an affinity measurement in nanomolar units as well as an optional binary
    interaction label. The `source` field identifies the originating dataset.
    """

    source: str
    group1: SequenceGroup
    group2: SequenceGroup
    affinity_nm: Optional[float] = None
    interaction_label: Optional[bool] = None


@dataclass(frozen=True)
class SourceSpec:
    """Specification for registering a new data source.

SOURCE_INSERT_BATCH_SIZE = 5_000

    The name is a human‑readable identifier and the loader is a callable that
    yields `InteractionEntry` objects for each record in the dataset.
    """

    name: str
    loader: SourceFactory


def _empty_loader() -> Iterable[InteractionEntry]:
    """Provide a no‑op source loader placeholder.

    This is only here so the file has a valid example loader shape before any
    real data sources are registered. It returns an empty iterable and does
    not perform any I/O or validation.
    """
    return []


def iter_ppb_affinity_rows() -> Iterable[InteractionEntry]:
    """Yield interaction entries from the PPB‑Affinity filtered dataset.

    The [PPB‑Affinity](https://huggingface.co/datasets/proteinea/ppb_affinity)
    dataset contains protein–protein complexes with binding affinity
    measurements. To include these data in the aggregation pipeline, download
    the `filtered.csv` file from the dataset repository and save it to
    ``data/raw/ppb_affinity_filtered.csv``.  The CSV contains columns
    ``Ligand Sequences``, ``Receptor Sequences`` and ``KD(M)``.  Each row is
    converted into an :class:`InteractionEntry`, where the ligand and receptor
    sequence strings are split on commas (and stripped of whitespace and
    embedded newlines).  The KD values provided in molar units are converted
    into nanomolar (nM) units by multiplying by 1e9.  Rows without sequence
    information or with unparsable affinity values are silently skipped.

    Yields
    ------
    Iterable[InteractionEntry]
        A generator over cleaned and normalized interaction entries.
    """
    import csv  # local import so that csv is only loaded when needed

    csv_path = Path("data/raw/ppb_affinity_filtered.csv")
    if not csv_path.exists():
        # Inform the caller that the dataset is missing.  Returning an empty
        # list preserves the iterable contract and allows the aggregation
        # pipeline to proceed with other sources.
        print(
            f"Warning: expected PPB‑Affinity data at {csv_path}. "
            "Download the dataset and place it in data/raw."
        )
        return []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Retrieve ligand and receptor sequence columns.  Accept both
            # capitalized and lowercase variants to be forgiving of casing.
            lig_seq_col = row.get("Ligand Sequences") or row.get("ligand sequences")
            rec_seq_col = row.get("Receptor Sequences") or row.get("receptor sequences")
            if not lig_seq_col or not rec_seq_col:
                continue  # skip incomplete rows

            # Normalize sequences: remove newlines, split on comma and strip
            ligand_seqs = [
                seq.strip()
                for seq in lig_seq_col.replace("\n", ",").split(",")
                if seq.strip()
            ]
            receptor_seqs = [
                seq.strip()
                for seq in rec_seq_col.replace("\n", ",").split(",")
                if seq.strip()
            ]

            # Parse KD in molar units and convert to nanomolar.  If the value
            # cannot be parsed, leave affinity as None.  Some rows may be
            # missing affinities entirely.
            affinity_nm: Optional[float] = None
            kd_m_str = row.get("KD(M)") or row.get("kd(m)")
            if kd_m_str:
                try:
                    kd_m = float(kd_m_str)
                    affinity_nm = kd_m * 1e9
                except Exception:
                    affinity_nm = None

            # Create and yield the interaction entry.  PPB‑Affinity does not
            # include explicit binary interaction labels, so the label is set
            # to None (which defaults to a positive observation downstream).
            yield InteractionEntry(
                source="ppb_affinity_filtered",
                group1=ligand_seqs,
                group2=receptor_seqs,
                affinity_nm=affinity_nm,
                interaction_label=None,
            )


def iter_skempi_rows() -> Iterable[InteractionEntry]:
    """Yield interaction entries from the SKEMPI v2.0 mutation dataset.

    The [SKEMPI](https://life.bsc.es/pid/skempi2/) database is a curated
    collection of protein–protein complexes with experimentally measured
    changes in binding affinity upon mutation. Each entry provides
    thermodynamic and kinetic parameters for a specific mutation relative
    to a wild‐type complex. To include these data in the aggregation
    pipeline, download the `skempi_v2.csv` file and the associated
    `SKEMPI2_PDBs.tgz` archive from the SKEMPI website and place them
    under ``data/raw``. The CSV file contains many columns, including
    identifiers for the interacting partners (PDB chains), affinity
    measurements for the wild‐type (`affinity_wt`) and mutant
    (`affinity_mut`) complexes, and mutation annotations.

    This loader focuses on the wild‑type affinity values and does not
    attempt to apply mutations or parse PDB files. Instead, it treats
    the wild‑type complex as a protein–protein interaction where the
    interacting partners are represented by the PDB chains listed in the
    ``Protein"`` column (e.g., ``1A2K_A_B``). Because SKEMPI does not
    include raw sequences in the CSV, this loader canonicalizes the
    chain identifiers into the group strings without sequence data.
    Affinities are parsed in molar units (M) and converted to nM by
    multiplying by ``1e9``. Rows lacking a valid wild‑type affinity
    measurement are skipped. Mutant affinities are ignored because they
    represent altered complexes.

    Note that a more sophisticated integration of SKEMPI would apply
    mutations to the wild‑type sequences extracted from the provided
    PDB files. Implementing such functionality requires external
    sequence extraction and is beyond the scope of this simple loader.

    Yields
    ------
    Iterable[InteractionEntry]
        A generator over interaction entries derived from SKEMPI v2.0.
    """
    import csv  # local import to avoid unnecessary dependency during import

    csv_path = Path("data/raw/skempi_v2.csv")
    if not csv_path.exists():
        print(
            f"Warning: expected SKEMPI v2 data at {csv_path}. "
            "Download skempi_v2.csv from the SKEMPI website and place it in data/raw."
        )
        return []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract the protein complex identifier.  The SKEMPI CSV
            # contains a column named 'Protein' (e.g., '1A2K_A_B') which
            # encodes the PDB ID and interacting chains separated by
            # underscores.  We split this into two group identifiers.
            protein = row.get("Protein") or row.get("protein")
            if not protein or "_" not in protein:
                continue
            parts = protein.split("_")
            # Expect exactly three parts: PDBID, chain1, chain2.  If the
            # format differs, skip the row.
            if len(parts) < 3:
                continue
            _, chain1, chain2 = parts[:3]
            group1 = chain1
            group2 = chain2

            # Parse the wild‑type affinity.  SKEMPI records affinities
            # in molar units under 'affinity_wt'.  Convert to nM.
            affinity_nm: Optional[float] = None
            kd_str = row.get("affinity_wt") or row.get("affinity_wt (M)") or row.get("affinity_wt(M)")
            if kd_str:
                try:
                    kd_m = float(kd_str)
                    if kd_m > 0.0:
                        affinity_nm = kd_m * 1e9
                except Exception:
                    affinity_nm = None

            # Skip rows without affinity values
            if affinity_nm is None:
                continue

            yield InteractionEntry(
                source="skempi_v2",
                group1=group1,
                group2=group2,
                affinity_nm=affinity_nm,
                interaction_label=None,
            )


# Source priority is defined by list order.  Earlier sources win if multiple
# sources provide the same canonicalized group pair.
SOURCE_SPECS: List[SourceSpec] = [
    # Register the PPB‑Affinity filtered dataset.  Additional sources may be
    # appended to this list in order of decreasing quality or priority.
    SourceSpec(name="ppb_affinity_filtered", loader=iter_ppb_affinity_rows),
    # Register the SKEMPI v2.0 dataset.  This source provides mutation‑aware
    # affinity measurements for protein–protein complexes.  See
    # `iter_skempi_rows` for parsing details.
    SourceSpec(name="skempi_v2", loader=iter_skempi_rows),
    # Example for future datasets:
    # SourceSpec(name="bindingdb_curated", loader=iter_bindingdb_rows),
  SourceSpec(name="intact", loader=lambda: iter_intact_entries()),
]


def normalize_chain_group(group: SequenceGroup) -> str:
    """Convert one interaction‑side group into its canonical serialized form.

    The aggregation layer needs a stable representation for each side of a
    protein complex so deduplication does not depend on how an upstream source
    happened to order chains.  This helper accepts either a pre‑delimited
    string or an iterable of raw amino‑acid sequences, strips empty items,
    uppercases all sequences, sorts them alphabetically, and joins them with
    `":"`.

    The result is the only group representation that should be written to the
    database or used for downstream comparisons.  If the incoming value
    contains no usable sequences after cleanup, the row is treated as invalid
    and the caller should skip it.
    """
    if isinstance(group, str):
        parts = [part.strip().upper() for part in group.split(":") if part.strip()]
    else:
        parts = [str(part).strip().upper() for part in group if str(part).strip()]

    if not parts:
        raise ValueError("Encountered an empty chain group.")

    return ":".join(sorted(parts))


def canonicalize_pair(group1: SequenceGroup, group2: SequenceGroup) -> Tuple[str, str]:
    """Canonicalize a two‑sided interaction pair into an order‑invariant key.

    The model still distinguishes the partition between the two sides of the
    interaction, so each group is preserved as its own normalized unit.  What is
    intentionally removed is the arbitrary top‑level ordering between those two
    units.  After each side is normalized independently, the pair itself is
    sorted so that ``(A:B, C:D)`` and ``(C:D, A:B)`` become the same
    canonical database key.

    This preserves bipartite structure while preventing duplicate rows caused
    purely by source‑specific left/right ordering conventions.
    """
    normalized = [normalize_chain_group(group1), normalize_chain_group(group2)]
    normalized.sort()
    return normalized[0], normalized[1]


def affinity_nm_to_pkd(affinity_nm: Optional[float]) -> Optional[float]:
    """Convert an affinity measurement from nanomolar units into pKd space.

    The raw sources may provide affinity in nM, but the training pipeline is
    expected to operate on a log‑scale target because it is numerically better
    behaved and aligns more naturally with standard biochemical conventions.
    This helper applies the identity `pKd = 9 - log10(Kd_nM)`, which is
    equivalent to `-log10(Kd_M)`.

    Missing affinity values remain missing so the regression task can be masked
    cleanly downstream.  Non‑positive affinity values are rejected because the
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
    before task‑specific casting.  At the moment, rows that do not explicitly
    provide a binary label are assumed to be positive observations.  That
    matches the current project assumption that curated source entries represent
    known interactions unless a source explicitly contributes negatives.

    If that assumption changes later, this is the single place where the
    default interaction‑label policy should be tightened.
    """
    if value is None:
        return 1.0
    return 1.0 if bool(value) else 0.0


def _prepare_db(con: duckdb.DuckDBPyConnection):
    """Recreate the target DuckDB schema from scratch.

    Aggregation is designed to be deterministic and rerunnable, so this
    function drops any existing ``samples`` table and creates a fresh one
    with the canonical columns expected by the rest of the project.  The
    uniqueness constraint is intentionally applied to ``(group1, group2)`` only,
    after canonicalization, so duplicate source rows cannot survive just
    because their original order differed.
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
    database schema before any insertion happens.  Invalid rows are skipped
    with a diagnostic message rather than aborting the whole aggregation run.
    After the rows are materialized into a temporary DataFrame, they are
    inserted with an ``ON CONFLICT DO NOTHING`` policy on the canonical pair
    key.

    Because sources are processed in the order they appear in
    :data:`SOURCE_SPECS`, earlier sources take precedence when multiple
    datasets contain the same canonicalized pair.  This gives the registry a
    simple, explicit source‑quality priority mechanism without needing per‑row
    merge logic.
    """
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
        print(
            f"Source={spec.name} inserted=0 skipped_invalid={skipped_invalid} "
            f"skipped_duplicate={skipped_duplicates}"
        )
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
    print(
        f"Source={spec.name} inserted={inserted} "
        f"skipped_invalid={skipped_invalid} skipped_duplicate={skipped_duplicates}"
    )
  """Normalize and insert all rows produced by one registered source.

  Each yielded entry is validated, canonicalized, and converted into the
  database schema before any insertion happens. Invalid rows are skipped with a
  diagnostic message rather than aborting the whole aggregation run. After the
  rows are materialized into a temporary DataFrame, they are inserted with an
  `ON CONFLICT DO NOTHING` policy on the canonical pair key.

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
      print(f"Source={spec.name} processed={processed_total} inserted={inserted_total} skipped_duplicate={skipped_duplicates}", flush=True)

  flush_pending_rows()
  print(f"Source={spec.name} inserted={inserted_total} skipped_invalid={skipped_invalid} skipped_duplicate={skipped_duplicates}", flush=True)


def _print_dataset_audit(con: duckdb.DuckDBPyConnection):
    """Print a high‑level audit summary for the aggregated dataset.

    The goal is not exhaustive reporting, just a fast sanity check after an
    aggregation run.  The summary makes it easy to confirm that rows were
    loaded, multiple sources were seen when expected, and both interaction and
    affinity supervision are present at non‑zero counts.
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

    This is the main orchestration entrypoint for data aggregation.  It creates the
    output directory if needed, recreates the target schema, processes each
    source in registry order, and finally prints a compact audit of the resulting
    table.  The output database is self‑contained and intended to be the single
    handoff artifact consumed by tokenization and downstream training scripts.
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
    """Parse command‑line options for the aggregation CLI.

    The current interface is intentionally minimal because source registration is
    code‑driven rather than command‑line driven.  At the moment the only
    runtime option is the destination path for the aggregated DuckDB file.
    """
    parser = argparse.ArgumentParser(
        description="Aggregate interaction‑group datasets into DuckDB."
    )
    parser.add_argument(
        "--out-db",
        default="data/aggregated/aggregated.duckdb",
        help="Output DuckDB path.",
    )
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

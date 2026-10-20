"""Load the plain-sequence, experimentally supported subset of SNOOPPI.

Only the positive and negative Parquet partitions are needed, downloaded from
https://huggingface.co/datasets/ChatterjeeLab/SNOOPPI into data/raw/snooppi/.
Downloads are pinned to SNOOPPI_REVISION below. This loader is local-only: it
never fetches the unknown partition or downloads anything during aggregation.

Modified-sequence markup is rejected rather than stripped. Negative examples
must have direct IntAct negative evidence or mutation/PTM-derived experimental
evidence; inferred binding-region deletion/shuffle/substitution negatives are
excluded even when a row also carries other evidence. Actual mutant sequences
are preserved, never replaced with wild-type sequences.

The registry appends this source after existing sources. The aggregator's
order-invariant sequence-pair uniqueness therefore discards duplicates without
overwriting existing labels, including disagreements with SNOOPPI.
"""

import re
from collections import Counter
from pathlib import Path
from typing import Iterator, Mapping, Optional

import duckdb

from contract import InteractionEntry


SNOOPPI_DIR = Path("data/raw/snooppi")
SNOOPPI_REVISION = "45e4f02204a135eeb26158e189bb253d0e677679"
SNOOPPI_URL = f"https://huggingface.co/datasets/ChatterjeeLab/SNOOPPI/resolve/{SNOOPPI_REVISION}/data"
# Standard residues plus ambiguity/rare-residue codes supported by preprocessing.
PLAIN_SEQUENCE = re.compile(r"[ACDEFGHIKLMNPQRSTVWYBXZOU]+")
COLUMNS = (
  "SNOOPPI_final_label", "partner_A_sequence", "partner_B_sequence",
  "partner_A_mol_type", "partner_B_mol_type", "seq_sort_origin_categories",
  "has_binding_site_relationship", "feature_relationship_types",
  "feature_relationship_summary", "exact_sequence_pair_intact_label_summary",
  "intact_interaction_ids",
)


def _text(value: Optional[str]) -> str:
  """Normalize nullable Parquet text fields without inventing missing values."""
  return "" if value is None else str(value).strip()


def _categories(value: Optional[str]):
  """Parse the pipe-delimited evidence or molecule-type fields in this export."""
  return {part.strip().lower() for part in _text(value).split("|") if part.strip()}


def _negative_exclusion(row: Mapping[str, str]) -> Optional[str]:
  """Explain why a negative is unsupported, or accept experimental provenance.

  Binding-region-derived negatives are synthetic, unlike annotated tested
  mutations/PTMs. Reject mixed binding-region records conservatively. Direct
  negatives must explicitly agree with IntAct's negative label; feature-derived
  negatives must identify the corresponding modified state as negative.
  """
  origins = _categories(row.get("seq_sort_origin_categories"))
  if (_text(row.get("has_binding_site_relationship")).lower() == "true"
      or "binding_site" in _categories(row.get("feature_relationship_types"))
      or "derived_from_binding_site_feature" in origins):
    return "inferred_negative"
  if not _text(row.get("intact_interaction_ids")):
    return "unsupported_negative"
  if ("exact_seq_sort_seen_in_original_intact" in origins
      and _text(row.get("exact_sequence_pair_intact_label_summary")).lower() == "negative"):
    return None
  for kind in ("mutation", "ptm"):
    if f"derived_from_{kind}_feature" not in origins:
      continue
    for evidence in _text(row.get("feature_relationship_summary")).split("|"):
      fields = {key.strip(): value.strip() for field in evidence.split(";") if "=" in field
                for key, value in [field.split("=", 1)]}
      if fields.get("type") == kind and fields.get("modified_label") == "negative":
        return None
  return "unsupported_negative"


def _row_to_entry(row: Mapping[str, str], expected_label: str):
  """Return an entry or an audit reason; only literal amino-acid inputs survive."""
  if expected_label not in {"positive", "negative"} or _text(row.get("SNOOPPI_final_label")).lower() != expected_label:
    return None, "unexpected_label"
  for partner in ("A", "B"):
    molecules = _categories(row.get(f"partner_{partner}_mol_type"))
    if not molecules or not molecules <= {"protein", "peptide"}:
      return None, "non_protein"
  # Reject annotation markup instead of trying to turn it into residue tokens.
  sequences = [_text(row.get(f"partner_{partner}_sequence")) for partner in ("A", "B")]
  if any(PLAIN_SEQUENCE.fullmatch(sequence) is None for sequence in sequences):
    return None, "non_plain_sequence"
  if expected_label == "negative":
    reason = _negative_exclusion(row)
    if reason:
      return None, reason
  return InteractionEntry(
    source="snooppi", group1=[sequences[0]], group2=[sequences[1]],
    interaction_label=expected_label == "positive",
  ), None


def iter_snooppi(raw_dir: Path = SNOOPPI_DIR) -> Iterator[InteractionEntry]:
  """Stream local labeled partitions with conservative filtering and audit counts.

  No species restriction is imposed. Missing files are reported individually;
  available partitions can still be loaded. Use the printed pinned URLs to
  obtain missing files without downloading a complete Hugging Face snapshot.
  """
  for label in ("positive", "negative"):
    filename = f"{label}-00000-of-00001.parquet"
    path = raw_dir / filename
    if not path.is_file():
      print(f"Source=snooppi partition={label} skipped: missing {path}. "
            f"Download with: wget -P {raw_dir} {SNOOPPI_URL}/{filename}", flush=True)
      continue
    counts = Counter()
    with duckdb.connect() as con:
      columns = ", ".join(f'"{column}"' for column in COLUMNS)
      cursor = con.execute(f"SELECT {columns} FROM read_parquet(?)", [str(path)])
      while True:
        rows = cursor.fetchmany(2048)
        if not rows:
          break
        for values in rows:
          counts["rows"] += 1
          entry, reason = _row_to_entry(dict(zip(COLUMNS, values)), label)
          if reason:
            counts[f"skipped_{reason}"] += 1
            continue
          counts["accepted"] += 1
          yield entry
    print(f"Source=snooppi partition={label} " + " ".join(f"{key}={value}" for key, value in sorted(counts.items())), flush=True)

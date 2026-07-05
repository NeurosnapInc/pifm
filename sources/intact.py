"""
IntAct -- curated, experimentally observed molecular interactions (EMBL-EBI).

IntAct is distributed in PSI-MITAB format (one interaction per tab-separated
line). We keep only physical/direct protein-protein interactions between two
UniProt accessions, optionally gated by the IntAct MI-score, and resolve each
accession to a sequence via ``data/raw/uniprot/`` (see README).

Download:
  data/raw/intact/intact.txt
"""

from typing import Iterator, Optional

from contract import InteractionEntry

from ._common import RAW_DIR, hint_missing, open_text, resolve_uniprot, uniprot_sequences

INTACT_FILE = RAW_DIR / "intact" / "intact.txt"

# Interaction-type MI codes accepted as physical evidence:
#   MI:0915 physical association, MI:0407 direct interaction, MI:0914 association.
PHYSICAL_MI_CODES = ("MI:0915", "MI:0407", "MI:0914")

# Minimum IntAct MI-score to accept when a confidence value is present.
MIN_MISCORE = 0.40

# Column indices in the PSI-MITAB 2.5+ layout used by IntAct.
_COL_ID_A = 0
_COL_ID_B = 1
_COL_INTERACTION_TYPE = 11
_COL_CONFIDENCE = 14


def _extract_uniprot(field: str) -> Optional[str]:
  """Return the UniProt accession from a MITAB identifier cell, if any."""
  for token in field.split("|"):
    token = token.strip()
    if token.startswith("uniprotkb:"):
      return token.split(":", 1)[1].strip().strip('"')
  return None


def _mi_score(confidence_field: str) -> Optional[float]:
  """Return the IntAct MI-score from a MITAB confidence cell, if present."""
  for token in confidence_field.split("|"):
    token = token.strip()
    if token.startswith("intact-miscore:"):
      try:
        return float(token.split(":", 1)[1])
      except ValueError:
        return None
  return None


def iter_intact() -> Iterator[InteractionEntry]:
  if not INTACT_FILE.is_file():
    hint_missing(
      "intact",
      INTACT_FILE,
      "Download intact.txt from https://ftp.ebi.ac.uk/pub/databases/intact/current/psimitab/intact.txt",
    )
    return

  sequences = uniprot_sequences()
  resolved = 0
  unresolved = 0
  filtered = 0

  with open_text(INTACT_FILE) as handle:
    for line in handle:
      if line.startswith("#"):  # MITAB header row
        continue
      columns = line.rstrip("\n").split("\t")
      if len(columns) <= _COL_CONFIDENCE:
        continue

      accession_a = _extract_uniprot(columns[_COL_ID_A])
      accession_b = _extract_uniprot(columns[_COL_ID_B])
      if not accession_a or not accession_b:
        filtered += 1
        continue

      interaction_type = columns[_COL_INTERACTION_TYPE]
      if not any(code in interaction_type for code in PHYSICAL_MI_CODES):
        filtered += 1
        continue

      score = _mi_score(columns[_COL_CONFIDENCE])
      if score is not None and score < MIN_MISCORE:
        filtered += 1
        continue

      seq_a = resolve_uniprot(sequences, accession_a)
      seq_b = resolve_uniprot(sequences, accession_b)
      if not seq_a or not seq_b:
        unresolved += 1
        continue

      resolved += 1
      yield InteractionEntry(
        source="intact",
        group1=seq_a,
        group2=seq_b,
        interaction_label=True,
      )

  print(f"Source=intact resolved_pairs={resolved} unresolved_pairs={unresolved} filtered={filtered}")
"""IntAct bulk archive source loader.

The local archive is configured by `config.INTACT_ARCHIVE_PATH` and is expected
to contain the MITAB positive/negative exports plus the bundled IntAct FASTA
sequence file. See `README.md` for download instructions.
"""

import csv
import re
from typing import Dict, Iterable, Iterator, List, Optional, Sequence
from zipfile import ZipFile

import pandas as pd

from config import (
  INTACT_ARCHIVE_PATH,
  INTACT_INTERACTION_TYPES,
  INTACT_INTERACTOR_TYPES,
  INTACT_SPECIES_TAXID,
)
from sources.source_types import InteractionEntry


INTACT_POSITIVE_MITAB_PATH = "psimitab/intact.txt"
INTACT_NEGATIVE_MITAB_PATH = "psimitab/intact_negative.txt"
INTACT_FASTA_PATH = "various/intact.fasta"
FASTA_HEADER_PATTERN = re.compile(r"^>INTACT:(\S+)\s+(\S+)")
SUPPORTED_INTACT_INTERACTOR_TYPES = {value.lower() for value in INTACT_INTERACTOR_TYPES}
SUPPORTED_INTACT_INTERACTION_TYPES = {value.lower() for value in INTACT_INTERACTION_TYPES}


def _iter_mitab_rows(zf: ZipFile, member_name: str) -> Iterator[List[str]]:
  """Yield parsed MITAB rows from one member inside the IntAct archive."""
  with zf.open(member_name) as handle:
    text_handle = (line.decode("utf-8", errors="replace") for line in handle)
    reader = csv.reader(text_handle, delimiter="\t")
    for row in reader:
      if not row or row[0].startswith("#"):
        continue
      yield row


def _load_intact_fasta_sequences(zf: ZipFile) -> Dict[str, str]:
  """Load sequence mappings from the bundled IntAct FASTA export.

  Each FASTA header contains both an IntAct accession and a UniProt accession.
  The returned mapping is keyed by both identifiers so MITAB rows can resolve
  sequences regardless of whether they reference `intact:EBI-...` or
  `uniprotkb:...` identifiers.
  """
  sequences: Dict[str, str] = {}
  current_keys: List[str] = []
  sequence_parts: List[str] = []

  def flush_record():
    if not current_keys:
      return
    sequence = "".join(sequence_parts).upper()
    if not sequence:
      return
    for key in current_keys:
      sequences[key] = sequence

  with zf.open(INTACT_FASTA_PATH) as handle:
    for raw_line in handle:
      line = raw_line.decode("utf-8", errors="replace").strip()
      if not line:
        continue
      if line.startswith(">"):
        flush_record()
        sequence_parts = []
        current_keys = []
        match = FASTA_HEADER_PATTERN.match(line)
        if not match:
          continue
        intact_id, accession = match.groups()
        current_keys = [f"intact:{intact_id}".lower(), f"uniprotkb:{accession}".lower(), accession.upper()]
      else:
        sequence_parts.append(line)

  flush_record()
  return sequences


def _extract_identifier_candidates(*values: object) -> List[str]:
  """Extract normalized IntAct and UniProt identifier candidates from MITAB fields."""
  candidates: List[str] = []

  for value in values:
    if value is None or (isinstance(value, float) and pd.isna(value)):
      continue

    for token in str(value).split("|"):
      stripped = token.strip()
      if not stripped:
        continue

      lowered = stripped.lower()
      if lowered.startswith("uniprotkb:"):
        accession = lowered.split(":", 1)[1].upper()
        candidates.append(lowered)
        candidates.append(accession)
        if "-PRO_" in accession:
          candidates.append(accession.split("-PRO_", 1)[0])
        elif "-" in accession:
          candidates.append(accession.split("-", 1)[0])
      elif lowered.startswith("intact:"):
        candidates.append(lowered)

  deduped: List[str] = []
  seen = set()
  for candidate in candidates:
    if candidate not in seen:
      deduped.append(candidate)
      seen.add(candidate)
  return deduped


def _parse_taxid_set(field: str) -> set[str]:
  """Extract taxon IDs from one MITAB taxid field."""
  taxids = set()
  for token in str(field or "").split("|"):
    token = token.strip().lower()
    if token.startswith("taxid:"):
      taxid = token.split("taxid:", 1)[1].split("(", 1)[0]
      if taxid:
        taxids.add(taxid)
  return taxids


def _mitab_terms(field: str) -> set[str]:
  """Extract normalized controlled-vocabulary labels from a MITAB field."""
  values = set()
  for token in str(field or "").split("|"):
    token = token.strip().lower()
    if "(" in token and token.endswith(")"):
      values.add(token.rsplit("(", 1)[1][:-1].strip())
  return values


def _resolve_sequence(candidates: Sequence[str], sequence_map: Dict[str, str]) -> Optional[str]:
  """Resolve the first available sequence for a list of identifier candidates."""
  for candidate in candidates:
    sequence = sequence_map.get(candidate)
    if sequence:
      return sequence
  return None


def _row_to_intact_entry(row: Sequence[str], interaction_label: bool, sequence_map: Dict[str, str]) -> Optional[InteractionEntry]:
  """Convert one MITAB row into a sequence-level singleton interaction pair.

  Only concrete human protein or peptide interactors are kept. Rows that
  reference ambiguous IntAct entities, unsupported taxa, or identifiers absent
  from the bundled FASTA export are skipped.
  """
  if len(row) < 15:
    return None

  types_a = _mitab_terms(row[20]) if len(row) > 20 else set()
  types_b = _mitab_terms(row[21]) if len(row) > 21 else set()
  if types_a.isdisjoint(SUPPORTED_INTACT_INTERACTOR_TYPES) or types_b.isdisjoint(SUPPORTED_INTACT_INTERACTOR_TYPES):
    return None

  taxids_a = _parse_taxid_set(row[9])
  taxids_b = _parse_taxid_set(row[10])
  if INTACT_SPECIES_TAXID not in taxids_a or INTACT_SPECIES_TAXID not in taxids_b:
    return None

  interaction_types = _mitab_terms(row[11])
  if interaction_types and interaction_types.isdisjoint(SUPPORTED_INTACT_INTERACTION_TYPES):
    return None

  candidates_a = _extract_identifier_candidates(row[0], row[2])
  candidates_b = _extract_identifier_candidates(row[1], row[3])
  if not candidates_a or not candidates_b:
    return None

  sequence_a = _resolve_sequence(candidates_a, sequence_map)
  sequence_b = _resolve_sequence(candidates_b, sequence_map)
  if not sequence_a or not sequence_b:
    return None

  return InteractionEntry(
    source=f"intact_{'positive' if interaction_label else 'negative'}",
    group1=[sequence_a],
    group2=[sequence_b],
    interaction_label=interaction_label,
  )


def iter_intact_entries() -> Iterable[InteractionEntry]:
  """Yield IntAct positive and negative rows from the local bulk archive.

  The loader operates entirely on the local `all.zip` download. It reads the
  canonical MITAB exports for positive and negative interactions and resolves
  interactor sequences from the bundled IntAct FASTA file, avoiding live
  IntAct or UniProt requests during aggregation.
  """
  if not INTACT_ARCHIVE_PATH.exists():
    raise FileNotFoundError(f"Missing IntAct archive at {INTACT_ARCHIVE_PATH}")

  with ZipFile(INTACT_ARCHIVE_PATH) as zf:
    sequence_map = _load_intact_fasta_sequences(zf)
    print(f"Source=intact loaded_sequences={len(sequence_map)}", flush=True)
    for member_name, interaction_label, mode in (
      (INTACT_POSITIVE_MITAB_PATH, True, "positive"),
      (INTACT_NEGATIVE_MITAB_PATH, False, "negative"),
    ):
      for row in _iter_mitab_rows(zf, member_name):
        try:
          entry = _row_to_intact_entry(row, interaction_label=interaction_label, sequence_map=sequence_map)
        except Exception as exc:
          interaction_id = row[13] if len(row) > 13 else "unknown"
          print(f"Source=intact skipped_row mode={mode} interaction={interaction_id} error={exc}", flush=True)
          continue
        if entry is not None:
          yield entry

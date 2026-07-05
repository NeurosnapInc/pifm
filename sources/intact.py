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

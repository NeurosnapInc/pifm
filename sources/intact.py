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

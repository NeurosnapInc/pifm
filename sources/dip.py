"""
DIP -- Database of Interacting Proteins (experimentally determined PPIs).

DIP is a small, high-quality, curated set of physical interactions. It requires
a (free) login to download, so the raw file must be fetched manually (see
README). Files are PSI-MITAB; UniProt accessions appear in the identifier and
alternative-identifier columns and are resolved to sequences via
``data/raw/uniprot/``.

Download (manual, registration required):
  data/raw/dip/dip.txt
"""

from typing import Iterator, Optional

from contract import InteractionEntry

from ._common import RAW_DIR, hint_missing, open_text, resolve_uniprot, uniprot_sequences

DIP_FILE = RAW_DIR / "dip" / "dip.txt"

# MITAB identifier + alternative-identifier columns for each interactor. DIP
# records the DIP id in the primary column and cross-references (incl. UniProt)
# in the alternative-id column, so both are scanned.
_ID_COLUMNS_A = (0, 2)
_ID_COLUMNS_B = (1, 3)


def _extract_uniprot(columns, indices) -> Optional[str]:
  """Return the first UniProt accession found across the given MITAB columns."""
  for index in indices:
    if index >= len(columns):
      continue
    for token in columns[index].split("|"):
      token = token.strip()
      if token.startswith("uniprotkb:"):
        return token.split(":", 1)[1].strip().strip('"')
  return None


def iter_dip() -> Iterator[InteractionEntry]:
  if not DIP_FILE.is_file():
    hint_missing(
      "dip",
      DIP_FILE,
      "Register and download the full MITAB file from https://dip.doe-mbi.ucla.edu/dip/Download.cgi, saved as data/raw/dip/dip.txt",
    )
    return

  sequences = uniprot_sequences()
  resolved = 0
  unresolved = 0

  with open_text(DIP_FILE) as handle:
    for line in handle:
      if line.startswith("#"):  # MITAB header row
        continue
      columns = line.rstrip("\n").split("\t")
      if len(columns) < 4:
        continue

      accession_a = _extract_uniprot(columns, _ID_COLUMNS_A)
      accession_b = _extract_uniprot(columns, _ID_COLUMNS_B)
      if not accession_a or not accession_b:
        unresolved += 1
        continue

      seq_a = resolve_uniprot(sequences, accession_a)
      seq_b = resolve_uniprot(sequences, accession_b)
      if not seq_a or not seq_b:
        unresolved += 1
        continue

      resolved += 1
      yield InteractionEntry(
        source="dip",
        group1=seq_a,
        group2=seq_b,
        interaction_label=True,
      )

  print(f"Source=dip resolved_pairs={resolved} unresolved_pairs={unresolved}")

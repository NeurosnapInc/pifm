"""
STRING -- functional protein association networks (per-species).

STRING edges are functional *associations*, not necessarily physical binding,
so this loader is deliberately conservative:

- it reads only the ``protein.physical.links`` (physical subnetwork) files,
- it requires a high combined score, and
- it requires nonzero experimental/database evidence (not text-mining alone).

STRING ships per-species sequence FASTAs, so -- unlike the UniProt-keyed sources
-- it needs no external sequence map. Each ``*.protein.physical.links*`` file is
paired with the matching ``*.protein.sequences*`` FASTA by taxid prefix.

Download (per species, into ``data/raw/string/``):
  <taxid>.protein.physical.links.detailed.v12.0.txt.gz
  <taxid>.protein.sequences.v12.0.fa.gz
"""

from pathlib import Path
from typing import Dict, Iterator

from contract import InteractionEntry

from ._common import RAW_DIR, hint_missing, open_text, read_fasta

STRING_DIR = RAW_DIR / "string"

# Minimum STRING combined score (0-1000). 700 == "high confidence".
STRING_MIN_SCORE = 700

# Require direct experimental or curated-database evidence, so we keep physical
# interactions rather than associations inferred only from text-mining.
REQUIRE_EXPERIMENTAL_EVIDENCE = True
_EVIDENCE_COLUMNS = ("experiments", "experiments_transferred", "database", "database_transferred")

# Benchmark organism held out from training (M. genitalium).
MGEN_TAXID = "2097"


def _load_string_sequences(fasta_path: Path) -> Dict[str, str]:
  """Map ``<taxid>.<protein_id>`` -> sequence for one species FASTA."""
  mapping: Dict[str, str] = {}
  for header, seq in read_fasta(fasta_path):
    string_id = header.split()[0]
    if string_id and seq:
      mapping[string_id] = seq.upper()
  return mapping


def _sequences_path_for(taxid: str) -> Path:
  """Find the sequence FASTA that pairs with a links file for ``taxid``."""
  candidates = sorted(STRING_DIR.glob(f"{taxid}.protein.sequences*.fa*"))
  return candidates[0] if candidates else STRING_DIR / f"{taxid}.protein.sequences.MISSING.fa.gz"


def _iter_species(links_path: Path) -> Iterator[InteractionEntry]:
  taxid = links_path.name.split(".", 1)[0]
  if taxid == MGEN_TAXID:
    print(f"Source=string quarantined taxid={taxid} (M. genitalium benchmark) file={links_path.name}")
    return

  sequences_path = _sequences_path_for(taxid)
  if not sequences_path.is_file():
    print(f"Source=string skipped taxid={taxid}: missing sequences FASTA for {links_path.name}")
    return

  id_to_seq = _load_string_sequences(sequences_path)
  resolved = 0
  unresolved = 0
  filtered = 0

  with open_text(links_path) as handle:
    header = handle.readline().split()
    index = {name: pos for pos, name in enumerate(header)}
    if not {"protein1", "protein2", "combined_score"} <= index.keys():
      print(f"Source=string skipped taxid={taxid}: unexpected header in {links_path.name}")
      return
    evidence_indices = [index[name] for name in _EVIDENCE_COLUMNS if name in index]

    for line in handle:
      columns = line.split()
      if len(columns) < len(header):
        continue

      try:
        combined_score = int(columns[index["combined_score"]])
      except ValueError:
        continue
      if combined_score < STRING_MIN_SCORE:
        filtered += 1
        continue

      if REQUIRE_EXPERIMENTAL_EVIDENCE and evidence_indices:
        evidence = sum(int(columns[i]) for i in evidence_indices)
        if evidence <= 0:
          filtered += 1
          continue

      seq_a = id_to_seq.get(columns[index["protein1"]])
      seq_b = id_to_seq.get(columns[index["protein2"]])
      if not seq_a or not seq_b:
        unresolved += 1
        continue

      resolved += 1
      yield InteractionEntry(
        source="string",
        group1=seq_a,
        group2=seq_b,
        interaction_label=True,
      )

  print(f"Source=string taxid={taxid} resolved_pairs={resolved} unresolved_pairs={unresolved} filtered={filtered}")


def iter_string() -> Iterator[InteractionEntry]:
  if not STRING_DIR.is_dir():
    hint_missing(
      "string",
      STRING_DIR,
      "Download per-species <taxid>.protein.physical.links.detailed and .protein.sequences files from https://stringdb-downloads.org/ (not taxid 2097).",
    )
    return

  links_files = sorted(STRING_DIR.glob("*.protein.physical.links*.txt*"))
  if not links_files:
    hint_missing(
      "string",
      STRING_DIR / "<taxid>.protein.physical.links.detailed.v12.0.txt.gz",
      "Download per-species physical-links + sequences files from https://stringdb-downloads.org/ (not taxid 2097).",
    )
    return

  for links_path in links_files:
    yield from _iter_species(links_path)

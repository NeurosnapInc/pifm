"""
Negatome 2.0 -- experimentally supported NON-interacting protein pairs.

Negatome is the pipeline's only explicit source of *negative* interaction
labels; every other registered source contributes positives. Pairs are
distributed as UniProt accession pairs, so a UniProt FASTA must be present in
``data/raw/uniprot/`` for sequences to resolve (see README).

Download (the "combined_stringent" list excludes any pair whose members are
seen interacting in IntAct, making it the safest negative set):
  data/raw/negatome/combined_stringent.txt
"""

from typing import Iterator

from contract import InteractionEntry

from ._common import RAW_DIR, hint_missing, resolve_uniprot, uniprot_sequences

NEGATOME_FILE = RAW_DIR / "negatome" / "combined_stringent.txt"


def iter_negatome() -> Iterator[InteractionEntry]:
  if not NEGATOME_FILE.is_file():
    hint_missing(
      "negatome",
      NEGATOME_FILE,
      "Download combined_stringent.txt from https://mips.helmholtz-muenchen.de/proj/ppi/negatome/",
    )
    return

  sequences = uniprot_sequences()
  resolved = 0
  unresolved = 0

  with open(NEGATOME_FILE, "r", encoding="utf-8", errors="replace") as handle:
    for line in handle:
      line = line.strip()
      if not line or line.startswith("#"):
        continue
      # The stringent files are tab-separated accession pairs; be tolerant of
      # stray whitespace and only rely on the first two columns.
      fields = [field for field in line.split("\t") if field.strip()]
      if len(fields) < 2:
        continue

      accession_a, accession_b = fields[0].strip(), fields[1].strip()
      seq_a = resolve_uniprot(sequences, accession_a)
      seq_b = resolve_uniprot(sequences, accession_b)
      if not seq_a or not seq_b:
        unresolved += 1
        continue

      resolved += 1
      yield InteractionEntry(
        source="negatome",
        group1=seq_a,
        group2=seq_b,
        interaction_label=False,
      )

  print(f"Source=negatome resolved_pairs={resolved} unresolved_pairs={unresolved}")

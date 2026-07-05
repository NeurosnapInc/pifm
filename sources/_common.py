"""
Shared helpers for dataset source loaders.

Raw downloads live under ``data/raw/`` (see the README section "Data Sources &
Downloads"). Some source loaders use source-specific subdirectories when the
source needs several files. Loaders are defensive: when their expected files
are absent they print a short, actionable hint and yield nothing, so
``aggregate_data.py`` always runs end to end even before any data has been
downloaded.
"""

import gzip
import io
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

# Repository-relative root for all raw dataset downloads.
RAW_DIR = Path("data/raw")


def open_text(path: Path) -> io.TextIOBase:
  """Open a path as UTF-8 text, transparently decompressing ``.gz`` files."""
  path = Path(path)
  if path.suffix == ".gz":
    return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
  return open(path, "r", encoding="utf-8", errors="replace")


def read_fasta(path: Path) -> Iterator[Tuple[str, str]]:
  """Yield ``(header, sequence)`` for each record in a (optionally gzipped) FASTA.

  ``header`` excludes the leading ``>`` and preserves the full description line;
  the sequence is concatenated across wrapped lines with whitespace stripped.
  """
  header: Optional[str] = None
  chunks = []
  with open_text(path) as handle:
    for line in handle:
      line = line.rstrip("\n")
      if line.startswith(">"):
        if header is not None:
          yield header, "".join(chunks)
        header = line[1:]
        chunks = []
      elif line:
        chunks.append(line.strip())
  if header is not None:
    yield header, "".join(chunks)


def strip_isoform(accession: str) -> str:
  """Return the base UniProt accession, dropping any ``-N`` isoform suffix."""
  return accession.split("-")[0]


def _uniprot_accession(header: str) -> str:
  """Extract the primary accession from a UniProt FASTA header.

  Handles ``sp|P12345|NAME_ORG`` / ``tr|P12345|NAME_ORG`` style headers as well
  as bare ``P12345 ...`` headers, falling back to the first whitespace token.
  """
  first = header.split()[0] if header else ""
  parts = first.split("|")
  if len(parts) >= 2 and parts[0] in ("sp", "tr"):
    return parts[1]
  return first


@lru_cache(maxsize=1)
def uniprot_sequences() -> Dict[str, str]:
  """Load a UniProt accession -> sequence map from ``data/raw/uniprot/``.

  Some sources, such as Negatome, distribute interactions as UniProt
  accession pairs rather than sequences. Drop one or more UniProt FASTA files
  (e.g. ``uniprot_sprot.fasta.gz`` or per-proteome FASTAs) into
  ``data/raw/uniprot/`` and every accession-based loader can resolve sequences
  locally with no network calls. The map is cached for the lifetime of the
  process, so it is parsed at most once per aggregation run.
  """
  mapping: Dict[str, str] = {}
  udir = RAW_DIR / "uniprot"
  if not udir.is_dir():
    return mapping
  patterns = ("*.fasta", "*.fasta.gz", "*.fa", "*.fa.gz")
  files = sorted(p for pattern in patterns for p in udir.glob(pattern))
  for fasta in files:
    for header, seq in read_fasta(fasta):
      accession = _uniprot_accession(header)
      if accession and seq and accession not in mapping:
        mapping[accession] = seq.upper()
  return mapping


def resolve_uniprot(sequences: Dict[str, str], accession: str) -> Optional[str]:
  """Look up a sequence by accession, retrying with the isoform suffix removed."""
  if accession in sequences:
    return sequences[accession]
  return sequences.get(strip_isoform(accession))


def hint_missing(source: str, expected: Path, how: str) -> None:
  """Print a one-line, actionable hint when a source's raw file is absent."""
  print(f"Source={source} skipped: missing {expected}. {how}")

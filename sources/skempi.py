"""SKEMPI v2.0 source placeholder.

SKEMPI rows identify complexes by PDB chain IDs, while this project stores
interaction groups as amino-acid sequences. A correct SKEMPI integration must
extract wild-type chain sequences from the companion PDB archive before yielding
`InteractionEntry` objects. This module intentionally does not emit chain IDs as
sequences because that would create invalid training examples.
"""

from pathlib import Path
from typing import Iterable

from sources.source_types import InteractionEntry


SKEMPI_CSV_PATH = Path("data/raw/skempi_v2.csv")


def iter_skempi_entries(csv_path: Path = SKEMPI_CSV_PATH) -> Iterable[InteractionEntry]:
  """Yield SKEMPI sequence-level entries once PDB sequence extraction is added."""
  if csv_path.exists():
    print(
      "Warning: SKEMPI data was found, but SKEMPI aggregation is disabled until "
      "PDB chain IDs are resolved to amino-acid sequences.",
      flush=True,
    )
  return []

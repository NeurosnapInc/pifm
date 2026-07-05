"""PPB-Affinity source loader."""

import csv
from pathlib import Path
from typing import Iterable, List, Optional

from contract import InteractionEntry


PPB_AFFINITY_FILTERED_PATH = Path("data/raw/ppb_affinity_filtered.csv")


def _split_sequence_column(value: str) -> List[str]:
  return [seq.strip() for seq in value.replace("\n", ",").split(",") if seq.strip()]


def iter_ppb_affinity(csv_path: Path = PPB_AFFINITY_FILTERED_PATH) -> Iterable[InteractionEntry]:
  """Yield interaction entries from the local PPB-Affinity filtered CSV."""
  if not csv_path.exists():
    print(
      f"Warning: expected PPB-Affinity data at {csv_path}. "
      "Download it with the wget command in README.md.",
      flush=True,
    )
    return

  with csv_path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
      ligand_column = row.get("Ligand Sequences") or row.get("ligand sequences")
      receptor_column = row.get("Receptor Sequences") or row.get("receptor sequences")
      if not ligand_column or not receptor_column:
        continue

      ligand_sequences = _split_sequence_column(ligand_column)
      receptor_sequences = _split_sequence_column(receptor_column)
      if not ligand_sequences or not receptor_sequences:
        continue

      affinity_nm: Optional[float] = None
      kd_m_value = row.get("KD(M)") or row.get("kd(m)")
      if kd_m_value:
        try:
          kd_m = float(kd_m_value)
          if kd_m > 0.0:
            affinity_nm = kd_m * 1e9
        except ValueError:
          affinity_nm = None

      yield InteractionEntry(
        source="ppb_affinity",
        group1=ligand_sequences,
        group2=receptor_sequences,
        affinity_nm=affinity_nm,
        interaction_label=True,
      )


def iter_ppb_affinity_entries(csv_path: Path = PPB_AFFINITY_FILTERED_PATH) -> Iterable[InteractionEntry]:
  """Compatibility alias for older imports."""
  yield from iter_ppb_affinity(csv_path)

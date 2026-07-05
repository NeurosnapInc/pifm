"""
SKEMPI v2.0 -- protein-protein complexes with experimentally measured binding
affinities for wild-type and point-mutant variants.

Unlike every other source, SKEMPI ships NO sequences in its CSV: rows reference a
PDB id and the chains forming each side of the complex (e.g. ``1CSE_E_I``).
Sequences are therefore reconstructed from the bundled cleaned PDB structures
using ``neurosnap`` (the project's structure dependency). We rebuild each chain's
sequence and an author-residue-number -> offset index from the same filtered
residue iteration, so the numbering that SKEMPI mutation strings (e.g. ``LI38G``)
use lines up exactly with the sequence we mutate.

For each CSV row we emit:
  - the wild-type complex once per structure (with ``Affinity_wt``), and
  - each mutant complex, with its point mutation(s) applied to the wild-type
    chain sequences (with ``Affinity_mut``).

Affinities are Kd in molar units; they are converted to nM (``* 1e9``) for the
aggregation layer's ``pKd`` conversion. Both wild-type and mutant complexes are
labeled as positive interactions (SKEMPI only contains complexes that form).

Download + extract:
  data/raw/skempi_v2.csv                    (semicolon-delimited)
  data/raw/PDBs/<PDBID>.pdb                  (from SKEMPI2_PDBs.tgz)
"""

import csv
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from neurosnap.io.pdb import parse_pdb

from contract import InteractionEntry

from ._common import RAW_DIR, hint_missing

SKEMPI_CSV = RAW_DIR / "skempi_v2.csv"
SKEMPI_PDB_DIR = RAW_DIR / "PDBs"

_THREE_TO_ONE = {
  "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
  "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
  "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
  "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
  "MSE": "M", "SEC": "U", "PYL": "O",
}

# Parsed PDB structure: chain -> sequence, and chain -> {author_resSeq: index}.
_Structure = Tuple[Dict[str, str], Dict[str, Dict[str, int]]]


def _parse_pdb_chains(path: Path) -> _Structure:
  """Extract per-chain sequences and residue-number indices via neurosnap.

  Only polymer (non-hetero) residues are kept, matching neurosnap's
  ``Chain.sequence(polymer_type="protein")`` while letting us build, in the same
  pass, an index from each author residue number (as a string) to its offset in
  the chain sequence. Building both together guarantees the offsets used to apply
  SKEMPI mutations stay aligned with the sequence. Multi-model files collapse to
  the first model.
  """
  model = parse_pdb(str(path)).first()

  chain_seq: Dict[str, str] = {}
  chain_pos_index: Dict[str, Dict[str, int]] = {}

  for chain_id in model.chain_ids():
    residues: List[str] = []
    index_map: Dict[str, int] = {}
    for residue in model[chain_id].residues():
      if residue.hetero:  # skip waters, ions, ligands
        continue
      index_map.setdefault(str(residue.res_id), len(residues))
      residues.append(_THREE_TO_ONE.get(residue.res_name, "X"))
    if residues:
      chain_seq[chain_id] = "".join(residues)
      chain_pos_index[chain_id] = index_map

  return chain_seq, chain_pos_index


def _load_structure(pdb_id: str, cache: Dict[str, Optional[_Structure]]) -> Optional[_Structure]:
  """Load and cache a parsed structure, trying common filename casings."""
  if pdb_id in cache:
    return cache[pdb_id]
  for name in (f"{pdb_id}.pdb", f"{pdb_id.upper()}.pdb", f"{pdb_id.lower()}.pdb"):
    path = SKEMPI_PDB_DIR / name
    if path.is_file():
      cache[pdb_id] = _parse_pdb_chains(path)
      return cache[pdb_id]
  cache[pdb_id] = None
  return None


def _parse_affinity_nm(parsed_value: Optional[str]) -> Optional[float]:
  """Convert a SKEMPI ``*_parsed`` Kd value (molar) into nanomolar."""
  if not parsed_value or not parsed_value.strip():
    return None
  try:
    kd_molar = float(parsed_value.strip())
  except ValueError:
    return None
  if kd_molar <= 0.0:
    return None
  return kd_molar * 1e9


def _parse_mutation(token: str) -> Optional[Tuple[str, str, str, str]]:
  """Parse a cleaned SKEMPI mutation ``<WT><Chain><ResNum><MUT>`` (e.g. ``LI38G``)."""
  token = token.strip()
  if len(token) < 4:
    return None
  wild_type, chain, mutant = token[0], token[1], token[-1]
  position = token[2:-1]
  if not position:
    return None
  return wild_type, chain, position, mutant


def _apply_mutations(chain_seq: Dict[str, str], chain_pos_index: Dict[str, Dict[str, int]],
                     mutations: List[Tuple[str, str, str, str]]) -> Optional[Dict[str, str]]:
  """Return chain sequences with mutations applied, or None on any mismatch.

  A mutation is rejected (whole row skipped) if the chain is absent, the residue
  number is not found, or the wild-type residue does not match the structure --
  any of which signals a numbering mismatch we should not silently paper over.
  """
  mutable = {chain: list(seq) for chain, seq in chain_seq.items()}
  for wild_type, chain, position, mutant in mutations:
    index_map = chain_pos_index.get(chain)
    if index_map is None:
      return None
    index = index_map.get(position)
    if index is None:
      return None
    if mutable[chain][index] != wild_type:
      return None
    mutable[chain][index] = mutant
  return {chain: "".join(residues) for chain, residues in mutable.items()}


def _chain_group(chain_seq: Dict[str, str], chains: str) -> Optional[List[str]]:
  """Collect the sequences for a side's chains, or None if any is missing/empty."""
  group = []
  for chain in chains:
    sequence = chain_seq.get(chain)
    if not sequence:
      return None
    group.append(sequence)
  return group


def iter_skempi() -> Iterator[InteractionEntry]:
  if not SKEMPI_CSV.is_file():
    hint_missing(
      "skempi",
      SKEMPI_CSV,
      "Download skempi_v2.csv from https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv",
    )
    return
  if not SKEMPI_PDB_DIR.is_dir():
    hint_missing(
      "skempi",
      SKEMPI_PDB_DIR,
      "Download SKEMPI2_PDBs.tgz and extract it with: tar -xzf data/raw/SKEMPI2_PDBs.tgz -C data/raw",
    )
    return

  structure_cache: Dict[str, Optional[_Structure]] = {}
  seen_wildtype = set()
  wildtype_complexes = 0
  mutant_complexes = 0
  skipped = 0

  with open(SKEMPI_CSV, newline="", encoding="utf-8", errors="replace") as handle:
    reader = csv.DictReader(handle, delimiter=";")
    for row in reader:
      pdb_field = (row.get("#Pdb") or "").strip()
      parts = pdb_field.split("_")
      if len(parts) != 3:
        skipped += 1
        continue
      pdb_id, chains1, chains2 = parts

      structure = _load_structure(pdb_id, structure_cache)
      if structure is None:
        skipped += 1
        continue
      chain_seq, chain_pos_index = structure

      wt_group1 = _chain_group(chain_seq, chains1)
      wt_group2 = _chain_group(chain_seq, chains2)
      if wt_group1 is None or wt_group2 is None:
        skipped += 1
        continue

      # Wild-type complex: emit once per distinct PDB/chain combination.
      if pdb_field not in seen_wildtype:
        seen_wildtype.add(pdb_field)
        wildtype_complexes += 1
        yield InteractionEntry(
          source="skempi",
          group1=wt_group1,
          group2=wt_group2,
          affinity_nm=_parse_affinity_nm(row.get("Affinity_wt_parsed")),
          interaction_label=True,
        )

      # Mutant complex: apply the cleaned mutation(s) to the wild-type sequences.
      mutation_field = (row.get("Mutation(s)_cleaned") or "").strip()
      mutations = [_parse_mutation(token) for token in mutation_field.split(",") if token.strip()]
      if not mutations or any(mutation is None for mutation in mutations):
        skipped += 1
        continue

      mutated_seq = _apply_mutations(chain_seq, chain_pos_index, mutations)
      if mutated_seq is None:
        skipped += 1
        continue

      mut_group1 = _chain_group(mutated_seq, chains1)
      mut_group2 = _chain_group(mutated_seq, chains2)
      if mut_group1 is None or mut_group2 is None:
        skipped += 1
        continue

      mutant_complexes += 1
      yield InteractionEntry(
        source="skempi",
        group1=mut_group1,
        group2=mut_group2,
        affinity_nm=_parse_affinity_nm(row.get("Affinity_mut_parsed")),
        interaction_label=True,
      )

  print(f"Source=skempi wildtype_complexes={wildtype_complexes} mutant_complexes={mutant_complexes} skipped={skipped}")

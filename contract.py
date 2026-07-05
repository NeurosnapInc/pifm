"""
Data contract shared between the aggregation entrypoint and the dataset source
loaders in the ``sources`` package.

These types live in their own module so that individual source loaders can
import them without importing ``aggregate_data`` (which imports the loaders in
turn). Keeping the contract here avoids a circular dependency and the
double-import that would otherwise occur when ``aggregate_data`` is run as a
script.

The expected source contract is intentionally lightweight so new datasets can
be registered without modifying the rest of the training pipeline:

- add one ``SourceSpec`` to the registry in ``sources`` (``build_source_specs``)
- implement a generator function that yields ``InteractionEntry`` objects
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Union

# A single interaction side. Either a pre-delimited ``":"`` string of amino-acid
# sequences, or an iterable of raw amino-acid sequences (one per chain).
SequenceGroup = Union[str, Sequence[str]]
SourceFactory = Callable[[], Iterable["InteractionEntry"]]


@dataclass(frozen=True)
class InteractionEntry:
  """One protein-group interaction observation produced by a source loader.

  ``group1``/``group2`` are amino-acid sequence groups only. The downstream
  tokenizer (``tokenize_data.py``) feeds every group member to ProstT5 as a
  protein sequence, so non-protein modalities (e.g. SMILES ligands) must not be
  emitted here until a molecule modality is added to the pipeline.

  Source loaders should provide Kd-like affinity measurements in nanomolar via
  ``affinity_nm``. The aggregation layer converts this source-facing unit into
  standardized ``affinity_pkd`` before writing the canonical DuckDB table.
  Exactly one of ``affinity_nm`` / ``interaction_label`` is required in practice,
  but both may be provided when a source carries both kinds of supervision.
  """

  source: str
  group1: SequenceGroup
  group2: SequenceGroup
  affinity_nm: Optional[float] = None
  interaction_label: Optional[bool] = None


@dataclass(frozen=True)
class SourceSpec:
  """Registration record binding a source name to its loader generator."""

  name: str
  loader: SourceFactory

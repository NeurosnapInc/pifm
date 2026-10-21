"""
Dataset source loaders registered into the aggregation pipeline.

To add a new source:
  1. implement a generator ``iter_<name>() -> Iterator[InteractionEntry]`` in a
     new module here (yield sequences only; see ``contract.InteractionEntry``),
  2. append a ``SourceSpec`` for it to ``build_source_specs()`` below,
  3. document its download under "Data Sources & Downloads" in the README.

Registry order defines priority: when two sources produce the same canonical
group pair, the earlier source wins (see ``aggregate_data._insert_source_rows``).
Higher-trust, curated sources are therefore listed before noisier ones.
"""

from typing import List

from contract import SourceSpec

from .intact import iter_intact
from .literature_affinity import iter_literature_affinity
from .negatome import iter_negatome
from .ppb_affinity import iter_ppb_affinity
from .skempi import iter_skempi
from .string_db import iter_string


def build_source_specs() -> List[SourceSpec]:
  return [
    SourceSpec(name="ppb_affinity", loader=iter_ppb_affinity),
    SourceSpec(name="skempi", loader=iter_skempi),
    SourceSpec(name="literature_affinity", loader=iter_literature_affinity),
    SourceSpec(name="intact", loader=iter_intact),
    SourceSpec(name="negatome", loader=iter_negatome),
    SourceSpec(name="string", loader=iter_string),
  ]


__all__ = ["build_source_specs"]

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Union


SequenceGroup = Union[str, Sequence[str]]
SourceFactory = Callable[[], Iterable["InteractionEntry"]]


@dataclass(frozen=True)
class InteractionEntry:
  source: str
  group1: SequenceGroup
  group2: SequenceGroup
  affinity_nm: Optional[float] = None
  interaction_label: Optional[bool] = None


@dataclass(frozen=True)
class SourceSpec:
  name: str
  loader: SourceFactory

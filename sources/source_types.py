"""Compatibility re-export for source loader contracts.

New code should import these names from `contract`, but this module remains so
older source modules or notebooks that import `sources.source_types` keep
working.
"""

from contract import InteractionEntry, SequenceGroup, SourceFactory, SourceSpec


__all__ = ["InteractionEntry", "SequenceGroup", "SourceFactory", "SourceSpec"]

"""Relation scorers. Importing this package registers every relation."""

from . import comparative, ordinal, projective, proximity, vertical  # noqa: F401
from .base import (REGISTRY, RelationConfig, RelationScore, RelationSpec,
                   all_relations, canonical_relation, family,
                   frame_dependent_relations, is_frame_dependent, spec)

__all__ = ["RelationConfig", "RelationScore", "RelationSpec", "REGISTRY",
           "all_relations", "canonical_relation", "family",
           "frame_dependent_relations", "is_frame_dependent", "spec",
           "projective", "vertical", "proximity", "ordinal", "comparative"]

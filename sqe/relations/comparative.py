"""Comparatives and superlatives: taller, bigger, the largest, the nearest.

Frame-independent, cheap, and included because a benchmark that only measures
the hard frame-dependent cases cannot show that the *rest* of the system works.
The interesting number in the final report is the gap between these and the
projective relations on the same scenes.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..geom.obb import OBB
from .base import (RelationConfig, RelationScore, RelationSpec, ramp, register)

register(RelationSpec("taller", "comparative", False, 1,
                      aliases=("taller than", "higher than than"),
                      doc="target is taller than the anchor"))
register(RelationSpec("shorter", "comparative", False, 1,
                      aliases=("shorter than", "lower than than"),
                      doc="target is shorter than the anchor"))
register(RelationSpec("bigger", "comparative", False, 1,
                      aliases=("bigger than", "larger than", "greater than"),
                      doc="target has a larger volume than the anchor"))
register(RelationSpec("smaller", "comparative", False, 1,
                      aliases=("smaller than", "tinier than"),
                      doc="target has a smaller volume than the anchor"))
register(RelationSpec("wider", "comparative", False, 1,
                      aliases=("wider than", "broader than"),
                      doc="target has a larger horizontal footprint"))

SIZE_METRICS: Dict[str, Callable[[OBB], float]] = {
    "height": lambda o: float(o.height),
    "volume": lambda o: float(o.volume),
    "footprint": lambda o: float(o.footprint_area),
    "width": lambda o: float(max(o.extent[0], o.extent[1])),
    "length": lambda o: float(max(o.extent[0], o.extent[1])),
    "max_dim": lambda o: float(np.max(o.extent)),
}

COMPARATIVE_METRIC = {
    "taller": ("height", +1), "shorter": ("height", -1),
    "bigger": ("volume", +1), "smaller": ("volume", -1),
    "wider": ("footprint", +1),
}

#: Superlative words mapped to (metric, sign). Sign +1 means "the largest".
SUPERLATIVES: Dict[str, Tuple[str, int]] = {
    "tallest": ("height", +1), "highest": ("height", +1),
    "shortest": ("height", -1), "lowest": ("height", -1),
    "biggest": ("volume", +1), "largest": ("volume", +1),
    "smallest": ("volume", -1), "tiniest": ("volume", -1),
    "widest": ("footprint", +1), "narrowest": ("footprint", -1),
    "longest": ("length", +1),
}

#: Bare size adjectives, treated as soft superlatives over the candidate set.
SIZE_ADJECTIVES: Dict[str, Tuple[str, int]] = {
    "big": ("volume", +1), "large": ("volume", +1), "huge": ("volume", +1),
    "small": ("volume", -1), "little": ("volume", -1), "tiny": ("volume", -1),
    "tall": ("height", +1), "short": ("height", -1),
    "wide": ("footprint", +1), "narrow": ("footprint", -1),
    "long": ("length", +1),
}


def comparative_score(target: OBB, anchor: OBB, relation: str,
                      cfg: RelationConfig) -> RelationScore:
    """Score "target is <relation> than anchor" on a smooth ratio."""
    if relation not in COMPARATIVE_METRIC:
        return RelationScore(0.0, {}, [f"{relation!r} is not a comparative"])
    metric, sign = COMPARATIVE_METRIC[relation]
    f = SIZE_METRICS[metric]
    tv, av = f(target), f(anchor)
    if av <= 1e-9 or tv <= 1e-9:
        return RelationScore(0.0, {"target": tv, "anchor": av},
                             ["degenerate size"])
    ratio = tv / av if sign > 0 else av / tv
    value = ramp(ratio, 1.0, cfg.size_ratio_significant)
    notes = []
    if 1.0 / cfg.size_ratio_significant < ratio < cfg.size_ratio_significant:
        notes.append(f"the two are within {cfg.size_ratio_significant:.2f}x on "
                     f"{metric}; the comparison is weak")
    return RelationScore(value, {"metric_target": tv, "metric_anchor": av,
                                 "ratio": float(ratio)}, notes)


def superlative_rank(candidates: Sequence[OBB], word: str
                     ) -> Tuple[List[int], List[float], str, bool]:
    """Rank candidates by a superlative. Returns (order, values, metric, tie).

    `tie` is set when the top two are within a few per cent, which is the case
    a benchmark item should be labelled ambiguous for.
    """
    w = (word or "").strip().lower()
    entry = SUPERLATIVES.get(w) or SIZE_ADJECTIVES.get(w)
    if entry is None:
        return list(range(len(candidates))), [], "", False
    metric, sign = entry
    f = SIZE_METRICS[metric]
    vals = [f(c) for c in candidates]
    order = sorted(range(len(candidates)), key=lambda i: -sign * vals[i])
    tie = False
    if len(order) >= 2:
        a, b = vals[order[0]], vals[order[1]]
        m = max(abs(a), abs(b), 1e-9)
        tie = abs(a - b) / m < 0.06
    return order, vals, metric, tie


def is_superlative(word: str) -> bool:
    w = (word or "").strip().lower()
    return w in SUPERLATIVES


def is_size_adjective(word: str) -> bool:
    w = (word or "").strip().lower()
    return w in SIZE_ADJECTIVES

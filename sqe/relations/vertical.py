"""Vertical and topological relations: on, above, below, under, in, inside.

These are frame-independent -- gravity settles them -- which is exactly why they
belong in the benchmark. If a system's accuracy on `on`/`above` is high while
its accuracy on `left`/`right` is poor, the gap is not a perception failure.
That contrast is the cleanest evidence available that the projective errors come
from the frame convention, and it needs the frame-free relations measured on the
same scenes with the same detector to be worth anything.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..geom.obb import OBB, Interval, horizontal_footprint_overlap
from ..geom.support import Level, support_score, support_score_on_level
from .base import (RelationConfig, RelationScore, RelationSpec, gmean, ramp,
                   register)

register(RelationSpec("on", "vertical", False, 1,
                      aliases=("on top of", "resting on", "sitting on",
                               "placed on", "lying on", "atop", "on the top of"),
                      needs_support=True,
                      doc="target rests on the anchor's upper surface"))
register(RelationSpec("above", "vertical", False, 1,
                      aliases=("over", "higher than", "up from"),
                      doc="target is higher and horizontally overlapping"))
register(RelationSpec("below", "vertical", False, 1,
                      aliases=("under", "underneath", "beneath", "lower than",
                               "below of", "down from"),
                      doc="target is lower and horizontally overlapping"))
register(RelationSpec("inside", "vertical", False, 1,
                      aliases=("in", "within", "contained in", "inside of",
                               "in the"),
                      needs_container=True,
                      doc="target is contained by the anchor's volume"))


def on_score(target: OBB, anchor: OBB, cfg: RelationConfig,
             anchor_levels: Optional[Sequence[Level]] = None,
             predicted: bool = False) -> RelationScore:
    """Rest-on score, considering the anchor's internal shelves as well as its
    top face."""
    tol = cfg.contact_tol_predicted if predicted else cfg.contact_tol
    top = support_score(target, anchor, tol, cfg.support_min_overlap)
    best_level, best_level_score = None, 0.0
    for l in (anchor_levels or []):
        s = support_score_on_level(target, anchor, l.z, max(tol, 0.08),
                                  cfg.support_min_overlap * 0.85)
        if s > best_level_score:
            best_level, best_level_score = l.index, s
    value = max(top, best_level_score)
    comp = {"top_face": top, "best_level": best_level_score,
            "gap_to_top": float(target.bottom - anchor.top),
            "footprint_overlap": horizontal_footprint_overlap(target, anchor)}
    notes = []
    if best_level is not None and best_level_score >= top and best_level_score > 0:
        comp["level_index"] = float(best_level)
        notes.append(f"supported by internal level {best_level}")
    return RelationScore(value, comp, notes)


def above_score(target: OBB, anchor: OBB, cfg: RelationConfig,
                require_contact: bool = False) -> RelationScore:
    overlap = horizontal_footprint_overlap(target, anchor)
    if overlap < cfg.above_min_overlap:
        return RelationScore(0.0, {"footprint_overlap": overlap},
                             ["no horizontal overlap with the anchor"])
    gap = float(target.bottom - anchor.top)
    if gap < -0.5 * min(target.height, anchor.height):
        return RelationScore(0.0, {"footprint_overlap": overlap, "gap": gap},
                             ["target is not higher than the anchor"])
    height = 1.0 - ramp(max(0.0, gap), 0.0, cfg.above_gap_zero)
    ordering = ramp(float(target.center[2] - anchor.center[2]), 0.0, 0.05)
    value = gmean([overlap, max(height, 0.15), ordering], [1.0, 0.6, 1.0])
    return RelationScore(value, {"footprint_overlap": overlap, "gap": gap,
                                 "height_term": height, "ordering": ordering})


def below_score(target: OBB, anchor: OBB, cfg: RelationConfig) -> RelationScore:
    sc = above_score(anchor, target, cfg)
    return RelationScore(sc.value, sc.components, sc.notes)


def inside_score(target: OBB, anchor: OBB, cfg: RelationConfig,
                 target_points: Optional[np.ndarray] = None) -> RelationScore:
    """Containment. Uses the target's points when available, its box otherwise.

    Real containers are concave and their contents are often only partly
    observed, so the criterion is the *fraction* of the target inside the
    anchor's padded box, not full enclosure.
    """
    pts = target_points if target_points is not None and len(target_points) \
        else target.corners()
    inside = anchor.contains(pts, pad=cfg.containment_pad)
    frac = float(np.mean(inside)) if len(pts) else 0.0
    vol_ratio = target.volume / max(anchor.volume, 1e-9)
    value = ramp(frac, cfg.containment_min_fraction * 0.5,
                 cfg.containment_min_fraction)
    notes = []
    if vol_ratio > 1.0:
        value *= 0.0
        notes.append("target is larger than the anchor, so it cannot be inside")
    return RelationScore(value, {"inside_fraction": frac,
                                 "volume_ratio": float(vol_ratio)}, notes)

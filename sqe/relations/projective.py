"""Projective relations: left, right, in front of, behind.

These are the frame-dependent ones, and the whole reason this project exists.
The score has four terms, all reported separately so the benchmark can see
which one failed:

``cone``
    Is the anchor-to-target bearing actually pointing along the requested axis?
    Measured in the frame's horizontal plane.

``separation``
    Is the target clear of the anchor's face along that axis? Centroids are not
    enough: a chair whose centroid sits left of a 3 m table's centroid may still
    be sitting squarely in the middle of the table.

``depth``
    "Left of" implies roughly the same distance along the orthogonal horizontal
    axis. A mug two metres nearer the viewer than the laptop is not "left of the
    laptop", it is "in front of" it, and scoring it highly is a classic
    false positive.

``height``
    A gentle penalty for a large vertical offset. "Left of the laptop" does not
    normally reach the ceiling light.

``proximity``
    A weak preference for the nearer of two objects on the correct side. "In
    front of the sofa" means the coffee table, not the bookshelf on the far wall
    that also happens to be on that side. Weighted low, with a floor, so it can
    break a tie but never overturn the direction terms.

None of these terms know which frame they are in. They take the frame's axes and
project. That separation is the point: swapping the frame changes the answer
without changing a line of the geometry.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..frames.reference_frame import ReferenceFrame
from ..geom.obb import OBB, Interval
from .base import (RelationConfig, RelationScore, RelationSpec, gmean, ramp,
                   register)

# axis index and sign for each direction word, in frame coordinates
# (0 = right, 1 = front, 2 = up)
DIRECTIONS = {
    "left": (0, -1),
    "right": (0, +1),
    "front": (1, +1),
    "behind": (1, -1),
}

register(RelationSpec("left", "projective", True, 1,
                      aliases=("left of", "to the left of", "on the left of",
                               "left hand side", "leftward of", "left side"),
                      doc="target is on the frame's negative right axis"))
register(RelationSpec("right", "projective", True, 1,
                      aliases=("right of", "to the right of", "on the right of",
                               "right hand side", "rightward of", "right side"),
                      doc="target is on the frame's positive right axis"))
register(RelationSpec("front", "projective", True, 1,
                      aliases=("in front of", "infront of", "ahead of",
                               "before", "in front"),
                      doc="target is on the side the frame calls the front"))
register(RelationSpec("behind", "projective", True, 1,
                      aliases=("in back of", "at the back of", "back of",
                               "beyond", "behind of"),
                      doc="target is on the far side from the frame's front"))


def _cone_term(dx: float, dy: float, axis: int, sign: int,
               cfg: RelationConfig) -> float:
    """How well the horizontal bearing agrees with the requested direction."""
    along = (dx if axis == 0 else dy) * sign
    across = dy if axis == 0 else dx
    r = float(np.hypot(along, across))
    if r < 1e-6:
        return 0.0
    # arctan2 already returns the obtuse angle for negative `along`, so no
    # separate branch is needed. There used to be one; over a 2001x401 grid its
    # maximum effect was 2.8e-14.
    ang = float(np.rad2deg(np.arctan2(abs(across), along)))
    return 1.0 - ramp(ang, cfg.cone_full_credit_deg, cfg.cone_half_angle_deg)


def _separation_term(t_iv: Interval, a_iv: Interval, sign: int,
                     cfg: RelationConfig) -> float:
    """Fraction of the target beyond the anchor's face, ramped."""
    bound = a_iv.hi if sign > 0 else a_iv.lo
    frac = t_iv.fraction_beyond(bound, sign)
    return ramp(frac, cfg.lateral_separation_zero, cfg.lateral_separation_full)


def _depth_term(t_iv: Interval, a_iv: Interval, cfg: RelationConfig) -> float:
    """Overlap on the orthogonal horizontal axis, with slack."""
    gap = t_iv.gap_to(a_iv)
    if gap <= 0.0:
        return 1.0
    return 1.0 - ramp(gap, 0.0, cfg.depth_overlap_slack)


def _proximity_terms(t: OBB, a: OBB, cfg: RelationConfig):
    """Returns (preference, locality_gate), both in [0, 1].

    Two different jobs, and they must not be combined into one:

    * **preference** is a weak tie-break inside the weighted mean -- of two
      objects on the right side, the nearer one is meant.
    * **locality** is a multiplicative *gate*, because a projective relation
      presupposes that the two objects share a local context. A low-weight term
      inside a geometric mean cannot express that: with `w_proximity = 0.35`
      against a total weight of 3.35, a term of 0.18 only pulls the score from
      0.94 to 0.84, so "the monitor to the left of the keyboard" was still
      satisfied by a monitor on a different desk 3.2 m away. A gate can say no.

    Both scale with the anchor's size, which is what lets "in front of the
    whiteboard" reach across a room while "left of the keyboard" does not.
    """
    from ..geom.obb import obb_gap
    gap = max(0.0, obb_gap(t, a))
    ref = max(0.35, a.horizontal_radius)
    lo = cfg.proximity_full_factor * ref
    hi = cfg.proximity_zero_factor * ref
    cut = cfg.proximity_cutoff_factor * ref
    preference = cfg.proximity_floor + (1.0 - cfg.proximity_floor) * (
        1.0 - ramp(gap, lo, hi))
    locality = 1.0 - ramp(gap, hi, cut)
    return float(preference), float(locality)


def _height_term(t: OBB, a: OBB, cfg: RelationConfig) -> float:
    dz = abs(float(t.center[2] - a.center[2]))
    overlap = Interval(t.bottom, t.top).overlap(Interval(a.bottom, a.top))
    if overlap > 1e-3:
        return 1.0
    return 1.0 - ramp(dz, cfg.height_band, cfg.height_band_hard)


def projective_score(target: OBB, anchor: OBB, frame: ReferenceFrame,
                     direction: str, cfg: RelationConfig,
                     target_points: Optional[np.ndarray] = None) -> RelationScore:
    """Score "target is <direction> of anchor" under `frame`."""
    if direction not in DIRECTIONS:
        raise ValueError(f"unknown projective direction {direction!r}")
    if not frame.available:
        return RelationScore(0.0, {}, [f"frame unavailable: {frame.reason}"])

    axis, sign = DIRECTIONS[direction]
    other = 1 - axis

    f = frame.at(anchor.center)
    c = f.coords(target.center[None, :])[0]
    cone = _cone_term(float(c[0]), float(c[1]), axis, sign, cfg)

    t_along = f.interval(target, axis)
    a_along = f.interval(anchor, axis)
    sep = _separation_term(t_along, a_along, sign, cfg)

    t_across = f.interval(target, other)
    a_across = f.interval(anchor, other)
    depth = _depth_term(t_across, a_across, cfg)

    height = _height_term(target, anchor, cfg)
    prox, locality = _proximity_terms(target, anchor, cfg)

    value = gmean([cone, sep, depth, height, prox],
                  [cfg.w_cone, cfg.w_separation, cfg.w_depth, cfg.w_height,
                   cfg.w_proximity]) * locality

    notes = []
    if locality < 0.05:
        notes.append(
            f"the two are too far apart to be described this way "
            f"(surface gap beyond {cfg.proximity_cutoff_factor:.0f}x the "
            f"anchor's radius)")
    if frame.confidence < 0.5:
        notes.append(f"low frame confidence {frame.confidence:.2f} ({frame.kind})")
    return RelationScore(value, {
        "cone": cone, "separation": sep, "depth": depth, "height": height,
        "proximity": prox, "locality": locality,
        "signed_offset": float(c[axis]) * sign,
        "orthogonal_offset": float(c[other]),
        "frame_confidence": frame.confidence,
    }, notes)


def signed_axis_value(target: OBB, frame: ReferenceFrame, axis: int = 0) -> float:
    """Where a target sits along one frame axis. Used by ordinal ordering."""
    return float(frame.coords(target.center[None, :])[0][axis])

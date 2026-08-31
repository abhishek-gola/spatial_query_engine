"""Proximity and betweenness: near, next to, beside, adjacent, between.

Two decisions worth stating.

Surface distance, not centroid distance. A mug on the corner of a three-metre
table has a centroid 1.5 m from the table's centroid, and is 2 cm from the
table. "Next to" means the second number.

Scale-relative thresholds. Two mugs 40 cm apart are not next to each other; two
sofas 40 cm apart are. The threshold grows with the size of the objects
involved, which is why `near_size_factor` exists and why no fixed metre value
appears in the code.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..geom.obb import OBB, Interval, obb_gap
from ..geom.pointcloud import cloud_quantile_gap
from ..geom.transforms import horizontal, normalize
from .base import (RelationConfig, RelationScore, RelationSpec, gmean, ramp,
                   register)

register(RelationSpec("near", "proximity", False, 1,
                      aliases=("close to", "closest to", "by", "at", "nearby",
                               "near to", "around", "beside of", "nearest",
                               "nearest to"),
                      doc="target is close to the anchor"))
register(RelationSpec("next_to", "proximity", False, 1,
                      aliases=("next to", "beside", "adjacent to", "alongside",
                               "side by side with", "nextto", "adjacent"),
                      doc="target is close, at a similar height, side by side"))
register(RelationSpec("far_from", "proximity", False, 1,
                      aliases=("far from", "away from", "distant from",
                               "farthest from", "furthest from", "farthest",
                               "furthest"),
                      doc="target is far from the anchor"))
register(RelationSpec("between", "proximity", False, 2,
                      aliases=("in between", "in the middle of", "amid"),
                      doc="target lies between two anchors"))


def surface_gap(target: OBB, anchor: OBB,
                target_points: Optional[np.ndarray] = None,
                anchor_points: Optional[np.ndarray] = None) -> float:
    """Surface-to-surface distance, from point clouds when we have them."""
    if (target_points is not None and len(target_points) > 8
            and anchor_points is not None and len(anchor_points) > 8):
        return cloud_quantile_gap(target_points, anchor_points, q=0.02)
    return max(0.0, obb_gap(target, anchor))


def near_scale(target: OBB, anchor: OBB, cfg: RelationConfig) -> float:
    """The distance at which "near" stops being fully true, for this pair."""
    size = 0.5 * (target.horizontal_radius + anchor.horizontal_radius)
    return cfg.near_base + cfg.near_size_factor * size


def near_score(target: OBB, anchor: OBB, cfg: RelationConfig,
               target_points=None, anchor_points=None) -> RelationScore:
    gap = surface_gap(target, anchor, target_points, anchor_points)
    d0 = near_scale(target, anchor, cfg)
    value = 1.0 - ramp(gap, d0, d0 * cfg.near_zero_multiplier)
    return RelationScore(value, {"surface_gap": gap, "near_scale": d0})


def far_score(target: OBB, anchor: OBB, cfg: RelationConfig,
              target_points=None, anchor_points=None,
              room_diagonal: float = 6.0) -> RelationScore:
    gap = surface_gap(target, anchor, target_points, anchor_points)
    value = ramp(gap, 0.15 * room_diagonal, 0.55 * room_diagonal)
    return RelationScore(value, {"surface_gap": gap,
                                 "room_diagonal": room_diagonal})


def next_to_score(target: OBB, anchor: OBB, cfg: RelationConfig,
                  target_points=None, anchor_points=None) -> RelationScore:
    """Near, plus side-by-side rather than stacked, plus a similar height.

    The extra conditions are what separates "next to" from "near": a lamp on the
    table is near the table but not next to it.
    """
    gap = surface_gap(target, anchor, target_points, anchor_points)
    d0 = near_scale(target, anchor, cfg)
    close = 1.0 - ramp(gap, d0, d0 * cfg.near_zero_multiplier)

    dz = abs(float(target.center[2] - anchor.center[2]))
    z_overlap = Interval(target.bottom, target.top).iou(
        Interval(anchor.bottom, anchor.top))
    same_level = max(1.0 - ramp(dz, cfg.beside_height_tol,
                                cfg.beside_height_tol * 3.0), z_overlap)

    d = target.center - anchor.center
    horiz = float(np.linalg.norm(d[:2]))
    lateral = ramp(horiz / max(abs(float(d[2])) + 1e-6, 1e-6), 0.6, 1.5)

    value = gmean([close, same_level, lateral], [1.0, 0.7, 0.5])
    return RelationScore(value, {"surface_gap": gap, "near_scale": d0,
                                 "close": close, "same_level": same_level,
                                 "lateral": lateral, "z_iou": z_overlap})


def between_score(target: OBB, anchor_a: OBB, anchor_b: OBB,
                  cfg: RelationConfig) -> RelationScore:
    """Target inside the corridor joining the two anchors.

    Frame-independent: "between the sofa and the table" needs no viewpoint,
    which makes it a useful control in the benchmark against the projective
    relations on the same objects.
    """
    a, b = anchor_a.center, anchor_b.center
    ab = b - a
    L = float(np.linalg.norm(ab[:2]))
    if L < 1e-3:
        return RelationScore(0.0, {}, ["the two anchors are at the same place"])
    u = normalize(np.array([ab[0], ab[1], 0.0]))
    d = target.center - a
    t = float(np.dot(np.array([d[0], d[1], 0.0]), u)) / L
    perp = float(np.linalg.norm(np.array([d[0], d[1], 0.0]) - t * L * u))

    span = min(ramp(t, cfg.between_span_lo * 0.5, cfg.between_span_lo),
               1.0 - ramp(t, cfg.between_span_hi, cfg.between_span_hi
                          + cfg.between_span_lo))
    corridor_w = cfg.between_corridor + 0.5 * (
        anchor_a.horizontal_radius + anchor_b.horizontal_radius)
    corridor = 1.0 - ramp(perp, corridor_w * 0.5, corridor_w * 1.5)
    value = gmean([max(span, 1e-6), max(corridor, 1e-6)], [1.0, 0.8])
    return RelationScore(value, {"along_fraction": t, "perpendicular": perp,
                                 "corridor_width": corridor_w,
                                 "span": span, "corridor": corridor})

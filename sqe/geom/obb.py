"""Gravity-aligned (yaw-only) oriented boxes, and the 1-D interval algebra the
projective relations are built on.

Why yaw-only: indoor objects sit on the floor. A full 3-DoF box fit on a noisy
partial point cloud routinely tilts a bookshelf by 15 degrees, and that tilt
then leaks straight into "left of", which is the exact failure mode this repo
is trying to measure rather than introduce. Up is pinned to gravity and only
the heading is fitted.

The fitted heading carries a 90-degree and a 180-degree ambiguity. We resolve
the 90 by convention (local x is the longer side) and leave the 180 alone --
`sqe.perception.orientation` decides which way is "front", with a confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .transforms import horizontal, normalize, rot_about_up

WORLD_UP = np.array([0.0, 0.0, 1.0])


# --------------------------------------------------------------------------
# 1-D intervals
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    @property
    def length(self) -> float:
        return max(0.0, self.hi - self.lo)

    @property
    def mid(self) -> float:
        return 0.5 * (self.lo + self.hi)

    def overlap(self, other: "Interval") -> float:
        return max(0.0, min(self.hi, other.hi) - max(self.lo, other.lo))

    def iou(self, other: "Interval") -> float:
        inter = self.overlap(other)
        union = max(self.hi, other.hi) - min(self.lo, other.lo)
        return inter / union if union > 1e-9 else 1.0

    def contained_fraction(self, other: "Interval") -> float:
        """How much of *self* lies inside `other`, in [0, 1]."""
        if self.length < 1e-9:
            return 1.0 if (other.lo - 1e-9 <= self.lo <= other.hi + 1e-9) else 0.0
        return self.overlap(other) / self.length

    def gap_to(self, other: "Interval") -> float:
        """Signed 1-D gap: positive when disjoint, negative when overlapping
        (magnitude = overlap length)."""
        return max(other.lo - self.hi, self.lo - other.hi, -self.overlap(other))

    def fraction_beyond(self, bound: float, sign: int) -> float:
        """Fraction of self strictly on the `sign` side of `bound`.

        `sign = +1` counts the part above `bound`, `sign = -1` the part below.
        This is what "is the mug really to the left of the table, or just
        overlapping it" reduces to.
        """
        if self.length < 1e-9:
            return 1.0 if (self.lo - bound) * sign > 0 else 0.0
        if sign > 0:
            return max(0.0, self.hi - max(self.lo, bound)) / self.length
        return max(0.0, min(self.hi, bound) - self.lo) / self.length


# --------------------------------------------------------------------------
# Oriented box
# --------------------------------------------------------------------------

@dataclass
class OBB:
    """Yaw-only oriented box.

    `R` columns are the local x, y, z axes in world coordinates; column 2 is
    always world up. `half` are half-extents along those axes.
    """
    center: np.ndarray
    half: np.ndarray
    R: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=np.float64).reshape(3)
        self.half = np.abs(np.asarray(self.half, dtype=np.float64).reshape(3))
        self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)

    # -- basic geometry ---------------------------------------------------
    @property
    def extent(self) -> np.ndarray:
        return 2.0 * self.half

    @property
    def volume(self) -> float:
        return float(np.prod(self.extent))

    @property
    def footprint_area(self) -> float:
        return float(self.extent[0] * self.extent[1])

    @property
    def height(self) -> float:
        return float(self.extent[2])

    @property
    def top(self) -> float:
        return float(self.center[2] + self.half[2])

    @property
    def bottom(self) -> float:
        return float(self.center[2] - self.half[2])

    @property
    def yaw(self) -> float:
        """Heading of local +x, radians, in world XY."""
        return float(np.arctan2(self.R[1, 0], self.R[0, 0]))

    @property
    def long_axis(self) -> np.ndarray:
        """World direction of the longer horizontal side."""
        return self.R[:, 0] if self.half[0] >= self.half[1] else self.R[:, 1]

    @property
    def short_axis(self) -> np.ndarray:
        return self.R[:, 1] if self.half[0] >= self.half[1] else self.R[:, 0]

    def corners(self) -> np.ndarray:
        signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
        return self.center + (signs * self.half) @ self.R.T

    def to_local(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
        return (pts - self.center) @ self.R

    def contains(self, pts: np.ndarray, pad: float = 0.0) -> np.ndarray:
        local = np.abs(self.to_local(pts))
        return np.all(local <= self.half + pad, axis=1)

    # -- projections ------------------------------------------------------
    def interval(self, axis: np.ndarray) -> Interval:
        """Exact support interval of the box along a world direction."""
        axis = normalize(axis)
        c = float(np.dot(self.center, axis))
        rad = float(np.sum(self.half * np.abs(self.R.T @ axis)))
        return Interval(c - rad, c + rad)

    def radius_along(self, axis: np.ndarray) -> float:
        axis = normalize(axis)
        return float(np.sum(self.half * np.abs(self.R.T @ axis)))

    @property
    def horizontal_radius(self) -> float:
        return float(np.hypot(self.half[0], self.half[1]))

    def transformed(self, R: np.ndarray = None, t: np.ndarray = None) -> "OBB":
        R = np.eye(3) if R is None else np.asarray(R, dtype=np.float64)
        t = np.zeros(3) if t is None else np.asarray(t, dtype=np.float64)
        return OBB(R @ self.center + t, self.half.copy(), R @ self.R)

    def to_dict(self) -> dict:
        return {"center": self.center.tolist(),
                "extent": self.extent.tolist(),
                "yaw": self.yaw,
                "R": self.R.tolist()}

    @staticmethod
    def from_dict(d: dict) -> "OBB":
        if "R" in d and d["R"] is not None:
            R = np.asarray(d["R"], dtype=np.float64)
        else:
            R = rot_about_up(float(d.get("yaw", 0.0)))
        half = 0.5 * np.asarray(d["extent"], dtype=np.float64)
        return OBB(np.asarray(d["center"], dtype=np.float64), half, R)


def aabb(points: np.ndarray) -> OBB:
    points = np.asarray(points, dtype=np.float64)
    lo, hi = points.min(axis=0), points.max(axis=0)
    return OBB(0.5 * (lo + hi), 0.5 * (hi - lo), np.eye(3))


def _min_area_rect(xy: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Minimum-area enclosing rectangle of 2-D points via rotating calipers.

    Returns (angle of rectangle's first axis, centre, half-extents).
    """
    from scipy.spatial import ConvexHull, QhullError

    xy = np.asarray(xy, dtype=np.float64)
    uniq = np.unique(np.round(xy, 6), axis=0)
    if len(uniq) < 3:
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        return 0.0, 0.5 * (lo + hi), 0.5 * (hi - lo)
    try:
        hull = uniq[ConvexHull(uniq).vertices]
    except (QhullError, ValueError):
        # degenerate / collinear
        c = uniq.mean(axis=0)
        d = uniq - c
        _, _, Vt = np.linalg.svd(d, full_matrices=False)
        ang = float(np.arctan2(Vt[0, 1], Vt[0, 0]))
        proj = d @ np.array([[np.cos(ang), -np.sin(ang)],
                             [np.sin(ang), np.cos(ang)]])
        lo, hi = proj.min(axis=0), proj.max(axis=0)
        mid = 0.5 * (lo + hi)
        Rm = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        return ang, c + Rm @ mid, 0.5 * (hi - lo)

    edges = np.roll(hull, -1, axis=0) - hull
    angles = np.arctan2(edges[:, 1], edges[:, 0]) % (np.pi / 2.0)
    angles = np.unique(np.round(angles, 9))

    best = None
    for a in angles:
        ca, sa = np.cos(a), np.sin(a)
        # rotate hull by -a so the candidate edge lies on the x axis
        rot = np.array([[ca, sa], [-sa, ca]])
        p = hull @ rot.T
        lo, hi = p.min(axis=0), p.max(axis=0)
        area = float(np.prod(hi - lo))
        if best is None or area < best[0] - 1e-12:
            mid = 0.5 * (lo + hi)
            centre = np.array([[ca, -sa], [sa, ca]]) @ mid
            best = (area, a, centre, 0.5 * (hi - lo))
    _, ang, centre, half = best
    return float(ang), centre, half


def fit_obb(points: np.ndarray, up: np.ndarray = WORLD_UP,
            trim: float = 0.0, min_extent: float = 1e-3) -> OBB:
    """Fit a gravity-aligned yaw-only box to `points`.

    `trim` (each side, as a fraction) clips extremes before fitting, for noisy
    reconstructed clouds where a handful of flyers would otherwise double the
    box. It is 0 by default: ground-truth clouds should be fitted exactly.

    Local x ends up as the longer horizontal side, which fixes the 90-degree
    ambiguity of the rectangle fit deterministically.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        raise ValueError("cannot fit a box to zero points")
    up = normalize(up)

    # Build a horizontal 2-D basis to work in.
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(tmp, up))) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e0 = horizontal(tmp, up)
    e1 = np.cross(up, e0)

    z = points @ up
    xy = np.stack([points @ e0, points @ e1], axis=1)

    if trim > 0.0 and len(points) > 20:
        qlo, qhi = 100.0 * trim, 100.0 * (1.0 - trim)
        keep = np.ones(len(points), dtype=bool)
        for col in (xy[:, 0], xy[:, 1], z):
            lo, hi = np.percentile(col, [qlo, qhi])
            keep &= (col >= lo) & (col <= hi)
        if keep.sum() >= 4:
            xy, z = xy[keep], z[keep]

    ang, centre2, half2 = _min_area_rect(xy)
    zlo, zhi = float(z.min()), float(z.max())

    a0 = np.cos(ang) * e0 + np.sin(ang) * e1
    a1 = -np.sin(ang) * e0 + np.cos(ang) * e1
    # `centre2` already comes back in (e0, e1) coordinates, not in the
    # rotated rectangle frame -- do not rotate it again.
    center = centre2[0] * e0 + centre2[1] * e1 + 0.5 * (zlo + zhi) * up
    half = np.array([half2[0], half2[1], 0.5 * (zhi - zlo)])

    # local x = longer horizontal side
    if half[1] > half[0]:
        a0, a1 = a1, -a0
        half = np.array([half[1], half[0], half[2]])

    half = np.maximum(half, min_extent)
    return OBB(center, half, np.stack([a0, a1, up], axis=1))


def obb_from_extent_yaw(center, extent, yaw: float, up=WORLD_UP) -> OBB:
    R = rot_about_up(yaw, up)
    return OBB(np.asarray(center, float), 0.5 * np.asarray(extent, float), R)


# --------------------------------------------------------------------------
# Pairwise measures
# --------------------------------------------------------------------------

def horizontal_footprint_overlap(a: OBB, b: OBB, axes: Optional[np.ndarray] = None) -> float:
    """Fraction of `a`'s footprint that overlaps `b`'s, approximated on the
    two horizontal axes of `b` (exact when the boxes share a heading).

    Used by the vertical relations, where "on" needs the target to actually sit
    over the supporter rather than merely be higher than it.
    """
    if axes is None:
        axes = [b.R[:, 0], b.R[:, 1]]
    frac = 1.0
    for ax in axes:
        frac = min(frac, a.interval(ax).contained_fraction(b.interval(ax)))
    return float(frac)


def obb_center_distance(a: OBB, b: OBB) -> float:
    return float(np.linalg.norm(a.center - b.center))


def obb_gap(a: OBB, b: OBB) -> float:
    """Conservative separation between two boxes: the largest gap found over
    the separating-axis candidates. Positive means disjoint.

    This is a lower bound on the true surface distance and is only used when
    point clouds are unavailable; `sqe.geom.pointcloud.cloud_gap` is preferred
    because it is the real thing.
    """
    axes = [a.R[:, 0], a.R[:, 1], a.R[:, 2], b.R[:, 0], b.R[:, 1], b.R[:, 2]]
    for i in range(3):
        for j in range(3):
            c = np.cross(a.R[:, i], b.R[:, j])
            if np.linalg.norm(c) > 1e-6:
                axes.append(normalize(c))
    return max(a.interval(ax).gap_to(b.interval(ax)) for ax in axes)

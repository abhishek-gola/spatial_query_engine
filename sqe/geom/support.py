"""Horizontal surfaces and support relations.

Two jobs:

1. Find the individual horizontal surfaces *inside* one object, so that "the
   middle shelf of the bookshelf" has something to refer to. A bookshelf is a
   single instance in every 3-D segmentation dataset; its shelves are not
   annotated anywhere and have to be recovered.
2. Decide what is resting on what, which is how `on` is distinguished from
   `above` and how ordinal queries get scoped to one surface.

Level counting is honestly ambiguous and treated as such. A unit with boards at
0.45 / 0.90 / 1.35 has four compartments, and "the middle shelf" then has no
single referent. `middle_level` returns both candidates instead of picking one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .obb import OBB, Interval, horizontal_footprint_overlap
from .pointcloud import estimate_normals, occupied_area_2d
from .transforms import normalize

WORLD_UP = np.array([0.0, 0.0, 1.0])


@dataclass
class Level:
    """A horizontal surface found inside an object."""
    z: float
    area: float                  # observed area of the surface, m^2
    support: int                 # points on it
    is_object_top: bool = False
    index: int = -1              # 0 = lowest, set by `shelf_levels`

    def to_dict(self) -> dict:
        return {"z": self.z, "area": self.area, "support": self.support,
                "is_object_top": self.is_object_top, "index": self.index}


def detect_levels(points: np.ndarray,
                  obb: Optional[OBB] = None,
                  up: np.ndarray = WORLD_UP,
                  normals: Optional[np.ndarray] = None,
                  bin_size: float = 0.02,
                  min_gap: float = 0.12,
                  min_area: float = 0.02,
                  min_area_fraction: float = 0.12,
                  normal_tol_deg: float = 25.0,
                  cell: float = 0.05) -> List[Level]:
    """Find horizontal surfaces in a point cloud.

    Points are restricted to those with an up-facing normal, then their heights
    are binned and peaks taken with a minimum vertical separation. Each peak's
    area is measured as occupied cells, and peaks that are too small in absolute
    terms or relative to the largest surface are dropped -- otherwise the top
    edge of every book on a shelf registers as its own shelf.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 60:
        return []
    up = normalize(up)
    if normals is None:
        normals = estimate_normals(points, k=18, orient_up=up)
    cos_tol = np.cos(np.deg2rad(normal_tol_deg))
    flat = np.abs(normals @ up) > cos_tol
    if flat.sum() < 40:
        return []
    pf = points[flat]
    z = pf @ up

    nbins = max(4, int(np.ceil((z.max() - z.min()) / bin_size)) + 1)
    hist, edges = np.histogram(z, bins=nbins)
    if hist.max() < 10:
        return []

    # local maxima with a minimum separation, taken greedily by height of peak
    min_sep_bins = max(1, int(round(min_gap / bin_size)))
    order = np.argsort(-hist)
    chosen: List[int] = []
    for b in order:
        if hist[b] < max(10, 0.06 * hist.max()):
            break
        if all(abs(b - c) >= min_sep_bins for c in chosen):
            chosen.append(int(b))

    # horizontal basis for area measurement
    if obb is not None:
        e0, e1 = obb.R[:, 0], obb.R[:, 1]
    else:
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(float(tmp @ up)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        e0 = normalize(tmp - float(tmp @ up) * up)
        e1 = np.cross(up, e0)

    levels: List[Level] = []
    for b in chosen:
        lo, hi = edges[b] - bin_size, edges[b + 1] + bin_size
        sel = (z >= lo) & (z <= hi)
        n = int(sel.sum())
        if n < 20:
            continue
        pts = pf[sel]
        area = occupied_area_2d(np.stack([pts @ e0, pts @ e1], axis=1), cell)
        # a support surface's usable height is its *top*, not the middle of
        # the slab, so take a high quantile rather than the median
        levels.append(Level(z=float(np.quantile(pts @ up, 0.8)), area=float(area),
                            support=n))
    if not levels:
        return []

    biggest = max(l.area for l in levels)
    levels = [l for l in levels
              if l.area >= min_area and l.area >= min_area_fraction * biggest]
    levels.sort(key=lambda l: l.z)

    if obb is not None:
        for l in levels:
            l.is_object_top = abs(l.z - obb.top) < max(0.04, 0.03 * obb.height)
    for i, l in enumerate(levels):
        l.index = i
    return levels


def shelf_levels(levels: Sequence[Level], obb: Optional[OBB] = None,
                 drop_object_top: bool = True) -> List[Level]:
    """The levels a person would call "shelves", lowest first.

    The unit's own top face is dropped by default: people say "on top of the
    bookshelf", not "on the top shelf", when they mean that surface.
    """
    out = [l for l in levels if not (drop_object_top and l.is_object_top)]
    # One surface is a table top, not a shelf system. Reporting "the middle
    # shelf of the coffee table" for a single detected plane is worse than
    # reporting nothing.
    if len(out) < 2:
        return []
    out.sort(key=lambda l: l.z)
    for i, l in enumerate(out):
        l.index = i
    return out


def middle_level(levels: Sequence[Level]) -> Tuple[List[Level], bool]:
    """The "middle" shelf, or both candidates when the count is even.

    Returns `(candidates, ambiguous)`. An even number of shelves has no middle,
    and saying so is more useful than rounding.
    """
    n = len(levels)
    if n == 0:
        return [], False
    if n % 2 == 1:
        return [levels[n // 2]], False
    return [levels[n // 2 - 1], levels[n // 2]], True


def ordinal_level(levels: Sequence[Level], word: str) -> Tuple[List[Level], bool]:
    """Resolve 'top' / 'bottom' / 'middle' / 'second from the bottom' etc."""
    if not levels:
        return [], False
    w = (word or "").strip().lower()
    if w in ("top", "topmost", "highest", "upper"):
        return [levels[-1]], False
    if w in ("bottom", "bottommost", "lowest", "lower"):
        return [levels[0]], False
    if w in ("middle", "centre", "center", "central"):
        return middle_level(levels)
    return [], False


def assign_level(target: OBB, levels: Sequence[Level],
                 tol: float = 0.08) -> Optional[int]:
    """Index of the level a target is resting on, if any.

    Matches the target's underside to a surface height. `tol` is generous
    because reconstruction thins surfaces and small objects lose their bottom
    few centimetres to the supporter.
    """
    if not levels:
        return None
    bottom = target.bottom
    best, best_d = None, tol
    for l in levels:
        d = abs(bottom - l.z)
        if d < best_d:
            best, best_d = l.index, d
    return best


# --------------------------------------------------------------------------
# support between objects
# --------------------------------------------------------------------------

def _contact(gap: float, tol: float) -> float:
    """Vertical contact credit for a gap of `gap` metres.

    Full credit inside `tol`, then a linear tail out to `3 * tol`, rather than
    decaying to zero at `tol`. The earlier hard version scored a monitor sitting
    5.5 cm above a desk top -- because its stand is not part of the instance --
    at 0.08, so "the tallest object on the desk" came back as the laptop. Real
    reconstructions routinely leave gaps of that size under an object, and a
    graceful tail is the difference between a threshold that models contact and
    one that models mesh quality.
    """
    a = abs(gap)
    if a <= tol:
        return 1.0
    return float(np.clip(1.0 - (a - tol) / (2.0 * max(tol, 1e-6)), 0.0, 1.0))


def support_score(target: OBB, supporter: OBB,
                  contact_tol: float = 0.08,
                  min_overlap: float = 0.35) -> float:
    """How strongly `target` looks like it is resting on `supporter`, in [0,1].

    Combines a vertical contact term (the target's underside near the
    supporter's top) with a footprint containment term (the target actually over
    the supporter). Both are needed: a lamp beside a table satisfies the first,
    a book in a drawer satisfies the second.
    """
    gap = target.bottom - supporter.top
    if gap < -0.5 * target.height - contact_tol:
        return 0.0
    contact = _contact(gap, contact_tol)
    overlap = horizontal_footprint_overlap(target, supporter)
    if overlap < min_overlap:
        return 0.0
    return float(contact * overlap)


def support_score_on_level(target: OBB, supporter: OBB, level_z: float,
                           contact_tol: float = 0.08,
                           min_overlap: float = 0.30) -> float:
    """Like `support_score` but against an internal surface at `level_z`."""
    gap = target.bottom - level_z
    contact = _contact(gap, contact_tol)
    if contact <= 0.0:
        return 0.0
    overlap = horizontal_footprint_overlap(target, supporter)
    if overlap < min_overlap:
        return 0.0
    return float(contact * overlap)

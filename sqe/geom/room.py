"""Room-level structure: gravity, floor, walls, and the Manhattan axes that the
allocentric ("world canonical") reference frame is built on.

The part worth reading is `canonical_forward`. A Manhattan frame fixes the room
axes only up to a 4-fold rotation, so "the left side of the room" is undefined
until something breaks that tie. Pipelines usually inherit whatever rotation the
dataset's alignment happened to produce and never mention it. Here the tie-break
is a named convention, all four candidates are kept, and the confidence that the
choice is meaningful is reported alongside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .obb import OBB, WORLD_UP
from .pointcloud import (Plane, estimate_normals, fit_plane_ransac,
                         occupied_area_2d, triangle_normals)
from .transforms import basis_from_forward_up, horizontal, normalize

FORWARD_CONVENTIONS = ("composite", "principal_wall", "trajectory",
                       "longest_axis", "dataset_axes")

#: Below this margin the room-canonical frame is treated as genuinely
#: undetermined and any query that leans on it gets flagged rather than
#: answered. See docs/METHOD.md.
FORWARD_MARGIN_AMBIGUOUS = 0.12


@dataclass
class Wall:
    direction: np.ndarray      # outward-from-room-centre normal, horizontal
    offset: float              # position along `direction`
    area: float                # observed extent, m^2
    support: int               # number of points


@dataclass
class RoomStructure:
    up: np.ndarray
    floor_z: float
    ceiling_z: Optional[float]
    axes: np.ndarray                  # 3x3, columns = primary, secondary, up
    axis_confidence: float            # 0..1, how Manhattan the room actually is
    bounds: OBB                       # room extent in the Manhattan frame
    walls: List[Wall] = field(default_factory=list)
    canonical_forward: np.ndarray = None
    forward_convention: str = "unset"
    forward_confidence: float = 0.0
    forward_candidates: List[np.ndarray] = field(default_factory=list)
    notes: dict = field(default_factory=dict)

    @property
    def center(self) -> np.ndarray:
        return self.bounds.center

    @property
    def diagonal(self) -> float:
        return float(np.linalg.norm(self.bounds.extent[:2]))

    def to_dict(self) -> dict:
        return {
            "up": self.up.tolist(),
            "floor_z": self.floor_z,
            "ceiling_z": self.ceiling_z,
            "axes": self.axes.tolist(),
            "axis_confidence": self.axis_confidence,
            "bounds": self.bounds.to_dict(),
            "walls": [{"direction": w.direction.tolist(), "offset": w.offset,
                       "area": w.area, "support": w.support} for w in self.walls],
            "canonical_forward": None if self.canonical_forward is None
            else self.canonical_forward.tolist(),
            "forward_convention": self.forward_convention,
            "forward_confidence": self.forward_confidence,
            "forward_candidates": [c.tolist() for c in self.forward_candidates],
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(d: dict) -> "RoomStructure":
        return RoomStructure(
            up=np.asarray(d["up"], float),
            floor_z=float(d["floor_z"]),
            ceiling_z=None if d.get("ceiling_z") is None else float(d["ceiling_z"]),
            axes=np.asarray(d["axes"], float),
            axis_confidence=float(d["axis_confidence"]),
            bounds=OBB.from_dict(d["bounds"]),
            walls=[Wall(np.asarray(w["direction"], float), float(w["offset"]),
                        float(w["area"]), int(w["support"])) for w in d.get("walls", [])],
            canonical_forward=None if d.get("canonical_forward") is None
            else np.asarray(d["canonical_forward"], float),
            forward_convention=d.get("forward_convention", "unset"),
            forward_confidence=float(d.get("forward_confidence", 0.0)),
            forward_candidates=[np.asarray(c, float)
                                for c in d.get("forward_candidates", [])],
            notes=d.get("notes", {}),
        )


# --------------------------------------------------------------------------
# Gravity
# --------------------------------------------------------------------------

def estimate_up(points: np.ndarray, normals: Optional[np.ndarray] = None,
                assume: np.ndarray = WORLD_UP,
                max_correction_deg: float = 10.0) -> tuple:
    """Verify (and mildly correct) the up axis using the floor.

    Datasets claim to be gravity-aligned; most are, to within a fraction of a
    degree, but not all frames of all captures. We fit the dominant horizontal
    plane in the bottom slab and use its normal, refusing corrections larger
    than `max_correction_deg` because at that point something is wrong with the
    input and silently rotating the scene would hide it.

    Returns `(up, correction_degrees, ok)`.
    """
    points = np.asarray(points, dtype=np.float64)
    assume = normalize(assume)
    if len(points) < 100:
        return assume, 0.0, True
    h = points @ assume
    lo = np.quantile(h, 0.002)
    slab = points[h < lo + 0.15]
    if len(slab) < 50:
        return assume, 0.0, True
    plane = fit_plane_ransac(slab, thresh=0.02, iters=150,
                             normal_prior=assume, prior_tol_deg=max_correction_deg)
    if plane is None or plane.inlier_ratio < 0.3:
        return assume, 0.0, True
    n = normalize(plane.normal)
    if float(n @ assume) < 0:
        n = -n
    ang = float(np.rad2deg(np.arccos(np.clip(n @ assume, -1, 1))))
    if ang > max_correction_deg:
        return assume, ang, False
    return n, ang, True


# --------------------------------------------------------------------------
# Manhattan axes
# --------------------------------------------------------------------------

def manhattan_axes(points: np.ndarray, up: np.ndarray = WORLD_UP,
                   normals: Optional[np.ndarray] = None,
                   faces: Optional[np.ndarray] = None,
                   vertical_cos_max: float = 0.25):
    """Dominant horizontal axis pair, from the azimuths of vertical surfaces.

    Wall normals are only defined mod 180 degrees and the grid is only defined
    mod 90, so azimuths are mapped through `4 * theta` and averaged on the
    circle. The resultant length of that mean doubles as the confidence: a
    round room or a cluttered scan gives a short resultant, and callers should
    then distrust anything phrased in room-canonical terms.
    """
    points = np.asarray(points, dtype=np.float64)
    up = normalize(up)

    if faces is not None and len(faces) > 0:
        fn, area = triangle_normals(points, faces)
        n_use, w_use = fn, area
    else:
        if normals is None:
            normals = estimate_normals(points, k=24)
        n_use, w_use = normals, np.ones(len(normals))

    vert_mask = np.abs(n_use @ up) < vertical_cos_max
    if vert_mask.sum() < 20:
        axes = basis_from_forward_up(np.array([0.0, 1.0, 0.0]) if abs(up[1]) < 0.9
                                    else np.array([1.0, 0.0, 0.0]), up)
        return np.stack([axes[:, 0], axes[:, 1], up], axis=1), 0.0

    # horizontal 2-D basis
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(float(tmp @ up)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e0 = horizontal(tmp, up)
    e1 = np.cross(up, e0)

    nv = n_use[vert_mask]
    wv = w_use[vert_mask]
    nh = nv - (nv @ up)[:, None] * up
    ln = np.linalg.norm(nh, axis=1)
    keep = ln > 1e-6
    nh, wv = nh[keep] / ln[keep, None], wv[keep]
    theta = np.arctan2(nh @ e1, nh @ e0)

    c = float(np.sum(wv * np.cos(4.0 * theta)))
    s = float(np.sum(wv * np.sin(4.0 * theta)))
    total = float(np.sum(wv)) + 1e-12
    resultant = float(np.hypot(c, s) / total)
    ang = 0.25 * float(np.arctan2(s, c))

    a0 = np.cos(ang) * e0 + np.sin(ang) * e1
    a1 = np.cross(up, a0)

    # fraction of vertical surface area actually snapped to the grid
    resid = np.abs(((theta - ang + np.pi / 4.0) % (np.pi / 2.0)) - np.pi / 4.0)
    snapped = float(np.sum(wv[resid < np.deg2rad(10.0)]) / total)
    confidence = float(np.clip(0.5 * resultant + 0.5 * snapped, 0.0, 1.0))
    return np.stack([a0, a1, up], axis=1), confidence


def detect_walls(points: np.ndarray, axes: np.ndarray, center: np.ndarray,
                 normals: Optional[np.ndarray] = None,
                 slab: float = 0.25, min_support: int = 200,
                 cell: float = 0.10, normal_tol_deg: float = 30.0) -> List[Wall]:
    """One wall per Manhattan direction, taken as the outermost slab of points.

    Area is measured as occupied 10 cm cells in the wall's own plane, not as a
    convex hull: a wall with a doorway in it must come out smaller than a solid
    one, since wall area is what decides the room's canonical forward.

    When per-point normals are available the slab is additionally restricted to
    points whose normal faces along the wall direction, which stops furniture
    standing against the wall from filling in its holes.
    """
    points = np.asarray(points, dtype=np.float64)
    up = axes[:, 2]
    cos_tol = np.cos(np.deg2rad(normal_tol_deg))
    walls: List[Wall] = []
    for sign in (+1, -1):
        for k in (0, 1):
            d = sign * axes[:, k]
            proj = points @ d
            far = float(np.quantile(proj, 0.995))
            mask = proj > far - slab
            if normals is not None:
                mask &= np.abs(normals @ d) > cos_tol
            if int(mask.sum()) < min_support:
                continue
            pw = points[mask]
            other = axes[:, 1 - k]
            area = occupied_area_2d(np.stack([pw @ other, pw @ up], axis=1), cell)
            walls.append(Wall(direction=d, offset=far, area=float(area),
                              support=int(mask.sum())))
    walls.sort(key=lambda w: -w.area)
    return walls


def canonical_forward(room_axes: np.ndarray, walls: List[Wall],
                      center: np.ndarray,
                      convention: str = "composite",
                      trajectory_heading: Optional[np.ndarray] = None,
                      trajectory_concentration: float = 0.0,
                      bounds: Optional[OBB] = None):
    """Break the 4-fold rotation ambiguity of the room frame.

    Conventions:

    * `composite` (default) -- blends two independent signals, wall area and how
      the room was actually looked at:

          score(c) = 0.6 * area(c) / max_area
                   + 0.4 * max(0, heading . c) * heading_concentration

      Rationale: rooms have a business end (the whiteboard, the TV, the window
      wall) and people describe "the left of the room" while facing it; and if
      the capture spent its time pointed one way, that is evidence about which
      way "forward" reads as natural. Neither signal is reliable alone, and when
      they are both weak the margin collapses, which is the answer we want.

    * `principal_wall` -- wall area only.
    * `trajectory` -- mean capture heading only, snapped to the grid.
    * `longest_axis` -- the longer horizontal room axis. Cheap, and included
      deliberately as a baseline because it is roughly what a pipeline gets by
      accident.
    * `dataset_axes` -- take the dataset's own +Y as forward. This is the
      "silently pick one and never mention it" baseline that the benchmark
      compares against.

    Returns `(forward, margin, all_four_candidates)`. `margin` is how much
    better the winner is than the runner-up, normalised to [0, 1]. Below
    `FORWARD_MARGIN_AMBIGUOUS` the room genuinely has no canonical front and the
    resolver must flag rather than answer. A square room with four equal walls
    and a capture that panned all the way round scores ~0, correctly.
    """
    up = room_axes[:, 2]
    cands = [room_axes[:, 0], -room_axes[:, 0], room_axes[:, 1], -room_axes[:, 1]]

    if convention == "dataset_axes":
        f = horizontal(np.array([0.0, 1.0, 0.0]), up)
        if not np.any(f):
            f = horizontal(np.array([1.0, 0.0, 0.0]), up)
        return f, 0.0, cands

    def wall_scores() -> np.ndarray:
        s = np.zeros(4)
        for i, c in enumerate(cands):
            s[i] = sum(w.area for w in walls if float(w.direction @ c) > 0.7)
        m = float(s.max())
        return s / m if m > 1e-9 else s

    def heading_scores() -> np.ndarray:
        s = np.zeros(4)
        if trajectory_heading is None:
            return s
        h = horizontal(trajectory_heading, up)
        if not np.any(h):
            return s
        conc = float(np.clip(trajectory_concentration, 0.0, 1.0))
        for i, c in enumerate(cands):
            s[i] = max(0.0, float(h @ c)) * conc
        return s

    if convention == "composite":
        scores = 0.6 * wall_scores() + 0.4 * heading_scores()
    elif convention == "principal_wall":
        if not walls:
            return cands[0], 0.0, cands
        scores = wall_scores()
    elif convention == "trajectory":
        if trajectory_heading is None:
            return cands[0], 0.0, cands
        scores = heading_scores()
    elif convention == "longest_axis":
        if bounds is None:
            return cands[0], 0.0, cands
        ext = np.array([bounds.extent[0], bounds.extent[0],
                        bounds.extent[1], bounds.extent[1]], float)
        scores = ext / max(1e-9, float(ext.max()))
        # a deterministic nudge so the two ends of the long axis do not tie
        scores = scores + 1e-3 * np.array([1.0, 0.0, 1.0, 0.0])
    else:
        raise ValueError(f"unknown forward convention: {convention!r}")

    order = np.argsort(-scores)
    best, second = float(scores[order[0]]), float(scores[order[1]])
    margin = 0.0 if best <= 1e-9 else float(np.clip(best - second, 0.0, 1.0))
    return cands[int(order[0])], margin, cands


def build_room(points: np.ndarray,
               normals: Optional[np.ndarray] = None,
               faces: Optional[np.ndarray] = None,
               up_hint: np.ndarray = WORLD_UP,
               convention: str = "composite",
               trajectory_heading: Optional[np.ndarray] = None,
               trajectory_concentration: float = 0.0) -> RoomStructure:
    """Full room analysis from a scene cloud (mesh faces used if available)."""
    points = np.asarray(points, dtype=np.float64)
    up, corr_deg, ok = estimate_up(points, normals, up_hint)
    axes, axis_conf = manhattan_axes(points, up, normals, faces)

    h = points @ up
    floor_z = float(np.quantile(h, 0.002))
    ceil_z = float(np.quantile(h, 0.998))
    if ceil_z - floor_z < 1.2:
        ceil_z = None

    # room bounds expressed in the Manhattan frame
    local = points @ axes
    lo, hi = np.quantile(local, 0.001, axis=0), np.quantile(local, 0.999, axis=0)
    mid = 0.5 * (lo + hi)
    bounds = OBB(axes @ mid, 0.5 * (hi - lo), axes)

    walls = detect_walls(points, axes, bounds.center, normals)
    fwd, fwd_conf, cands = canonical_forward(
        axes, walls, bounds.center, convention, trajectory_heading,
        trajectory_concentration, bounds)

    return RoomStructure(
        up=up, floor_z=floor_z, ceiling_z=ceil_z, axes=axes,
        axis_confidence=axis_conf, bounds=bounds, walls=walls,
        canonical_forward=fwd, forward_convention=convention,
        forward_confidence=fwd_conf, forward_candidates=cands,
        notes={"up_correction_deg": corr_deg, "up_check_passed": bool(ok),
               "n_points": int(len(points))},
    )

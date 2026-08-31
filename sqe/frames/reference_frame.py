"""Reference frames, and the sign conventions that make them disagree.

A projective term like "left of the laptop" is not a property of the scene. It
is a property of the scene *plus* a frame, and there are at least five frames a
speaker could plausibly mean. This module builds them all explicitly.

The frames
----------

``egocentric``
    Relative / viewer-centric. Left and right are the *viewer's* left and right;
    "in front of A" means between the viewer and A. Lateral offsets are measured
    by projecting onto the viewer's lateral axis.

``egocentric_bearing``
    The same viewpoint, but left and right are measured as a difference in
    *bearing* rather than as a projected displacement, and front/behind as a
    difference in range. This is what "appears to the left of" means when you
    look at a photograph. It parts company with ``egocentric`` whenever the two
    objects sit at appreciably different depths -- a case that pipelines
    silently get wrong because they only ever implement the projected version.

``egocentric_image``
    Image-plane axes, camera roll included. The one frame that is deliberately
    not gravity-aligned, because "left in the picture" of a tilted phone really
    does tilt.

``intrinsic``
    Object-centric. Left and right are the *anchor's own* left and right, as if
    the anchor were an agent looking out of its front. "In front of the sofa" is
    the side the sofa faces. Available only when the anchor's category has a
    front and we managed to estimate it.

``addressee``
    The mirrored reading of the same anchor: left and right as seen by someone
    standing in front of the anchor, looking at it. "The chair's left" has both
    readings in ordinary English. Note the identity, which the tests check:
    ``addressee(A)`` equals ``egocentric`` for a viewer standing on A's front
    axis.

``world``
    Allocentric / room-canonical. Built on the Manhattan axes and the canonical
    forward from ``sqe.geom.room``, whose 4-fold ambiguity is the frame's main
    weakness and is carried through as a confidence.

The handedness trap
-------------------

Fix the convention that a frame's ``front`` axis points in the direction
"in front of" means, and its ``right`` axis in the direction "to the right of"
means. Then:

* for ``intrinsic`` and ``world``, ``right = front x up`` and the triple is
  right-handed;
* for ``egocentric``, ``front`` points from the anchor back towards the viewer
  while ``right`` stays the viewer's own right, and the triple comes out
  **left-handed**.

That handedness flip is not a bug to be normalised away. It *is* the mirror
error: it is why the mug on your left is on the chair's right, and why a
pipeline that builds one basis and reuses it for both readings is wrong half the
time. `ReferenceFrame.handedness` reports it, and the tests assert it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..geom.obb import OBB, Interval
from ..geom.transforms import (camera_down, camera_forward, camera_right,
                               horizontal, normalize)

WORLD_UP = np.array([0.0, 0.0, 1.0])

EGOCENTRIC_KINDS = ("egocentric", "egocentric_bearing", "egocentric_image")
ANCHOR_KINDS = ("intrinsic", "addressee")
ALLOCENTRIC_KINDS = ("world",)
ALL_KINDS = EGOCENTRIC_KINDS + ANCHOR_KINDS + ALLOCENTRIC_KINDS


@dataclass
class ReferenceFrame:
    """One interpretation of "left", "right", "front" and "behind"."""

    kind: str
    right: np.ndarray
    front: np.ndarray
    up: np.ndarray
    origin: np.ndarray                       # anchor centre the frame is read at
    viewpoint: Optional[np.ndarray] = None   # eye position, egocentric frames only
    confidence: float = 1.0
    available: bool = True
    reason: str = ""
    provenance: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.right = normalize(np.asarray(self.right, float).reshape(3))
        self.front = normalize(np.asarray(self.front, float).reshape(3))
        self.up = normalize(np.asarray(self.up, float).reshape(3))
        self.origin = np.asarray(self.origin, float).reshape(3)
        if self.viewpoint is not None:
            self.viewpoint = np.asarray(self.viewpoint, float).reshape(3)

    # -- basic properties --------------------------------------------------
    @property
    def basis(self) -> np.ndarray:
        """3x3 with columns (right, front, up). Not necessarily right-handed."""
        return np.stack([self.right, self.front, self.up], axis=1)

    @property
    def handedness(self) -> int:
        """+1 right-handed, -1 left-handed. Egocentric frames are -1."""
        return 1 if float(np.linalg.det(self.basis)) > 0 else -1

    @property
    def is_egocentric(self) -> bool:
        return self.kind in EGOCENTRIC_KINDS

    def axis(self, name: str) -> np.ndarray:
        return {"right": self.right, "left": -self.right,
                "front": self.front, "back": -self.front,
                "behind": -self.front, "up": self.up, "down": -self.up}[name]

    # -- coordinates -------------------------------------------------------
    def coords(self, points: np.ndarray) -> np.ndarray:
        """Frame-local coordinates of world points, relative to `origin`.

        Columns are (rightward, frontward, upward) in metres. For
        ``egocentric_bearing`` the lateral and frontal components are the
        nonlinear bearing and range measures, converted to metres at the
        anchor's distance so that they stay comparable with the other frames.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if self.kind != "egocentric_bearing":
            d = pts - self.origin
            return np.stack([d @ self.right, d @ self.front, d @ self.up], axis=1)

        v = self.viewpoint
        up = self.up
        a_h = horizontal(self.origin - v, up)
        radius = float(np.linalg.norm((self.origin - v) - ((self.origin - v) @ up) * up))
        radius = max(radius, 1e-3)
        lat_axis = np.cross(a_h, up)                 # viewer's right at the anchor
        d = pts - v
        d_h = d - (d @ up)[:, None] * up
        rng = np.linalg.norm(d_h, axis=1)
        az = np.arctan2(d_h @ lat_axis, d_h @ a_h)   # 0 = straight at the anchor
        lateral = radius * az                        # arc length, metres
        frontal = radius - rng                       # closer to the viewer = +front
        vertical = (pts - self.origin) @ up
        return np.stack([lateral, frontal, vertical], axis=1)

    def local_basis(self) -> np.ndarray:
        """A linear basis valid near the origin.

        Identical to `basis` for every frame except ``egocentric_bearing``,
        where it is the tangent basis at the anchor. Relations that need object
        *extents* rather than centres project onto this, since a bearing is not
        a direction you can take a support interval along.
        """
        if self.kind != "egocentric_bearing":
            return self.basis
        a_h = horizontal(self.origin - self.viewpoint, self.up)
        lat = np.cross(a_h, self.up)
        return np.stack([normalize(lat), normalize(-a_h), self.up], axis=1)

    def interval(self, obb: OBB, axis: int) -> Interval:
        """Support interval of a box along one of the frame's axes, measured
        from the origin. `axis` is 0 = right, 1 = front, 2 = up."""
        a = self.local_basis()[:, axis]
        iv = obb.interval(a)
        o = float(self.origin @ a)
        return Interval(iv.lo - o, iv.hi - o)

    def mirrored(self, kind: Optional[str] = None) -> "ReferenceFrame":
        """The same frame with left and right swapped."""
        return ReferenceFrame(
            kind=kind or (self.kind + "_mirrored"), right=-self.right,
            front=self.front, up=self.up, origin=self.origin,
            viewpoint=self.viewpoint, confidence=self.confidence,
            available=self.available, reason=self.reason,
            provenance={**self.provenance, "mirrored_from": self.kind})

    def at(self, origin: np.ndarray) -> "ReferenceFrame":
        """The same frame re-anchored at a different origin."""
        return ReferenceFrame(kind=self.kind, right=self.right, front=self.front,
                              up=self.up, origin=np.asarray(origin, float),
                              viewpoint=self.viewpoint,
                              confidence=self.confidence, available=self.available,
                              reason=self.reason, provenance=dict(self.provenance))

    def describe(self) -> str:
        bits = [self.kind]
        if self.viewpoint is not None:
            v = self.viewpoint
            bits.append(f"eye ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})")
        bits.append(f"right ({self.right[0]:+.2f}, {self.right[1]:+.2f})")
        bits.append(f"conf {self.confidence:.2f}")
        if not self.available:
            bits.append(f"UNAVAILABLE: {self.reason}")
        return " | ".join(bits)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "right": self.right.tolist(),
                "front": self.front.tolist(), "up": self.up.tolist(),
                "origin": self.origin.tolist(),
                "viewpoint": None if self.viewpoint is None else self.viewpoint.tolist(),
                "confidence": self.confidence, "available": self.available,
                "reason": self.reason, "handedness": self.handedness,
                "provenance": self.provenance}


def unavailable(kind: str, origin: np.ndarray, reason: str,
                up: np.ndarray = WORLD_UP) -> ReferenceFrame:
    """A placeholder frame that refuses to answer. Used instead of a default,
    so that "we do not know which way the sofa faces" never gets rendered as a
    confident left/right."""
    return ReferenceFrame(kind=kind, right=np.array([1.0, 0.0, 0.0]),
                          front=np.array([0.0, 1.0, 0.0]), up=up, origin=origin,
                          confidence=0.0, available=False, reason=reason)


# --------------------------------------------------------------------------
# constructors
# --------------------------------------------------------------------------

def egocentric_frame(anchor_center: np.ndarray, viewpoint: np.ndarray,
                     look_dir: Optional[np.ndarray] = None,
                     up: np.ndarray = WORLD_UP,
                     kind: str = "egocentric",
                     confidence: float = 1.0,
                     provenance: Optional[Dict] = None) -> ReferenceFrame:
    """Viewer-centric frame read at `anchor_center` from `viewpoint`.

    `right` is the viewer's own right, taken about the line of sight to the
    anchor (not about the camera's optical axis) because "left of the laptop"
    is judged around the laptop, not around wherever the lens happened to point.
    `front` points from the anchor back towards the viewer, so "in front of the
    laptop" is the near side. The result is left-handed; see the module
    docstring.
    """
    anchor_center = np.asarray(anchor_center, float)
    viewpoint = np.asarray(viewpoint, float)
    to_anchor = horizontal(anchor_center - viewpoint, up)
    if not np.any(to_anchor):
        # the viewer is directly above or below the anchor
        if look_dir is not None:
            to_anchor = horizontal(look_dir, up)
        if not np.any(to_anchor):
            return unavailable(kind, anchor_center,
                               "viewpoint is vertically aligned with the anchor,"
                               " so the viewer's left and right are undefined", up)
    right = normalize(np.cross(to_anchor, up))
    front = -to_anchor
    return ReferenceFrame(kind=kind, right=right, front=front, up=up,
                          origin=anchor_center, viewpoint=viewpoint,
                          confidence=confidence,
                          provenance={"line_of_sight": to_anchor.tolist(),
                                      **(provenance or {})})


def egocentric_image_frame(anchor_center: np.ndarray, pose_c2w: np.ndarray,
                           confidence: float = 1.0,
                           provenance: Optional[Dict] = None) -> ReferenceFrame:
    """Image-plane frame: keeps camera roll, so it is not gravity-aligned."""
    anchor_center = np.asarray(anchor_center, float)
    right = camera_right(pose_c2w)
    up = -camera_down(pose_c2w)            # OpenCV +y is down
    front = -camera_forward(pose_c2w)      # towards the camera
    return ReferenceFrame(kind="egocentric_image", right=right, front=front,
                          up=up, origin=anchor_center,
                          viewpoint=pose_c2w[:3, 3].copy(),
                          confidence=confidence,
                          provenance={"note": "camera roll preserved",
                                      **(provenance or {})})


def intrinsic_frame(anchor_center: np.ndarray, anchor_front: Optional[np.ndarray],
                    up: np.ndarray = WORLD_UP, confidence: float = 1.0,
                    provenance: Optional[Dict] = None) -> ReferenceFrame:
    """Object-centric frame: the anchor's own left, right and front."""
    anchor_center = np.asarray(anchor_center, float)
    if anchor_front is None:
        return unavailable("intrinsic", anchor_center,
                           "the anchor has no estimated front", up)
    f = horizontal(anchor_front, up)
    if not np.any(f):
        return unavailable("intrinsic", anchor_center,
                           "the anchor's front is vertical, so its left and right"
                           " are undefined", up)
    right = normalize(np.cross(f, up))
    return ReferenceFrame(kind="intrinsic", right=right, front=f, up=up,
                          origin=anchor_center, confidence=confidence,
                          provenance=dict(provenance or {}))


def addressee_frame(anchor_center: np.ndarray, anchor_front: Optional[np.ndarray],
                    up: np.ndarray = WORLD_UP, confidence: float = 1.0,
                    provenance: Optional[Dict] = None) -> ReferenceFrame:
    """Mirrored object-centric frame: left and right as seen by someone facing
    the anchor. Equal to the egocentric frame of a viewer on the anchor's front
    axis, which `tests/test_frames.py` verifies numerically."""
    base = intrinsic_frame(anchor_center, anchor_front, up, confidence, provenance)
    if not base.available:
        return unavailable("addressee", anchor_center, base.reason, up)
    out = base.mirrored("addressee")
    out.provenance["equivalent_to"] = "egocentric viewer on the anchor's front axis"
    return out


def world_frame(anchor_center: np.ndarray, room, up: Optional[np.ndarray] = None,
                provenance: Optional[Dict] = None) -> ReferenceFrame:
    """Room-canonical frame. Confidence carries the 4-fold ambiguity margin."""
    anchor_center = np.asarray(anchor_center, float)
    if room is None:
        return unavailable("world", anchor_center,
                           "no room structure was estimated",
                           up if up is not None else WORLD_UP)
    u = room.up if up is None else up
    f = room.canonical_forward
    if f is None:
        return unavailable("world", anchor_center,
                           "the room has no canonical forward direction", u)
    f = horizontal(f, u)
    right = normalize(np.cross(f, u))
    conf = float(np.clip(min(room.forward_confidence / 0.30,
                             room.axis_confidence), 0.0, 1.0))
    return ReferenceFrame(
        kind="world", right=right, front=f, up=u, origin=anchor_center,
        confidence=conf,
        provenance={"convention": room.forward_convention,
                    "forward_margin": room.forward_confidence,
                    "manhattan_confidence": room.axis_confidence,
                    "candidates": [c.tolist() for c in room.forward_candidates],
                    **(provenance or {})})

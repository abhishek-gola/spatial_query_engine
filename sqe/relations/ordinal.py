"""Ordinal selection: "the second mug from the left on the middle shelf".

This is where the frame problem gets sharpest, because an ordinal needs three
things and pipelines usually get two of them from one place:

1. an **ordering axis** -- which line the objects are strung out along;
2. a **sign** -- which end of that line counts as "the left";
3. an **index**.

The axis is best taken from the geometry the objects sit on. Four mugs on a
shelf are ordered along the shelf's long horizontal axis, and using that rather
than the viewer's lateral axis makes the ordering stable when the viewer stands
slightly off to one side. But the *sign* cannot come from the shelf, because the
shelf's long axis has no preferred end. The sign has to come from a reference
frame, and different frames give opposite signs -- so "the second from the left"
picks a different mug depending on the frame while the ordering itself is
identical.

The module also refuses to answer in two situations that pipelines answer
anyway:

* **degenerate spread** -- the candidates are not actually separated along the
  axis. Four mugs strung out along a shelf that runs left-to-right have no
  meaningful order along the room's front-back axis, and asking for the second
  from the left in a frame whose lateral axis points along the shelf's depth is
  a question with no answer.
* **fragile ordering** -- two candidates are so close along the axis that their
  order is inside the noise. The answer is reported with the tie flagged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..frames.reference_frame import ReferenceFrame
from ..geom.obb import OBB
from ..geom.transforms import horizontal, normalize
from .base import RelationConfig, RelationSpec, register

register(RelationSpec("ordinal", "ordinal", True, 0,
                      aliases=("nth from", "counting from", "ordinal"),
                      doc="pick the n-th candidate along a frame-signed axis"))

#: Note the absence of "one", "two", "three". They are cardinals far more often
#: than ordinals -- "the two mugs on the shelf" is a count, not a position --
#: and treating them as ordinals made every plural query parse as an ordinal.
ORDINAL_WORDS: Dict[str, int] = {
    "first": 0, "1st": 0,
    "second": 1, "2nd": 1,
    "third": 2, "3rd": 2,
    "fourth": 3, "4th": 3,
    "fifth": 4, "5th": 4,
    "sixth": 5, "6th": 5,
    "seventh": 6, "7th": 6,
    "eighth": 7, "8th": 7,
    "ninth": 8, "9th": 8,
    "tenth": 9, "10th": 9,
    "last": -1, "final": -1, "furthest": -1, "farthest": -1,
    "second to last": -2, "second last": -2, "penultimate": -2,
}

#: Words that name the end an ordinal counts from, mapped to a frame axis name.
FROM_WORDS: Dict[str, str] = {
    "left": "left", "leftmost": "left", "left hand": "left",
    "right": "right", "rightmost": "right", "right hand": "right",
    "front": "front", "nearest": "front", "near": "front", "closest": "front",
    "back": "behind", "behind": "behind", "rear": "behind", "far": "behind",
    "top": "up", "upper": "up", "highest": "up",
    "bottom": "down", "lower": "down", "lowest": "down",
}

#: Superlatives that are ordinals in disguise.
EXTREME_WORDS: Dict[str, Tuple[str, int]] = {
    "leftmost": ("left", 0), "left most": ("left", 0),
    "rightmost": ("right", 0), "right most": ("right", 0),
    "topmost": ("up", 0), "top most": ("up", 0),
    "nearest": ("front", 0), "closest": ("front", 0),
    "farthest": ("behind", 0), "furthest": ("behind", 0),
    "bottommost": ("down", 0),
}


def parse_ordinal_word(word: str, allow_bare_digits: bool = False) -> Optional[int]:
    """'second' -> 1, 'last' -> -1, '3rd' -> 2. None if not an ordinal.

    A bare digit is not accepted by default: "the 2 chairs" is a count. Pass
    `allow_bare_digits` when the caller already knows the slot is positional.
    """
    if word is None:
        return None
    w = " ".join(str(word).strip().lower().replace("-", " ").split())
    if w in ORDINAL_WORDS:
        return ORDINAL_WORDS[w]
    m = re.fullmatch(r"(\d+)(st|nd|rd|th)?", w)
    if m and (m.group(2) or allow_bare_digits):
        return max(0, int(m.group(1)) - 1)
    return None


# --------------------------------------------------------------------------

@dataclass
class OrderingAxis:
    direction: np.ndarray            # world unit vector; index 0 is at the tail
    source: str                      # frame_axis | support_long_axis | landmark
    confidence: float
    frame_kind: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"direction": self.direction.tolist(), "source": self.source,
                "confidence": self.confidence, "frame_kind": self.frame_kind,
                "notes": list(self.notes)}


@dataclass
class OrdinalResult:
    order: List[int]                 # candidate indices, in order
    keys: List[float]
    axis: OrderingAxis
    picked: List[int]                # candidate indices selected (>1 if tied)
    index: Optional[int]
    spread: float
    mean_width: float
    min_gap: float
    median_gap: float
    degenerate: bool
    fragile: bool
    out_of_range: bool
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.picked) and not self.degenerate and not self.out_of_range

    def to_dict(self) -> dict:
        return {"order": list(self.order), "keys": [float(k) for k in self.keys],
                "axis": self.axis.to_dict(), "picked": list(self.picked),
                "index": self.index, "spread": self.spread,
                "mean_width": self.mean_width, "min_gap": self.min_gap,
                "median_gap": self.median_gap, "degenerate": self.degenerate,
                "fragile": self.fragile, "out_of_range": self.out_of_range,
                "notes": list(self.notes)}


def frame_ordering_axis(frame: ReferenceFrame, from_word: str) -> OrderingAxis:
    """Ordering direction implied by counting "from the <from_word>".

    Counting from the left means the leftmost item is index 0, so the ordering
    direction is the frame's *right*: positions increase as you move away from
    the end you started at.
    """
    name = FROM_WORDS.get((from_word or "left").strip().lower(), "left")
    if not frame.available:
        return OrderingAxis(np.array([1.0, 0.0, 0.0]), "frame_axis", 0.0,
                            frame.kind, [f"frame unavailable: {frame.reason}"])
    d = -frame.axis(name)
    return OrderingAxis(normalize(d), "frame_axis", frame.confidence, frame.kind,
                        [f"counting from the {name} in the {frame.kind} frame"])


def support_ordering_axis(support: OBB, frame: ReferenceFrame, from_word: str,
                          min_alignment: float = 0.35) -> OrderingAxis:
    """Ordering along the support surface's long axis, signed by the frame.

    The magnitude of the alignment between the support's long axis and the
    frame's axis is carried as the confidence: looking at a shelf end-on makes
    "the second from the left" close to meaningless, and this is where that gets
    detected rather than in the answer.
    """
    base = frame_ordering_axis(frame, from_word)
    if base.confidence <= 0.0 and not frame.available:
        return base
    long_axis = horizontal(support.long_axis, frame.up)
    if not np.any(long_axis):
        base.notes.append("support has no horizontal long axis; using frame axis")
        return base
    dot = float(long_axis @ base.direction)
    signed = long_axis if dot >= 0 else -long_axis
    align = abs(dot)
    notes = list(base.notes) + [
        f"ordering along the support's long axis, signed by the "
        f"{frame.kind} frame (alignment {align:.2f})"]
    if align < min_alignment:
        notes.append("the support's long axis is nearly perpendicular to the "
                     "frame's counting axis, so the ordering is ill-posed")
    return OrderingAxis(normalize(signed), "support_long_axis",
                        float(min(base.confidence, align / max(min_alignment, 1e-6)))
                        if align < min_alignment else base.confidence,
                        frame.kind, notes)


def landmark_ordering_axis(landmark_center: np.ndarray,
                           candidates: Sequence[OBB],
                           up: np.ndarray) -> OrderingAxis:
    """"The third chair from the door": order by distance from a landmark.

    Not really an axis -- the key is a radial distance -- but it shares the
    interface, and unlike the projective axes it is frame-independent, which
    makes landmark ordinals a useful control in the benchmark.
    """
    return OrderingAxis(np.zeros(3), "landmark", 1.0, "",
                        ["ordering by distance from the landmark "
                         "(frame-independent)"])


def order_candidates(candidates: Sequence[OBB], axis: OrderingAxis,
                     landmark_center: Optional[np.ndarray] = None
                     ) -> Tuple[List[int], List[float], List[float]]:
    """Sort candidates along the axis. Returns (order, keys, widths)."""
    n = len(candidates)
    if axis.source == "landmark":
        if landmark_center is None:
            raise ValueError("landmark ordering needs a landmark centre")
        keys = [float(np.linalg.norm(c.center - landmark_center)) for c in candidates]
        widths = [float(c.horizontal_radius * 2.0) for c in candidates]
    else:
        d = axis.direction
        keys = [float(c.center @ d) for c in candidates]
        widths = [float(2.0 * c.radius_along(d)) for c in candidates]
    order = sorted(range(n), key=lambda i: keys[i])
    return order, keys, widths


def apply_ordinal(candidates: Sequence[OBB], axis: OrderingAxis,
                  index: Optional[int], cfg: RelationConfig,
                  landmark_center: Optional[np.ndarray] = None) -> OrdinalResult:
    """Order the candidates and take the `index`-th, with degeneracy checks."""
    n = len(candidates)
    notes: List[str] = list(axis.notes)
    if n == 0:
        return OrdinalResult([], [], axis, [], index, 0.0, 0.0, 0.0, 0.0,
                             True, False, True, notes + ["no candidates"])

    order, keys, widths = order_candidates(candidates, axis, landmark_center)
    sorted_keys = [keys[i] for i in order]
    spread = float(sorted_keys[-1] - sorted_keys[0]) if n > 1 else 0.0
    mean_width = float(np.mean(widths)) if widths else 0.0
    gaps = [sorted_keys[i + 1] - sorted_keys[i] for i in range(n - 1)]
    min_gap = float(min(gaps)) if gaps else 0.0
    median_gap = float(np.median(gaps)) if gaps else 0.0

    degenerate = False
    if n > 1 and spread < cfg.ordinal_min_spread_factor * max(mean_width, 1e-6):
        degenerate = True
        notes.append(
            f"candidates span only {spread:.3f} m along this axis but are "
            f"{mean_width:.3f} m wide on average, so they are not ordered "
            f"along it at all")
    if axis.confidence <= 0.0:
        degenerate = True
        notes.append("the ordering axis has zero confidence")

    fragile = False
    if gaps:
        if min_gap < cfg.ordinal_min_gap:
            fragile = True
            notes.append(f"the two closest candidates are {min_gap:.3f} m apart "
                         f"along the axis; their order is inside the noise")
        elif median_gap > 0 and min_gap < cfg.ordinal_tie_ratio * median_gap:
            fragile = True
            notes.append(f"one neighbour gap ({min_gap:.3f} m) is much smaller "
                         f"than the typical spacing ({median_gap:.3f} m)")

    picked: List[int] = []
    out_of_range = False
    if index is None:
        picked = list(order)
    else:
        i = index if index >= 0 else n + index
        if 0 <= i < n:
            picked = [order[i]]
        else:
            out_of_range = True
            notes.append(f"asked for item {index} but only {n} candidates exist")

    return OrdinalResult(order, keys, axis, picked, index, spread, mean_width,
                         min_gap, median_gap, degenerate, fragile, out_of_range,
                         notes)


def middle_candidate(candidates: Sequence[OBB], axis: OrderingAxis,
                     cfg: RelationConfig) -> OrdinalResult:
    """"the middle mug". Returns both neighbours when the count is even."""
    n = len(candidates)
    res = apply_ordinal(candidates, axis, None, cfg)
    if n == 0:
        return res
    if n % 2 == 1:
        res.picked = [res.order[n // 2]]
        res.index = n // 2
    else:
        res.picked = [res.order[n // 2 - 1], res.order[n // 2]]
        res.index = None
        res.notes.append(f"{n} candidates: 'middle' has two equally good "
                         f"referents")
    return res

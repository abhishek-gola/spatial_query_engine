"""Choosing a reference frame, and being explicit that a choice was made.

The default policy is a small table, written out below and in docs/METHOD.md.
Its most important property is the asymmetry between axes:

* **front / behind** prefer the *intrinsic* frame whenever the anchor has a
  front. "In front of the sofa" means the side the sofa faces; almost nobody
  means "between me and the sofa" when the anchor is a sofa. But "in front of
  the box" can only mean the near side, because a box has no front.
* **left / right** prefer the *egocentric* frame even when the anchor has a
  front. "The mug to the left of the laptop" is overwhelmingly read as the
  speaker's left.

That asymmetry is a documented fact about how people use these words, and it is
the single most useful thing a resolver can know. Anchoring both axes to the
same frame -- which is what you get by building one basis and reusing it -- is
wrong for one of them whichever frame you pick.

Every decision carries `explicit`: whether the sentence named a frame, or the
policy supplied one. The benchmark reports accuracy separately for the two, and
the second number is the interesting one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..categories import is_room_fixed
from ..geom.transforms import horizontal, normalize
from .cues import FrameCue, extract_cues
from .reference_frame import (ALL_KINDS, ReferenceFrame, addressee_frame,
                              egocentric_frame, egocentric_image_frame,
                              intrinsic_frame, unavailable, world_frame)

#: Eye height used when a viewpoint is placed at a landmark rather than a camera.
EYE_HEIGHT = 1.55

#: Minimum front confidence before the intrinsic frame is offered at full prior.
INTRINSIC_MIN_FRONT_CONFIDENCE = 0.25

#: Frames below this confidence are not allowed to be the chosen frame unless
#: the sentence explicitly asked for them.
MIN_USABLE_CONFIDENCE = 0.15


# --------------------------------------------------------------------------
# viewpoints
# --------------------------------------------------------------------------

@dataclass
class ViewpointSpec:
    """Where the egocentric observer stands.

    `best_view` is the default and means "the frame of the capture that shows
    this anchor most squarely" -- the closest thing to "as filmed". The viewer
    UI uses `position` so that orbiting the camera really does change which mug
    is "the second from the left", which is the demo that makes the whole point
    in three seconds.
    """
    mode: str = "best_view"     # best_view | nearest | mean | index | position | landmark
    index: Optional[int] = None
    position: Optional[np.ndarray] = None
    look_at: Optional[np.ndarray] = None
    landmark: Optional[str] = None

    def to_dict(self) -> dict:
        return {"mode": self.mode, "index": self.index,
                "position": None if self.position is None
                else np.asarray(self.position).tolist(),
                "look_at": None if self.look_at is None
                else np.asarray(self.look_at).tolist(),
                "landmark": self.landmark}


@dataclass
class ResolvedViewpoint:
    eye: Optional[np.ndarray]
    look_dir: Optional[np.ndarray]
    pose: Optional[np.ndarray]
    source: str
    ok: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return {"eye": None if self.eye is None else self.eye.tolist(),
                "look_dir": None if self.look_dir is None else self.look_dir.tolist(),
                "source": self.source, "ok": self.ok, "reason": self.reason}


def resolve_viewpoint(spec: ViewpointSpec, scene, anchor_center: np.ndarray
                      ) -> ResolvedViewpoint:
    """Turn a `ViewpointSpec` into an eye position and look direction."""
    anchor_center = np.asarray(anchor_center, float)
    traj = scene.trajectory
    up = scene.up

    if spec.mode == "position":
        if spec.position is None:
            return ResolvedViewpoint(None, None, None, "position", False,
                                     "viewpoint mode 'position' with no position")
        eye = np.asarray(spec.position, float)
        target = (np.asarray(spec.look_at, float) if spec.look_at is not None
                  else anchor_center)
        return ResolvedViewpoint(eye, normalize(target - eye), None,
                                 "explicit position", True)

    if spec.mode == "landmark":
        obj = _find_landmark(scene, spec.landmark)
        if obj is None:
            return ResolvedViewpoint(
                None, None, None, f"landmark:{spec.landmark}", False,
                f"no object in the scene matches the landmark "
                f"{spec.landmark!r}, so the viewpoint it names cannot be placed")
        eye = obj.center.copy()
        floor = scene.room.floor_z if scene.room is not None else 0.0
        eye[2] = floor + EYE_HEIGHT
        # step slightly off the landmark towards the room, so a viewer standing
        # "at the door" is not inside the door's own geometry
        into = anchor_center - eye
        into[2] = 0.0
        if np.linalg.norm(into) > 1e-6:
            eye = eye + 0.25 * normalize(into)
        return ResolvedViewpoint(eye, normalize(anchor_center - eye), None,
                                 f"landmark:{obj.label}", True)

    if traj is None or len(traj) == 0:
        return ResolvedViewpoint(
            None, None, None, spec.mode, False,
            "the scene has no camera trajectory, so no egocentric viewpoint "
            "is available; give an explicit position instead")

    if spec.mode == "index":
        i = int(spec.index if spec.index is not None else 0)
        if not (0 <= i < len(traj)):
            return ResolvedViewpoint(None, None, None, "index", False,
                                     f"frame index {i} is outside the "
                                     f"{len(traj)}-pose trajectory")
    elif spec.mode == "nearest":
        i = traj.nearest_index(anchor_center)
    elif spec.mode == "mean":
        eye = traj.centers.mean(axis=0)
        h = traj.mean_heading(up)
        if not np.any(h):
            return ResolvedViewpoint(
                None, None, None, "mean", False,
                "the capture panned all the way round, so its mean heading is "
                "degenerate and there is no canonical viewpoint")
        return ResolvedViewpoint(eye, h, None, "mean trajectory pose", True)
    else:   # best_view
        i = traj.best_view(anchor_center_obb(anchor_center), up)
        if i < 0:
            i = traj.nearest_index(anchor_center)

    pose = traj.poses[i]
    eye = pose[:3, 3].copy()
    return ResolvedViewpoint(eye, normalize(pose[:3, 2]), pose,
                             f"{spec.mode} frame {i}"
                             + (f" ({traj.frame_ids[i]})"
                                if i < len(traj.frame_ids) else ""), True)


def anchor_center_obb(center: np.ndarray):
    """Tiny helper so `best_view` can be called with a bare centre."""
    from ..geom.obb import OBB
    return OBB(np.asarray(center, float), np.array([0.05, 0.05, 0.05]))


def _find_landmark(scene, label: Optional[str]):
    if not label:
        return None
    from ..categories import normalize_label
    want = normalize_label(label)
    hits = [o for o in scene.objects if o.canonical_label == want]
    if not hits:
        hits = [o for o in scene.objects if want and want in o.canonical_label]
    if not hits:
        return None
    # the biggest match, so "the door" is the doorway and not a door handle
    return max(hits, key=lambda o: o.obb.volume)


# --------------------------------------------------------------------------
# frame construction
# --------------------------------------------------------------------------

def build_frames(scene, anchor, kinds: Sequence[str] = ALL_KINDS,
                 viewpoint: Optional[ViewpointSpec] = None
                 ) -> Tuple[Dict[str, ReferenceFrame], ResolvedViewpoint]:
    """Construct every requested frame around `anchor`.

    Frames that cannot be built come back as unavailable objects with a reason,
    never as a silent fallback to another frame.
    """
    viewpoint = viewpoint or ViewpointSpec()
    center = anchor.obb.center if hasattr(anchor, "obb") else np.asarray(anchor, float)
    rv = resolve_viewpoint(viewpoint, scene, center)
    out: Dict[str, ReferenceFrame] = {}

    for kind in kinds:
        if kind in ("egocentric", "egocentric_bearing"):
            if not rv.ok:
                out[kind] = unavailable(kind, center, rv.reason, scene.up)
            else:
                out[kind] = egocentric_frame(
                    center, rv.eye, rv.look_dir, scene.up, kind,
                    provenance={"viewpoint": rv.source})
        elif kind == "egocentric_image":
            if not rv.ok or rv.pose is None:
                out[kind] = unavailable(
                    kind, center,
                    rv.reason or "image-plane axes need an actual camera pose, "
                                 "and this viewpoint is not a captured frame",
                    scene.up)
            else:
                out[kind] = egocentric_image_frame(
                    center, rv.pose, provenance={"viewpoint": rv.source})
        elif kind == "intrinsic":
            conf = float(getattr(anchor, "front_confidence", 0.0))
            front = getattr(anchor, "front", None)
            if front is None and getattr(anchor, "has_intrinsic_front", False):
                out[kind] = unavailable(
                    kind, center,
                    f"{getattr(anchor, 'label', 'the anchor')} has a front in "
                    f"principle but it could not be estimated "
                    f"({getattr(anchor, 'front_method', '?')})", scene.up)
            elif front is None:
                out[kind] = unavailable(
                    kind, center,
                    f"a {getattr(anchor, 'label', 'anchor')} has no intrinsic "
                    f"front, so its own left and right do not exist", scene.up)
            else:
                out[kind] = intrinsic_frame(center, front, scene.up, conf,
                                            {"front_method":
                                             getattr(anchor, "front_method", "")})
        elif kind == "addressee":
            conf = float(getattr(anchor, "front_confidence", 0.0))
            front = getattr(anchor, "front", None)
            if front is None:
                out[kind] = unavailable(kind, center,
                                        "the anchor's front is unknown, so the "
                                        "mirrored reading is unavailable",
                                        scene.up)
            else:
                out[kind] = addressee_frame(center, front, scene.up, conf)
        elif kind == "world":
            out[kind] = world_frame(center, scene.room, scene.up)
        else:
            raise ValueError(f"unknown frame kind {kind!r}")
    return out, rv


# --------------------------------------------------------------------------
# the policy
# --------------------------------------------------------------------------

@dataclass
class FramePrior:
    kind: str
    prior: float
    rationale: str


@dataclass
class FrameDecision:
    chosen: Optional[str]
    frame: Optional[ReferenceFrame]
    priors: List[FramePrior]
    frames: Dict[str, ReferenceFrame]
    viewpoint: ResolvedViewpoint
    explicit: bool
    cues: List[FrameCue] = field(default_factory=list)
    conflicting_cues: bool = False
    notes: List[str] = field(default_factory=list)
    forced: bool = False

    def usable(self) -> bool:
        return self.frame is not None and self.frame.available

    def to_dict(self) -> dict:
        return {
            "chosen": self.chosen,
            "explicit": self.explicit,
            "forced": self.forced,
            "conflicting_cues": self.conflicting_cues,
            "priors": [{"kind": p.kind, "prior": p.prior,
                        "rationale": p.rationale} for p in self.priors],
            "frames": {k: v.to_dict() for k, v in self.frames.items()},
            "viewpoint": self.viewpoint.to_dict(),
            "cues": [c.to_dict() for c in self.cues],
            "notes": list(self.notes),
        }


#: The default policy table. `axis` is which projective axis the relation uses.
DEFAULT_PRIORS: Dict[str, Dict[str, float]] = {
    # relation group -> {frame kind: prior}
    "front_back": {"intrinsic": 1.00, "egocentric": 0.55,
                   "addressee": 0.25, "world": 0.15},
    "left_right": {"egocentric": 1.00, "intrinsic": 0.55,
                   "addressee": 0.35, "world": 0.15},
}

RELATION_GROUP = {"front": "front_back", "behind": "front_back",
                  "left": "left_right", "right": "left_right",
                  "ordinal": "left_right"}


def relation_group(relation: str) -> Optional[str]:
    return RELATION_GROUP.get(relation)


def decide_frame(relation: str, anchor, scene, text: str = "",
                 viewpoint: Optional[ViewpointSpec] = None,
                 force_kind: Optional[str] = None,
                 kinds: Sequence[str] = ALL_KINDS) -> FrameDecision:
    """Pick a reference frame for one relation on one anchor.

    `force_kind` bypasses the policy entirely and is how the benchmark runs the
    fixed-frame baselines that most pipelines implement by accident.
    """
    frames, rv = build_frames(scene, anchor, kinds, viewpoint)
    cues = extract_cues(text)
    cue_kinds = {c.kind: c.weight for c in cues}
    notes: List[str] = []

    group = relation_group(relation)
    if group is None:
        # frame-independent relation: nothing to choose, but keep the frames
        # around so explanations and the viewer can still show axes
        return FrameDecision(None, None, [], frames, rv, bool(cues), cues,
                             len(cue_kinds) > 1,
                             [f"{relation!r} does not depend on a reference frame"])

    if force_kind:
        f = frames.get(force_kind)
        if f is None:
            raise ValueError(f"unknown frame kind {force_kind!r}")
        return FrameDecision(force_kind, f, [FramePrior(force_kind, 1.0,
                                                        "forced by caller")],
                             frames, rv, bool(cues), cues, len(cue_kinds) > 1,
                             [f"frame forced to {force_kind!r}"], forced=True)

    base = dict(DEFAULT_PRIORS[group])

    # The anchor's category shifts the priors.
    label = getattr(anchor, "label", "")
    if is_room_fixed(label) and getattr(anchor, "obb", None) is not None:
        if anchor.obb.footprint_area > 2.0 or "wall" in str(label).lower():
            base["world"] = max(base["world"], 0.90)
            notes.append(f"{label!r} is part of the room shell, so the "
                         f"room-canonical frame is a strong candidate")

    front_conf = float(getattr(anchor, "front_confidence", 0.0))
    if front_conf < INTRINSIC_MIN_FRONT_CONFIDENCE:
        for k in ("intrinsic", "addressee"):
            if k in base:
                base[k] *= 0.35
        if getattr(anchor, "has_intrinsic_front", False):
            notes.append(
                f"the {label}'s front is only {front_conf:.2f} confident, so "
                f"the intrinsic reading is down-weighted rather than trusted")

    # Explicit cues dominate everything.
    for kind, w in cue_kinds.items():
        base[kind] = base.get(kind, 0.0) + 10.0 * w
    if len(cue_kinds) > 1:
        notes.append("the sentence contains more than one frame marker "
                     f"({', '.join(sorted(cue_kinds))}); this is a genuine "
                     f"ambiguity, not a tie to be broken")

    priors = [FramePrior(k, float(v), _rationale(k, group, bool(cue_kinds)))
              for k, v in sorted(base.items(), key=lambda kv: -kv[1])]

    chosen, frame = None, None
    for p in priors:
        f = frames.get(p.kind)
        if f is None or not f.available:
            if f is not None and p.prior >= 0.9:
                notes.append(f"{p.kind} was preferred but is unavailable: "
                             f"{f.reason}")
            continue
        if not cue_kinds and f.confidence < MIN_USABLE_CONFIDENCE:
            notes.append(f"{p.kind} is available but only {f.confidence:.2f} "
                         f"confident, which is too low to choose by default")
            continue
        chosen, frame = p.kind, f
        break

    if frame is None:
        notes.append("no reference frame could be established for this query")
    return FrameDecision(chosen, frame, priors, frames, rv, bool(cues), cues,
                         len(cue_kinds) > 1, notes)


def _rationale(kind: str, group: str, explicit: bool) -> str:
    if explicit:
        return "the sentence names this frame"
    if group == "front_back":
        return {
            "intrinsic": "'in front of' means the side the anchor faces, when "
                         "the anchor has a face",
            "egocentric": "with no anchor front, 'in front of' means the side "
                          "nearer the viewer",
            "addressee": "the mirrored reading, attested but uncommon for "
                         "front/back",
            "world": "room-canonical front/back, used for room-shell anchors",
        }.get(kind, "")
    return {
        "egocentric": "'left of X' is read as the speaker's left far more often "
                      "than as X's own left",
        "intrinsic": "the anchor's own left, attested when the anchor has a "
                     "strong front",
        "addressee": "the left seen by someone facing the anchor",
        "world": "room-canonical left, used for room-shell anchors",
    }.get(kind, "")

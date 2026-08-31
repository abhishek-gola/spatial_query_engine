"""Estimating an object's intrinsic front.

"In front of the sofa" is meaningless without knowing which way the sofa faces,
and no dataset ships that for most classes. This module estimates it from
geometry and reports a confidence, so the intrinsic reference frame can be
declared *unavailable* rather than silently wrong.

Two stages, deliberately separate
---------------------------------

1. **Which axis** does the front lie on? Answered mostly by shape priors: a
   monitor faces along its thin horizontal axis, a desk normal to its long side.
2. **Which of the two directions** along that axis? Answered by placement and
   structure: the roomier side, the side away from the backrest, the hollow side.

Keeping them apart matters. The first version scored all four directions on one
scale, and on real ScanNet++ monitors it picked the wrong *axis* -- because a
monitor with a stand is 0.56 x 0.20 m at the panel but 0.32 m deep once the
base is included, so the "thin slab" test never fired and clearance decided the
axis instead. Splitting the decision lets a weak shape prior still settle the
axis while placement settles only the sign, which is what each kind of evidence
is actually good for.

Cues, gated per category by `sqe.categories`:

* `backrest`       -- an upright mass offset to one side. Covers chairs, sofas,
                      beds with headboards, and open laptops, which are
                      geometrically the same thing at a different scale.
* `away_from_wall` -- furniture is placed against walls; the front faces the room.
* `thin_face`      -- flat panels face along their thin axis.
* `long_side`      -- fronts are usually normal to the longer horizontal side.
* `open_face`      -- shelves and cabinets are hollow at the front.

One cue is deliberately **not** used: which side the camera looked at. It is by
far the strongest signal for televisions and wall units, and using it would make
the "intrinsic" frame partly egocentric -- quietly contaminating the
egocentric-versus-intrinsic comparison this repo exists to measure. It is
implemented, off by default, and only ever reported as a diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..categories import front_cues, has_intrinsic_front, prior
from ..geom.obb import OBB
from ..geom.transforms import normalize

#: Clearance difference (metres) that counts as fully decisive.
CLEARANCE_SCALE = 1.2
#: An object is "placed against something" when its tightest side has less than
#: this much room. Beyond 2.5x this, clearance carries no information at all.
AGAINST_TOL = 0.45

#: How much each cue is trusted. A backrest is near-conclusive about which way a
#: chair faces; "normal to the long side" is a weak statistical regularity.
CUE_PRIOR = {
    "backrest": 1.00,
    "thin_face": 0.90,
    "open_face": 0.80,
    "away_from_wall": 0.70,
    "long_side": 0.40,
    "visibility": 0.50,
}
CUE_FUNCS = tuple(CUE_PRIOR)


def _ramp(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


@dataclass
class FrontEstimate:
    front: Optional[np.ndarray]
    confidence: float
    method: str
    scores: List[Tuple[List[float], float]] = field(default_factory=list)
    axis_confidence: float = 0.0
    sign_confidence: float = 0.0
    detail: Dict = field(default_factory=dict)


@dataclass
class Cue:
    """One cue's opinion. Either opinion may be absent."""
    name: str
    weight: float
    axis: Optional[np.ndarray] = None     # (2,) preference for local axis 0 / 1
    sign: Optional[np.ndarray] = None     # (4,) preference per candidate
    detail: Dict = field(default_factory=dict)


def _candidates(obb: OBB) -> List[np.ndarray]:
    """The four horizontal face normals, ordered +x, -x, +y, -y in box axes."""
    return [obb.R[:, 0], -obb.R[:, 0], obb.R[:, 1], -obb.R[:, 1]]


def _short_axis_pref(obb: OBB) -> np.ndarray:
    """Preference vector selecting the shorter horizontal axis."""
    return (np.array([1.0, 0.0]) if obb.half[0] <= obb.half[1]
            else np.array([0.0, 1.0]))


def _clearance(obb: OBB, d: np.ndarray, room_bounds: Optional[OBB],
               blockers: Optional[List[OBB]] = None) -> float:
    """Empty room in front of the box's face along `d`."""
    if room_bounds is None:
        return 0.0
    face = obb.interval(d).hi
    free = float(room_bounds.interval(d).hi - face)
    if blockers:
        lat = [obb.R[:, 0], obb.R[:, 1], obb.R[:, 2]]
        for b in blockers:
            if b.footprint_area < 0.15:
                continue
            blocks = True
            for ax in lat:
                if abs(float(ax @ d)) > 0.7:
                    continue
                if obb.interval(ax).overlap(b.interval(ax)) < 1e-3:
                    blocks = False
                    break
            if blocks:
                gap = float(b.interval(d).lo - face)
                if 0.0 < gap < free:
                    free = gap
    return max(0.0, free)


# --------------------------------------------------------------------------
# cues
# --------------------------------------------------------------------------

def _cue_away_from_wall(obb, cands, room_bounds, blockers) -> Cue:
    """Placement against a wall. Votes on both axis and sign.

    The weight is gated by whether the object is actually against anything: for
    a chair in the middle of a room, clearance to the walls says nothing about
    which way it faces, and the gate decays to zero rather than cutting off.
    """
    if room_bounds is None:
        return Cue("away_from_wall", 0.0)
    free = np.array([_clearance(obb, d, room_bounds, blockers) for d in cands])
    asym = np.array([free[0] - free[1], free[2] - free[3]])
    axis_asym = np.abs(asym)
    gate = 1.0 - _ramp(float(free.min()), AGAINST_TOL, AGAINST_TOL * 2.5)
    strength = _ramp(float(axis_asym.max()), 0.25, CLEARANCE_SCALE)
    weight = gate * strength
    axis_pref = axis_asym / max(float(axis_asym.sum()), 1e-9)
    sign_pref = np.array([max(0.0, asym[0]), max(0.0, -asym[0]),
                          max(0.0, asym[1]), max(0.0, -asym[1])])
    m = float(sign_pref.max())
    sign_pref = sign_pref / m if m > 1e-9 else sign_pref
    return Cue("away_from_wall", weight, axis_pref, sign_pref,
               {"clearance": free.tolist(), "gate": gate,
                "min_free": float(free.min())})


def _cue_long_side(obb, cands) -> Cue:
    """Desks, beds and bookshelves face normal to their long side. Axis only."""
    e = obb.extent
    long_h, short_h = max(e[0], e[1]), min(e[0], e[1])
    ratio = long_h / max(short_h, 1e-6)
    return Cue("long_side", _ramp(ratio, 1.15, 2.20), _short_axis_pref(obb),
               None, {"aspect_ratio": float(ratio)})


def _cue_thin_face(obb, cands) -> Cue:
    """Panels face along their thin horizontal axis. Axis only.

    The ramp starts at 1.35 rather than 2.5. Because this cue now only votes on
    the axis and never on the direction, a mildly flattened box is still good
    evidence -- and real monitors, doors and pictures are mildly flattened once
    stands, frames and handles are included in the instance.
    """
    e = obb.extent
    thin = min(e[0], e[1])
    big = max(max(e[0], e[1]), e[2])
    ratio = big / max(thin, 1e-6)
    return Cue("thin_face", _ramp(ratio, 1.35, 3.00), _short_axis_pref(obb),
               None, {"slab_ratio": float(ratio)})


def _cue_backrest(obb, cands, points) -> Cue:
    """An upright mass offset to one side: the front faces away from it.

    Votes on both axis and sign, and is the most reliable cue when it fires.
    """
    if points is None or len(points) < 60:
        return Cue("backrest", 0.0)
    local = obb.to_local(points)
    z = local[:, 2]
    span = max(1e-6, float(z.max() - z.min()))
    hi = z > (z.min() + 0.62 * span)
    if int(hi.sum()) < 25:
        return Cue("backrest", 0.0)
    off = local[hi, :2].mean(axis=0) - local[:, :2].mean(axis=0)
    rel = off / np.maximum(obb.half[:2], 1e-6)
    mag = float(np.linalg.norm(rel))
    if mag < 0.12:
        return Cue("backrest", 0.0, detail={"backrest_offset": rel.tolist()})
    axis_pref = np.abs(rel) / max(float(np.abs(rel).sum()), 1e-9)
    # the front points away from the upper mass
    sign_pref = np.array([max(0.0, -rel[0]), max(0.0, rel[0]),
                          max(0.0, -rel[1]), max(0.0, rel[1])])
    m = float(sign_pref.max())
    sign_pref = sign_pref / m if m > 1e-9 else sign_pref
    return Cue("backrest", _ramp(mag, 0.12, 0.45), axis_pref, sign_pref,
               {"backrest_offset": rel.tolist(), "magnitude": mag})


def _cue_open_face(obb, cands, points) -> Cue:
    """Shelves and cabinets are hollow at the front: the front slab of the box
    holds noticeably fewer points than the back. Votes on axis and sign."""
    if points is None or len(points) < 200:
        return Cue("open_face", 0.0)
    local = obb.to_local(points)
    counts = np.zeros(4)
    for i, ax in enumerate((0, 0, 1, 1)):
        sign = 1 if i % 2 == 0 else -1
        sel = local[:, ax] * sign > obb.half[ax] * 0.72
        counts[i] = float(sel.sum())
    total = float(counts.sum())
    if total < 100:
        return Cue("open_face", 0.0)
    frac = counts / total
    deficit = np.array([frac[1] - frac[0], frac[3] - frac[2]])
    axis_pref = np.abs(deficit) / max(float(np.abs(deficit).sum()), 1e-9)
    sign_pref = np.array([max(0.0, deficit[0]), max(0.0, -deficit[0]),
                          max(0.0, deficit[1]), max(0.0, -deficit[1])])
    m = float(sign_pref.max())
    sign_pref = sign_pref / m if m > 1e-9 else sign_pref
    weight = _ramp(float(np.abs(deficit).max()), 0.06, 0.30)
    return Cue("open_face", weight, axis_pref, sign_pref,
               {"face_fractions": frac.tolist()})


def _cue_visibility(obb, cands, view_dirs) -> Cue:
    """Diagnostic only, and off by default -- see the module docstring."""
    if view_dirs is None or len(view_dirs) == 0:
        return Cue("visibility", 0.0)
    v = np.asarray(view_dirs, float)
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    sign_pref = np.array([float(np.mean(np.maximum(0.0, v @ d))) for d in cands])
    m = float(sign_pref.max())
    if m < 1e-6:
        return Cue("visibility", 0.0)
    sign_pref = sign_pref / m
    axis_pref = np.array([sign_pref[0] + sign_pref[1],
                          sign_pref[2] + sign_pref[3]])
    axis_pref = axis_pref / max(float(axis_pref.sum()), 1e-9)
    return Cue("visibility", 0.6, axis_pref, sign_pref,
               {"view_scores": sign_pref.tolist()})


# --------------------------------------------------------------------------
# estimator
# --------------------------------------------------------------------------

def estimate_front(obb: OBB,
                   points: Optional[np.ndarray] = None,
                   label: str = "",
                   room_bounds: Optional[OBB] = None,
                   blockers: Optional[List[OBB]] = None,
                   view_dirs: Optional[np.ndarray] = None,
                   use_visibility: bool = False,
                   min_confidence: float = 0.18) -> FrontEstimate:
    """Estimate the intrinsic front of one object.

    Returns `front = None` when the category has no front at all, and also when
    the cues cannot settle the axis or the sign. Both matter and are
    distinguished downstream: the first makes the intrinsic frame inapplicable,
    the second makes it unreliable.
    """
    if not has_intrinsic_front(label):
        return FrontEstimate(None, 0.0, "no_front_category", detail={"label": label})

    cands = _candidates(obb)
    wanted = front_cues(label) or ("away_from_wall", "long_side")
    cues: List[Cue] = []
    for name in wanted:
        if name == "away_from_wall":
            c = _cue_away_from_wall(obb, cands, room_bounds, blockers)
        elif name == "long_side":
            c = _cue_long_side(obb, cands)
        elif name == "thin_face":
            c = _cue_thin_face(obb, cands)
        elif name == "backrest":
            c = _cue_backrest(obb, cands, points)
        elif name == "open_face":
            c = _cue_open_face(obb, cands, points)
        else:
            continue
        c.weight *= CUE_PRIOR.get(name, 0.5)
        cues.append(c)

    if use_visibility:
        c = _cue_visibility(obb, cands, view_dirs)
        c.weight *= CUE_PRIOR["visibility"]
        cues.append(c)
    vis_diag = (_cue_visibility(obb, cands, view_dirs).detail
                if (view_dirs is not None and not use_visibility) else {})

    detail: Dict = {"cues": {c.name: {"weight": float(c.weight), **c.detail}
                             for c in cues}}
    if vis_diag:
        detail["visibility_diagnostic"] = vis_diag

    active = [c for c in cues if c.weight > 1e-6]
    if not active:
        return FrontEstimate(None, 0.0, "no_cue_fired", [], 0.0, 0.0, detail)

    # -- stage 1: which horizontal axis ---------------------------------
    axis_acc = np.zeros(2)
    axis_w = 0.0
    axis_voters: List[str] = []
    for c in active:
        if c.axis is None:
            continue
        axis_acc += c.weight * c.axis
        axis_w += c.weight
        axis_voters.append(c.name)
    if axis_w <= 1e-9:
        return FrontEstimate(None, 0.0, "no_axis_evidence", [], 0.0, 0.0, detail)
    axis_scores = axis_acc / axis_w
    k = int(np.argmax(axis_scores))
    tot = float(axis_scores.sum())
    axis_conf = float(abs(axis_scores[0] - axis_scores[1]) / tot) if tot > 1e-9 else 0.0

    # -- stage 2: which sign along that axis ----------------------------
    pair = (0, 1) if k == 0 else (2, 3)
    sign_acc = np.zeros(2)
    sign_w = 0.0
    sign_voters: List[str] = []
    for c in active:
        if c.sign is None:
            continue
        vote = np.array([c.sign[pair[0]], c.sign[pair[1]]])
        if float(vote.sum()) <= 1e-9:
            continue
        sign_acc += c.weight * vote
        sign_w += c.weight
        sign_voters.append(c.name)
    detail["axis_scores"] = axis_scores.tolist()
    detail["chosen_axis"] = k
    detail["axis_voters"] = axis_voters

    if sign_w <= 1e-9:
        # The axis is known but nothing distinguishes its two ends. This is the
        # honest outcome for a keyboard or a bare panel, and it is a different
        # failure from not knowing the axis.
        return FrontEstimate(None, 0.0, "axis_only:" + ",".join(axis_voters),
                             [(list(map(float, cands[i])), 0.0)
                              for i in (pair[0], pair[1])],
                             axis_conf, 0.0, detail)

    sign_scores = sign_acc / sign_w
    s = int(np.argmax(sign_scores))
    stot = float(sign_scores.sum())
    sign_conf = float(abs(sign_scores[0] - sign_scores[1]) / stot) if stot > 1e-9 else 0.0
    detail["sign_scores"] = sign_scores.tolist()
    detail["sign_voters"] = sign_voters

    confidence = float(np.clip(min(axis_conf, 1.0) * min(sign_conf, 1.0), 0.0, 1.0))
    chosen = cands[pair[s]]
    alternatives = [(list(map(float, cands[pair[s]])), float(sign_scores[s])),
                    (list(map(float, cands[pair[1 - s]])), float(sign_scores[1 - s])),
                    (list(map(float, cands[2 if k == 0 else 0])), 0.0),
                    (list(map(float, cands[3 if k == 0 else 1])), 0.0)]

    if confidence < min_confidence:
        return FrontEstimate(None, confidence,
                             "inconclusive:axis=%.2f,sign=%.2f" % (axis_conf, sign_conf),
                             alternatives, axis_conf, sign_conf, detail)

    method = "+".join(dict.fromkeys(axis_voters + sign_voters))
    return FrontEstimate(normalize(chosen), confidence, method, alternatives,
                         axis_conf, sign_conf, detail)


def annotate_scene_fronts(scene, use_visibility: bool = False,
                          use_gt: bool = False) -> Dict:
    """Fill in `front`/`front_confidence` for every object in a scene.

    `use_gt` takes the front from `meta["gt_front"]` when present. That is the
    oracle-orientation setting, used by the benchmark to separate "the frame
    convention was wrong" from "we did not know which way the sofa faced".
    """
    room_bounds = scene.room.bounds if scene.room is not None else None
    # Only structures that genuinely wall an object in count as blockers. A
    # chair tucked under a desk must not: the desk's front is precisely where
    # the chair is, and counting it flipped every desk by 90 degrees.
    big = [o.obb for o in scene.objects
           if o.is_room_fixed
           or (o.obb.height > 1.4 and o.obb.footprint_area > 0.30)]
    stats = {"n": 0, "estimated": 0, "front_categories": 0,
             "abstained": 0, "oracle": bool(use_gt),
             "reasons": {}}

    for o in scene.objects:
        stats["n"] += 1
        if o.has_intrinsic_front:
            stats["front_categories"] += 1
        if use_gt and o.meta.get("gt_front"):
            o.front = normalize(np.asarray(o.meta["gt_front"], float))
            o.front_confidence = 1.0
            o.front_method = "oracle"
            stats["estimated"] += 1
            continue
        view_dirs = None
        if scene.trajectory is not None and len(scene.trajectory):
            view_dirs = o.center - scene.trajectory.centers
        blockers = [b for b in big if b is not o.obb]
        est = estimate_front(o.obb, o.points, o.label, room_bounds, blockers,
                             view_dirs, use_visibility)
        o.front = est.front
        o.front_confidence = est.confidence
        o.front_method = est.method
        o.front_alternatives = est.scores
        o.meta["front_detail"] = est.detail
        o.meta["front_axis_confidence"] = est.axis_confidence
        o.meta["front_sign_confidence"] = est.sign_confidence
        if est.front is not None:
            stats["estimated"] += 1
        elif o.has_intrinsic_front:
            stats["abstained"] += 1
            key = est.method.split(":")[0]
            stats["reasons"][key] = stats["reasons"].get(key, 0) + 1
    return stats


def score_fronts(scene) -> Dict:
    """Compare estimated fronts against `meta["gt_front"]`, when available."""
    errs, n_gt, n_ok, misses, flips = [], 0, 0, [], 0
    per_label: Dict[str, List[float]] = {}
    for o in scene.objects:
        gt = o.meta.get("gt_front")
        if not gt:
            continue
        n_gt += 1
        if o.front is None:
            misses.append(o.label)
            continue
        c = float(np.clip(np.dot(normalize(np.asarray(gt, float)),
                                 normalize(o.front)), -1, 1))
        ang = float(np.rad2deg(np.arccos(c)))
        errs.append(ang)
        per_label.setdefault(o.canonical_label, []).append(ang)
        if ang < 45.0:
            n_ok += 1
        elif ang > 135.0:
            flips += 1
    return {
        "n_with_gt": n_gt,
        "n_estimated": len(errs),
        "n_correct_within_45deg": n_ok,
        "n_flipped_180": flips,
        "abstained": misses,
        "mean_error_deg": float(np.mean(errs)) if errs else None,
        "median_error_deg": float(np.median(errs)) if errs else None,
        "per_label_deg": {k: float(np.median(v)) for k, v in sorted(per_label.items())},
    }


def score_against_dominant_normal(scene) -> Dict:
    """Cross-check estimated fronts against ScanNet++'s `dominantNormal`.

    Not ground truth for *facing* -- a dominant plane normal has no sign and,
    for a chair, may be the seat rather than the backrest. It is used only as an
    independent check that the estimated front lies on the right *axis*, which
    is what caught the monitor bug.
    """
    agree, total, per_label = 0, 0, {}
    for o in scene.objects:
        dn = o.meta.get("dominant_normal")
        if dn is None or o.front is None:
            continue
        d = np.asarray(dn, float)[:2]
        n = float(np.linalg.norm(d))
        if n < 0.30:                     # a near-vertical dominant normal says
            continue                     # nothing about a horizontal front
        c = abs(float(np.dot(d / n, o.front[:2] /
                             max(float(np.linalg.norm(o.front[:2])), 1e-9))))
        total += 1
        ok = c > 0.87                    # within 30 degrees of the same axis
        agree += int(ok)
        per_label.setdefault(o.canonical_label, []).append(round(c, 3))
    return {"n_compared": total, "n_same_axis": agree,
            "fraction_same_axis": (agree / total) if total else None,
            "per_label_cos": per_label}

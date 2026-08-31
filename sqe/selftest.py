"""End-to-end self-test on the synthetic rooms.

The synthetic rooms are built so that the egocentric, intrinsic and allocentric
frames give *provably different* answers to particular sentences, and the answers
were worked out by hand from the layout rather than read off this code. That is
what makes this a test rather than a snapshot: if a change collapses the frames
into one, or flips a handedness, or quietly re-signs an ordering axis, these
assertions fail.

Objects are located by label and position, never by id, so inserting a new object
into a room spec does not silently invalidate every expectation.

Run with `sqe selftest`, or under pytest via tests/test_selftest.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .frames.policy import ViewpointSpec, build_frames
from .frames.reference_frame import addressee_frame, egocentric_frame
from .pipeline import finish_scene
from .query.parser_rules import parse
from .query.resolver import Resolver
from .relations.base import RelationConfig
from .scenegraph.objects import Scene


def find(scene: Scene, label: str, near: Tuple[float, float, float],
         tol: float = 0.25):
    """The object with this label closest to `near`, within `tol` metres."""
    from .categories import normalize_label
    want = normalize_label(label)
    best, best_d = None, tol
    for o in scene.objects:
        if o.canonical_label != want:
            continue
        d = float(np.linalg.norm(o.center - np.asarray(near, float)))
        if d < best_d:
            best, best_d = o, d
    if best is None:
        raise AssertionError(
            f"no {label!r} within {tol} m of {near} in scene {scene.scene_id}")
    return best


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def _fmt(scene, oid):
    if oid is None:
        return "none"
    o = scene.by_id(oid)
    return f"{o.label}#{oid}@y={o.center[1]:.2f},x={o.center[0]:.2f}"


def frame_invariants(scene: Scene) -> List[Check]:
    """Sign and handedness properties that must hold for any scene."""
    out: List[Check] = []
    up = scene.up
    anchor = None
    for o in scene.objects:
        if o.front is not None:
            anchor = o
            break
    if anchor is None:
        return [Check("frame invariants", True, "no oriented anchor in this room")]

    A, F = anchor.center, anchor.front
    eye = A + 2.0 * F                       # a viewer standing at the front
    ego = egocentric_frame(A, eye, up=up)
    addr = addressee_frame(A, F, up=up)
    from .frames.reference_frame import intrinsic_frame
    intr = intrinsic_frame(A, F, up=up)

    out.append(Check(
        "egocentric frame is left-handed under 'front = towards the viewer'",
        ego.handedness == -1, f"handedness {ego.handedness}"))
    out.append(Check(
        "intrinsic frame is right-handed",
        intr.handedness == +1, f"handedness {intr.handedness}"))
    out.append(Check(
        "addressee is the mirror of intrinsic",
        bool(np.allclose(addr.right, -intr.right, atol=1e-9)),
        f"{np.round(addr.right, 4)} vs {np.round(-intr.right, 4)}"))
    out.append(Check(
        "addressee equals the egocentric frame of a viewer on the anchor's "
        "front axis",
        bool(np.allclose(addr.right, ego.right, atol=1e-6)
             and np.allclose(addr.front, ego.front, atol=1e-6)),
        f"right {np.round(addr.right, 4)} vs {np.round(ego.right, 4)}"))

    # a point offset along the frame's right axis must read as positive x
    probe = A + 0.5 * intr.right
    x = float(intr.coords(probe[None, :])[0][0])
    out.append(Check("intrinsic right axis maps to positive local x", x > 0.4,
                     f"x={x:.3f}"))
    return out


STUDIO_CASES = [
    # (query, viewpoint position, expected per frame). Worked out by hand from
    # the layout in sqe/data/synthetic.py: four mugs on the bookshelf's middle
    # board at y = 1.2, 1.6, 2.0, 2.4, and the bookshelf's open front faces -x.
    # A viewer standing in the room and looking at the shelf therefore has -y on
    # their right, while the shelf's own right is +y: exactly opposite.
    dict(text="the second mug from the left on the middle shelf",
         eye=(2.0, 2.0, 1.55),
         expect={"egocentric": ("mug", (4.72, 2.00, 0.67)),
                 "intrinsic": ("mug", (4.72, 1.60, 0.67))},
         must_disagree=True),
    dict(text="the leftmost mug on the middle shelf",
         eye=(2.0, 2.0, 1.55),
         expect={"egocentric": ("mug", (4.72, 2.40, 0.67)),
                 "intrinsic": ("mug", (4.72, 1.20, 0.67))},
         must_disagree=True),
    dict(text="the rightmost mug on the middle shelf",
         eye=(2.0, 2.0, 1.55),
         expect={"egocentric": ("mug", (4.72, 1.20, 0.67)),
                 "intrinsic": ("mug", (4.72, 2.40, 0.67))},
         must_disagree=True),
    # On the desk: the laptop faces -y, so its own right is -x while a viewer
    # standing in the room and looking at it has +x on their right.
    dict(text="the mug to the right of the laptop",
         eye=(1.2, 2.0, 1.55),
         expect={"egocentric": ("mug", (1.72, 3.45, 0.80)),
                 "intrinsic": None},
         must_disagree=False),
    dict(text="the bottle to the left of the laptop",
         eye=(1.2, 2.0, 1.55),
         expect={"egocentric": ("bottle", (0.42, 3.45, 0.86)),
                 "intrinsic": None},
         must_disagree=False),
    # The office chair faces +y (towards the desk), so its own right is +x. A
    # viewer standing behind the desk looking back at the chair has -x on their
    # right, so the two readings pick opposite objects.
    dict(text="the trash can to the right of the office chair",
         eye=(1.2, 3.90, 1.55),
         expect={"egocentric": None,
                 "intrinsic": ("trash can", (2.05, 2.85, 0.20))},
         must_disagree=False),
    dict(text="the backpack to the right of the office chair",
         eye=(1.2, 3.90, 1.55),
         expect={"egocentric": ("backpack", (0.40, 2.85, 0.22)),
                 "intrinsic": None},
         must_disagree=False),
    # Frame-free controls: these must be identical under every frame.
    dict(text="the remote on the coffee table", eye=(2.5, 2.0, 1.55),
         expect={"any": ("remote", (1.75, 1.70, 0.42))}, must_disagree=False),
    dict(text="the book on the top shelf", eye=(2.5, 2.0, 1.55),
         expect={"any": ("book", (4.72, 1.70, 1.33))}, must_disagree=False),
    dict(text="the trash can under the desk", eye=(2.5, 2.0, 1.55),
         expect={"any": ("trash can", (2.05, 2.85, 0.20))}, must_disagree=False),
]


def run_studio(verbose: bool = True) -> List[Check]:
    from .data.synthetic import make
    scene = make("studio")
    finish_scene(scene, verbose=False)
    r = Resolver(scene, RelationConfig.load())
    out: List[Check] = []

    for case in STUDIO_CASES:
        vp = ViewpointSpec(mode="position",
                           position=np.asarray(case["eye"], float))
        q = parse(case["text"])
        res = r.resolve(q, vp)
        for frame, want in case["expect"].items():
            want_id = None
            if want is not None:
                want_id = find(scene, want[0], want[1]).id
            if frame == "any":
                got = res.target_id
                ok = got == want_id
                out.append(Check(f"{case['text']!r} -> {want[0]}",
                                 ok, f"got {_fmt(scene, got)}, "
                                     f"want {_fmt(scene, want_id)}"))
                # a frame-free relation must not vary with the frame
                for fk in ("egocentric", "intrinsic", "world"):
                    sub = r.resolve(q, vp, force_frame=fk,
                                    evaluate_alternative_frames=False)
                    out.append(Check(
                        f"{case['text']!r} is frame-free (forced {fk})",
                        sub.target_id == want_id,
                        f"got {_fmt(scene, sub.target_id)}"))
                continue
            sub = r.resolve(q, vp, force_frame=frame,
                            evaluate_alternative_frames=False)
            got = sub.target_id
            if want_id is None:
                from .query.resolver import MIN_ANSWER_SCORE
                top = sub.candidates[0].score if sub.candidates else 0.0
                ok = top < MIN_ANSWER_SCORE
                out.append(Check(
                    f"{case['text']!r} has no answer under {frame}",
                    ok, f"got {_fmt(scene, got)} with score {top:.3f}"))
            else:
                ok = got == want_id
                out.append(Check(
                    f"{case['text']!r} under {frame} -> {want[0]} "
                    f"at {want[1]}",
                    ok, f"got {_fmt(scene, got)}, want {_fmt(scene, want_id)}"))
        if case["must_disagree"]:
            answers = {k: v for k, v in res.frame_answers.items() if v is not None}
            distinct = len(set(answers.values())) > 1
            out.append(Check(
                f"{case['text']!r} is reported as frame-ambiguous",
                distinct and res.ambiguity.frame_ambiguous,
                f"frame answers {answers}, kinds {res.ambiguity.kinds}"))

    out.extend(frame_invariants(scene))

    # the room's canonical forward must be the hand-specified +y
    gt = scene.meta.get("gt_canonical_forward")
    if gt is not None:
        got = scene.room.canonical_forward
        out.append(Check("studio canonical forward is +y",
                         bool(np.dot(got, np.asarray(gt, float)) > 0.9),
                         f"got {np.round(got, 3)}, want {gt}, "
                         f"margin {scene.room.forward_confidence:.3f}"))
    return out


def run_square(verbose: bool = True) -> List[Check]:
    """The hostile room: the allocentric frame must be reported as undetermined,
    and an even shelf count must make 'the middle shelf' ambiguous."""
    from .data.synthetic import make
    from .geom.room import FORWARD_MARGIN_AMBIGUOUS
    from .geom.support import detect_levels, middle_level, shelf_levels
    scene = make("square")
    finish_scene(scene, verbose=False)
    out: List[Check] = []

    out.append(Check(
        "square room's canonical forward is reported as a near-tie",
        scene.room.forward_confidence < FORWARD_MARGIN_AMBIGUOUS,
        f"margin {scene.room.forward_confidence:.4f} "
        f"(threshold {FORWARD_MARGIN_AMBIGUOUS})"))

    shelf = find(scene, "shelf", (0.20, 2.00, 0.90), tol=0.5)
    levels = shelf_levels(detect_levels(shelf.points, shelf.obb), shelf.obb)
    picked, amb = middle_level(levels)
    out.append(Check("four-shelf unit is detected as four shelves",
                     len(levels) == 4, f"found {len(levels)}: "
                                       f"{[round(l.z, 2) for l in levels]}"))
    out.append(Check("'the middle shelf' of four is flagged ambiguous",
                     amb and len(picked) == 2,
                     f"ambiguous={amb}, {len(picked)} candidates"))

    r = Resolver(scene, RelationConfig.load())
    res = r.resolve(parse("the bottle on the middle shelf of the shelf"),
                    ViewpointSpec(mode="position",
                                  position=np.array([2.0, 2.0, 1.55])))
    out.append(Check("query on an even shelf count is flagged",
                     "level_even" in res.ambiguity.kinds,
                     f"kinds {res.ambiguity.kinds}"))

    # a front-less anchor must make the intrinsic frame unavailable
    table = find(scene, "dining table", (2.0, 2.0, 0.37), tol=0.5)
    frames, _ = build_frames(scene, table)
    out.append(Check("intrinsic frame unavailable for a front-less table",
                     not frames["intrinsic"].available,
                     frames["intrinsic"].reason))
    return out


def run_orientation(verbose: bool = True) -> List[Check]:
    """Front estimation against the synthetic ground truth."""
    from .data.synthetic import make
    from .perception.orientation import score_fronts
    out: List[Check] = []
    for room in ("studio", "square"):
        scene = make(room)
        finish_scene(scene, verbose=False)
        sc = score_fronts(scene)
        n_est = sc["n_estimated"]
        out.append(Check(
            f"{room}: no estimated front is flipped by 180 degrees",
            sc["n_flipped_180"] == 0,
            f"{sc['n_flipped_180']} flipped of {n_est} estimated"))
        out.append(Check(
            f"{room}: every estimated front is within 45 degrees of truth",
            sc["n_correct_within_45deg"] == n_est,
            f"{sc['n_correct_within_45deg']}/{n_est} correct, "
            f"median error {sc['median_error_deg']}"))
        out.append(Check(
            f"{room}: at least half the front-bearing objects get a front",
            n_est >= 0.5 * max(sc["n_with_gt"], 1),
            f"{n_est} of {sc['n_with_gt']} with ground truth"))
    return out


def run_selftest(verbose: bool = True) -> int:
    groups = [("frame semantics and studio queries", run_studio),
              ("hostile room and ambiguity", run_square),
              ("front estimation", run_orientation)]
    all_checks: List[Check] = []
    for name, fn in groups:
        checks = fn(verbose)
        all_checks.extend(checks)
        n_ok = sum(1 for c in checks if c.passed)
        if verbose:
            print(f"\n== {name}: {n_ok}/{len(checks)} passed")
            for c in checks:
                mark = "ok  " if c.passed else "FAIL"
                line = f"  [{mark}] {c.name}"
                if not c.passed or verbose:
                    line += f"\n           {c.detail}" if c.detail else ""
                print(line)
    failed = [c for c in all_checks if not c.passed]
    print(f"\n{len(all_checks) - len(failed)}/{len(all_checks)} checks passed")
    if failed:
        print("\nFAILURES:")
        for c in failed:
            print(f"  - {c.name}\n      {c.detail}")
        return 1
    return 0

#!/usr/bin/env python
"""Emit a small benchmark on the synthetic rooms, with hand-derived answers.

The answers here were worked out by reading the room layout in
`sqe/data/synthetic.py`, not by running the resolver. That is what makes this a
usable (if tiny) benchmark rather than a snapshot of current behaviour: if the
resolver regresses, these items fail.

Its purpose is to exercise the whole evaluation path -- conditions, baselines,
attribution, report -- before any real annotation exists, and to show what the
report looks like. It is far too small and too clean to support any claim about
real scenes, and `results/synthetic_example/report.md` says so.

    python scripts/make_synthetic_benchmark.py benchmark/queries/synthetic.jsonl
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sqe.bench.schema import BenchItem, describe_split, write_jsonl
from sqe.data.synthetic import make
from sqe.pipeline import finish_scene
from sqe.query.parser_rules import parse
from sqe.selftest import find

STUDIO_EYE_ROOM = (2.0, 2.0, 1.55)      # standing in the room, facing the shelf
STUDIO_EYE_DESK = (1.2, 2.0, 1.55)      # standing in the room, facing the desk
STUDIO_EYE_WALL = (1.2, 3.90, 1.55)     # behind the desk, facing back at the chair

# (text, eye, gold frame, target as (label, position) or None, ambiguous, kind,
#  answers_by_frame as {frame: (label, pos) or None}, difficulty)
STUDIO = [
    # --- frame-sensitive ordinals on the bookshelf ----------------------
    ("the second mug from the left on the middle shelf", STUDIO_EYE_ROOM,
     "egocentric", ("mug", (4.72, 2.00, 0.67)), True, "frame",
     {"egocentric": ("mug", (4.72, 2.00, 0.67)),
      "intrinsic": ("mug", (4.72, 1.60, 0.67))}, "hard"),
    ("the leftmost mug on the middle shelf", STUDIO_EYE_ROOM,
     "egocentric", ("mug", (4.72, 2.40, 0.67)), True, "frame",
     {"egocentric": ("mug", (4.72, 2.40, 0.67)),
      "intrinsic": ("mug", (4.72, 1.20, 0.67))}, "hard"),
    ("the rightmost mug on the middle shelf", STUDIO_EYE_ROOM,
     "egocentric", ("mug", (4.72, 1.20, 0.67)), True, "frame",
     {"egocentric": ("mug", (4.72, 1.20, 0.67)),
      "intrinsic": ("mug", (4.72, 2.40, 0.67))}, "hard"),
    ("from where I'm standing, the second mug from the left on the middle "
     "shelf", STUDIO_EYE_ROOM, "egocentric", ("mug", (4.72, 2.00, 0.67)),
     False, "none", {}, "medium"),
    ("the second mug from the bookshelf's own left on the middle shelf",
     STUDIO_EYE_ROOM, "intrinsic", ("mug", (4.72, 1.60, 0.67)), False, "none",
     {}, "hard"),

    # --- lateral relations on the desk ---------------------------------
    ("the mug to the right of the laptop", STUDIO_EYE_DESK, "egocentric",
     ("mug", (1.72, 3.45, 0.80)), False, "none",
     {"egocentric": ("mug", (1.72, 3.45, 0.80)), "intrinsic": None}, "medium"),
    ("the bottle to the left of the laptop", STUDIO_EYE_DESK, "egocentric",
     ("bottle", (0.42, 3.45, 0.86)), False, "none",
     {"egocentric": ("bottle", (0.42, 3.45, 0.86)), "intrinsic": None},
     "medium"),
    ("the mug on the laptop's left", STUDIO_EYE_DESK, "intrinsic",
     ("mug", (1.72, 3.45, 0.80)), False, "none", {}, "hard"),

    # --- the chair, whose own right is opposite the viewer's -----------
    ("the trash can to the right of the office chair", STUDIO_EYE_WALL,
     "intrinsic", ("trash can", (2.05, 2.85, 0.20)), True, "frame",
     {"egocentric": ("backpack", (0.40, 2.85, 0.22)),
      "intrinsic": ("trash can", (2.05, 2.85, 0.20))}, "hard"),
    ("the backpack to the right of the office chair", STUDIO_EYE_WALL,
     "egocentric", ("backpack", (0.40, 2.85, 0.22)), True, "frame",
     {"egocentric": ("backpack", (0.40, 2.85, 0.22)), "intrinsic": None},
     "hard"),

    # --- frontal relations, where intrinsic is the default reading -----
    ("the coffee table in front of the sofa", STUDIO_EYE_ROOM, "intrinsic",
     ("coffee table", (1.75, 1.50, 0.20)), False, "none", {}, "medium"),
    ("the keyboard in front of the monitor", STUDIO_EYE_DESK, "intrinsic",
     ("keyboard", (1.20, 3.30, 0.77)), False, "none", {}, "medium"),

    # --- frame-free controls -------------------------------------------
    ("the remote on the coffee table", STUDIO_EYE_ROOM, "any",
     ("remote", (1.75, 1.70, 0.42)), False, "none", {}, "easy"),
    ("the book on the top shelf", STUDIO_EYE_ROOM, "any",
     ("book", (4.72, 1.70, 1.33)), False, "none", {}, "easy"),
    ("the trash can under the desk", STUDIO_EYE_ROOM, "any",
     ("trash can", (2.05, 2.85, 0.20)), False, "none", {}, "easy"),
    ("the monitor on the desk", STUDIO_EYE_ROOM, "any",
     ("monitor", (1.20, 3.85, 0.98)), False, "none", {}, "easy"),
    ("the coffee table between the sofa and the bookshelf", STUDIO_EYE_ROOM,
     "any", ("coffee table", (1.75, 1.50, 0.20)), False, "none", {}, "medium"),
    ("the plant nearest to the window", STUDIO_EYE_ROOM, "any",
     ("plant", (4.45, 3.60, 0.45)), False, "none", {}, "easy"),
    ("the tallest object on the desk", STUDIO_EYE_DESK, "any",
     ("monitor", (1.20, 3.85, 0.98)), False, "none", {}, "medium"),
    ("the keyboard next to the laptop", STUDIO_EYE_DESK, "any",
     ("keyboard", (1.20, 3.30, 0.77)), False, "none", {}, "easy"),
]

SQUARE = [
    # the square room's canonical forward is a coin flip, so any query that
    # leans on it is ambiguous by construction
    ("the mug on the left side of the room", (2.0, 0.6, 1.55), "world",
     None, True, "world_undetermined", {}, "hard"),
    ("the bottle on the middle shelf of the shelf", (2.0, 2.0, 1.55), "any",
     None, True, "level_even", {}, "hard"),
    ("the mug to the left of the bowl", (2.0, 0.6, 1.55), "egocentric",
     ("mug", (1.70, 2.00, 0.80)), False, "none", {}, "medium"),
    ("the bowl on the dining table", (2.0, 0.6, 1.55), "any",
     ("bowl", (2.00, 2.30, 0.79)), False, "none", {}, "easy"),
    ("the plant in the corner nearest the window", (2.0, 2.0, 1.55), "any",
     None, True, "vague", {}, "hard"),
]


def build(room: str, cases, prefix: str):
    scene = make(room)
    finish_scene(scene, verbose=False)
    items = []
    for i, (text, eye, frame, target, amb, kind, by_frame, diff) in enumerate(cases):
        tid = [] if target is None else [find(scene, target[0], target[1]).id]
        abf = {}
        for k, v in (by_frame or {}).items():
            abf[k] = None if v is None else find(scene, v[0], v[1]).id
        q = parse(text)
        items.append(BenchItem(
            id=f"{prefix}_{i:03d}", scene_id=scene.scene_id,
            dataset="synthetic", text=text, target_ids=tid,
            ambiguous=amb, ambiguity_kind=kind,
            relation=q.primary_relation, relation_type=q.relation_type(),
            frame=frame,
            frame_stated_in_text=bool(q.frame_hint),
            answers_by_frame=abf, gold_parse=q.to_dict(),
            viewpoint_mode="position",
            viewpoint_position=[float(x) for x in eye],
            difficulty=diff, annotator="hand-derived from the room layout",
            source="hand_derived_synthetic",
            notes="answer derived from sqe/data/synthetic.py by hand, not from "
                  "the resolver"))
    problems = [(it.id, p) for it in items for p in [it.validate(scene)] if p]
    return items, problems


def main():
    out = (sys.argv[1] if len(sys.argv) > 1
           else "benchmark/queries/synthetic.jsonl")
    items, problems = [], []
    for room, cases, prefix in (("studio", STUDIO, "synth_studio"),
                                ("square", SQUARE, "synth_square")):
        got, probs = build(room, cases, prefix)
        items.extend(got)
        problems.extend(probs)
    if problems:
        print("VALIDATION PROBLEMS:")
        for iid, p in problems:
            print(f"  {iid}: {p}")
    write_jsonl(items, out)
    print(f"wrote {len(items)} items -> {out}")
    import json
    print(json.dumps(describe_split(items), indent=1))


if __name__ == "__main__":
    main()

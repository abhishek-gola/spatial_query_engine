"""Terminal annotation tool.

Blind by default. The tool shows the query, the objects in the scene and a
top-down map, and asks for the answer. It does **not** show what the resolver
would have said, because a human confirming a system's own prediction produces a
benchmark that measures agreement rather than correctness -- and that is the
single easiest way to make the headline number meaningless.

`--show-prediction` exists for debugging and stamps every item it touches with
`source="annotated_with_prediction_shown"`, so the evaluator can report those
separately and you cannot forget you used it.

Per item you supply:

  target   the object id, or several ids separated by spaces when the query is
           genuinely ambiguous, or `-` for "no valid answer"
  frame    which reading the *sentence* means: e (egocentric), i (intrinsic),
           a (addressee), w (world), n (frame-free / any)
  amb      whether the query has a single referent at all
  diff     easy / medium / hard

Commands: `s` skip, `b` back, `r` rewrite the text, `q` save and quit,
`m` redraw the map, `l` list objects, `?` help.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..scenegraph.objects import Object3D, Scene
from .schema import (AMBIGUITY_KINDS, DIFFICULTY, BenchItem, read_jsonl,
                     write_jsonl)

FRAME_KEYS = {"e": "egocentric", "i": "intrinsic", "a": "addressee",
              "w": "world", "n": "any", "b": "egocentric_bearing",
              "m": "egocentric_image", "u": "unspecified"}

AMB_KEYS = {"f": "frame", "o": "ordinal_degenerate", "t": "ordinal_tie",
            "l": "level_even", "s": "score_tie", "a": "anchor",
            "w": "world_undetermined", "v": "vague", "m": "multiple_valid",
            "n": "none"}


def ascii_map(scene: Scene, highlight: Sequence[int] = (),
              width: int = 78, height: int = 24,
              viewpoint: Optional[np.ndarray] = None) -> str:
    """Top-down map of the scene, so geometry is judgeable in a terminal.

    Highlighted objects show their id; everything else shows a dot. North on the
    page is +Y in world coordinates, and the axes are labelled, because an
    unlabelled map would just be another place to smuggle in a frame convention.
    """
    objs = [o for o in scene.objects if not o.is_room_fixed or o.id in highlight]
    if not objs:
        return "(no objects)"
    pts = np.array([o.center[:2] for o in objs])
    lo, hi = pts.min(axis=0) - 0.3, pts.max(axis=0) + 0.3
    if scene.room is not None:
        b = scene.room.bounds
        lo = np.minimum(lo, b.center[:2] - b.half[:2])
        hi = np.maximum(hi, b.center[:2] + b.half[:2])
    span = np.maximum(hi - lo, 1e-3)

    grid = [[" "] * width for _ in range(height)]

    def place(x: float, y: float, text: str):
        cx = int((x - lo[0]) / span[0] * (width - 1))
        cy = int((1.0 - (y - lo[1]) / span[1]) * (height - 1))
        cx = max(0, min(width - len(text), cx))
        cy = max(0, min(height - 1, cy))
        for i, ch in enumerate(text):
            grid[cy][cx + i] = ch

    hl = set(highlight)
    for o in sorted(objs, key=lambda o: o.id in hl):
        place(o.center[0], o.center[1],
              f"[{o.id}]" if o.id in hl else ".")
    if viewpoint is not None:
        place(float(viewpoint[0]), float(viewpoint[1]), "@")

    body = "\n".join("".join(row) for row in grid)
    header = (f"  top-down map   x: {lo[0]:.1f}..{hi[0]:.1f} m (left->right)   "
              f"y: {lo[1]:.1f}..{hi[1]:.1f} m (bottom->top)"
              + ("   @ = viewpoint" if viewpoint is not None else ""))
    return header + "\n" + body


def object_table(scene: Scene, ids: Optional[Sequence[int]] = None,
                 columns: int = 3) -> str:
    objs = ([scene.by_id(i) for i in ids] if ids
            else [o for o in scene.objects if not o.is_room_fixed])
    objs = [o for o in objs if o is not None]
    cells = []
    for o in objs:
        c = o.center
        cells.append(f"{o.id:4d} {o.canonical_label[:16]:16s} "
                     f"({c[0]:5.2f},{c[1]:5.2f},{c[2]:5.2f})")
    rows = []
    for i in range(0, len(cells), columns):
        rows.append("  ".join(f"{c:44s}" for c in cells[i:i + columns]))
    return "\n".join(rows)


def _relevant_ids(scene: Scene, item: BenchItem) -> List[int]:
    """Objects whose class is mentioned in the query, for the map highlight."""
    from ..categories import label_matches, normalize_label
    words = item.text.lower().replace(",", " ").split()
    out = []
    for o in scene.objects:
        lab = o.canonical_label
        if not lab:
            continue
        head = lab.split()[-1]
        if head in words or lab in item.text.lower():
            out.append(o.id)
    return out


def _ask(prompt: str, default: str = "") -> str:
    try:
        s = input(prompt).strip()
    except EOFError:
        return "q"
    return s or default


#: Relation types in descending order of information per label. Lateral
#: relations show the highest measured frame disagreement (24.8% vs 15.4% for
#: frontal on the ScanNet++ scenes), so a fixed annotation budget spent there
#: settles the frame question fastest.
INFORMATIVE_TYPES = ("projective_lateral", "projective_frontal", "ordinal",
                     "vertical", "proximity", "between", "comparative")


def order_queue(items: List[BenchItem], indices: List[int], mode: str,
                scene_for=None, cfg=None, verbose: bool = True) -> List[int]:
    """Order the annotation queue.

    `file` keeps the file's order. `informative` puts the items that actually
    decide something first: queries where two reference frames currently select
    *different* objects, then by relation type, hardest and most frame-relevant
    first. Measuring the disagreement costs one pass over the queue (about 15 s
    for 900 items) and is worth it -- a label on a query where every frame agrees
    tells you almost nothing about the frame.
    """
    if mode == "file":
        return indices
    rank_type = {t: i for i, t in enumerate(INFORMATIVE_TYPES)}
    rank_diff = {"hard": 0, "medium": 1, "easy": 2}

    disagrees: Dict[str, bool] = {}
    if mode == "informative" and scene_for is not None:
        try:
            from .sensitivity import measure
            if verbose:
                print("measuring which queries the frames currently disagree "
                      "on, to put those first ...")
            rows = measure([items[i] for i in indices], scene_for, cfg)
            disagrees = {r.item_id: r.disagreed for r in rows}
        except Exception as exc:
            if verbose:
                print(f"  (could not measure frame disagreement: {exc}; "
                      f"falling back to relation-type order)")

    def key(i: int):
        it = items[i]
        return (0 if disagrees.get(it.id) else 1,
                rank_type.get(it.relation_type or "", 99),
                rank_diff.get(it.difficulty, 9),
                it.scene_id, it.id)

    out = sorted(indices, key=key)
    if verbose and disagrees:
        n = sum(1 for i in out if disagrees.get(items[i].id))
        print(f"  {n} of {len(out)} queries have frames that currently "
              f"disagree; those come first")
    return out


def annotate(items: List[BenchItem], scene_for, out_path: str,
             annotator: str = "", show_prediction: bool = False,
             resolver_for=None, start: int = 0,
             only_unannotated: bool = True,
             relation_types: Optional[Sequence[str]] = None,
             order: str = "informative", target_count: Optional[int] = None,
             cfg=None) -> List[BenchItem]:
    """Interactive loop. Saves after every item, so an interrupt loses nothing."""
    todo = list(range(len(items)))
    if only_unannotated:
        todo = [i for i in todo
                if not items[i].target_ids and not items[i].ambiguous]
    if relation_types:
        want = set(relation_types)
        todo = [i for i in todo if (items[i].relation_type or "") in want]
    todo = order_queue(items, todo, order, scene_for, cfg)
    todo = [i for i in todo if i >= 0][start:]
    if not todo:
        print("nothing left to annotate in this file")
        return items

    print(__doc__)
    pos = 0
    while 0 <= pos < len(todo):
        k = todo[pos]
        it = items[k]
        try:
            scene = scene_for(it.scene_id)
        except Exception as exc:
            print(f"! scene {it.scene_id} unavailable ({exc}); skipping")
            pos += 1
            continue

        hl = _relevant_ids(scene, it)
        vp = None
        if it.viewpoint_position:
            vp = np.asarray(it.viewpoint_position, float)
        elif (it.viewpoint_mode in ("best_view", "nearest", "mean")
              and scene.trajectory is not None and len(scene.trajectory)):
            vp = scene.trajectory.centers.mean(axis=0)

        cols = shutil.get_terminal_size((100, 30)).columns
        n_done = sum(1 for x in items if x.target_ids or x.ambiguous)
        goal = f" | goal {target_count}" if target_count else ""
        print("\n" + "=" * min(cols, 100))
        print(f"[{pos + 1}/{len(todo)}]  {it.id}   scene {it.scene_id}   "
              f"({it.relation_type or '?'}, suggested {it.difficulty})"
              f"   annotated so far: {n_done}{goal}")
        print(f'QUERY: "{it.text}"')
        if it.notes:
            visible = " | ".join(p for p in it.notes.split(" | ")
                                 if not p.startswith("_proposal"))
            if visible:
                print(f"  note: {visible}")
        print(f"  viewpoint: {it.viewpoint_mode}"
              + (f" {np.round(vp, 2).tolist()}" if vp is not None else ""))
        print()
        print(ascii_map(scene, hl, min(cols - 2, 96), 22, vp))
        print()
        print("candidates mentioned in the query:")
        print(object_table(scene, hl) or "  (none matched by class name)")

        if show_prediction and resolver_for is not None:
            from ..query.parser_rules import parse
            res = resolver_for(it.scene_id).resolve(parse(it.text),
                                                    it.viewpoint_spec())
            print(f"\n  [PREDICTION SHOWN] system says: {res.target_id} "
                  f"frame={res.frame_used} amb={res.ambiguity.kinds}")

        cmd = _ask("\n  target id(s) / s,b,r,m,l,q > ")
        low = cmd.lower()
        if low == "q":
            break
        if low == "s":
            pos += 1
            continue
        if low == "b":
            pos = max(0, pos - 1)
            continue
        if low == "m":
            continue
        if low == "l":
            print(object_table(scene))
            continue
        if low == "?":
            print(__doc__)
            continue
        if low == "r":
            new = _ask("  new text > ")
            if new:
                it.text = new
                it.source = "generated_rewritten"
                write_jsonl(items, out_path)
            continue

        if low in ("-", "none"):
            ids: List[int] = []
        else:
            try:
                ids = [int(x) for x in cmd.replace(",", " ").split()]
            except ValueError:
                print("  ! not a list of ids")
                continue
            bad = [i for i in ids if scene.by_id(i) is None]
            if bad:
                print(f"  ! not in this scene: {bad}")
                continue

        fk = _ask("  frame  e=egocentric i=intrinsic a=addressee w=world "
                  "n=frame-free > ", "n").lower()[:1]
        frame = FRAME_KEYS.get(fk, "unspecified")
        stated = _ask("  is the frame stated in the sentence? [y/N] > ",
                      "n").lower().startswith("y")
        amb_default = "y" if len(ids) != 1 else "n"
        is_amb = _ask(f"  ambiguous? [y/N] ({amb_default}) > ",
                      amb_default).lower().startswith("y")
        kind = "none"
        if is_amb:
            ak = _ask("  ambiguity  f=frame o=ord-degenerate t=ord-tie "
                      "l=level-even s=score-tie a=anchor v=vague "
                      "m=multiple-valid > ", "v").lower()[:1]
            kind = AMB_KEYS.get(ak, "vague")
        diff = _ask(f"  difficulty [e/m/h] ({it.difficulty[0]}) > ",
                    it.difficulty[0]).lower()[:1]

        it.target_ids = ids
        it.frame = frame
        it.frame_stated_in_text = stated
        it.ambiguous = is_amb
        it.ambiguity_kind = kind
        it.difficulty = {"e": "easy", "m": "medium", "h": "hard"}.get(
            diff, it.difficulty)
        it.annotator = annotator
        it.source = ("annotated_with_prediction_shown" if show_prediction
                     else ("generated_reviewed" if it.source.startswith("generated")
                           else "manual"))
        from ..query.parser_rules import parse
        it.gold_parse = parse(it.text).to_dict()

        problems = it.validate(scene)
        if problems:
            print(f"  ! item still has problems: {problems}")
        write_jsonl(items, out_path)
        n_done = sum(1 for x in items if x.target_ids or x.ambiguous)
        print(f"  saved ({n_done}/{len(items)} annotated)")
        if target_count and n_done >= target_count:
            print(f"\n  reached the goal of {target_count} annotated items. "
                  f"Carry on, or press q to stop and run the benchmark.")
        pos += 1

    write_jsonl(items, out_path)
    done = sum(1 for x in items if x.target_ids or x.ambiguous)
    print(f"\nwrote {out_path}: {done}/{len(items)} annotated")
    return items

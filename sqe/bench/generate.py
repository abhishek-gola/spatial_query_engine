"""Proposing benchmark queries for a human to check.

Hand-writing 300-500 spatial queries with correct answers is a lot of work, and
most of it is wasted on queries that are trivially easy. This module proposes
candidates from the scene geometry, weighted towards the ones worth annotating:

* pairs where the reference frames **disagree** -- the items the whole benchmark
  turns on;
* ordinals over three or more same-class objects on a shared support surface;
* frame-free controls (`on`, `above`, `between`, `near`, comparatives) on the
  same scenes, so the report can compare frame-dependent against
  frame-independent accuracy with perception held constant.

What it deliberately does **not** do is fill in the answer. The generator writes
`target_ids: []` and `frame: "unspecified"`, and the annotation tool shows the
geometry without showing what the system would have said. Pre-filling the
system's own prediction and asking a human to confirm it produces a benchmark
that measures agreement with the system rather than correctness -- the failure
mode that makes a lot of auto-generated 3-D grounding data worthless.

The one thing the generator does pre-compute is `answers_by_frame`, written into
a separate `_proposal` block that the annotation tool hides by default and the
evaluator ignores. It is there so that, *after* annotating, you can see which
items turned out to be frame-sensitive without re-running anything.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..categories import is_room_fixed
from ..frames.policy import ViewpointSpec, build_frames, decide_frame
from ..frames.reference_frame import ReferenceFrame
from ..geom.support import detect_levels, shelf_levels
from ..relations.base import RelationConfig
from ..relations.projective import projective_score
from ..relations.proximity import between_score, near_score, next_to_score
from ..relations.vertical import above_score, on_score
from ..scenegraph.objects import Object3D, Scene
from .schema import BenchItem

#: Templates, kept plain. Natural phrasing variety is the annotator's job --
#: they are free to rewrite the text, and the tool records that they did.
LATERAL = {"left": "the {t} to the left of the {a}",
           "right": "the {t} to the right of the {a}"}
FRONTAL = {"front": "the {t} in front of the {a}",
           "behind": "the {t} behind the {a}"}
VERTICAL = {"on": "the {t} on the {a}",
            "above": "the {t} above the {a}"}
PROXIMITY = {"next_to": "the {t} next to the {a}",
             "near": "the {t} nearest to the {a}"}

ORDINAL_WORDS = ["first", "second", "third", "fourth"]

#: Objects smaller than this are too small to be reliable referents.
MIN_TARGET_DIAMETER = 0.05

#: Classes that make good anchors: big, nameable, and the kind of thing a person
#: would actually orient by. A benchmark full of "the box to the left of the
#: blind rail" would be measuring nothing anyone cares about, and alphabetical
#: ordering over a ScanNet++ label set produces exactly that.
SALIENT_ANCHORS = {
    "whiteboard": 3.0, "door": 3.0, "window": 2.6, "tv": 2.6, "monitor": 2.2,
    "table": 2.4, "desk": 2.6, "sofa": 2.6, "bed": 2.6, "bookshelf": 2.4,
    "shelf": 2.2, "cabinet": 2.0, "storage cabinet": 2.0, "wardrobe": 2.2,
    "office chair": 2.0, "chair": 2.0, "sink": 2.2, "toilet": 2.2,
    "refrigerator": 2.4, "stairs": 2.2, "pillar": 1.8, "dresser": 2.0,
    "nightstand": 1.8, "coffee table": 2.0, "dining table": 2.4,
    "counter": 2.2, "kitchen counter": 2.2, "printer": 1.6, "heater": 1.4,
    "keyboard": 1.4, "laptop": 1.8, "computer tower": 1.4, "trash can": 1.4,
    "plant": 1.6, "lamp": 1.4, "picture": 1.6, "mirror": 1.8, "curtain": 1.2,
}

#: Fittings nobody orients by. Kept out of anchor position entirely.
POOR_ANCHORS = {
    "power socket", "wall outlet", "light switch", "thermostat",
    "electrical duct", "electric duct", "blind rail", "door frame",
    "window frame", "fake ceiling", "ceiling", "vent", "pipe", "rug",
    "ceiling lamp", "whiteboard eraser", "object", "wall",
}

#: Only these classes get "the middle shelf of the X" proposals. A table with
#: two detected horizontal surfaces is a table with a lower shelf, and
#: "the top shelf of the table" is not something anyone says.
LEVEL_IDIOMATIC = {"shelf", "bookshelf", "shelving unit", "cabinet",
                   "storage cabinet", "wardrobe", "dresser", "nightstand",
                   "kitchen cabinet", "sideboard", "closet"}

#: How many anchor classes to consider, best-salience first.
MAX_ANCHOR_CLASSES = 12


@dataclass
class Proposal:
    text: str
    relation: str
    relation_type: str
    target_hint: Optional[int]
    anchor_id: Optional[int]
    answers_by_frame: Dict[str, Optional[int]]
    frame_sensitive: bool
    difficulty: str
    note: str = ""


def _usable(o: Object3D, allow_room_fixed: bool = False) -> bool:
    if o.meta.get("unlabelled"):
        return False
    # a mislabelled or merged annotation makes an unanswerable query whose
    # failure would be attributed to geometry
    if o.meta.get("suspect_instance"):
        return False
    if not allow_room_fixed and o.is_room_fixed:
        return False
    if o.diameter < MIN_TARGET_DIAMETER:
        return False
    return True


def _count(scene: Scene, label: str) -> int:
    return sum(1 for x in scene.objects if x.canonical_label == label)


def _classes(objs: Sequence[Object3D]) -> Dict[str, List[Object3D]]:
    out: Dict[str, List[Object3D]] = {}
    for o in objs:
        out.setdefault(o.canonical_label, []).append(o)
    return out


def _frame_answers(scene: Scene, targets: Sequence[Object3D],
                   anchors: Sequence[Object3D], relation: str,
                   viewpoint: ViewpointSpec, cfg: RelationConfig
                   ) -> Dict[str, Optional[int]]:
    """Best (target, anchor) pair for `relation` under each frame.

    Scored jointly over anchor instances, the same way the resolver does it, so
    a proposal's frame-sensitivity flag means the same thing as the resolver's.
    """
    out: Dict[str, Optional[int]] = {}
    for kind in ("egocentric", "intrinsic", "addressee", "world"):
        best, best_s = None, 0.0
        for a in anchors:
            frames, _ = build_frames(scene, a, (kind,), viewpoint)
            f = frames[kind]
            if not f.available:
                continue
            for t in targets:
                if t.id == a.id:
                    continue
                sc = projective_score(t.obb, a.obb, f, relation, cfg).value
                if sc > best_s:
                    best, best_s = t.id, sc
        out[kind] = best if best_s >= 0.12 else None
    return out


def _pairs(targets, anchors, score_fn, thresh: float = 0.5):
    """Every (target, anchor) pair whose score clears `thresh`, best first."""
    hits = []
    for a in anchors:
        for t in targets:
            if t.id == a.id:
                continue
            v = score_fn(t, a)
            if v >= thresh:
                hits.append((v, t, a))
    hits.sort(key=lambda h: -h[0])
    return hits


def anchor_salience(cls: str, objs: Sequence[Object3D]) -> float:
    """How good a class is as an anchor. Higher is better; 0 means unusable."""
    if cls in POOR_ANCHORS or not cls:
        return 0.0
    base = SALIENT_ANCHORS.get(cls, 1.0)
    vol = float(np.median([o.obb.volume for o in objs]))
    size = float(np.clip(np.log1p(vol * 20.0) / 3.0, 0.0, 1.0))
    # a class with one instance names itself; ten instances of it do not
    uniqueness = 1.0 / (1.0 + 0.25 * (len(objs) - 1))
    oriented = 0.3 if any(o.front is not None for o in objs) else 0.0
    return base * (0.45 + 0.35 * size + 0.20 * uniqueness) + oriented


def _ranked_anchor_classes(scene: Scene, allow_room_fixed: bool = True
                           ) -> List[Tuple[str, List[Object3D]]]:
    groups = _classes([o for o in scene.objects
                       if _usable(o, allow_room_fixed=allow_room_fixed)])
    scored = [(anchor_salience(c, o), c, o) for c, o in groups.items()]
    scored = [t for t in scored if t[0] > 0.0]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [(c, o) for _, c, o in scored[:MAX_ANCHOR_CLASSES]]


def _difficulty(n_targets_satisfying: int, frame_sensitive: bool,
                n_anchor_instances: int) -> str:
    if frame_sensitive:
        return "hard"
    if n_targets_satisfying > 1 or n_anchor_instances > 2:
        return "medium"
    return "easy"


def propose_projective(scene: Scene, cfg: RelationConfig,
                       viewpoint: Optional[ViewpointSpec] = None,
                       max_per_relation: int = 25,
                       enrich: bool = True,
                       seed: int = 0) -> List[Proposal]:
    """Lateral and frontal proposals, frame-disagreement cases first.

    Anchor classes with several instances are allowed. In a real office almost
    every anchor class is repeated -- four tables, six monitors, five keyboards
    -- and excluding them left nothing to propose. What matters instead is how
    many (target, anchor) pairs satisfy the relation, which is recorded so the
    annotator knows what they are looking at.
    """
    viewpoint = viewpoint or ViewpointSpec()
    out: List[Proposal] = []
    by_class = _classes([o for o in scene.objects if _usable(o)])
    anchor_classes = _ranked_anchor_classes(scene)

    for relation in list(LATERAL) + list(FRONTAL):
        template = LATERAL.get(relation) or FRONTAL[relation]
        rt = ("projective_lateral" if relation in LATERAL
              else "projective_frontal")
        found: List[Proposal] = []
        for a_cls, a_objs in anchor_classes:
            for t_cls, t_objs in sorted(by_class.items()):
                if t_cls == a_cls:
                    continue
                # only bother when some instance pair is reasonably close
                if not any(float(np.linalg.norm(t.center - a.center)) < 3.5
                           for t in t_objs for a in a_objs):
                    continue
                fa = _frame_answers(scene, t_objs, a_objs, relation,
                                    viewpoint, cfg)
                real = {k: v for k, v in fa.items() if v is not None}
                if not real:
                    continue
                sensitive = len(set(real.values())) > 1
                found.append(Proposal(
                    text=template.format(t=t_cls, a=a_cls),
                    relation=relation, relation_type=rt,
                    target_hint=None, anchor_id=a_objs[0].id,
                    answers_by_frame=fa, frame_sensitive=sensitive,
                    difficulty=_difficulty(len(set(real.values())), sensitive,
                                           len(a_objs)),
                    note=(("the frames disagree here; " if sensitive else "")
                          + f"{len(t_objs)} {t_cls}(s), {len(a_objs)} "
                            f"{a_cls}(s) in the scene")))
        # `enrich` puts frame-sensitive candidates first before the cap, which
        # makes annotation efficient -- and biases any rate measured over the
        # result. Any statistic computed on an enriched set is a rate over
        # "queries chosen for being frame-sensitive", not over queries. Use
        # enrich=False for the unbiased denominator; both are reported.
        if enrich:
            found.sort(key=lambda p: (not p.frame_sensitive,
                                      -sum(1 for v in p.answers_by_frame.values()
                                           if v is not None)))
            out.extend(found[:max_per_relation])
        else:
            rng = np.random.default_rng(seed + hash(relation) % 10_000)
            idx = rng.permutation(len(found))[:max_per_relation]
            out.extend([found[int(i)] for i in idx])
    return out


def propose_ordinals(scene: Scene, cfg: RelationConfig,
                     viewpoint: Optional[ViewpointSpec] = None,
                     max_items: int = 30) -> List[Proposal]:
    """Ordinals over three or more same-class objects sharing a support."""
    viewpoint = viewpoint or ViewpointSpec()
    out: List[Proposal] = []
    supports = [o for o in scene.objects
                if o.is_support_surface and o.points is not None
                and len(o.points) > 200
                and o.canonical_label not in POOR_ANCHORS
                and o.canonical_label not in ("floor",)]
    seen_support_class = set()

    for sup in supports:
        levels = shelf_levels(detect_levels(sup.points, sup.obb), sup.obb)
        on_it: Dict[str, List[Object3D]] = {}
        for o in scene.objects:
            if o.id == sup.id or not _usable(o):
                continue
            if on_score(o.obb, sup.obb, cfg, levels).value > 0.25:
                on_it.setdefault(o.canonical_label, []).append(o)
        for cls, group in sorted(on_it.items()):
            if len(group) >= 3:
                for word in ORDINAL_WORDS[:min(3, len(group))]:
                    for side in ("left", "right"):
                        out.append(Proposal(
                            text=(f"the {word} {cls} from the {side} on the "
                                  f"{sup.canonical_label}"),
                            relation="ordinal", relation_type="ordinal",
                            target_hint=None, anchor_id=sup.id,
                            answers_by_frame={}, frame_sensitive=True,
                            difficulty="hard",
                            note=f"{len(group)} {cls}s on this "
                                 f"{sup.canonical_label}"))
        if len(levels) >= 2 and sup.canonical_label in LEVEL_IDIOMATIC:
            key = (sup.canonical_label, len(levels))
            if key not in seen_support_class:
                seen_support_class.add(key)
                word = "middle" if len(levels) % 2 == 1 else "top"
                for cls in sorted(on_it):
                    out.append(Proposal(
                        text=f"the {cls} on the {word} shelf of the "
                             f"{sup.canonical_label}",
                        relation="on", relation_type="vertical",
                        target_hint=None, anchor_id=sup.id,
                        answers_by_frame={}, frame_sensitive=False,
                        difficulty="medium",
                        note=f"{len(levels)} shelf levels detected"
                             + ("; even count, so 'middle' would be ambiguous"
                                if len(levels) % 2 == 0 else "")))
    # whole-scene ordinals, which need no support surface
    by_class = _classes([o for o in scene.objects if _usable(o)])
    for cls, group in sorted(by_class.items()):
        if len(group) < 3:
            continue
        for word in ORDINAL_WORDS[:2]:
            out.append(Proposal(
                text=f"the {word} {cls} from the left",
                relation="ordinal", relation_type="ordinal",
                target_hint=None, anchor_id=None, answers_by_frame={},
                frame_sensitive=True, difficulty="hard",
                note=f"{len(group)} {cls}s in the scene, no support surface "
                     f"named, so the ordering axis comes from the frame alone"))
    return out[:max_items]


def propose_controls(scene: Scene, cfg: RelationConfig,
                     max_items: int = 70) -> List[Proposal]:
    """Frame-independent controls: on, above, next to, nearest, between, size.

    These carry the report's most important comparison. If frame-free relations
    score well on the same scenes with the same instances, the projective gap is
    not a perception problem -- and that inference needs the controls to exist.
    """
    out: List[Proposal] = []
    by_class = _classes([o for o in scene.objects if _usable(o)])
    anchor_classes = _ranked_anchor_classes(scene)

    scorers = {
        "on": (VERTICAL["on"], "vertical",
               lambda t, a: on_score(t.obb, a.obb, cfg).value),
        "above": (VERTICAL["above"], "vertical",
                  lambda t, a: above_score(t.obb, a.obb, cfg).value),
        "next_to": (PROXIMITY["next_to"], "proximity",
                    lambda t, a: next_to_score(t.obb, a.obb, cfg, t.points,
                                               a.points).value),
        "near": (PROXIMITY["near"], "proximity",
                 lambda t, a: near_score(t.obb, a.obb, cfg, t.points,
                                         a.points).value),
    }

    for relation, (template, rt, fn) in scorers.items():
        for a_cls, a_objs in anchor_classes:
            for t_cls, t_objs in sorted(by_class.items()):
                if t_cls == a_cls:
                    continue
                hits = _pairs(t_objs, a_objs, fn, 0.55)
                if not hits:
                    continue
                n_targets = len({t.id for _, t, _ in hits})
                out.append(Proposal(
                    text=template.format(t=t_cls, a=a_cls),
                    relation=relation, relation_type=rt, target_hint=None,
                    anchor_id=hits[0][2].id, answers_by_frame={},
                    frame_sensitive=False,
                    difficulty=_difficulty(n_targets, False, len(a_objs)),
                    note=(f"frame-independent control; {n_targets} of "
                          f"{len(t_objs)} {t_cls}(s) satisfy it")))

    # between, over pairs of anchor classes
    a_list = list(anchor_classes)
    for (ac1, ao1), (ac2, ao2) in itertools.islice(
            itertools.combinations(a_list, 2), 600):
        for t_cls, t_objs in sorted(by_class.items()):
            if t_cls in (ac1, ac2):
                continue
            hit = None
            for a in ao1:
                for b in ao2:
                    for t in t_objs:
                        if t.id in (a.id, b.id):
                            continue
                        if between_score(t.obb, a.obb, b.obb, cfg).value > 0.6:
                            hit = (t, a, b)
                            break
                    if hit:
                        break
                if hit:
                    break
            if hit:
                out.append(Proposal(
                    text=f"the {t_cls} between the {ac1} and the {ac2}",
                    relation="between", relation_type="between",
                    target_hint=None, anchor_id=hit[1].id, answers_by_frame={},
                    frame_sensitive=False, difficulty="medium",
                    note="frame-independent control"))
                break

    # comparatives over classes with several instances
    for cls, group in sorted(by_class.items()):
        if len(group) < 2:
            continue
        for word in ("tallest", "biggest"):
            out.append(Proposal(
                text=f"the {word} {cls}", relation=word,
                relation_type="comparative", target_hint=None, anchor_id=None,
                answers_by_frame={}, frame_sensitive=False, difficulty="easy",
                note=f"{len(group)} instances of {cls}"))

    # interleave the relation families so a truncation keeps the mix balanced
    buckets: Dict[str, List[Proposal]] = {}
    for p in out:
        buckets.setdefault(p.relation, []).append(p)
    mixed: List[Proposal] = []
    while any(buckets.values()) and len(mixed) < max_items:
        for k in list(buckets):
            if buckets[k]:
                mixed.append(buckets[k].pop(0))
                if len(mixed) >= max_items:
                    break
    seen, uniq_out = set(), []
    for p in mixed:
        if p.text in seen:
            continue
        seen.add(p.text)
        uniq_out.append(p)
    return uniq_out


def propose_scene(scene: Scene, cfg: Optional[RelationConfig] = None,
                  viewpoint: Optional[ViewpointSpec] = None,
                  max_projective: int = 25, max_ordinal: int = 30,
                  max_controls: int = 70, enrich: bool = True,
                  seed: int = 0) -> List[Proposal]:
    cfg = cfg or RelationConfig.load()
    props = (propose_projective(scene, cfg, viewpoint, max_projective,
                                enrich=enrich, seed=seed)
             + propose_ordinals(scene, cfg, viewpoint, max_ordinal)
             + propose_controls(scene, cfg, max_controls))
    seen, out = set(), []
    for p in props:
        if p.text in seen:
            continue
        seen.add(p.text)
        out.append(p)
    return out


def to_items(scene: Scene, proposals: Sequence[Proposal],
             prefix: str = "", viewpoint: Optional[ViewpointSpec] = None
             ) -> List[BenchItem]:
    """Turn proposals into unannotated benchmark items.

    Answers and frames are left blank on purpose; the proposal's computed
    `answers_by_frame` goes into `notes` as a hidden hint, never into the gold
    fields.
    """
    viewpoint = viewpoint or ViewpointSpec()
    out: List[BenchItem] = []
    for i, p in enumerate(proposals):
        note_bits = [p.note] if p.note else []
        if p.answers_by_frame:
            note_bits.append("_proposal frame answers: " + ", ".join(
                f"{k}={v}" for k, v in sorted(p.answers_by_frame.items())))
        out.append(BenchItem(
            id=f"{prefix or scene.scene_id}_{i:04d}",
            scene_id=scene.scene_id, dataset=scene.dataset, text=p.text,
            target_ids=[], ambiguous=False, ambiguity_kind="none",
            relation=p.relation, relation_type=p.relation_type,
            frame="unspecified", frame_stated_in_text=False,
            answers_by_frame={}, gold_parse=None,
            viewpoint_mode=viewpoint.mode,
            viewpoint_index=viewpoint.index,
            viewpoint_position=(None if viewpoint.position is None
                                else list(map(float, viewpoint.position))),
            viewpoint_landmark=viewpoint.landmark,
            difficulty=p.difficulty, source="generated",
            notes=" | ".join(note_bits)))
    return out

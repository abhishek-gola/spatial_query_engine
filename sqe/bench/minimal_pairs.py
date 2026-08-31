"""Minimal pairs that differ only in the frame cue.

The stimulus for the experiment that matters: two utterances identical except
for an explicit marker of *whose* left is meant, on a scene where the two
readings provably select different objects.

    A  "from where I am standing, the mug to the left of the laptop"
    B  "the mug to the laptop's own left"

Both are grammatical, both are unambiguous, and the correct answer differs. A
system that answers them identically is not making a mistake about geometry -- it
is failing to represent the frame at all, and is unresponsive to being told which
one to use. That is a much sharper claim than an accuracy gap, and a minimal pair
is what makes it airtight: nothing else varies.

Every pair is validated before it is emitted, and the validation is strict
because a bad pair would produce a spurious result:

1. **The cue reads as intended.** `sqe.frames.cues` must recover exactly the
   frame the phrasing was built for. If our own cue extractor cannot tell the two
   apart, the pair is not minimal in the way that matters.
2. **The readings genuinely differ.** Forcing each frame must give a different
   object, each scoring above the resolver's answer threshold under its own
   frame -- so neither arm is "no answer", which a model could match by accident.
3. **Same class.** Both candidate answers are instances of the target class, so
   the distinction cannot be made on the class name alone.
4. **The neutral phrasing is recorded** -- the same sentence with no cue -- which
   is what measures a system's *default* convention rather than its ability to
   switch.

The pairs are also the right stimulus for the human study: if people split on the
neutral phrasing but agree on the cued arms, that is direct evidence that the
ambiguity is in the language rather than in anyone's implementation.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..frames.cues import extract_cues
from ..frames.policy import ViewpointSpec
from ..query.parser_rules import parse
from ..query.resolver import MIN_ANSWER_SCORE, Resolver
from ..relations.base import RelationConfig
from ..scenegraph.objects import Object3D, Scene

#: Phrasings that pin a frame. Each must be recovered by `extract_cues` as the
#: frame it claims, which `validate_templates` checks at import-time cost zero
#: and `sqe.bench.minimal_pairs` tests assert.
#:
#: `neutral` carries no marker and is what reveals a default convention.
#: Every cued phrasing must leave the anchor as an explicit noun in the part
#: that survives cue-stripping. The first version had
#: `"from the {a}'s point of view, the {t} in front of it"` and
#: `"facing the {a}, the {t} to the {d}"` -- strip the cue and the anchor is a
#: pronoun, or gone entirely, so that arm was resolving against *no anchor*. Both
#: passed a cue-recovery check and were still wrong, which is why
#: `validate_templates` now parses the filled sentence and insists the intended
#: relation and anchor class both survive.
TEMPLATES: Dict[str, Dict[str, str]] = {
    "lateral": {
        "neutral": "the {t} to the {d} of the {a}",
        "egocentric": "from where I am standing, the {t} to the {d} of the {a}",
        "intrinsic": "the {t} to the {a}'s own {d}",
        "addressee": "the {t} to the {d} of the {a}, facing the {a}",
    },
    # No `addressee` arm for front/behind, and that is a fact about the frames
    # rather than an omission: the addressee frame shares the anchor's front axis
    # and mirrors only left/right, so for front/behind it is identical to
    # intrinsic and could never form a minimal pair with it.
    "frontal": {
        "neutral": "the {t} {d} the {a}",
        "egocentric": "from where I am standing, the {t} {d} the {a}",
        "intrinsic": "the {t} {d} the {a}, from the {a}'s point of view",
    },
}

#: Control paraphrases: equally awkward, **not** frame-contrastive, same correct
#: answer as the neutral phrasing.
#:
#: These exist to separate two explanations of a `frame_blind` result. A model
#: that answers both arms of a minimal pair identically might have no
#: representation of the frame -- the claim -- or might simply be failing to
#: parse an unusually-shaped sentence, since both cued arms are wordier and more
#: contorted than anything in its training distribution. If the same model also
#: gives one answer to both of these controls, which are equally contorted but
#: differ in nothing that should change the answer, then parse fragility is
#: ruled out and frame-blindness is the remaining explanation.
#:
#: The design requirement is that the pair members must be *matched for
#: awkwardness* and *identical in meaning*. Each carries a clause of comparable
#: length and subordination to the frame cues, phrased so it adds no spatial
#: information: a redundant restatement, or an irrelevant hedge.
#: One leading clause and one trailing clause, matching the shape of the frame
#: cues themselves ("from where I am standing, …" leads; ", from the X's point of
#: view" trails). Matching the *shape* as well as the length is the point: a
#: model that is thrown by a trailing subordinate clause should fail here too,
#: and then its minimal-pair result cannot be read as frame-blindness.
CONTROL_TEMPLATES: Dict[str, Dict[str, str]] = {
    "lateral": {
        "a": "as far as I can tell, the {t} to the {d} of the {a}",
        "b": "the {t} to the {d} of the {a}, if I am not mistaken",
    },
    "frontal": {
        "a": "as far as I can tell, the {t} {d} the {a}",
        "b": "the {t} {d} the {a}, if I am not mistaken",
    },
}

#: Surface forms for each relation, per template family.
SURFACE = {
    "left": ("lateral", "left"),
    "right": ("lateral", "right"),
    "front": ("frontal", "in front of"),
    "behind": ("frontal", "behind"),
}


@dataclass
class Arm:
    """One side of a pair: a phrasing, the frame it names, its gold answer."""
    frame: str
    text: str
    answer_id: int
    score: float
    cue_recovered: Optional[str] = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ControlPair:
    """Two equally-awkward paraphrases with the same correct answer.

    The negative control for `frame_blind`: a system SHOULD answer these
    identically. If it does not, its answers are unstable to surface form and no
    minimal-pair result from it can be interpreted.
    """
    id: str
    scene_id: str
    relation: str
    anchor_id: int
    anchor_label: str
    target_class: str
    texts: List[str]
    expected_answer_id: int
    viewpoint: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @staticmethod
    def from_dict(d: dict) -> "ControlPair":
        return ControlPair(**d)


@dataclass
class MinimalPair:
    id: str
    scene_id: str
    relation: str
    anchor_id: int
    anchor_label: str
    target_class: str
    arms: List[Arm]
    neutral_text: str
    neutral_answer_id: Optional[int]
    neutral_frame_chosen: Optional[str]
    candidate_ids: List[int]
    n_class_instances: int
    viewpoint: dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def answers(self) -> Dict[str, int]:
        return {a.frame: a.answer_id for a in self.arms}

    def to_dict(self) -> dict:
        return {"id": self.id, "scene_id": self.scene_id,
                "relation": self.relation, "anchor_id": self.anchor_id,
                "anchor_label": self.anchor_label,
                "target_class": self.target_class,
                "arms": [a.to_dict() for a in self.arms],
                "neutral_text": self.neutral_text,
                "neutral_answer_id": self.neutral_answer_id,
                "neutral_frame_chosen": self.neutral_frame_chosen,
                "candidate_ids": list(self.candidate_ids),
                "n_class_instances": self.n_class_instances,
                "viewpoint": self.viewpoint, "notes": list(self.notes)}

    @staticmethod
    def from_dict(d: dict) -> "MinimalPair":
        return MinimalPair(
            id=d["id"], scene_id=d["scene_id"], relation=d["relation"],
            anchor_id=int(d["anchor_id"]), anchor_label=d["anchor_label"],
            target_class=d["target_class"],
            arms=[Arm(**a) for a in d["arms"]],
            neutral_text=d["neutral_text"],
            neutral_answer_id=d.get("neutral_answer_id"),
            neutral_frame_chosen=d.get("neutral_frame_chosen"),
            candidate_ids=list(d.get("candidate_ids", [])),
            n_class_instances=int(d.get("n_class_instances", 0)),
            viewpoint=d.get("viewpoint", {}), notes=list(d.get("notes", [])))


def minimality_check(arms: Sequence["Arm"]) -> List[str]:
    """Confirm the arms differ *only* in the frame.

    The definition of a minimal pair, enforced rather than assumed. Two arms once
    differed in the viewpoint as well: "from the bed's point of view" also
    matched the bare landmark-viewpoint pattern, so that arm moved the observer
    to the bed. Both arms still named the right frame and still parsed to the
    right relation, anchor and target -- and the pair was still not minimal,
    because a system pinned to a single frame gave two different answers.
    """
    problems: List[str] = []
    sigs = {}
    for a in arms:
        q = parse(a.text)
        cons = q.target.constraints
        anchor = (cons[0].anchors[0].label if cons and cons[0].anchors
                  else None)
        sigs[a.frame] = {
            "relation": q.primary_relation,
            "target": q.target.label,
            "anchor": anchor,
            "viewpoint_hint": q.viewpoint_hint,
            "expects_multiple": q.expects_multiple,
            "ordinal": None if q.target.ordinal is None
            else q.target.ordinal.to_dict(),
        }
    frames = list(sigs)
    base = sigs[frames[0]]
    for f in frames[1:]:
        for k, v in base.items():
            if sigs[f][k] != v:
                problems.append(
                    f"arms differ in {k!r} as well as the frame: "
                    f"{frames[0]}={v!r} vs {f}={sigs[f][k]!r}")
    return problems


def parse_check(text: str, relation: str, anchor_class: str,
                target_class: str) -> List[str]:
    """Problems with how a filled template actually parses.

    Cue recovery is not enough. A phrasing can name its frame correctly and
    still lose the anchor -- "from the door's point of view, the cabinet in front
    of **it**" strips to a pronoun, and the resolver then answers against an
    unspecified anchor. This checks that the relation, the anchor class and the
    target class all survive parsing.
    """
    from ..categories import label_matches, normalize_label
    problems: List[str] = []
    q = parse(text)
    if q.primary_relation != relation:
        problems.append(f"relation parsed as {q.primary_relation!r}, "
                        f"expected {relation!r}")
    tl = q.target.label
    if tl is None or label_matches(target_class, tl) < 0.6:
        problems.append(f"target parsed as {tl!r}, expected {target_class!r}")
    cons = q.target.constraints
    if not cons or not cons[0].anchors:
        problems.append("no anchor survived parsing")
        return problems
    al = cons[0].anchors[0].label
    if al is None or label_matches(anchor_class, al) < 0.6:
        problems.append(f"anchor parsed as {al!r}, expected {anchor_class!r} "
                        f"(a pronoun or a dropped noun leaves the resolver "
                        f"with no anchor)")
    return problems


def validate_templates() -> List[str]:
    """Check every template both names its frame and keeps its anchor.

    Returns a list of problems; empty means the templates are usable. Run before
    generating anything: a template whose cue our own extractor misreads, or
    whose anchor does not survive parsing, silently produces pairs that are not
    minimal in the way that matters.
    """
    problems: List[str] = []
    for family, tpl in TEMPLATES.items():
        rel, d = ("left", "left") if family == "lateral" else ("front",
                                                              "in front of")
        for frame, text in tpl.items():
            filled = text.format(t="mug", a="laptop", d=d)
            kinds = {c.kind for c in extract_cues(filled)}
            if frame == "neutral":
                if kinds:
                    problems.append(
                        f"{family}/neutral carries a cue: {filled!r} -> "
                        f"{sorted(kinds)}")
            elif frame not in kinds:
                problems.append(
                    f"{family}/{frame}: {filled!r} -> cues {sorted(kinds)}, "
                    f"expected {frame}")
            for pb in parse_check(filled, rel, "laptop", "mug"):
                problems.append(f"{family}/{frame}: {pb}   [{filled!r}]")
        # every cued arm must be minimal against the neutral phrasing
        d2 = "left" if family == "lateral" else "in front of"
        neutral = Arm("neutral", tpl["neutral"].format(t="mug", a="laptop",
                                                        d=d2), -1, 0.0)
        for frame, text in tpl.items():
            if frame == "neutral":
                continue
            arm = Arm(frame, text.format(t="mug", a="laptop", d=d2), -1, 0.0)
            for pb in minimality_check([neutral, arm]):
                problems.append(f"{family}/{frame}: {pb}")
    return problems


def _class_counts(scene: Scene) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for o in scene.objects:
        out[o.canonical_label] = out.get(o.canonical_label, 0) + 1
    return out


def generate_for_scene(scene: Scene, cfg: Optional[RelationConfig] = None,
                       viewpoint: Optional[ViewpointSpec] = None,
                       frames: Sequence[str] = ("egocentric", "intrinsic"),
                       relations: Sequence[str] = ("left", "right",
                                                    "front", "behind"),
                       max_pairs: Optional[int] = None,
                       require_visible_together: bool = False,
                       frame_source=None,
                       verbose: bool = True) -> List[MinimalPair]:
    """All valid minimal pairs in one scene."""
    cfg = cfg or RelationConfig.load()
    viewpoint = viewpoint or ViewpointSpec()
    r = Resolver(scene, cfg)
    counts = _class_counts(scene)
    out: List[MinimalPair] = []
    rejected: Dict[str, int] = {}

    def reject(why: str):
        rejected[why] = rejected.get(why, 0) + 1

    # The anchor is named by *class* in the sentence, so a pair is only
    # well-defined when that class has exactly one instance. Otherwise the same
    # sentence gets generated once per instance and the resolver's joint anchor
    # scoring picks whichever instance works best -- so the recorded anchor_id is
    # meaningless and the pairs are duplicates. This is the single biggest cut in
    # yield and it is not optional.
    anchors = [o for o in scene.objects
               if not o.meta.get("suspect_instance")
               and o.front is not None            # an intrinsic arm needs a front
               and o.front_confidence >= 0.25
               and counts.get(o.canonical_label, 0) == 1]
    # The target class needs at least two instances, or "the mug to the left of
    # X" would be answerable from the class name alone and the frame could not
    # change the answer.
    target_classes = sorted({o.canonical_label for o in scene.objects
                             if not o.is_room_fixed
                             and not o.meta.get("unlabelled")
                             and not o.meta.get("suspect_instance")
                             and counts.get(o.canonical_label, 0) >= 2})

    for anchor in anchors:
        for cls in target_classes:
            if cls == anchor.canonical_label:
                continue
            for rel in relations:
                family, surface = SURFACE[rel]
                tpl = TEMPLATES[family]
                arms: List[Arm] = []
                ok = True
                for frame in frames:
                    if frame not in tpl:
                        ok = False
                        break
                    text = tpl[frame].format(t=cls, a=anchor.canonical_label,
                                             d=surface)
                    kinds = {c.kind for c in extract_cues(text)}
                    if frame not in kinds:
                        reject(f"cue not recovered for {frame}")
                        ok = False
                        break
                    pb = parse_check(text, rel, anchor.canonical_label, cls)
                    if pb:
                        reject(f"parse lost something: {pb[0][:48]}")
                        ok = False
                        break
                    res = r.resolve(parse(text), viewpoint, force_frame=frame,
                                    evaluate_alternative_frames=False)
                    if res.target is None or not res.candidates:
                        reject("no answer under a forced frame")
                        ok = False
                        break
                    score = float(res.candidates[0].score)
                    if score < MIN_ANSWER_SCORE:
                        reject("answer below the score threshold")
                        ok = False
                        break
                    if res.target.canonical_label != cls:
                        reject("answer is not an instance of the target class")
                        ok = False
                        break
                    arms.append(Arm(frame=frame, text=text,
                                    answer_id=res.target.id, score=score,
                                    cue_recovered=sorted(kinds)[0]))
                if not ok or len(arms) < 2:
                    continue
                mp = minimality_check(arms)
                if mp:
                    reject(f"not minimal: {mp[0][:44]}")
                    continue
                ids = [a.answer_id for a in arms]
                if len(set(ids)) < 2:
                    reject("the frames agree, so the pair is not minimal")
                    continue

                neutral_text = tpl["neutral"].format(
                    t=cls, a=anchor.canonical_label, d=surface)
                nres = r.resolve(parse(neutral_text), viewpoint,
                                 evaluate_alternative_frames=False)

                if require_visible_together and frame_source is not None:
                    from ..viz.overlay import best_joint_view
                    objs = [scene.by_id(i) for i in ids] + [anchor]
                    if best_joint_view(frame_source, [o.obb for o in objs],
                                       scene.up,
                                       scene_background=scene.background) < 0:
                        reject("no camera frame shows the anchor and both "
                               "answers")
                        continue

                out.append(MinimalPair(
                    id=f"{scene.scene_id}_{rel}_{anchor.id}_{cls.replace(' ', '-')}",
                    scene_id=scene.scene_id, relation=rel,
                    anchor_id=anchor.id, anchor_label=anchor.canonical_label,
                    target_class=cls, arms=arms, neutral_text=neutral_text,
                    neutral_answer_id=nres.target_id,
                    neutral_frame_chosen=nres.frame_used,
                    candidate_ids=sorted(set(ids)),
                    n_class_instances=counts.get(cls, 0),
                    viewpoint=viewpoint.to_dict(),
                    notes=[f"anchor front confidence "
                           f"{anchor.front_confidence:.2f} "
                           f"({anchor.front_method})"]))
                if max_pairs and len(out) >= max_pairs:
                    if verbose:
                        print(f"  {scene.scene_id}: {len(out)} pairs "
                              f"(capped)  rejected: {rejected}")
                    return out
    if verbose:
        print(f"  {scene.scene_id}: {len(out)} pairs   rejected: {rejected}")
    return out


def generate(scene_ids: Sequence[str], scene_for: Callable,
             cfg: Optional[RelationConfig] = None,
             frames: Sequence[str] = ("egocentric", "intrinsic"),
             max_per_scene: Optional[int] = None,
             verbose: bool = True, **kw) -> List[MinimalPair]:
    problems = validate_templates()
    if problems:
        raise RuntimeError("the phrasing templates do not read as intended:\n  "
                           + "\n  ".join(problems))
    out: List[MinimalPair] = []
    for sid in scene_ids:
        try:
            scene = scene_for(sid)
        except Exception as exc:
            if verbose:
                print(f"  [skip] {sid}: {exc}")
            continue
        out.extend(generate_for_scene(scene, cfg, frames=frames,
                                      max_pairs=max_per_scene,
                                      verbose=verbose, **kw))
    return out


def make_controls(pairs: Sequence[MinimalPair], scene_for: Callable,
                  cfg: Optional[RelationConfig] = None,
                  verbose: bool = True) -> List[ControlPair]:
    """One control pair per minimal pair, on the same scene and anchor.

    Built from the *neutral* phrasing so the expected answer is unambiguous, and
    validated the same way: both members must parse to the same relation, target
    and anchor, must carry no frame cue, and must resolve to the same object.
    """
    cfg = cfg or RelationConfig.load()
    resolvers: Dict[str, Resolver] = {}
    out: List[ControlPair] = []
    rejected: Dict[str, int] = {}
    for p in pairs:
        family = "lateral" if p.relation in ("left", "right") else "frontal"
        _, surface = SURFACE[p.relation]
        tpl = CONTROL_TEMPLATES[family]
        texts = [tpl[k].format(t=p.target_class, a=p.anchor_label, d=surface)
                 for k in ("a", "b")]
        try:
            scene = scene_for(p.scene_id)
        except Exception:
            continue
        if p.scene_id not in resolvers:
            resolvers[p.scene_id] = Resolver(scene, cfg)
        r = resolvers[p.scene_id]
        vp = ViewpointSpec()
        bad = False
        for t in texts:
            if extract_cues(t):
                rejected["control carries a frame cue"] = rejected.get(
                    "control carries a frame cue", 0) + 1
                bad = True
                break
            if parse_check(t, p.relation, p.anchor_label, p.target_class):
                rejected["control does not parse the same"] = rejected.get(
                    "control does not parse the same", 0) + 1
                bad = True
                break
        if bad:
            continue
        answers = [r.resolve(parse(t), vp,
                             evaluate_alternative_frames=False).target_id
                   for t in texts]
        if answers[0] is None or len(set(answers)) != 1:
            rejected["controls do not agree with each other"] = rejected.get(
                "controls do not agree with each other", 0) + 1
            continue
        out.append(ControlPair(
            id=p.id + "__control", scene_id=p.scene_id, relation=p.relation,
            anchor_id=p.anchor_id, anchor_label=p.anchor_label,
            target_class=p.target_class, texts=texts,
            expected_answer_id=int(answers[0]), viewpoint=p.viewpoint))
    if verbose:
        print(f"  {len(out)} control pairs   rejected: {rejected}")
    return out


def summarise(pairs: Sequence[MinimalPair]) -> Dict:
    from collections import Counter
    by_rel = Counter(p.relation for p in pairs)
    by_scene = Counter(p.scene_id for p in pairs)
    by_anchor_class = Counter(p.anchor_label for p in pairs)
    neutral_frames = Counter(p.neutral_frame_chosen for p in pairs
                             if p.neutral_frame_chosen)
    # does the neutral phrasing land on one of the cued answers?
    neutral_matches = Counter()
    for p in pairs:
        if p.neutral_answer_id is None:
            neutral_matches["no answer"] += 1
            continue
        hit = [a.frame for a in p.arms if a.answer_id == p.neutral_answer_id]
        neutral_matches[hit[0] if len(hit) == 1 else
                        ("several" if hit else "neither")] += 1
    return {
        "n_pairs": len(pairs),
        "by_relation": dict(by_rel),
        "by_scene": dict(by_scene),
        "by_anchor_class": dict(by_anchor_class.most_common(12)),
        "resolver_frame_on_neutral": dict(neutral_frames),
        "neutral_answer_matches": dict(neutral_matches),
        "distinct_anchors": len({(p.scene_id, p.anchor_id) for p in pairs}),
        "distinct_target_classes": len({p.target_class for p in pairs}),
    }


def write_jsonl(pairs: Sequence[MinimalPair], path: str):
    import json
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p.to_dict(), sort_keys=True) + "\n")


def read_jsonl(path: str) -> List[MinimalPair]:
    import json
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(MinimalPair.from_dict(json.loads(line)))
    return out

"""The resolver: a parsed query plus a scene, resolved to an object.

Shape of the thing:

1. **Candidates.** Every object whose class matches the target phrase, filtered
   by attributes. Room-shell objects are excluded unless the query names one.
2. **Anchors.** Each constraint's anchor is itself a referring expression, so it
   is resolved recursively. When several objects match an anchor equally well
   that is reported, not silently broken by an id ordering.
3. **Frames.** For each frame-dependent constraint the policy picks a frame --
   and the resolver *also* scores the query under every other plausible frame,
   because the difference between those answers is the output this project
   exists to produce.
4. **Relations.** Continuous scores, combined as a weighted geometric mean so a
   single hard failure cannot be averaged away by three easy passes.
5. **Ordinals** run last, over the candidates that survived the constraints,
   because "the second mug from the left on the middle shelf" counts only the
   mugs that are actually on that shelf.
6. **Ambiguity.** Reported alongside the answer, never instead of it.

The whole thing is pure geometry over cached numpy arrays: no model loading, no
GPU, and a query on a real ScanNet++ scene resolves in single-digit
milliseconds. That is deliberate -- the expensive part happened once, at build
time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..categories import is_room_fixed, label_matches, normalize_label
from ..frames.policy import (FrameDecision, ViewpointSpec, decide_frame,
                             relation_group, resolve_viewpoint)
from ..frames.reference_frame import ReferenceFrame
from ..geom.obb import OBB
from ..geom.support import (Level, assign_level, detect_levels, middle_level,
                            ordinal_level, shelf_levels)
from ..relations.base import RelationConfig, RelationScore, family, gmean, spec
from ..relations.comparative import (SIZE_ADJECTIVES, comparative_score,
                                     superlative_rank)
from ..relations.ordinal import (OrdinalResult, OrderingAxis, apply_ordinal,
                                 frame_ordering_axis, middle_candidate,
                                 support_ordering_axis)
from ..relations.projective import projective_score
from ..relations.proximity import (between_score, far_score, near_score,
                                   next_to_score)
from ..relations.vertical import (above_score, below_score, inside_score,
                                  on_score)
from ..scenegraph.objects import Object3D, Scene
from .ambiguity import (AmbiguityReport, WORLD_MARGIN_MIN,
                        check_frame_disagreement, score_tie)
from .schema import Constraint, LevelSpec, OrdinalSpec, Phrase, Query

#: Frames worth evaluating for disagreement: prior at least this fraction of
#: the best prior. Below that the reading is too marginal to call an ambiguity.
ALTERNATIVE_FRAME_PRIOR_RATIO = 0.30

#: Minimum class-match score for an object to be a candidate.
MIN_LABEL_MATCH = 0.55

#: How many equally-plausible anchor candidates to keep and score jointly with
#: the target. "The keyboard on the table" in a room with four tables means the
#: keyboard that is on *some* table; committing to one table first and scoring
#: against only that one gets it wrong whenever the greedy pick is not the right
#: table, which on real ScanNet++ office scenes is most of the time.
MAX_ANCHOR_CANDIDATES = 6

#: A top score below this means nothing in the scene really satisfies the
#: query. The best candidate is still returned -- an argmax is what a benchmark
#: scores -- but the answer is flagged, and a frame whose best candidate is this
#: weak is treated as having no answer when frames are compared. Otherwise
#: "the mug to the right of the laptop" reports a confident disagreement
#: between frames when in truth one frame has no mug on that side at all.
MIN_ANSWER_SCORE = 0.12

#: Colour names to approximate RGB, for attribute filtering.
COLOUR_RGB = {
    "red": (0.72, 0.15, 0.15), "green": (0.18, 0.55, 0.22),
    "blue": (0.18, 0.32, 0.72), "yellow": (0.88, 0.80, 0.20),
    "black": (0.10, 0.10, 0.10), "white": (0.92, 0.92, 0.92),
    "grey": (0.52, 0.52, 0.52), "gray": (0.52, 0.52, 0.52),
    "brown": (0.45, 0.30, 0.18), "orange": (0.90, 0.50, 0.15),
    "pink": (0.92, 0.60, 0.70), "purple": (0.50, 0.25, 0.60),
    "silver": (0.75, 0.75, 0.78), "beige": (0.85, 0.78, 0.65),
    "cream": (0.95, 0.92, 0.82), "gold": (0.80, 0.65, 0.20),
    "wooden": (0.55, 0.40, 0.25), "wood": (0.55, 0.40, 0.25),
}


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------

@dataclass
class Candidate:
    obj: Object3D
    score: float
    terms: Dict[str, float] = field(default_factory=dict)
    detail: Dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"object_id": self.obj.id, "label": self.obj.label,
                "score": float(self.score),
                "center": self.obj.center.tolist(),
                "terms": {k: float(v) for k, v in self.terms.items()},
                "detail": self.detail, "notes": list(self.notes)}


@dataclass
class ResolvedAnchor:
    phrase: Phrase
    obj: Optional[Object3D]
    alternatives: List[Candidate] = field(default_factory=list)
    levels: List[Level] = field(default_factory=list)
    level_index: Optional[int] = None
    level_z: Optional[float] = None
    level_ambiguous: bool = False
    notes: List[str] = field(default_factory=list)


@dataclass
class Resolution:
    query: Query
    scene_id: str
    target: Optional[Object3D]
    candidates: List[Candidate]
    anchors: List[ResolvedAnchor]
    frame_decision: Optional[FrameDecision]
    frame_answers: Dict[str, Optional[int]]
    ambiguity: AmbiguityReport
    ordinal: Optional[OrdinalResult] = None
    elapsed_ms: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def target_id(self) -> Optional[int]:
        return None if self.target is None else self.target.id

    @property
    def frame_used(self) -> Optional[str]:
        return self.frame_decision.chosen if self.frame_decision else None

    def to_dict(self) -> dict:
        return {
            "query": self.query.to_dict(),
            "scene_id": self.scene_id,
            "target_id": self.target_id,
            "target_label": None if self.target is None else self.target.label,
            "candidates": [c.to_dict() for c in self.candidates[:12]],
            "anchors": [{"phrase": a.phrase.to_dict(),
                         "object_id": None if a.obj is None else a.obj.id,
                         "label": None if a.obj is None else a.obj.label,
                         "level_index": a.level_index,
                         "level_z": a.level_z,
                         "level_ambiguous": a.level_ambiguous,
                         "n_levels": len(a.levels),
                         "alternatives": [c.to_dict() for c in a.alternatives[:4]],
                         "notes": a.notes} for a in self.anchors],
            "frame": None if self.frame_decision is None
            else self.frame_decision.to_dict(),
            "frame_answers": dict(self.frame_answers),
            "ambiguity": self.ambiguity.to_dict(),
            "ordinal": None if self.ordinal is None else self.ordinal.to_dict(),
            "elapsed_ms": self.elapsed_ms,
            "notes": list(self.notes),
        }

    def explain(self) -> str:
        """A short human-readable account of how the answer was reached."""
        lines = [f'query: "{self.query.text}"',
                 f"  parsed as: {self.query.describe()}"]
        for a in self.anchors:
            if a.obj is not None:
                extra = ""
                if a.level_index is not None:
                    extra = (f", level {a.level_index} of {len(a.levels)}"
                             f" at z={a.level_z:.2f}")
                lines.append(f"  anchor {a.phrase.label or a.phrase.text!r} -> "
                             f"{a.obj.short()}{extra}")
            else:
                lines.append(f"  anchor {a.phrase.label or a.phrase.text!r} -> "
                             f"NOT FOUND ({'; '.join(a.notes) or 'no match'})")
        if self.frame_decision is not None and self.frame_decision.chosen:
            fd = self.frame_decision
            lines.append(f"  frame: {fd.chosen} "
                         f"({'stated in the query' if fd.explicit else 'policy default'})"
                         f" | {fd.frame.describe()}")
            lines.append(f"    viewpoint: {fd.viewpoint.source}")
        if self.ordinal is not None:
            o = self.ordinal
            lines.append(f"  ordering: {o.axis.source} along "
                         f"({o.axis.direction[0]:+.2f}, {o.axis.direction[1]:+.2f}) "
                         f"spread {o.spread:.2f} m over {len(o.order)} candidates")
        if self.target is not None:
            lines.append(f"  ANSWER: {self.target.short()} "
                         f"(score {self.candidates[0].score:.3f})")
        else:
            lines.append("  ANSWER: none")
        if len(self.candidates) > 1:
            runner = self.candidates[1]
            lines.append(f"  runner-up: {runner.obj.short()} "
                         f"(score {runner.score:.3f})")
        if self.ambiguity.ambiguous:
            lines.append(f"  {self.ambiguity.summary()}")
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# resolver
# --------------------------------------------------------------------------

class Resolver:
    """Resolves queries against one scene. Cheap to construct, reusable."""

    def __init__(self, scene: Scene, cfg: Optional[RelationConfig] = None,
                 predicted: bool = False, text_encoder=None,
                 openvocab_threshold: float = 0.06):
        self.scene = scene
        self.cfg = cfg or RelationConfig.load()
        self.predicted = predicted
        #: Optional CLIP text encoder, for queries whose class is outside the
        #: vocabulary the scene was labelled with. Not required: in-vocabulary
        #: queries are answered from the stored per-object label distribution,
        #: so query time needs no torch even on an open-vocabulary scene.
        self.text_encoder = text_encoder
        self.openvocab_threshold = openvocab_threshold
        self._levels: Dict[int, List[Level]] = {}
        self._frame_scores: Dict[str, float] = {}
        self._has_embeddings = any(o.embedding is not None
                                   for o in scene.objects)
        self._text_cache: Dict[str, Optional[np.ndarray]] = {}

    # -- helpers ---------------------------------------------------------
    def levels_of(self, obj: Object3D) -> List[Level]:
        """Internal horizontal surfaces of an object, computed once per object."""
        if obj.id in self._levels:
            return self._levels[obj.id]
        lv: List[Level] = []
        if obj.points is not None and len(obj.points) > 200:
            raw = detect_levels(obj.points, obj.obb)
            lv = shelf_levels(raw, obj.obb)
        obj.levels = [l.z for l in lv]
        self._levels[obj.id] = lv
        return lv

    def _encloses(self, cand: Object3D, anchor: Object3D) -> Optional[str]:
        """Whether `cand` supports or contains `anchor`.

        A projective relation between an object and the thing holding it up is
        not a thing people say. "The object in front of the monitor" must not
        return the desk the monitor stands on, even though the desk's centroid
        genuinely lies on the monitor's front side -- it is simply the wrong
        kind of answer, and large supporting furniture wins these queries on
        overlap alone unless it is ruled out.
        """
        from ..geom.obb import horizontal_footprint_overlap
        from ..geom.support import support_score
        if support_score(anchor.obb, cand.obb, 0.12,
                         self.cfg.support_min_overlap) > 0.25:
            return "supports"
        if cand.obb.volume > 2.0 * anchor.obb.volume:
            inside = float(np.mean(cand.obb.contains(anchor.obb.corners(),
                                                    pad=0.02)))
            if inside > 0.75:
                return "contains"
        return None

    def _colour_score(self, obj: Object3D, attributes: Sequence[str]) -> float:
        """Soft agreement between an object's mean colour and a colour word.

        Mean vertex colour is a blunt instrument and this is scored softly for
        that reason: it narrows a field of four mugs, it does not decide a query
        on its own.
        """
        wanted = [a for a in attributes if a in COLOUR_RGB]
        if not wanted:
            return 1.0
        c = np.asarray(obj.color, float)[:3]
        best = 0.0
        for w in wanted:
            target = np.asarray(COLOUR_RGB[w], float)
            d = float(np.linalg.norm(c - target)) / np.sqrt(3.0)
            best = max(best, float(np.clip(1.0 - d / 0.55, 0.0, 1.0)))
        return max(best, 0.05)

    def _label_score(self, obj: Object3D, want: str) -> float:
        """How well an object matches a requested class.

        On a ground-truth scene this is a string match against the annotation.
        On an open-vocabulary scene the top-1 predicted label is noisy -- a
        geometric proposal of part of a cleaning trolley might come back as
        "pan" -- so the whole stored label distribution is consulted, and the
        query class counts if it appears anywhere in it with reasonable mass.
        That needs no model at query time, because the distribution was computed
        once at build time.
        """
        best = label_matches(want, obj.label)
        if best >= 1.0 or not obj.label_scores:
            return best
        for lab, prob in obj.label_scores:
            m = label_matches(want, lab)
            if m <= 0.0:
                continue
            # a lower-ranked label counts, discounted by its probability
            best = max(best, m * float(np.clip(prob / 0.35, 0.0, 1.0)))
        return best

    def _openvocab_score(self, want: str) -> Optional[np.ndarray]:
        """Cosine similarity of every object's embedding to a text query.

        Only used when the class is outside the scene's vocabulary and a text
        encoder was supplied. Returns None otherwise.
        """
        if not self._has_embeddings or self.text_encoder is None:
            return None
        if want in self._text_cache:
            t = self._text_cache[want]
        else:
            try:
                t = self.text_encoder.encode_texts([want])[0]
            except Exception:
                t = None
            self._text_cache[want] = t
        if t is None:
            return None
        sims = np.full(len(self.scene.objects), -1.0)
        for i, o in enumerate(self.scene.objects):
            if o.embedding is not None:
                sims[i] = float(np.dot(o.embedding, t))
        return sims

    def _class_candidates(self, phrase: Phrase) -> List[Candidate]:
        """Objects matching the phrase's class and attributes."""
        want = phrase.label
        out: List[Candidate] = []
        names_room_fixed = bool(want) and is_room_fixed(want)

        ov = None
        if want:
            # try the closed / stored-distribution path first, and only reach
            # for embeddings when nothing matches at all
            any_match = any(self._label_score(o, want) >= MIN_LABEL_MATCH
                            for o in self.scene.objects)
            if not any_match:
                ov = self._openvocab_score(want)

        for i, o in enumerate(self.scene.objects):
            if o.meta.get("unlabelled") and want and ov is None:
                continue
            if want is None:
                if o.is_room_fixed:
                    continue
                lab = 1.0
            else:
                lab = self._label_score(o, want)
                if ov is not None:
                    # rescale cosine similarity into a comparable score
                    sim = float(ov[i])
                    lab = max(lab, float(np.clip(
                        (sim - self.openvocab_threshold) / 0.20, 0.0, 1.0)))
                if lab < MIN_LABEL_MATCH:
                    continue
                if o.is_room_fixed and not names_room_fixed:
                    continue
            col = self._colour_score(o, phrase.attributes)
            out.append(Candidate(o, lab * col,
                                 {"label_match": lab, "colour": col}))
        out.sort(key=lambda c: (-c.score, c.obj.id))
        return out

    # -- anchors ---------------------------------------------------------
    def _resolve_anchor(self, phrase: Phrase, text: str,
                        viewpoint: Optional[ViewpointSpec],
                        depth: int = 0) -> List[ResolvedAnchor]:
        """Resolve an anchor phrase to a *ranked list* of possibilities.

        Returns one `ResolvedAnchor` per plausible object rather than a single
        winner. The caller scores the relation against each and keeps the best
        combination, which is what makes "the keyboard on the table" work in a
        room with four tables.
        """
        cands = self._class_candidates(phrase)
        if not cands:
            ra = ResolvedAnchor(phrase=phrase, obj=None)
            ra.notes.append(
                f"no object in the scene matches {phrase.label or phrase.text!r}")
            return [ra]

        # an anchor may itself carry constraints ("the shelf next to the door")
        if phrase.constraints and depth < 3:
            cands = self._apply_constraints(cands, phrase, text, viewpoint,
                                            depth + 1)[0]
        if phrase.superlative or phrase.size_word:
            cands = self._apply_size(cands, phrase)
        if phrase.ordinal is not None and len(cands) > 1:
            cands = self._apply_ordinal_to(cands, phrase, text, viewpoint,
                                           None, None)[0]
        if not cands:
            ra = ResolvedAnchor(phrase=phrase, obj=None)
            ra.notes.append("the anchor's own constraints excluded every match")
            return [ra]

        top = cands[0].score
        # keep everything scoring near the best, largest first among equals so
        # that "the door" prefers the doorway over the handle
        near_best = [c for c in cands if c.score >= top - 1e-6]
        near_best.sort(key=lambda c: -c.obj.obb.volume)
        rest = [c for c in cands if c.score < top - 1e-6]
        ordered = (near_best + rest)[:MAX_ANCHOR_CANDIDATES]

        out: List[ResolvedAnchor] = []
        for c in ordered:
            ra = ResolvedAnchor(phrase=phrase, obj=c.obj, alternatives=cands)
            if len(near_best) > 1:
                ra.notes.append(
                    f"{len(near_best)} objects match "
                    f"{phrase.label or phrase.text!r} equally well; all were "
                    f"scored and the best combination kept")
            if phrase.level is not None:
                ra.levels = self.levels_of(c.obj)
                if not ra.levels:
                    ra.notes.append(
                        f"no distinct horizontal surfaces were found inside the "
                        f"{c.obj.label}, so {phrase.level.word!r} cannot be "
                        f"resolved to a shelf")
                else:
                    picked, amb = self._pick_level(ra.levels, phrase.level)
                    ra.level_ambiguous = amb
                    if picked:
                        ra.level_index = picked[0].index
                        ra.level_z = picked[0].z
                    if amb and len(picked) > 1:
                        ra.notes.append(
                            f"the {c.obj.label} has {len(ra.levels)} shelves, "
                            f"so {phrase.level.word!r} has two equally good "
                            f"referents (z={picked[0].z:.2f} and "
                            f"z={picked[1].z:.2f})")
            out.append(ra)
        return out

    @staticmethod
    def _pick_level(levels: List[Level], spec_: LevelSpec):
        if spec_.middle or spec_.word in ("middle", "centre", "center"):
            return middle_level(levels)
        if spec_.word in ("top", "bottom"):
            return ordinal_level(levels, spec_.word)
        if spec_.index is not None:
            seq = levels if spec_.from_bottom else list(reversed(levels))
            i = spec_.index
            if 0 <= i < len(seq):
                return [seq[i]], False
            if -len(seq) <= i < 0:
                return [seq[i]], False
            return [], False
        return [], False

    # -- constraints -----------------------------------------------------
    def _score_constraint(self, cand: Object3D, constraint: Constraint,
                          anchors: List[ResolvedAnchor],
                          frame: Optional[ReferenceFrame]) -> RelationScore:
        rel = constraint.relation
        a0 = anchors[0].obj if anchors and anchors[0].obj is not None else None
        if a0 is None:
            return RelationScore(0.0, {}, ["anchor unresolved"])
        if cand.id == a0.id:
            return RelationScore(0.0, {}, ["target and anchor are the same object"])

        fam = family(rel)
        if fam == "projective":
            blocked = self._encloses(cand, a0)
            if blocked:
                return RelationScore(
                    0.0, {},
                    [f"the {cand.label} {blocked} the {a0.label}, so it is not "
                     f"{rel} of it"])
            if frame is None or not frame.available:
                reason = ("no reference frame" if frame is None
                          else frame.reason)
                return RelationScore(0.0, {}, [f"cannot evaluate {rel!r}: {reason}"])
            return projective_score(cand.obb, a0.obb, frame, rel, self.cfg)
        if rel == "on":
            lv = anchors[0].levels or self.levels_of(a0)
            if anchors[0].level_z is not None:
                from ..geom.support import support_score_on_level
                v = support_score_on_level(cand.obb, a0.obb, anchors[0].level_z,
                                           0.10, self.cfg.support_min_overlap * 0.85)
                return RelationScore(v, {"level_z": anchors[0].level_z,
                                         "support_on_level": v})
            return on_score(cand.obb, a0.obb, self.cfg, lv, self.predicted)
        if rel == "above":
            return above_score(cand.obb, a0.obb, self.cfg)
        if rel == "below":
            return below_score(cand.obb, a0.obb, self.cfg)
        if rel == "inside":
            return inside_score(cand.obb, a0.obb, self.cfg, cand.points)
        if rel == "near":
            return near_score(cand.obb, a0.obb, self.cfg, cand.points, a0.points)
        if rel == "next_to":
            return next_to_score(cand.obb, a0.obb, self.cfg, cand.points, a0.points)
        if rel == "far_from":
            return far_score(cand.obb, a0.obb, self.cfg, cand.points, a0.points,
                             self.scene.scale())
        if rel == "between":
            if len(anchors) < 2 or anchors[1].obj is None:
                return RelationScore(0.0, {}, ["'between' needs two anchors"])
            return between_score(cand.obb, a0.obb, anchors[1].obj.obb, self.cfg)
        if fam == "comparative":
            return comparative_score(cand.obb, a0.obb, rel, self.cfg)
        return RelationScore(0.0, {}, [f"relation {rel!r} is not implemented"])

    def _apply_constraints(self, cands: List[Candidate], phrase: Phrase,
                           text: str, viewpoint: Optional[ViewpointSpec],
                           depth: int = 0,
                           force_frame: Optional[str] = None,
                           collect: Optional[dict] = None):
        """Score every candidate against every constraint, jointly over anchors.

        For each constraint and each target candidate, the relation is evaluated
        against every plausible anchor and the best score kept, along with which
        anchor produced it. The anchor reported back is the one that won for the
        eventual answer, not a guess made before any scoring happened.
        """
        chosen_anchors: List[ResolvedAnchor] = []
        decisions: List[Optional[FrameDecision]] = []

        for c in phrase.constraints:
            anchor_options: List[List[ResolvedAnchor]] = [
                self._resolve_anchor(a, text, viewpoint, depth)
                for a in c.anchors]
            primary = anchor_options[0] if anchor_options else [
                ResolvedAnchor(Phrase(), None)]
            secondary = (anchor_options[1] if len(anchor_options) > 1
                         else [ResolvedAnchor(Phrase(), None)])

            # One frame decision per anchor option, but a single shared
            # viewpoint. Letting each candidate anchor resolve its own
            # `best_view` moved the viewer between candidates, so the frame that
            # scored the winner was not the frame that got reported -- the
            # explanation named a keyboard 1.7 m from the one actually used. The
            # viewpoint is a property of the query, not of each candidate, so it
            # is resolved once from the best-ranked anchor and reused.
            shared_vp = viewpoint
            if relation_group(c.relation) is not None and primary and \
                    primary[0].obj is not None:
                rv = resolve_viewpoint(viewpoint or ViewpointSpec(), self.scene,
                                       primary[0].obj.center)
                if rv.ok and rv.eye is not None:
                    shared_vp = ViewpointSpec(
                        mode="position", position=rv.eye,
                        look_at=(primary[0].obj.center if rv.look_dir is None
                                 else rv.eye + rv.look_dir))
                    shared_vp_source = rv.source
                else:
                    shared_vp_source = rv.reason
            else:
                shared_vp_source = ""

            frames: Dict[int, Optional[FrameDecision]] = {}
            if relation_group(c.relation) is not None:
                for ra in primary:
                    if ra.obj is None:
                        continue
                    fd = decide_frame(c.relation, ra.obj, self.scene, text,
                                      shared_vp, force_frame)
                    if shared_vp_source:
                        fd.viewpoint.source = shared_vp_source
                    frames[ra.obj.id] = fd

            key_base = f"{c.relation}:{c.anchors[0].label if c.anchors else '?'}"
            best_per_target: Dict[int, Tuple[float, ResolvedAnchor,
                                             Optional[FrameDecision], dict]] = {}
            for cand in cands:
                best = (-1.0, primary[0], frames.get(
                    primary[0].obj.id if primary[0].obj else -1), {})
                for ra in primary:
                    for rb in secondary:
                        pair = [ra] + ([rb] if len(anchor_options) > 1 else [])
                        fd = frames.get(ra.obj.id) if ra.obj else None
                        sc = self._score_constraint(
                            cand.obj, c, pair,
                            fd.frame if fd is not None else None)
                        if sc.value > best[0]:
                            best = (sc.value, ra, fd, sc.to_dict())
                best_per_target[cand.obj.id] = best
                cand.terms[key_base] = max(0.0, best[0])
                cand.detail.setdefault("relations", {})[key_base] = best[3]
                if best[1].obj is not None:
                    cand.detail.setdefault("anchor_used", {})[key_base] = \
                        best[1].obj.id
                for n in best[3].get("notes", []):
                    if n not in cand.notes:
                        cand.notes.append(n)

            # superlative proximity is a ranking, not a threshold
            if c.superlative and c.relation in ("near", "far_from"):
                self._rank_proximity(cands, c, primary, key_base)

            decisions.append(frames)
            chosen_anchors.append((primary, best_per_target, key_base))

        kept = []
        for cand in cands:
            rel_terms = [v for k, v in cand.terms.items()
                         if k not in ("label_match", "colour")]
            base = cand.terms.get("label_match", 1.0) * cand.terms.get("colour", 1.0)
            if rel_terms:
                cand.score = float(gmean([max(t, 1e-6) for t in rel_terms])) * base
            else:
                cand.score = base
            kept.append(cand)
        kept.sort(key=lambda c: (-c.score, c.obj.id))

        # Report the anchor that won for the top candidate, and the frame that
        # belongs to *that* anchor -- not whichever one happened to be first.
        resolved: List[ResolvedAnchor] = []
        chosen_decisions: List[Optional[FrameDecision]] = []
        winner = kept[0] if kept else None
        for (primary, best_per_target, key_base), frames in zip(chosen_anchors,
                                                                decisions):
            if winner is not None and winner.obj.id in best_per_target:
                ra = best_per_target[winner.obj.id][1]
                resolved.append(ra)
                chosen_decisions.append(
                    frames.get(ra.obj.id) if ra.obj is not None else None)
            else:
                resolved.append(primary[0])
                chosen_decisions.append(
                    frames.get(primary[0].obj.id)
                    if primary[0].obj is not None else None)
        info = {"anchors": resolved, "decisions": chosen_decisions}
        if collect is not None:
            collect.update(info)
        return kept, info

    def _rank_proximity(self, cands: List[Candidate], c: Constraint,
                        primary: List[ResolvedAnchor], key: str) -> None:
        """Turn "nearest to X" into a ranking over the candidate set.

        "Near the door" is an absolute claim and is simply false at three metres;
        "nearest to the door" always has a winner. Scoring the superlative with
        the absolute threshold gave every candidate zero and made the answer
        arbitrary, which is how a real query on a real scene came back with a
        confident wrong trash can.
        """
        from ..relations.proximity import surface_gap
        gaps: Dict[int, float] = {}
        for cand in cands:
            best = float("inf")
            for ra in primary:
                if ra.obj is None or ra.obj.id == cand.obj.id:
                    continue
                g = surface_gap(cand.obb if hasattr(cand, "obb") else cand.obj.obb,
                                ra.obj.obb, cand.obj.points, ra.obj.points)
                best = min(best, g)
            gaps[cand.obj.id] = best
        finite = [g for g in gaps.values() if np.isfinite(g)]
        if not finite:
            return
        lo, hi = min(finite), max(finite)
        for cand in cands:
            g = gaps[cand.obj.id]
            if not np.isfinite(g):
                cand.terms[key] = 0.0
                continue
            if c.relation == "near":
                v = 1.0 if g <= lo + 1e-9 else float(
                    np.clip((lo + 0.05) / (g + 0.05), 0.0, 1.0))
            else:
                v = 1.0 if g >= hi - 1e-9 else float(
                    np.clip((g + 0.05) / (hi + 0.05), 0.0, 1.0))
            cand.terms[key] = v
            cand.detail.setdefault("relations", {}).setdefault(key, {})[
                "superlative_gap"] = g

    def _apply_size(self, cands: List[Candidate], phrase: Phrase) -> List[Candidate]:
        word = phrase.superlative or phrase.size_word
        if not word or not cands:
            return cands
        order, vals, metric, tie = superlative_rank([c.obj.obb for c in cands], word)
        if not vals:
            return cands
        ranked = [cands[i] for i in order]
        # a superlative selects; a bare size adjective only re-weights
        if phrase.superlative:
            top = ranked[0]
            top.notes.append(f"selected as the {word} by {metric}")
            if tie:
                top.notes.append(f"the two {word} candidates are within 6% on "
                                 f"{metric}")
            return [top] + ranked[1:]
        n = len(ranked)
        for rank, c in enumerate(ranked):
            c.score *= float(np.clip(1.0 - 0.5 * rank / max(n - 1, 1), 0.4, 1.0))
        ranked.sort(key=lambda c: (-c.score, c.obj.id))
        return ranked

    # -- ordinals --------------------------------------------------------
    def _ordering_axis(self, phrase: Phrase, text: str,
                       viewpoint: Optional[ViewpointSpec],
                       support: Optional[Object3D],
                       force_frame: Optional[str],
                       anchor_for_frame: Optional[Object3D]
                       ) -> Tuple[OrderingAxis, Optional[FrameDecision],
                                  Optional[np.ndarray]]:
        o = phrase.ordinal
        assert o is not None
        if o.from_landmark:
            from ..frames.policy import _find_landmark
            lm = _find_landmark(self.scene, o.from_landmark)
            if lm is None:
                axis = OrderingAxis(np.zeros(3), "landmark", 0.0, "",
                                    [f"no object matches the landmark "
                                     f"{o.from_landmark!r}"])
                return axis, None, None
            from ..relations.ordinal import landmark_ordering_axis
            axis = landmark_ordering_axis(lm.center, [], self.scene.up)
            axis.notes.append(f"counting outwards from the {lm.label}")
            return axis, None, lm.center

        frame_anchor = anchor_for_frame or support
        if frame_anchor is None:
            # no anchor: read the frame at the centroid of the candidates
            frame_anchor = None
        fd = decide_frame("ordinal",
                          frame_anchor if frame_anchor is not None
                          else _PseudoAnchor(self.scene),
                          self.scene, text, viewpoint, force_frame)
        if fd.frame is None:
            axis = OrderingAxis(np.array([1.0, 0.0, 0.0]), "frame_axis", 0.0, "",
                                fd.notes or ["no reference frame available"])
            return axis, fd, None
        if support is not None:
            axis = support_ordering_axis(support.obb, fd.frame, o.from_word)
        else:
            axis = frame_ordering_axis(fd.frame, o.from_word)
        return axis, fd, None

    def _apply_ordinal_to(self, cands: List[Candidate], phrase: Phrase,
                          text: str, viewpoint: Optional[ViewpointSpec],
                          support: Optional[Object3D],
                          anchor_for_frame: Optional[Object3D],
                          force_frame: Optional[str] = None):
        o = phrase.ordinal
        if o is None or not cands:
            return cands, None, None
        axis, fd, landmark = self._ordering_axis(phrase, text, viewpoint,
                                                 support, force_frame,
                                                 anchor_for_frame)
        boxes = [c.obj.obb for c in cands]
        if o.middle:
            res = middle_candidate(boxes, axis, self.cfg)
        else:
            res = apply_ordinal(boxes, axis, o.index, self.cfg, landmark)
        if not res.picked:
            return [], res, fd
        picked = [cands[i] for i in res.picked]
        for p in picked:
            p.notes.extend(n for n in res.notes if n not in p.notes)
        rest = [cands[i] for i in res.order if i not in res.picked]
        return picked + rest, res, fd

    # -- main entry ------------------------------------------------------
    def resolve(self, query: Query,
                viewpoint: Optional[ViewpointSpec] = None,
                force_frame: Optional[str] = None,
                evaluate_alternative_frames: bool = True) -> Resolution:
        t0 = time.perf_counter()
        amb = AmbiguityReport()
        notes: List[str] = []
        phrase = query.target

        if query.viewpoint_hint and (viewpoint is None
                                     or viewpoint.mode == "best_view"):
            viewpoint = ViewpointSpec(mode="landmark",
                                      landmark=query.viewpoint_hint)
            notes.append(f"viewpoint taken from the query: "
                         f"the {query.viewpoint_hint}")

        cands = self._class_candidates(phrase)
        if not cands:
            amb.add("no_candidate",
                    f"nothing in the scene matches "
                    f"{phrase.label or phrase.text or 'the target'!r}")
            return Resolution(query, self.scene.scene_id, None, [], [], None, {},
                              amb, None,
                              (time.perf_counter() - t0) * 1000.0, notes)

        info: dict = {}
        chosen_frame_kind = force_frame or query.frame_hint
        kept, info = self._apply_constraints(cands, phrase, query.text, viewpoint,
                                            0, force_frame, info)
        anchors: List[ResolvedAnchor] = list(info.get("anchors", []))
        decisions = [d for d in info.get("decisions", []) if d is not None]
        frame_decision = decisions[0] if decisions else None

        # drop candidates that flatly fail a constraint
        if phrase.constraints:
            survivors = [c for c in kept if c.score > 1e-3]
            if survivors:
                kept = survivors

        if phrase.superlative or phrase.size_word:
            kept = self._apply_size(kept, phrase)

        # the support object an ordinal should be counted along
        support = None
        anchor_for_frame = None
        for a, c in zip(anchors, phrase.constraints):
            if a.obj is None:
                continue
            if anchor_for_frame is None:
                anchor_for_frame = a.obj
            if c.relation in ("on", "inside", "above") and a.obj.is_support_surface:
                support = a.obj
                break

        ordinal_res = None
        if phrase.ordinal is not None:
            kept, ordinal_res, ord_fd = self._apply_ordinal_to(
                kept, phrase, query.text, viewpoint, support, anchor_for_frame,
                force_frame)
            if ord_fd is not None:
                frame_decision = frame_decision or ord_fd
            if ordinal_res is not None:
                if ordinal_res.degenerate:
                    amb.add("ordinal_degenerate",
                            "; ".join(n for n in ordinal_res.notes) or
                            "the candidates are not ordered along the "
                            "counting axis",
                            spread=ordinal_res.spread,
                            mean_width=ordinal_res.mean_width)
                if ordinal_res.fragile:
                    amb.add("ordinal_tie",
                            f"two candidates are only "
                            f"{ordinal_res.min_gap:.3f} m apart along the "
                            f"counting axis",
                            min_gap=ordinal_res.min_gap,
                            median_gap=ordinal_res.median_gap)
                if ordinal_res.out_of_range:
                    amb.add("no_candidate",
                            f"asked for item {ordinal_res.index} but only "
                            f"{len(ordinal_res.order)} candidates exist")
                if len(ordinal_res.picked) > 1:
                    amb.add("level_even",
                            f"'{phrase.ordinal.word}' has "
                            f"{len(ordinal_res.picked)} equally good referents")

        for a in anchors:
            if a.obj is None:
                amb.add("no_candidate", "; ".join(a.notes) or "an anchor "
                        "could not be resolved")
            if a.level_ambiguous:
                amb.add("level_even", "; ".join(a.notes))
            if len(a.alternatives) > 1 and a.obj is not None:
                top = a.alternatives[0].score
                second = a.alternatives[1].score
                if score_tie(top, second):
                    amb.add("anchor",
                            f"{len(a.alternatives)} objects match the anchor "
                            f"{a.phrase.label or a.phrase.text!r} about equally "
                            f"well",
                            candidates=[c.obj.id for c in a.alternatives[:4]])

        target = kept[0].obj if kept else None
        if kept and phrase.constraints and kept[0].score < MIN_ANSWER_SCORE:
            amb.add("weak_match",
                    f"the best candidate only scores {kept[0].score:.3f}; "
                    f"nothing in the scene really satisfies this query",
                    top_score=float(kept[0].score))
        if len(kept) >= 2 and score_tie(kept[0].score, kept[1].score) \
                and phrase.ordinal is None:
            amb.add("score_tie",
                    f"{kept[0].obj.label} #{kept[0].obj.id} and "
                    f"{kept[1].obj.label} #{kept[1].obj.id} score within "
                    f"{abs(kept[0].score - kept[1].score):.3f} of each other")

        # -- the frame comparison ---------------------------------------
        frame_answers: Dict[str, Optional[int]] = {}
        if evaluate_alternative_frames and frame_decision is not None \
                and frame_decision.chosen:
            self._frame_scores = {}
            frame_answers = self._answers_under_frames(query, viewpoint,
                                                       frame_decision)
            amb.detail["frame_scores"] = dict(self._frame_scores)
            check_frame_disagreement(frame_answers, frame_decision.chosen, amb,
                                     label_of=self._label_of)
            f = frame_decision.frame
            if f is not None and f.kind == "world":
                margin = float(f.provenance.get("forward_margin", 0.0))
                if margin < WORLD_MARGIN_MIN:
                    amb.add("world_undetermined",
                            f"the room-canonical frame was used but its "
                            f"forward direction is a near-tie "
                            f"(margin {margin:.3f} < {WORLD_MARGIN_MIN})",
                            margin=margin)
            for p in frame_decision.priors[:1]:
                fr = frame_decision.frames.get(p.kind)
                if fr is not None and not fr.available:
                    amb.add("frame_unavailable",
                            f"the preferred {p.kind} frame is unavailable: "
                            f"{fr.reason}")

        notes.extend(query.notes)
        if frame_decision is not None:
            notes.extend(frame_decision.notes)

        return Resolution(query, self.scene.scene_id, target, kept, anchors,
                          frame_decision, frame_answers, amb, ordinal_res,
                          (time.perf_counter() - t0) * 1000.0, notes)

    def _label_of(self, oid: int) -> str:
        o = self.scene.by_id(oid)
        return f"{o.label} #{oid}" if o is not None else f"#{oid}"

    def _answers_under_frames(self, query: Query,
                              viewpoint: Optional[ViewpointSpec],
                              decision: FrameDecision
                              ) -> Dict[str, Optional[int]]:
        """Re-resolve under every plausible frame, to see whether they agree.

        Only frames with a prior worth taking seriously are tried: an
        interpretation nobody would actually mean is not an ambiguity. Frames
        that cannot be built map to None.
        """
        best_prior = max((p.prior for p in decision.priors), default=0.0)
        out: Dict[str, Optional[int]] = {}
        for p in decision.priors:
            if best_prior > 0 and p.prior < ALTERNATIVE_FRAME_PRIOR_RATIO * best_prior:
                continue
            fr = decision.frames.get(p.kind)
            if fr is None or not fr.available:
                out[p.kind] = None
                continue
            # Recomputed for every frame including the chosen one, so the
            # comparison cannot be biased by one branch taking a different code
            # path from the others.
            sub = self.resolve(query, viewpoint, force_frame=p.kind,
                               evaluate_alternative_frames=False)
            top = sub.candidates[0].score if sub.candidates else 0.0
            out[p.kind] = sub.target_id if top >= MIN_ANSWER_SCORE else None
            self._frame_scores[p.kind] = float(top)
        return out


class _PseudoAnchor:
    """Stand-in anchor for an ordinal with no anchor at all, e.g. "the leftmost
    monitor". Frames still need a point to be read at; the room centre is the
    least arbitrary choice and is recorded as such."""

    def __init__(self, scene: Scene):
        c = (scene.room.bounds.center if scene.room is not None
             else np.zeros(3))
        self.obb = OBB(c, np.array([0.05, 0.05, 0.05]))
        self.label = "the room"
        self.front = None
        self.front_confidence = 0.0
        self.front_method = "n/a"
        self.has_intrinsic_front = False


def resolve_text(scene: Scene, text: str,
                 viewpoint: Optional[ViewpointSpec] = None,
                 cfg: Optional[RelationConfig] = None,
                 force_frame: Optional[str] = None,
                 parser: str = "rules") -> Resolution:
    """Convenience: parse and resolve in one call."""
    from .parser_rules import parse
    q = parse(text)
    return Resolver(scene, cfg).resolve(q, viewpoint, force_frame)

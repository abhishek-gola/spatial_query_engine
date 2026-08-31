"""Evaluation, and the counterfactual attribution of failures.

An accuracy number on spatial queries is close to uninterpretable on its own,
because at least four different things can produce a wrong answer:

* the sentence was misparsed,
* the object was not detected or was mislabelled,
* the *frame convention* was wrong -- the geometry was right and the answer was
  the correct object under a different, equally reasonable reading,
* the relation scoring itself ranked the wrong object.

This module separates them by re-running each failed item under counterfactual
conditions and seeing which one repairs it. That is the difference between
"71% accuracy on projective relations" and "of the projective failures, 43% are
repaired by forcing the frame the annotator meant, and only 12% by switching to
ground-truth perception" -- the second is a finding, the first is a leaderboard
entry.

Attribution order
-----------------

A failure is attributed to the **first** cause below that repairs it, and the
order is fixed and reported so the numbers cannot be reshuffled after the fact:

1. ``unresolvable``   -- no candidate at all (a missing anchor, usually)
2. ``parse``          -- repaired by the gold parse
3. ``perception``     -- repaired by ground-truth instances
4. ``frame_unavailable`` -- the annotated frame could not be constructed
5. ``frame_convention``  -- repaired by forcing the annotated frame
6. ``geometry``       -- repaired by no condition; the scoring ranked wrong
7. ``ambiguous_item`` -- the item has no single gold answer

Putting `parse` and `perception` *before* `frame_convention` is the conservative
choice: it attributes a failure to the frame only when nothing upstream explains
it, so the headline frame number is a lower bound rather than a flattering one.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..frames.policy import ViewpointSpec
from ..query.resolver import Resolution, Resolver
from ..query.schema import Query, is_frame_dependent_type
from ..relations.base import RelationConfig
from ..scenegraph.objects import Scene
from .schema import BenchItem

#: Conditions the evaluator can run. `frame` values beyond these are passed
#: through to the resolver as a forced frame kind.
PARSE_MODES = ("rules", "gold", "llm")
FRAME_MODES = ("policy", "oracle", "egocentric", "egocentric_bearing",
               "intrinsic", "addressee", "world")
PERCEPTION_MODES = ("gt", "openvocab")

ATTRIBUTION_ORDER = ("unresolvable", "parse", "perception", "frame_unavailable",
                     "frame_convention", "geometry", "ambiguous_item")

#: Ambiguity kinds scored separately. `frame` is the one this project claims;
#: the others are reported so a reader can see that a poor pooled number comes
#: from `anchor` and `score_tie`, which are properties of how many instances a
#: real room contains rather than of the frame resolver.
AMBIGUITY_KINDS_SCORED = ("frame", "anchor", "score_tie", "level_even",
                          "ordinal_degenerate", "ordinal_tie",
                          "world_undetermined", "weak_match", "no_candidate")


def _binary_scores(pairs: Sequence[Tuple[bool, bool]]) -> Dict:
    """precision / recall / F1 / confusion counts for (gold, predicted) pairs."""
    tp = sum(1 for g, p in pairs if g and p)
    fp = sum(1 for g, p in pairs if not g and p)
    fn = sum(1 for g, p in pairs if g and not p)
    tn = sum(1 for g, p in pairs if not g and not p)
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(pairs),
            "precision": prec, "recall": rec, "f1": f1}


@dataclass
class Outcome:
    item_id: str
    scene_id: str
    relation_type: Optional[str]
    gold_frame: str
    frame_stated: bool
    ambiguous_gold: bool
    difficulty: str

    predicted_id: Optional[int]
    gold_ids: List[int]
    correct: bool
    frame_used: Optional[str]
    flagged_ambiguous: bool
    ambiguity_kinds: List[str]
    gold_ambiguity_kind: str = "none"
    frame_answers: Dict[str, Optional[int]] = field(default_factory=dict)
    top_score: float = 0.0
    elapsed_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _correct(pred: Optional[int], gold: Sequence[int], ambiguous: bool,
             flagged: bool = False) -> bool:
    """Whether an outcome counts as correct.

    Three cases, and the third is the one that matters:

    * unambiguous item -- the prediction must be the single gold target;
    * ambiguous item *with* acceptable targets listed -- any of them counts,
      since there is no single right one;
    * ambiguous item with **no** acceptable answer ("the plant in the corner
      nearest the window", where there is no corner object) -- the correct
      behaviour is to *say* the query cannot be answered, so the item is scored
      on the ambiguity flag alone.

    Without that third case a resolver is penalised for the one behaviour we
    actually want from it, and the accuracy column silently punishes honesty.
    """
    if ambiguous and not gold:
        return bool(flagged)
    if pred is None:
        return not gold
    return pred in set(gold)


def run_condition(items: Sequence[BenchItem],
                  scene_for: Callable[[str], Scene],
                  parse_mode: str = "rules",
                  frame_mode: str = "policy",
                  cfg: Optional[RelationConfig] = None,
                  predicted_perception: bool = False,
                  parser_fn: Optional[Callable[[str], Query]] = None,
                  progress: bool = False) -> List[Outcome]:
    """Resolve every item under one condition."""
    from ..query.parser_rules import parse as rule_parse
    cfg = cfg or RelationConfig.load()
    resolvers: Dict[str, Resolver] = {}
    outcomes: List[Outcome] = []

    for n, it in enumerate(items):
        if progress and n % 25 == 0:
            print(f"  [{parse_mode}/{frame_mode}] {n}/{len(items)}", flush=True)
        try:
            scene = scene_for(it.scene_id)
        except Exception as exc:
            outcomes.append(Outcome(it.id, it.scene_id, it.relation_type,
                                    it.frame, it.frame_stated_in_text,
                                    it.ambiguous, it.difficulty, None,
                                    list(it.target_ids), False, None, False, [],
                                    gold_ambiguity_kind=it.ambiguity_kind,
                                    error=f"scene unavailable: {exc}"))
            continue
        if it.scene_id not in resolvers:
            resolvers[it.scene_id] = Resolver(scene, cfg,
                                              predicted=predicted_perception)
        res_engine = resolvers[it.scene_id]

        # -- the query --------------------------------------------------
        if parse_mode == "gold":
            q = it.gold_query()
            if q is None:
                q = rule_parse(it.text)
                q.notes.append("no gold parse for this item; fell back to rules")
        elif parse_mode == "llm":
            if parser_fn is None:
                raise ValueError("parse_mode 'llm' needs a parser_fn")
            q = parser_fn(it.text)
        else:
            q = rule_parse(it.text)

        # -- the frame --------------------------------------------------
        force = None
        if frame_mode == "oracle":
            if it.frame in ("unspecified", "any"):
                force = None
            else:
                force = it.frame
        elif frame_mode != "policy":
            force = frame_mode

        try:
            res = res_engine.resolve(q, it.viewpoint_spec(), force_frame=force)
        except Exception as exc:      # a crash is a failure, not an exception
            outcomes.append(Outcome(it.id, it.scene_id, it.relation_type,
                                    it.frame, it.frame_stated_in_text,
                                    it.ambiguous, it.difficulty, None,
                                    list(it.target_ids), False, None, False, [],
                                    gold_ambiguity_kind=it.ambiguity_kind,
                                    error=f"{type(exc).__name__}: {exc}"))
            continue

        outcomes.append(Outcome(
            item_id=it.id, scene_id=it.scene_id, relation_type=it.relation_type,
            gold_frame=it.frame, frame_stated=it.frame_stated_in_text,
            ambiguous_gold=it.ambiguous, difficulty=it.difficulty,
            predicted_id=res.target_id, gold_ids=list(it.target_ids),
            correct=_correct(res.target_id, it.target_ids, it.ambiguous,
                             res.ambiguity.ambiguous),
            frame_used=res.frame_used,
            flagged_ambiguous=res.ambiguity.ambiguous,
            ambiguity_kinds=list(res.ambiguity.kinds),
            gold_ambiguity_kind=it.ambiguity_kind,
            frame_answers=dict(res.frame_answers),
            top_score=(res.candidates[0].score if res.candidates else 0.0),
            elapsed_ms=res.elapsed_ms))
    return outcomes


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def _acc(oc: Sequence[Outcome]) -> Optional[float]:
    return (sum(1 for o in oc if o.correct) / len(oc)) if oc else None


def aggregate(outcomes: Sequence[Outcome]) -> Dict:
    """Accuracy overall and split by every axis the report needs."""
    by_type: Dict[str, List[Outcome]] = defaultdict(list)
    by_frame: Dict[str, List[Outcome]] = defaultdict(list)
    by_diff: Dict[str, List[Outcome]] = defaultdict(list)
    by_scene: Dict[str, List[Outcome]] = defaultdict(list)
    for o in outcomes:
        by_type[o.relation_type or "unknown"].append(o)
        by_frame[o.gold_frame].append(o)
        by_diff[o.difficulty].append(o)
        by_scene[o.scene_id].append(o)

    unamb = [o for o in outcomes if not o.ambiguous_gold]
    amb = [o for o in outcomes if o.ambiguous_gold]
    fdep = [o for o in outcomes if is_frame_dependent_type(o.relation_type)]
    findep = [o for o in outcomes if not is_frame_dependent_type(o.relation_type)]
    stated = [o for o in fdep if o.frame_stated]
    unstated = [o for o in fdep if not o.frame_stated]

    # Ambiguity detection, scored per kind and not pooled.
    #
    # Pooling is misleading here. On real scenes the `anchor` and `score_tie`
    # flags fire on the large majority of queries -- because a room genuinely
    # has five keyboards and four tables -- while the `frame` flag, the one this
    # project actually claims, fires on far fewer. A single pooled
    # precision/recall is then dominated by the two kinds that have nothing to
    # do with the contribution, and reads as a system that cries wolf.
    # `amb_metrics`, not `amb`: there is already a local `amb` holding the
    # ambiguous outcomes, and shadowing it silently broke
    # accuracy_ambiguous_only.
    amb_metrics = {"pooled": _binary_scores(
        [(o.ambiguous_gold, o.flagged_ambiguous) for o in outcomes])}
    for kind in AMBIGUITY_KINDS_SCORED:
        pairs = [(o.ambiguous_gold and o.gold_ambiguity_kind == kind,
                  kind in o.ambiguity_kinds) for o in outcomes]
        amb_metrics[kind] = _binary_scores(pairs)
    # `frame` is the claimed one: does the system flag the queries the annotator
    # judged frame-ambiguous, among frame-dependent queries only?
    amb_metrics["frame_on_frame_dependent"] = _binary_scores(
        [(o.ambiguous_gold and o.gold_ambiguity_kind == "frame",
          "frame" in o.ambiguity_kinds)
         for o in outcomes if is_frame_dependent_type(o.relation_type)])
    prec = amb_metrics["pooled"]["precision"]
    rec = amb_metrics["pooled"]["recall"]
    f1 = amb_metrics["pooled"]["f1"]

    # how often the frames actually disagreed, and how often that mattered
    disagreed = 0
    disagree_and_wrong = 0
    for o in fdep:
        real = {v for v in o.frame_answers.values() if v is not None}
        if len(real) > 1:
            disagreed += 1
            if not o.correct:
                disagree_and_wrong += 1

    return {
        "n": len(outcomes),
        "accuracy": _acc(outcomes),
        "accuracy_unambiguous_only": _acc(unamb),
        "accuracy_ambiguous_only": _acc(amb),
        "accuracy_frame_dependent": _acc(fdep),
        "accuracy_frame_independent": _acc(findep),
        "accuracy_frame_stated": _acc(stated),
        "accuracy_frame_unstated": _acc(unstated),
        "n_frame_dependent": len(fdep),
        "n_frame_independent": len(findep),
        "by_relation_type": {k: {"n": len(v), "accuracy": _acc(v)}
                             for k, v in sorted(by_type.items())},
        "by_gold_frame": {k: {"n": len(v), "accuracy": _acc(v)}
                          for k, v in sorted(by_frame.items())},
        "by_difficulty": {k: {"n": len(v), "accuracy": _acc(v)}
                          for k, v in sorted(by_diff.items())},
        "by_scene": {k: {"n": len(v), "accuracy": _acc(v)}
                     for k, v in sorted(by_scene.items())},
        "frames_used": dict(Counter(o.frame_used for o in outcomes
                                    if o.frame_used)),
        "ambiguity_detection": amb_metrics["pooled"],
        "ambiguity_detection_by_kind": amb_metrics,
        "frame_disagreement": {
            "n_frame_dependent": len(fdep),
            "n_frames_disagreed": disagreed,
            "fraction_frames_disagreed": (disagreed / len(fdep)) if fdep else None,
            "n_disagreed_and_wrong": disagree_and_wrong},
        "errors": dict(Counter(o.error for o in outcomes if o.error)),
        "median_latency_ms": (float(np.median([o.elapsed_ms for o in outcomes]))
                              if outcomes else None),
    }


# --------------------------------------------------------------------------
# attribution
# --------------------------------------------------------------------------

def attribute(baseline: Sequence[Outcome],
              conditions: Dict[str, Sequence[Outcome]],
              frame_available: Optional[Dict[str, bool]] = None) -> Dict:
    """Attribute each baseline failure to the first condition that repairs it.

    `conditions` maps a condition name to its outcomes, keyed the same way as
    `baseline` by item id. Expected keys, any of which may be absent:

    * ``gold_parse``      -- baseline but with the annotated parse
    * ``gt_perception``   -- baseline but with ground-truth instances
    * ``oracle_frame``    -- baseline but forced into the annotated frame

    `frame_available` maps item id to whether the annotated frame could be
    constructed at all, which separates "we chose the wrong frame" from "the
    frame the annotator meant does not exist in our scene graph".
    """
    idx = {name: {o.item_id: o for o in oc} for name, oc in conditions.items()}
    counts: Counter = Counter()
    per_item: Dict[str, str] = {}
    per_type: Dict[str, Counter] = defaultdict(Counter)
    examples: Dict[str, List[str]] = defaultdict(list)

    for o in baseline:
        if o.correct:
            continue
        cause = None
        if o.ambiguous_gold:
            cause = "ambiguous_item"
        elif o.predicted_id is None and o.gold_ids:
            cause = "unresolvable"
        else:
            for cond, name in (("gold_parse", "parse"),
                               ("gt_perception", "perception")):
                other = idx.get(cond, {}).get(o.item_id)
                if other is not None and other.correct:
                    cause = name
                    break
            if cause is None:
                avail = (frame_available or {}).get(o.item_id, True)
                other = idx.get("oracle_frame", {}).get(o.item_id)
                if not avail:
                    cause = "frame_unavailable"
                elif other is not None and other.correct:
                    cause = "frame_convention"
                else:
                    cause = "geometry"
        counts[cause] += 1
        per_item[o.item_id] = cause
        per_type[o.relation_type or "unknown"][cause] += 1
        if len(examples[cause]) < 6:
            examples[cause].append(o.item_id)

    n_fail = sum(counts.values())
    frame_dep_fail = sum(
        c for t, cc in per_type.items() if is_frame_dependent_type(t)
        for c in [sum(cc.values())])
    frame_dep_frame_errors = sum(
        cc.get("frame_convention", 0) + cc.get("frame_unavailable", 0)
        for t, cc in per_type.items() if is_frame_dependent_type(t))

    return {
        "n_failures": n_fail,
        "counts": {k: counts.get(k, 0) for k in ATTRIBUTION_ORDER},
        "fractions": {k: (counts.get(k, 0) / n_fail) if n_fail else None
                      for k in ATTRIBUTION_ORDER},
        "attribution_order": list(ATTRIBUTION_ORDER),
        "by_relation_type": {t: dict(cc) for t, cc in sorted(per_type.items())},
        "per_item": per_item,
        "examples": {k: v for k, v in examples.items()},
        "headline": {
            "frame_dependent_failures": frame_dep_fail,
            "of_which_frame_errors": frame_dep_frame_errors,
            "fraction_of_frame_dependent_failures_that_are_frame_errors":
                (frame_dep_frame_errors / frame_dep_fail)
                if frame_dep_fail else None,
            "fraction_of_all_failures_that_are_frame_errors":
                ((counts.get("frame_convention", 0)
                  + counts.get("frame_unavailable", 0)) / n_fail)
                if n_fail else None,
            "fraction_of_all_failures_that_are_perception":
                (counts.get("perception", 0) / n_fail) if n_fail else None,
        },
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _pct(x: Optional[float]) -> str:
    return "  n/a" if x is None else f"{100.0 * x:5.1f}%"


def render_report(split: Dict, conditions: Dict[str, Dict],
                  attribution: Optional[Dict] = None,
                  title: str = "Spatial query benchmark") -> str:
    """A plain-text report. Also the thing pasted into a README."""
    L: List[str] = [f"# {title}", ""]
    L.append(f"{split['n_items']} queries over {split['n_scenes']} scenes.  "
             f"{split['n_ambiguous']} ({_pct(split['fraction_ambiguous']).strip()}) "
             f"marked ambiguous by the annotator.  "
             f"{split['n_frame_dependent']} are frame-dependent.")
    L.append("")
    L.append("## Composition")
    L.append("")
    L.append("| relation type | n |")
    L.append("|---|---|")
    for k, v in sorted(split["by_relation_type"].items(),
                       key=lambda kv: -kv[1]):
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("| annotated frame | n |")
    L.append("|---|---|")
    for k, v in sorted(split["by_frame"].items(), key=lambda kv: -kv[1]):
        L.append(f"| {k} | {v} |")
    L.append("")

    L.append("## Accuracy by condition")
    L.append("")
    L.append("| condition | n | overall | frame-dependent | frame-free | "
             "frame stated | frame unstated |")
    L.append("|---|---|---|---|---|---|---|")
    for name, m in conditions.items():
        L.append(f"| {name} | {m['n']} | {_pct(m['accuracy'])} | "
                 f"{_pct(m['accuracy_frame_dependent'])} | "
                 f"{_pct(m['accuracy_frame_independent'])} | "
                 f"{_pct(m['accuracy_frame_stated'])} | "
                 f"{_pct(m['accuracy_frame_unstated'])} |")
    L.append("")

    types = sorted({t for m in conditions.values()
                    for t in m["by_relation_type"]})
    L.append("## Accuracy by relation type")
    L.append("")
    L.append("| relation type | n | " + " | ".join(conditions) + " |")
    L.append("|---" * (2 + len(conditions)) + "|")
    for t in types:
        n = next((m["by_relation_type"][t]["n"] for m in conditions.values()
                  if t in m["by_relation_type"]), 0)
        cells = []
        for m in conditions.values():
            e = m["by_relation_type"].get(t)
            cells.append(_pct(e["accuracy"]) if e else "  n/a")
        L.append(f"| {t} | {n} | " + " | ".join(cells) + " |")
    L.append("")

    frames = sorted({f for m in conditions.values() for f in m["by_gold_frame"]})
    L.append("## Accuracy by annotated frame")
    L.append("")
    L.append("| annotated frame | n | " + " | ".join(conditions) + " |")
    L.append("|---" * (2 + len(conditions)) + "|")
    for fr in frames:
        n = next((m["by_gold_frame"][fr]["n"] for m in conditions.values()
                  if fr in m["by_gold_frame"]), 0)
        cells = []
        for m in conditions.values():
            e = m["by_gold_frame"].get(fr)
            cells.append(_pct(e["accuracy"]) if e else "  n/a")
        L.append(f"| {fr} | {n} | " + " | ".join(cells) + " |")
    L.append("")

    first = next(iter(conditions.values()))
    by_kind = first.get("ambiguity_detection_by_kind", {})
    L.append("## Ambiguity detection, per kind")
    L.append("")
    L.append("Scored against the annotator's `ambiguous` flag and "
             "`ambiguity_kind`, on the primary condition. **Reported per kind, "
             "not pooled.** On real scenes the `anchor` and `score_tie` flags "
             "fire on most queries, because a room genuinely does contain five "
             "keyboards and four tables; a pooled number is dominated by those "
             "and says nothing about the frame resolver. `frame` is the kind "
             "this project claims.")
    L.append("")
    L.append("| kind | n | gold positives | precision | recall | F1 |")
    L.append("|---|---|---|---|---|---|")
    order = ["frame", "frame_on_frame_dependent"] + [
        k for k in by_kind if k not in ("frame", "frame_on_frame_dependent",
                                        "pooled")] + ["pooled"]
    for k in order:
        v = by_kind.get(k)
        if not v:
            continue
        gold_pos = v["tp"] + v["fn"]
        name = f"**{k}**" if k.startswith("frame") else k
        L.append(f"| {name} | {v['n']} | {gold_pos} | "
                 f"{_pct(v['precision'])} | {_pct(v['recall'])} | "
                 f"{_pct(v['f1'])} |")
    L.append("")
    L.append("`n/a` means the annotator marked no query with that kind, so "
             "there is nothing to score. `pooled` is shown last and "
             "deliberately de-emphasised: it is the number a reader would "
             "otherwise quote, and it is the least informative one here.")
    L.append("")

    fd = first["frame_disagreement"]
    if fd["fraction_frames_disagreed"] is not None:
        L.append(f"On {fd['n_frame_dependent']} frame-dependent queries, the "
                 f"plausible reference frames picked different objects in "
                 f"{fd['n_frames_disagreed']} cases "
                 f"({_pct(fd['fraction_frames_disagreed']).strip()}).")
        L.append("")

    if attribution:
        L.append("## Failure attribution")
        L.append("")
        L.append("Each failure of the primary condition is attributed to the "
                 "first cause that repairs it, in this fixed order: "
                 + " -> ".join(attribution["attribution_order"]) + ".")
        L.append("")
        L.append("| cause | n | share of failures |")
        L.append("|---|---|---|")
        for k in attribution["attribution_order"]:
            n = attribution["counts"].get(k, 0)
            L.append(f"| {k} | {n} | {_pct(attribution['fractions'].get(k))} |")
        L.append("")
        h = attribution["headline"]
        fr = h["fraction_of_frame_dependent_failures_that_are_frame_errors"]
        if fr is not None:
            L.append(f"**Of the {h['frame_dependent_failures']} failures on "
                     f"frame-dependent queries, "
                     f"{_pct(fr).strip()} are frame-convention errors rather "
                     f"than perception errors** "
                     f"(perception accounts for "
                     f"{_pct(h['fraction_of_all_failures_that_are_perception']).strip()} "
                     f"of all failures).")
            L.append("")
    return "\n".join(L)


def save_results(out_dir: str, split: Dict, conditions: Dict[str, Dict],
                 attribution: Optional[Dict],
                 outcomes: Optional[Dict[str, List[Outcome]]] = None,
                 report_text: str = "", meta: Optional[Dict] = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    payload = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "split": split, "conditions": conditions,
               "attribution": attribution, "meta": meta or {}}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(payload, f, indent=1, default=float)
    if report_text:
        with open(os.path.join(out_dir, "report.md"), "w") as f:
            f.write(report_text)
    if outcomes:
        with open(os.path.join(out_dir, "outcomes.jsonl"), "w") as f:
            for cond, oc in outcomes.items():
                for o in oc:
                    d = o.to_dict()
                    d["condition"] = cond
                    f.write(json.dumps(d, default=float) + "\n")

"""Frame sensitivity: how much the answer depends on the reference frame.

Measurable **without any annotation**, which makes it the one quantitative
result available before the benchmark is labelled. It answers a different
question from accuracy:

* *accuracy* asks "is the answer the one a person meant" -- needs gold labels;
* *sensitivity* asks "does the choice of frame change the answer at all" --
  needs only the scenes and the queries.

Sensitivity is the precondition for the accuracy claim. If forcing a different
frame never changed the answer, the frame would not matter and this project
would have nothing to say. If it changes the answer often, then any pipeline
that picks a frame silently is making a consequential choice it does not report,
and the size of "often" is the size of the problem.

Two numbers per query set:

``disagreement rate``
    fraction of frame-dependent queries where two plausible frames, both
    available and both confident, select different objects.
``flip rate against a fixed frame``
    fraction where forcing one fixed convention changes the answer away from the
    policy's. This is the more directly meaningful of the two, because a fixed
    convention is exactly what an unexamined pipeline has.

Both are lower bounds on how much the frame matters, because a frame that could
not be built at all (an anchor whose front we could not estimate) is counted as
agreement rather than as disagreement.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..query.parser_rules import parse
from ..query.resolver import Resolver
from ..query.schema import is_frame_dependent_type
from ..relations.base import RelationConfig
from .schema import BenchItem

FIXED_FRAMES = ("egocentric", "intrinsic", "addressee", "world")


@dataclass
class SensitivityRow:
    item_id: str
    scene_id: str
    text: str
    relation_type: Optional[str]
    policy_frame: Optional[str]
    policy_answer: Optional[int]
    frame_answers: Dict[str, Optional[int]] = field(default_factory=dict)
    forced: Dict[str, Optional[int]] = field(default_factory=dict)
    ambiguity_kinds: List[str] = field(default_factory=list)

    @property
    def frame_dependent(self) -> bool:
        return is_frame_dependent_type(self.relation_type)

    @property
    def disagreed(self) -> bool:
        real = {v for v in self.frame_answers.values() if v is not None}
        return len(real) > 1


def measure(items: Sequence[BenchItem], scene_for: Callable,
            cfg: Optional[RelationConfig] = None,
            frames: Sequence[str] = FIXED_FRAMES,
            progress: bool = False) -> List[SensitivityRow]:
    cfg = cfg or RelationConfig.load()
    resolvers: Dict[str, Resolver] = {}
    rows: List[SensitivityRow] = []
    for n, it in enumerate(items):
        if progress and n % 100 == 0:
            print(f"  {n}/{len(items)}", flush=True)
        try:
            scene = scene_for(it.scene_id)
        except Exception:
            continue
        if it.scene_id not in resolvers:
            resolvers[it.scene_id] = Resolver(scene, cfg)
        r = resolvers[it.scene_id]
        q = parse(it.text)
        vp = it.viewpoint_spec()
        try:
            res = r.resolve(q, vp)
        except Exception:
            continue
        row = SensitivityRow(
            item_id=it.id, scene_id=it.scene_id, text=it.text,
            relation_type=it.relation_type or q.relation_type(),
            policy_frame=res.frame_used, policy_answer=res.target_id,
            frame_answers=dict(res.frame_answers),
            ambiguity_kinds=list(res.ambiguity.kinds))
        if row.frame_dependent:
            for fk in frames:
                try:
                    sub = r.resolve(q, vp, force_frame=fk,
                                    evaluate_alternative_frames=False)
                except Exception:
                    row.forced[fk] = None
                    continue
                from ..query.resolver import MIN_ANSWER_SCORE
                top = sub.candidates[0].score if sub.candidates else 0.0
                row.forced[fk] = (sub.target_id if top >= MIN_ANSWER_SCORE
                                  else None)
        rows.append(row)
    return rows


def summarise(rows: Sequence[SensitivityRow],
              frames: Sequence[str] = FIXED_FRAMES) -> Dict:
    fdep = [r for r in rows if r.frame_dependent]
    out: Dict = {
        "n_queries": len(rows),
        "n_frame_dependent": len(fdep),
        "n_frame_independent": len(rows) - len(fdep),
        "policy_frames_used": dict(Counter(r.policy_frame for r in rows
                                           if r.policy_frame)),
    }
    dis = sum(1 for r in fdep if r.disagreed)
    out["disagreement"] = {
        "n": dis,
        "rate": (dis / len(fdep)) if fdep else None,
        "note": "two plausible frames, both available and confident, pick "
                "different objects",
    }

    flips: Dict[str, Dict] = {}
    for fk in frames:
        considered = [r for r in fdep
                      if r.policy_answer is not None and fk in r.forced]
        changed = [r for r in considered if r.forced[fk] != r.policy_answer]
        no_answer = [r for r in considered if r.forced[fk] is None]
        # Two flip rates. The plain one counts "no answer under this frame" as a
        # change, because None != an object id -- so a frame that simply cannot
        # be built inflates it. The second excludes those, and is the one to
        # read. An earlier version of this report claimed no-answer cases were
        # counted as unchanged, which was the opposite of what the code did.
        answered = [r for r in considered if r.forced[fk] is not None]
        changed_answered = [r for r in answered
                            if r.forced[fk] != r.policy_answer]
        flips[fk] = {
            "n_considered": len(considered),
            "n_changed": len(changed),
            "flip_rate": (len(changed) / len(considered)) if considered else None,
            "n_no_answer_under_this_frame": len(no_answer),
            "n_answered": len(answered),
            "flip_rate_excluding_no_answer": (
                len(changed_answered) / len(answered)) if answered else None,
        }
    out["flip_rate_vs_fixed_frame"] = flips

    by_type: Dict[str, Dict] = {}
    grouped: Dict[str, List[SensitivityRow]] = defaultdict(list)
    for r in fdep:
        grouped[r.relation_type or "unknown"].append(r)
    for t, rs in sorted(grouped.items()):
        worst = 0.0
        for fk in frames:
            cons = [r for r in rs if r.policy_answer is not None and fk in r.forced]
            ch = sum(1 for r in cons if r.forced[fk] != r.policy_answer)
            if cons:
                worst = max(worst, ch / len(cons))
        by_type[t] = {
            "n": len(rs),
            "disagreement_rate": sum(1 for r in rs if r.disagreed) / len(rs),
            "worst_fixed_frame_flip_rate": worst,
        }
    out["by_relation_type"] = by_type

    out["ambiguity_kinds"] = dict(Counter(
        k for r in rows for k in r.ambiguity_kinds))
    out["n_flagged_ambiguous"] = sum(1 for r in rows if r.ambiguity_kinds)
    return out


def render(summary: Dict, title: str = "Frame sensitivity") -> str:
    """Report the policy-independent statistic first.

    Two numbers are available and they are not equally good:

    * **mutual disagreement** -- two plausible frames, both available and both
      confident, select different objects. This makes no reference to which
      frame is correct, so it stands on its own.
    * **flip rate against a fixed frame** -- forcing one convention changes the
      answer away from the policy's. This is *not* an error rate. It measures
      divergence from my choice, and if my policy is wrong a high flip rate only
      says "disagrees with me". It is reported second, and labelled.

    An earlier version of this report put the flip table first and largest,
    which invited exactly the conflation it should prevent.
    """
    def pct(x):
        return "  n/a" if x is None else f"{100.0 * x:5.1f}%"

    d = summary["disagreement"]
    L = [f"# {title}", ""]
    if summary.get("enriched") is not None:
        if summary["enriched"]:
            L.append("> **This item set is ENRICHED for frame sensitivity.** The "
                     "proposal generator sorts frame-sensitive candidates first "
                     "before capping, which makes annotation efficient and makes "
                     "any rate measured here a rate over *queries selected for "
                     "being frame-sensitive*, not over queries. Do not quote it "
                     "as a population rate; see the unenriched run.")
        else:
            L.append("> Item set sampled **without** frame-sensitivity "
                     "enrichment, so rates here are population rates over the "
                     "generator's candidate space.")
        L.append("")
    L.append("Measured without annotation. **This is not accuracy.** It is how "
             "much the answer depends on which reference frame is used, which "
             "is the precondition for any accuracy claim: if the frame never "
             "changed the answer, the frame would not matter.")
    L.append("")
    L.append(f"{summary['n_queries']} queries, "
             f"{summary['n_frame_dependent']} of them frame-dependent "
             f"({summary['n_frame_independent']} frame-free).")
    L.append("")
    L.append("## Headline: frames disagree with each other")
    L.append("")
    L.append(f"### {pct(d['rate']).strip()} of frame-dependent queries "
             f"({d['n']} of {summary['n_frame_dependent']})")
    L.append("")
    L.append("Two plausible reference frames, both constructible and both "
             "returning a confident answer, select **different objects**. This "
             "is the statistic to quote: it says nothing about which frame is "
             "right, so it does not depend on my policy being correct. It is a "
             "lower bound -- a frame that could not be built at all, because an "
             "anchor's front was not estimable, counts as agreement here rather "
             "than as disagreement.")
    L.append("")
    L.append("| relation type | n | frames disagree |")
    L.append("|---|---|---|")
    for t, v in summary["by_relation_type"].items():
        L.append(f"| {t} | {v['n']} | {pct(v['disagreement_rate'])} |")
    L.append("")

    L.append("## Secondary, and weaker: flip rate against a fixed convention")
    L.append("")
    L.append("How often forcing a single fixed frame -- what a pipeline that "
             "never names its convention effectively has -- picks a different "
             "object from the policy.")
    L.append("")
    L.append("**This is not an error rate.** It measures divergence from *my* "
             "policy's answer, and the policy has not been validated against "
             "human labels yet. If the policy is wrong, a high flip rate means "
             "only \"disagrees with my choice\". Treat it as an upper bound on "
             "how much the frame choice could matter, not as a measurement of "
             "how much anyone gets wrong.")
    L.append("")
    L.append("| forced frame | queries | no answer under it | flip rate "
             "(counting no-answer as a change) | flip rate (answered only) |")
    L.append("|---|---|---|---|---|")
    for fk, v in summary["flip_rate_vs_fixed_frame"].items():
        L.append(f"| {fk} | {v['n_considered']} | "
                 f"{v['n_no_answer_under_this_frame']} | "
                 f"{pct(v['flip_rate'])} | "
                 f"{pct(v.get('flip_rate_excluding_no_answer'))} |")
    L.append("")
    L.append("Read the last column. The middle one counts a query with **no "
             "answer at all** under the forced frame as a change, so a frame "
             "that simply cannot be constructed -- usually because the anchor's "
             "front is not estimable -- inflates it. That is a third reason the "
             "flip rate is the weaker statistic.")
    L.append("")

    L.append("## Frames the policy chose")
    L.append("")
    for k, v in sorted(summary["policy_frames_used"].items(),
                       key=lambda kv: -kv[1]):
        L.append(f"* {k}: {v}")
    L.append("")
    L.append("## Ambiguity flags raised, by kind")
    L.append("")
    L.append(f"{summary['n_flagged_ambiguous']} of {summary['n_queries']} "
             f"queries were flagged. These are the system's own flags, not "
             f"annotator judgements. Note the composition: `anchor` and "
             f"`score_tie` dominate because a real room contains several "
             f"instances of most classes, and they are not what this project "
             f"claims. The benchmark scores each kind separately for exactly "
             f"this reason.")
    L.append("")
    L.append("| kind | n flagged |")
    L.append("|---|---|")
    for k, v in sorted(summary["ambiguity_kinds"].items(),
                       key=lambda kv: -kv[1]):
        L.append(f"| {k} | {v} |")
    return "\n".join(L)

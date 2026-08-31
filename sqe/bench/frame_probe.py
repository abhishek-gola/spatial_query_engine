"""Can a system be *instructed* into a reference frame?

Run a system over minimal pairs -- two sentences differing only in an explicit
marker of whose left is meant -- and classify what it does:

``switched_correctly``
    Answered each arm with that arm's gold object. The system represents the
    frame and responds to being told which one to use.
``frame_blind``
    Gave the **same** answer to both arms. Not a geometry mistake: the system has
    no representation of the frame to instruct, and is unresponsive to an
    explicit instruction to change it. This is the finding worth reporting.
``switched_incorrectly``
    Answered differently but matched neither gold. It noticed the sentences
    differ without recovering what the difference means.
``partial``
    One arm right, one wrong.

Two built-in controls, and they matter more than they look. A metric like this is
easy to get wrong, so it is calibrated against systems whose behaviour is known
in advance:

* ``resolver`` -- this repo's resolver, following the cue. Must come out
  `switched_correctly` on essentially everything, because the pairs were
  constructed by forcing its own frames. That is a **circularity check, not a
  result**: it verifies the stimulus is well-formed and the scoring works. It is
  labelled as such everywhere and must never be quoted as a capability claim.
* ``pinned:<frame>`` -- the same resolver with the frame forced, ignoring the
  cue. Must come out `frame_blind` on 100%. This is the positive control for the
  `frame_blind` label: if a system that provably cannot switch is not scored
  frame-blind, the metric is broken.

Any external model is then measured on the same scale, against those two poles.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from ..frames.policy import ViewpointSpec
from ..query.parser_rules import parse
from ..query.resolver import Resolver
from ..relations.base import RelationConfig
from .minimal_pairs import MinimalPair

OUTCOMES = ("switched_correctly", "switched_incorrectly", "frame_blind",
            "partial", "no_answer", "error")


class Probe(Protocol):
    """Anything that answers "which object id does this sentence refer to"."""

    name: str

    def answer(self, scene, text: str, pair: MinimalPair) -> Optional[int]:
        ...


@dataclass
class ResolverProbe:
    """This repo's resolver. `pinned` forces one frame and ignores the cue."""
    scene_for: Callable
    cfg: Optional[RelationConfig] = None
    pinned: Optional[str] = None
    _resolvers: Dict[str, Resolver] = field(default_factory=dict, repr=False)

    @property
    def name(self) -> str:
        return f"pinned:{self.pinned}" if self.pinned else "resolver (cue-following)"

    @property
    def is_control(self) -> bool:
        return True

    def answer(self, scene, text: str, pair: MinimalPair) -> Optional[int]:
        if scene.scene_id not in self._resolvers:
            self._resolvers[scene.scene_id] = Resolver(
                scene, self.cfg or RelationConfig.load())
        r = self._resolvers[scene.scene_id]
        vp = ViewpointSpec(**{k: v for k, v in (pair.viewpoint or {}).items()
                              if k in ("mode", "index", "landmark")}) \
            if pair.viewpoint else ViewpointSpec()
        res = r.resolve(parse(text), vp, force_frame=self.pinned,
                        evaluate_alternative_frames=False)
        return res.target_id


@dataclass
class LLMProbe:
    """An LLM given the scene graph as text. Needs ANTHROPIC_API_KEY."""
    model: str = "claude-sonnet-5"
    cache_path: Optional[str] = None
    shuffle_ids: bool = True
    seed: int = 0
    _runner: object = None

    @property
    def name(self) -> str:
        return self.model

    @property
    def is_control(self) -> bool:
        return False

    def _get_runner(self):
        if self._runner is None:
            from .vlm_baseline import Runner
            self._runner = Runner(self.model, cache_path=self.cache_path)
        return self._runner

    def answer(self, scene, text: str, pair: MinimalPair) -> Optional[int]:
        import random

        from .vlm_baseline import _fmt_scene, parse_reply
        rng = random.Random(f"{self.seed}:{pair.id}")
        order = [o.id for o in scene.objects
                 if not o.is_room_fixed
                 or o.canonical_label in ("door", "window", "whiteboard",
                                          "heater", "sink", "toilet")]
        if self.shuffle_ids:
            rng.shuffle(order)
        vp = pair.viewpoint or {}
        eye = np.asarray(vp.get("position") or [0.0, 0.0, 1.6], float)
        look = np.asarray(vp.get("look_at") or [0.0, 1.0, 0.0], float)
        prompt = (_fmt_scene(scene, eye, look, order)
                  + f"\n\nWhich object does this refer to?\n\"{text}\"")
        reply = self._get_runner().ask(prompt)
        oid, _ = parse_reply(reply.get("raw", ""))
        return oid


@dataclass
class PairResult:
    pair_id: str
    scene_id: str
    relation: str
    answers: Dict[str, Optional[int]]     # arm frame -> answer given
    gold: Dict[str, int]
    neutral_answer: Optional[int]
    neutral_implies: Optional[str]
    outcome: str
    error: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def classify_pair(given: Dict[str, Optional[int]], gold: Dict[str, int]) -> str:
    """Outcome for one pair. See the module docstring for the categories."""
    frames = [f for f in gold if f in given]
    vals = [given[f] for f in frames]
    if any(v is None for v in vals):
        return "no_answer"
    if len(set(vals)) == 1:
        return "frame_blind"
    n_right = sum(1 for f in frames if given[f] == gold[f])
    if n_right == len(frames):
        return "switched_correctly"
    if n_right == 0:
        return "switched_incorrectly"
    return "partial"


def run(pairs: Sequence[MinimalPair], scene_for: Callable, probe,
        verbose: bool = True) -> List[PairResult]:
    out: List[PairResult] = []
    for n, p in enumerate(pairs):
        try:
            scene = scene_for(p.scene_id)
        except Exception as exc:
            out.append(PairResult(p.id, p.scene_id, p.relation, {}, p.answers,
                                  None, None, "error", str(exc)))
            continue
        given: Dict[str, Optional[int]] = {}
        err = ""
        for arm in p.arms:
            try:
                given[arm.frame] = probe.answer(scene, arm.text, p)
            except Exception as exc:
                given[arm.frame] = None
                err = f"{type(exc).__name__}: {exc}"
        neutral = None
        try:
            neutral = probe.answer(scene, p.neutral_text, p)
        except Exception as exc:
            err = err or f"{type(exc).__name__}: {exc}"
        implies = None
        if neutral is not None:
            hits = [f for f, v in p.answers.items() if v == neutral]
            implies = hits[0] if len(hits) == 1 else (
                "several" if hits else "neither")
        outcome = "error" if err and not any(
            v is not None for v in given.values()) else classify_pair(
            given, p.answers)
        out.append(PairResult(p.id, p.scene_id, p.relation, given, p.answers,
                              neutral, implies, outcome, err))
        if verbose and (n + 1) % 10 == 0:
            print(f"  {probe.name}: {n + 1}/{len(pairs)}", flush=True)
    return out


@dataclass
class ControlResult:
    control_id: str
    answers: List[Optional[int]]
    expected: int
    stable: bool          # same answer to both paraphrases
    correct: bool         # and that answer is the expected one
    error: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def run_controls(controls, scene_for: Callable, probe,
                 verbose: bool = True) -> List[ControlResult]:
    """Score the negative control: equally-awkward, non-contrastive paraphrases.

    A system SHOULD answer these identically. If it does not, its answers are
    unstable to surface form, and a `frame_blind` result from the same system
    cannot be read as "has no frame" -- it could just as well be "cannot parse an
    unusual sentence". This is the check that turns the minimal-pair result from
    an artefact into a finding.
    """
    out: List[ControlResult] = []
    for n, c in enumerate(controls):
        try:
            scene = scene_for(c.scene_id)
        except Exception as exc:
            out.append(ControlResult(c.id, [], c.expected_answer_id, False,
                                     False, str(exc)))
            continue
        answers: List[Optional[int]] = []
        err = ""
        for t in c.texts:
            try:
                answers.append(probe.answer(scene, t, c))
            except Exception as exc:
                answers.append(None)
                err = f"{type(exc).__name__}: {exc}"
        stable = (len(set(answers)) == 1 and answers[0] is not None)
        out.append(ControlResult(c.id, answers, c.expected_answer_id, stable,
                                 stable and answers[0] == c.expected_answer_id,
                                 err))
        if verbose and (n + 1) % 10 == 0:
            print(f"  {probe.name} controls: {n + 1}/{len(controls)}", flush=True)
    return out


def summarise_controls(results: Sequence[ControlResult]) -> Dict:
    n = len(results)
    return {
        "n_controls": n,
        "n_stable": sum(1 for r in results if r.stable),
        "stable_rate": (sum(1 for r in results if r.stable) / n) if n else None,
        "n_correct": sum(1 for r in results if r.correct),
        "correct_rate": (sum(1 for r in results if r.correct) / n) if n else None,
        "n_errors": sum(1 for r in results if r.error),
    }


def summarise(results: Sequence[PairResult]) -> Dict:
    n = len(results)
    counts = Counter(r.outcome for r in results)
    by_rel: Dict[str, Dict[str, int]] = {}
    for r in results:
        by_rel.setdefault(r.relation, Counter())[r.outcome] += 1
    implies = Counter(r.neutral_implies for r in results
                      if r.neutral_implies)
    return {
        "n_pairs": n,
        "outcomes": {k: counts.get(k, 0) for k in OUTCOMES},
        "rates": {k: (counts.get(k, 0) / n) if n else None for k in OUTCOMES},
        "by_relation": {k: dict(v) for k, v in sorted(by_rel.items())},
        "neutral_implies": dict(implies),
        "n_errors": sum(1 for r in results if r.error),
    }


def render(summaries: Dict[str, Dict], title: str = "Frame instructability") -> str:
    """Table across systems. Controls are marked and explained."""
    def pct(x):
        return "  n/a" if x is None else f"{100.0 * x:5.1f}%"

    L = [f"# {title}", ""]
    L.append("Minimal pairs: two sentences differing only in an explicit marker "
             "of whose left is meant, on scenes where the two readings pick "
             "different objects. **frame_blind** means the system gave the same "
             "answer to both -- it has no frame to instruct.")
    L.append("")
    L.append("| system | pairs | switched correctly | frame blind | "
             "switched wrongly | partial | no answer | control stable |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, s in summaries.items():
        r = s["rates"]
        c = s.get("controls") or {}
        L.append(f"| {name} | {s['n_pairs']} | {pct(r['switched_correctly'])} | "
                 f"**{pct(r['frame_blind'])}** | "
                 f"{pct(r['switched_incorrectly'])} | {pct(r['partial'])} | "
                 f"{pct(r['no_answer'])} | {pct(c.get('stable_rate'))} |")
    L.append("")
    L.append("**`control stable` is load-bearing.** It is the fraction of "
             "*non-contrastive* paraphrase pairs -- equally awkward, matched for "
             "shape, differing in nothing that should change the answer -- that "
             "the system answers identically. A high `frame_blind` rate only "
             "means \"has no frame to instruct\" if `control stable` is also "
             "high. A system with low `control stable` is unstable to surface "
             "form, and its frame-blindness is unattributable.")
    L.append("")
    L.append("## Default convention on the uncued sentence")
    L.append("")
    L.append("With no marker, which arm's answer does the system give?")
    L.append("")
    keys = sorted({k for s in summaries.values() for k in s["neutral_implies"]})
    L.append("| system | " + " | ".join(keys) + " |")
    L.append("|---" * (1 + len(keys)) + "|")
    for name, s in summaries.items():
        L.append(f"| {name} | " + " | ".join(
            str(s["neutral_implies"].get(k, 0)) for k in keys) + " |")
    L.append("")
    L.append("## Reading the controls")
    L.append("")
    L.append("* **`resolver (cue-following)`** is a circularity check, not a "
             "result. The pairs were built by forcing this resolver's own "
             "frames, so it must score near 100% switched-correctly; that only "
             "confirms the stimulus is well-formed and the scoring works. It is "
             "not evidence that the resolver is right about anything.")
    L.append("* **`pinned:<frame>`** is the positive control for the "
             "`frame_blind` label: a system that provably cannot switch must "
             "score 100% frame-blind. If it does not, the metric is broken and "
             "no other row means anything.")
    return "\n".join(L)


def save(out_dir: str, summaries: Dict[str, Dict],
         results: Dict[str, List[PairResult]], text: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "frame_probe.md"), "w") as f:
        f.write(text)
    with open(os.path.join(out_dir, "frame_probe.json"), "w") as f:
        json.dump({"summaries": summaries,
                   "results": {k: [r.to_dict() for r in v]
                               for k, v in results.items()}},
                  f, indent=1, default=float)

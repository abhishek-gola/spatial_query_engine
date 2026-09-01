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

import hashlib
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


def _spec_from_dict(d: Optional[dict]) -> ViewpointSpec:
    """Rebuild the full spec, including an explicit position.

    Dropping `position` here would silently move the observer to the default and
    make every egocentric gold answer wrong -- a bug that looks like a result.
    """
    if not d:
        return ViewpointSpec()
    kw: Dict = {}
    for k in ("mode", "index", "landmark"):
        if d.get(k) is not None:
            kw[k] = d[k]
    for k in ("position", "look_at"):
        if d.get(k) is not None:
            kw[k] = np.asarray(d[k], float)
    return ViewpointSpec(**kw)


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
        vp = _spec_from_dict(pair.viewpoint)
        res = r.resolve(parse(text), vp, force_frame=self.pinned,
                        evaluate_alternative_frames=False)
        return res.target_id


def build_prompt(scene, text: str, pair, shuffle_ids: bool = True,
                 seed: int = 0) -> str:
    """The exact question put to a model, identical across probes.

    The object id order is shuffled per *pair*, not per call, so both arms of a
    minimal pair see the same list in the same order. Shuffling per call would
    let an answer change because the list changed -- exactly the confound the
    pair design exists to remove.
    """
    import random

    from .vlm_baseline import _fmt_scene

    rng = random.Random(f"{seed}:{pair.id}")
    order = [o.id for o in scene.objects if not o.is_room_fixed
             or o.canonical_label in ("door", "window", "whiteboard")]
    # The listing MUST contain the anchor and every candidate answer. The filter
    # above drops room-fixed objects, and some anchors are room-fixed -- a
    # heater, a wall clock -- so without this the sentence names an object that
    # is not in the list and the trial is unanswerable rather than hard. Found by
    # an answerer reporting four trials whose anchor it could not find.
    required = [getattr(pair, "anchor_id", None)]
    required += list(getattr(pair, "candidate_ids", []) or [])
    for oid in required:
        if oid is None or oid in order or scene.by_id(oid) is None:
            continue
        order.insert(rng.randrange(len(order) + 1), oid)
    if shuffle_ids:
        rng.shuffle(order)
    vp = pair.viewpoint or {}
    if vp.get("position") is None:
        raise ValueError(
            f"{pair.id}: the pair records viewpoint mode "
            f"{vp.get('mode')!r} with no concrete position, so a model cannot "
            f"be told where the observer stands and its egocentric arm is "
            f"unanswerable. Regenerate the pairs.")
    eye = np.asarray(vp["position"], float)
    # `look_at` is a point; the prompt states a direction.
    if vp.get("look_at") is not None:
        d = np.asarray(vp["look_at"], float) - eye
        n = float(np.linalg.norm(d))
        look = d / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
    else:
        look = np.array([0.0, 1.0, 0.0])
    return (_fmt_scene(scene, eye, look, order)
            + f"\n\nWhich object does this refer to?\n\"{text}\"")


def prompt_key(prompt: str) -> str:
    """Stable opaque handle for one trial. Carries no hint of the frame."""
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]


@dataclass
class LLMProbe:
    """A model given the scene graph as text. Any supported vendor.

    Object ids are shuffled per *pair* rather than per call, so both arms of a
    minimal pair see the identical object list. Shuffling per call would let a
    model's answer change because the list changed, which is precisely the
    confound the pair design exists to remove.
    """
    model: str = "anthropic:claude-sonnet-5"
    cache_path: Optional[str] = None
    shuffle_ids: bool = True
    seed: int = 0
    max_tokens: int = 300
    _client: object = None
    _cache: Dict[str, str] = field(default_factory=dict)
    _loaded: bool = False

    @property
    def name(self) -> str:
        from .vendors import split_name
        v, m = split_name(self.model)
        return f"{v}:{m}"

    @property
    def is_control(self) -> bool:
        return False

    def _get_client(self):
        if self._client is None:
            from .vendors import VendorClient
            self._client = VendorClient(self.model, max_tokens=self.max_tokens)
        return self._client

    def _load_cache(self):
        if self._loaded:
            return
        self._loaded = True
        if self.cache_path and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path) as f:
                    self._cache = json.load(f)
            except (OSError, json.JSONDecodeError):
                self._cache = {}

    def _save_cache(self):
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)) or ".",
                    exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f, indent=1)

    def answer(self, scene, text: str, pair) -> Optional[int]:
        from .vlm_baseline import SYSTEM, parse_reply
        self._load_cache()
        prompt = build_prompt(scene, text, pair, self.shuffle_ids, self.seed)
        key = f"{self.name}::{prompt}"
        if key in self._cache:
            oid, _ = parse_reply(self._cache[key])
            return oid
        raw = self._get_client().complete(SYSTEM, prompt)
        self._cache[key] = raw
        self._save_cache()
        oid, _ = parse_reply(raw)
        return oid


@dataclass
class FileProbe:
    """A system whose answers were collected out-of-band and read from a file.

    This exists so a model that has no API reachable from here -- or an agent, or
    a person -- can be measured on exactly the same scale as an API model. The
    trials are exported as opaque prompt/key pairs, answered elsewhere, and
    scored by the same `run()`/`classify_pair()` path. Nothing about the scoring
    knows or cares where the answers came from.

    A missing key is an error, not a `no_answer`: silently scoring an unanswered
    trial would let an incomplete run look like a finding.
    """
    label: str
    answers: Dict[str, Optional[int]]
    shuffle_ids: bool = True
    seed: int = 0

    @property
    def name(self) -> str:
        return self.label

    @property
    def is_control(self) -> bool:
        return False

    def answer(self, scene, text: str, pair) -> Optional[int]:
        key = prompt_key(build_prompt(scene, text, pair, self.shuffle_ids,
                                      self.seed))
        if key not in self.answers:
            raise KeyError(f"{self.label}: trial {key} was never answered "
                           f"({text!r})")
        return self.answers[key]

    @staticmethod
    def load(label: str, path: str, **kw) -> "FileProbe":
        with open(path) as f:
            d = json.load(f)
        rows = d.get("answers", d)
        out: Dict[str, Optional[int]] = {}
        for k, v in rows.items():
            if isinstance(v, dict):
                v = v.get("id")
            out[k] = None if v is None else int(v)
        return FileProbe(label, out, **kw)


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


def stable_only(pair_results: Sequence[PairResult],
                control_results: Sequence[ControlResult]) -> Dict:
    """Re-summarise over the pairs whose OWN control the system got consistent.

    Pooled control stability tells you a system is noisy; it does not tell you
    which rows to distrust. Every control is built from one pair -- same scene,
    same anchor, same neutral sentence, two equally awkward paraphrases -- so
    each pair has a matched, item-level check on whether that system's answers
    move for reasons unrelated to the frame.

    Restricting to the pairs that passed their own check is the honest reading of
    a `frame_blind` rate: on those items the system answers the same sentence the
    same way twice, so if it *also* answers two differently-framed sentences the
    same way, that is about the frame and not about surface form. It is a smaller
    n, and the n is reported alongside.
    """
    stable = {c.control_id for c in control_results if c.stable}
    kept = [r for r in pair_results
            if f"{r.pair_id}__control" in stable]
    out = summarise(kept)
    out["n_dropped"] = len(pair_results) - len(kept)
    out["basis"] = "pairs whose matched control pair was answered consistently"
    return out


def stratify_by_baseline(pair_results: Sequence[PairResult],
                         pairs: Sequence[MinimalPair]) -> Dict:
    """Re-summarise over the pairs where the system and this resolver agree on
    the **unmarked** sentence.

    This is the stratification that matters, and it demolishes the pooled
    `frame_blind` rate. `frame_blind` means "gave the same answer to both cued
    arms". But if the system already disagrees with me about the plain,
    uncued sentence, then its two identical answers may be a perfectly
    consistent reading of a sentence I resolve differently -- the disagreement is
    about the baseline, not about whether the frame cue landed. Pooling the two
    populations attributes baseline disagreement to frame-blindness.

    Conditioning on baseline agreement isolates the thing being claimed: for
    these items the system and I read the plain sentence the same way, so if a
    cue then fails to move it, that is about the cue.

    The subset is small, and the count is reported rather than a percentage for
    exactly that reason.
    """
    neutral = {p.id: p.neutral_answer_id for p in pairs}
    kept = [r for r in pair_results
            if r.pair_id in neutral
            and r.neutral_answer is not None
            and r.neutral_answer == neutral[r.pair_id]]
    out = summarise(kept)
    out["n_dropped"] = len(pair_results) - len(kept)
    out["basis"] = ("pairs where the system's answer to the unmarked sentence "
                    "matches this resolver's")
    return out


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
    if any("baseline_agree" in s for s in summaries.values()):
        L.append("## Conditioned on agreement about the *unmarked* sentence")
        L.append("")
        L.append("`frame_blind` means \"same answer to both cued arms\". If a "
                 "system already disagrees with this resolver about the plain, "
                 "uncued sentence, its two identical answers can be a "
                 "consistent reading of a sentence I resolve differently -- the "
                 "disagreement is about the baseline, not about whether the cue "
                 "landed. Pooling the two populations charges baseline "
                 "disagreement to frame-blindness. Counts, not percentages: the "
                 "subset is small.")
        L.append("")
        L.append("| system | pairs kept | dropped | switched correctly | "
                 "frame blind | switched wrongly | partial |")
        L.append("|---|---|---|---|---|---|---|")
        for name, s in summaries.items():
            ba = s.get("baseline_agree")
            if not ba:
                continue
            o = ba["outcomes"]
            L.append(f"| {name} | {ba['n_pairs']} | {ba['n_dropped']} | "
                     f"{o['switched_correctly']} | **{o['frame_blind']}** | "
                     f"{o['switched_incorrectly']} | {o['partial']} |")
        L.append("")
        L.append("**This is the row to read.** A pooled `frame_blind` rate over "
                 "all 35 pairs mixes in every item where the system and I simply "
                 "read the plain sentence differently, and there is no ground "
                 "truth yet saying which of us is right. On the conditioned "
                 "subset the cue mostly does land, and the honest summary is "
                 "that the pooled rate was not measuring what its name says.")
        L.append("")
    if any("stable_only" in s for s in summaries.values()):
        L.append("## The same table, restricted to control-matched pairs")
        L.append("")
        L.append("Each control is built from one pair, so every pair has an "
                 "item-level check on whether that system's answers move for "
                 "reasons unrelated to the frame. Restricting to the pairs that "
                 "passed their own check is the reading of `frame_blind` that "
                 "survives a noisy system: on these items the system answers "
                 "the same sentence the same way twice.")
        L.append("")
        L.append("| system | pairs kept | dropped | switched correctly | "
                 "frame blind | switched wrongly | partial |")
        L.append("|---|---|---|---|---|---|---|")
        for name, s in summaries.items():
            so = s.get("stable_only")
            if not so:
                continue
            r = so["rates"]
            L.append(f"| {name} | {so['n_pairs']} | {so['n_dropped']} | "
                     f"{pct(r['switched_correctly'])} | "
                     f"**{pct(r['frame_blind'])}** | "
                     f"{pct(r['switched_incorrectly'])} | "
                     f"{pct(r['partial'])} |")
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

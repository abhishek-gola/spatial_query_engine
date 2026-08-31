"""Which reference frame does an LLM implicitly use?

The experiment. Take the queries where two frames genuinely disagree, hand the
model the same scene graph the resolver sees, ask it which object the sentence
refers to, and then classify its answer by *which frame it matches*. The model is
never asked about frames and the prompt never mentions them.

This is worth more than an accuracy comparison. A model that scores 60% on
projective relations tells you it is imperfect; a model whose answers match the
egocentric reading 80% of the time on queries where the object's own frame gives
a different object tells you *what convention it has absorbed* -- and that it has
one, silently, in a way nobody labelled or chose.

Four outcomes per query, and the third is the interesting one:

* matches exactly one frame -> that frame is what the model used here;
* matches several (they coincide) -> uninformative, excluded;
* matches none -> the model picked a third object; a plain error;
* unparseable -> excluded, counted separately.

Design choices that keep it honest:

* **Only frame-split queries.** On a query where every frame agrees, a correct
  answer says nothing about the model's convention.
* **The prompt carries no frame vocabulary.** No "egocentric", no "from your
  point of view". Mentioning frames would prime the very thing being measured.
* **Object ids are shuffled** per query, so a model cannot do well by preferring
  low ids, and the id ordering carries no positional hint.
* **The viewpoint is given explicitly**, as a camera position and look direction,
  because an egocentric reading is undefined without one. Withholding it and then
  reporting that the model failed to be egocentric would be rigging the result.
* Responses are cached, so a re-run costs nothing and the numbers are stable.

Needs `pip install anthropic` and `ANTHROPIC_API_KEY`. Nothing else in the repo
depends on it.
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..query.parser_rules import parse
from ..query.resolver import Resolver
from ..relations.base import RelationConfig
from .schema import BenchItem

DEFAULT_MODEL = "claude-sonnet-5"

SYSTEM = """\
You are given a list of objects in a room, each with an id, a class name, the \
centre of its 3D bounding box in metres, and its size. You are also told where \
the observer is standing and which way they are looking.

Answer with the id of the single object the sentence refers to.

Reply with ONLY a JSON object: {"id": <integer>, "why": "<one short sentence>"}

If no object fits, reply {"id": null, "why": "..."}.
"""


def _fmt_scene(scene, viewpoint_eye, viewpoint_look, order: Sequence[int]) -> str:
    lines = []
    for oid in order:
        o = scene.by_id(oid)
        if o is None:
            continue
        c, e = o.center, o.extent
        lines.append(
            f"  id={o.id}  {o.canonical_label}  "
            f"centre=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})  "
            f"size=({e[0]:.2f} x {e[1]:.2f} x {e[2]:.2f})")
    head = [
        "Coordinates are metres in a right-handed frame with +Z up.",
        f"The observer is standing at "
        f"({viewpoint_eye[0]:.2f}, {viewpoint_eye[1]:.2f}, "
        f"{viewpoint_eye[2]:.2f})",
        f"and looking in direction "
        f"({viewpoint_look[0]:.2f}, {viewpoint_look[1]:.2f}, "
        f"{viewpoint_look[2]:.2f}).",
        "",
        "Objects:",
    ]
    return "\n".join(head + lines)


@dataclass
class VlmRow:
    item_id: str
    scene_id: str
    text: str
    relation_type: Optional[str]
    frame_answers: Dict[str, Optional[int]]
    model_id: Optional[int]
    model_why: str = ""
    matched_frames: List[str] = field(default_factory=list)
    outcome: str = "unparsed"   # one_frame | several_frames | no_frame | unparsed
    policy_frame: Optional[str] = None
    error: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def classify(model_id: Optional[int],
             frame_answers: Dict[str, Optional[int]]) -> Tuple[List[str], str]:
    """Which frames the model's answer is consistent with."""
    if model_id is None:
        return [], "no_frame"
    matched = sorted(k for k, v in frame_answers.items()
                     if v is not None and int(v) == int(model_id))
    if not matched:
        return [], "no_frame"
    if len(matched) == 1:
        return matched, "one_frame"
    return matched, "several_frames"


def select_queries(items: Sequence[BenchItem], scene_for: Callable,
                   cfg: Optional[RelationConfig] = None,
                   relation_types: Sequence[str] = ("projective_lateral",
                                                     "projective_frontal"),
                   limit: Optional[int] = None,
                   verbose: bool = True) -> List[Tuple[BenchItem, Dict]]:
    """Queries where two frames pick different objects, with those answers."""
    cfg = cfg or RelationConfig.load()
    resolvers: Dict[str, Resolver] = {}
    out: List[Tuple[BenchItem, Dict]] = []
    for it in items:
        if it.relation_type not in relation_types:
            continue
        try:
            scene = scene_for(it.scene_id)
        except Exception:
            continue
        if it.scene_id not in resolvers:
            resolvers[it.scene_id] = Resolver(scene, cfg)
        r = resolvers[it.scene_id]
        res = r.resolve(parse(it.text), it.viewpoint_spec())
        real = {k: v for k, v in res.frame_answers.items() if v is not None}
        if len(set(real.values())) < 2:
            continue
        out.append((it, {"frame_answers": real,
                         "policy_frame": res.frame_used,
                         "viewpoint": res.frame_decision.viewpoint.to_dict()
                         if res.frame_decision else {}}))
        if limit and len(out) >= limit:
            break
    if verbose:
        print(f"{len(out)} frame-split queries selected")
    return out


class Runner:
    """Calls the model, with an on-disk cache keyed by (model, prompt)."""

    def __init__(self, model: str = DEFAULT_MODEL,
                 cache_path: Optional[str] = None, max_tokens: int = 300):
        self.model = model
        self.max_tokens = max_tokens
        self.cache_path = cache_path
        self.cache: Dict[str, dict] = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                self.cache = json.load(f)
        self._client = None

    def _client_or_raise(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "the VLM baseline needs `pip install anthropic`") from exc
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set, so the baseline cannot run")
            self._client = anthropic.Anthropic()
        return self._client

    def _save(self):
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)) or ".",
                    exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f, indent=1)

    def ask(self, prompt: str) -> dict:
        key = f"{self.model}::{prompt}"
        if key in self.cache:
            return self.cache[key]
        client = self._client_or_raise()
        msg = client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=SYSTEM,
            messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in msg.content
                       if getattr(b, "type", "") == "text")
        out = {"raw": text}
        self.cache[key] = out
        self._save()
        return out


def parse_reply(text: str) -> Tuple[Optional[int], str]:
    """Pull (id, why) out of a reply. Tolerant of fences and stray prose."""
    if not text:
        return None, ""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-z]*\n", "", s)
        s = re.sub(r"\n```$", "", s.strip())
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        try:
            d = json.loads(s[i:j + 1])
            v = d.get("id")
            return (None if v is None else int(v)), str(d.get("why", ""))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    m = re.search(r"\bid\D{0,5}(\d+)", s, re.I)
    if m:
        return int(m.group(1)), s[:160]
    return None, s[:160]


def run(selected: Sequence[Tuple[BenchItem, Dict]], scene_for: Callable,
        runner: Runner, shuffle_ids: bool = True, seed: int = 0,
        verbose: bool = True) -> List[VlmRow]:
    rng = random.Random(seed)
    rows: List[VlmRow] = []
    for n, (it, info) in enumerate(selected):
        scene = scene_for(it.scene_id)
        vp = info.get("viewpoint") or {}
        eye = np.asarray(vp.get("eye") or [0.0, 0.0, 1.6], float)
        look = np.asarray(vp.get("look_dir") or [0.0, 1.0, 0.0], float)
        order = [o.id for o in scene.objects if not o.is_room_fixed
                 or o.canonical_label in ("door", "window", "whiteboard")]
        if shuffle_ids:
            rng.shuffle(order)
        prompt = (_fmt_scene(scene, eye, look, order)
                  + f"\n\nWhich object does this refer to?\n\"{it.text}\"")
        row = VlmRow(item_id=it.id, scene_id=it.scene_id, text=it.text,
                     relation_type=it.relation_type,
                     frame_answers=info["frame_answers"],
                     policy_frame=info.get("policy_frame"), model_id=None)
        try:
            reply = runner.ask(prompt)
            mid, why = parse_reply(reply.get("raw", ""))
            row.model_id, row.model_why = mid, why
            row.matched_frames, row.outcome = classify(mid, row.frame_answers)
        except Exception as exc:
            row.error = f"{type(exc).__name__}: {exc}"
            row.outcome = "unparsed"
        rows.append(row)
        if verbose and (n + 1) % 10 == 0:
            print(f"  {n + 1}/{len(selected)}", flush=True)
    return rows


def summarise(rows: Sequence[VlmRow]) -> Dict:
    informative = [r for r in rows if r.outcome == "one_frame"]
    frames = Counter(r.matched_frames[0] for r in informative)
    n_inf = len(informative)
    return {
        "n_queries": len(rows),
        "outcomes": dict(Counter(r.outcome for r in rows)),
        "n_errors": sum(1 for r in rows if r.error),
        "n_informative": n_inf,
        "implied_frame_counts": dict(frames),
        "implied_frame_share": {k: v / n_inf for k, v in frames.items()}
        if n_inf else {},
        "by_relation_type": {
            t: dict(Counter(r.matched_frames[0] for r in informative
                            if r.relation_type == t))
            for t in sorted({r.relation_type for r in rows if r.relation_type})},
        "agreement_with_policy": (
            sum(1 for r in informative
                if r.policy_frame and r.matched_frames[0] == r.policy_frame)
            / n_inf) if n_inf else None,
    }


def render(summary: Dict, model: str, title: str = "Which frame does the model use?") -> str:
    def pct(x):
        return "n/a" if x is None else f"{100.0 * x:.1f}%"

    L = [f"# {title}", "", f"Model: `{model}`", ""]
    L.append("Only queries where two reference frames pick **different** "
             "objects. The model gets the same object list the resolver sees, "
             "plus the observer's position and look direction. The prompt never "
             "mentions reference frames; object ids are shuffled per query.")
    L.append("")
    L.append(f"{summary['n_queries']} queries, "
             f"{summary['n_informative']} of which matched exactly one frame "
             f"and are therefore informative.")
    L.append("")
    L.append("| outcome | n |")
    L.append("|---|---|")
    for k, v in sorted(summary["outcomes"].items(), key=lambda kv: -kv[1]):
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("## Frame the model's answers imply")
    L.append("")
    L.append("| frame | n | share of informative |")
    L.append("|---|---|---|")
    for k, v in sorted(summary["implied_frame_counts"].items(),
                       key=lambda kv: -kv[1]):
        L.append(f"| {k} | {v} | "
                 f"{pct(summary['implied_frame_share'].get(k))} |")
    L.append("")
    if summary["by_relation_type"]:
        L.append("| relation type | " + " | ".join(
            sorted({k for d in summary["by_relation_type"].values()
                    for k in d})) + " |")
        keys = sorted({k for d in summary["by_relation_type"].values()
                       for k in d})
        L.append("|---" * (1 + len(keys)) + "|")
        for t, d in summary["by_relation_type"].items():
            L.append(f"| {t} | " + " | ".join(str(d.get(k, 0)) for k in keys)
                     + " |")
        L.append("")
    L.append(f"Agreement with this repo's frame-selection policy: "
             f"{pct(summary['agreement_with_policy'])}. That is a comparison of "
             f"two unvalidated conventions, not a correctness claim -- neither "
             f"has been checked against human labels on these queries.")
    L.append("")
    L.append("**Reading this.** A concentrated distribution means the model has "
             "absorbed one convention and applies it silently. A flat one means "
             "it has no stable convention, which is a different and arguably "
             "worse finding. Either way the model was never asked which frame "
             "to use, and never said.")
    return "\n".join(L)

"""LLM query parser, as an alternative to the rule parser.

Reported as a separate benchmark condition rather than as the default. The rule
parser is deterministic, and a benchmark that has to separate parsing failures
from spatial-reasoning failures needs at least one parser whose behaviour does
not change between runs. This one covers the phrasings the rules miss.

The model is asked for the same `Query` schema the rule parser produces and the
annotation tool stores, so all three are directly comparable. Output is
validated against the schema and against the relation lexicon; anything the
model invents is rejected and the item falls back to the rule parser with a note
saying so, which keeps a hallucinated relation from silently becoming a
benchmark result.

Needs `pip install anthropic` and `ANTHROPIC_API_KEY`.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

from ..relations.base import all_relations, canonical_relation
from .schema import Constraint, LevelSpec, OrdinalSpec, Phrase, Query

DEFAULT_MODEL = "claude-sonnet-5"

SYSTEM = """\
You convert an English spatial referring expression about an indoor scene into a \
strict JSON structure. You do not resolve it -- you have no access to the scene \
and must not guess object identities.

Return ONLY a JSON object, no prose, with this shape:

{
  "target": {
    "label": "mug" | null,          // the head noun's class; null for "thing"/"object"
    "attributes": ["red"],          // colours or materials only
    "size_word": "big" | null,      // bare size adjective
    "superlative": "tallest" | null,
    "plural": false,
    "ordinal": null | {
       "word": "second",
       "index": 1,                  // 0-based; null when "middle"
       "middle": false,
       "from_word": "left",         // left|right|front|behind|up|down
       "from_landmark": null        // e.g. "door" for "third chair from the door"
    },
    "level": null | {               // a surface INSIDE the object, e.g. a shelf
       "word": "middle", "index": null, "middle": true, "from_bottom": true
    },
    "constraints": [
      {"relation": "on", "anchors": [ <same shape as target> ], "superlative": false}
    ]
  },
  "frame_hint": null | "egocentric" | "intrinsic" | "addressee" | "world"
                     | "egocentric_image",
  "viewpoint_hint": null | "door" | "window" | "camera" | ...
}

Rules:
- `relation` must be one of: RELATIONS
- Nest constraints: "the mug on the shelf next to the door" puts `next_to door`
  inside the shelf anchor, not on the mug.
- "between A and B" gets two anchors.
- Set `frame_hint` ONLY when the sentence explicitly says whose left/right is
  meant: "from where I'm standing" (egocentric), "the chair's own left"
  (intrinsic), "facing the sofa" (addressee), "the left side of the room"
  (world), "in the photo" (egocentric_image). Otherwise leave it null -- do not
  guess a default, the caller has a documented policy for that.
- "the second X from the left" is an `ordinal` on X, not a relation.
- "the top shelf of the bookshelf" is `label: "bookshelf"` with a `level`.
- Use `superlative: true` on a constraint for "nearest to" / "closest to" /
  "farthest from", as opposed to plain "near" / "far from".
"""


def _prompt() -> str:
    return SYSTEM.replace("RELATIONS", ", ".join(all_relations()))


def _phrase_from(d: dict, depth: int = 0) -> Phrase:
    if not isinstance(d, dict) or depth > 4:
        return Phrase()
    ordinal = None
    o = d.get("ordinal")
    if isinstance(o, dict):
        ordinal = OrdinalSpec(
            word=str(o.get("word") or ""),
            index=(None if o.get("index") is None else int(o["index"])),
            middle=bool(o.get("middle", False)),
            from_word=str(o.get("from_word") or "left"),
            from_landmark=(None if not o.get("from_landmark")
                           else str(o["from_landmark"])))
    level = None
    lv = d.get("level")
    if isinstance(lv, dict):
        level = LevelSpec(
            word=str(lv.get("word") or ""),
            index=(None if lv.get("index") is None else int(lv["index"])),
            middle=bool(lv.get("middle", False)),
            from_bottom=bool(lv.get("from_bottom", True)))
    cons: List[Constraint] = []
    for c in (d.get("constraints") or []):
        if not isinstance(c, dict):
            continue
        rel = canonical_relation(str(c.get("relation") or ""))
        if rel is None:
            continue                     # invented relation: drop it
        anchors = [_phrase_from(a, depth + 1) for a in (c.get("anchors") or [])]
        cons.append(Constraint(relation=rel, anchors=anchors,
                               negated=bool(c.get("negated", False)),
                               superlative=bool(c.get("superlative", False))))
    label = d.get("label")
    return Phrase(
        label=(None if label in (None, "", "null") else str(label)),
        text=str(d.get("text") or label or ""),
        attributes=[str(a) for a in (d.get("attributes") or [])],
        size_word=(None if not d.get("size_word") else str(d["size_word"])),
        superlative=(None if not d.get("superlative") else str(d["superlative"])),
        ordinal=ordinal, level=level, plural=bool(d.get("plural", False)),
        constraints=cons)


def query_from_json(text: str, payload: dict) -> Query:
    """Validate a model response into a `Query`. Raises on anything unusable."""
    if not isinstance(payload, dict) or "target" not in payload:
        raise ValueError("response has no 'target'")
    target = _phrase_from(payload["target"])
    fh = payload.get("frame_hint")
    valid_frames = {"egocentric", "egocentric_bearing", "egocentric_image",
                    "intrinsic", "addressee", "world"}
    if fh is not None and str(fh) not in valid_frames:
        fh = None
    if not target.label and not target.constraints and target.ordinal is None:
        raise ValueError("response has neither a class, a relation nor an ordinal")
    return Query(text=text, target=target,
                 frame_hint=(None if fh is None else str(fh)),
                 viewpoint_hint=(None if not payload.get("viewpoint_hint")
                                 else str(payload["viewpoint_hint"])),
                 parser="llm", parse_confidence=1.0)


def _extract_json(s: str) -> dict:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-z]*\n", "", s)
        s = re.sub(r"\n```$", "", s.strip())
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("no JSON object in the response")
    return json.loads(s[i:j + 1])


class LLMParser:
    """Caches by query text, so repeated benchmark runs cost one call per item."""

    def __init__(self, model: str = DEFAULT_MODEL, cache_path: Optional[str] = None,
                 fallback: bool = True, max_tokens: int = 1200):
        self.model = model
        self.fallback = fallback
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
                raise ImportError("the LLM parser needs `pip install anthropic`"
                                  ) from exc
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            self._client = anthropic.Anthropic()
        return self._client

    def _save(self):
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)) or ".",
                    exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f, indent=1)

    def __call__(self, text: str) -> Query:
        if text in self.cache:
            try:
                return query_from_json(text, self.cache[text])
            except ValueError:
                pass
        try:
            client = self._client_or_raise()
            msg = client.messages.create(
                model=self.model, max_tokens=self.max_tokens,
                system=_prompt(),
                messages=[{"role": "user", "content": text}])
            payload = _extract_json("".join(
                b.text for b in msg.content if getattr(b, "type", "") == "text"))
            q = query_from_json(text, payload)
            self.cache[text] = payload
            self._save()
            return q
        except Exception as exc:
            if not self.fallback:
                raise
            from .parser_rules import parse as rule_parse
            q = rule_parse(text)
            q.parser = "rules(llm-fallback)"
            q.parse_confidence = min(q.parse_confidence, 0.6)
            q.notes.append(f"the LLM parser failed ({type(exc).__name__}: "
                           f"{exc}); fell back to the rule parser")
            return q


def parse(text: str, **kw) -> Query:
    return LLMParser(**kw)(text)

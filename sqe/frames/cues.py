"""Reading the frame out of the sentence.

Most spatial queries carry no explicit frame marker, but a useful minority do,
and those markers are unambiguous when present: "from where I'm standing" is
egocentric, "the chair's left" is intrinsic, "the left side of the room" is
allocentric. This module finds them.

What it does *not* do is guess when there is no marker. That decision belongs to
`sqe.frames.policy`, which applies a documented default and records that it was
a default. Keeping the two apart is what makes it possible to report accuracy
separately for queries that specified a frame and queries that did not -- and
the second number is the one that matters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

#: Phrases that pin the frame to the viewer. The value is the frame kind and,
#: where the phrase also names the viewpoint, a viewpoint hint.
VIEWER_PHRASES: List[Tuple[str, str, Optional[str]]] = [
    (r"\bfrom (?:my|the viewer'?s?|our) (?:point of view|pov|perspective|view|side)\b", "egocentric", None),
    (r"\bfrom where (?:i|we|you)(?:'m| am|'re| are)? (?:standing|stood|sitting)\b", "egocentric", None),
    (r"\bfrom (?:my|our) (?:vantage|angle|position|viewpoint)\b", "egocentric", None),
    (r"\bfrom (?:here|there|this angle|this side|this position)\b", "egocentric", None),
    (r"\bas (?:i|we) (?:see|look at|am looking at|view)\b", "egocentric", None),
    (r"\bfrom the camera(?:'s)?(?: point of view| perspective| view)?\b", "egocentric", "camera"),
    (r"\bin (?:the |this )?(?:image|picture|photo|photograph|frame|shot|view|screen)\b", "egocentric_image", None),
    (r"\bon (?:the )?screen\b", "egocentric_image", None),
    (r"\bmy (?:left|right)\b", "egocentric", None),
    (r"\bto me\b", "egocentric", None),
    (r"\bnearer (?:to )?(?:me|us|the camera)\b", "egocentric", None),
]

#: A noun phrase: at most four words, and never spanning a preposition. A lazy
#: `[a-z ]{1,30}?` here is a trap -- on "the second mug from the bookshelf's own
#: left" it captures "second mug from the bookshelf" and the marker span then
#: swallows the whole target phrase.
_NP = r"((?:[a-z]+ ){0,3}[a-z]+)"

#: Phrases that pin the frame to the anchor object's own orientation. The
#: possessive forms ("the sofa's left", "the sofa's own left") are handled by
#: POSSESSIVE_RE below, which also names the anchor.
INTRINSIC_PHRASES: List[Tuple[str, str]] = [
    (rf"\bfrom (?:the )?{_NP}'s (?:point of view|pov|perspective|view)\b",
     "intrinsic"),
    (r"\bits own (?:left|right|front|back)\b", "intrinsic"),
    (r"\bon its (?:left|right)\b", "intrinsic"),
    (rf"\bas (?:the )?{_NP} faces\b", "intrinsic"),
    (rf"\bin the direction (?:the )?{_NP} (?:faces|is facing)\b", "intrinsic"),
]

#: Possessive left/right, e.g. "the sofa's left". Captured separately because
#: it is by far the most common intrinsic marker and it names the anchor.
POSSESSIVE_RE = re.compile(
    r"\b((?:[a-z]+ ){0,3}[a-z]+)'s (?:own )?(left|right|front|back|rear)\b")

#: Tokens that cannot be part of the possessed noun phrase. Without this,
#: "the mug on the laptop's left" captures "mug on the laptop" as the anchor
#: instead of "laptop", and the intrinsic frame then gets built on the wrong
#: object.
_NP_STOP = {"the", "a", "an", "of", "on", "in", "at", "to", "from", "with",
            "and", "or", "is", "that", "this", "which", "what", "near",
            "beside", "next", "above", "below", "under", "behind", "by",
            "for", "over", "inside", "onto", "into"}


def _trim_noun_phrase(phrase: str) -> str:
    """Keep only the trailing noun phrase, dropping anything up to and
    including the last preposition or article."""
    toks = [t for t in phrase.split() if t]
    cut = 0
    for i, t in enumerate(toks):
        if t in _NP_STOP:
            cut = i + 1
    toks = toks[cut:]
    return " ".join(toks[-3:]) if toks else ""

#: Phrases meaning "as seen by someone facing the anchor" -- the mirrored
#: reading of the anchor's own sides.
ADDRESSEE_PHRASES: List[str] = [
    r"\b(?:if|when|as) you(?:'re| are)? (?:facing|stood facing|"
    r"standing in front of|looking at)\b",
    rf"\bfacing (?:the )?{_NP}\b",
    rf"\bfrom the front of (?:the )?{_NP}\b",
    r"\bstanding in front of\b",
]

#: Phrases that pin the frame to the room.
WORLD_PHRASES: List[str] = [
    r"\bthe (?:left|right) side of the room\b",
    r"\bthe (?:left|right) (?:half|part|end) of the room\b",
    r"\bthe room'?s? (?:left|right|front|back)\b",
    r"\bthe (?:north|south|east|west) (?:wall|side|end|corner)\b",
    r"\bthe (?:far|near) (?:end|side|wall) of the room\b",
    r"\bthe (?:left|right) wall\b",
    r"\bin room coordinates\b",
]

#: Landmarks a viewpoint can be placed at.
_LANDMARKS = (r"door|doorway|entrance|entry|window|sofa|couch|bed|desk|table|"
              r"kitchen|corner|whiteboard|sink")

#: Unambiguous landmark viewpoints: the wording can only be about where the
#: speaker stands.
LANDMARK_VIEWPOINT_RE = re.compile(
    r"\b(?:standing at|standing by|standing in|seen from|viewed from|looking "
    r"from|if you stand at|when you stand at)"
    rf" (?:the )?({_LANDMARKS})\b")

#: The bare "from the X" form, which is genuinely ambiguous: it is a viewpoint
#: in "the chair, seen from the door" and a counting origin in "the third chair
#: from the door". Tagged separately so the parser can drop it when the sentence
#: is counting something.
#: The `(?!'s)` matters. "from the bed's point of view" names the bed's own
#: frame, not a place a person stands, and without the lookahead this pattern
#: fired inside it and silently relocated the observer to the bed.
LANDMARK_VIEWPOINT_FROM_RE = re.compile(
    rf"\bfrom (?:the )?({_LANDMARKS})\b(?!'s)")
ENTERING_RE = re.compile(
    r"\b(?:as|when) (?:you|i|we) (?:walk|come|enter|step) (?:in|into|inside|"
    r"through the door)\b")


@dataclass
class FrameCue:
    """One piece of linguistic evidence about the frame."""
    kind: str                      # frame kind implied
    weight: float
    evidence: str                  # the span that fired
    rule: str                      # which rule matched
    anchor_hint: Optional[str] = None
    viewpoint_hint: Optional[str] = None

    def to_dict(self) -> dict:
        return {"kind": self.kind, "weight": self.weight,
                "evidence": self.evidence, "rule": self.rule,
                "anchor_hint": self.anchor_hint,
                "viewpoint_hint": self.viewpoint_hint}


def extract_cues(text: str) -> List[FrameCue]:
    """All frame cues in a query, strongest first.

    A query can carry more than one, and they can conflict ("from where I'm
    standing, what is on the chair's left?"). Conflicts are not resolved here;
    they are handed to the policy and end up as a flagged ambiguity, which is
    the honest outcome.
    """
    if not text:
        return []
    s = " ".join(str(text).lower().split())
    cues: List[FrameCue] = []

    for pattern, kind, vp in VIEWER_PHRASES:
        m = re.search(pattern, s)
        if m:
            cues.append(FrameCue(kind, 1.0, m.group(0), "viewer_phrase",
                                 viewpoint_hint=vp))

    for pattern, kind in INTRINSIC_PHRASES:
        m = re.search(pattern, s)
        if m:
            hint = (_trim_noun_phrase(m.group(1))
                    if m.re.groups >= 1 and m.lastindex else None) or None
            cues.append(FrameCue(kind, 1.0, m.group(0), "intrinsic_phrase",
                                 anchor_hint=hint))

    for m in POSSESSIVE_RE.finditer(s):
        hint = _trim_noun_phrase(m.group(1))
        if not hint:
            continue
        cues.append(FrameCue("intrinsic", 0.9,
                             f"{hint}'s {m.group(2)}", "possessive",
                             anchor_hint=hint))
        break

    for pattern in ADDRESSEE_PHRASES:
        m = re.search(pattern, s)
        if m:
            hint = (_trim_noun_phrase(m.group(1))
                    if m.re.groups >= 1 and m.lastindex else None) or None
            cues.append(FrameCue("addressee", 0.95, m.group(0),
                                 "addressee_phrase", anchor_hint=hint))
            break

    for pattern in WORLD_PHRASES:
        m = re.search(pattern, s)
        if m:
            cues.append(FrameCue("world", 1.0, m.group(0), "world_phrase"))
            break

    m = LANDMARK_VIEWPOINT_RE.search(s)
    if m:
        cues.append(FrameCue("egocentric", 1.0, m.group(0),
                             "landmark_viewpoint", viewpoint_hint=m.group(1)))
    else:
        m = LANDMARK_VIEWPOINT_FROM_RE.search(s)
        if m:
            cues.append(FrameCue("egocentric", 0.8, m.group(0),
                                 "landmark_viewpoint_from",
                                 viewpoint_hint=m.group(1)))
    if ENTERING_RE.search(s):
        cues.append(FrameCue("egocentric", 1.0, "as you walk in",
                             "landmark_viewpoint", viewpoint_hint="door"))

    # de-duplicate by (kind, rule), keeping the strongest
    best: Dict[Tuple[str, str], FrameCue] = {}
    for c in cues:
        key = (c.kind, c.rule)
        if key not in best or c.weight > best[key].weight:
            best[key] = c
    out = sorted(best.values(), key=lambda c: -c.weight)
    return out


def cue_summary(cues: List[FrameCue]) -> Dict:
    kinds = {}
    for c in cues:
        kinds[c.kind] = max(kinds.get(c.kind, 0.0), c.weight)
    return {"kinds": kinds, "conflicting": len(kinds) > 1,
            "explicit": bool(cues),
            "cues": [c.to_dict() for c in cues]}

"""Deciding when a query has no single right answer.

A resolver that always returns one object is easier to score and less useful.
Several distinct things can make a spatial query genuinely underdetermined, and
they need different fixes, so they are reported separately rather than as one
confidence number:

``frame``
    The plausible reference frames disagree about the answer. This is the
    interesting one: it means the sentence is fine, the perception is fine, and
    the question simply does not have an answer until someone says whose left
    they meant.
``frame_unavailable``
    The frame the sentence asked for cannot be built -- typically the anchor's
    front could not be estimated.
``world_undetermined``
    The room-canonical frame was used but its 4-fold rotation is a near-tie.
``ordinal_degenerate``
    The candidates are not ordered along the counting axis at all.
``ordinal_tie``
    Two candidates are so close along the axis that their order is noise.
``level_even``
    "The middle shelf" of a unit with an even number of shelves.
``score_tie``
    The top two candidates score within a whisker of each other.
``anchor``
    The anchor itself is ambiguous -- several objects match it equally well.
``no_candidate``
    Nothing matched.

The benchmark labels every item for whether it *should* be ambiguous, so these
flags are scored the same way the answers are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

#: Two candidate scores this close together are treated as a tie.
SCORE_TIE_ABS = 0.04
SCORE_TIE_REL = 0.08

#: Below this margin the room-canonical frame is a coin flip.
WORLD_MARGIN_MIN = 0.12


@dataclass
class AmbiguityReport:
    ambiguous: bool = False
    kinds: List[str] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    frame_answers: Dict[str, Optional[int]] = field(default_factory=dict)
    detail: Dict = field(default_factory=dict)

    def add(self, kind: str, message: str, **detail):
        self.ambiguous = True
        if kind not in self.kinds:
            self.kinds.append(kind)
        self.messages.append(message)
        if detail:
            self.detail.setdefault(kind, {}).update(detail)

    @property
    def frame_ambiguous(self) -> bool:
        return "frame" in self.kinds

    def to_dict(self) -> dict:
        return {"ambiguous": self.ambiguous, "kinds": list(self.kinds),
                "messages": list(self.messages),
                "frame_answers": dict(self.frame_answers),
                "detail": self.detail}

    def summary(self) -> str:
        if not self.ambiguous:
            return "unambiguous"
        return f"ambiguous ({', '.join(self.kinds)}): " + " ".join(self.messages)


def score_tie(top: float, second: Optional[float]) -> bool:
    if second is None:
        return False
    if top <= 1e-9:
        return False
    return (top - second) <= max(SCORE_TIE_ABS, SCORE_TIE_REL * top)


def check_frame_disagreement(frame_answers: Dict[str, Optional[int]],
                             chosen_kind: Optional[str],
                             report: AmbiguityReport,
                             label_of=None) -> None:
    """Flag when plausible frames pick different objects.

    `frame_answers` maps a frame kind to the object id it selects. Frames that
    could not be built are expected to map to None and are ignored here --
    `frame_unavailable` covers those.
    """
    answers = {k: v for k, v in frame_answers.items() if v is not None}
    report.frame_answers.update(frame_answers)
    if len(answers) < 2:
        return
    distinct = set(answers.values())
    if len(distinct) < 2:
        return
    groups: Dict[int, List[str]] = {}
    for kind, oid in answers.items():
        groups.setdefault(oid, []).append(kind)
    parts = []
    for oid, kinds in sorted(groups.items()):
        name = label_of(oid) if label_of else f"#{oid}"
        parts.append(f"{name} under {'/'.join(sorted(kinds))}")
    chosen_note = (f"; answering with the {chosen_kind} reading"
                   if chosen_kind else "")
    report.add("frame",
               "the reference frames disagree: " + "; ".join(parts) + chosen_note,
               groups={str(k): v for k, v in groups.items()},
               chosen=chosen_kind)

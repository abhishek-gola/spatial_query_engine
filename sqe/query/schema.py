"""The structured form of a referring expression.

This schema does double duty. It is what the parser produces, and it is what a
benchmark item stores as its ground-truth parse. That matters for the error
decomposition: with a gold parse available, a failed query can be re-run with
the parser bypassed, and if it then succeeds the failure was linguistic rather
than spatial. Without that, every number in the report is a mixture of three
different kinds of mistake.

Everything here is JSON round-trippable, because the annotation tool writes it
and the evaluator reads it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

FRAME_KINDS = ("egocentric", "egocentric_bearing", "egocentric_image",
               "intrinsic", "addressee", "world")

#: The label set a benchmark item's relation is filed under, for the
#: accuracy-by-relation-type table.
RELATION_TYPES = ("projective_lateral", "projective_frontal", "vertical",
                  "proximity", "ordinal", "comparative", "between",
                  "containment")


@dataclass
class OrdinalSpec:
    """"the second ... from the left", "the middle one", "the third from the door"."""
    word: str = ""                       # surface form: 'second', 'middle'
    index: Optional[int] = None          # 0-based; None for 'middle'
    middle: bool = False
    from_word: str = "left"              # left | right | front | behind | up | down
    from_landmark: Optional[str] = None   # 'door' -> order by distance instead

    def to_dict(self) -> dict:
        return {"word": self.word, "index": self.index, "middle": self.middle,
                "from_word": self.from_word, "from_landmark": self.from_landmark}

    @staticmethod
    def from_dict(d: dict) -> "OrdinalSpec":
        return OrdinalSpec(word=d.get("word", ""), index=d.get("index"),
                           middle=bool(d.get("middle", False)),
                           from_word=d.get("from_word", "left"),
                           from_landmark=d.get("from_landmark"))


@dataclass
class LevelSpec:
    """A horizontal surface inside an object: "the middle shelf"."""
    word: str = ""                       # 'middle' | 'top' | 'bottom' | 'second'
    index: Optional[int] = None          # 0-based from the bottom
    middle: bool = False
    from_bottom: bool = True

    def to_dict(self) -> dict:
        return {"word": self.word, "index": self.index, "middle": self.middle,
                "from_bottom": self.from_bottom}

    @staticmethod
    def from_dict(d: dict) -> "LevelSpec":
        return LevelSpec(word=d.get("word", ""), index=d.get("index"),
                         middle=bool(d.get("middle", False)),
                         from_bottom=bool(d.get("from_bottom", True)))


@dataclass
class Phrase:
    """A noun phrase: what to look for, and how to narrow it down.

    Recursive through `constraints`, so "the mug on the shelf next to the door"
    parses without flattening.
    """
    label: Optional[str] = None          # canonical-ish class name
    text: str = ""                       # the surface noun phrase, for open vocab
    attributes: List[str] = field(default_factory=list)   # colours, materials
    size_word: Optional[str] = None      # 'big', 'tall' -- a soft superlative
    superlative: Optional[str] = None    # 'largest', 'tallest'
    ordinal: Optional[OrdinalSpec] = None
    level: Optional[LevelSpec] = None
    plural: bool = False
    constraints: List["Constraint"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"label": self.label, "text": self.text,
                "attributes": list(self.attributes), "size_word": self.size_word,
                "superlative": self.superlative,
                "ordinal": None if self.ordinal is None else self.ordinal.to_dict(),
                "level": None if self.level is None else self.level.to_dict(),
                "plural": self.plural,
                "constraints": [c.to_dict() for c in self.constraints]}

    @staticmethod
    def from_dict(d: dict) -> "Phrase":
        return Phrase(
            label=d.get("label"), text=d.get("text", ""),
            attributes=list(d.get("attributes", [])),
            size_word=d.get("size_word"), superlative=d.get("superlative"),
            ordinal=None if d.get("ordinal") is None
            else OrdinalSpec.from_dict(d["ordinal"]),
            level=None if d.get("level") is None
            else LevelSpec.from_dict(d["level"]),
            plural=bool(d.get("plural", False)),
            constraints=[Constraint.from_dict(c) for c in d.get("constraints", [])])

    def describe(self) -> str:
        bits = []
        if self.ordinal:
            bits.append(self.ordinal.word or f"#{self.ordinal.index}")
        if self.superlative:
            bits.append(self.superlative)
        if self.size_word:
            bits.append(self.size_word)
        bits += self.attributes
        bits.append(self.label or self.text or "thing")
        s = " ".join(b for b in bits if b)
        if self.level:
            s += f" <level: {self.level.word or self.level.index}>"
        for c in self.constraints:
            s += f" [{c.relation} {' & '.join(a.describe() for a in c.anchors)}]"
        return s


@dataclass
class Constraint:
    """One relation the target must satisfy with respect to its anchors."""
    relation: str
    anchors: List[Phrase] = field(default_factory=list)
    negated: bool = False
    #: True when the surface form was a superlative -- "nearest to" rather than
    #: "near". The two mean different things: "near the door" is an absolute
    #: threshold and is simply false at three metres, while "nearest to the
    #: door" is a ranking and always has a winner.
    superlative: bool = False

    def to_dict(self) -> dict:
        return {"relation": self.relation, "negated": self.negated,
                "superlative": self.superlative,
                "anchors": [a.to_dict() for a in self.anchors]}

    @staticmethod
    def from_dict(d: dict) -> "Constraint":
        return Constraint(relation=d["relation"],
                          anchors=[Phrase.from_dict(a) for a in d.get("anchors", [])],
                          negated=bool(d.get("negated", False)),
                          superlative=bool(d.get("superlative", False)))


@dataclass
class Query:
    """A parsed spatial query."""
    text: str = ""
    target: Phrase = field(default_factory=Phrase)
    frame_hint: Optional[str] = None          # from an explicit cue
    viewpoint_hint: Optional[str] = None      # 'door', 'camera', ...
    frame_cues: List[dict] = field(default_factory=list)
    expects_multiple: bool = False
    parser: str = "unknown"
    parse_confidence: float = 1.0
    notes: List[str] = field(default_factory=list)

    @property
    def constraints(self) -> List[Constraint]:
        return self.target.constraints

    @property
    def primary_relation(self) -> Optional[str]:
        return self.target.constraints[0].relation if self.target.constraints else None

    def relation_type(self) -> Optional[str]:
        """Bucket the query for the accuracy-by-relation-type table."""
        if self.target.ordinal is not None:
            return "ordinal"
        rel = self.primary_relation
        if rel is None:
            if self.target.superlative or self.target.size_word:
                return "comparative"
            return None
        return relation_type_of(rel)

    def to_dict(self) -> dict:
        return {"text": self.text, "target": self.target.to_dict(),
                "frame_hint": self.frame_hint,
                "viewpoint_hint": self.viewpoint_hint,
                "frame_cues": list(self.frame_cues),
                "expects_multiple": self.expects_multiple,
                "parser": self.parser,
                "parse_confidence": self.parse_confidence,
                "notes": list(self.notes)}

    @staticmethod
    def from_dict(d: dict) -> "Query":
        return Query(text=d.get("text", ""),
                     target=Phrase.from_dict(d.get("target", {})),
                     frame_hint=d.get("frame_hint"),
                     viewpoint_hint=d.get("viewpoint_hint"),
                     frame_cues=list(d.get("frame_cues", [])),
                     expects_multiple=bool(d.get("expects_multiple", False)),
                     parser=d.get("parser", "unknown"),
                     parse_confidence=float(d.get("parse_confidence", 1.0)),
                     notes=list(d.get("notes", [])))

    def describe(self) -> str:
        s = self.target.describe()
        if self.frame_hint:
            s += f" <frame: {self.frame_hint}>"
        return s


def relation_type_of(relation: str) -> str:
    """Map a relation name onto its benchmark bucket."""
    if relation in ("left", "right"):
        return "projective_lateral"
    if relation in ("front", "behind"):
        return "projective_frontal"
    if relation in ("on", "above", "below"):
        return "vertical"
    if relation == "inside":
        return "containment"
    if relation == "between":
        return "between"
    if relation in ("near", "next_to", "far_from"):
        return "proximity"
    if relation in ("taller", "shorter", "bigger", "smaller", "wider"):
        return "comparative"
    if relation == "ordinal":
        return "ordinal"
    return "other"


def is_frame_dependent_type(relation_type: Optional[str]) -> bool:
    """Whether a bucket's answer can change with the reference frame."""
    return relation_type in ("projective_lateral", "projective_frontal", "ordinal")

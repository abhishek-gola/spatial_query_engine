"""Benchmark item schema.

One JSONL line per query. The fields that make the benchmark worth more than an
accuracy number:

``frame``
    Which reference frame the annotator judged the sentence to mean. Without
    this the report cannot say whether a failure was a frame-convention error,
    which is the whole point.

``ambiguous`` / ``ambiguity_kind``
    Whether the sentence has a single referent *at all*. A benchmark that
    forces one gold answer onto an ambiguous query rewards systems for guessing
    confidently, which is the opposite of what a spatial reasoner should do.
    Roughly a fifth of naturally-phrased queries land here.

``gold_parse``
    The structured form. Lets the evaluator re-run a failed item with the parser
    bypassed and attribute the failure to language rather than to geometry.

``answers_by_frame``
    Optional: what the answer would be under each frame. When the annotator
    fills this in, the evaluator can tell "the system picked the wrong frame"
    apart from "the system picked the right frame and still got it wrong",
    which are different bugs with different fixes.

``viewpoint``
    Which camera pose or position the annotator had in mind. Egocentric queries
    are meaningless without it, and leaving it implicit is one of the ways
    existing benchmarks quietly bake in a frame convention.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..query.schema import Query

SCHEMA_VERSION = 2

#: What kind of ambiguity the annotator saw, when they marked one.
AMBIGUITY_KINDS = ("frame", "ordinal_degenerate", "ordinal_tie", "level_even",
                   "score_tie", "anchor", "world_undetermined", "vague",
                   "multiple_valid", "none")

FRAME_LABELS = ("egocentric", "egocentric_bearing", "egocentric_image",
                "intrinsic", "addressee", "world", "any", "unspecified")

DIFFICULTY = ("easy", "medium", "hard")


@dataclass
class BenchItem:
    id: str
    scene_id: str
    dataset: str
    text: str

    # -- gold answer ------------------------------------------------------
    target_ids: List[int] = field(default_factory=list)
    #: True when the query has no single referent. `target_ids` may then hold
    #: every acceptable answer, or be empty if none is acceptable.
    ambiguous: bool = False
    ambiguity_kind: str = "none"

    # -- gold interpretation ----------------------------------------------
    relation: Optional[str] = None
    relation_type: Optional[str] = None
    frame: str = "unspecified"
    frame_stated_in_text: bool = False
    answers_by_frame: Dict[str, Optional[int]] = field(default_factory=dict)
    gold_parse: Optional[dict] = None

    # -- viewpoint --------------------------------------------------------
    viewpoint_mode: str = "best_view"
    viewpoint_index: Optional[int] = None
    viewpoint_position: Optional[List[float]] = None
    viewpoint_landmark: Optional[str] = None

    # -- bookkeeping ------------------------------------------------------
    difficulty: str = "medium"
    annotator: str = ""
    source: str = "manual"
    notes: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        return d

    @staticmethod
    def from_dict(d: dict) -> "BenchItem":
        known = set(BenchItem.__dataclass_fields__)
        extra = {k: v for k, v in d.items() if k not in known}
        clean = {k: v for k, v in d.items() if k in known}
        item = BenchItem(**clean)
        if extra:
            item.notes = (item.notes + " | unknown fields: "
                          + ",".join(sorted(extra))).strip(" |")
        return item

    def viewpoint_spec(self):
        from ..frames.policy import ViewpointSpec
        import numpy as np
        return ViewpointSpec(
            mode=self.viewpoint_mode,
            index=self.viewpoint_index,
            position=(None if self.viewpoint_position is None
                      else np.asarray(self.viewpoint_position, float)),
            landmark=self.viewpoint_landmark)

    def gold_query(self) -> Optional[Query]:
        return None if self.gold_parse is None else Query.from_dict(self.gold_parse)

    def validate(self, scene=None) -> List[str]:
        """Return a list of problems with this item; empty means valid."""
        problems: List[str] = []
        if not self.id:
            problems.append("missing id")
        if not self.text.strip():
            problems.append("empty text")
        if self.frame not in FRAME_LABELS:
            problems.append(f"frame {self.frame!r} is not one of {FRAME_LABELS}")
        if self.ambiguity_kind not in AMBIGUITY_KINDS:
            problems.append(f"ambiguity_kind {self.ambiguity_kind!r} is unknown")
        if self.ambiguous and self.ambiguity_kind == "none":
            problems.append("marked ambiguous but ambiguity_kind is 'none'")
        if not self.ambiguous and len(self.target_ids) != 1:
            problems.append(f"unambiguous items need exactly one target, got "
                            f"{len(self.target_ids)}")
        if self.difficulty not in DIFFICULTY:
            problems.append(f"difficulty {self.difficulty!r} is unknown")
        if self.viewpoint_mode == "position" and not self.viewpoint_position:
            problems.append("viewpoint_mode 'position' needs viewpoint_position")
        if self.viewpoint_mode == "landmark" and not self.viewpoint_landmark:
            problems.append("viewpoint_mode 'landmark' needs viewpoint_landmark")
        if self.viewpoint_mode == "index" and self.viewpoint_index is None:
            problems.append("viewpoint_mode 'index' needs viewpoint_index")
        if scene is not None:
            for t in self.target_ids:
                if scene.by_id(t) is None:
                    problems.append(f"target id {t} is not in scene "
                                    f"{scene.scene_id}")
            for k, v in self.answers_by_frame.items():
                if k not in FRAME_LABELS:
                    problems.append(f"answers_by_frame has unknown frame {k!r}")
                if v is not None and scene.by_id(v) is None:
                    problems.append(f"answers_by_frame[{k}] = {v} is not in the "
                                    f"scene")
        return problems


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------

def write_jsonl(items: List[BenchItem], path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), sort_keys=True) + "\n")


def read_jsonl(path: str) -> List[BenchItem]:
    items: List[BenchItem] = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            try:
                items.append(BenchItem.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"{path}:{n} is not a valid benchmark item: "
                                 f"{exc}") from exc
    return items


def read_many(paths: List[str]) -> List[BenchItem]:
    out: List[BenchItem] = []
    seen = set()
    for p in paths:
        for it in read_jsonl(p):
            if it.id in seen:
                raise ValueError(f"duplicate benchmark item id {it.id!r} "
                                 f"(second occurrence in {p})")
            seen.add(it.id)
            out.append(it)
    return out


def describe_split(items: List[BenchItem]) -> Dict[str, Any]:
    """Composition of a benchmark file, for the report's first table."""
    from collections import Counter
    by_type = Counter(i.relation_type or "unknown" for i in items)
    by_frame = Counter(i.frame for i in items)
    by_scene = Counter(i.scene_id for i in items)
    n_amb = sum(1 for i in items if i.ambiguous)
    n_stated = sum(1 for i in items if i.frame_stated_in_text)
    frame_dep = sum(1 for i in items
                    if i.relation_type in ("projective_lateral",
                                            "projective_frontal", "ordinal"))
    return {
        "n_items": len(items),
        "n_scenes": len(by_scene),
        "by_relation_type": dict(by_type),
        "by_frame": dict(by_frame),
        "by_scene": dict(by_scene),
        "by_difficulty": dict(Counter(i.difficulty for i in items)),
        "n_ambiguous": n_amb,
        "fraction_ambiguous": (n_amb / len(items)) if items else 0.0,
        "n_frame_stated_in_text": n_stated,
        "n_frame_dependent": frame_dep,
        "fraction_frame_dependent": (frame_dep / len(items)) if items else 0.0,
        "n_with_answers_by_frame": sum(1 for i in items if i.answers_by_frame),
        "n_with_gold_parse": sum(1 for i in items if i.gold_parse),
    }

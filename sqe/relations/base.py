"""Relation scoring framework.

Every relation returns a continuous score in [0, 1] together with the named
components that produced it. The components matter as much as the score: when
the benchmark says a query failed, the error taxonomy needs to know *which*
term went wrong -- the cone test, the lateral separation, the depth overlap --
in order to attribute the failure to a frame convention rather than to
perception.

Thresholds live in `configs/relations.yaml` and are loaded into
`RelationConfig`. None of them are inline in the scoring code.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def ramp(x: float, lo: float, hi: float) -> float:
    """0 at or below `lo`, 1 at or above `hi`, linear in between."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def falloff(x: float, zero_at: float, one_at: float = 0.0) -> float:
    """1 at `one_at`, decaying to 0 at `zero_at`."""
    return ramp(-x, -zero_at, -one_at)


def gmean(values: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
    """Weighted geometric mean, with a floor so one zero does not erase all
    diagnostic information from the score."""
    v = np.clip(np.asarray(list(values), dtype=np.float64), 1e-6, 1.0)
    w = np.ones_like(v) if weights is None else np.asarray(list(weights), float)
    w = w / max(w.sum(), 1e-9)
    return float(np.exp(np.sum(w * np.log(v))))


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

@dataclass
class RelationConfig:
    # -- projective (left / right / front / behind) ------------------------
    cone_half_angle_deg: float = 55.0
    cone_full_credit_deg: float = 25.0
    lateral_separation_full: float = 0.75    # fraction of target beyond the face
    lateral_separation_zero: float = 0.05
    depth_overlap_slack: float = 0.45        # metres of tolerated depth offset
    height_band: float = 1.10                # metres before a height penalty bites
    height_band_hard: float = 2.20
    # "in front of X" implies relevance as well as direction: of two objects on
    # the anchor's front side, the near one is meant. Weighted low so it breaks
    # ties without ever overturning the direction terms.
    proximity_full_factor: float = 1.5       # x anchor radius, full credit
    proximity_zero_factor: float = 5.0       # x anchor radius, down to the floor
    proximity_floor: float = 0.30
    # Beyond this the pair is not in a shared local context at all and the
    # relation is false, not merely weak. "The monitor to the left of the
    # keyboard" does not describe a monitor on a different desk 3.6 m away.
    proximity_cutoff_factor: float = 8.0
    w_cone: float = 1.0
    w_separation: float = 0.9
    w_depth: float = 0.7
    w_height: float = 0.35
    w_proximity: float = 0.35

    # -- vertical ----------------------------------------------------------
    contact_tol: float = 0.08
    contact_tol_predicted: float = 0.15      # looser for reconstructed clouds
    support_min_overlap: float = 0.35
    above_min_overlap: float = 0.20
    above_gap_zero: float = 2.50
    containment_pad: float = 0.05
    containment_min_fraction: float = 0.60

    # -- proximity ---------------------------------------------------------
    near_base: float = 0.30                  # metres added to the size term
    near_size_factor: float = 0.60
    near_zero_multiplier: float = 3.0
    beside_height_tol: float = 0.35
    between_corridor: float = 0.70
    between_span_lo: float = 0.12
    between_span_hi: float = 0.88

    # -- ordinal -----------------------------------------------------------
    ordinal_min_spread_factor: float = 1.4   # spread vs mean object width
    ordinal_min_gap: float = 0.06            # metres between neighbours
    ordinal_tie_ratio: float = 0.35          # gap vs median gap before "fragile"

    # -- comparative -------------------------------------------------------
    size_ratio_significant: float = 1.25

    @staticmethod
    def load(path: Optional[str] = None) -> "RelationConfig":
        """Load from YAML, falling back to the defaults above.

        Unknown keys are an error rather than a silent no-op: a typo in a
        threshold name would otherwise look like a working experiment.
        """
        if path is None:
            path = os.path.join(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))),
                "configs", "relations.yaml")
        if not os.path.exists(path):
            return RelationConfig()
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        flat: Dict = {}
        for section in data.values():
            if isinstance(section, dict):
                flat.update(section)
        known = {f.name for f in fields(RelationConfig)}
        unknown = set(flat) - known
        if unknown:
            raise ValueError(f"unknown relation config keys in {path}: "
                             f"{sorted(unknown)}")
        return RelationConfig(**flat)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------

@dataclass
class RelationScore:
    value: float
    components: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def __float__(self) -> float:
        return float(self.value)

    def to_dict(self) -> dict:
        return {"value": float(self.value),
                "components": {k: float(v) for k, v in self.components.items()},
                "notes": list(self.notes)}


ZERO = RelationScore(0.0)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

@dataclass
class RelationSpec:
    name: str
    family: str                  # projective | vertical | proximity | ordinal |
                                 # comparative | topological
    frame_dependent: bool
    n_anchors: int
    aliases: Tuple[str, ...] = ()
    needs_container: bool = False
    needs_support: bool = False
    doc: str = ""


REGISTRY: Dict[str, RelationSpec] = {}
_ALIASES: Dict[str, str] = {}


def register(spec: RelationSpec) -> RelationSpec:
    REGISTRY[spec.name] = spec
    _ALIASES[spec.name] = spec.name
    for a in spec.aliases:
        _ALIASES[a] = spec.name
    return spec


def canonical_relation(name: str) -> Optional[str]:
    """Map a surface phrase like 'to the left of' onto a relation name.

    Candidate forms are tried from most to least literal. The order matters:
    stripping first would turn "on top of" into "on top" and "next to" into
    "next", neither of which is registered, and the phrase would come back
    unrecognised.
    """
    if not name:
        return None
    s = " ".join(str(name).strip().lower().replace("-", " ")
                 .replace("_", " ").split())
    if not s:
        return None

    def strip_prefix(x: str) -> str:
        for prefix in ("to the ", "on the ", "at the ", "in the ", "the "):
            if x.startswith(prefix):
                return x[len(prefix):].strip()
        return x

    def strip_suffix(x: str) -> str:
        for suffix in (" hand side of", " side of", " of", " from", " than",
                       " to", " with"):
            if x.endswith(suffix) and len(x) > len(suffix) + 1:
                return x[: -len(suffix)].strip()
        return x

    seen = set()
    for cand in (s, strip_prefix(s), strip_suffix(s),
                 strip_suffix(strip_prefix(s)),
                 strip_prefix(strip_suffix(s))):
        if cand and cand not in seen:
            seen.add(cand)
            hit = _ALIASES.get(cand)
            if hit:
                return hit
    return None


def spec(name: str) -> Optional[RelationSpec]:
    c = canonical_relation(name)
    return REGISTRY.get(c) if c else None


def is_frame_dependent(name: str) -> bool:
    sp = spec(name)
    return bool(sp and sp.frame_dependent)


def family(name: str) -> Optional[str]:
    sp = spec(name)
    return sp.family if sp else None


def all_relations() -> List[str]:
    return sorted(REGISTRY)


def frame_dependent_relations() -> List[str]:
    return sorted(n for n, s in REGISTRY.items() if s.frame_dependent)

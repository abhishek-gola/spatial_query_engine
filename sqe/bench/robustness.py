"""Does the headline number survive perturbing the hand-set constants?

The resolver has 47 hand-set numeric constants -- 34 in `RelationConfig` and 13
module-level -- and, until the benchmark is annotated, zero labelled examples to
have fitted them on. That is a real problem with the claim, not a cosmetic one:
pre-registering thresholds is the right discipline, but pre-registration only
buys credibility once an evaluation has actually run against them. Before that,
"18.8% of frame-dependent queries are frame-sensitive" could in principle be an
artefact of where the thresholds happen to sit.

This module makes that falsifiable. It jitters every query-time constant by a
log-uniform factor, re-measures frame sensitivity, and reports the distribution.
If the headline moves a little, it is a property of the scenes. If it moves a
lot, it is a property of my thresholds and should not be quoted.

What is *not* perturbed, and why: `sqe.perception.orientation`'s two constants
and `sqe.geom.room`'s ambiguity margin act at scene-build time, so varying them
means rebuilding every scene per trial. They are held fixed and listed in the
report, so the coverage claim is not overstated.

Constraints between thresholds are repaired after jittering -- a full-credit
cone angle wider than the half-angle, or a proximity cutoff inside the falloff,
would not be a perturbed configuration but a broken one.
"""

from __future__ import annotations

import contextlib
from dataclasses import fields, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..relations.base import RelationConfig
from .schema import BenchItem
from .sensitivity import measure, summarise

#: Query-time module constants, as (module path, attribute). These are read
#: inside functions, so rebinding the module attribute takes effect immediately.
MODULE_CONSTANTS: Tuple[Tuple[str, str], ...] = (
    ("sqe.query.resolver", "ALTERNATIVE_FRAME_PRIOR_RATIO"),
    ("sqe.query.resolver", "MIN_LABEL_MATCH"),
    ("sqe.query.resolver", "MIN_ANSWER_SCORE"),
    ("sqe.query.ambiguity", "SCORE_TIE_ABS"),
    ("sqe.query.ambiguity", "SCORE_TIE_REL"),
    ("sqe.query.resolver", "WORLD_MARGIN_MIN"),
    ("sqe.frames.policy", "INTRINSIC_MIN_FRONT_CONFIDENCE"),
    ("sqe.frames.policy", "MIN_USABLE_CONFIDENCE"),
    ("sqe.frames.policy", "EYE_HEIGHT"),
)

#: Held fixed because they act at scene-build time, not query time.
BUILD_TIME_CONSTANTS: Tuple[str, ...] = (
    "sqe.perception.orientation.CLEARANCE_SCALE",
    "sqe.perception.orientation.AGAINST_TOL",
    "sqe.geom.room.FORWARD_MARGIN_AMBIGUOUS",
    "sqe.query.resolver.MAX_ANCHOR_CANDIDATES",   # an integer count, not a threshold
)

#: Fields that are fractions in [0, 1] and must stay there.
_FRACTIONS = {
    "lateral_separation_full", "lateral_separation_zero", "proximity_floor",
    "support_min_overlap", "above_min_overlap", "containment_min_fraction",
    "ordinal_tie_ratio", "between_span_lo", "between_span_hi",
}


def perturb_config(cfg: RelationConfig, scale: float,
                   rng: np.random.Generator) -> RelationConfig:
    """Log-uniform jitter of every numeric field, then repair the constraints."""
    kw: Dict[str, float] = {}
    lo = np.log(1.0 / (1.0 + scale))
    hi = np.log(1.0 + scale)
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if not isinstance(v, (int, float)):
            continue
        nv = float(v) * float(np.exp(rng.uniform(lo, hi)))
        if f.name in _FRACTIONS:
            nv = float(np.clip(nv, 1e-3, 1.0))
        kw[f.name] = nv
    out = replace(cfg, **kw)

    # repair orderings: a jittered configuration must still be a coherent one
    if out.cone_full_credit_deg >= out.cone_half_angle_deg:
        out = replace(out, cone_full_credit_deg=0.5 * out.cone_half_angle_deg)
    if out.lateral_separation_zero >= out.lateral_separation_full:
        out = replace(out,
                      lateral_separation_zero=0.2 * out.lateral_separation_full)
    if out.height_band >= out.height_band_hard:
        out = replace(out, height_band=0.5 * out.height_band_hard)
    if not (out.proximity_full_factor < out.proximity_zero_factor
            < out.proximity_cutoff_factor):
        out = replace(out,
                      proximity_zero_factor=max(out.proximity_zero_factor,
                                                1.5 * out.proximity_full_factor),
                      proximity_cutoff_factor=max(
                          out.proximity_cutoff_factor,
                          1.5 * max(out.proximity_zero_factor,
                                    1.5 * out.proximity_full_factor)))
    if out.between_span_lo >= out.between_span_hi:
        out = replace(out, between_span_lo=0.1, between_span_hi=0.9)
    if out.above_gap_zero <= 0.1:
        out = replace(out, above_gap_zero=0.1)
    return out


@contextlib.contextmanager
def perturbed_module_constants(scale: float, rng: np.random.Generator):
    """Temporarily jitter the query-time module constants."""
    import importlib
    saved: List[Tuple[object, str, float]] = []
    lo = np.log(1.0 / (1.0 + scale))
    hi = np.log(1.0 + scale)
    try:
        for mod_path, attr in MODULE_CONSTANTS:
            mod = importlib.import_module(mod_path)
            if not hasattr(mod, attr):
                continue
            old = getattr(mod, attr)
            saved.append((mod, attr, old))
            new = float(old) * float(np.exp(rng.uniform(lo, hi)))
            if attr in ("MIN_LABEL_MATCH", "ALTERNATIVE_FRAME_PRIOR_RATIO",
                        "SCORE_TIE_REL", "INTRINSIC_MIN_FRONT_CONFIDENCE",
                        "MIN_USABLE_CONFIDENCE"):
                new = float(np.clip(new, 1e-3, 0.98))
            setattr(mod, attr, new)
        yield
    finally:
        for mod, attr, old in saved:
            setattr(mod, attr, old)


def sweep(items: Sequence[BenchItem], scene_for: Callable,
          n_trials: int = 20, scale: float = 0.30, seed: int = 0,
          base_cfg: Optional[RelationConfig] = None,
          verbose: bool = True) -> Dict:
    """Re-measure frame sensitivity under `n_trials` perturbed configurations."""
    base_cfg = base_cfg or RelationConfig.load()
    baseline = summarise(measure(items, scene_for, base_cfg))
    trials: List[Dict] = []
    rng = np.random.default_rng(seed)

    for t in range(n_trials):
        cfg = perturb_config(base_cfg, scale, rng)
        with perturbed_module_constants(scale, rng):
            s = summarise(measure(items, scene_for, cfg))
        trials.append({
            "trial": t,
            "disagreement_rate": s["disagreement"]["rate"],
            "n_frame_dependent": s["n_frame_dependent"],
            "by_relation_type": {k: v["disagreement_rate"]
                                 for k, v in s["by_relation_type"].items()},
            "n_flagged_ambiguous": s["n_flagged_ambiguous"],
        })
        if verbose:
            r = trials[-1]["disagreement_rate"]
            print(f"  trial {t + 1}/{n_trials}: disagreement "
                  f"{100.0 * r:.1f}%" if r is not None else
                  f"  trial {t + 1}/{n_trials}: n/a", flush=True)

    rates = np.array([t["disagreement_rate"] for t in trials
                      if t["disagreement_rate"] is not None], float)
    per_type: Dict[str, List[float]] = {}
    for t in trials:
        for k, v in t["by_relation_type"].items():
            if v is not None:
                per_type.setdefault(k, []).append(v)

    return {
        "n_trials": n_trials,
        "jitter_scale": scale,
        "n_config_fields_perturbed": sum(
            1 for f in fields(base_cfg)
            if isinstance(getattr(base_cfg, f.name), (int, float))),
        "n_module_constants_perturbed": len(MODULE_CONSTANTS),
        "held_fixed": list(BUILD_TIME_CONSTANTS),
        "baseline_disagreement_rate": baseline["disagreement"]["rate"],
        "baseline_n_frame_dependent": baseline["n_frame_dependent"],
        "disagreement_rate": {
            "median": float(np.median(rates)) if len(rates) else None,
            "p10": float(np.percentile(rates, 10)) if len(rates) else None,
            "p90": float(np.percentile(rates, 90)) if len(rates) else None,
            "min": float(rates.min()) if len(rates) else None,
            "max": float(rates.max()) if len(rates) else None,
        },
        "by_relation_type": {
            k: {"median": float(np.median(v)), "min": float(np.min(v)),
                "max": float(np.max(v))}
            for k, v in sorted(per_type.items())},
        "trials": trials,
    }


def render(result: Dict, title: str = "Threshold robustness") -> str:
    def pct(x):
        return "n/a" if x is None else f"{100.0 * x:.1f}%"

    r = result["disagreement_rate"]
    L = [f"# {title}", ""]
    L.append(f"The resolver has 47 hand-set numeric constants and, until the "
             f"benchmark is annotated, no labelled examples they could have "
             f"been fitted on. So the question is whether the headline "
             f"sensitivity number is a property of the scenes or of the "
             f"thresholds.")
    L.append("")
    L.append(f"{result['n_trials']} trials. Every one of "
             f"{result['n_config_fields_perturbed']} `RelationConfig` fields and "
             f"{result['n_module_constants_perturbed']} query-time module "
             f"constants jittered by a log-uniform factor of up to "
             f"±{100 * result['jitter_scale']:.0f}%, with ordering constraints "
             f"repaired afterwards.")
    L.append("")
    L.append(f"| | frame disagreement rate |")
    L.append(f"|---|---|")
    L.append(f"| **as configured** | **{pct(result['baseline_disagreement_rate'])}** |")
    L.append(f"| perturbed, median | {pct(r['median'])} |")
    L.append(f"| perturbed, 10th–90th pct | {pct(r['p10'])} – {pct(r['p90'])} |")
    L.append(f"| perturbed, full range | {pct(r['min'])} – {pct(r['max'])} |")
    L.append("")
    L.append("| relation type | median | min | max |")
    L.append("|---|---|---|---|")
    for k, v in result["by_relation_type"].items():
        L.append(f"| {k} | {pct(v['median'])} | {pct(v['min'])} | "
                 f"{pct(v['max'])} |")
    L.append("")
    L.append("Held fixed, because they act at scene-build time rather than "
             "query time and varying them means rebuilding every scene per "
             "trial:")
    L.append("")
    for c in result["held_fixed"]:
        L.append(f"* `{c}`")
    L.append("")
    L.append("**What this does and does not establish.** It shows whether the "
             "sensitivity number is stable against the thresholds. It does not "
             "validate the thresholds: only annotated data can say whether the "
             "resolver's answers are the ones a person meant. A stable number "
             "here plus an unvalidated policy is still an unvalidated policy -- "
             "it just means the *size of the frame problem* does not depend on "
             "my particular choices.")
    return "\n".join(L)

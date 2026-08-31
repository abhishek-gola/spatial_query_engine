"""Minimal pairs and the frame-instructability probe.

The most important tests here are the two control invariants. The probe's whole
value rests on them: if a cue-following resolver does not score
`switched_correctly`, the stimulus is malformed; if a resolver pinned to one
frame is not scored `frame_blind`, the label means nothing and no result from the
probe can be trusted.
"""

import numpy as np
import pytest

from sqe.bench.frame_probe import (OUTCOMES, ResolverProbe, classify_pair,
                                   stable_only,
                                   render)
from sqe.bench.frame_probe import run as probe_run
from sqe.bench.frame_probe import summarise as probe_summarise
from sqe.bench.minimal_pairs import (TEMPLATES, generate_for_scene,
                                     minimality_check, parse_check, summarise,
                                     validate_templates)
from sqe.bench.minimal_pairs import Arm
from sqe.relations.base import RelationConfig


def test_templates_name_their_frame_and_keep_their_anchor():
    """Guards two bugs that both passed a weaker check.

    A template can name its frame correctly and still lose the anchor to a
    pronoun ("...in front of **it**"), and it can keep everything and still not
    be minimal because it moves the observer as well.
    """
    assert validate_templates() == []


def test_parse_check_catches_a_pronoun_anchor():
    problems = parse_check("the mug in front of it", "front", "laptop", "mug")
    assert problems
    assert any("anchor" in p for p in problems)


def test_parse_check_catches_a_wrong_relation():
    problems = parse_check("the mug on the laptop", "left", "laptop", "mug")
    assert any("relation" in p for p in problems)


def test_minimality_check_catches_a_moved_observer():
    """The exact confound that broke the pinned control.

    "from the bed's point of view" also matched the bare landmark-viewpoint
    pattern, so that arm relocated the observer to the bed. Both arms named the
    right frame and parsed to the right relation, anchor and target; the pair
    still was not minimal.
    """
    good = [Arm("egocentric",
                "from where I am standing, the mug in front of the bed", 1, 1.0),
            Arm("intrinsic",
                "the mug in front of the bed, from the bed's point of view",
                2, 1.0)]
    assert minimality_check(good) == []

    moved = [Arm("egocentric", "the mug in front of the bed", 1, 1.0),
             Arm("other", "seen from the bed, the mug in front of the bed",
                 2, 1.0)]
    problems = minimality_check(moved)
    assert problems and "viewpoint_hint" in problems[0]


@pytest.mark.parametrize("given,want", [
    ({"egocentric": 7, "intrinsic": 9}, "switched_correctly"),
    ({"egocentric": 7, "intrinsic": 7}, "frame_blind"),
    ({"egocentric": 9, "intrinsic": 7}, "switched_incorrectly"),
    ({"egocentric": 7, "intrinsic": 3}, "partial"),
    ({"egocentric": None, "intrinsic": 9}, "no_answer"),
])
def test_classify_pair(given, want):
    assert classify_pair(given, {"egocentric": 7, "intrinsic": 9}) == want


@pytest.fixture(scope="module")
def studio_pairs(studio):
    return generate_for_scene(studio, RelationConfig(),
                              frames=("egocentric", "intrinsic"),
                              verbose=False)


def test_generated_pairs_are_well_formed(studio, studio_pairs):
    assert studio_pairs, "the synthetic studio should yield some pairs"
    counts = {}
    for o in studio.objects:
        counts[o.canonical_label] = counts.get(o.canonical_label, 0) + 1
    for p in studio_pairs:
        # two arms, different answers, both real objects of the target class
        assert len(p.arms) == 2
        ids = [a.answer_id for a in p.arms]
        assert len(set(ids)) == 2, p.id
        for oid in ids:
            o = studio.by_id(oid)
            assert o is not None and o.canonical_label == p.target_class
        # the anchor class is unique, or the sentence would not identify it
        assert counts[p.anchor_label] == 1, p.id
        # the target class has alternatives, or the frame could not matter
        assert p.n_class_instances >= 2, p.id
        # and the arms differ only in the frame
        assert minimality_check(p.arms) == []


def test_control_a_cue_following_resolver_switches(studio, studio_pairs):
    """If this fails the stimulus is malformed, not the resolver."""
    probe = ResolverProbe(lambda s: studio, RelationConfig())
    res = probe_run(studio_pairs, lambda s: studio, probe, verbose=False)
    s = probe_summarise(res)
    assert s["rates"]["switched_correctly"] == pytest.approx(1.0), s["outcomes"]


@pytest.mark.parametrize("frame", ["egocentric", "intrinsic", "addressee",
                                    "world"])
def test_control_a_pinned_resolver_is_frame_blind(studio, studio_pairs, frame):
    """The positive control for the `frame_blind` label.

    A system that provably cannot switch must be scored frame-blind on every
    pair. Anything less means the label is picking up something other than the
    frame, and every probe result would be suspect.
    """
    probe = ResolverProbe(lambda s: studio, RelationConfig(), pinned=frame)
    res = probe_run(studio_pairs, lambda s: studio, probe, verbose=False)
    s = probe_summarise(res)
    assert s["rates"]["frame_blind"] == pytest.approx(1.0), \
        f"{frame}: {s['outcomes']}"


def test_summarise_and_render(studio, studio_pairs):
    s = summarise(studio_pairs)
    assert s["n_pairs"] == len(studio_pairs)
    assert s["distinct_target_classes"] >= 1
    probe = ResolverProbe(lambda s_: studio, RelationConfig())
    res = probe_run(studio_pairs, lambda s_: studio, probe, verbose=False)
    text = render({"resolver (cue-following)": probe_summarise(res)})
    assert "frame blind" in text
    assert "circularity check" in text


def test_probe_records_an_error_rather_than_raising(studio, studio_pairs):
    class Broken:
        name = "broken"
        is_control = False
        def answer(self, scene, text, pair):
            raise RuntimeError("no api key")

    res = probe_run(studio_pairs[:2], lambda s: studio, Broken(), verbose=False)
    assert all(r.outcome == "error" for r in res)
    assert probe_summarise(res)["n_errors"] == len(res)


def test_stable_only_drops_the_pairs_whose_control_moved():
    """The item-level check is what makes a noisy system's row readable."""
    from sqe.bench.frame_probe import ControlResult, PairResult

    pairs = [PairResult(f"p{i}", "s", "left", {}, {}, None, None,
                        "frame_blind") for i in range(4)]
    controls = [ControlResult(f"p{i}__control", [1, 1], 1, i < 2, i < 2)
                for i in range(4)]
    out = stable_only(pairs, controls)
    assert out["n_pairs"] == 2
    assert out["n_dropped"] == 2
    assert out["outcomes"]["frame_blind"] == 2


def test_stable_only_keeps_nothing_when_no_control_is_stable():
    from sqe.bench.frame_probe import ControlResult, PairResult

    pairs = [PairResult("p0", "s", "left", {}, {}, None, None, "frame_blind")]
    controls = [ControlResult("p0__control", [1, 2], 1, False, False)]
    out = stable_only(pairs, controls)
    assert out["n_pairs"] == 0 and out["n_dropped"] == 1

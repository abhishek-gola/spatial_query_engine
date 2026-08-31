"""Benchmark schema, generation and the failure-attribution logic."""

import json
import os
import tempfile

import numpy as np
import pytest

from sqe.bench.evaluate import (ATTRIBUTION_ORDER, Outcome, _correct,
                                aggregate, attribute, render_report,
                                run_condition)
from sqe.bench.generate import anchor_salience, propose_scene, to_items
from sqe.bench.schema import BenchItem, describe_split, read_jsonl, write_jsonl
from sqe.relations.base import RelationConfig


def _item(**kw):
    base = dict(id="i1", scene_id="s", dataset="synthetic", text="the mug",
                target_ids=[3])
    base.update(kw)
    return BenchItem(**base)


def test_item_validation():
    assert _item().validate() == []
    assert _item(target_ids=[1, 2]).validate()          # two targets, not ambiguous
    assert _item(ambiguous=True, ambiguity_kind="none").validate()
    assert _item(frame="sideways").validate()
    assert _item(viewpoint_mode="position").validate()  # no position given
    assert _item(viewpoint_mode="landmark").validate()


def test_jsonl_roundtrip_and_duplicate_detection():
    items = [_item(id="a"), _item(id="b", ambiguous=True,
                                  ambiguity_kind="frame", target_ids=[1, 2])]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.jsonl")
        write_jsonl(items, p)
        back = read_jsonl(p)
        assert [i.id for i in back] == ["a", "b"]
        assert back[1].ambiguous and back[1].ambiguity_kind == "frame"
        from sqe.bench.schema import read_many
        with pytest.raises(ValueError):
            read_many([p, p])


def test_unknown_fields_are_recorded_not_dropped_silently():
    it = BenchItem.from_dict({"id": "z", "scene_id": "s", "dataset": "d",
                              "text": "t", "target_ids": [1],
                              "mystery_field": 7})
    assert "mystery_field" in it.notes


def test_correctness_semantics():
    # unambiguous: must match exactly
    assert _correct(3, [3], False)
    assert not _correct(4, [3], False)
    # ambiguous with several acceptable answers: any of them
    assert _correct(4, [3, 4], True)
    # ambiguous with NO acceptable answer: scored on the flag alone, because
    # saying "this cannot be answered" is the behaviour we want
    assert _correct(9, [], True, flagged=True)
    assert not _correct(9, [], True, flagged=False)


def test_describe_split_counts_what_the_report_needs():
    items = [_item(id="a", relation_type="projective_lateral",
                   frame="egocentric", frame_stated_in_text=True),
             _item(id="b", relation_type="vertical", frame="any"),
             _item(id="c", relation_type="ordinal", frame="intrinsic",
                   ambiguous=True, ambiguity_kind="frame", target_ids=[1, 2])]
    d = describe_split(items)
    assert d["n_items"] == 3 and d["n_ambiguous"] == 1
    assert d["n_frame_dependent"] == 2
    assert d["n_frame_stated_in_text"] == 1


def _oc(item_id, correct, rtype="projective_lateral", ambiguous=False):
    return Outcome(item_id=item_id, scene_id="s", relation_type=rtype,
                   gold_frame="intrinsic", frame_stated=False,
                   ambiguous_gold=ambiguous, difficulty="hard",
                   predicted_id=1 if not correct else 2, gold_ids=[2],
                   correct=correct, frame_used="egocentric",
                   flagged_ambiguous=False, ambiguity_kinds=[])


def test_attribution_order_is_respected():
    base = [_oc("a", False), _oc("b", False), _oc("c", False), _oc("d", False)]
    conds = {
        "gold_parse": [_oc("a", True), _oc("b", False), _oc("c", False),
                       _oc("d", False)],
        "gt_perception": [_oc("a", True), _oc("b", True), _oc("c", False),
                          _oc("d", False)],
        "oracle_frame": [_oc("a", True), _oc("b", True), _oc("c", True),
                         _oc("d", False)],
    }
    att = attribute(base, conds)
    assert att["per_item"] == {"a": "parse", "b": "perception",
                               "c": "frame_convention", "d": "geometry"}
    assert att["counts"]["frame_convention"] == 1
    assert list(att["counts"]) == list(ATTRIBUTION_ORDER)
    h = att["headline"]
    assert h["frame_dependent_failures"] == 4
    assert h["of_which_frame_errors"] == 1


def test_frame_unavailable_is_distinguished_from_wrong_frame():
    base = [_oc("a", False)]
    conds = {"oracle_frame": [_oc("a", True)]}
    att = attribute(base, conds, frame_available={"a": False})
    assert att["per_item"]["a"] == "frame_unavailable"


def test_aggregate_splits_and_ambiguity_metrics():
    oc = [_oc("a", True), _oc("b", False),
          _oc("c", True, rtype="vertical"),
          _oc("d", True, rtype="vertical")]
    oc[1].flagged_ambiguous = True
    m = aggregate(oc)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["accuracy_frame_dependent"] == pytest.approx(0.5)
    assert m["accuracy_frame_independent"] == pytest.approx(1.0)
    assert m["by_relation_type"]["vertical"]["n"] == 2
    ad = m["ambiguity_detection"]
    assert ad["fp"] == 1 and ad["tp"] == 0


def test_report_renders(studio):
    items = [_item(id="a", relation_type="projective_lateral",
                   frame="egocentric")]
    m = aggregate([_oc("a", True)])
    text = render_report(describe_split(items), {"ours": m},
                         attribute([_oc("a", True)], {}))
    assert "Accuracy by condition" in text and "ours" in text


# ------------------------------------------------------------------ generate

def test_generator_leaves_the_answer_blank(studio):
    props = propose_scene(studio, RelationConfig(), max_projective=3,
                          max_ordinal=3, max_controls=6)
    items = to_items(studio, props)
    assert items, "no proposals generated"
    for it in items:
        assert it.target_ids == []           # blind by construction
        assert it.frame == "unspecified"
        assert it.source == "generated"
        assert it.relation_type


def test_generator_hides_its_own_prediction_in_notes_only(studio):
    props = propose_scene(studio, RelationConfig(), max_projective=4,
                          max_ordinal=0, max_controls=0)
    items = to_items(studio, props)
    for it in items:
        assert not it.answers_by_frame       # gold field stays empty
    assert any("_proposal" in it.notes for it in items)


def test_generator_covers_frame_free_controls(studio):
    props = propose_scene(studio, RelationConfig())
    kinds = {p.relation_type for p in props}
    assert "projective_lateral" in kinds
    assert kinds & {"vertical", "proximity", "between", "comparative"}


def test_anchor_salience_rejects_fittings(studio):
    sockets = [o for o in studio.objects if o.canonical_label == "window"]
    assert anchor_salience("power socket", sockets) == 0.0
    assert anchor_salience("whiteboard", sockets) > \
        anchor_salience("mug", sockets)


def test_run_condition_on_the_synthetic_benchmark(studio, square):
    scenes = {"synth_studio": studio, "synth_square": square}
    items = [BenchItem(
        id="t0", scene_id="synth_studio", dataset="synthetic",
        text="the remote on the coffee table",
        target_ids=[next(o.id for o in studio.objects
                         if o.canonical_label == "remote")],
        relation="on", relation_type="vertical", frame="any",
        viewpoint_mode="position", viewpoint_position=[2.0, 2.0, 1.55])]
    oc = run_condition(items, lambda sid: scenes[sid])
    assert len(oc) == 1 and oc[0].correct
    assert oc[0].error is None


def test_run_condition_survives_a_missing_scene():
    items = [BenchItem(id="t0", scene_id="nope", dataset="synthetic",
                       text="the mug", target_ids=[1])]
    def getter(sid):
        raise FileNotFoundError("no such scene")
    oc = run_condition(items, getter)
    assert len(oc) == 1 and not oc[0].correct and oc[0].error

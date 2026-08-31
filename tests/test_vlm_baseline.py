"""The VLM baseline's selection, classification and reporting.

Everything except the API call is tested. The call itself is a thin wrapper with
an on-disk cache; the logic that matters is which queries get selected and how a
reply is turned into "the model used frame X".
"""

import json

import pytest

from sqe.bench.schema import BenchItem
from sqe.bench.vlm_baseline import (Runner, classify, parse_reply, render, run,
                                    select_queries, summarise)
from sqe.relations.base import RelationConfig


def test_classify_needs_exactly_one_matching_frame():
    fa = {"egocentric": 7, "intrinsic": 9}
    assert classify(7, fa) == (["egocentric"], "one_frame")
    assert classify(9, fa) == (["intrinsic"], "one_frame")
    # a third object is a plain error, not evidence about a frame
    assert classify(3, fa) == ([], "no_frame")
    assert classify(None, fa) == ([], "no_frame")
    # when the frames coincide the answer carries no information
    both = {"egocentric": 7, "intrinsic": 7}
    matched, outcome = classify(7, both)
    assert outcome == "several_frames" and len(matched) == 2


def test_classify_ignores_frames_with_no_answer():
    fa = {"egocentric": 7, "intrinsic": None}
    assert classify(7, fa) == (["egocentric"], "one_frame")


@pytest.mark.parametrize("text,want_id", [
    ('{"id": 12, "why": "x"}', 12),
    ('```json\n{"id": 5}\n```', 5),
    ('  {"id": null, "why": "none fit"}  ', None),
    ('I think it is id 8 because ...', 8),
    ('no idea', None),
    ('', None),
])
def test_parse_reply(text, want_id):
    got, _ = parse_reply(text)
    assert got == want_id


def test_selection_keeps_only_frame_split_queries(studio, square):
    """A query every frame agrees on says nothing about the model's convention."""
    scenes = {"synth_studio": studio, "synth_square": square}
    from sqe.selftest import find
    mug = find(studio, "mug", (4.72, 2.00, 0.67))
    items = [
        BenchItem(id="split", scene_id="synth_studio", dataset="synthetic",
                  text="the mug to the left of the bookshelf",
                  relation_type="projective_lateral", target_ids=[mug.id],
                  viewpoint_mode="position",
                  viewpoint_position=[2.0, 2.0, 1.55]),
        BenchItem(id="frame_free", scene_id="synth_studio", dataset="synthetic",
                  text="the remote on the coffee table",
                  relation_type="vertical", target_ids=[1],
                  viewpoint_mode="position",
                  viewpoint_position=[2.0, 2.0, 1.55]),
    ]
    sel = select_queries(items, lambda s: scenes[s], RelationConfig(),
                         verbose=False)
    # the frame-free one must never be selected, whatever happens to the other
    assert all(it.relation_type in ("projective_lateral", "projective_frontal")
               for it, _ in sel)
    assert "frame_free" not in [it.id for it, _ in sel]


def test_run_and_summarise_with_a_stub_model(studio):
    """End-to-end through run/summarise/render without touching an API."""
    scenes = {"synth_studio": studio}
    fa = {"egocentric": 3, "intrinsic": 2}
    it = BenchItem(id="q0", scene_id="synth_studio", dataset="synthetic",
                   text="the mug to the left of the bookshelf",
                   relation_type="projective_lateral", target_ids=[3],
                   viewpoint_mode="position",
                   viewpoint_position=[2.0, 2.0, 1.55])
    selected = [(it, {"frame_answers": fa, "policy_frame": "egocentric",
                      "viewpoint": {"eye": [2.0, 2.0, 1.55],
                                    "look_dir": [1.0, 0.0, 0.0]}})]

    class Stub(Runner):
        def __init__(self, pick):
            super().__init__("stub", cache_path=None)
            self.pick = pick
        def ask(self, prompt):
            assert "reference frame" not in prompt.lower(), \
                "the prompt must not mention frames; that would prime the answer"
            return {"raw": json.dumps({"id": self.pick, "why": "stub"})}

    rows = run(selected, lambda s: scenes[s], Stub(3), verbose=False)
    assert len(rows) == 1
    assert rows[0].outcome == "one_frame"
    assert rows[0].matched_frames == ["egocentric"]
    s = summarise(rows)
    assert s["n_informative"] == 1
    assert s["implied_frame_counts"] == {"egocentric": 1}
    assert s["agreement_with_policy"] == pytest.approx(1.0)
    text = render(s, "stub")
    assert "Frame the model's answers imply" in text
    assert "not a correctness claim" in text


def test_a_model_error_is_recorded_not_raised(studio):
    scenes = {"synth_studio": studio}
    it = BenchItem(id="q0", scene_id="synth_studio", dataset="synthetic",
                   text="the mug to the left of the bookshelf",
                   relation_type="projective_lateral", target_ids=[3],
                   viewpoint_mode="position",
                   viewpoint_position=[2.0, 2.0, 1.55])
    selected = [(it, {"frame_answers": {"egocentric": 3, "intrinsic": 2},
                      "policy_frame": "egocentric", "viewpoint": {}})]

    class Broken(Runner):
        def __init__(self):
            super().__init__("broken", cache_path=None)
        def ask(self, prompt):
            raise RuntimeError("no api key")

    rows = run(selected, lambda s: scenes[s], Broken(), verbose=False)
    assert rows[0].error and rows[0].outcome == "unparsed"
    assert summarise(rows)["n_errors"] == 1


def test_prompt_shuffles_ids_and_omits_frame_words(studio):
    """Two guards: no frame vocabulary, and no low-id preference to exploit."""
    from sqe.bench.vlm_baseline import SYSTEM, _fmt_scene
    import numpy as np
    for word in ("egocentric", "intrinsic", "allocentric", "reference frame",
                 "point of view"):
        assert word not in SYSTEM.lower()
    order = [o.id for o in studio.objects][:6]
    txt = _fmt_scene(studio, np.zeros(3), np.array([0.0, 1.0, 0.0]), order)
    assert "id=" in txt and "+Z up" in txt
    # the order given is the order rendered, so shuffling upstream has an effect
    ids = [int(l.split("id=")[1].split()[0]) for l in txt.splitlines()
           if "id=" in l]
    assert ids == order

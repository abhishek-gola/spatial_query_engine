"""Tests for the blocked trial export and the file-backed probe.

The load-bearing test here is `test_no_block_holds_both_arms_of_a_pair`. The
minimal-pair design only measures anything if the answerer cannot see the two
arms side by side; an export that put them in the same block would still produce
a clean-looking table, and the table would be worthless. So the blocking is
asserted rather than trusted.
"""

from __future__ import annotations

import json
import os

import pytest

from sqe.bench.frame_probe import (FileProbe, build_prompt, prompt_key,
                                   summarise)
from sqe.bench.frame_probe import run as probe_run
from sqe.bench.minimal_pairs import (ControlPair, generate_for_scene,
                                     make_controls)
from sqe.bench.selfprobe import audit_answers, export_trials, merge_answers
from sqe.relations.base import RelationConfig


@pytest.fixture(scope="module")
def pairs(studio):
    return generate_for_scene(studio, RelationConfig(),
                              frames=("egocentric", "intrinsic"),
                              verbose=False)


@pytest.fixture(scope="module")
def controls(studio, pairs):
    return make_controls(pairs, lambda s: studio, RelationConfig(),
                        verbose=False)


@pytest.fixture(scope="module")
def exported(tmp_path_factory, studio, pairs, controls):
    out = str(tmp_path_factory.mktemp("selfprobe"))
    man = export_trials(pairs, controls, lambda s: studio, out)
    return out, man


def test_every_pair_pins_a_concrete_observer(pairs):
    """Without this the egocentric arm is unanswerable by an outside system.

    The gold answer for the egocentric arm was computed from some observer
    position. If the pair only records the *rule* that found it, a model handed
    the scene has no way to know where it stood, and its egocentric arm is being
    marked against a viewpoint it was never told about.
    """
    assert pairs
    for p in pairs:
        assert p.viewpoint.get("position") is not None, p.id
        assert p.viewpoint.get("look_at") is not None, p.id


def test_build_prompt_refuses_an_unpinned_pair(studio, pairs):
    p = pairs[0]
    saved = dict(p.viewpoint)
    p.viewpoint = {"mode": "best_view", "position": None, "look_at": None}
    try:
        with pytest.raises(ValueError, match="concrete position"):
            build_prompt(studio, p.neutral_text, p)
    finally:
        p.viewpoint = saved


def test_both_arms_of_a_pair_see_the_same_object_list(studio, pairs):
    """Shuffling per call rather than per pair would be a silent confound."""
    for p in pairs:
        a = build_prompt(studio, p.arms[0].text, p).splitlines()
        b = build_prompt(studio, p.arms[1].text, p).splitlines()
        # everything except the trailing sentence must be identical
        assert a[:-1] == b[:-1], p.id
        assert a[-1] != b[-1], p.id


def test_no_block_holds_both_arms_of_a_pair(exported):
    out, man = exported
    with open(os.path.join(out, "keymap_private.json")) as f:
        keymap = json.load(f)
    seen = {}
    for meta in keymap.values():
        gid = meta.get("pair_id") or meta.get("control_id")
        key = (meta["block"], gid)
        assert key not in seen, f"{gid} appears twice in block {meta['block']}"
        seen[key] = True
    # and the two arms of any pair are in different blocks
    by_pair = {}
    for meta in keymap.values():
        if meta["kind"] != "pair_arm":
            continue
        by_pair.setdefault(meta["pair_id"], set()).add(meta["block"])
    for pid, blocks in by_pair.items():
        assert len(blocks) == 2, pid


def test_trial_files_leak_nothing(exported):
    """A block file must carry the sentence and the scene, and nothing else."""
    out, man = exported
    for name, b in man["blocks"].items():
        with open(b["path"]) as f:
            blob = json.load(f)
        for t in blob["trials"]:
            assert set(t) == {"trial", "prompt"}
        text = json.dumps(blob["trials"])
        for leak in ("gold", "answer_id", "expected", "pair_id", "control_id",
                     "egocentric", "intrinsic", "addressee"):
            assert leak not in text, f"{leak!r} leaked into block {name}"
    # the answer key is not in the directory the answerer is pointed at
    assert not os.path.exists(os.path.join(out, "trials",
                                           "keymap_private.json"))
    assert os.path.exists(os.path.join(out, "keymap_private.json"))


def test_file_probe_scores_like_any_other(exported, studio, pairs):
    """Answer every trial from the key and the probe must switch correctly."""
    out, man = exported
    with open(os.path.join(out, "keymap_private.json")) as f:
        keymap = json.load(f)
    answers = {k: m.get("gold") for k, m in keymap.items()
               if m["kind"] == "pair_arm"}
    # the neutral sentence still has to be answerable, or run() records an error
    for k, m in keymap.items():
        if m["kind"] == "pair_neutral":
            answers[k] = None
    path = os.path.join(out, "perfect.json")
    with open(path, "w") as f:
        json.dump({"answers": answers}, f)
    probe = FileProbe.load("oracle", path)
    res = probe_run(pairs, lambda s: studio, probe, verbose=False)
    s = summarise(res)
    assert s["rates"]["switched_correctly"] == 1.0


def test_file_probe_raises_on_a_trial_it_never_answered(studio, pairs):
    probe = FileProbe("empty", {})
    with pytest.raises(KeyError):
        probe.answer(studio, pairs[0].arms[0].text, pairs[0])


def test_merge_rejects_a_duplicated_trial(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"answers": {"deadbeef": 1}}))
    b.write_text(json.dumps({"answers": {"deadbeef": 2}}))
    with pytest.raises(ValueError, match="two blocks"):
        merge_answers([str(a), str(b)], str(tmp_path / "m.json"))


def test_audit_reports_what_is_missing(exported):
    out, man = exported
    with open(os.path.join(out, "keymap_private.json")) as f:
        keymap = json.load(f)
    keys = sorted(keymap)
    rep = audit_answers({k: 1 for k in keys[:-3]},
                        os.path.join(out, "keymap_private.json"))
    assert len(rep["missing"]) == 3
    assert rep["n_expected"] == len(keys)


def test_the_anchor_and_every_candidate_are_in_the_listing(studio, pairs):
    """A sentence naming an object absent from the list is unanswerable.

    Room-fixed objects are filtered out of the listing, and some anchors are
    room-fixed. Without an explicit re-insertion the trial asks about a heater
    that the answerer was never shown.
    """
    for p in pairs:
        prompt = build_prompt(studio, p.neutral_text, p)
        for oid in [p.anchor_id] + list(p.candidate_ids):
            assert f"id={oid} " in prompt, f"{p.id} omits #{oid}"


def test_prompt_key_is_stable(studio, pairs):
    p = pairs[0]
    k1 = prompt_key(build_prompt(studio, p.neutral_text, p))
    k2 = prompt_key(build_prompt(studio, p.neutral_text, p))
    assert k1 == k2 and len(k1) == 12

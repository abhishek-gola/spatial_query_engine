"""The blind annotation UI.

The property that matters most here is not rendering, it is *blindness*: the
payload the browser receives must not contain anything the resolver produced.
If that leaks, every label collected with this tool measures agreement with the
system rather than correctness, and the leak is invisible in the pixels. So it
gets a test that reads the payload keys directly.

The rest pins the two geometric decisions the tool makes: badge order runs
left-to-right along the *viewer's* lateral axis, and the viewpoint the annotator
judged from is written into the item on save rather than being re-resolved later.
"""

import json
import numpy as np
import pytest

from sqe.annotate_ui.server import (ITEM_KEYS, PALETTE, Queue,
                                    _anchor_and_targets, build_view, draw_plan,
                                    item_payload)
from sqe.bench.schema import BenchItem

cv2 = pytest.importorskip("cv2")


class StubSource:
    """A `FrameSource` that has poses but no video, for tests without data.

    `rgb()` returning None is a real case -- a scene cached without its capture
    -- and the thumbnail path has to survive it.
    """

    def __init__(self, scene, n=24):
        c = np.mean([o.center for o in scene.movable_objects()], axis=0)
        poses = []
        for k in range(n):
            a = 2 * np.pi * k / n
            eye = c + np.array([2.5 * np.cos(a), 2.5 * np.sin(a), 0.0])
            fwd = c - eye
            fwd[2] = 0.0
            fwd /= np.linalg.norm(fwd)
            right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
            down = np.cross(fwd, right)
            poses.append(np.block([
                [np.stack([right, down, fwd], axis=1), eye.reshape(3, 1)],
                [np.zeros((1, 3)), np.ones((1, 1))]]))
        self.poses = np.stack(poses)
        self.Ks = np.tile(np.array([[600.0, 0, 320.0],
                                    [0, 600.0, 240.0],
                                    [0, 0, 1.0]]), (n, 1, 1))
        self.names = [f"f{k:04d}" for k in range(n)]
        self.image_size = (640, 480)
        self.video_path = None

    def __len__(self):
        return len(self.poses)

    def rgb(self, i):
        return None

    def depth(self, i):
        return None


def _item(scene, text):
    return BenchItem(id="t0", scene_id="s", dataset="synthetic", text=text)


def test_badges_run_left_to_right_along_the_viewers_lateral_axis(studio):
    """Otherwise the annotator has to rotate the map in their head."""
    src = StubSource(studio)
    it = _item(studio, "the mug to the left of the laptop")
    v = build_view(it, studio, src)
    if len(v.candidates) < 2:
        pytest.skip("this room does not have two candidates for that sentence")

    look = v.look_dir - (v.look_dir @ studio.up) * studio.up
    lat = np.cross(look, studio.up)
    lat /= np.linalg.norm(lat)
    proj = [float(np.asarray(c["centre"]) @ lat) for c in v.candidates]
    assert proj == sorted(proj), "badge order is not left-to-right for the viewer"
    assert [c["badge"] for c in v.candidates] == list(
        range(1, len(v.candidates) + 1))


def test_the_anchor_is_identified_separately_from_the_targets(studio):
    anchors, targets = _anchor_and_targets(
        studio, "the mug to the left of the laptop")
    if not anchors:
        pytest.skip("no laptop/mug pair in this room")
    a_labels = {studio.by_id(i).canonical_label for i in anchors}
    t_labels = {studio.by_id(i).canonical_label for i in targets}
    assert not (a_labels & t_labels), "an object is both anchor and target"


def test_the_plan_renders_without_any_capture_imagery(studio):
    """A cached scene with no video must still be annotatable."""
    v = build_view(_item(studio, "the mug to the left of the laptop"),
                   studio, StubSource(studio))
    png = draw_plan(v, size=400)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[0] > 50


def test_no_palette_colour_is_the_answer_amber():
    """Amber means "this is the answer" in every other renderer."""
    from sqe.viz.overlay import COLOURS
    assert COLOURS["target"] not in PALETTE


# --------------------------------------------------------------------------
# blindness
# --------------------------------------------------------------------------

FORBIDDEN = ("predicted", "prediction", "answer", "resolved", "frame_scores",
             "frame_used", "score", "runner", "disagree", "ambiguity_kinds",
             "policy")


def _payload(scene, text):
    """The real payload the handler sends -- not a copy of it.

    Built through `item_payload`, which the GET handler also calls, so this
    test cannot drift out of agreement with what the browser actually receives.
    """
    it = _item(scene, text)
    q = Queue([it], [0], out_path="/dev/null", annotator="t", goal=1)
    v = build_view(it, scene, StubSource(scene))
    return item_payload(q, 0, v)


def test_the_item_payload_carries_no_resolver_output(studio):
    """The one leak that would silently invalidate every label collected."""
    payload = _payload(studio, "the mug to the left of the laptop")
    assert set(payload) == set(ITEM_KEYS), "a key was added without review"
    low = json.dumps(payload).lower()
    for word in FORBIDDEN:
        assert word not in low, f"payload mentions {word!r}: possible leak"


def test_the_annotation_server_never_builds_a_resolver():
    """Structural guarantee, not a behavioural one: grep the module."""
    import inspect

    from sqe.annotate_ui import server
    src = inspect.getsource(server)
    body = src[src.index("def make_handler"):]
    assert "Resolver(" not in body
    assert "resolve(" not in body


def test_save_pins_the_viewpoint_the_annotator_judged_from(studio, tmp_path):
    """Without this the evaluator re-resolves `best_view` and can move the eye."""
    it = _item(studio, "the mug to the left of the laptop")
    v = build_view(it, studio, StubSource(studio))
    out = tmp_path / "labels.jsonl"
    q = Queue([it], [0], out_path=str(out), annotator="t", goal=1)

    # what the POST handler does, minus HTTP
    it.target_ids = [v.candidates[0]["id"]]
    it.frame = "egocentric"
    it.viewpoint_mode = "position"
    it.viewpoint_position = [float(x) for x in v.eye]
    q.save()

    from sqe.bench.schema import read_jsonl
    back = read_jsonl(str(out))[0]
    assert back.viewpoint_mode == "position"
    assert np.allclose(back.viewpoint_position, v.eye, atol=1e-9)
    # and the pinned eye must reproduce the same egocentric frame
    spec = back.viewpoint_spec()
    assert spec.mode == "position"
    assert np.allclose(spec.position, v.eye, atol=1e-9)

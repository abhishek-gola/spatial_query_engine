"""Reference-frame semantics: the sign and handedness properties the whole
project rests on. If any of these break, every accuracy number is meaningless.
"""

import numpy as np
import pytest

from sqe.frames.cues import extract_cues
from sqe.frames.policy import (ViewpointSpec, build_frames, decide_frame,
                               relation_group)
from sqe.frames.reference_frame import (addressee_frame, egocentric_frame,
                                        egocentric_image_frame,
                                        intrinsic_frame, world_frame)
from sqe.geom.transforms import se3

UP = np.array([0.0, 0.0, 1.0])


def test_egocentric_is_left_handed_and_intrinsic_is_right_handed():
    A = np.array([4.0, 2.0, 0.7])
    front = np.array([-1.0, 0.0, 0.0])
    ego = egocentric_frame(A, A + 2.0 * front, up=UP)
    intr = intrinsic_frame(A, front, up=UP)
    # Fixing "front means the direction 'in front of' points" makes the viewer
    # frame left-handed. That flip *is* the mirror error; it must not be
    # normalised away.
    assert ego.handedness == -1
    assert intr.handedness == +1


def test_addressee_is_the_mirror_of_intrinsic():
    A = np.array([1.0, 1.0, 0.5])
    front = np.array([0.0, 1.0, 0.0])
    intr = intrinsic_frame(A, front, up=UP)
    addr = addressee_frame(A, front, up=UP)
    assert np.allclose(addr.right, -intr.right, atol=1e-12)
    assert np.allclose(addr.front, intr.front, atol=1e-12)


def test_addressee_equals_egocentric_from_the_front_axis():
    """The identity that ties the two families together."""
    A = np.array([2.0, 3.0, 0.8])
    for front in (np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]),
                  np.array([0.6, 0.8, 0.0])):
        addr = addressee_frame(A, front, up=UP)
        ego = egocentric_frame(A, A + 3.0 * front, up=UP)
        assert np.allclose(addr.right, ego.right, atol=1e-9)
        assert np.allclose(addr.front, ego.front, atol=1e-9)


def test_viewer_and_object_frames_disagree_by_a_mirror():
    """A shelf facing -x, viewed from inside the room: opposite lateral axes."""
    A = np.array([4.8, 2.0, 0.9])
    front = np.array([-1.0, 0.0, 0.0])
    ego = egocentric_frame(A, np.array([2.0, 2.0, 1.5]), up=UP)
    intr = intrinsic_frame(A, front, up=UP)
    assert np.allclose(ego.right, [0, -1, 0], atol=1e-9)
    assert np.allclose(intr.right, [0, 1, 0], atol=1e-9)
    # a point at larger y is to the viewer's left but to the shelf's right
    p = A + np.array([0.0, 0.4, 0.0])
    assert ego.coords(p[None, :])[0][0] < 0
    assert intr.coords(p[None, :])[0][0] > 0


def test_in_front_of_means_opposite_things_in_the_two_frames():
    A = np.array([1.0, 3.0, 0.8])
    front = np.array([0.0, -1.0, 0.0])         # the object faces -y
    eye = np.array([1.0, 1.0, 1.5])            # viewer is also at -y
    ego = egocentric_frame(A, eye, up=UP)
    intr = intrinsic_frame(A, front, up=UP)
    near = A + np.array([0.0, -0.5, 0.0])      # between viewer and object
    # here the two agree, because the viewer stands on the object's front side
    assert ego.coords(near[None, :])[0][1] > 0
    assert intr.coords(near[None, :])[0][1] > 0
    # move the viewer behind the object and they disagree
    ego2 = egocentric_frame(A, np.array([1.0, 5.0, 1.5]), up=UP)
    assert ego2.coords(near[None, :])[0][1] < 0
    assert intr.coords(near[None, :])[0][1] > 0


def test_bearing_frame_disagrees_with_the_projected_one_at_different_depths():
    A = np.array([4.0, 2.0, 0.7])
    # the viewer stands closer to the near candidate than to the anchor, which
    # is where an angular reading and a projected reading part company
    eye = np.array([2.0, 2.0, 1.5])
    ego = egocentric_frame(A, eye, up=UP)
    bear = egocentric_frame(A, eye, kind="egocentric_bearing", up=UP)
    far = np.array([[4.9, 1.0, 0.7]])
    near = np.array([[2.6, 1.6, 0.7]])
    pts = np.vstack([far, near])
    lat_proj = ego.coords(pts)[:, 0]
    lat_bear = bear.coords(pts)[:, 0]
    # same side, but the *order* of the two reverses -- the depth-dependent
    # disagreement that pipelines implementing only the projected version miss
    assert np.all(np.sign(lat_proj) == np.sign(lat_bear))
    assert (lat_proj[0] > lat_proj[1]) != (lat_bear[0] > lat_bear[1])


def test_image_frame_keeps_camera_roll():
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pose = se3(R, [0.0, 0.0, 1.5])            # rolled 90 degrees
    f = egocentric_image_frame(np.array([0.0, 2.0, 1.0]), pose)
    assert not np.allclose(f.up, UP, atol=1e-6)
    gravity = egocentric_frame(np.array([0.0, 2.0, 1.0]),
                               np.array([0.0, 0.0, 1.5]), up=UP)
    assert np.allclose(gravity.up, UP, atol=1e-12)


def test_intrinsic_frame_unavailable_without_a_front():
    f = intrinsic_frame(np.zeros(3), None)
    assert not f.available and "no estimated front" in f.reason


def test_frame_reanchoring_and_serialisation():
    f = intrinsic_frame(np.zeros(3), np.array([0.0, 1.0, 0.0]))
    g = f.at(np.array([1.0, 1.0, 1.0]))
    assert np.allclose(g.origin, [1, 1, 1]) and np.allclose(g.right, f.right)
    d = f.to_dict()
    assert d["handedness"] == 1 and d["kind"] == "intrinsic"


# ------------------------------------------------------------------ policy

def test_policy_asymmetry_between_lateral_and_frontal(studio):
    """left/right default to the viewer; front/behind to the object."""
    mon = next(o for o in studio.objects if o.canonical_label == "monitor")
    vp = ViewpointSpec(mode="position", position=np.array([1.2, 2.0, 1.55]))
    assert decide_frame("left", mon, studio, "", vp).chosen == "egocentric"
    assert decide_frame("right", mon, studio, "", vp).chosen == "egocentric"
    assert decide_frame("front", mon, studio, "", vp).chosen == "intrinsic"
    assert decide_frame("behind", mon, studio, "", vp).chosen == "intrinsic"


def test_explicit_cues_override_the_policy(studio):
    mon = next(o for o in studio.objects if o.canonical_label == "monitor")
    vp = ViewpointSpec(mode="position", position=np.array([1.2, 2.0, 1.55]))
    d = decide_frame("front", mon, studio, "from where I'm standing", vp)
    assert d.chosen == "egocentric" and d.explicit
    d = decide_frame("left", mon, studio, "the monitor's left", vp)
    assert d.chosen == "intrinsic" and d.explicit


def test_front_less_anchor_makes_the_intrinsic_frame_unavailable(square):
    table = next(o for o in square.objects
                 if o.canonical_label == "dining table")
    frames, _ = build_frames(square, table)
    assert not frames["intrinsic"].available
    assert "no intrinsic front" in frames["intrinsic"].reason


def test_world_frame_confidence_tracks_the_rooms_ambiguity(studio, square):
    a = world_frame(np.zeros(3), studio.room)
    b = world_frame(np.zeros(3), square.room)
    assert a.available and b.available
    # the square room has four equal walls, so its canonical forward is a
    # coin flip and the frame must say so
    assert b.confidence < a.confidence
    assert b.provenance["forward_margin"] < 0.02


def test_frame_free_relations_have_no_group():
    for r in ("on", "above", "below", "inside", "near", "next_to", "between"):
        assert relation_group(r) is None
    for r in ("left", "right", "front", "behind", "ordinal"):
        assert relation_group(r) is not None


# -------------------------------------------------------------------- cues

@pytest.mark.parametrize("text,kind,anchor", [
    ("from where I'm standing, the mug left of the laptop", "egocentric", None),
    ("the mug on the laptop's left", "intrinsic", "laptop"),
    ("what is on the left side of the room", "world", None),
    ("the object to the left of the chair in the image", "egocentric_image", None),
    ("facing the sofa, the lamp on its left", "addressee", "sofa"),
])
def test_cue_extraction(text, kind, anchor):
    cues = extract_cues(text)
    got = {c.kind: c for c in cues}
    assert kind in got, f"{text!r} -> {sorted(got)}"
    if anchor:
        assert got[kind].anchor_hint == anchor


def test_no_cue_when_the_sentence_states_no_frame():
    assert extract_cues("the mug to the left of the laptop") == []


def test_possessive_capture_stops_at_the_head_noun():
    cues = extract_cues("the mug on the laptop's left")
    poss = next(c for c in cues if c.rule == "possessive")
    assert poss.anchor_hint == "laptop"

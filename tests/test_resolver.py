"""End-to-end resolution on the synthetic rooms, plus the behaviours that make
the resolver's output honest: joint anchor scoring, weak-match reporting, and
frame-disagreement flagging.
"""

import numpy as np
import pytest

from sqe.frames.policy import ViewpointSpec
from sqe.query.parser_rules import parse
from sqe.query.resolver import MIN_ANSWER_SCORE, Resolver, resolve_text
from sqe.relations.base import RelationConfig
from sqe.selftest import find

EYE_ROOM = ViewpointSpec(mode="position", position=np.array([2.0, 2.0, 1.55]))
EYE_DESK = ViewpointSpec(mode="position", position=np.array([1.2, 2.0, 1.55]))
EYE_WALL = ViewpointSpec(mode="position", position=np.array([1.2, 3.90, 1.55]))


@pytest.fixture(scope="module")
def R(studio):
    return Resolver(studio, RelationConfig())


def test_flagship_query_and_its_alternative_reading(studio, R):
    res = R.resolve(parse("the second mug from the left on the middle shelf"),
                    EYE_ROOM)
    ego = find(studio, "mug", (4.72, 2.00, 0.67)).id
    intr = find(studio, "mug", (4.72, 1.60, 0.67)).id
    assert res.target_id == ego
    assert res.frame_used == "egocentric"
    assert res.frame_answers["intrinsic"] == intr
    assert res.ambiguity.frame_ambiguous
    # the anchor resolved to the bookshelf, on its middle board
    anchor = res.anchors[0]
    assert anchor.obj.canonical_label == "bookshelf"
    assert anchor.level_index == 1


def test_forcing_a_frame_changes_the_answer(studio, R):
    q = parse("the leftmost mug on the middle shelf")
    a = R.resolve(q, EYE_ROOM, force_frame="egocentric")
    b = R.resolve(q, EYE_ROOM, force_frame="intrinsic")
    assert a.target_id != b.target_id
    assert a.target_id == find(studio, "mug", (4.72, 2.40, 0.67)).id
    assert b.target_id == find(studio, "mug", (4.72, 1.20, 0.67)).id


def test_frame_free_relations_are_invariant(studio, R):
    for text, want in (("the remote on the coffee table",
                        ("remote", (1.75, 1.70, 0.42))),
                       ("the book on the top shelf",
                        ("book", (4.72, 1.70, 1.33))),
                       ("the trash can under the desk",
                        ("trash can", (2.05, 2.85, 0.20)))):
        gold = find(studio, want[0], want[1]).id
        ids = {R.resolve(parse(text), EYE_ROOM, force_frame=f,
                         evaluate_alternative_frames=False).target_id
               for f in (None, "egocentric", "intrinsic", "world")}
        assert ids == {gold}, text


def test_no_answer_under_a_frame_is_reported_as_no_answer(studio, R):
    """No mug is on the laptop's own right, and saying so beats guessing."""
    res = R.resolve(parse("the mug to the right of the laptop"), EYE_DESK)
    assert res.target_id == find(studio, "mug", (1.72, 3.45, 0.80)).id
    assert res.frame_answers.get("intrinsic") is None
    sub = R.resolve(parse("the mug to the right of the laptop"), EYE_DESK,
                    force_frame="intrinsic", evaluate_alternative_frames=False)
    assert sub.candidates[0].score < MIN_ANSWER_SCORE


def test_weak_match_is_flagged(studio, R):
    res = R.resolve(parse("the object behind the sofa"), EYE_ROOM)
    assert "weak_match" in res.ambiguity.kinds


def test_supporting_object_is_not_in_front_of_what_it_holds(studio, R):
    """The desk must not answer 'the object in front of the monitor'."""
    res = R.resolve(parse("the object in front of the monitor"), EYE_DESK)
    desk = find(studio, "desk", (1.20, 3.62, 0.37)).id
    assert res.target_id != desk
    assert res.target_id in {find(studio, "laptop", (1.05, 3.45, 0.86)).id,
                             find(studio, "keyboard", (1.20, 3.30, 0.77)).id}


def test_joint_anchor_scoring_picks_the_anchor_that_works(studio, R):
    """Four mugs exist; 'the mug on the coffee table' must not pick a shelf mug."""
    res = R.resolve(parse("the book on the coffee table"), EYE_ROOM)
    assert res.target_id == find(studio, "book", (1.75, 1.30, 0.43)).id


def test_missing_anchor_is_reported_not_invented(studio, R):
    res = R.resolve(parse("the mug to the left of the helicopter"), EYE_ROOM)
    assert "no_candidate" in res.ambiguity.kinds
    assert any(a.obj is None for a in res.anchors)


def test_missing_target_class_returns_nothing(studio, R):
    res = R.resolve(parse("the helicopter on the desk"), EYE_ROOM)
    assert res.target_id is None
    assert "no_candidate" in res.ambiguity.kinds


def test_explicit_frame_in_the_text_is_obeyed(studio, R):
    ego = find(studio, "mug", (4.72, 2.00, 0.67)).id
    intr = find(studio, "mug", (4.72, 1.60, 0.67)).id
    a = R.resolve(parse("from where I'm standing, the second mug from the left "
                        "on the middle shelf"), EYE_ROOM)
    b = R.resolve(parse("the second mug from the bookshelf's own left on the "
                        "middle shelf"), EYE_ROOM)
    assert a.target_id == ego and a.frame_used == "egocentric"
    assert b.target_id == intr and b.frame_used == "intrinsic"


def test_chair_whose_own_right_opposes_the_viewers(studio, R):
    res = R.resolve(parse("the trash can to the right of the office chair"),
                    EYE_WALL)
    assert res.frame_answers.get("intrinsic") == \
        find(studio, "trash can", (2.05, 2.85, 0.20)).id


def test_even_shelf_count_is_flagged(square):
    r = Resolver(square, RelationConfig())
    res = r.resolve(parse("the bottle on the middle shelf of the shelf"),
                    ViewpointSpec(mode="position",
                                  position=np.array([2.0, 2.0, 1.55])))
    assert "level_even" in res.ambiguity.kinds


def test_world_frame_undetermined_is_flagged(square):
    r = Resolver(square, RelationConfig())
    res = r.resolve(parse("the mug on the left side of the room"),
                    ViewpointSpec(mode="position",
                                  position=np.array([2.0, 0.6, 1.55])))
    assert res.ambiguity.ambiguous


def test_resolution_serialises_and_explains(studio, R):
    res = R.resolve(parse("the second mug from the left on the middle shelf"),
                    EYE_ROOM)
    d = res.to_dict()
    import json
    json.dumps(d)                       # must be JSON-clean
    assert d["target_id"] == res.target_id
    text = res.explain()
    assert "ANSWER" in text and "frame:" in text


def test_resolution_is_fast(studio, R):
    """Query time is pure geometry over cached arrays."""
    res = R.resolve(parse("the mug to the right of the laptop"), EYE_DESK)
    assert res.elapsed_ms < 250.0


def test_resolve_text_convenience(studio):
    res = resolve_text(studio, "the remote on the coffee table")
    assert res.target_id == find(studio, "remote", (1.75, 1.70, 0.42)).id

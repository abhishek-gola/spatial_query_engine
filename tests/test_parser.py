"""Query parsing. Parse errors would otherwise be scored as spatial errors."""

import pytest

from sqe.categories import label_matches, normalize_label, prior
from sqe.query.parser_rules import parse
from sqe.relations.base import canonical_relation
from sqe.relations.ordinal import parse_ordinal_word


@pytest.mark.parametrize("text,label,relation,rtype", [
    ("the mug to the left of the laptop", "mug", "left", "projective_lateral"),
    ("what is to the left of the laptop?", None, "left", "projective_lateral"),
    ("the chair in front of the whiteboard", "chair", "front",
     "projective_frontal"),
    ("the monitor behind the keyboard", "monitor", "behind",
     "projective_frontal"),
    ("the trash can under the desk", "trash can", "below", "vertical"),
    ("the red mug on the desk", "mug", "on", "vertical"),
    ("the object inside the cabinet", None, "inside", "containment"),
    ("the monitor above the keyboard", "monitor", "above", "vertical"),
    ("which monitor is closest to the window", "monitor", "near", "proximity"),
    ("the chair nearest to the whiteboard", "chair", "near", "proximity"),
])
def test_basic_shapes(text, label, relation, rtype):
    q = parse(text)
    assert q.target.label == label
    assert q.primary_relation == relation
    assert q.relation_type() == rtype


def test_the_flagship_query():
    q = parse("the second mug from the left on the middle shelf")
    assert q.target.label == "mug"
    assert q.target.ordinal is not None
    assert q.target.ordinal.index == 1
    assert q.target.ordinal.from_word == "left"
    assert q.relation_type() == "ordinal"
    c = q.target.constraints[0]
    assert c.relation == "on"
    assert c.anchors[0].level is not None and c.anchors[0].level.middle


def test_bare_left_is_a_modifier_not_a_relation():
    """'the left monitor' is the leftmost monitor, not a relation with an anchor."""
    q = parse("the left monitor")
    assert q.target.label == "monitor"
    assert q.target.ordinal is not None and q.target.ordinal.index == 0
    assert not q.target.constraints


def test_post_nominal_left_becomes_an_ordinal():
    q = parse("the chair on the right")
    assert q.target.label == "chair"
    assert q.target.ordinal is not None
    assert q.target.ordinal.from_word == "right"
    # but the real relation survives
    q2 = parse("the mug on the left of the desk")
    assert q2.primary_relation == "left"
    assert q2.target.constraints[0].anchors[0].label == "desk"


def test_from_the_left_is_not_split_as_a_relation():
    """'from the left' belongs to the ordinal; 'on' is the only relation here."""
    q = parse("the second mug from the left on the middle shelf")
    assert len(q.target.constraints) == 1
    assert q.target.constraints[0].relation == "on"


def test_counting_origin_attaches_to_the_phrase_that_counts():
    """In 'the bottle on the second shelf from the bottom' the origin is the
    shelf's, not the bottle's."""
    q = parse("the bottle on the second shelf from the bottom")
    assert q.target.label == "bottle"
    assert q.target.ordinal is None
    anchor = q.target.constraints[0].anchors[0]
    assert anchor.level is not None and anchor.level.index == 1


def test_landmark_ordinal_beats_the_viewpoint_reading():
    q = parse("the third chair from the door")
    assert q.target.ordinal is not None
    assert q.target.ordinal.index == 2
    assert q.target.ordinal.from_landmark == "door"
    assert q.viewpoint_hint is None      # not read as where the speaker stands


def test_explicit_viewpoint_survives_when_nothing_is_counted():
    q = parse("seen from the door, the chair on the left")
    assert q.viewpoint_hint == "door"


def test_cardinals_are_not_ordinals():
    q = parse("the two mugs on the shelf")
    assert q.target.ordinal is None
    assert q.expects_multiple
    assert q.relation_type() == "vertical"
    assert parse_ordinal_word("two") is None
    assert parse_ordinal_word("2") is None
    assert parse_ordinal_word("2nd") == 1


def test_possessive_rewrites_into_a_relation():
    q = parse("the mug on the laptop's left")
    assert q.primary_relation == "left"
    assert q.target.label == "mug"
    assert q.target.constraints[0].anchors[0].label == "laptop"
    assert q.frame_hint == "intrinsic"


def test_between_takes_two_anchors():
    q = parse("the mug between the laptop and the keyboard")
    c = q.target.constraints[0]
    assert c.relation == "between" and len(c.anchors) == 2
    assert [a.label for a in c.anchors] == ["laptop", "keyboard"]


def test_nested_relations_attach_low():
    q = parse("the cup on the shelf next to the door")
    assert q.target.label == "cup"
    shelf = q.target.constraints[0].anchors[0]
    assert shelf.label == "shelf"
    assert shelf.constraints[0].relation == "next_to"
    assert shelf.constraints[0].anchors[0].label == "door"
    assert any("attached low" in n for n in q.notes)


def test_level_of_object_genitive():
    q = parse("what is on the middle shelf of the bookshelf")
    anchor = q.target.constraints[0].anchors[0]
    assert anchor.label == "bookshelf"
    assert anchor.level is not None and anchor.level.middle


def test_superlative_proximity_is_flagged():
    assert parse("the trash can nearest to the door"
                 ).target.constraints[0].superlative
    assert not parse("the trash can near the door"
                     ).target.constraints[0].superlative


def test_attributes_and_superlatives():
    q = parse("the tallest object on the table")
    assert q.target.label is None and q.target.superlative == "tallest"
    q = parse("the big red mug on the desk")
    assert q.target.label == "mug"
    assert "red" in q.target.attributes and q.target.size_word == "big"


def test_intensifiers_do_not_become_the_head_noun():
    q = parse("the mug immediately right of the keyboard")
    assert q.target.label == "mug"


def test_parser_never_raises():
    for text in ("", "???", "the", "left of", "aaaa bbbb cccc",
                 "the the the of of", "42"):
        q = parse(text)
        assert q.parser == "rules"


# ------------------------------------------------------------- vocabulary

@pytest.mark.parametrize("raw,want", [
    ("Couch", "sofa"), ("the TVs", "tv"), ("book_shelf", "bookshelf"),
    ("chair/armchair", "chair"), ("Trash Cans", "trash can"),
    ("potted plant", "plant"), ("wastebasket", "trash can"),
    ("shelves", "shelf"), ("ceiling light", "ceiling lamp"),
])
def test_label_normalisation(raw, want):
    assert normalize_label(raw) == want


def test_unknown_labels_fall_through_without_a_front():
    p = prior("xyzzy widget")
    assert not p.has_front and not p.support_surface


def test_shelf_matches_bookshelf():
    """Needed so 'on the middle shelf' finds a bookshelf."""
    assert label_matches("shelf", "bookshelf") >= 0.65
    assert label_matches("mug", "cup") == 0.0


@pytest.mark.parametrize("phrase,rel", [
    ("to the left of", "left"), ("on top of", "on"), ("underneath", "below"),
    ("next to", "next_to"), ("closest to", "near"), ("in", "inside"),
    ("right hand side of", "right"), ("taller than", "taller"),
    ("nearest to", "near"), ("nonsense", None),
])
def test_relation_phrases(phrase, rel):
    assert canonical_relation(phrase) == rel


def test_vocabulary_tables_do_not_collide():
    """A synonym key that is also a category name never fires.

    Enforced at import time in sqe/categories.py; asserted here so the reason is
    visible in the test output rather than only as an ImportError.
    """
    from sqe.categories import CATEGORIES, SYNONYMS
    assert not (set(SYNONYMS) & set(CATEGORIES))
    assert set(SYNONYMS.values()) <= set(CATEGORIES)

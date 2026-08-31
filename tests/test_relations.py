"""Relation scorers, including the cases that used to be wrong."""

import numpy as np
import pytest

from sqe.frames.reference_frame import egocentric_frame, intrinsic_frame
from sqe.geom.obb import OBB, obb_from_extent_yaw
from sqe.geom.support import (detect_levels, middle_level, shelf_levels,
                              support_score)
from sqe.relations.base import RelationConfig
from sqe.relations.comparative import comparative_score, superlative_rank
from sqe.relations.ordinal import (apply_ordinal, frame_ordering_axis,
                                   support_ordering_axis)
from sqe.relations.projective import projective_score
from sqe.relations.proximity import between_score, near_score, next_to_score
from sqe.relations.vertical import above_score, inside_score, on_score

CFG = RelationConfig()
UP = np.array([0.0, 0.0, 1.0])


def box(c, e, yaw=0.0):
    return obb_from_extent_yaw(np.asarray(c, float), np.asarray(e, float), yaw)


# --------------------------------------------------------------- projective

def test_left_and_right_are_opposite_under_one_frame():
    anchor = box((0, 0, 0.5), (0.3, 0.3, 0.3))
    frame = egocentric_frame(anchor.center, np.array([0.0, -3.0, 1.5]), up=UP)
    # the viewer looks along +y, so their right is +x
    assert np.allclose(frame.right, [1, 0, 0], atol=1e-9)
    right_obj = box((0.6, 0, 0.5), (0.2, 0.2, 0.2))
    assert projective_score(right_obj, anchor, frame, "right", CFG).value > 0.7
    assert projective_score(right_obj, anchor, frame, "left", CFG).value < 0.1


def test_the_same_object_flips_side_when_the_frame_flips():
    anchor = box((0, 0, 0.5), (0.3, 0.3, 0.3))
    target = box((0.6, 0, 0.5), (0.2, 0.2, 0.2))
    ego = egocentric_frame(anchor.center, np.array([0.0, -3.0, 1.5]), up=UP)
    # an object facing the viewer has its own right on the viewer's left
    intr = intrinsic_frame(anchor.center, np.array([0.0, -1.0, 0.0]), up=UP)
    assert projective_score(target, anchor, ego, "right", CFG).value > 0.7
    assert projective_score(target, anchor, intr, "right", CFG).value < 0.1
    assert projective_score(target, anchor, intr, "left", CFG).value > 0.7


def test_separation_term_rejects_an_object_over_the_anchor():
    """A mug in the middle of a long table is not left of the table."""
    table = box((0, 0, 0.7), (3.0, 1.0, 0.05))
    mug_middle = box((-0.2, 0, 0.8), (0.1, 0.1, 0.1))
    mug_clear = box((-1.9, 0, 0.8), (0.1, 0.1, 0.1))
    frame = egocentric_frame(table.center, np.array([0.0, -4.0, 1.5]), up=UP)
    s_mid = projective_score(mug_middle, table, frame, "left", CFG)
    s_clear = projective_score(mug_clear, table, frame, "left", CFG)
    assert s_mid.components["separation"] < 0.2
    assert s_clear.components["separation"] > 0.9
    assert s_clear.value > s_mid.value


def test_depth_term_rejects_an_object_far_nearer_the_viewer():
    anchor = box((0, 3.0, 0.8), (0.3, 0.3, 0.3))
    frame = egocentric_frame(anchor.center, np.array([0.0, 0.0, 1.5]), up=UP)
    aligned = box((0.7, 3.0, 0.8), (0.2, 0.2, 0.2))
    nearer = box((0.7, 1.0, 0.8), (0.2, 0.2, 0.2))
    assert projective_score(aligned, anchor, frame, "right", CFG
                            ).components["depth"] > 0.95
    assert projective_score(nearer, anchor, frame, "right", CFG
                            ).components["depth"] < 0.2


def test_proximity_term_prefers_the_nearer_of_two_valid_candidates():
    """'in front of the sofa' means the coffee table, not the far bookshelf."""
    sofa = box((0.5, 1.5, 0.4), (0.9, 2.0, 0.8))
    frame = intrinsic_frame(sofa.center, np.array([1.0, 0.0, 0.0]), up=UP)
    near = box((1.8, 1.5, 0.2), (0.6, 1.0, 0.4))
    far = box((4.6, 1.5, 0.9), (0.4, 1.6, 1.8))
    s_near = projective_score(near, sofa, frame, "front", CFG)
    s_far = projective_score(far, sofa, frame, "front", CFG)
    assert s_near.value > s_far.value
    # but it only breaks the tie; the far object is still on the front side
    assert s_far.components["cone"] > 0.8


def test_unavailable_frame_scores_zero_with_a_reason():
    anchor = box((0, 0, 0.5), (0.3, 0.3, 0.3))
    frame = intrinsic_frame(anchor.center, None)
    s = projective_score(box((1, 0, 0.5), (0.2, 0.2, 0.2)), anchor, frame,
                         "right", CFG)
    assert s.value == 0.0 and s.notes


# ----------------------------------------------------------------- vertical

def test_on_needs_contact_and_overlap():
    table = box((0, 0, 0.375), (1.2, 0.8, 0.75))
    on_it = box((0.1, 0.1, 0.80), (0.1, 0.1, 0.1))
    beside = box((1.0, 0.0, 0.80), (0.1, 0.1, 0.1))
    high = box((0.1, 0.1, 1.60), (0.1, 0.1, 0.1))
    assert on_score(on_it, table, CFG).value > 0.8
    assert on_score(beside, table, CFG).value < 0.2
    assert on_score(high, table, CFG).value < 0.2
    assert above_score(high, table, CFG).value > 0.3


def test_contact_tolerance_has_a_graceful_tail():
    """An object a few centimetres above a surface is still on it.

    Reconstructions lose the bottom of an object, or its stand, and a hard
    cutoff here measures mesh quality rather than contact.
    """
    table = box((0, 0, 0.375), (1.2, 0.8, 0.75))
    for gap in (0.0, 0.04, 0.07):
        obj = box((0, 0, 0.75 + gap + 0.05), (0.1, 0.1, 0.1))
        assert on_score(obj, table, CFG).value > 0.9, gap
    far = box((0, 0, 0.75 + 0.30 + 0.05), (0.1, 0.1, 0.1))
    assert on_score(far, table, CFG).value < 0.2


def test_below_is_the_mirror_of_above():
    desk = box((0, 0, 0.7), (1.2, 0.7, 0.05))
    bin_ = box((0, 0, 0.2), (0.3, 0.3, 0.4))
    assert above_score(desk, bin_, CFG).value > 0.3
    from sqe.relations.vertical import below_score
    assert below_score(bin_, desk, CFG).value > 0.3


def test_inside_rejects_a_target_bigger_than_the_container():
    small = box((0, 0, 0.2), (0.1, 0.1, 0.1))
    crate = box((0, 0, 0.25), (0.5, 0.5, 0.5))
    assert inside_score(small, crate, CFG).value > 0.8
    assert inside_score(crate, small, CFG).value == 0.0


# ---------------------------------------------------------------- proximity

def test_next_to_is_stricter_than_near():
    table = box((0, 0, 0.375), (1.2, 0.8, 0.75))
    lamp_on = box((0.2, 0.2, 0.95), (0.15, 0.15, 0.4))
    chair_beside = box((0.95, 0.0, 0.45), (0.5, 0.5, 0.9))
    assert near_score(lamp_on, table, CFG).value > 0.8
    # a lamp on the table is near it but not beside it
    assert next_to_score(lamp_on, table, CFG).value < \
        next_to_score(chair_beside, table, CFG).value


def test_near_threshold_scales_with_object_size():
    """Two mugs 40 cm apart are not next to each other; two sofas are."""
    mug_a = box((0, 0, 0.8), (0.1, 0.1, 0.1))
    mug_b = box((0.5, 0, 0.8), (0.1, 0.1, 0.1))
    sofa_a = box((0, 0, 0.4), (0.9, 2.0, 0.8))
    sofa_b = box((1.35, 0, 0.4), (0.9, 2.0, 0.8))
    assert near_score(mug_b, mug_a, CFG).value < \
        near_score(sofa_b, sofa_a, CFG).value


def test_between_needs_the_corridor():
    a = box((0, 0, 0.5), (0.4, 0.4, 1.0))
    b = box((4, 0, 0.5), (0.4, 0.4, 1.0))
    mid = box((2, 0, 0.3), (0.4, 0.4, 0.6))
    off = box((2, 3.5, 0.3), (0.4, 0.4, 0.6))
    beyond = box((5, 0, 0.3), (0.4, 0.4, 0.6))
    assert between_score(mid, a, b, CFG).value > 0.7
    assert between_score(off, a, b, CFG).value < 0.2
    assert between_score(beyond, a, b, CFG).value < 0.2


# ------------------------------------------------------------------ ordinal

def _four_mugs_on_a_shelf():
    shelf = box((4.8, 2.0, 0.9), (0.4, 1.6, 1.8))
    mugs = [box((4.7, y, 0.67), (0.1, 0.1, 0.1))
            for y in (1.2, 1.6, 2.0, 2.4)]
    return shelf, mugs


def test_ordinal_sign_comes_from_the_frame_axis_not_the_support():
    shelf, mugs = _four_mugs_on_a_shelf()
    ego = egocentric_frame(shelf.center, np.array([2.0, 2.0, 1.55]), up=UP)
    intr = intrinsic_frame(shelf.center, np.array([-1.0, 0.0, 0.0]), up=UP)
    ax_e = support_ordering_axis(shelf, ego, "left")
    ax_i = support_ordering_axis(shelf, intr, "left")
    # the ordering axis is the shelf's long axis in both cases, signed opposite
    assert np.allclose(np.abs(ax_e.direction), np.abs(ax_i.direction), atol=1e-9)
    assert np.dot(ax_e.direction, ax_i.direction) < -0.99
    second_e = apply_ordinal(mugs, ax_e, 1, CFG)
    second_i = apply_ordinal(mugs, ax_i, 1, CFG)
    assert second_e.picked != second_i.picked
    assert mugs[second_e.picked[0]].center[1] == pytest.approx(2.0)
    assert mugs[second_i.picked[0]].center[1] == pytest.approx(1.6)


def test_degenerate_ordering_is_flagged():
    """Four mugs strung out along y have no order along x."""
    shelf, mugs = _four_mugs_on_a_shelf()
    # a frame whose lateral axis runs along the shelf's depth
    ego = egocentric_frame(shelf.center, np.array([4.8, -2.0, 1.55]), up=UP)
    ax = frame_ordering_axis(ego, "left")
    res = apply_ordinal(mugs, ax, 1, CFG)
    assert res.degenerate
    assert any("not ordered along it" in n for n in res.notes)


def test_ordinal_out_of_range_is_flagged():
    shelf, mugs = _four_mugs_on_a_shelf()
    ego = egocentric_frame(shelf.center, np.array([2.0, 2.0, 1.55]), up=UP)
    res = apply_ordinal(mugs, support_ordering_axis(shelf, ego, "left"), 9, CFG)
    assert res.out_of_range and not res.picked


def test_ordinal_from_the_right_reverses_the_order():
    shelf, mugs = _four_mugs_on_a_shelf()
    ego = egocentric_frame(shelf.center, np.array([2.0, 2.0, 1.55]), up=UP)
    left0 = apply_ordinal(mugs, support_ordering_axis(shelf, ego, "left"), 0, CFG)
    right0 = apply_ordinal(mugs, support_ordering_axis(shelf, ego, "right"), 0,
                           CFG)
    assert left0.picked != right0.picked


# -------------------------------------------------------------------- levels

def test_even_shelf_count_has_no_middle():
    from sqe.geom.support import Level
    levels = [Level(z=z, area=0.4, support=500, index=i)
              for i, z in enumerate((0.0, 0.45, 0.9, 1.35))]
    picked, amb = middle_level(levels)
    assert amb and len(picked) == 2
    odd = [Level(z=z, area=0.4, support=500, index=i)
           for i, z in enumerate((0.0, 0.6, 1.2))]
    picked, amb = middle_level(odd)
    assert not amb and len(picked) == 1 and picked[0].z == pytest.approx(0.6)


def test_a_single_surface_is_not_a_shelf_system():
    from sqe.geom.support import Level
    one = [Level(z=0.75, area=0.8, support=900, index=0)]
    assert shelf_levels(one) == []


# --------------------------------------------------------------- comparative

def test_comparatives_and_superlatives():
    tall = box((0, 0, 0.9), (0.3, 0.3, 1.8))
    short = box((1, 0, 0.2), (0.3, 0.3, 0.4))
    assert comparative_score(tall, short, "taller", CFG).value > 0.9
    assert comparative_score(short, tall, "taller", CFG).value == 0.0
    order, vals, metric, tie = superlative_rank([short, tall], "tallest")
    assert order[0] == 1 and metric == "height" and not tie
    same = [box((0, 0, 0.5), (0.3, 0.3, 1.0)),
            box((1, 0, 0.5), (0.3, 0.3, 1.01))]
    assert superlative_rank(same, "tallest")[3] is True

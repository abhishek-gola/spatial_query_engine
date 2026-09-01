"""Tests for the feed-composed GIF.

The crop is the part worth testing. It is easy to write a crop that looks right
on the one frame it was tuned on and silently pushes a candidate out of shot on
the next, and a figure that omits one of the two answers cannot make the point
the figure exists to make.
"""

from __future__ import annotations

import numpy as np
import pytest

from sqe.viz.poster import ZOOM, _fit_scale, crop_rect


def _box(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def _inside(rect, box):
    x, y, w, h = rect
    return (box[:, 0].min() >= x and box[:, 0].max() <= x + w
            and box[:, 1].min() >= y and box[:, 1].max() <= y + h)


def test_crop_keeps_every_candidate_in_shot():
    size = (1920, 1440)
    boxes = [_box(600, 300, 780, 420), _box(900, 260, 1120, 400)]
    rect = crop_rect(boxes, size)
    for b in boxes:
        assert _inside(rect, b)


def test_crop_holds_its_aspect_and_stays_inside_the_image():
    size = (1920, 1440)
    for boxes in ([_box(10, 10, 60, 40)],
                  [_box(1850, 1380, 1910, 1430)],
                  [_box(100, 100, 1800, 1300)]):
        x, y, w, h = crop_rect(boxes, size, aspect=4.0 / 3.0)
        assert 0 <= x and 0 <= y and x + w <= size[0] and y + h <= size[1]
        assert abs((w / h) - 4.0 / 3.0) < 0.02


def test_crop_widens_rather_than_clipping_content_that_does_not_fit():
    size = (1920, 1440)
    boxes = [_box(50, 50, 1870, 900)]
    rect = crop_rect(boxes, size, zoom=1.2)
    assert _inside(rect, boxes[0])


def test_headroom_moves_the_window_up_but_never_drops_the_content():
    """Room for the caption band is granted out of slack, not out of content."""
    size = (1920, 1440)
    boxes = [_box(700, 500, 900, 620)]
    plain = crop_rect(boxes, size, headroom=0.0)
    lifted = crop_rect(boxes, size, headroom=200.0)
    assert lifted[1] <= plain[1]
    for r in (plain, lifted):
        assert _inside(r, boxes[0])
    # an absurd request must not push the candidate out of the bottom
    silly = crop_rect(boxes, size, headroom=10_000.0)
    assert _inside(silly, boxes[0])


def test_zoom_is_tight_enough_to_change_most_of_the_frame():
    """The whole point of cropping is that little of the image is static."""
    assert ZOOM <= 2.5


def test_fit_scale_shrinks_a_long_sentence_to_one_line():
    import cv2

    long = '"the second mug from the left on the middle shelf of the cabinet"'
    s = _fit_scale(long, 1000, 1.55, 3)
    assert s <= 1.55
    assert cv2.getTextSize(long, cv2.FONT_HERSHEY_SIMPLEX, s, 3)[0][0] <= 1000
    # a short one keeps the requested size
    assert _fit_scale('"the mug"', 1000, 1.55, 3) == pytest.approx(1.55)


def test_fit_scale_has_a_floor_rather_than_vanishing():
    s = _fit_scale("x" * 400, 300, 1.55, 3, floor=0.7)
    assert s == pytest.approx(0.7)

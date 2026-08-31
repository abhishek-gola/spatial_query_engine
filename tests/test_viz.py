"""Overlay rendering. Checks the projection maths, not the pixels.

The renderers exist to catch pose and intrinsic mistakes by eye, so the tests
here pin the one thing a test can check: that a box placed in front of a known
camera projects to where it should, and that a box behind the camera is rejected.
"""

import os

import numpy as np
import pytest

from sqe.geom.obb import obb_from_extent_yaw
from sqe.geom.transforms import intrinsics_matrix, se3
from sqe.viz.overlay import (BOX_EDGES, box_screen_area, is_occluded,
                             project_box, visible_objects)

cv2 = pytest.importorskip("cv2")

K = intrinsics_matrix(600.0, 600.0, 320.0, 240.0)
SIZE = (640, 480)


def _camera_at_origin_looking_along_y():
    """OpenCV camera: +x right, +y down, +z forward. Look along world +y."""
    R = np.stack([np.array([1.0, 0.0, 0.0]),     # camera x -> world +x
                  np.array([0.0, 0.0, -1.0]),    # camera y (down) -> world -z
                  np.array([0.0, 1.0, 0.0])],    # camera z (fwd) -> world +y
                 axis=1)
    return se3(R, [0.0, 0.0, 1.5])


def test_a_box_dead_ahead_projects_to_the_image_centre():
    pose = _camera_at_origin_looking_along_y()
    box = obb_from_extent_yaw((0.0, 3.0, 1.5), (0.4, 0.4, 0.4), 0.0)
    uv, z, front, on = project_box(box, K, pose, SIZE)
    assert front and on
    assert uv[:, 0].mean() == pytest.approx(320.0, abs=1.0)
    assert uv[:, 1].mean() == pytest.approx(240.0, abs=1.0)
    assert np.all(z > 0)


def test_a_box_to_the_world_right_projects_right_of_centre():
    pose = _camera_at_origin_looking_along_y()
    box = obb_from_extent_yaw((1.0, 3.0, 1.5), (0.2, 0.2, 0.2), 0.0)
    uv, _, _, _ = project_box(box, K, pose, SIZE)
    assert uv[:, 0].mean() > 340.0


def test_a_box_higher_up_projects_above_centre():
    pose = _camera_at_origin_looking_along_y()
    box = obb_from_extent_yaw((0.0, 3.0, 2.3), (0.2, 0.2, 0.2), 0.0)
    uv, _, _, _ = project_box(box, K, pose, SIZE)
    assert uv[:, 1].mean() < 220.0


def test_a_box_behind_the_camera_is_rejected():
    pose = _camera_at_origin_looking_along_y()
    box = obb_from_egocentric = obb_from_extent_yaw((0.0, -3.0, 1.5),
                                                   (0.4, 0.4, 0.4), 0.0)
    _, _, front, _ = project_box(box, K, pose, SIZE)
    assert not front


def test_box_edges_cover_a_cube():
    assert len(BOX_EDGES) == 12
    seen = set()
    for a, b in BOX_EDGES:
        seen.add(a)
        seen.add(b)
    assert seen == set(range(8))


def test_screen_area_grows_as_the_object_nears():
    pose = _camera_at_origin_looking_along_y()
    near = obb_from_extent_yaw((0.0, 1.5, 1.5), (0.4, 0.4, 0.4), 0.0)
    far = obb_from_extent_yaw((0.0, 6.0, 1.5), (0.4, 0.4, 0.4), 0.0)
    a_near = box_screen_area(project_box(near, K, pose, SIZE)[0])
    a_far = box_screen_area(project_box(far, K, pose, SIZE)[0])
    assert a_near > 4 * a_far


def test_occlusion_uses_the_depth_map():
    pose = _camera_at_origin_looking_along_y()
    box = obb_from_extent_yaw((0.0, 4.0, 1.5), (0.3, 0.3, 0.3), 0.0)
    # a depth map saying everything is 1 m away: the box at 4 m is hidden
    close = np.full((192, 256), 1.0, np.float32)
    assert is_occluded(box, K, pose, close)
    # a depth map agreeing with the box's own distance: visible
    agree = np.full((192, 256), 4.0, np.float32)
    assert not is_occluded(box, K, pose, agree)
    assert not is_occluded(box, K, pose, None)


def test_topdown_and_pointcloud_renderers_write_files(studio, tmp_path):
    from sqe.viz.overlay import render_pointcloud_view, render_topdown
    target = studio.objects[1]
    p = render_topdown(studio, str(tmp_path / "td.png"),
                       highlight={target.id: "target"})
    assert os.path.exists(p) and os.path.getsize(p) > 2000
    q = render_pointcloud_view(studio, str(tmp_path / "view.png"),
                               highlight={target.id: "target"})
    assert os.path.exists(q) and os.path.getsize(q) > 2000
    img = cv2.imread(q)
    assert img is not None and img.shape[0] > 100


def test_visible_objects_filters_by_distance(studio):
    pose = _camera_at_origin_looking_along_y()
    near = visible_objects(studio, K, pose, SIZE, max_distance=1.0)
    far = visible_objects(studio, K, pose, SIZE, max_distance=20.0)
    assert len(far) >= len(near)

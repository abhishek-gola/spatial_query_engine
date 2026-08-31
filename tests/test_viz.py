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
    far = visible_objects(studio, K, pose, SIZE, max_distance=20.0,
                          max_objects=None)
    assert len(far) >= len(near)
    # rows are (object, corners, distance, visible_fraction), far first so a
    # caller painting in order draws near objects last
    for row in far:
        assert len(row) == 4
    dists = [r[2] for r in far]
    assert dists == sorted(dists, reverse=True)


def test_visible_objects_caps_the_count(studio):
    pose = _camera_at_origin_looking_along_y()
    capped = visible_objects(studio, K, pose, SIZE, max_distance=20.0,
                             max_objects=3)
    assert len(capped) <= 3


def test_a_box_enclosing_the_camera_is_dropped(studio):
    """Its wireframe is four lines sprawling off every edge of the frame."""
    from sqe.geom.obb import obb_from_extent_yaw
    from sqe.scenegraph.objects import Object3D, Scene
    pose = _camera_at_origin_looking_along_y()
    eye = pose[:3, 3]
    big = Object3D(id=0, label="table",
                   obb=obb_from_extent_yaw(eye, (4.0, 4.0, 4.0), 0.0))
    tiny = Object3D(id=1, label="mug",
                    obb=obb_from_extent_yaw((0.0, 3.0, 1.5),
                                            (0.3, 0.3, 0.3), 0.0))
    sc = Scene(scene_id="t", objects=[big, tiny])
    rows = visible_objects(sc, K, pose, SIZE, max_distance=20.0)
    assert [o.id for o, _, _, _ in rows] == [1]
    # `skip_enclosing` drops it before projection; the all-corners-in-front
    # test would have caught this particular one anyway, since a box around the
    # camera has corners behind it. The explicit check matters for a box that
    # merely grazes the camera.
    kept = visible_objects(sc, K, pose, SIZE, max_distance=20.0,
                           skip_enclosing=False)
    assert 0 not in [o.id for o, _, _, _ in kept]


def test_hidden_line_removal_splits_an_occluded_edge():
    """A box behind a near wall must come back with almost nothing visible."""
    from sqe.viz.overlay import (DepthBuffer, box_visible_fraction,
                                 edge_visibility)
    pose = _camera_at_origin_looking_along_y()
    box = obb_from_extent_yaw((0.0, 4.0, 1.5), (0.6, 0.6, 0.6), 0.0)
    wall = DepthBuffer.from_sensor(np.full((192, 256), 1.0, np.float32), SIZE)
    clear = DepthBuffer.from_sensor(np.full((192, 256), 9.0, np.float32), SIZE)
    assert box_visible_fraction(box, K, pose, wall) < 0.02
    assert box_visible_fraction(box, K, pose, clear) > 0.98
    assert box_visible_fraction(box, K, pose, None) > 0.98


def test_a_box_is_occluded_by_its_own_front_surface():
    """The back edges of a box must not be drawn as if in front of the object.

    This is what made a full wireframe look like it was floating: a real
    wireframe of a solid shows only the edges the object itself does not hide.
    """
    from sqe.viz.overlay import DepthBuffer, box_visible_fraction
    pose = _camera_at_origin_looking_along_y()
    box = obb_from_extent_yaw((0.0, 4.0, 1.5), (0.8, 0.8, 0.8), 0.0)
    # the object's front surface sits at y = 3.6, so anything deeper is hidden
    surface = DepthBuffer.from_sensor(np.full((192, 256), 3.6, np.float32), SIZE)
    frac = box_visible_fraction(box, K, pose, surface)
    assert 0.1 < frac < 0.75, frac


def test_partial_occlusion_lands_between_the_extremes():
    from sqe.viz.overlay import DepthBuffer, box_visible_fraction
    pose = _camera_at_origin_looking_along_y()
    box = obb_from_extent_yaw((0.0, 4.0, 1.5), (1.6, 0.4, 1.6), 0.0)
    d = np.full((192, 256), 9.0, np.float32)
    d[:, :128] = 1.0                      # a near wall over the left half
    half = DepthBuffer.from_sensor(d, SIZE)
    frac = box_visible_fraction(box, K, pose, half)
    assert 0.2 < frac < 0.85, frac


def test_point_splat_depth_buffer_closes_its_holes(studio):
    """A sparse z-buffer full of gaps would let occluded edges show through."""
    from sqe.viz.overlay import DepthBuffer
    pose = _camera_at_origin_looking_along_y()
    db = DepthBuffer.from_points(studio.background, K, pose, SIZE)
    assert db.source == "points"
    covered = (db.depth > 0.05).mean()
    assert covered > 0.15, covered


def test_rendered_camera_is_not_upside_down_or_mirrored(studio, tmp_path):
    """Pins image orientation.

    `up x f` and `f x up` are both valid rotations with determinant +1, so a
    handedness check does not catch swapping them -- but one of them renders the
    image rotated 180 degrees, and a left/right judgement made on a flipped
    picture is worse than no picture at all. This asserts the actual convention:
    world up projects above the centre, and the viewer's right projects to the
    right of it.
    """
    from sqe.geom.transforms import (intrinsics_matrix, normalize, project,
                                     se3, se3_inverse, transform_points)
    eye = np.array([2.0, 2.0, 1.55])
    look = np.array([4.78, 2.0, 1.0])

    # rebuild the basis exactly as render_pointcloud_view does
    f = normalize(look - eye)
    right = normalize(np.cross(f, np.array([0.0, 0.0, 1.0])))
    down = normalize(np.cross(f, right))
    pose = se3(np.stack([right, down, f], axis=1), eye)
    Kv = intrinsics_matrix(600.0, 600.0, 320.0, 240.0)

    def px(world):
        cam = transform_points(se3_inverse(pose), np.atleast_2d(world))
        uv, z, _ = project(cam, Kv)
        return uv[0], float(z[0])

    mid, _ = px(look)
    higher, _ = px(look + np.array([0.0, 0.0, 0.5]))
    assert higher[1] < mid[1], "world up must project ABOVE the centre"

    # the viewer's right, by this repo's frame convention
    from sqe.frames.reference_frame import egocentric_frame
    frame = egocentric_frame(look, eye, up=np.array([0.0, 0.0, 1.0]))
    to_right, _ = px(look + 0.5 * frame.right)
    assert to_right[0] > mid[0], "the viewer's right must project RIGHT of centre"


def test_synthetic_trajectory_cameras_are_level(studio):
    """A synthetic capture's cameras must not be upside down."""
    from sqe.geom.transforms import camera_down
    traj = studio.trajectory
    assert traj is not None and len(traj)
    downs = np.array([camera_down(traj.poses[i]) for i in range(len(traj))])
    # camera +y is image-down, so it must point broadly along world -z
    assert float(downs[:, 2].max()) < 0.0, downs[:, 2].max()


def test_a_box_entirely_off_screen_is_not_visible():
    """The bug this pins was silently inflating every visibility number.

    `on_screen` used to allow a full image-width of margin on each side, and
    `box_visible_fraction` counted out-of-bounds samples as visible, because the
    depth lookup returns "no measurement" there and nothing was found in the
    way. A door projecting to x in [-4947, -1345] on a 1920-wide frame scored
    on_screen=True and visible=1.00, which made the joint-view chooser pick
    frames containing none of the objects.
    """
    from sqe.viz.overlay import DepthBuffer, box_visible_fraction
    pose = _camera_at_origin_looking_along_y()
    # far off to the world -x side: behind the left edge of the frame
    off = obb_from_extent_yaw((-12.0, 3.0, 1.5), (0.6, 0.6, 0.6), 0.0)
    uv, _, front, on = project_box(off, K, pose, SIZE)
    assert front, "it is still in front of the camera"
    assert not on, f"but it is off screen; uv x range {uv[:, 0].min():.0f}..{uv[:, 0].max():.0f}"
    clear = DepthBuffer.from_sensor(np.full((192, 256), 99.0, np.float32), SIZE)
    assert box_visible_fraction(off, K, pose, clear, image_size=SIZE) == 0.0
    # and with no depth buffer at all, an explicit image size still rules it out
    assert box_visible_fraction(off, K, pose, None, image_size=SIZE) == 0.0
    # a box in the middle of the frame is visible
    inframe = obb_from_extent_yaw((0.0, 3.0, 1.5), (0.6, 0.6, 0.6), 0.0)
    assert box_visible_fraction(inframe, K, pose, clear, image_size=SIZE) > 0.9


def test_partially_in_frame_scores_between():
    from sqe.viz.overlay import DepthBuffer, box_visible_fraction
    pose = _camera_at_origin_looking_along_y()
    clear = DepthBuffer.from_sensor(np.full((192, 256), 99.0, np.float32), SIZE)
    # straddling the left edge
    straddle = obb_from_extent_yaw((-1.6, 2.0, 1.5), (1.6, 0.4, 1.6), 0.0)
    f = box_visible_fraction(straddle, K, pose, clear, image_size=SIZE)
    assert 0.02 < f < 0.95, f


def test_best_joint_view_returns_minus_one_when_nothing_shows_all():
    """Two objects at opposite ends of a room cannot share one 55-degree frame.

    It used to fall back to the best geometric candidate, which meant returning
    a frame where some objects were invisible.
    """
    from sqe.geom.transforms import se3
    from sqe.viz.overlay import FrameSource, best_joint_view
    poses = np.stack([_camera_at_origin_looking_along_y()])
    src = FrameSource(poses=poses, Ks=np.stack([K]), names=["f0"],
                      image_size=SIZE, video_path=None, depth_reader=None)
    near = obb_from_extent_yaw((0.0, 3.0, 1.5), (0.4, 0.4, 0.4), 0.0)
    far_left = obb_from_extent_yaw((-14.0, 3.0, 1.5), (0.4, 0.4, 0.4), 0.0)
    assert best_joint_view(src, [near], np.array([0.0, 0.0, 1.0]), stride=1) == 0
    assert best_joint_view(src, [near, far_left], np.array([0.0, 0.0, 1.0]),
                           stride=1) == -1


def test_viewer_refuses_to_resolve_while_annotating(studio, tmp_path):
    """Blindness must be structural, not a convention the annotator can break.

    With --items set, the resolution endpoints return 403. Previously they served
    the answer, the runner-up and the ambiguity flags, so an annotator could
    resolve first and label second without meaning to -- while the README claimed
    the tool was blind.
    """
    import json
    import threading
    import time
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    from sqe.relations.base import RelationConfig
    from sqe.viewer.server import make_handler

    items = str(tmp_path / "items.jsonl")
    scenes = {"synth_studio": studio}

    def start(items_path, port):
        h = make_handler(scenes, RelationConfig(), items_path)
        srv = ThreadingHTTPServer(("127.0.0.1", port), h)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        time.sleep(0.3)
        return srv

    def post(port, path, body):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    body = {"scene_id": "synth_studio",
            "text": "the remote on the coffee table"}

    # annotating: refused
    srv = start(items, 8811)
    try:
        code, out = post(8811, "/api/query", body)
        assert code == 403, out
        assert "disabled while annotating" in out["error"]
        code, out = post(8811, "/api/frames",
                         {"scene_id": "synth_studio", "object_id": 0})
        assert code == 403
        with urllib.request.urlopen("http://127.0.0.1:8811/api/scenes",
                                    timeout=10) as r:
            info = json.loads(r.read())
        assert info["annotating"] and info["resolution_disabled"]
    finally:
        srv.shutdown()

    # not annotating: resolution works
    srv = start(None, 8812)
    try:
        code, out = post(8812, "/api/query", body)
        assert code == 200
        assert out["target_id"] is not None
        with urllib.request.urlopen("http://127.0.0.1:8812/api/scenes",
                                    timeout=10) as r:
            info = json.loads(r.read())
        assert not info["annotating"] and not info["resolution_disabled"]
    finally:
        srv.shutdown()

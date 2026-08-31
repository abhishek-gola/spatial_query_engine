"""Geometry primitives. These are the layer everything else trusts."""

import numpy as np
import pytest

from sqe.geom.obb import Interval, OBB, fit_obb, obb_gap
from sqe.geom.pointcloud import (cloud_gap, dbscan, fit_plane_ransac,
                                 occupied_area_2d, voxel_downsample)
from sqe.geom.transforms import (basis_from_forward_up, quat_to_rot,
                                 rot_about_up, rot_to_quat, rotation_between,
                                 se3, se3_inverse, transform_points)


@pytest.mark.parametrize("yaw_deg", [0, 17, 30, 45, 80, 123, -40, 179])
def test_fit_obb_recovers_a_known_box(yaw_deg):
    rng = np.random.default_rng(0)
    half = np.array([1.0, 0.3, 0.4])
    R = rot_about_up(np.deg2rad(yaw_deg))
    ctr = np.array([3.0, 1.0, 0.4])
    pts = ((rng.random((8000, 3)) * 2 - 1) * half) @ R.T + ctr
    b = fit_obb(pts)
    assert np.allclose(b.extent, 2 * half, atol=0.01)
    assert np.linalg.norm(b.center - ctr) < 0.01
    # local x must be the longer horizontal side, which fixes the 90-degree
    # ambiguity deterministically
    assert b.half[0] >= b.half[1]
    err = (np.rad2deg(b.yaw) - yaw_deg + 90) % 180 - 90
    assert abs(err) < 0.5


def test_fit_obb_degenerate_inputs():
    b = fit_obb(np.array([[1.0, 2.0, 3.0]]))
    assert np.allclose(b.center, [1, 2, 3])
    line = np.stack([np.linspace(0, 1, 50), np.linspace(0, 1, 50),
                     np.zeros(50)], axis=1)
    b = fit_obb(line)
    assert b.extent[0] == pytest.approx(np.sqrt(2), abs=0.01)
    with pytest.raises(ValueError):
        fit_obb(np.zeros((0, 3)))


def test_obb_interval_is_exact():
    b = OBB(np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.25, 0.1]),
            rot_about_up(0.7))
    for axis in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                 np.array([0.6, 0.8, 0.0])):
        iv = b.interval(axis)
        proj = b.corners() @ axis
        assert iv.lo == pytest.approx(proj.min(), abs=1e-9)
        assert iv.hi == pytest.approx(proj.max(), abs=1e-9)


def test_interval_algebra():
    a, b = Interval(0, 1), Interval(0.5, 2)
    assert a.overlap(b) == pytest.approx(0.5)
    assert a.iou(b) == pytest.approx(0.25)
    assert a.gap_to(b) == pytest.approx(-0.5)
    assert Interval(0, 1).gap_to(Interval(1.5, 2)) == pytest.approx(0.5)
    assert a.fraction_beyond(0.5, +1) == pytest.approx(0.5)
    assert a.fraction_beyond(0.5, -1) == pytest.approx(0.5)
    assert a.contained_fraction(Interval(-1, 2)) == pytest.approx(1.0)


def test_basis_from_forward_up_is_right_handed():
    up = np.array([0.0, 0.0, 1.0])
    B = basis_from_forward_up(np.array([0.3, 1.0, 0.7]), up)
    r, f, u = B[:, 0], B[:, 1], B[:, 2]
    assert np.allclose(np.cross(r, f), u, atol=1e-9)
    assert np.linalg.det(B) == pytest.approx(1.0, abs=1e-9)
    # forward is flattened into the gravity plane
    assert f[2] == pytest.approx(0.0, abs=1e-9)


def test_basis_rejects_a_vertical_forward():
    with pytest.raises(ValueError):
        basis_from_forward_up(np.array([0.0, 0.0, 1.0]),
                              np.array([0.0, 0.0, 1.0]))


def test_se3_roundtrip_and_quaternions():
    R = rot_about_up(0.7) @ rotation_between([0, 0, 1.0], [0.1, 0.2, 0.97])
    T = se3(R, [1, 2, 3])
    pts = np.random.default_rng(1).random((20, 3))
    back = transform_points(se3_inverse(T), transform_points(T, pts))
    assert np.allclose(back, pts, atol=1e-9)
    assert np.allclose(quat_to_rot(rot_to_quat(R)), R, atol=1e-9)


def test_rotation_between_handles_antiparallel():
    R = rotation_between([0, 0, 1.0], [0, 0, -1.0])
    assert np.allclose(R @ np.array([0, 0, 1.0]), [0, 0, -1.0], atol=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)


def test_cloud_gap_is_surface_distance_not_centroid():
    rng = np.random.default_rng(0)
    # a long slab and a small blob touching one end: centroids are far apart,
    # surfaces are not
    slab = np.c_[rng.random(4000) * 3.0, rng.random(4000) * 0.1,
                 rng.random(4000) * 0.05]
    blob = np.c_[rng.random(500) * 0.1 + 3.02, rng.random(500) * 0.1,
                 rng.random(500) * 0.05]
    assert cloud_gap(slab, blob) < 0.05
    assert np.linalg.norm(slab.mean(0) - blob.mean(0)) > 1.4


def test_voxel_downsample_preserves_labels():
    rng = np.random.default_rng(0)
    pts = rng.random((5000, 3))
    labels = np.arange(5000)
    down, inverse, attrs = voxel_downsample(pts, 0.1, attrs={"l": labels})
    assert len(down) == int(inverse.max()) + 1
    assert attrs["l"].shape[0] == len(down)
    assert len(inverse) == 5000


def test_occupied_area_sees_a_hole_that_hull_area_cannot():
    from sqe.geom.pointcloud import hull_area_2d
    rng = np.random.default_rng(0)
    xy = rng.random((20000, 2)) * np.array([2.0, 1.0])
    solid = occupied_area_2d(xy, 0.05)
    holed = occupied_area_2d(
        xy[~((xy[:, 0] > 0.8) & (xy[:, 0] < 1.2) & (xy[:, 1] < 0.5))], 0.05)
    assert holed < solid - 0.15
    # the convex hull is blind to the hole, which is why walls are not measured
    # that way
    assert hull_area_2d(xy) == pytest.approx(
        hull_area_2d(xy[~((xy[:, 0] > 0.8) & (xy[:, 0] < 1.2)
                          & (xy[:, 1] < 0.5))]), rel=0.05)


def test_plane_and_dbscan():
    rng = np.random.default_rng(0)
    plane = np.c_[rng.random((3000, 2)) * 2, np.full(3000, 1.5)
                  + rng.normal(0, 0.003, 3000)]
    P = fit_plane_ransac(np.vstack([plane, rng.random((400, 3)) * 2]),
                         thresh=0.02, normal_prior=np.array([0, 0, 1.0]))
    assert abs(abs(float(P.normal[2])) - 1.0) < 0.02
    assert abs(abs(P.offset) - 1.5) < 0.02

    a = rng.random((400, 3)) * 0.2
    b = rng.random((400, 3)) * 0.2 + np.array([1.0, 0, 0])
    labels = dbscan(np.vstack([a, b]), 0.06, 8)
    assert labels.max() + 1 == 2


def test_obb_gap_is_a_lower_bound_on_surface_distance():
    a = OBB(np.zeros(3), np.array([0.5, 0.5, 0.5]))
    b = OBB(np.array([2.0, 0, 0]), np.array([0.5, 0.5, 0.5]))
    assert obb_gap(a, b) == pytest.approx(1.0, abs=1e-9)
    assert obb_gap(a, a) <= 0.0

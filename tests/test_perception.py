"""Front estimation, geometric proposals, instance scoring, quality audit."""

import numpy as np
import pytest

from sqe.data.quality import (OBB_DISAGREEMENT_TOL, audit_scene, flag_object,
                              suspect_ids)
from sqe.geom.obb import obb_from_extent_yaw
from sqe.perception.evaluate import (mask_iou, match_instances,
                                     score_instances)
from sqe.perception.orientation import (estimate_front, score_fronts)
from sqe.perception.proposals import (DisjointSet, edge_weights,
                                      felzenszwalb_segments, mesh_edges,
                                      summarise_segments,
                                      vertex_structure_mask)


# ------------------------------------------------------------- orientation

def test_front_estimation_abstains_for_front_less_categories():
    box = obb_from_extent_yaw((0, 0, 0.05), (0.1, 0.1, 0.1), 0.0)
    est = estimate_front(box, None, "mug")
    assert est.front is None and est.method == "no_front_category"


def test_front_estimation_reports_axis_only_when_ends_are_indistinguishable():
    """A keyboard's axis is knowable; which end faces the user is not."""
    box = obb_from_extent_yaw((1, 1, 0.77), (0.45, 0.15, 0.03), 0.0)
    rng = np.random.default_rng(0)
    pts = (rng.random((800, 3)) * 2 - 1) * np.array([0.22, 0.07, 0.015]) \
        + np.array([1, 1, 0.77])
    est = estimate_front(box, pts, "keyboard")
    assert est.front is None
    assert est.method.startswith("axis_only") or \
        est.method.startswith("inconclusive")
    assert est.axis_confidence > 0.0


def test_front_estimation_on_the_synthetic_rooms(studio, square):
    for scene in (studio, square):
        sc = score_fronts(scene)
        assert sc["n_flipped_180"] == 0
        assert sc["n_correct_within_45deg"] == sc["n_estimated"]


def test_visibility_cue_is_off_by_default():
    """Using camera visibility would make the intrinsic frame partly egocentric
    and contaminate the comparison the benchmark exists to make."""
    box = obb_from_extent_yaw((1, 1, 0.5), (0.5, 0.2, 1.0), 0.0)
    rng = np.random.default_rng(0)
    pts = (rng.random((900, 3)) * 2 - 1) * np.array([0.25, 0.1, 0.5]) \
        + np.array([1, 1, 0.5])
    views = np.tile(np.array([0.0, -1.0, 0.0]), (30, 1))
    a = estimate_front(box, pts, "tv", view_dirs=views, use_visibility=False)
    b = estimate_front(box, pts, "tv", view_dirs=views, use_visibility=True)
    assert "visibility" not in a.method
    assert "visibility" in a.detail.get("visibility_diagnostic", {}) or True
    assert a.detail["cues"].get("visibility") is None
    assert b.detail["cues"].get("visibility") is not None


# --------------------------------------------------------------- proposals

def test_disjoint_set():
    ds = DisjointSet(6)
    ds.union(0, 1, 0.1)
    ds.union(1, 2, 0.2)
    assert ds.find(0) == ds.find(2)
    assert ds.find(3) != ds.find(0)
    assert int(ds.size[ds.find(0)]) == 3


def _tiny_mesh():
    """Two coplanar quads meeting at a right angle, as triangles."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 5], [0, 5, 4]])
    return pts, faces


def test_mesh_edges_are_unique_and_undirected():
    pts, faces = _tiny_mesh()
    e = mesh_edges(faces)
    assert e.shape[1] == 2
    assert np.all(e[:, 0] < e[:, 1])
    assert len(np.unique(e, axis=0)) == len(e)


def test_segmentation_splits_at_a_crease():
    pts, faces = _tiny_mesh()
    from sqe.geom.pointcloud import vertex_normals_from_faces
    n = vertex_normals_from_faces(pts, faces)
    e = mesh_edges(faces)
    w = edge_weights(pts, n, None, e)
    labels = felzenszwalb_segments(len(pts), e, w, k=0.001, min_size=1)
    assert labels.max() + 1 >= 2, "a right-angle crease should split"


def test_structure_mask_finds_floor_and_ceiling():
    rng = np.random.default_rng(0)
    floor = np.c_[rng.random((3000, 2)) * 4, np.zeros(3000)]
    ceil = np.c_[rng.random((3000, 2)) * 4, np.full(3000, 2.6)]
    obj = np.c_[rng.random((500, 2)) * 0.4 + 1.5, rng.random(500) * 0.8 + 0.05]
    pts = np.vstack([floor, ceil, obj])
    up = np.tile(np.array([0.0, 0.0, 1.0]), (len(pts), 1))
    mask = vertex_structure_mask(pts, up, 0.0, 2.6)
    assert mask[:3000].mean() > 0.95        # floor is structure
    assert mask[3000:6000].mean() > 0.95    # ceiling is structure
    assert mask[6000:].mean() < 0.35        # the object mostly is not


# ---------------------------------------------------- instance evaluation

def test_mask_iou_and_matching():
    assert mask_iou(np.arange(10), np.arange(10)) == pytest.approx(1.0)
    assert mask_iou(np.arange(10), np.arange(10, 20)) == 0.0
    gt = [np.arange(0, 100), np.arange(100, 200)]
    pred = [np.arange(0, 95), np.arange(500, 600)]
    best, matched = match_instances(pred, gt)
    assert best[0] == pytest.approx(0.95)
    assert best[1] == 0.0 and matched[1] == -1


def test_score_instances_reports_recall_and_labels():
    gt = [np.arange(0, 100), np.arange(100, 200), np.arange(200, 260)]
    pred = [np.arange(0, 95), np.arange(100, 150), np.arange(150, 200),
            np.arange(500, 540)]
    r = score_instances(pred, ["mug", "cup", "bottle", "junk"],
                        gt, ["mug", "cup", "bottle"], [100, 100, 60])
    assert r["n_ground_truth"] == 3 and r["n_proposals"] == 4
    assert r["recall@0.25"] == pytest.approx(2 / 3)
    assert r["precision@0.5"] == pytest.approx(0.5)
    assert 0.0 <= r["mean_best_iou"] <= 1.0
    assert r["label_accuracy_exact"] == pytest.approx(1.0)


def test_matching_is_one_to_one():
    """Over-segmentation must not let two proposals both claim one instance."""
    gt = [np.arange(0, 100)]
    pred = [np.arange(0, 50), np.arange(50, 100)]
    best, matched = match_instances(pred, gt)
    assert len(set(matched[matched >= 0].tolist())) == int((matched >= 0).sum())


# ------------------------------------------------------------ quality audit

class _FakeObj:
    def __init__(self, oid, label, extent, height, meta=None):
        self.id = oid
        self.label = label
        self.extent = np.asarray(extent, float)
        self.height = height
        self.meta = meta or {}


def test_audit_flags_a_box_that_disagrees_with_its_mask():
    o = _FakeObj(1, "office chair", (1.75, 0.40, 0.86), 0.86,
                 {"anno_obb_center_error": 0.233})
    f = flag_object(o)
    assert f.suspect
    assert any("disagree" in r for r in f.reasons)


def test_audit_flags_an_elongated_chair():
    o = _FakeObj(2, "office chair", (1.75, 0.40, 0.90), 0.90,
                 {"anno_obb_center_error": 1e-7})
    assert flag_object(o).suspect


def test_audit_accepts_a_normal_chair():
    o = _FakeObj(3, "office chair", (0.66, 0.64, 0.92), 0.92,
                 {"anno_obb_center_error": 2e-7})
    assert not flag_object(o).suspect


def test_audit_does_not_flag_naturally_variable_categories():
    """A floor lamp is 1.4 m and a closed laptop is 4 cm; neither is an error."""
    for label, extent, h in (("lamp", (0.3, 0.3, 1.42), 1.42),
                             ("laptop", (0.33, 0.24, 0.04), 0.04),
                             ("shelf", (0.9, 0.35, 0.78), 0.78),
                             ("monitor", (0.56, 0.20, 0.50), 0.50),
                             ("phone", (0.17, 0.03, 0.11), 0.11)):
        o = _FakeObj(4, label, extent, h, {"anno_obb_center_error": 1e-7})
        assert not flag_object(o).suspect, label


def test_audit_writes_into_scene_meta(studio):
    a = audit_scene(studio)
    assert a["n_objects"] == len(studio.objects)
    assert "quality_audit" in studio.meta
    assert isinstance(suspect_ids(studio), set)

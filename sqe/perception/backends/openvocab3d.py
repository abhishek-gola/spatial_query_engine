"""Open-vocabulary 3-D instances: geometric proposals + multi-view CLIP labels.

The perception half of the pipeline, and explicitly a replaceable component.
It produces the same `Scene` type as the ground-truth loader, which is what
makes the two comparable in the benchmark, and its instance quality is *measured*
against the ground truth rather than assumed -- otherwise the `perception` row of
the failure-attribution table means nothing.

Pipeline:

1. class-agnostic proposals from mesh geometry (`sqe.perception.proposals`);
2. for each proposal, the frames that best see it, judged with the frame's own
   depth map so occluded views are rejected;
3. padded square crops from those frames, encoded with CLIP and averaged with
   visibility weights;
4. labels by similarity against a vocabulary, with the embedding kept on the
   object so a query outside the vocabulary can still be matched.

No learned proposal network is used, so recall on small objects is limited and
large objects fragment. That is reported, not hidden. A Mask3D-style model is the
obvious drop-in: produce masks and call `scene_from_masks`.

Cost is dominated by video decoding, not by CLIP, so every frame the whole scene
needs is decoded once in a single forward pass and shared.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...categories import known_categories, normalize_label
from ...geom.pointcloud import (read_ply, subsample,
                                vertex_normals_from_faces, voxel_downsample)
from ...geom.room import build_room
from ...scenegraph.objects import CameraTrajectory, Object3D, Scene
from ..clip_features import (ClipEncoder, ViewScore, aggregate_view_features,
                             classify, crop_square, score_views)
from ..proposals import Proposal, propose_instances

#: Extra vocabulary entries beyond `sqe.categories`. Structure classes matter:
#: without them CLIP labels a wall crop as whatever furniture is nearest.
EXTRA_VOCAB = ("wall", "floor", "ceiling", "window", "door", "curtain",
               "radiator", "pipe", "ceiling lamp", "power socket",
               "light switch", "cardboard box", "clothes", "bag", "poster")

#: Structure proposals get their label from geometry, not from CLIP: their kind
#: is already known and a crop of a wall is not informative.
STRUCTURE_LABELS = {"floor": "floor", "ceiling": "ceiling", "wall": "wall"}


def default_vocabulary() -> List[str]:
    vocab = list(dict.fromkeys(list(known_categories()) + list(EXTRA_VOCAB)))
    return vocab


def _frame_source(root: str, scene_id: str):
    from ...viz.overlay import scannetpp_frame_source
    return scannetpp_frame_source(root, scene_id, with_depth=True)


def label_proposals(proposals: Sequence[Proposal], mesh_points: np.ndarray,
                    src, encoder: ClipEncoder,
                    vocab: Optional[Sequence[str]] = None,
                    views_per_proposal: int = 4,
                    frame_stride: int = 12,
                    min_crop_side: int = 48,
                    verbose: bool = True):
    """Label proposals from multi-view CLIP crops.

    Returns `(labels, label_scores, embeddings, stats)`, all indexed like
    `proposals`. Structure proposals are labelled geometrically and skip CLIP.
    """
    vocab = list(vocab or default_vocabulary())
    n = len(proposals)
    labels: List[str] = ["object"] * n
    scores: List[List[Tuple[str, float]]] = [[] for _ in range(n)]
    embeddings: List[Optional[np.ndarray]] = [None] * n

    # -- which frames does each proposal need ---------------------------
    needed: Dict[int, List[Tuple[int, ViewScore]]] = defaultdict(list)
    per_proposal_views: Dict[int, List[ViewScore]] = {}
    for i, p in enumerate(proposals):
        if p.structure is not None:
            labels[i] = STRUCTURE_LABELS.get(p.structure, "wall")
            scores[i] = [(labels[i], 1.0)]
            continue
        vs = score_views(mesh_points[p.indices],
                         src.poses, src.Ks, src.image_size,
                         depth_reader=src.depth_reader,
                         frame_stride=frame_stride, seed=i)
        vs = vs[:views_per_proposal]
        per_proposal_views[i] = vs
        for v in vs:
            needed[v.frame_index].append((i, v))

    if verbose:
        print(f"[openvocab] {len(needed)} frames needed for "
              f"{len(per_proposal_views)} proposals "
              f"(stride {frame_stride}, {views_per_proposal} views each)")

    # -- decode each frame once, crop everything that needs it ----------
    crops: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    frames = src.rgb_many(sorted(needed))
    n_crops = 0
    for fidx, users in needed.items():
        img = frames.get(fidx)
        if img is None:
            continue
        for i, v in users:
            c = crop_square(img, v.bbox, min_side=min_crop_side)
            if c is None:
                continue
            crops[i].append((fidx, c))
            n_crops += 1
    if verbose:
        print(f"[openvocab] {n_crops} crops from {len(frames)} decoded frames")

    # -- encode ---------------------------------------------------------
    flat: List[np.ndarray] = []
    owner: List[int] = []
    weight: List[float] = []
    for i, items in crops.items():
        vs = {v.frame_index: v.score for v in per_proposal_views.get(i, [])}
        for fidx, c in items:
            flat.append(c)
            owner.append(i)
            weight.append(vs.get(fidx, 0.1))
    if flat:
        feats = encoder.encode_images(flat)
        by_owner: Dict[int, List[int]] = defaultdict(list)
        for k, i in enumerate(owner):
            by_owner[i].append(k)
        for i, ks in by_owner.items():
            embeddings[i] = aggregate_view_features(
                feats[ks], [weight[k] for k in ks])
        idx = [i for i in by_owner]
        emb = np.stack([embeddings[i] for i in idx])
        ranked = classify(emb, vocab, encoder)
        for i, r in zip(idx, ranked):
            scores[i] = r
            labels[i] = r[0][0] if r else "object"

    stats = {
        "n_proposals": n,
        "n_structure": sum(1 for p in proposals if p.structure is not None),
        "n_labelled_by_clip": sum(1 for e in embeddings if e is not None),
        "n_crops": n_crops,
        "n_frames_decoded": len(frames),
        "frame_stride": frame_stride,
        "views_per_proposal": views_per_proposal,
        "vocabulary_size": len(vocab),
        "clip_model": encoder.model_name,
        "device": encoder.device,
    }
    if verbose:
        print(f"[openvocab] labelled {stats['n_labelled_by_clip']} proposals "
              f"with CLIP ({encoder.model_name} on {encoder.device})")
    return labels, scores, embeddings, stats


def build_scene_openvocab(root: str, scene_id: str,
                          object_voxel: float = 0.015,
                          background_voxel: float = 0.03,
                          max_points_per_object: int = 30000,
                          trajectory_stride: int = 12,
                          forward_convention: str = "composite",
                          clip_model: Optional[str] = None,
                          device: Optional[str] = None,
                          views_per_proposal: int = 4,
                          frame_stride: int = 12,
                          felzenszwalb_k: float = 0.06,
                          score_against_gt: bool = True,
                          verbose: bool = True, **_) -> Scene:
    """Build a `Scene` from open-vocabulary predicted instances."""
    from ...data.scannetpp import ScanNetPPPaths
    paths = ScanNetPPPaths(root, scene_id)
    problems = paths.check()
    if problems:
        raise FileNotFoundError(f"scene {scene_id} is not usable:\n  " +
                                "\n  ".join(problems))

    points, colors, faces = read_ply(paths.mesh)
    if faces is None or not len(faces):
        raise ValueError(f"{paths.mesh} has no faces; the geometric proposal "
                         f"stage needs mesh connectivity")
    if verbose:
        print(f"[openvocab] {scene_id}: {len(points)} vertices, "
              f"{len(faces)} faces")

    normals = vertex_normals_from_faces(points, faces)
    room = build_room(points, normals=normals, faces=faces,
                      convention=forward_convention)

    proposals, prop_stats = propose_instances(
        points, faces, colors, room=room, k=felzenszwalb_k, verbose=verbose)

    src = _frame_source(root, scene_id)
    if verbose:
        print(f"[openvocab] frame source: {len(src)} frames, "
              f"video={'yes' if src.video_path else 'no'}, "
              f"depth={'yes' if src.depth_reader else 'no'}")

    encoder = ClipEncoder(clip_model or "openai/clip-vit-base-patch16", device)
    labels, label_scores, embeddings, clip_stats = label_proposals(
        proposals, points, src, encoder, views_per_proposal=views_per_proposal,
        frame_stride=frame_stride, verbose=verbose)

    objects: List[Object3D] = []
    for i, p in enumerate(proposals):
        pts = points[p.indices]
        red = pts
        if object_voxel > 0:
            red, _, _ = voxel_downsample(pts, object_voxel)
        if len(red) > max_points_per_object:
            red = red[subsample(red, max_points_per_object, i)]
        objects.append(Object3D(
            id=len(objects), label=labels[i], obb=p.obb, points=red,
            point_count=int(len(p.indices)),
            color=tuple(float(x) for x in p.colour),
            label_scores=list(label_scores[i]),
            embedding=embeddings[i], source="openvocab",
            meta={"vertex_indices": p.indices.tolist(),
                  "structure": p.structure,
                  "n_segments": len(p.segment_ids)}))

    bg, _, _ = voxel_downsample(points, background_voxel)
    bg_col = None
    if colors is not None:
        keys = np.floor((points - points.min(axis=0)) / background_voxel
                        ).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        order = np.argsort(first)
        bg = points[first[order]]
        bg_col = colors[first[order]]

    traj = None
    if len(src):
        step = max(1, len(src) // 700)
        traj = CameraTrajectory(src.poses[::step], src.Ks[::step],
                                src.image_size, list(src.names[::step]))

    meta: Dict = {"root": os.path.abspath(root), "scene_id": scene_id,
                  "perception": "openvocab",
                  "proposals": prop_stats, "clip": clip_stats}

    scene = Scene(scene_id=scene_id, objects=objects, room=room,
                  trajectory=traj, background=bg, background_color=bg_col,
                  up=room.up, dataset="scannetpp", meta=meta)

    if score_against_gt:
        try:
            from ...data.scannetpp import gt_instances
            gt = gt_instances(paths, len(points))
            from ..evaluate import score_instances
            meta["vs_ground_truth"] = score_instances(
                [np.asarray(o.meta["vertex_indices"], np.int64)
                 for o in objects],
                [o.label for o in objects],
                [g.vertex_indices for g in gt], [g.label for g in gt],
                [len(g.vertex_indices) for g in gt])
            if verbose:
                v = meta["vs_ground_truth"]
                print(f"[openvocab] vs ground truth: "
                      f"recall@0.25 {v['recall@0.25']:.2f}, "
                      f"recall@0.5 {v['recall@0.5']:.2f}, "
                      f"mean best IoU {v['mean_best_iou']:.2f}, "
                      f"label acc {v['label_accuracy_exact']}")
        except Exception as exc:
            meta["vs_ground_truth"] = {"error": f"{type(exc).__name__}: {exc}"}
    return scene


def scene_from_masks(root: str, scene_id: str,
                     masks: Sequence[np.ndarray],
                     labels: Sequence[str],
                     scores: Optional[Sequence[float]] = None,
                     **kw) -> Scene:
    """Build a `Scene` from externally supplied instance masks.

    The drop-in point for a learned proposal network: hand over per-instance
    vertex index arrays and labels and everything downstream is unchanged.
    """
    from ...data.scannetpp import ScanNetPPPaths
    from ...geom.obb import fit_obb
    paths = ScanNetPPPaths(root, scene_id)
    points, colors, faces = read_ply(paths.mesh)
    normals = (vertex_normals_from_faces(points, faces)
               if faces is not None and len(faces) else None)
    room = build_room(points, normals=normals, faces=faces,
                      convention=kw.get("forward_convention", "composite"))
    objects: List[Object3D] = []
    for i, (m, lab) in enumerate(zip(masks, labels)):
        idx = np.asarray(m, dtype=np.int64)
        if len(idx) < 30:
            continue
        pts = points[idx]
        red, _, _ = voxel_downsample(pts, kw.get("object_voxel", 0.015))
        col = ((colors[idx].mean(axis=0) / 255.0).tolist()
               if colors is not None else [0.6, 0.6, 0.6])
        objects.append(Object3D(
            id=len(objects), label=str(lab), obb=fit_obb(pts, trim=0.005),
            points=red, point_count=int(len(idx)), color=tuple(col),
            label_scores=[(str(lab), float(scores[i]) if scores else 1.0)],
            source="openvocab",
            meta={"vertex_indices": idx.tolist(), "external_masks": True}))
    bg, _, _ = voxel_downsample(points, kw.get("background_voxel", 0.03))
    return Scene(scene_id=scene_id, objects=objects, room=room, background=bg,
                 up=room.up, dataset="scannetpp",
                 meta={"perception": "openvocab_external"})

"""Scene building, one call per dataset, plus the post-processing every backend
shares (fronts, shelf levels).

Kept separate from the loaders so the ground-truth path and the open-vocabulary
path get *identical* downstream treatment. If the two diverged here, the
benchmark's perception comparison would be measuring the difference between two
pipelines rather than the difference between two detectors.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from .geom.support import detect_levels, shelf_levels
from .perception.orientation import annotate_scene_fronts
from .scenegraph.objects import Scene

DATASETS = ("scannetpp", "synthetic", "arkitscenes")


def finish_scene(scene: Scene, use_gt_fronts: bool = False,
                 use_visibility: bool = False,
                 detect_shelf_levels: bool = True,
                 verbose: bool = True) -> Dict:
    """Post-process a freshly loaded scene: fronts, then internal surfaces."""
    stats = {"fronts": annotate_scene_fronts(scene, use_visibility, use_gt_fronts)}
    if detect_shelf_levels:
        n_with = 0
        for o in scene.objects:
            if o.points is None or len(o.points) < 200:
                continue
            if not (o.is_support_surface or o.is_container):
                continue
            lv = shelf_levels(detect_levels(o.points, o.obb), o.obb)
            o.levels = [l.z for l in lv]
            o.level_confidence = 1.0 if lv else 0.0
            if lv:
                n_with += 1
        stats["objects_with_shelf_levels"] = n_with
    # The annotation audit compares a fitted box against an *annotated* box, so
    # it only means anything on a ground-truth scene. On predicted instances the
    # size heuristics alone would just relabel ordinary over-segmentation as
    # "dubious annotation".
    if any(o.source == "gt" for o in scene.objects):
        from .data.quality import audit_scene
        stats["quality"] = audit_scene(scene)
    scene.meta["build_stats"] = stats
    if verbose:
        f = stats["fronts"]
        print(f"[pipeline] fronts: {f['estimated']} estimated of "
              f"{f['front_categories']} front-bearing categories, "
              f"{f.get('abstained', 0)} abstentions {f.get('reasons', {})}")
        if detect_shelf_levels:
            print(f"[pipeline] {stats['objects_with_shelf_levels']} objects have "
                  f"detected internal shelf levels")
        q = stats.get("quality")
        if q and q["n_suspect"]:
            print(f"[pipeline] {q['n_suspect']} of {q['n_objects']} instances "
                  f"flagged as dubious annotations (excluded from benchmark "
                  f"proposals); see `sqe audit`")
    return stats


def build(dataset: str, root: str, scene_id: str,
          perception: str = "gt", use_gt_fronts: bool = False,
          forward_convention: str = "composite",
          verbose: bool = True, **kw) -> Scene:
    """Build a scene from a dataset.

    `perception` selects the instance source: `gt` uses the dataset's
    annotations, `openvocab` runs the open-vocabulary backend.
    """
    if dataset == "synthetic":
        from .data.synthetic import make
        scene = make(scene_id, forward_convention=forward_convention, **kw)
    elif dataset == "scannetpp":
        if perception == "gt":
            from .data.scannetpp import build_scene
            scene = build_scene(root, scene_id,
                                forward_convention=forward_convention,
                                verbose=verbose, **kw)
        elif perception == "openvocab":
            from .perception.backends.openvocab3d import build_scene_openvocab
            scene = build_scene_openvocab(root, scene_id,
                                          forward_convention=forward_convention,
                                          verbose=verbose, **kw)
        else:
            raise ValueError(f"unknown perception mode {perception!r}")
    elif dataset == "arkitscenes":
        from .data.arkitscenes import build_scene as ark_build
        scene = ark_build(root, scene_id,
                          forward_convention=forward_convention,
                          verbose=verbose, **kw)
    else:
        raise ValueError(f"unknown dataset {dataset!r}; have {DATASETS}")
    scene.meta["perception"] = perception
    finish_scene(scene, use_gt_fronts=use_gt_fronts, verbose=verbose)
    return scene

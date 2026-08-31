"""Scoring predicted instances against ground-truth instances.

The open-vocabulary path is a component, not a contribution, but the benchmark's
`perception` attribution row is only interpretable if the perception quality is
*measured*. Reporting "43% of failures are frame errors and 12% are perception
errors" is meaningless without knowing whether the detector recalled 90% or 40%
of the objects.

Metrics, all mask-based over mesh vertices so they do not depend on box fitting:

* **recall at IoU t** -- fraction of ground-truth instances matched by some
  proposal at mask IoU >= t. The number that bounds how well any downstream
  resolver can do.
* **precision at IoU t** -- fraction of proposals matching some ground-truth
  instance, i.e. how much of the proposal set is junk.
* **best-IoU distribution** -- per ground-truth instance, so systematic
  over-segmentation of large objects is visible rather than averaged away.
* **label accuracy** -- of the matched instances, how often the open-vocabulary
  label agrees with the annotation, both exactly and after normalisation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..categories import label_matches, normalize_label

IOU_THRESHOLDS = (0.25, 0.5, 0.75)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two index sets over the same vertex array."""
    sa, sb = set(a.tolist()), set(b.tolist())
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    return inter / float(len(sa) + len(sb) - inter)


def match_instances(pred_indices: Sequence[np.ndarray],
                    gt_indices: Sequence[np.ndarray]
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy one-to-one matching by IoU. Returns (best_iou_per_gt, matched_pred).

    Greedy rather than Hungarian: with heavy over-segmentation the assignment is
    dominated by one obvious match per instance, and greedy makes the numbers
    reproducible without an extra dependency.
    """
    n_gt, n_pred = len(gt_indices), len(pred_indices)
    best_iou = np.zeros(n_gt)
    matched = np.full(n_gt, -1, dtype=np.int64)
    if not n_gt or not n_pred:
        return best_iou, matched

    # candidate pairs via a vertex -> proposal index
    pred_of_vertex: Dict[int, List[int]] = {}
    for j, idx in enumerate(pred_indices):
        for v in idx.tolist():
            pred_of_vertex.setdefault(v, []).append(j)

    pairs: List[Tuple[float, int, int]] = []
    for i, g in enumerate(gt_indices):
        touched: Dict[int, int] = {}
        for v in g.tolist():
            for j in pred_of_vertex.get(v, ()):
                touched[j] = touched.get(j, 0) + 1
        for j, inter in touched.items():
            union = len(g) + len(pred_indices[j]) - inter
            if union > 0:
                pairs.append((inter / float(union), i, j))
    pairs.sort(key=lambda t: -t[0])

    used_pred, used_gt = set(), set()
    for iou, i, j in pairs:
        if i in used_gt or j in used_pred:
            continue
        used_gt.add(i)
        used_pred.add(j)
        best_iou[i] = iou
        matched[i] = j
    return best_iou, matched


def score_instances(pred_indices: Sequence[np.ndarray],
                    pred_labels: Sequence[str],
                    gt_indices: Sequence[np.ndarray],
                    gt_labels: Sequence[str],
                    gt_sizes: Optional[Sequence[int]] = None) -> Dict:
    """Recall, precision, IoU distribution and label accuracy."""
    best_iou, matched = match_instances(pred_indices, gt_indices)
    n_gt, n_pred = len(gt_indices), len(pred_indices)

    out: Dict = {"n_ground_truth": n_gt, "n_proposals": n_pred}
    for t in IOU_THRESHOLDS:
        hit = best_iou >= t
        out[f"recall@{t}"] = float(hit.mean()) if n_gt else None
        out[f"precision@{t}"] = (float(hit.sum()) / n_pred) if n_pred else None
    out["mean_best_iou"] = float(best_iou.mean()) if n_gt else None
    out["median_best_iou"] = float(np.median(best_iou)) if n_gt else None

    # label agreement on the instances that were found at all
    exact = soft = considered = 0
    per_label: Dict[str, List[float]] = {}
    for i in range(n_gt):
        per_label.setdefault(normalize_label(gt_labels[i]), []).append(
            float(best_iou[i]))
        j = int(matched[i])
        if j < 0 or best_iou[i] < 0.25:
            continue
        considered += 1
        if normalize_label(pred_labels[j]) == normalize_label(gt_labels[i]):
            exact += 1
        if label_matches(pred_labels[j], gt_labels[i]) >= 0.6:
            soft += 1
    out["n_labelled_comparisons"] = considered
    out["label_accuracy_exact"] = (exact / considered) if considered else None
    out["label_accuracy_soft"] = (soft / considered) if considered else None
    out["mean_best_iou_by_label"] = {
        k: round(float(np.mean(v)), 3) for k, v in sorted(per_label.items())}

    # is over-segmentation size-dependent?
    if gt_sizes is not None and n_gt:
        sizes = np.asarray(list(gt_sizes), float)
        big = sizes >= np.median(sizes)
        out["mean_best_iou_large_instances"] = float(best_iou[big].mean())
        out["mean_best_iou_small_instances"] = float(best_iou[~big].mean())
    return out


def score_scene_against_gt(pred_scene, gt_scene, mesh_points=None) -> Dict:
    """Compare two `Scene` objects built from the same mesh.

    Requires both to carry `meta["vertex_indices"]` per object, which the
    ScanNet++ ground-truth loader and the open-vocabulary backend both write.
    Falls back to a nearest-point mask when they do not.
    """
    def indices_of(scene):
        out = []
        for o in scene.objects:
            idx = o.meta.get("vertex_indices")
            if idx is None:
                return None
            out.append(np.asarray(idx, dtype=np.int64))
        return out

    pi, gi = indices_of(pred_scene), indices_of(gt_scene)
    if pi is None or gi is None:
        return {"error": "one of the scenes has no per-object vertex indices, "
                         "so masks cannot be compared"}
    return score_instances(pi, [o.label for o in pred_scene.objects],
                           gi, [o.label for o in gt_scene.objects],
                           [len(x) for x in gi])

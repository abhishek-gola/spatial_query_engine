"""Class-agnostic 3-D instance proposals from mesh geometry alone.

This is the perception half of the open-vocabulary path, and it is deliberately
modest. The contribution of this repo is downstream; the job here is to produce
*real* predicted instances -- with real over- and under-segmentation -- so that
the benchmark's perception condition measures something honest rather than
perturbed ground truth.

Two stages, both classical:

1. **Over-segmentation.** Felzenszwalb-style graph segmentation over the mesh
   edge graph, with edge weights combining normal disagreement and colour
   difference. This is what ScanNet's `segmentator` does and what
   over-segmentation-based instance methods have used for a decade.
2. **Agglomeration into proposals.** Adjacent segments are merged while they
   stay convex-ish, similarly coloured and below a size cap, with room structure
   (large horizontal and vertical planes) held back so a chair does not get
   merged into the floor.

Quality is measured, not assumed: `sqe.perception.evaluate` scores the
proposals against the ground-truth instances and the numbers go into the scene
metadata, so the perception row of the failure-attribution table is
interpretable.

No learned proposal network is used. A Mask3D-style model would be better and is
the obvious drop-in: produce `(labels, masks)` and hand them to
`instances_from_masks`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from ..geom.obb import OBB, fit_obb
from ..geom.pointcloud import (largest_component, triangle_normals,
                               vertex_normals_from_faces, voxel_downsample)
from ..geom.transforms import normalize


# --------------------------------------------------------------------------
# union-find
# --------------------------------------------------------------------------

class DisjointSet:
    """Union-find with size and internal-difference tracking, as Felzenszwalb."""

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int32)
        self.size = np.ones(n, dtype=np.int64)
        self.internal = np.zeros(n, dtype=np.float64)

    def find(self, a: int) -> int:
        p = self.parent
        root = a
        while p[root] != root:
            root = p[root]
        while p[a] != root:            # path compression
            p[a], a = root, p[a]
        return int(root)

    def union(self, a: int, b: int, weight: float) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.size[ra] += self.size[rb]
        self.internal[ra] = weight
        return int(ra)


# --------------------------------------------------------------------------
# over-segmentation
# --------------------------------------------------------------------------

def mesh_edges(faces: np.ndarray) -> np.ndarray:
    """Unique undirected vertex pairs of a triangle mesh."""
    faces = np.asarray(faces, dtype=np.int64)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
                       axis=0)
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def edge_weights(points: np.ndarray, normals: np.ndarray,
                 colors: Optional[np.ndarray], edges: np.ndarray,
                 w_normal: float = 1.0, w_colour: float = 0.35,
                 w_length: float = 0.10) -> np.ndarray:
    """Dissimilarity per edge: normal disagreement, colour, and edge length.

    Normals dominate because object boundaries in indoor scans are creases far
    more reliably than they are colour changes -- a white mug on a white shelf
    has no colour edge at all.
    """
    a, b = edges[:, 0], edges[:, 1]
    na, nb = normals[a], normals[b]
    cos = np.abs(np.einsum("ij,ij->i", na, nb))
    w = w_normal * (1.0 - np.clip(cos, 0.0, 1.0))
    if colors is not None:
        ca = colors[a].astype(np.float64) / 255.0
        cb = colors[b].astype(np.float64) / 255.0
        w = w + w_colour * np.linalg.norm(ca - cb, axis=1) / np.sqrt(3.0)
    if w_length > 0:
        d = np.linalg.norm(points[a] - points[b], axis=1)
        w = w + w_length * np.clip(d / 0.05, 0.0, 1.0)
    return w


def felzenszwalb_segments(n_vertices: int, edges: np.ndarray,
                          weights: np.ndarray, k: float = 0.06,
                          min_size: int = 40) -> np.ndarray:
    """Graph segmentation. Returns a segment id per vertex.

    `k` sets the scale: larger merges more. `min_size` post-merges runt
    segments into their cheapest neighbour.
    """
    order = np.argsort(weights, kind="stable")
    ds = DisjointSet(n_vertices)
    ea, eb = edges[:, 0], edges[:, 1]

    for idx in order:
        a, b = int(ea[idx]), int(eb[idx])
        w = float(weights[idx])
        ra, rb = ds.find(a), ds.find(b)
        if ra == rb:
            continue
        ta = ds.internal[ra] + k / ds.size[ra]
        tb = ds.internal[rb] + k / ds.size[rb]
        if w <= min(ta, tb):
            ds.union(ra, rb, w)

    # absorb runts
    for idx in order:
        a, b = int(ea[idx]), int(eb[idx])
        ra, rb = ds.find(a), ds.find(b)
        if ra != rb and (ds.size[ra] < min_size or ds.size[rb] < min_size):
            ds.union(ra, rb, float(weights[idx]))

    roots = np.array([ds.find(i) for i in range(n_vertices)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64)


# --------------------------------------------------------------------------
# segment summaries and agglomeration
# --------------------------------------------------------------------------

@dataclass
class Segment:
    id: int
    indices: np.ndarray
    center: np.ndarray
    normal: np.ndarray               # dominant normal
    colour: np.ndarray               # mean, 0..1
    extent: np.ndarray
    planarity: float                 # 1 = flat
    height_span: Tuple[float, float]
    n: int

    @property
    def is_horizontal(self) -> bool:
        return abs(float(self.normal[2])) > 0.85

    @property
    def is_vertical(self) -> bool:
        return abs(float(self.normal[2])) < 0.25

    @property
    def footprint(self) -> float:
        return float(self.extent[0] * self.extent[1])


def summarise_segments(points: np.ndarray, labels: np.ndarray,
                       colors: Optional[np.ndarray]) -> List[Segment]:
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    bounds = np.flatnonzero(np.diff(sorted_labels)) + 1
    groups = np.split(order, bounds)
    out: List[Segment] = []
    for g in groups:
        if len(g) < 4:
            continue
        pts = points[g]
        c = pts.mean(axis=0)
        d = pts - c
        cov = d.T @ d / len(g)
        w, V = np.linalg.eigh(cov)
        nrm = V[:, 0]
        if nrm[2] < 0:
            nrm = -nrm
        planar = float(1.0 - w[0] / max(w[2], 1e-12))
        col = (colors[g].mean(axis=0) / 255.0 if colors is not None
               else np.array([0.6, 0.6, 0.6]))
        out.append(Segment(
            id=int(labels[g[0]]), indices=g, center=c, normal=normalize(nrm),
            colour=col, extent=pts.max(axis=0) - pts.min(axis=0),
            planarity=planar,
            height_span=(float(pts[:, 2].min()), float(pts[:, 2].max())),
            n=len(g)))
    return out


def vertex_structure_mask(points: np.ndarray, normals: np.ndarray,
                          floor_z: float, ceiling_z: Optional[float],
                          room=None, slab: float = 0.06,
                          wall_slab: float = 0.10) -> np.ndarray:
    """Per-vertex mask of room shell: floor, ceiling and walls.

    Computed from *planes*, not from per-segment area. The area test cannot work:
    the over-segmentation shatters a floor into hundreds of small pieces, none of
    which is individually large, so nothing was ever marked as structure and the
    agglomeration happily merged chairs into the floor they stand on. Two
    proposals came out with 120 000 points each.

    Returns a boolean array over vertices.
    """
    points = np.asarray(points, dtype=np.float64)
    z = points[:, 2]
    up = np.array([0.0, 0.0, 1.0])
    flat = np.abs(normals @ up) > 0.80
    mask = flat & (z < floor_z + slab)
    if ceiling_z is not None:
        mask |= flat & (z > ceiling_z - slab)

    vertical = np.abs(normals @ up) < 0.35
    if room is not None and getattr(room, "walls", None):
        for w in room.walls:
            d = np.asarray(w.direction, float)
            proj = points @ d
            near = proj > float(w.offset) - wall_slab
            aligned = np.abs(normals @ d) > 0.70
            mask |= near & aligned & vertical
    else:
        # fall back to the outermost vertical slabs on each horizontal axis
        for k in (0, 1):
            for sign in (+1, -1):
                d = np.zeros(3)
                d[k] = sign
                proj = points @ d
                far = float(np.quantile(proj, 0.995))
                mask |= (proj > far - wall_slab) & vertical
    return mask


def structure_mask(segments: Sequence[Segment], floor_z: float,
                   ceiling_z: Optional[float],
                   vertex_mask: Optional[np.ndarray] = None,
                   min_structure_fraction: float = 0.60,
                   min_structure_area: float = 2.0) -> Dict[int, str]:
    """Label the segments that are room shell rather than objects.

    A segment counts as structure when most of its vertices fall in
    `vertex_mask`. Held back from agglomeration so that a chair does not get
    merged into the floor it stands on, which is the single most destructive
    merge in a purely geometric pipeline.
    """
    out: Dict[int, str] = {}
    for s in segments:
        lo, hi = s.height_span
        kind = None
        if vertex_mask is not None:
            frac = float(np.mean(vertex_mask[s.indices]))
            if frac >= min_structure_fraction:
                if s.is_horizontal and lo < floor_z + 0.15:
                    kind = "floor"
                elif (s.is_horizontal and ceiling_z is not None
                      and hi > ceiling_z - 0.25):
                    kind = "ceiling"
                else:
                    kind = "wall"
        if kind is None:
            # a large flat plane at floor or ceiling height is structure even
            # if the plane fit missed it
            if s.is_horizontal and s.footprint > min_structure_area:
                if lo < floor_z + 0.12:
                    kind = "floor"
                elif ceiling_z is not None and hi > ceiling_z - 0.20:
                    kind = "ceiling"
            elif (s.is_vertical and s.planarity > 0.96 and (hi - lo) > 1.4
                  and max(s.extent[0], s.extent[1]) > 1.5):
                kind = "wall"
        if kind is not None:
            out[s.id] = kind
    return out


def segment_adjacency(labels: np.ndarray, edges: np.ndarray) -> Dict[Tuple[int, int], int]:
    """Count mesh edges crossing each pair of segments."""
    la, lb = labels[edges[:, 0]], labels[edges[:, 1]]
    cross = la != lb
    pairs = np.stack([np.minimum(la[cross], lb[cross]),
                      np.maximum(la[cross], lb[cross])], axis=1)
    if not len(pairs):
        return {}
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)
    return {(int(a), int(b)): int(c) for (a, b), c in zip(uniq, counts)}


@dataclass
class Proposal:
    indices: np.ndarray
    obb: OBB
    colour: np.ndarray
    segment_ids: List[int] = field(default_factory=list)
    structure: Optional[str] = None
    n_points: int = 0


def agglomerate(points: np.ndarray, labels: np.ndarray,
                segments: Sequence[Segment],
                adjacency: Dict[Tuple[int, int], int],
                structure: Dict[int, str],
                colour_tol: float = 0.30,
                max_extent: float = 2.6,
                min_shared_edges: int = 12,
                max_merge_rounds: int = 6) -> List[Proposal]:
    """Merge adjacent non-structure segments into object-sized proposals."""
    by_id = {s.id: s for s in segments}
    ds = DisjointSet(int(max(by_id) + 1) if by_id else 1)

    # merge candidates, cheapest (most similar) first
    cands = []
    for (a, b), shared in adjacency.items():
        if a not in by_id or b not in by_id:
            continue
        if a in structure or b in structure:
            continue
        if shared < min_shared_edges:
            continue
        sa, sb = by_id[a], by_id[b]
        colour_d = float(np.linalg.norm(sa.colour - sb.colour) / np.sqrt(3.0))
        normal_d = 1.0 - abs(float(np.dot(sa.normal, sb.normal)))
        cost = colour_d + 0.5 * normal_d - 0.02 * np.log1p(shared)
        cands.append((cost, a, b, colour_d))
    cands.sort(key=lambda t: t[0])

    members: Dict[int, List[int]] = {s.id: [s.id] for s in segments}
    for _ in range(max_merge_rounds):
        merged_any = False
        for cost, a, b, colour_d in cands:
            ra, rb = ds.find(a), ds.find(b)
            if ra == rb or colour_d > colour_tol:
                continue
            idx = np.concatenate([points[np.concatenate(
                [by_id[m].indices for m in members.get(r, [r]) if m in by_id])]
                for r in (ra, rb) if members.get(r)] or [np.zeros((0, 3))])
            if len(idx) == 0:
                continue
            ext = idx.max(axis=0) - idx.min(axis=0)
            if float(max(ext[0], ext[1])) > max_extent or float(ext[2]) > max_extent:
                continue
            root = ds.union(ra, rb, cost)
            other = rb if root == ra else ra
            members[root] = members.get(root, [root]) + members.pop(other, [other])
            merged_any = True
        if not merged_any:
            break

    out: List[Proposal] = []
    for root, segs in members.items():
        # Structure segments must not come through here as singletons. They are
        # seeded into `members` so the union-find is complete, but they are
        # emitted once per plane by `_merge_structure` below. Letting both paths
        # run gave 1076 "wall" objects for one small bathroom.
        segs = [m for m in segs if m in by_id and m not in structure]
        if not segs:
            continue
        idx = np.concatenate([by_id[m].indices for m in segs])
        if len(idx) < 40:
            continue
        pts = points[idx]
        # drop stray components before fitting the box, or a blob of wall
        # picked up behind the object doubles its extent
        keep = largest_component(pts, eps=0.08, min_samples=8)
        if len(keep) >= 40:
            idx, pts = idx[keep], pts[keep]
        try:
            box = fit_obb(pts, trim=0.005)
        except (ValueError, IndexError):
            continue
        col = np.mean([by_id[m].colour for m in segs], axis=0)
        struct = next((structure[m] for m in segs if m in structure), None)
        out.append(Proposal(indices=idx, obb=box, colour=col,
                            segment_ids=segs, structure=struct,
                            n_points=len(idx)))
    out.extend(_merge_structure(points, segments, structure))
    return out


def _merge_structure(points: np.ndarray, segments: Sequence[Segment],
                     structure: Dict[int, str],
                     wall_offset_bucket: float = 0.30) -> List[Proposal]:
    """Collapse structure segments into one proposal per plane.

    Emitting a proposal per structure segment gave two thousand of them for a
    single bathroom, because the over-segmentation shatters every wall. A room
    has one floor, one ceiling and a handful of walls, and those are the only
    structure objects a query can name.
    """
    groups: Dict[Tuple[str, int, int], List[Segment]] = {}
    for sg in segments:
        kind = structure.get(sg.id)
        if kind is None:
            continue
        if kind in ("floor", "ceiling"):
            key = (kind, 0, 0)
        else:
            # bucket walls by facing direction and by distance along it
            azimuth = int(round(np.arctan2(sg.normal[1], sg.normal[0])
                                / (np.pi / 2.0))) % 2
            axis = 0 if azimuth == 0 else 1
            offset = int(round(float(sg.center[axis]) / wall_offset_bucket))
            key = (kind, axis, offset)
        groups.setdefault(key, []).append(sg)

    out: List[Proposal] = []
    for (kind, _, _), segs in groups.items():
        idx = np.concatenate([sg.indices for sg in segs])
        if len(idx) < 200:
            continue
        pts = points[idx]
        try:
            box = fit_obb(pts, trim=0.002)
        except (ValueError, IndexError):
            continue
        col = np.mean([sg.colour for sg in segs], axis=0)
        out.append(Proposal(indices=idx, obb=box, colour=col,
                            segment_ids=[sg.id for sg in segs],
                            structure=kind, n_points=int(len(idx))))
    return out


def filter_proposals(proposals: Sequence[Proposal],
                     min_points: int = 60,
                     min_extent: float = 0.05,
                     max_footprint: float = 8.0,
                     keep_structure: bool = True) -> List[Proposal]:
    out = []
    for p in proposals:
        if p.structure is not None:
            if keep_structure:
                out.append(p)
            continue
        if p.n_points < min_points:
            continue
        if float(np.max(p.obb.extent)) < min_extent:
            continue
        if p.obb.footprint_area > max_footprint:
            continue
        out.append(p)
    return out


def propose_instances(points: np.ndarray, faces: np.ndarray,
                      colors: Optional[np.ndarray] = None,
                      floor_z: Optional[float] = None,
                      ceiling_z: Optional[float] = None,
                      room=None,
                      k: float = 0.06, min_size: int = 40,
                      colour_tol: float = 0.30,
                      verbose: bool = True) -> Tuple[List[Proposal], Dict]:
    """Full geometric proposal pipeline. Returns (proposals, stats)."""
    points = np.asarray(points, dtype=np.float64)
    normals = vertex_normals_from_faces(points, faces)
    edges = mesh_edges(faces)
    weights = edge_weights(points, normals, colors, edges)
    labels = felzenszwalb_segments(len(points), edges, weights, k, min_size)
    segments = summarise_segments(points, labels, colors)
    if floor_z is None:
        floor_z = (float(room.floor_z) if room is not None
                   else float(np.quantile(points[:, 2], 0.002)))
    if ceiling_z is None:
        ceiling_z = (float(room.ceiling_z) if room is not None
                     and room.ceiling_z is not None
                     else float(np.quantile(points[:, 2], 0.998)))
    vmask = vertex_structure_mask(points, normals, floor_z, ceiling_z, room)
    structure = structure_mask(segments, floor_z, ceiling_z, vmask)
    adjacency = segment_adjacency(labels, edges)
    proposals = agglomerate(points, labels, segments, adjacency, structure,
                            colour_tol=colour_tol)
    proposals = filter_proposals(proposals)
    stats = {"n_vertices": int(len(points)), "n_edges": int(len(edges)),
             "structure_vertex_fraction": float(np.mean(vmask)),
             "n_oversegments": int(labels.max() + 1),
             "n_segments_summarised": len(segments),
             "n_structure_segments": len(structure),
             "structure_kinds": {v: sum(1 for x in structure.values() if x == v)
                                 for v in set(structure.values())},
             "n_proposals": len(proposals),
             "felzenszwalb_k": k, "min_size": min_size,
             "colour_tol": colour_tol}
    if verbose:
        print(f"[proposals] {stats['n_oversegments']} over-segments -> "
              f"{stats['n_proposals']} proposals "
              f"({stats['n_structure_segments']} structure: "
              f"{stats['structure_kinds']})")
    return proposals, stats

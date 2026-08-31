"""Point-cloud primitives: IO, voxel reduction, neighbour queries, normals,
plane fitting and clustering.

Only numpy + scipy. Open3D would shorten a few of these but it is a heavy
dependency to put underneath the part of the repo people are meant to read,
and DBSCAN over a cKDTree is fifteen lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------

def read_ply(path: str):
    """Read a PLY. Returns (points (N,3) float64, colors (N,3) uint8 or None,
    faces (M,3) int or None)."""
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    pts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    colors = None
    names = v.dtype.names
    if names and {"red", "green", "blue"} <= set(names):
        colors = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.uint8)
    faces = None
    for name in ("face", "faces"):
        if name in [e.name for e in ply.elements]:
            fd = ply[name].data
            key = fd.dtype.names[0]
            # plyfile hands back an object array of per-face index arrays. For
            # a few million triangles, a list comprehension plus vstack takes
            # tens of seconds; concatenating the flat buffer takes under one.
            sizes = np.fromiter((len(f) for f in fd[key]), dtype=np.int64,
                                count=len(fd[key]))
            if len(sizes) and np.all(sizes == 3):
                faces = np.concatenate(fd[key]).astype(np.int64).reshape(-1, 3)
            else:
                tri = [np.asarray(f, dtype=np.int64) for f in fd[key]
                       if len(f) == 3]
                faces = np.vstack(tri) if tri else None
            break
    return pts, colors, faces


def write_ply(path: str, points: np.ndarray, colors: Optional[np.ndarray] = None):
    from plyfile import PlyData, PlyElement

    points = np.asarray(points, dtype=np.float32)
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    cols = [points[:, 0], points[:, 1], points[:, 2]]
    if colors is not None:
        colors = np.asarray(colors)
        if colors.dtype != np.uint8:
            colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        cols += [colors[:, 0], colors[:, 1], colors[:, 2]]
    arr = np.empty(len(points), dtype=fields)
    for (name, _), col in zip(fields, cols):
        arr[name] = col
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


# --------------------------------------------------------------------------
# Reduction
# --------------------------------------------------------------------------

def voxel_key(points: np.ndarray, voxel: float, origin: Optional[np.ndarray] = None):
    if origin is None:
        origin = points.min(axis=0)
    return np.floor((points - origin) / voxel).astype(np.int64)


def voxel_downsample(points: np.ndarray, voxel: float,
                     attrs: Optional[dict] = None,
                     mode: str = "mean"):
    """Reduce `points` to one sample per voxel.

    Returns `(down_points, inverse, attrs_down)` where `inverse[i]` is the
    index into `down_points` for original point `i` -- needed so per-vertex
    labels survive the reduction. `attrs` values are averaged (`mean`) or
    taken from the voxel's first point (`first`).
    """
    points = np.asarray(points, dtype=np.float64)
    if voxel <= 0 or len(points) == 0:
        return points, np.arange(len(points)), (attrs or {})
    keys = voxel_key(points, voxel)
    _, first_idx, inverse = np.unique(keys, axis=0, return_index=True,
                                      return_inverse=True)
    inverse = inverse.reshape(-1)
    n = int(inverse.max()) + 1 if len(inverse) else 0
    if mode == "first":
        down = points[first_idx]
    else:
        down = np.zeros((n, 3))
        counts = np.bincount(inverse, minlength=n).astype(np.float64)
        for c in range(3):
            down[:, c] = np.bincount(inverse, weights=points[:, c], minlength=n)
        down /= counts[:, None]
    out_attrs = {}
    for k, v in (attrs or {}).items():
        v = np.asarray(v)
        if mode == "first" or v.dtype.kind not in "fiu":
            out_attrs[k] = v[first_idx]
        else:
            v2 = v.reshape(len(points), -1).astype(np.float64)
            acc = np.stack([np.bincount(inverse, weights=v2[:, c], minlength=n)
                            for c in range(v2.shape[1])], axis=1)
            acc /= np.bincount(inverse, minlength=n)[:, None]
            out_attrs[k] = acc.reshape((n,) + v.shape[1:])
    return down, inverse, out_attrs


def subsample(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    """Uniform random subsample, returned as indices. Deterministic."""
    n = len(points)
    if n <= max_points:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, max_points, replace=False)


# --------------------------------------------------------------------------
# Neighbour queries
# --------------------------------------------------------------------------

def cloud_gap(a: np.ndarray, b: np.ndarray, max_points: int = 4000) -> float:
    """Minimum surface-to-surface distance between two clouds.

    This is what proximity relations should use. Centroid distance says a mug
    sitting on the edge of a 3 m table is 1.5 m from it; surface distance says
    2 cm, which is what a person means by "next to".
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    a = a[subsample(a, max_points, 0)]
    b = b[subsample(b, max_points, 1)]
    d, _ = cKDTree(b).query(a, k=1)
    return float(d.min())


def cloud_quantile_gap(a: np.ndarray, b: np.ndarray, q: float = 0.05,
                       max_points: int = 4000) -> float:
    """Like `cloud_gap` but takes the q-quantile of nearest distances, which is
    robust to a single stray point bridging two objects."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    a = a[subsample(a, max_points, 0)]
    b = b[subsample(b, max_points, 1)]
    d, _ = cKDTree(b).query(a, k=1)
    return float(np.quantile(d, q))


def estimate_normals(points: np.ndarray, k: int = 24,
                     orient_up: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-point normals from local PCA. Sign is arbitrary unless `orient_up`
    is given, in which case normals are flipped to have a non-negative dot
    with it (useful when you only care about horizontal-vs-vertical)."""
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 3:
        return np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    k = int(min(max(4, k), n))
    tree = cKDTree(points)
    # chunked: the (n, k, 3) neighbour tensor is 24x the cloud, which on a
    # million-vertex ScanNet++ mesh is enough to swap a 16 GB machine
    normals = np.empty((n, 3), dtype=np.float64)
    chunk = max(1, int(4_000_000 // max(k, 1)))
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        _, idx = tree.query(points[start:stop], k=k, workers=-1)
        nb = points[idx]
        nb = nb - nb.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", nb, nb) / float(k)
        _, V = np.linalg.eigh(cov)
        normals[start:stop] = V[:, :, 0]          # smallest eigenvalue
    if orient_up is not None:
        s = np.sign(normals @ np.asarray(orient_up, dtype=np.float64))
        s[s == 0] = 1.0
        normals = normals * s[:, None]
    return normals


def vertex_normals_from_faces(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals from mesh connectivity.

    Preferred over `estimate_normals` whenever faces exist: it is far cheaper,
    and the signs are globally consistent, which matters for telling a wall's
    inward side from its outward one.
    """
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    v0, v1, v2 = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)            # magnitude = 2 * area
    out = np.zeros_like(points)
    for c in range(3):
        for k in range(3):
            np.add.at(out[:, k], faces[:, c], cross[:, k])
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return np.where(n < 1e-12, np.array([0.0, 0.0, 1.0]), out / np.maximum(n, 1e-12))


def triangle_normals(points: np.ndarray, faces: np.ndarray):
    """Face normals and areas of a triangle mesh."""
    v0, v1, v2 = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    nrm = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-12)
    return nrm, area


# --------------------------------------------------------------------------
# Fitting and clustering
# --------------------------------------------------------------------------

@dataclass
class Plane:
    normal: np.ndarray
    offset: float          # plane is {x : normal . x = offset}
    inlier_ratio: float

    def distance(self, pts: np.ndarray) -> np.ndarray:
        return np.abs(np.asarray(pts) @ self.normal - self.offset)


def fit_plane_ransac(points: np.ndarray, thresh: float = 0.02,
                     iters: int = 200, seed: int = 0,
                     normal_prior: Optional[np.ndarray] = None,
                     prior_tol_deg: float = 25.0) -> Optional[Plane]:
    """RANSAC plane. `normal_prior` restricts accepted planes to those whose
    normal is within `prior_tol_deg` of it (e.g. horizontal-only)."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return None
    rng = np.random.default_rng(seed)
    best = None
    cos_tol = np.cos(np.deg2rad(prior_tol_deg))
    for _ in range(iters):
        i = rng.choice(len(points), 3, replace=False)
        p0, p1, p2 = points[i]
        nrm = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm = nrm / ln
        if normal_prior is not None and abs(float(nrm @ normal_prior)) < cos_tol:
            continue
        off = float(nrm @ p0)
        inl = np.abs(points @ nrm - off) < thresh
        cnt = int(inl.sum())
        if best is None or cnt > best[0]:
            best = (cnt, nrm, off, inl)
    if best is None:
        return None
    cnt, nrm, off, inl = best
    # one least-squares refit on the inliers
    pin = points[inl]
    c = pin.mean(axis=0)
    _, _, Vt = np.linalg.svd(pin - c, full_matrices=False)
    nrm = Vt[-1]
    off = float(nrm @ c)
    return Plane(nrm, off, cnt / len(points))


def dbscan(points: np.ndarray, eps: float, min_samples: int = 8) -> np.ndarray:
    """DBSCAN over a cKDTree. Returns labels, -1 for noise."""
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    labels = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return labels
    tree = cKDTree(points)
    neighbours = tree.query_ball_point(points, eps, workers=-1)
    core = np.array([len(nb) >= min_samples for nb in neighbours])
    visited = np.zeros(n, dtype=bool)
    cid = 0
    for i in range(n):
        if visited[i] or not core[i]:
            continue
        stack = [i]
        visited[i] = True
        labels[i] = cid
        while stack:
            j = stack.pop()
            for m in neighbours[j]:
                if labels[m] == -1:
                    labels[m] = cid
                if core[j] and not visited[m]:
                    visited[m] = True
                    if core[m]:
                        stack.append(m)
        cid += 1
    return labels


def largest_component(points: np.ndarray, eps: float,
                      min_samples: int = 8) -> np.ndarray:
    """Indices of the biggest DBSCAN cluster; all indices if clustering fails.

    Predicted instances often pick up a blob of wall behind the object. Keeping
    the dominant component before fitting a box removes most of that.
    """
    labels = dbscan(points, eps, min_samples)
    if labels.max() < 0:
        return np.arange(len(points))
    counts = np.bincount(labels[labels >= 0])
    return np.flatnonzero(labels == int(np.argmax(counts)))


def hull_area_2d(xy: np.ndarray) -> float:
    from scipy.spatial import ConvexHull, QhullError
    xy = np.unique(np.round(np.asarray(xy, dtype=np.float64), 6), axis=0)
    if len(xy) < 3:
        return 0.0
    try:
        return float(ConvexHull(xy).volume)   # 'volume' is area in 2-D
    except (QhullError, ValueError):
        return 0.0


def occupied_area_2d(xy: np.ndarray, cell: float = 0.10) -> float:
    """Observed surface area of a 2-D point set, as occupied grid cells.

    Preferred over `hull_area_2d` for walls. A wall with a doorway in it has the
    same convex hull as a solid wall, so hull area cannot tell them apart --
    which matters, because "which wall is the biggest" is what decides the
    room's canonical forward direction. Counting occupied cells sees the hole,
    and is insensitive to how densely the scan happened to sample the surface.
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(xy) == 0:
        return 0.0
    keys = np.floor(xy / cell).astype(np.int64)
    n = len(np.unique(keys, axis=0))
    return float(n) * cell * cell

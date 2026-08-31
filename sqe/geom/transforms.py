"""Rigid transforms, camera projection and the small linear-algebra helpers.

Conventions are in docs/CONVENTIONS.md. The short version: world is Z-up,
cameras are OpenCV (x right, y down, z forward), and a `pose` is
camera-to-world unless the name says otherwise.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9


def normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """Unit-length version of `v`. Zero vectors are returned unchanged."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.where(n < EPS, v, v / np.maximum(n, EPS))


def project_out(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Component of `v` orthogonal to the unit vector `axis`."""
    v = np.asarray(v, dtype=np.float64)
    axis = normalize(axis)
    return v - np.dot(v, axis) * axis


def horizontal(v: np.ndarray, up: np.ndarray) -> np.ndarray:
    """`v` flattened into the gravity plane and renormalised.

    Returns a zero vector when `v` is (near) parallel to `up`, which callers
    must treat as "this direction has no horizontal meaning" rather than
    silently picking an axis.
    """
    flat = project_out(v, up)
    if np.linalg.norm(flat) < 1e-6:
        return np.zeros(3)
    return normalize(flat)


def rot_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix."""
    axis = normalize(axis)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def rot_about_up(angle: float, up: np.ndarray = None) -> np.ndarray:
    """Yaw rotation of `angle` radians about `up` (default world Z)."""
    if up is None:
        up = np.array([0.0, 0.0, 1.0])
    return rot_axis_angle(up, angle)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector `a` onto unit vector `b`."""
    a, b = normalize(a), normalize(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        # antiparallel: rotate pi about any axis orthogonal to a
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        return rot_axis_angle(normalize(np.cross(a, tmp)), np.pi)
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def basis_from_forward_up(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Right-handed `[r | f | u]` basis from a forward hint and an up axis.

    `forward` is flattened into the plane orthogonal to `up` first, so the
    returned basis never tilts out of the gravity plane. Raises if `forward`
    is parallel to `up`, because then "right" is genuinely undefined and
    guessing is exactly the class of bug this project is about.
    """
    up = normalize(up)
    f = horizontal(forward, up)
    if not np.any(f):
        raise ValueError("forward is parallel to up; 'right' is undefined")
    r = np.cross(f, up)          # r x f = u for a right-handed (r, f, u)
    r = normalize(r)
    return np.stack([r, f, up], axis=1)


def orthonormalize(B: np.ndarray) -> np.ndarray:
    """Nearest orthonormal matrix to `B` (polar decomposition)."""
    U, _, Vt = np.linalg.svd(B)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


# --------------------------------------------------------------------------
# SE(3)
# --------------------------------------------------------------------------

def se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def se3_inverse(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 to an (N,3) array."""
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ T[:3, :3].T + T[:3, 3]


def quat_to_rot(q: np.ndarray, order: str = "wxyz") -> np.ndarray:
    """Quaternion to rotation matrix. COLMAP writes `wxyz`."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    if order == "xyzw":
        q = q[[3, 0, 1, 2]]
    w, x, y, z = q / max(np.linalg.norm(q), EPS)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to `wxyz` quaternion."""
    m = np.asarray(R, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        return np.array([0.25 / s,
                         (m[2, 1] - m[1, 2]) * s,
                         (m[0, 2] - m[2, 0]) * s,
                         (m[1, 0] - m[0, 1]) * s])
    i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(max(1e-12, 1.0 + m[i, i] - m[j, j] - m[k, k]))
    q = np.zeros(4)
    q[0] = (m[k, j] - m[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (m[j, i] + m[i, j]) / s
    q[1 + k] = (m[k, i] + m[i, k]) / s
    return q


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------

def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def project(pts_cam: np.ndarray, K: np.ndarray):
    """(N,3) camera-frame points to pixels. Returns (uv, depth, valid)."""
    pts_cam = np.asarray(pts_cam, dtype=np.float64)
    z = pts_cam[:, 2]
    valid = z > 1e-6
    zz = np.where(valid, z, 1.0)
    uv = np.stack([K[0, 0] * pts_cam[:, 0] / zz + K[0, 2],
                   K[1, 1] * pts_cam[:, 1] / zz + K[1, 2]], axis=1)
    return uv, z, valid


def unproject(depth: np.ndarray, K: np.ndarray, pose_c2w: np.ndarray = None,
              stride: int = 1, depth_range=(0.1, 10.0)):
    """Depth image to points. Returns (points, pixel_index) with the points in
    world frame if `pose_c2w` is given, camera frame otherwise."""
    depth = np.asarray(depth, dtype=np.float64)
    H, W = depth.shape
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    d = depth[vs, us]
    ok = np.isfinite(d) & (d > depth_range[0]) & (d < depth_range[1])
    us, vs, d = us[ok], vs[ok], d[ok]
    x = (us - K[0, 2]) * d / K[0, 0]
    y = (vs - K[1, 2]) * d / K[1, 1]
    pts = np.stack([x, y, d], axis=1)
    if pose_c2w is not None:
        pts = transform_points(pose_c2w, pts)
    return pts, np.stack([vs, us], axis=1)


def camera_center(pose_c2w: np.ndarray) -> np.ndarray:
    return pose_c2w[:3, 3].copy()


def camera_forward(pose_c2w: np.ndarray) -> np.ndarray:
    """World-frame optical axis (camera +z)."""
    return normalize(pose_c2w[:3, 2])


def camera_right(pose_c2w: np.ndarray) -> np.ndarray:
    """World-frame camera +x."""
    return normalize(pose_c2w[:3, 0])


def camera_down(pose_c2w: np.ndarray) -> np.ndarray:
    """World-frame camera +y, which points down in OpenCV convention."""
    return normalize(pose_c2w[:3, 1])

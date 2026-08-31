"""Projecting 3-D boxes onto real camera frames, for visual verification.

The single most useful check on a 3-D pipeline: take a real photograph from the
capture, project the fitted oriented boxes into it using the dataset's own poses
and intrinsics, and look. If the mesh alignment, the pose convention, the
intrinsic scaling or the box fit is wrong, the boxes land in the wrong place and
you see it immediately. Every one of those is a silent failure otherwise -- a
query returns a confident wrong object and nothing in the numbers says why.

Three renderers:

* `render_frame_overlay` -- boxes on a real RGB frame, optionally occlusion-
  tested against the frame's depth map.
* `render_query_overlay` -- the same, but showing one resolution: target,
  anchors, runners-up, and the reference-frame axes actually used.
* `render_topdown` -- a plan view, which is the quickest way to check a
  left/right answer, since lateral relations are a plan-view property.

`render_pointcloud_view` is the fallback when a dataset has no RGB: a small
software point splatter, so the same verification works on synthetic rooms.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..geom.obb import OBB
from ..geom.transforms import project, se3_inverse, transform_points
from ..scenegraph.objects import Object3D, Scene

#: BGR, because that is what OpenCV writes.
COLOURS = {
    "target": (80, 200, 255),      # amber
    "anchor": (110, 90, 240),      # red-pink
    "runner": (230, 170, 90),      # blue
    "other": (150, 150, 150),
    "structure": (90, 90, 90),
    "right": (80, 80, 255),
    "front": (110, 240, 110),
    "up": (255, 160, 80),
}

BOX_EDGES = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))


def _cv2():
    try:
        import cv2
    except ImportError as exc:      # pragma: no cover
        raise ImportError("rendering overlays needs opencv "
                          "(pip install opencv-python)") from exc
    return cv2


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------

def project_box(obb: OBB, K: np.ndarray, pose_c2w: np.ndarray,
                image_size: Tuple[int, int]):
    """Project a box's 8 corners. Returns (uv, depths, all_in_front, on_screen).

    Corner order matches `OBB.corners()`, so `BOX_EDGES` indexes it directly.
    """
    w, h = image_size
    corners_w = obb.corners()
    corners_c = transform_points(se3_inverse(pose_c2w), corners_w)
    uv, z, valid = project(corners_c, K)
    all_front = bool(np.all(z > 0.05))
    on_screen = bool(np.any((uv[:, 0] > -w) & (uv[:, 0] < 2 * w)
                            & (uv[:, 1] > -h) & (uv[:, 1] < 2 * h) & valid))
    return uv, z, all_front, on_screen


def box_screen_area(uv: np.ndarray) -> float:
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    return float(max(0.0, hi[0] - lo[0]) * max(0.0, hi[1] - lo[1]))


class DepthBuffer:
    """Per-pixel scene depth, for hidden-line removal.

    A box drawn as a complete wireframe reads as floating in front of
    everything, because a real wireframe of a solid object shows only the edges
    that are not behind something -- including its own front faces. So every
    edge is tested against the scene depth, sample by sample, and split into
    visible and hidden runs.

    Two sources. The ScanNet++ iPhone depth maps are fully dense (100% valid on
    the frames measured here) and are the primary source; a z-buffer splatted
    from the scene point cloud is the fallback for captures and synthetic rooms
    with no depth, where it also keeps the same code path working.
    """

    def __init__(self, depth: np.ndarray, rgb_size: Tuple[int, int],
                 source: str = "sensor"):
        self.depth = np.asarray(depth, dtype=np.float32)
        self.h, self.w = self.depth.shape[:2]
        self.rgb_w, self.rgb_h = rgb_size
        self.sx = self.w / float(max(self.rgb_w, 1))
        self.sy = self.h / float(max(self.rgb_h, 1))
        self.source = source

    @classmethod
    def from_sensor(cls, depth: np.ndarray,
                    rgb_size: Tuple[int, int]) -> "DepthBuffer":
        """The capture's own depth map, rescaled to RGB pixel coordinates.

        The depth is a downscaled version of the same image, so the mapping is a
        pure scale -- no need to rebuild intrinsics, which is where the earlier
        version guessed a factor from the principal point.
        """
        return cls(depth, rgb_size, "sensor")

    @classmethod
    def from_points(cls, points: np.ndarray, K: np.ndarray,
                    pose_c2w: np.ndarray, rgb_size: Tuple[int, int],
                    downscale: int = 4, splat: int = 2,
                    max_points: int = 600_000) -> "DepthBuffer":
        """Nearest-surface z-buffer splatted from a point cloud.

        Points are splatted as small squares and the buffer is then min-filtered
        over a `splat`-pixel neighbourhood, because a raw one-pixel-per-point
        buffer is full of holes and every hole reads as "nothing here", which
        would make occluded edges pop back into view.
        """
        from ..geom.pointcloud import subsample
        w, h = rgb_size
        bw, bh = max(1, w // downscale), max(1, h // downscale)
        buf = np.full((bh, bw), np.inf, np.float32)
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) == 0:
            return cls(buf, rgb_size, "points")
        if len(pts) > max_points:
            pts = pts[subsample(pts, max_points, 0)]
        Kb = np.asarray(K, dtype=np.float64).copy()
        Kb[0, :] *= bw / float(w)
        Kb[1, :] *= bh / float(h)
        cam = transform_points(se3_inverse(pose_c2w), pts)
        uv, z, valid = project(cam, Kb)
        u = np.round(uv[:, 0]).astype(np.int64)
        v = np.round(uv[:, 1]).astype(np.int64)
        ok = valid & (u >= 0) & (u < bw) & (v >= 0) & (v < bh) & (z > 0.05)
        u, v, z = u[ok], v[ok], z[ok]
        # np.minimum.at is the scatter-min the z-buffer needs
        np.minimum.at(buf, (v, u), z.astype(np.float32))
        if splat > 0:
            pad = splat
            padded = np.pad(buf, pad, constant_values=np.inf)
            out = buf.copy()
            for dy in range(-pad, pad + 1):
                for dx in range(-pad, pad + 1):
                    out = np.minimum(
                        out, padded[pad + dy:pad + dy + bh,
                                    pad + dx:pad + dx + bw])
            buf = out
        buf[~np.isfinite(buf)] = 0.0        # 0 means "no measurement"
        return cls(buf, rgb_size, "points")

    def sample(self, uv: np.ndarray):
        """Depth at RGB pixel coordinates. Returns (depth, valid)."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        iu = np.round(uv[:, 0] * self.sx).astype(np.int64)
        iv = np.round(uv[:, 1] * self.sy).astype(np.int64)
        inside = (iu >= 0) & (iu < self.w) & (iv >= 0) & (iv < self.h)
        d = np.zeros(len(uv), np.float32)
        if inside.any():
            d[inside] = self.depth[iv[inside], iu[inside]]
        return d, inside & (d > 0.05)


def _runs(values: np.ndarray):
    """Group an array into (start, stop_exclusive, value) runs.

    Works on ints as well as bools, because edge samples carry three states and
    collapsing them to a boolean is what made "behind the camera" and "occluded"
    render the same way.
    """
    out = []
    if len(values) == 0:
        return out
    start = 0
    cur = values[0]
    for i in range(1, len(values)):
        if values[i] != cur:
            out.append((start, i, cur))
            start, cur = i, values[i]
    out.append((start, len(values), cur))
    return out


def edge_visibility(p0: np.ndarray, p1: np.ndarray, K: np.ndarray,
                    pose_c2w: np.ndarray, dbuf: Optional[DepthBuffer],
                    tol: float = 0.04, max_samples: int = 96):
    """Sample one box edge and say which parts of it are visible.

    Returns `(uv, visible, in_front)`, all indexed by sample. The sample count
    follows the edge's projected pixel length, so a long edge across the frame
    gets a fine test and a two-pixel edge does not cost 96 depth lookups.
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    ends_c = transform_points(se3_inverse(pose_c2w), np.stack([p0, p1]))
    ends_uv, _, _ = project(ends_c, K)
    px = float(np.linalg.norm(ends_uv[1] - ends_uv[0]))
    n = int(np.clip(px / 6.0, 8, max_samples))

    t = np.linspace(0.0, 1.0, n)
    world = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
    cam = transform_points(se3_inverse(pose_c2w), world)
    uv, z, valid = project(cam, K)
    in_front = valid & (z > 0.05)
    if dbuf is None:
        return uv, in_front.copy(), in_front
    meas, ok = dbuf.sample(uv)
    # hidden where a measurement exists and the edge is behind it
    hidden = ok & (z > meas + tol)
    return uv, in_front & ~hidden, in_front


def _close_holes(buf: np.ndarray, radius: int = 2) -> np.ndarray:
    """Min-filter a sparse z-buffer so its holes stop reading as empty space.

    A point splat leaves gaps between samples, and every gap would let an
    occluded edge show through.
    """
    if radius <= 0:
        return buf
    h, w = buf.shape
    filled = np.where(buf > 0.0, buf, np.inf)
    padded = np.pad(filled, radius, constant_values=np.inf)
    out = filled.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out = np.minimum(out, padded[radius + dy:radius + dy + h,
                                         radius + dx:radius + dx + w])
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def _dim(colour, factor: float = 0.45, bg: int = 22):
    return tuple(int(bg + (c - bg) * factor) for c in colour)


def _draw_polyline(img, uv, i0, i1, colour, thickness, dashed=False,
                   dash_px: int = 9, gap_px: int = 7):
    cv2 = _cv2()
    h, w = img.shape[:2]
    lim = 4 * max(w, h)
    pts = np.clip(uv[i0:i1], -lim, lim).astype(np.int32)
    if len(pts) < 2:
        return
    if not dashed:
        cv2.polylines(img, [pts], False, colour, thickness, cv2.LINE_AA)
        return
    # walk the polyline in pixel space, alternating on and off
    on, carried = True, 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        seg = float(np.hypot(*(b - a)))
        if seg < 1e-6:
            continue
        pos = 0.0
        while pos < seg:
            span = (dash_px if on else gap_px) - carried
            step = min(span, seg - pos)
            if on and step > 0.5:
                p = a + (b - a) * (pos / seg)
                q = a + (b - a) * ((pos + step) / seg)
                cv2.line(img, tuple(p.astype(np.int32)),
                         tuple(q.astype(np.int32)), colour, thickness,
                         cv2.LINE_AA)
            pos += step
            carried += step
            if carried >= (dash_px if on else gap_px) - 1e-6:
                on, carried = not on, 0.0


def draw_box_hidden_line(img: np.ndarray, obb: OBB, K: np.ndarray,
                         pose_c2w: np.ndarray,
                         dbuf: Optional[DepthBuffer], colour,
                         thickness: int = 2, label: Optional[str] = None,
                         label_scale: float = 0.6, tol: float = 0.04,
                         hidden_style: str = "dashed") -> float:
    """Draw a box with its occluded portions removed. Returns visible fraction.

    `hidden_style`:
      * `dashed` -- hidden runs as faint dashes, so the box's full extent is
        still readable. This is the CAD convention and it looks right.
      * `dim`    -- hidden runs as faint solid lines.
      * `hide`   -- hidden runs not drawn at all.

    Because the test is against the *scene* depth, and the scene includes the
    object itself, a box's own back edges come out hidden without any special
    case: the front surface of the object is nearer than they are.
    """
    corners = obb.corners()
    n_vis = n_tot = 0
    vis_uv: List[np.ndarray] = []

    hidden_colour = _dim(colour)
    hidden_thickness = max(1, thickness - 1)

    # Three states per sample, so "behind the camera" is never confused with
    # "occluded": 0 skip, 1 visible, 2 hidden.
    SKIP, VISIBLE, HIDDEN = 0, 1, 2

    for a, b in BOX_EDGES:
        uv, visible, in_front = edge_visibility(corners[a], corners[b], K,
                                                pose_c2w, dbuf, tol)
        state = np.where(~in_front, SKIP,
                         np.where(visible, VISIBLE, HIDDEN)).astype(np.int8)
        n_tot += int(in_front.sum())
        n_vis += int((state == VISIBLE).sum())
        for i0, i1, val in _runs(state):
            if i1 - i0 < 2 or val == SKIP:
                continue
            if val == VISIBLE:
                _draw_polyline(img, uv, i0, i1, colour, thickness)
                vis_uv.append(uv[i0:i1])
            elif hidden_style != "hide":
                _draw_polyline(img, uv, i0, i1, hidden_colour,
                               hidden_thickness,
                               dashed=(hidden_style == "dashed"))

    fraction = (n_vis / n_tot) if n_tot else 0.0

    if label:
        _draw_label(img, vis_uv, corners, K, pose_c2w, label, colour,
                    label_scale, fraction)
    return fraction


def _draw_label(img, vis_uv, corners, K, pose_c2w, label, colour, scale,
                fraction: float, margin: int = 24):
    """Put the label on a visible part of the box, not over an occluder.

    Anchored to a *visible* sample well inside the frame. Without the margin
    test, a box whose visible portion is almost entirely off-screen parked its
    label against the image edge, which read as a stray annotation belonging to
    nothing.
    """
    cv2 = _cv2()
    h, w = img.shape[:2]
    if vis_uv:
        pts = np.concatenate(vis_uv, axis=0)
    else:
        cam = transform_points(se3_inverse(pose_c2w), corners)
        pts, _, _ = project(cam, K)
    inside = pts[(pts[:, 0] > margin) & (pts[:, 0] < w - margin)
                 & (pts[:, 1] > margin) & (pts[:, 1] < h - margin)]
    if len(inside) == 0:
        return              # nothing of this box is comfortably in frame
    anchor = inside[np.argmin(inside[:, 1])]
    x = int(np.clip(anchor[0], 4, w - 8))
    y = int(np.clip(anchor[1] - 8, 16, h - 6))
    text = label if fraction > 0.15 else f"{label} (mostly hidden)"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (x - 3, y - th - 5), (x + tw + 3, y + 4), (24, 24, 24), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                colour if fraction > 0.15 else _dim(colour, 0.6), 1, cv2.LINE_AA)


def is_occluded(obb: OBB, K: np.ndarray, pose_c2w: np.ndarray,
                depth: Optional[np.ndarray], tolerance: float = 0.25) -> bool:
    """Whether the box's nearest face is behind the measured depth.

    Uses the frame's own depth map, so it is a real visibility test rather than a
    guess. A box whose nearest corner is more than `tolerance` metres behind what
    the sensor saw at that pixel is hidden by something.
    """
    if depth is None:
        return False
    dh, dw = depth.shape
    corners_c = transform_points(se3_inverse(pose_c2w), obb.corners())
    Kd = K.copy()
    Kd[0, :] *= dw / float(K[0, 2] * 2.0) if K[0, 2] > 0 else 1.0
    Kd[1, :] *= dh / float(K[1, 2] * 2.0) if K[1, 2] > 0 else 1.0
    uv, z, valid = project(corners_c, Kd)
    seen = 0
    hidden = 0
    for (u, v), zz, ok in zip(uv, z, valid):
        if not ok:
            continue
        iu, iv = int(round(u)), int(round(v))
        if not (0 <= iu < dw and 0 <= iv < dh):
            continue
        d = float(depth[iv, iu])
        if d <= 0.05:
            continue
        seen += 1
        if zz > d + tolerance:
            hidden += 1
    if seen < 3:
        return False
    return hidden / seen > 0.75


def visible_objects(scene: Scene, K: np.ndarray, pose_c2w: np.ndarray,
                    image_size: Tuple[int, int],
                    depth: Optional[np.ndarray] = None,
                    max_distance: float = 6.0,
                    min_screen_area: float = 2500.0,
                    include_structure: bool = False,
                    occlusion_test: bool = True,
                    dbuf: Optional["DepthBuffer"] = None,
                    min_visible_fraction: float = 0.04,
                    max_objects: Optional[int] = 14,
                    skip_enclosing: bool = True
                    ) -> List[Tuple[Object3D, np.ndarray, float, float]]:
    """Objects worth drawing on this frame.

    Returns `(object, projected_corners, distance, visible_fraction)`, nearest
    last so a caller drawing in order paints near objects over far ones.

    Three filters exist to keep the picture readable rather than exhaustive.
    Forty-two boxes on one frame is not a visualisation, it is a haystack:

    * objects that are entirely hidden are dropped -- but a *partly* hidden one
      is kept and drawn with its occluded edges removed, which is the whole
      point of the hidden-line pass;
    * a box that encloses the camera is dropped, because its wireframe is just
      four lines sprawling off every edge of the frame and tells you nothing;
    * only the nearest `max_objects` survive.
    """
    eye = pose_c2w[:3, 3]
    w, h = image_size
    if dbuf is None and depth is not None and occlusion_test:
        dbuf = DepthBuffer.from_sensor(depth, image_size)
    rows: List[Tuple[Object3D, np.ndarray, float, float]] = []
    for o in scene.objects:
        if not include_structure and o.is_room_fixed:
            continue
        dist = float(np.linalg.norm(o.center - eye))
        if dist > max_distance:
            continue
        if skip_enclosing and bool(o.obb.contains(eye[None, :], pad=0.10)[0]):
            continue
        uv, z, all_front, on_screen = project_box(o.obb, K, pose_c2w, image_size)
        if not all_front or not on_screen:
            continue
        if box_screen_area(uv) < min_screen_area:
            continue
        frac = 1.0
        if dbuf is not None and occlusion_test:
            frac = box_visible_fraction(o.obb, K, pose_c2w, dbuf)
            if frac < min_visible_fraction:
                continue
        rows.append((o, uv, dist, frac))

    rows.sort(key=lambda t: t[2])                 # nearest first
    if max_objects is not None:
        rows = rows[:max_objects]
    rows.sort(key=lambda t: -t[2])                # far first for painting
    return rows


def box_visible_fraction(obb: OBB, K: np.ndarray, pose_c2w: np.ndarray,
                         dbuf: Optional["DepthBuffer"],
                         tol: float = 0.04) -> float:
    """Fraction of a box's in-front edge length that is not occluded."""
    corners = obb.corners()
    n_vis = n_tot = 0
    for a, b in BOX_EDGES:
        _, visible, in_front = edge_visibility(corners[a], corners[b], K,
                                               pose_c2w, dbuf, tol)
        n_tot += int(in_front.sum())
        n_vis += int((visible & in_front).sum())
    return (n_vis / n_tot) if n_tot else 0.0


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

def draw_box(img: np.ndarray, uv: np.ndarray, colour, thickness: int = 2,
             label: Optional[str] = None, label_scale: float = 0.6) -> None:
    cv2 = _cv2()
    h, w = img.shape[:2]
    pts = uv.astype(np.int32)
    for a, b in BOX_EDGES:
        p, q = tuple(pts[a]), tuple(pts[b])
        cv2.line(img, p, q, colour, thickness, cv2.LINE_AA)
    if label:
        top = pts[np.argmin(pts[:, 1])]
        x = int(np.clip(top[0], 4, w - 8))
        y = int(np.clip(top[1] - 8, 16, h - 6))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                      label_scale, 1)
        cv2.rectangle(img, (x - 3, y - th - 5), (x + tw + 3, y + 4),
                      (24, 24, 24), -1)
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, label_scale,
                    colour, 1, cv2.LINE_AA)


def draw_arrow_3d(img: np.ndarray, origin: np.ndarray, direction: np.ndarray,
                  length: float, K: np.ndarray, pose_c2w: np.ndarray,
                  colour, thickness: int = 3,
                  label: Optional[str] = None) -> None:
    """Project a 3-D arrow, for drawing reference-frame axes into the frame."""
    cv2 = _cv2()
    a = np.asarray(origin, float)
    b = a + length * np.asarray(direction, float)
    pts_c = transform_points(se3_inverse(pose_c2w), np.stack([a, b]))
    if np.any(pts_c[:, 2] < 0.05):
        return
    uv, _, _ = project(pts_c, K)
    p, q = tuple(uv[0].astype(np.int32)), tuple(uv[1].astype(np.int32))
    cv2.arrowedLine(img, p, q, colour, thickness, cv2.LINE_AA, tipLength=0.22)
    if label:
        cv2.putText(img, label, (q[0] + 4, q[1] - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, colour, 1, cv2.LINE_AA)


def draw_caption(img: np.ndarray, lines: Sequence[str],
                 scale: float = 0.6) -> None:
    cv2 = _cv2()
    pad, lh = 8, int(26 * scale / 0.6)
    height = pad * 2 + lh * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], height), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (pad, pad + lh * (i + 1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 235, 235), 1,
                    cv2.LINE_AA)


# --------------------------------------------------------------------------
# frame sources
# --------------------------------------------------------------------------

@dataclass
class FrameSource:
    """RGB + depth + pose + intrinsics for one capture, indexable by frame."""
    poses: np.ndarray                       # (T,4,4) camera-to-world
    Ks: np.ndarray                          # (T,3,3) at RGB resolution
    names: List[str]
    image_size: Tuple[int, int]
    video_path: Optional[str] = None
    depth_reader: Optional[object] = None   # callable(i) -> HxW metres or None

    def __len__(self) -> int:
        return len(self.poses)

    def rgb(self, i: int) -> Optional[np.ndarray]:
        if not self.video_path:
            return None
        cv2 = _cv2()
        cap = cv2.VideoCapture(self.video_path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            return frame if ok else None
        finally:
            cap.release()

    def rgb_many(self, indices: Sequence[int]) -> Dict[int, np.ndarray]:
        """Decode several frames in one pass, in increasing order.

        Seeking an mp4 per frame is slow and, with inter-frame compression, not
        always exact. Reading forward once is both faster and safer.
        """
        if not self.video_path:
            return {}
        cv2 = _cv2()
        want = sorted(set(int(i) for i in indices))
        out: Dict[int, np.ndarray] = {}
        cap = cv2.VideoCapture(self.video_path)
        try:
            pos = 0
            for idx in want:
                if idx - pos > 60 or idx < pos:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    pos = idx
                while pos <= idx:
                    ok, frame = cap.read()
                    if not ok:
                        return out
                    if pos == idx:
                        out[idx] = frame
                    pos += 1
        finally:
            cap.release()
        return out

    def depth(self, i: int) -> Optional[np.ndarray]:
        if self.depth_reader is None:
            return None
        return self.depth_reader(int(i))


def scannetpp_frame_source(root: str, scene_id: str,
                           with_depth: bool = True) -> FrameSource:
    """A `FrameSource` over the ScanNet++ iPhone stream.

    Frame `i` of `rgb.mp4` corresponds to the `i`-th key of
    `pose_intrinsic_imu.json` in sorted order; the two counts match exactly
    (4794 for these scenes) and the overlay is what verifies the correspondence
    rather than assuming it.
    """
    from ..data.scannetpp import (ScanNetPPPaths, depth_frame_offsets,
                                 load_iphone_poses, read_depth_frames)
    paths = ScanNetPPPaths(root, scene_id)
    names, poses, Ks, _, _ = load_iphone_poses(paths, stride=1)

    reader = None
    if with_depth and os.path.exists(paths.iphone_depth):
        try:
            offsets = depth_frame_offsets(paths)

            def reader(i, _p=paths, _o=offsets):
                for _, d in read_depth_frames(_p, [i], _o):
                    return d
                return None
        except Exception:
            reader = None

    return FrameSource(poses=poses, Ks=Ks, names=list(names),
                       image_size=(1920, 1440),
                       video_path=(paths.iphone_rgb
                                   if os.path.exists(paths.iphone_rgb) else None),
                       depth_reader=reader)


# --------------------------------------------------------------------------
# renderers
# --------------------------------------------------------------------------

def render_frame_overlay(scene: Scene, src: FrameSource, frame_index: int,
                         highlight: Optional[Dict[int, str]] = None,
                         rgb: Optional[np.ndarray] = None,
                         max_distance: float = 6.0,
                         include_structure: bool = False,
                         occlusion_test: bool = True,
                         caption: Optional[Sequence[str]] = None,
                         scale: float = 0.5,
                         hidden_style: str = "dashed",
                         label_all: bool = False,
                         max_objects: Optional[int] = 14,
                         label_top: int = 6,
                         fade_distance: bool = True) -> Optional[np.ndarray]:
    """Boxes on one real frame, with occluded edges removed.

    `highlight` maps object id to a colour key. `hidden_style` controls how the
    occluded parts of an edge are drawn -- `dashed` (default), `dim`, or `hide`.
    """
    cv2 = _cv2()
    if rgb is None:
        rgb = src.rgb(frame_index)
    if rgb is None:
        return None
    img = rgb.copy()
    pose = src.poses[frame_index]
    K = src.Ks[frame_index] if src.Ks.ndim == 3 else src.Ks
    depth = src.depth(frame_index) if occlusion_test else None
    dbuf = None
    if occlusion_test:
        if depth is not None:
            dbuf = DepthBuffer.from_sensor(depth, src.image_size)
        elif scene.background is not None and len(scene.background):
            dbuf = DepthBuffer.from_points(scene.background, K, pose,
                                           src.image_size)

    highlight = highlight or {}
    rows = visible_objects(scene, K, pose, src.image_size, depth,
                           max_distance=max_distance,
                           include_structure=include_structure,
                           occlusion_test=occlusion_test, dbuf=dbuf,
                           max_objects=max_objects)
    # anything highlighted is drawn even if the filters would have dropped it
    already = {o.id for o, _, _, _ in rows}
    eye = pose[:3, 3]
    for oid in highlight:
        if oid in already:
            continue
        o = scene.by_id(oid)
        if o is None:
            continue
        uv, _, front, on_screen = project_box(o.obb, K, pose, src.image_size)
        if front and on_screen:
            frac = (box_visible_fraction(o.obb, K, pose, dbuf)
                    if dbuf is not None else 1.0)
            rows.append((o, uv, float(np.linalg.norm(o.center - eye)), frac))
    rows.sort(key=lambda t: -t[2])

    # label only the nearest few unlabelled objects; highlighted ones always
    label_ids = {oid for oid in highlight}
    if label_all:
        nearest = sorted((r for r in rows if r[0].id not in highlight),
                         key=lambda t: t[2])[:label_top]
        label_ids |= {o.id for o, _, _, _ in nearest}

    for o, uv, dist, frac in rows:
        key = highlight.get(o.id, "structure" if o.is_room_fixed else "other")
        important = key in ("target", "anchor", "runner")
        thickness = 3 if key in ("target", "anchor") else 1
        colour = COLOURS.get(key, COLOURS["other"])
        if not important and fade_distance:
            # let the background recede instead of competing with the answer
            t = float(np.clip((dist - 1.0) / max(max_distance - 1.0, 1e-6),
                              0.0, 1.0))
            colour = _dim(colour, 1.0 - 0.55 * t)
        label = None
        if o.id in label_ids:
            label = (f"{o.label} #{o.id}" if important else o.canonical_label)
        draw_box_hidden_line(img, o.obb, K, pose, dbuf, colour, thickness,
                             label, tol=0.04, hidden_style=hidden_style)

    if caption:
        draw_caption(img, list(caption))
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    return img


def render_scene_frames(scene: Scene, src: FrameSource, out_dir: str,
                        n_frames: int = 6,
                        highlight: Optional[Dict[int, str]] = None,
                        max_distance: float = 6.0,
                        include_structure: bool = False,
                        scale: float = 0.5,
                        indices: Optional[Sequence[int]] = None,
                        hidden_style: str = "dashed",
                        label_all: bool = True,
                        max_objects: Optional[int] = 14) -> List[str]:
    """Overlay boxes on frames spread through the capture. Returns file paths."""
    cv2 = _cv2()
    os.makedirs(out_dir, exist_ok=True)
    if indices is None:
        if len(src) == 0:
            return []
        indices = np.linspace(0, len(src) - 1, num=min(n_frames, len(src)),
                              dtype=int).tolist()
    frames = src.rgb_many(indices)
    written: List[str] = []
    for i in indices:
        rgb = frames.get(int(i))
        if rgb is None:
            continue
        cap = [f"{scene.scene_id}  frame {i}"
               + (f" ({src.names[i]})" if i < len(src.names) else ""),
               "3D boxes projected with the dataset's own pose + intrinsics",
               "solid = visible, faint dashes = occluded by the scene depth",
               f"nearest {max_objects} objects shown"]
        img = render_frame_overlay(scene, src, int(i), highlight, rgb,
                                   max_distance, include_structure,
                                   caption=cap, scale=scale,
                                   hidden_style=hidden_style,
                                   label_all=label_all,
                                   max_objects=max_objects)
        if img is None:
            continue
        p = os.path.join(out_dir, f"{scene.scene_id}_frame{int(i):05d}.jpg")
        cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        written.append(p)
    return written


def best_joint_view(src: FrameSource, obbs: Sequence[OBB],
                    up: np.ndarray = np.array([0.0, 0.0, 1.0]),
                    stride: int = 15, refine_top: int = 6,
                    scene_background: Optional[np.ndarray] = None) -> int:
    """A frame that sees *all* of `obbs` at once.

    Picking the best view of the answer alone often leaves the anchor out of
    frame, and then the picture cannot be used to check the relation -- which is
    the only reason to draw it. This scores frames on the worst-seen box, so a
    frame that shows the target beautifully and the anchor not at all loses to
    one showing both.

    Two passes: a cheap geometric score over every `stride`-th frame, then a
    depth-based occlusion check on the best few, because each depth read costs
    an LZ4 decode.
    """
    if len(src) == 0 or not obbs:
        return -1
    w, h = src.image_size
    cand: List[Tuple[float, int]] = []
    for i in range(0, len(src), max(1, stride)):
        pose = src.poses[i]
        K = src.Ks[i] if src.Ks.ndim == 3 else src.Ks
        eye = pose[:3, 3]
        worst = 1e9
        for obb in obbs:
            uv, z, all_front, on_screen = project_box(obb, K, pose, (w, h))
            if not all_front:
                worst = 0.0
                break
            centre_uv = uv.mean(axis=0)
            inside = (0 <= centre_uv[0] < w) and (0 <= centre_uv[1] < h)
            d = float(np.linalg.norm(obb.center - eye))
            radius = max(0.05, float(np.linalg.norm(obb.half)))
            if d < 1.25 * radius:
                worst = 0.0
                break
            # want it in frame and at a sane apparent size
            fit = np.exp(-0.5 * (np.log(max(d, 1e-6)
                                        / max(radius / 0.18, 0.5)) / 0.75) ** 2)
            score = fit * (1.0 if inside else 0.25)
            worst = min(worst, float(score))
        if worst > 0.0:
            cand.append((worst, i))
    if not cand:
        return -1
    cand.sort(key=lambda t: -t[0])

    best_i, best_score = cand[0][1], -1.0
    for _, i in cand[:refine_top]:
        pose = src.poses[i]
        K = src.Ks[i] if src.Ks.ndim == 3 else src.Ks
        depth = src.depth(i)
        dbuf = (DepthBuffer.from_sensor(depth, src.image_size)
                if depth is not None else
                (DepthBuffer.from_points(scene_background, K, pose,
                                         src.image_size)
                 if scene_background is not None else None))
        fracs = [box_visible_fraction(o, K, pose, dbuf) for o in obbs]
        score = float(min(fracs))
        if score > best_score:
            best_i, best_score = i, score
    return int(best_i)


def render_query_overlay(scene: Scene, src: FrameSource, resolution,
                         out_path: str, scale: float = 0.6,
                         draw_axes: bool = True,
                         max_distance: float = 7.0,
                         hidden_style: str = "dashed") -> Optional[str]:
    """One resolution drawn on the frame that best sees its target.

    Target in amber, anchors in red, runner-up in blue, and the reference frame's
    right/front axes drawn at the anchor -- so a left/right answer can be checked
    against the photograph rather than against a number.
    """
    cv2 = _cv2()
    if resolution.target is None or len(src) == 0:
        return None
    from ..scenegraph.objects import CameraTrajectory
    traj = CameraTrajectory(src.poses, src.Ks, src.image_size, src.names)

    highlight: Dict[int, str] = {resolution.target.id: "target"}
    for a in resolution.anchors:
        if a.obj is not None:
            highlight[a.obj.id] = "anchor"
    if len(resolution.candidates) > 1:
        highlight.setdefault(resolution.candidates[1].obj.id, "runner")

    # prefer a frame showing the answer *and* its anchors: a picture with only
    # the answer in it cannot be used to check the relation
    want = [resolution.target.obb]
    want += [a.obj.obb for a in resolution.anchors if a.obj is not None]
    i = best_joint_view(src, want, scene.up,
                        scene_background=scene.background)
    if i < 0:
        i = traj.best_view(resolution.target.obb, scene.up)
    if i < 0:
        i = traj.nearest_index(resolution.target.center)

    rgb = src.rgb(int(i))
    if rgb is None:
        return None
    frame_kind = resolution.frame_used or "frame-free"
    lines = [f'"{resolution.query.text}"',
             f"answer: {resolution.target.label} #{resolution.target.id}"
             f"   frame: {frame_kind}",
             "amber = answer, red = anchor, blue = runner-up   "
             "solid = visible, faint dashes = occluded"]
    if resolution.ambiguity.ambiguous:
        lines.append("ambiguous: " + ", ".join(resolution.ambiguity.kinds))
    img = render_frame_overlay(scene, src, int(i), highlight, rgb,
                               max_distance, caption=lines, scale=1.0,
                               hidden_style=hidden_style, label_all=False)
    if img is None:
        return None

    if draw_axes and resolution.frame_decision is not None:
        fr = resolution.frame_decision.frame
        anchor = next((a.obj for a in resolution.anchors if a.obj is not None),
                      None)
        if fr is not None and fr.available and anchor is not None:
            K = src.Ks[i] if src.Ks.ndim == 3 else src.Ks
            org = anchor.center
            L = max(0.3, 0.6 * float(np.max(anchor.extent)))
            draw_arrow_3d(img, org, fr.right, L, K, src.poses[i],
                          COLOURS["right"], 3, "right")
            draw_arrow_3d(img, org, fr.front, L, K, src.poses[i],
                          COLOURS["front"], 3, "front")

    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_path


def render_topdown(scene: Scene, out_path: str,
                   highlight: Optional[Dict[int, str]] = None,
                   size: int = 1000, margin: int = 40,
                   viewpoint: Optional[np.ndarray] = None,
                   frame=None, caption: Optional[Sequence[str]] = None
                   ) -> str:
    """Plan view with box footprints. The quickest check on a left/right answer.

    Lateral relations are a plan-view property, so if an answer looks wrong in
    this picture it *is* wrong, without having to reason about the camera.
    """
    cv2 = _cv2()
    highlight = highlight or {}
    pts = scene.background
    if pts is None or not len(pts):
        pts = np.concatenate([o.cloud() for o in scene.objects], axis=0)
    lo = pts[:, :2].min(axis=0)
    hi = pts[:, :2].max(axis=0)
    span = np.maximum(hi - lo, 1e-3)
    s = (size - 2 * margin) / float(span.max())
    W = int(span[0] * s) + 2 * margin
    H = int(span[1] * s) + 2 * margin
    img = np.full((H, W, 3), 20, np.uint8)

    def to_px(xy):
        p = (np.asarray(xy, float)[:2] - lo) * s
        return int(margin + p[0]), int(H - margin - p[1])

    step = max(1, len(pts) // 60000)
    for p in pts[::step]:
        x, y = to_px(p[:2])
        if 0 <= x < W and 0 <= y < H:
            img[y, x] = (70, 70, 70)

    for o in sorted(scene.objects, key=lambda o: o.id in highlight):
        if o.is_room_fixed and o.id not in highlight:
            continue
        key = highlight.get(o.id, "other")
        colour = COLOURS.get(key, COLOURS["other"])
        if key == "other":
            colour = _dim(colour, 0.75)
        c = o.obb.corners()
        base = c[np.argsort(c[:, 2])[:4]]
        order = np.argsort(np.arctan2(base[:, 1] - o.center[1],
                                      base[:, 0] - o.center[0]))
        poly = np.array([to_px(base[i][:2]) for i in order], np.int32)
        cv2.polylines(img, [poly], True, colour,
                      3 if key in ("target", "anchor") else 1, cv2.LINE_AA)
        if key in ("target", "anchor", "runner"):
            x, y = to_px(o.center[:2])
            cv2.putText(img, f"{o.label} #{o.id}", (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

    if viewpoint is not None:
        x, y = to_px(np.asarray(viewpoint, float)[:2])
        cv2.circle(img, (x, y), 7, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(img, "viewpoint", (x + 10, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
    if frame is not None and getattr(frame, "available", False):
        org = frame.origin
        for vec, key, name in ((frame.right, "right", "right"),
                               (frame.front, "front", "front")):
            a = to_px(org[:2])
            b = to_px((org + 0.8 * vec)[:2])
            cv2.arrowedLine(img, a, b, COLOURS[key], 3, cv2.LINE_AA,
                            tipLength=0.25)
            cv2.putText(img, name, (b[0] + 4, b[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOURS[key], 1,
                        cv2.LINE_AA)

    lines = list(caption or [])
    lines.insert(0, f"{scene.scene_id}  top-down  "
                    f"+x right, +y up, {span[0]:.1f} x {span[1]:.1f} m")
    draw_caption(img, lines, 0.55)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    cv2.imwrite(out_path, img)
    return out_path


def render_pointcloud_view(scene: Scene, out_path: str,
                           eye: Optional[np.ndarray] = None,
                           look_at: Optional[np.ndarray] = None,
                           size: Tuple[int, int] = (1280, 960),
                           fov_deg: float = 60.0,
                           highlight: Optional[Dict[int, str]] = None,
                           caption: Optional[Sequence[str]] = None,
                           hidden_style: str = "dashed") -> str:
    """Software point splat from an arbitrary viewpoint, with boxes.

    The fallback for datasets and synthetic rooms with no RGB, so the same visual
    check works everywhere.
    """
    cv2 = _cv2()
    from ..geom.transforms import intrinsics_matrix, normalize, se3
    highlight = highlight or {}
    w, h = size
    pts = scene.background
    cols = scene.background_color
    if pts is None or not len(pts):
        pts = np.concatenate([o.cloud() for o in scene.objects], axis=0)
        cols = None
    centre = pts.mean(axis=0) if look_at is None else np.asarray(look_at, float)
    if eye is None:
        ext = pts.max(axis=0) - pts.min(axis=0)
        eye = centre + np.array([0.0, -1.1 * max(ext[0], ext[1]),
                                 0.55 * max(ext[2], 1.5)])
    eye = np.asarray(eye, float)

    f = normalize(centre - eye)
    # OpenCV camera-to-world columns are (right, down, forward). The image's
    # right is `f x up`, NOT `up x f`: both are valid rotations with det +1,
    # which is why the wrong one went unnoticed, but it renders the image
    # rotated 180 degrees -- and a left/right check on a mirrored picture is
    # worse than no picture.
    right = normalize(np.cross(f, np.array([0.0, 0.0, 1.0])))
    if not np.any(right):
        right = np.array([1.0, 0.0, 0.0])
    down = normalize(np.cross(f, right))
    pose = se3(np.stack([right, down, f], axis=1), eye)
    fx = 0.5 * w / np.tan(np.deg2rad(fov_deg) / 2.0)
    K = intrinsics_matrix(fx, fx, w / 2.0, h / 2.0)

    img = np.full((h, w, 3), 18, np.uint8)
    zbuf = np.full((h, w), np.inf)
    cam = transform_points(se3_inverse(pose), pts)
    uv, z, valid = project(cam, K)
    u = np.round(uv[:, 0]).astype(np.int64)
    v = np.round(uv[:, 1]).astype(np.int64)
    ok = valid & (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0.05)
    order = np.argsort(-z[ok])                 # far to near
    uu, vv, zz = u[ok][order], v[ok][order], z[ok][order]
    if cols is not None and len(cols) == len(pts):
        cc = cols[ok][order][:, ::-1]          # RGB -> BGR
    else:
        shade = np.clip(255 - 12 * zz, 60, 220).astype(np.uint8)
        cc = np.stack([shade] * 3, axis=1)
    img[vv, uu] = cc
    zbuf[vv, uu] = zz

    # The splatter already produced a z-buffer, so the same hidden-line test
    # works here and synthetic rooms get occlusion for free.
    zfilled = np.where(np.isfinite(zbuf), zbuf, 0.0).astype(np.float32)
    dbuf = DepthBuffer(zfilled, (w, h), "points")
    dbuf.depth = _close_holes(zfilled, radius=2)

    for o in sorted(scene.objects,
                    key=lambda o: -float(np.linalg.norm(o.center - eye))):
        if o.is_room_fixed and o.id not in highlight:
            continue
        key = highlight.get(o.id, "other")
        cuv, cz, front, on = project_box(o.obb, K, pose, (w, h))
        if not front or not on:
            continue
        draw_box_hidden_line(img, o.obb, K, pose, dbuf,
                             COLOURS.get(key, COLOURS["other"]),
                             3 if key in ("target", "anchor") else 1,
                             f"{o.label} #{o.id}" if key != "other" else None,
                             tol=0.06, hidden_style=hidden_style)

    lines = list(caption or [])
    lines.insert(0, f"{scene.scene_id}  rendered view  "
                    f"eye ({eye[0]:.1f}, {eye[1]:.1f}, {eye[2]:.1f})")
    draw_caption(img, lines, 0.55)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    cv2.imwrite(out_path, img)
    return out_path

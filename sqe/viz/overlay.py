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
                    min_screen_area: float = 400.0,
                    include_structure: bool = False,
                    occlusion_test: bool = True) -> List[Tuple[Object3D, np.ndarray]]:
    """Objects worth drawing on this frame, with their projected corners."""
    eye = pose_c2w[:3, 3]
    out = []
    for o in scene.objects:
        if not include_structure and o.is_room_fixed:
            continue
        if float(np.linalg.norm(o.center - eye)) > max_distance:
            continue
        uv, z, all_front, on_screen = project_box(o.obb, K, pose_c2w, image_size)
        if not all_front or not on_screen:
            continue
        if box_screen_area(uv) < min_screen_area:
            continue
        if occlusion_test and is_occluded(o.obb, K, pose_c2w, depth):
            continue
        out.append((o, uv))
    # far objects first, so near ones draw over them
    out.sort(key=lambda t: -float(np.linalg.norm(t[0].center - eye)))
    return out


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
                         scale: float = 0.5) -> Optional[np.ndarray]:
    """Boxes on one real frame. `highlight` maps object id to a colour key."""
    cv2 = _cv2()
    if rgb is None:
        rgb = src.rgb(frame_index)
    if rgb is None:
        return None
    img = rgb.copy()
    pose = src.poses[frame_index]
    K = src.Ks[frame_index] if src.Ks.ndim == 3 else src.Ks
    depth = src.depth(frame_index) if occlusion_test else None

    highlight = highlight or {}
    drawn = visible_objects(scene, K, pose, src.image_size, depth,
                            max_distance=max_distance,
                            include_structure=include_structure,
                            occlusion_test=occlusion_test)
    # anything highlighted is drawn even if the filters would have dropped it
    already = {o.id for o, _ in drawn}
    for oid in highlight:
        if oid in already:
            continue
        o = scene.by_id(oid)
        if o is None:
            continue
        uv, _, front, on_screen = project_box(o.obb, K, pose, src.image_size)
        if front and on_screen:
            drawn.append((o, uv))

    for o, uv in drawn:
        key = highlight.get(o.id, "structure" if o.is_room_fixed else "other")
        thickness = 3 if key in ("target", "anchor") else 1
        label = (f"{o.label} #{o.id}" if key in ("target", "anchor", "runner")
                 else o.canonical_label)
        draw_box(img, uv, COLOURS.get(key, COLOURS["other"]), thickness, label)

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
                        indices: Optional[Sequence[int]] = None) -> List[str]:
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
               "3D boxes projected with the dataset's own pose + intrinsics"]
        img = render_frame_overlay(scene, src, int(i), highlight, rgb,
                                   max_distance, include_structure,
                                   caption=cap, scale=scale)
        if img is None:
            continue
        p = os.path.join(out_dir, f"{scene.scene_id}_frame{int(i):05d}.jpg")
        cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        written.append(p)
    return written


def render_query_overlay(scene: Scene, src: FrameSource, resolution,
                         out_path: str, scale: float = 0.6,
                         draw_axes: bool = True,
                         max_distance: float = 7.0) -> Optional[str]:
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
    i = traj.best_view(resolution.target.obb, scene.up)
    if i < 0:
        i = traj.nearest_index(resolution.target.center)

    highlight: Dict[int, str] = {resolution.target.id: "target"}
    for a in resolution.anchors:
        if a.obj is not None:
            highlight[a.obj.id] = "anchor"
    if len(resolution.candidates) > 1:
        highlight.setdefault(resolution.candidates[1].obj.id, "runner")

    rgb = src.rgb(int(i))
    if rgb is None:
        return None
    frame_kind = resolution.frame_used or "frame-free"
    lines = [f'"{resolution.query.text}"',
             f"answer: {resolution.target.label} #{resolution.target.id}"
             f"   frame: {frame_kind}",
             f"amber = answer, red = anchor, blue = runner-up"]
    if resolution.ambiguity.ambiguous:
        lines.append("ambiguous: " + ", ".join(resolution.ambiguity.kinds))
    img = render_frame_overlay(scene, src, int(i), highlight, rgb,
                               max_distance, caption=lines, scale=1.0)
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
                           caption: Optional[Sequence[str]] = None) -> str:
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
    right = normalize(np.cross(np.array([0.0, 0.0, 1.0]), f))
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

    for o in sorted(scene.objects,
                    key=lambda o: -float(np.linalg.norm(o.center - eye))):
        if o.is_room_fixed and o.id not in highlight:
            continue
        key = highlight.get(o.id, "other")
        cuv, cz, front, on = project_box(o.obb, K, pose, (w, h))
        if not front or not on:
            continue
        draw_box(img, cuv, COLOURS.get(key, COLOURS["other"]),
                 3 if key in ("target", "anchor") else 1,
                 f"{o.label} #{o.id}" if key != "other" else None)

    lines = list(caption or [])
    lines.insert(0, f"{scene.scene_id}  rendered view  "
                    f"eye ({eye[0]:.1f}, {eye[1]:.1f}, {eye[2]:.1f})")
    draw_caption(img, lines, 0.55)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    cv2.imwrite(out_path, img)
    return out_path

"""An animated GIF of the same sentence resolving differently per frame.

The point of the whole project in three seconds: one query, one scene, one
camera, and the highlighted object jumps when the reference frame changes. It
mirrors what the web viewer does when you click a row of its frame table, and it
is rendered offline rather than screen-captured -- higher resolution, no browser
furniture, and controllable timing. It is a rendering of the same computation the
viewer performs, not a recording of the viewer.

The frame is chosen so that the anchor *and* every candidate answer are visible
at once. A picture showing only one of the two answers cannot make the point.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..frames.policy import ViewpointSpec
from ..query.parser_rules import parse
from ..query.resolver import Resolver
from ..relations.base import RelationConfig
from ..scenegraph.objects import Scene
from .overlay import (COLOURS, DepthBuffer, FrameSource, best_joint_view,
                      box_visible_fraction, draw_arrow_3d,
                      draw_box_hidden_line, project_box, _cv2, _dim)

PANEL_W = 430
ROW_H = 40

#: How each frame reads in plain words. The GIF has to explain itself to someone
#: who has read nothing about the project.
FRAME_GLOSS = {
    "egocentric": "from where I'm standing",
    "egocentric_bearing": "as it appears in my view",
    "egocentric_image": "left/right in the picture",
    "intrinsic": "the side the object itself faces",
    "addressee": "as seen by someone facing it",
    "world": "relative to the room's axes",
}

#: Plain-words gloss specialised by relation, where it differs.
FRONT_GLOSS = {
    "egocentric": "between me and it",
    "intrinsic": "the side it faces",
    "addressee": "as seen by someone facing it",
    "world": "towards the room's front",
}


def _projected_length(origin: np.ndarray, direction: np.ndarray, length: float,
                      K: np.ndarray, pose_c2w: np.ndarray) -> float:
    """Screen length in pixels of a 3-D arrow, or 0 if it is behind the camera."""
    from ..geom.transforms import project, se3_inverse, transform_points
    a = np.asarray(origin, float)
    b = a + length * np.asarray(direction, float)
    cam = transform_points(se3_inverse(pose_c2w), np.stack([a, b]))
    if np.any(cam[:, 2] < 0.05):
        return 0.0
    uv, _, _ = project(cam, K)
    return float(np.linalg.norm(uv[1] - uv[0]))


def _panel(img: np.ndarray, rows: Sequence[Tuple[str, str]], selected: int,
           pressed: Optional[int] = None, title: str = "answer under each frame"):
    """Draw the viewer's frame table, with one row selected.

    `rows` is a list of (frame name, answer label). `pressed` draws the row as
    if it were being clicked, which is what makes the loop read as an
    interaction rather than a slideshow.
    """
    cv2 = _cv2()
    h, w = img.shape[:2]
    x0 = w - PANEL_W - 26
    y0 = 100
    ph = 44 + ROW_H * len(rows) + 14

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + PANEL_W, y0 + ph), (18, 20, 24), -1)
    cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + PANEL_W, y0 + ph), (60, 66, 76), 1)
    cv2.putText(img, title, (x0 + 14, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (150, 158, 170), 1, cv2.LINE_AA)

    for i, (name, answer) in enumerate(rows):
        ry = y0 + 44 + ROW_H * i
        is_sel = (i == selected)
        is_press = (pressed == i)
        if is_sel:
            cv2.rectangle(img, (x0 + 6, ry + 3), (x0 + PANEL_W - 6, ry + ROW_H - 5),
                          (34, 46, 60), -1)
            cv2.rectangle(img, (x0 + 6, ry + 3), (x0 + 10, ry + ROW_H - 5),
                          COLOURS["target"], -1)
        elif is_press:
            cv2.rectangle(img, (x0 + 6, ry + 3), (x0 + PANEL_W - 6, ry + ROW_H - 5),
                          (44, 48, 56), -1)
        colour = COLOURS["target"] if is_sel else (170, 176, 186)
        cv2.putText(img, name, (x0 + 22, ry + 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, colour, 2 if is_sel else 1, cv2.LINE_AA)
        cv2.putText(img, answer, (x0 + 210, ry + 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, colour, 2 if is_sel else 1, cv2.LINE_AA)
        if is_press:
            _cursor(img, x0 + 250, ry + 30)
    return x0, y0


def _highlight(img: np.ndarray, centre_uv: np.ndarray, radius_px: float,
               colour, from_xy: Optional[Tuple[int, int]] = None):
    """Ring the selected answer, and tie it back to the clicked row.

    Needed because the answer is sometimes a small object near the frame edge --
    a bottle on a kitchen counter -- and an amber wireframe alone is easy to miss
    at the size a GIF is actually viewed. The ring says "this one"; the connector
    says "because of that row".
    """
    cv2 = _cv2()
    h, w = img.shape[:2]
    cx, cy = int(round(centre_uv[0])), int(round(centre_uv[1]))
    r = int(max(22, min(120, radius_px * 1.5)))
    if from_xy is not None:
        overlay = img.copy()
        cv2.line(overlay, from_xy, (cx, cy), colour, 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), r, colour, 3, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), r + 5, colour, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)


def _cursor(img: np.ndarray, x: int, y: int):
    """A small pointer, so a still frame reads as a click."""
    cv2 = _cv2()
    pts = np.array([[x, y], [x, y + 20], [x + 5, y + 15], [x + 9, y + 23],
                    [x + 12, y + 21], [x + 8, y + 14], [x + 15, y + 13]],
                   np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255), cv2.LINE_AA)
    cv2.polylines(img, [pts], True, (20, 20, 20), 1, cv2.LINE_AA)


def _caption(img: np.ndarray, lines: Sequence[str]):
    cv2 = _cv2()
    pad, lh = 16, 34
    height = pad * 2 + lh * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], height), (14, 16, 20), -1)
    cv2.addWeighted(overlay, 0.86, img, 0.14, 0, img)
    for i, line in enumerate(lines):
        big = i == 0
        cv2.putText(img, line, (pad + 4, pad + lh * (i + 1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85 if big else 0.58,
                    (245, 245, 245) if big else (165, 172, 184),
                    2 if big else 1, cv2.LINE_AA)


def render_state(scene: Scene, src: FrameSource, frame_index: int,
                 anchor, answers: Sequence[Tuple[str, int]], selected: int,
                 query_text: str, pressed: Optional[int] = None,
                 width: int = 1000, max_distance: float = 8.0,
                 max_context: int = 10,
                 relation: Optional[str] = None) -> np.ndarray:
    """One still: the answer under `answers[selected]` highlighted."""
    cv2 = _cv2()
    rgb = src.rgb(frame_index)
    if rgb is None:
        raise RuntimeError(f"no RGB for frame {frame_index}")
    img = rgb.copy()
    pose = src.poses[frame_index]
    K = src.Ks[frame_index] if src.Ks.ndim == 3 else src.Ks
    depth = src.depth(frame_index)
    dbuf = (DepthBuffer.from_sensor(depth, src.image_size) if depth is not None
            else DepthBuffer.from_points(scene.background, K, pose,
                                         src.image_size))

    sel_id = answers[selected][1]
    others = [oid for _, oid in answers if oid != sel_id]

    # context boxes, faint and capped -- they should say "this is a scene graph",
    # not compete with the answer
    eye = pose[:3, 3]
    context = []
    for o in scene.objects:
        if o.is_room_fixed or o.id == anchor.id or o.id == sel_id:
            continue
        if o.id in others:
            continue
        d = float(np.linalg.norm(o.center - eye))
        if d > max_distance:
            continue
        uv, _, front, on = project_box(o.obb, K, pose, src.image_size)
        if not (front and on):
            continue
        if box_visible_fraction(o.obb, K, pose, dbuf,
                                image_size=src.image_size) < 0.15:
            continue
        context.append((d, o))
    context.sort(key=lambda t: t[0])
    for _, o in context[:max_context]:
        draw_box_hidden_line(img, o.obb, K, pose, dbuf,
                             _dim(COLOURS["other"], 0.45), 1, None)

    # the not-selected candidate, so the viewer can see where it will jump to
    for oid in others:
        o = scene.by_id(oid)
        if o is None:
            continue
        draw_box_hidden_line(img, o.obb, K, pose, dbuf,
                             _dim(COLOURS["runner"], 0.8), 2,
                             f"{o.label} #{o.id}")

    # anchor, and the selected frame's axes read at it
    draw_box_hidden_line(img, anchor.obb, K, pose, dbuf, COLOURS["anchor"], 3,
                         f"anchor: {anchor.label} #{anchor.id}")
    kind = answers[selected][0]
    from ..frames.policy import build_frames
    frames, _ = build_frames(scene, anchor, (kind,),
                             ViewpointSpec(mode="index", index=frame_index))
    fr = frames.get(kind)
    if fr is not None and fr.available:
        L = max(0.45, 0.75 * float(np.max(anchor.extent)))
        for vec, ckey, name in ((fr.right, "right", "right"),
                                (fr.front, "front", "front")):
            # An axis pointing almost straight at the camera projects to a few
            # pixels; drawing it as an arrow is worse than not drawing it.
            if _projected_length(anchor.center, vec, L, K, pose) < 28.0:
                continue
            draw_arrow_3d(img, anchor.center, vec, L, K, pose,
                          COLOURS[ckey], 4, name)

    # the answer
    sel = scene.by_id(sel_id)
    draw_box_hidden_line(img, sel.obb, K, pose, dbuf, COLOURS["target"], 4,
                         f"{sel.label} #{sel.id}", label_scale=0.72)

    gloss = FRONT_GLOSS if relation in ("front", "behind") else FRAME_GLOSS
    legend = "     ".join(
        f"{k}: {gloss.get(k, '')}" for k, _ in answers)
    _caption(img, [f'"{query_text}"', legend,
                   "same scene, same camera, same geometry - only the "
                   "reference frame changes"])
    rows = [(k, f"{scene.by_id(i).label} #{i}") for k, i in answers]
    px0, py0 = _panel(img, rows, selected, pressed)

    # ring the answer and tie it to the row that selected it
    suv, _, sfront, son = project_box(sel.obb, K, pose, src.image_size)
    if sfront and son:
        ctr = suv.mean(axis=0)
        rad = 0.5 * float(max(suv[:, 0].max() - suv[:, 0].min(),
                              suv[:, 1].max() - suv[:, 1].min()))
        row_anchor = (px0 + 8, py0 + 44 + ROW_H * selected + ROW_H // 2)
        _highlight(img, ctr, rad, COLOURS["target"], row_anchor)

    scale = width / img.shape[1]
    return cv2.resize(img, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA)


def frame_switch_gif(scene: Scene, src: FrameSource, query_text: str,
                     out_path: str, cfg: Optional[RelationConfig] = None,
                     kinds: Sequence[str] = ("egocentric", "intrinsic"),
                     width: int = 1000, hold_ms: int = 1500,
                     click_ms: int = 260, colours: int = 128,
                     verbose: bool = True) -> Optional[str]:
    """Write the GIF. Returns the path, or None if the query is not frame-split."""
    from PIL import Image
    cfg = cfg or RelationConfig.load()
    r = Resolver(scene, cfg)
    res = r.resolve(parse(query_text))
    anchor = next((a.obj for a in res.anchors if a.obj is not None), None)
    if anchor is None:
        if verbose:
            print("  no anchor resolved")
        return None

    answers: List[Tuple[str, int]] = []
    for k in kinds:
        v = res.frame_answers.get(k)
        if v is not None and v not in [i for _, i in answers]:
            answers.append((k, v))
    if len(answers) < 2:
        if verbose:
            print(f"  frames do not disagree on this query: "
                  f"{res.frame_answers}")
        return None

    objs = [scene.by_id(i) for _, i in answers] + [anchor]
    idx = best_joint_view(src, [o.obb for o in objs], scene.up,
                          scene_background=scene.background)
    if idx < 0:
        if verbose:
            print("  no frame sees the anchor and both answers together")
        return None
    if verbose:
        print(f"  frame {idx}, answers {answers}, anchor #{anchor.id}")

    relation = res.query.primary_relation
    stills: List[np.ndarray] = []
    durations: List[int] = []
    n = len(answers)
    for i in range(n):
        stills.append(render_state(scene, src, idx, anchor, answers, i,
                                   query_text, None, width,
                                   relation=relation))
        durations.append(hold_ms)
        nxt = (i + 1) % n
        stills.append(render_state(scene, src, idx, anchor, answers, i,
                                   query_text, nxt, width,
                                   relation=relation))
        durations.append(click_ms)

    pil = [Image.fromarray(s[:, :, ::-1]).convert(
        "P", palette=Image.ADAPTIVE, colors=colours) for s in stills]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    pil[0].save(out_path, save_all=True, append_images=pil[1:],
                duration=durations, loop=0, optimize=True, disposal=2)
    if verbose:
        kb = os.path.getsize(out_path) / 1024.0
        print(f"  wrote {out_path} ({kb:.0f} KB, {len(pil)} frames, "
              f"{width}px wide)")
    return out_path

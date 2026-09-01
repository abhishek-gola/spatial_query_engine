"""The shareable version of the frame-switch GIF, composed for a feed.

`animate.py` renders a diagnostic: every participant boxed, the anchor's axes
drawn, occluded edges dashed so the geometry is auditable. That is the right
picture to debug with and the wrong picture to post. A 1100 px render arrives at
roughly 550 px wide in a feed, so a 16 px title lands at 8 px and the sentence --
which is the entire content of the image -- becomes unreadable, while a hairline
ring around the answer disappears altogether.

So this module composes the same computation for the other job:

* the query sentence set as large as will fit on one line, because it *is* the
  content;
* no anchor wireframe and no frame axes -- long lines sprawling to a corner and a
  green arrow running off the edge mean nothing to a reader who has seen no
  README, and they compete with the two boxes that carry the argument;
* the answer marked with a filled glow and a heavy stroke, not a hairline;
* cropped to the candidates, so none of the canvas is spent on furniture that is
  identical in both states.

Everything after the crop is drawn in **final** pixel space, so type and stroke
weights are chosen for the width actually delivered instead of surviving an
arbitrary downscale. Strokes drawn before the crop are pre-divided by the scale
the crop implies, so they land at the intended weight.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..query.parser_rules import parse
from ..query.resolver import Resolver
from ..relations.base import RelationConfig
from ..scenegraph.objects import Scene
from .animate import FRAME_GLOSS, FRONT_GLOSS, ORDINAL_GLOSS
from .overlay import (COLOURS, DepthBuffer, FrameSource, best_joint_view,
                      box_visible_fraction, draw_box_hidden_line, project_box,
                      _cv2, _dim)

#: How strongly the answer's fill is blended over the photograph.
GLOW_ALPHA = 0.32

#: Fraction of the final height reserved above the candidates for the caption.
HEADROOM = 0.26

#: Crop width as a multiple of the candidates' bounding box. Tight enough that
#: no large region of the frame is identical in both states.
ZOOM = 2.15


def best_poster_view(src: FrameSource, scene: Scene, anchor,
                     answer_ids: Sequence[int], aspect: float = 4.0 / 3.0,
                     zoom: float = None, stride: int = 1,
                     verbose: bool = False) -> int:
    """Pick the frame that makes the best *poster*, which is not the same frame
    that makes the best diagnostic.

    `best_joint_view` maximises how much of every participant is visible. That is
    the right objective for checking geometry and the wrong one here: it will
    happily choose a view where the two candidates are crammed together in one
    corner and half the image is floor. Cropping cannot rescue that -- if the
    participants already span most of the frame, there is nothing to crop away.

    So this scores what the composition actually needs:

    * the anchor **must** be visible. The sentence names it, and for a reader to
      see that the object's own left is a real reading, they have to be able to
      see which way it faces.
    * the two candidates must be separated on screen and must not overlap. Two
      boxes on top of each other cannot show "different objects".
    * everything must fit in well under the frame, so the crop has slack to work
      with and the picture is not mostly furniture.
    """
    zoom = ZOOM if zoom is None else zoom
    W, H = src.image_size
    objs = [scene.by_id(i) for i in answer_ids]
    best, best_i = -1.0, -1
    for i in range(0, len(src.poses), max(1, stride)):
        pose = src.poses[i]
        K = src.Ks[i] if src.Ks.ndim == 3 else src.Ks
        uvs, ok = [], True
        for o in objs + [anchor]:
            uv, _, front, _ = project_box(o.obb, K, pose, src.image_size)
            if not front or uv[:, 0].min() < 0 or uv[:, 0].max() > W \
                    or uv[:, 1].min() < 0 or uv[:, 1].max() > H:
                ok = False
                break
            uvs.append(uv)
        if not ok:
            continue
        cand = np.concatenate(uvs[:len(objs)], axis=0)
        uw = float(np.ptp(cand[:, 0])) / W
        uh = float(np.ptp(cand[:, 1])) / H
        if uw > 0.66 or uh > 0.66:
            continue          # nothing left for the crop to remove
        # the crop is taken on the candidates, so test the anchor against the
        # crop the poster will actually use
        rect = crop_rect(uvs[:len(objs)], src.image_size, zoom=zoom,
                         aspect=aspect)
        if _fraction_inside(uvs[-1], rect) < 0.20:
            continue
        c0, c1 = uvs[0].mean(axis=0), uvs[1].mean(axis=0)
        sep = float(np.linalg.norm(c0 - c1))
        if sep < 0.10 * W:
            continue
        if _hulls_overlap(uvs[0], uvs[1]):
            continue
        depth = src.depth(i)
        dbuf = (DepthBuffer.from_sensor(depth, src.image_size)
                if depth is not None
                else DepthBuffer.from_points(scene.background, K, pose,
                                            src.image_size))
        vis = [box_visible_fraction(o.obb, K, pose, dbuf,
                                    image_size=src.image_size)
               for o in objs + [anchor]]
        if min(vis[:len(objs)]) < 0.40 or vis[-1] < 0.25:
            continue
        size = float(np.mean([max(np.ptp(u[:, 0]), np.ptp(u[:, 1]))
                              for u in uvs[:len(objs)]]))
        score = (min(vis[:len(objs)])
                 * min(1.0, sep / (0.18 * W))
                 * min(1.0, size / (0.13 * W))
                 * (1.0 - 0.5 * max(uw, uh)))
        if score > best:
            best, best_i = score, i
    if verbose and best_i >= 0:
        print(f"  poster view: frame {best_i} (score {best:.3f})")
    return best_i


def _fraction_inside(uv: np.ndarray, rect: Tuple[int, int, int, int]) -> float:
    """How much of a projected box lands inside a crop rectangle, by area."""
    cv2 = _cv2()
    x, y, w, h = rect
    box = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]],
                   np.float32)
    hull = _hull(uv)
    area = cv2.contourArea(hull)
    if area <= 0:
        # a degenerate projection -- a whiteboard seen edge-on -- still counts as
        # present if its centre is in frame
        c = np.asarray(uv, float).reshape(-1, 2).mean(axis=0)
        return 1.0 if (x <= c[0] <= x + w and y <= c[1] <= y + h) else 0.0
    inter, _ = cv2.intersectConvexConvex(hull, box)
    return float(inter) / float(area)


def _hulls_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Do two projected boxes overlap enough to read as one object?"""
    cv2 = _cv2()
    ha, hb = _hull(a), _hull(b)
    inter, _ = cv2.intersectConvexConvex(ha, hb)
    if inter <= 0:
        return False
    smaller = min(cv2.contourArea(ha), cv2.contourArea(hb))
    return smaller > 0 and (inter / smaller) > 0.15


def _hull(uv: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    return cv2.convexHull(np.asarray(uv, np.float32).reshape(-1, 1, 2))


def crop_rect(boxes_uv: Sequence[np.ndarray], image_size: Tuple[int, int],
              zoom: float = ZOOM, aspect: float = 4.0 / 3.0,
              headroom: float = 0.0) -> Tuple[int, int, int, int]:
    """A window on the candidates: fixed aspect, inside the image, with headroom.

    `headroom` asks for that many source pixels of extra space above the
    candidates, so the caption band has somewhere to sit that is not on top of an
    answer. It is granted only out of the slack the window already has -- pushing
    the content down far enough to fall out of the bottom would be a worse bug
    than a caption overlapping a box.
    """
    W, H = image_size
    pts = np.concatenate([np.asarray(u, float).reshape(-1, 2)
                          for u in boxes_uv], axis=0)
    x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    content_w, content_h = x1 - x0, y1 - y0
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)

    cw = max(content_w * zoom, content_h * zoom * aspect, 240.0)
    cw = min(cw, float(W), float(H) * aspect)
    ch = cw / aspect
    # if the candidates do not fit at this zoom, the zoom loses
    if content_w > cw or content_h > ch:
        cw = min(float(W), float(H) * aspect,
                 max(content_w * 1.2, content_h * 1.2 * aspect))
        ch = cw / aspect

    slack = max(0.0, 0.5 * (ch - content_h) - 0.05 * ch)
    top = cy - 0.5 * ch - min(headroom, slack)
    left = cx - 0.5 * cw
    left = float(np.clip(left, 0.0, max(0.0, W - cw)))
    top = float(np.clip(top, 0.0, max(0.0, H - ch)))
    return int(round(left)), int(round(top)), int(round(cw)), int(round(ch))


def _fit_scale(text: str, max_w: int, want: float, thickness: int,
               floor: float = 0.55) -> float:
    """Largest scale at or below `want` that keeps `text` on one line."""
    cv2 = _cv2()
    scale = want
    while scale > floor:
        w = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale,
                            thickness)[0][0]
        if w <= max_w:
            return scale
        scale -= 0.02
    return floor


def _chip(img, text: str, xy, colour, scale: float, thickness: int,
          bg=(16, 18, 22), pad: int = 7, anchor: str = "tl"):
    """Text on an opaque plate, so a label never sits illegibly on clutter."""
    cv2 = _cv2()
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale,
                                     thickness)
    bw, bh = tw + 2 * pad, th + base + 2 * pad
    x, y = float(xy[0]), float(xy[1])
    if anchor == "bc":          # bottom-centre of the plate at xy
        x, y = x - bw / 2.0, y - bh
    h, w = img.shape[:2]
    x = int(np.clip(x, 2, max(2, w - bw - 2)))
    y = int(np.clip(y, 2, max(2, h - bh - 2)))
    cv2.rectangle(img, (x, y), (x + bw, y + bh), bg, -1)
    cv2.rectangle(img, (x, y), (x + bw, y + bh), colour, 1)
    cv2.putText(img, text, (x + pad, y + pad + th), cv2.FONT_HERSHEY_SIMPLEX,
                scale, colour, thickness, cv2.LINE_AA)
    return x, y, bw, bh


def _caption(img, query: str, sub: str, want_title: float) -> int:
    """Big sentence, one quiet line under it. Returns the band height."""
    cv2 = _cv2()
    w = img.shape[1]
    pad = int(round(0.020 * w))
    title_scale = _fit_scale(query, w - 2 * pad, want_title, 3, floor=0.7)
    sub_scale = 0.62 * (w / 1100.0)
    (_, th), _ = cv2.getTextSize(query, cv2.FONT_HERSHEY_SIMPLEX, title_scale, 3)
    (_, sh), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, sub_scale, 1)
    band = pad + th + int(0.75 * pad) + sh + pad

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, band), (11, 13, 17), -1)
    cv2.addWeighted(overlay, 0.92, img, 0.08, 0, img)
    cv2.putText(img, query, (pad, pad + th), cv2.FONT_HERSHEY_SIMPLEX,
                title_scale, (248, 248, 249), 3, cv2.LINE_AA)
    cv2.putText(img, sub, (pad + 2, band - pad), cv2.FONT_HERSHEY_SIMPLEX,
                sub_scale, (156, 164, 176), 1, cv2.LINE_AA)
    return band


def _panel(img, rows: Sequence[Tuple[str, str]], selected: int,
           pressed: Optional[int], scale: float) -> Tuple[int, int, int]:
    """The frame table, bottom-left, big enough to read at feed scale.

    Bottom-left because the caption band now owns the top of the image; a large
    title and a top-right panel together leave the picture nowhere to breathe.
    """
    cv2 = _cv2()
    h, w = img.shape[:2]
    fs = 0.66 * scale
    lines = [f"{n}: {g}" for n, g in rows]
    tw = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0][0]
             for t in lines)
    pad = int(round(15 * scale))
    row_h = int(round(42 * scale))
    pw = min(int(w - 2 * round(0.022 * w)), tw + 2 * pad + int(round(18 * scale)))
    ph = row_h * len(rows) + pad
    x0 = int(round(0.022 * w))
    y0 = h - ph - int(round(0.032 * h))

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + pw, y0 + ph), (13, 15, 19), -1)
    cv2.addWeighted(overlay, 0.90, img, 0.10, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (56, 62, 72), 1)

    for i, text in enumerate(lines):
        ry = y0 + int(round(pad * 0.5)) + row_h * i
        sel = i == selected
        if sel:
            cv2.rectangle(img, (x0 + 4, ry + 2), (x0 + pw - 4, ry + row_h - 5),
                          (30, 42, 56), -1)
            cv2.rectangle(img, (x0 + 4, ry + 2),
                          (x0 + 4 + int(round(7 * scale)), ry + row_h - 5),
                          COLOURS["target"], -1)
        elif pressed == i:
            cv2.rectangle(img, (x0 + 4, ry + 2), (x0 + pw - 4, ry + row_h - 5),
                          (42, 46, 54), -1)
        col = COLOURS["target"] if sel else (166, 172, 182)
        cv2.putText(img, text, (x0 + pad + int(round(8 * scale)),
                                ry + int(round(row_h * 0.66))),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, col, 2 if sel else 1,
                    cv2.LINE_AA)
    return x0, y0, row_h


def poster_state(scene: Scene, src: FrameSource, frame_index: int, anchor,
                 answers: Sequence[Tuple[str, int]], selected: int,
                 query_text: str, gloss: Dict[str, str],
                 pressed: Optional[int] = None, width: int = 1100,
                 zoom: float = ZOOM, aspect: float = 4.0 / 3.0,
                 title_scale: float = 1.55, include_anchor: bool = True,
                 label_anchor: bool = True) -> np.ndarray:
    """One still, composed for a feed rather than for debugging."""
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
    other_ids = [i for _, i in answers if i != sel_id]
    uv_by_id: Dict[int, np.ndarray] = {}
    for _, oid in answers:
        o = scene.by_id(oid)
        uv, _, front, _ = project_box(o.obb, K, pose, src.image_size)
        if front:
            uv_by_id[oid] = uv
    if not uv_by_id:
        raise RuntimeError("no candidate projects in front of the camera")

    # The crop sets the final scale, and the final scale sets how thick a stroke
    # has to be drawn *now* to land at the intended weight after the resize.
    boxes = list(uv_by_id.values())
    if include_anchor:
        # only when the anchor is small enough that including it whole does not
        # blow the crop open; `best_poster_view` separately guarantees the anchor
        # is at least partly in shot
        auv, _, afront, _ = project_box(anchor.obb, K, pose, src.image_size)
        if afront:
            cand = np.concatenate(boxes, axis=0)
            span = max(np.ptp(cand[:, 0]), np.ptp(cand[:, 1]))
            if max(np.ptp(auv[:, 0]), np.ptp(auv[:, 1])) < 1.4 * span:
                boxes = boxes + [auv]
    _, _, cw0, _ = crop_rect(boxes, src.image_size, zoom=zoom, aspect=aspect)
    s = width / float(cw0)
    x0, y0, cw, ch = crop_rect(boxes, src.image_size, zoom=zoom, aspect=aspect,
                               headroom=HEADROOM * (width / aspect) / s)

    def src_px(final_px: float) -> int:
        return max(1, int(round(final_px / s)))

    sel = scene.by_id(sel_id)
    if sel_id in uv_by_id:
        overlay = img.copy()
        cv2.fillConvexPoly(overlay, np.int32(_hull(uv_by_id[sel_id])),
                           COLOURS["target"])
        cv2.addWeighted(overlay, GLOW_ALPHA, img, 1.0 - GLOW_ALPHA, 0, img)
    for oid in other_ids:
        o = scene.by_id(oid)
        draw_box_hidden_line(img, o.obb, K, pose, dbuf,
                             _dim(COLOURS["runner"], 0.9), src_px(3.5), None)
    draw_box_hidden_line(img, sel.obb, K, pose, dbuf, COLOURS["target"],
                         src_px(8.0), None)

    img = np.ascontiguousarray(img[y0:y0 + ch, x0:x0 + cw])
    img = cv2.resize(img, (width, int(round(width / aspect))),
                     interpolation=cv2.INTER_AREA)
    fs = width / 1100.0

    def to_final(uv):
        a = np.asarray(uv, float)
        return np.stack([(a[..., 0] - x0) * s, (a[..., 1] - y0) * s], axis=-1)

    band = _caption(img, f'"{query_text}"',
                    "same scene, same camera - only the reference frame changes",
                    title_scale * fs)
    rows = [(k, gloss.get(k, "")) for k, _ in answers]
    px0, py0, row_h = _panel(img, rows, selected, pressed, fs)

    lab = 0.62 * fs
    for oid in other_ids:
        if oid not in uv_by_id:
            continue
        o = scene.by_id(oid)
        p = to_final(uv_by_id[oid])
        _chip(img, f"{o.label} #{oid}",
              (p[:, 0].mean(), max(band + 6 + 40 * fs, p[:, 1].min() - 6)),
              COLOURS["runner"], lab, 2, anchor="bc")
    if label_anchor:
        auv, _, afront, _ = project_box(anchor.obb, K, pose, src.image_size)
        if afront:
            c = to_final(auv).mean(axis=0)
            if band + 10 < c[1] < py0 - 10:
                _chip(img, anchor.label, (c[0], c[1]), (168, 174, 184), lab, 1,
                      anchor="bc")

    # The heavy marker. No connector line back to the panel: the selected row
    # and the answer are both amber, which ties them without drawing a 900 px
    # diagonal across the photograph, and at feed scale that line read as an
    # artefact rather than as a pointer.
    if sel_id in uv_by_id:
        p = to_final(uv_by_id[sel_id])
        ctr = p.mean(axis=0)
        rad = 0.5 * float(max(np.ptp(p[:, 0]), np.ptp(p[:, 1])))
        # hug the box. An earlier version used rad * 1.5 with a 0.17 * width
        # cap, which on a near object drew a ring around a quarter of the image
        # and swallowed the other candidate.
        r = int(np.clip(rad * 1.06 + 0.008 * width, 0.030 * width,
                        0.105 * width))
        overlay = img.copy()
        cv2.circle(overlay, (int(ctr[0]), int(ctr[1])), r, COLOURS["target"],
                   max(3, int(0.005 * width)), cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.9, img, 0.1, 0, img)
        _chip(img, f"{sel.label} #{sel_id}",
              (ctr[0], max(band + 8 + 44 * fs, ctr[1] - r - 4)),
              COLOURS["target"], lab * 1.2, 2, anchor="bc")
    return img


def frame_switch_poster(scene: Scene, src: FrameSource, query_text: str,
                        out_path: str, cfg: Optional[RelationConfig] = None,
                        kinds: Sequence[str] = ("egocentric", "intrinsic"),
                        width: int = 1100, hold_ms: int = 1700,
                        click_ms: int = 280, colours: int = 160,
                        frame_index: Optional[int] = None,
                        zoom: float = ZOOM, aspect: float = 4.0 / 3.0,
                        title_scale: float = 1.55,
                        include_anchor: bool = True,
                        verbose: bool = True) -> Optional[str]:
    """Write the shareable GIF. None if the query is not frame-split."""
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
    idx = int(frame_index) if frame_index is not None else best_poster_view(
        src, scene, anchor, [i for _, i in answers], aspect=aspect, zoom=zoom,
        verbose=verbose)
    if idx < 0:
        # fall back to the diagnostic chooser rather than refusing outright, and
        # say so, because the composition will be worse
        idx = best_joint_view(src, [o.obb for o in objs], scene.up,
                              scene_background=scene.background)
        if idx < 0:
            if verbose:
                print("  no frame sees the anchor and both answers together")
            return None
        if verbose:
            print("  no frame composes well; falling back to best_joint_view")

    relation = res.query.primary_relation
    od = res.query.target.ordinal
    if od is not None and od.from_word:
        gloss = {k: v.format(d=od.from_word, a=anchor.canonical_label)
                 for k, v in ORDINAL_GLOSS.items()}
    elif relation in ("front", "behind"):
        gloss = dict(FRONT_GLOSS)
    else:
        gloss = dict(FRAME_GLOSS)

    stills: List[np.ndarray] = []
    durations: List[int] = []
    n = len(answers)
    for i in range(n):
        for pressed, ms in ((None, hold_ms), ((i + 1) % n, click_ms)):
            stills.append(poster_state(
                scene, src, idx, anchor, answers, i, query_text, gloss,
                pressed=pressed, width=width, zoom=zoom, aspect=aspect,
                title_scale=title_scale, include_anchor=include_anchor))
            durations.append(ms)

    pil = [Image.fromarray(s[:, :, ::-1]).convert(
        "P", palette=Image.ADAPTIVE, colors=colours) for s in stills]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                exist_ok=True)
    pil[0].save(out_path, save_all=True, append_images=pil[1:],
                duration=durations, loop=0, optimize=True, disposal=2)
    if verbose:
        kb = os.path.getsize(out_path) / 1024.0
        print(f"  frame {idx}, answers {answers}, anchor #{anchor.id}")
        print(f"  wrote {out_path} ({kb:.0f} KB, {len(pil)} frames, "
              f"{width}px wide)")
    return out_path

"""HTTP server behind the blind annotation UI.

    sqe annotate-ui --root DATA --items proposals.jsonl --out labelled.jsonl \
                    --relation-type projective_lateral --goal 57

Three decisions in here matter more than the rest of the file.

**The plan view is the primary instrument, not a photograph.** The first version
of this tool tried to show one capture frame containing the anchor and every
candidate. It does not exist. On "the sink to the left of the door", scene
`0a184cf634` has two doors and two sinks spread over six metres, and *no* frame
of the capture sees even one of each: `best_joint_view` scores frames on the
worst-seen box, so when the set cannot be co-visible every frame scores equally
badly and the winner is arbitrary. Measured on the first three lateral items,
that approach put 1 of 5, 1 of 4 and 0 of 4 candidates on screen. So the plan
view carries the judgement -- which is the right instrument anyway, since
`render_topdown`'s own docstring notes that a lateral relation is a plan-view
property -- and photographs are demoted to one thumbnail *per candidate*,
answering the only question a picture is needed for: "which cabinet is #3?"

**The viewpoint is pinned on save.** A scene's cached trajectory is built with
`trajectory_stride=12`, so its frame *i* is not video frame *i*; the render path
has always resolved `best_view` against a full-rate `FrameSource` instead (see
`render_query_overlay`). That leaves a gap: the viewpoint the annotator judges
from would be re-resolved at evaluation time against the strided trajectory, and
a left/right judgement can turn on the difference between two nearby poses. So
the exact eye used for the plan view is written into the item as an explicit
`position` viewpoint. The annotator's reading and the evaluator's frame are then
the same viewpoint by construction rather than by luck.

**What is deliberately not drawn.** No estimated object fronts, no room
canonical forward, no labelled left/right axis, and nothing the resolver would
answer. Those are the pipeline's own outputs and estimates; on screen, a wrong
front estimate quietly becomes the annotator's intrinsic reading, which is a
system output contaminating its own ground truth. The camera eye and its facing
*are* drawn -- those are dataset ground truth, and the item specifies that
viewpoint. `serve()` never constructs a `Resolver`.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np

from ..bench.schema import BenchItem, read_jsonl, write_jsonl
from ..scenegraph.objects import CameraTrajectory, Scene

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

#: Badge colours, BGR. No amber: `overlay.COLOURS["target"]` means "this is the
#: answer" everywhere else in the tool and must never mark an unlabelled
#: candidate.
PALETTE = [
    (120, 220, 120),   # green
    (240, 150, 100),   # blue
    (200, 130, 230),   # violet
    (110, 220, 240),   # cyan
    (140, 170, 250),   # salmon
    (190, 200, 120),   # teal
    (230, 180, 250),   # pink
    (160, 230, 190),   # mint
]


def _horiz(v: np.ndarray, up: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    up = np.asarray(up, float)
    return v - (v @ up) * up


def _relevant_ids(scene: Scene, text: str) -> List[int]:
    """Objects whose class the sentence names. Same rule as the terminal tool."""
    words = text.lower().replace(",", " ").split()
    low = text.lower()
    out = []
    for o in scene.objects:
        lab = o.canonical_label
        if not lab:
            continue
        if lab.split()[-1] in words or lab in low:
            out.append(o.id)
    return out


def _anchor_and_targets(scene: Scene, text: str) -> Tuple[List[int], List[int]]:
    """Split the class-matched objects into (anchor ids, target ids).

    Uses the rule parser only to learn which noun is the anchor. The sentence is
    on screen anyway, so this leaks nothing the annotator cannot already read.
    """
    from ..categories import label_matches
    from ..query.parser_rules import parse

    ids = _relevant_ids(scene, text)
    try:
        q = parse(text)
        tgt_cls = q.target.label
        anc_cls = next((a.label for c in q.target.constraints
                        for a in c.anchors if a.label), None)
    except Exception:
        return [], ids
    if not anc_cls or anc_cls == tgt_cls:
        return [], ids

    anchors, targets = [], []
    for oid in ids:
        o = scene.by_id(oid)
        lab = o.canonical_label or o.label
        if label_matches(anc_cls, lab) > (label_matches(tgt_cls, lab)
                                          if tgt_cls else 0.0):
            anchors.append(oid)
        else:
            targets.append(oid)
    if not targets or not anchors:
        return [], ids
    return anchors, targets


class ItemView:
    """Everything needed to draw and save one item, resolved once and cached."""

    def __init__(self, item: BenchItem, scene: Scene, src,
                 eye: np.ndarray, look_dir: np.ndarray,
                 frame_index: int, candidates: List[dict]):
        self.item = item
        self.scene = scene
        self.src = src
        self.eye = eye
        self.look_dir = look_dir
        self.frame_index = frame_index
        self.candidates = candidates
        self.plan: Optional[bytes] = None
        self.thumbs: Dict[int, bytes] = {}


def _encode(img: np.ndarray, ext: str = ".jpg") -> bytes:
    from ..viz.overlay import _cv2
    cv2 = _cv2()
    params = ([int(cv2.IMWRITE_JPEG_QUALITY), 86] if ext == ".jpg" else [])
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError("failed to encode image")
    return buf.tobytes()


def build_view(item: BenchItem, scene: Scene, src) -> ItemView:
    """Resolve the viewpoint, then give every candidate its own best frame."""
    from ..frames.policy import anchor_center_obb

    anchor_ids, target_ids = _anchor_and_targets(scene, item.text)
    ids = list(dict.fromkeys(anchor_ids + target_ids)) or \
        [o.id for o in scene.movable_objects()][:12]

    traj = CameraTrajectory(src.poses, src.Ks, src.image_size, src.names)

    # The viewpoint is "as filmed, looking at the anchor" -- the same thing the
    # item's `best_view` mode means, resolved against the full-rate trajectory
    # so it matches the frames actually shown. With no anchor, fall back to the
    # centroid of the candidates.
    if anchor_ids:
        focus = np.mean([scene.by_id(i).center for i in anchor_ids], axis=0)
    else:
        focus = np.mean([scene.by_id(i).center for i in ids], axis=0)
    vi = traj.best_view(anchor_center_obb(focus), scene.up)
    if vi < 0:
        vi = traj.nearest_index(focus)
    vi = int(max(0, vi))
    pose = traj.poses[vi]
    eye = pose[:3, 3].copy()
    look_dir = pose[:3, 2].copy()

    cands = []
    for oid in ids:
        o = scene.by_id(oid)
        if o is None:
            continue
        fi = traj.best_view(o.obb, scene.up)
        if fi < 0:
            fi = traj.nearest_index(o.center)
        cands.append({
            "id": int(o.id), "label": o.label,
            "role": "anchor" if oid in anchor_ids else "target",
            "centre": [round(float(x), 2) for x in o.center],
            "frame": int(max(0, fi)),
        })

    # Badge order is left-to-right along the viewer's lateral axis, so the
    # numbering in the plan, the list and the thumbnails all agree, and matches
    # the order the sentence talks about.
    lat = np.cross(_horiz(look_dir, scene.up), scene.up)
    if np.linalg.norm(lat) < 1e-9:
        lat = np.array([1.0, 0.0, 0.0])
    lat /= np.linalg.norm(lat)
    cands.sort(key=lambda c: float(np.asarray(c["centre"]) @ lat))
    for n, c in enumerate(cands):
        c["badge"] = n + 1
        c["colour"] = PALETTE[n % len(PALETTE)]

    return ItemView(item, scene, src, eye, look_dir, vi, cands)


def draw_plan(v: ItemView, size: int = 1000) -> bytes:
    """Plan view: badged candidates, the camera, and nothing else.

    Drawn in the *camera's* horizontal frame, not world axes -- the camera's
    right is the image's right and it looks up the page. In world axes the
    camera happens to face down the image on most of these captures, so badge 1
    (the leftmost object from the viewer) landed on the right of the picture and
    the annotator had to rotate the map in their head before every judgement.
    That is a pure error source with nothing to recommend it.

    A world-axis compass is drawn in the corner so the room-canonical reading is
    still available. Object fronts are still not drawn: see the module
    docstring.
    """
    from ..viz.overlay import _cv2
    cv2 = _cv2()
    scene = v.scene

    look = _horiz(v.look_dir, scene.up)
    if np.linalg.norm(look) < 1e-9:
        look = np.array([0.0, 1.0, 0.0])
    look /= np.linalg.norm(look)
    right = np.cross(look, scene.up)
    right /= max(float(np.linalg.norm(right)), 1e-9)
    eye = np.asarray(v.eye, float)

    def cam_xy(p) -> np.ndarray:
        """World point -> (rightward, forward) metres from the eye.

        Accepts a 2- or 3-vector; a plan position only needs the horizontal
        part, and the vertical component cancels because `right` and `look` are
        both horizontal.
        """
        q = np.asarray(p, float).reshape(-1)
        w = np.zeros(3)
        w[:min(3, q.size)] = q[:3]
        d = w - eye
        return np.array([float(d @ right), float(d @ look)])

    pts = scene.background
    if pts is None or not len(pts):
        pts = np.concatenate([o.cloud() for o in scene.objects], axis=0)
    d = pts[:, :3] - eye
    puv = np.stack([d @ right, d @ look], axis=1)
    allxy = np.vstack([puv, np.zeros((1, 2))])
    lo, hi = allxy.min(axis=0), allxy.max(axis=0)
    span = np.maximum(hi - lo, 1e-3)
    margin = 48
    s = (size - 2 * margin) / float(span.max())
    W, H = int(span[0] * s) + 2 * margin, int(span[1] * s) + 2 * margin
    img = np.full((H, W, 3), 18, np.uint8)

    def px(world_pt):
        p = (cam_xy(world_pt) - lo) * s
        return int(margin + p[0]), int(H - margin - p[1])

    step = max(1, len(pts) // 60000)
    for q in puv[::step]:
        x = int(margin + (q[0] - lo[0]) * s)
        y = int(H - margin - (q[1] - lo[1]) * s)
        if 0 <= x < W and 0 <= y < H:
            img[y, x] = (72, 72, 72)

    def footprint(o):
        return cv2.convexHull(np.array([px(c) for c in o.obb.corners()],
                                       np.int32).reshape(-1, 1, 2))

    by_id = {c["id"]: c for c in v.candidates}
    for o in scene.objects:
        if o.id not in by_id:
            cv2.polylines(img, [footprint(o)], True, (70, 70, 70), 1,
                          cv2.LINE_AA)

    for c in v.candidates:
        o = scene.by_id(c["id"])
        col = tuple(int(x) for x in c["colour"])
        hull = footprint(o)
        ov = img.copy()
        cv2.fillPoly(ov, [hull], col)
        cv2.addWeighted(ov, 0.30, img, 0.70, 0, img)
        cv2.polylines(img, [hull], True, col,
                      3 if c["role"] == "anchor" else 2, cv2.LINE_AA)
        cx, cy = px(o.center)
        if c["role"] == "anchor":
            cv2.circle(img, (cx, cy), 18, col, 2, cv2.LINE_AA)
        b = str(c["badge"])
        (tw, th), _ = cv2.getTextSize(b, cv2.FONT_HERSHEY_SIMPLEX, 0.74, 2)
        cv2.rectangle(img, (cx - tw // 2 - 7, cy - th // 2 - 7),
                      (cx + tw // 2 + 7, cy + th // 2 + 7), col, -1)
        cv2.putText(img, b, (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.74, (18, 18, 18), 2,
                    cv2.LINE_AA)

    # the camera, at the bottom of the picture looking up it
    pale = (250, 240, 200)
    tip = eye + 0.60 * look
    wedge = np.array([px(eye + 0.04 * look), px(tip + 0.32 * right),
                      px(tip - 0.32 * right)], np.int32)
    cv2.fillPoly(img, [wedge.reshape(-1, 1, 2)], pale)
    cv2.circle(img, px(eye), 6, pale, -1, cv2.LINE_AA)

    # "left" and "right" are now literally left and right in this picture,
    # which is the whole point of drawing it in the camera's frame
    ex, ey = px(eye)
    cv2.putText(img, "<- viewer's left", (16, H - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, pale, 1, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize("viewer's right ->", cv2.FONT_HERSHEY_SIMPLEX,
                                 0.5, 1)
    cv2.putText(img, "viewer's right ->", (W - tw - 16, H - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, pale, 1, cv2.LINE_AA)

    # world-axis compass, so the room-canonical reading stays available
    r = 26
    cx, cy = 62, H - 100
    dull = (140, 140, 140)
    cv2.circle(img, (cx, cy), r + 8, (34, 34, 34), -1)
    for axis, name in ((np.array([1.0, 0.0, 0.0]), "x"),
                       (np.array([0.0, 1.0, 0.0]), "y")):
        u = float(axis @ right)
        w = float(axis @ look)
        n = max((u * u + w * w) ** 0.5, 1e-9)
        ax, ay = cx + int(r * u / n), cy - int(r * w / n)
        cv2.arrowedLine(img, (cx, cy), (ax, ay), dull, 1, cv2.LINE_AA,
                        tipLength=0.3)
        cv2.putText(img, name, (ax - 4, ay - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, dull, 1, cv2.LINE_AA)
    cv2.putText(img, "world", (cx - 17, cy + r + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, dull, 1, cv2.LINE_AA)

    return _encode(img, ".png")


def draw_thumb(v: ItemView, cand_id: int, width: int = 520) -> bytes:
    """One candidate, on the capture frame that sees it best, with its box."""
    from ..viz.overlay import _cv2, project_box
    cv2 = _cv2()
    c = next((x for x in v.candidates if x["id"] == cand_id), None)
    if c is None:
        raise KeyError(cand_id)
    fi = c["frame"]
    rgb = v.src.rgb(fi)
    if rgb is None:
        img = np.full((width * 3 // 4, width, 3), 30, np.uint8)
        cv2.putText(img, "no RGB frame", (20, width * 3 // 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (190, 190, 190), 2)
        return _encode(img)

    o = v.scene.by_id(cand_id)
    K = v.src.Ks[fi] if v.src.Ks.ndim == 3 else v.src.Ks
    pose = v.src.poses[fi]
    uv, _z, all_front, on_screen = project_box(o.obb, K, pose, v.src.image_size)

    scale = width / float(rgb.shape[1])
    img = cv2.resize(rgb, (width, int(rgb.shape[0] * scale)))
    col = tuple(int(x) for x in c["colour"])
    if all_front and on_screen and len(uv):
        lo, hi = uv.min(axis=0) * scale, uv.max(axis=0) * scale
        cv2.rectangle(img, (int(lo[0]), int(lo[1])), (int(hi[0]), int(hi[1])),
                      col, 3, cv2.LINE_AA)
    b = f"{c['badge']}  {c['label']}"
    (tw, th), _ = cv2.getTextSize(b, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (0, 0), (tw + 18, th + 16), col, -1)
    cv2.putText(img, b, (9, th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (18, 18, 18), 2, cv2.LINE_AA)
    return _encode(img)


#: Keys the browser is allowed to receive. Anything the resolver produces --
#: a predicted target, frame scores, the policy's chosen frame, an ambiguity
#: flag -- must never appear, because a label confirming a system's own guess
#: measures agreement rather than correctness and the leak is invisible in the
#: rendered pictures. `tests/test_annotate_ui.py` asserts this on the real
#: payload, which is why it is built here rather than inline in the handler.
ITEM_KEYS = ("pos", "n", "id", "scene_id", "text", "relation", "relation_type",
             "eye", "candidates", "annotated", "saved", "done")

CANDIDATE_KEYS = ("id", "label", "role", "badge", "centre", "colour")


def item_payload(queue: "Queue", pos: int, v: ItemView) -> dict:
    """What `/api/item` sends. See `ITEM_KEYS` for what is deliberately absent."""
    it = v.item
    return {
        "pos": pos, "n": len(queue.indices), "id": it.id,
        "scene_id": it.scene_id, "text": it.text,
        "relation": it.relation, "relation_type": it.relation_type,
        "eye": [round(float(x), 2) for x in v.eye],
        "candidates": [{k: c[k] for k in CANDIDATE_KEYS}
                       for c in v.candidates],
        "annotated": bool(it.target_ids or it.ambiguous),
        "saved": {"target_ids": list(it.target_ids), "frame": it.frame,
                  "frame_stated": bool(it.frame_stated_in_text),
                  "ambiguous": bool(it.ambiguous),
                  "ambiguity_kind": it.ambiguity_kind},
        "done": queue.done_count(),
    }


class Queue:
    """The ordered work list, and the file it is written to."""

    def __init__(self, items: List[BenchItem], indices: List[int],
                 out_path: str, annotator: str, goal: Optional[int]):
        self.items = items
        self.indices = indices
        self.out_path = out_path
        self.annotator = annotator
        self.goal = goal
        self.lock = threading.Lock()

    def done_count(self) -> int:
        return sum(1 for x in self.items if x.target_ids or x.ambiguous)

    def save(self) -> None:
        with self.lock:
            write_jsonl(self.items, self.out_path)


def make_handler(queue: Queue, scene_for: Callable, src_for: Callable):
    views: Dict[int, ItemView] = {}
    vlock = threading.Lock()

    def view(pos: int) -> ItemView:
        with vlock:
            if pos in views:
                return views[pos]
        it = queue.items[queue.indices[pos]]
        v = build_view(it, scene_for(it.scene_id), src_for(it.scene_id))
        with vlock:
            views[pos] = v
        return v

    def prefetch(pos: int) -> None:
        if not (0 <= pos < len(queue.indices)):
            return
        try:
            v = view(pos)
            if v.plan is None:
                v.plan = draw_plan(v)
            for c in v.candidates:
                if c["id"] not in v.thumbs:
                    v.thumbs[c["id"]] = draw_thumb(v, c["id"])
        except Exception:
            pass

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(json.dumps(obj).encode(), "application/json", code)

        def _pos(self, q) -> int:
            try:
                return int(q.get("pos", ["0"])[0])
            except ValueError:
                return 0

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)

            if u.path in ("/", "/index.html"):
                with open(os.path.join(STATIC, "index.html"), "rb") as f:
                    return self._send(f.read(), "text/html; charset=utf-8")

            if u.path == "/api/queue":
                return self._json({"n": len(queue.indices), "goal": queue.goal,
                                   "done": queue.done_count(),
                                   "out": queue.out_path,
                                   "annotator": queue.annotator})

            if u.path == "/api/item":
                pos = self._pos(q)
                if not (0 <= pos < len(queue.indices)):
                    return self._json({"error": "out of range"}, 404)
                it = queue.items[queue.indices[pos]]
                try:
                    v = view(pos)
                except Exception as exc:
                    return self._json(
                        {"error": f"{type(exc).__name__}: {exc}"}, 500)
                threading.Thread(target=prefetch, args=(pos + 1,),
                                 daemon=True).start()
                return self._json(item_payload(queue, pos, v))

            if u.path == "/api/plan":
                pos = self._pos(q)
                try:
                    v = view(pos)
                    if v.plan is None:
                        v.plan = draw_plan(v)
                    return self._send(v.plan, "image/png")
                except Exception as exc:
                    return self._json({"error": str(exc)}, 500)

            if u.path == "/api/thumb":
                pos = self._pos(q)
                try:
                    cid = int(q.get("cand", ["-1"])[0])
                    v = view(pos)
                    if cid not in v.thumbs:
                        v.thumbs[cid] = draw_thumb(v, cid)
                    return self._send(v.thumbs[cid], "image/jpeg")
                except Exception as exc:
                    return self._json({"error": str(exc)}, 500)

            return self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"error": "bad json"}, 400)
            if u.path != "/api/save":
                return self._send(b"not found", "text/plain", 404)

            pos = int(body.get("pos", -1))
            if not (0 <= pos < len(queue.indices)):
                return self._json({"error": "out of range"}, 400)
            it = queue.items[queue.indices[pos]]
            try:
                v = view(pos)
            except Exception as exc:
                return self._json({"error": str(exc)}, 500)

            scene = scene_for(it.scene_id)
            ids = [int(x) for x in (body.get("target_ids") or [])]
            bad = [i for i in ids if scene.by_id(i) is None]
            if bad:
                return self._json({"error": f"not in this scene: {bad}"}, 400)

            it.target_ids = ids
            it.frame = body.get("frame", "unspecified")
            it.frame_stated_in_text = bool(body.get("frame_stated", False))
            it.ambiguous = bool(body.get("ambiguous", False))
            it.ambiguity_kind = body.get("ambiguity_kind", "none")
            it.annotator = queue.annotator
            it.source = ("generated_reviewed_ui"
                         if it.source.startswith("generated")
                         else "annotated_ui")

            # Pin the viewpoint the annotator actually judged from. See the
            # module docstring: without this the evaluator re-resolves
            # `best_view` against the strided trajectory and can land on a
            # different pose than the plan view showed.
            it.viewpoint_mode = "position"
            it.viewpoint_position = [float(x) for x in v.eye]
            it.viewpoint_index = None
            it.viewpoint_landmark = None

            from ..query.parser_rules import parse
            it.gold_parse = parse(it.text).to_dict()

            problems = it.validate(scene)
            queue.save()
            return self._json({"saved": it.id, "problems": problems,
                               "done": queue.done_count()})

    return Handler


def serve(items_path: str, out_path: str, scene_for: Callable,
          src_for: Callable,
          relation_types: Optional[Sequence[str]] = None,
          order: str = "informative", goal: Optional[int] = None,
          annotator: str = "", cfg=None, host: str = "127.0.0.1",
          port: int = 8766, open_browser: bool = True) -> None:
    from ..bench.annotate import order_queue

    items = read_jsonl(items_path)
    if out_path != items_path and os.path.exists(out_path):
        done = {i.id: i for i in read_jsonl(out_path)}
        for n, it in enumerate(items):
            if it.id in done:
                items[n] = done[it.id]

    keep = [n for n, it in enumerate(items)
            if not relation_types or it.relation_type in relation_types]
    if not keep:
        raise SystemExit(f"no items match relation types {relation_types}")

    # `order_queue` runs the resolver once, to find which items the frames
    # currently disagree on and put those first. That is a property of the queue
    # order only: no per-item prediction is kept, returned or shown.
    indices = order_queue(items, keep, order, scene_for=scene_for, cfg=cfg)
    if goal:
        labelled = [i for i in indices
                    if items[i].target_ids or items[i].ambiguous]
        pending = [i for i in indices if i not in set(labelled)]
        indices = labelled + pending[:max(0, goal - len(labelled))]

    q = Queue(items, indices, out_path, annotator, goal)
    httpd = ThreadingHTTPServer((host, port), make_handler(q, scene_for,
                                                           src_for))
    url = f"http://{host}:{port}/"
    print(f"\n  blind annotation UI on {url}")
    print(f"  {len(indices)} queued, {q.done_count()} already annotated")
    print(f"  writing to {out_path}")
    print("  keys: 1-9 pick - e i a w n frame - t stated - x ambiguous"
          " - enter save+next - s skip - b back\n")
    if open_browser:
        import webbrowser
        threading.Thread(target=lambda: webbrowser.open(url),
                         daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n  stopped. {q.done_count()} annotated in {out_path}")
    finally:
        httpd.server_close()

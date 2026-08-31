"""Local web viewer: point cloud, object boxes, live query, relation graph.

Deliberately built on `http.server` from the standard library. The viewer is a
demo and a debugging tool, and making the repo's core depend on a web framework
to look at a point cloud is a bad trade -- this way `sqe viewer` works in any
environment that can run the rest of the package.

Geometry goes over the wire as raw little-endian float32/uint8 buffers rather
than JSON, because a 400 000-point cloud as JSON numbers is about 12 MB of text
and half a second of parsing.

Endpoints
---------
``GET  /``                     the page
``GET  /api/scenes``           scene list
``GET  /api/scene/<id>``       objects, room, frames metadata (JSON)
``GET  /api/cloud/<id>``       point cloud, binary: xyz float32 + rgb uint8
``POST /api/query``            resolve a query, returns the full resolution
``POST /api/annotate``         append an annotated item to the benchmark jsonl
"""

from __future__ import annotations

import json
import os
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional
from urllib.parse import unquote, urlparse

import numpy as np

from ..frames.policy import ViewpointSpec, build_frames
from ..geom.pointcloud import subsample
from ..query.parser_rules import parse
from ..query.resolver import Resolver
from ..relations.base import RelationConfig
from ..scenegraph.objects import Scene

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_VIEWER_POINTS = 400_000


def scene_payload(scene: Scene) -> dict:
    """Everything the front end needs except the point cloud itself."""
    objs = []
    for o in scene.objects:
        objs.append({
            "id": o.id,
            "label": o.label,
            "canonical_label": o.canonical_label,
            "center": o.center.tolist(),
            "extent": o.extent.tolist(),
            "yaw": o.obb.yaw,
            "R": o.obb.R.tolist(),
            "color": list(o.color),
            "front": None if o.front is None else o.front.tolist(),
            "front_confidence": o.front_confidence,
            "front_method": o.front_method,
            "has_intrinsic_front": o.has_intrinsic_front,
            "is_support_surface": o.is_support_surface,
            "is_room_fixed": o.is_room_fixed,
            "levels": list(o.levels),
            "point_count": o.point_count,
            "volume": o.volume,
            "height": o.height,
        })
    room = None
    if scene.room is not None:
        r = scene.room
        room = {
            "up": r.up.tolist(), "floor_z": r.floor_z, "ceiling_z": r.ceiling_z,
            "axes": r.axes.tolist(), "axis_confidence": r.axis_confidence,
            "bounds": r.bounds.to_dict(),
            "canonical_forward": (None if r.canonical_forward is None
                                  else r.canonical_forward.tolist()),
            "forward_convention": r.forward_convention,
            "forward_margin": r.forward_confidence,
            "forward_candidates": [c.tolist() for c in r.forward_candidates],
            "walls": [{"direction": w.direction.tolist(), "area": w.area}
                      for w in r.walls],
        }
    traj = None
    if scene.trajectory is not None and len(scene.trajectory):
        t = scene.trajectory
        step = max(1, len(t) // 400)
        traj = {"centers": t.centers[::step].tolist(),
                "forwards": t.forwards[::step].tolist(),
                "n": len(t), "step": step}
    return {"scene_id": scene.scene_id, "dataset": scene.dataset,
            "objects": objs, "room": room, "trajectory": traj,
            "meta": {k: v for k, v in scene.meta.items()
                     if k not in ("build_stats",)},
            "labels": scene.labels()}


def cloud_buffer(scene: Scene, max_points: int = MAX_VIEWER_POINTS) -> bytes:
    """xyz float32 then rgb uint8, with an 8-byte header giving the count."""
    pts = scene.background
    cols = scene.background_color
    if pts is None or not len(pts):
        pts = np.concatenate([o.cloud() for o in scene.objects], axis=0) \
            if scene.objects else np.zeros((0, 3))
        cols = None
    idx = subsample(pts, max_points, 0)
    p = np.asarray(pts, np.float32)[idx]
    if cols is not None and len(cols) == len(pts):
        c = np.asarray(cols, np.uint8)[idx]
    else:
        c = np.full((len(p), 3), 170, np.uint8)
    head = np.asarray([len(p)], "<u4").tobytes() + b"\0\0\0\0"
    return head + p.astype("<f4").tobytes() + c.tobytes()


def _viewpoint_from(body: dict) -> ViewpointSpec:
    mode = body.get("viewpoint_mode", "best_view")
    pos = body.get("viewpoint_position")
    return ViewpointSpec(
        mode=mode,
        index=body.get("viewpoint_index"),
        position=None if not pos else np.asarray(pos, float),
        look_at=(None if not body.get("look_at")
                 else np.asarray(body["look_at"], float)),
        landmark=body.get("viewpoint_landmark"))


def make_handler(scenes: Dict[str, Scene], cfg: RelationConfig,
                 items_path: Optional[str]):
    #: In annotation mode the resolution endpoints are switched off. Blindness
    #: was previously a matter of convention: `--items` turned the annotation
    #: panel on but left /api/query serving the answer, the runner-up and the
    #: ambiguity flags, so an annotator could resolve first and label second
    #: without meaning to. The README claimed the tool was blind by default,
    #: which was true of the terminal tool and overstated for the viewer.
    annotating = bool(items_path)
    resolvers: Dict[str, Resolver] = {}
    lock = threading.Lock()

    def resolver(sid: str) -> Resolver:
        with lock:
            if sid not in resolvers:
                resolvers[sid] = Resolver(scenes[sid], cfg)
            return resolvers[sid]

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):      # quieter than the default
            if os.environ.get("SQE_VIEWER_LOG"):
                super().log_message(fmt, *a)

        # -- plumbing --------------------------------------------------
        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj, default=float).encode(),
                       "application/json")

        def _read_body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            try:
                return json.loads(self.rfile.read(n).decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                return {}

        # -- routes ----------------------------------------------------
        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path in ("/", "/index.html"):
                return self._file("index.html", "text/html; charset=utf-8")
            if path == "/app.js":
                return self._file("app.js", "application/javascript")
            if path == "/style.css":
                return self._file("style.css", "text/css")
            if path == "/api/scenes":
                return self._json({"scenes": sorted(scenes),
                                   "annotating": annotating,
                                   "resolution_disabled": annotating})
            m = re.fullmatch(r"/api/scene/(.+)", path)
            if m:
                sid = m.group(1)
                if sid not in scenes:
                    return self._json({"error": f"no scene {sid!r}"}, 404)
                return self._json(scene_payload(scenes[sid]))
            m = re.fullmatch(r"/api/cloud/(.+)", path)
            if m:
                sid = m.group(1)
                if sid not in scenes:
                    return self._json({"error": f"no scene {sid!r}"}, 404)
                return self._send(200, cloud_buffer(scenes[sid]),
                                  "application/octet-stream")
            return self._json({"error": "not found"}, 404)

        def do_POST(self):
            path = unquote(urlparse(self.path).path)
            body = self._read_body()
            if path == "/api/query":
                if annotating:
                    return self._json(
                        {"error": "resolution is disabled while annotating, so "
                                  "the label cannot be influenced by the "
                                  "system's answer. Restart the viewer without "
                                  "--items to explore resolutions."}, 403)
                return self._query(body)
            if path == "/api/frames":
                if annotating:
                    # the frame axes alone are fine, but this endpoint is how the
                    # front end learns which frames are available and confident,
                    # which is a hint about the answer
                    return self._json(
                        {"error": "disabled while annotating"}, 403)
                return self._frames(body)
            if path == "/api/annotate":
                return self._annotate(body)
            return self._json({"error": "not found"}, 404)

        def _file(self, name: str, ctype: str):
            p = os.path.join(STATIC, name)
            if not os.path.exists(p):
                return self._json({"error": f"missing static file {name}"}, 404)
            with open(p, "rb") as f:
                return self._send(200, f.read(), ctype)

        def _query(self, body: dict):
            sid = body.get("scene_id")
            text = (body.get("text") or "").strip()
            if sid not in scenes:
                return self._json({"error": f"no scene {sid!r}"}, 400)
            if not text:
                return self._json({"error": "empty query"}, 400)
            q = parse(text)
            try:
                res = resolver(sid).resolve(q, _viewpoint_from(body),
                                            force_frame=body.get("force_frame")
                                            or None)
            except Exception as exc:
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
            out = res.to_dict()
            out["explanation"] = res.explain()
            return self._json(out)

        def _frames(self, body: dict):
            sid = body.get("scene_id")
            oid = body.get("object_id")
            if sid not in scenes:
                return self._json({"error": f"no scene {sid!r}"}, 400)
            scene = scenes[sid]
            obj = scene.by_id(int(oid)) if oid is not None else None
            if obj is None:
                return self._json({"error": f"no object {oid}"}, 400)
            frames, rv = build_frames(scene, obj,
                                      viewpoint=_viewpoint_from(body))
            return self._json({"object_id": obj.id, "label": obj.label,
                               "viewpoint": rv.to_dict(),
                               "frames": {k: v.to_dict()
                                          for k, v in frames.items()}})

        def _annotate(self, body: dict):
            if not items_path:
                return self._json({"error": "the viewer was started without "
                                            "--items, so it cannot annotate"}, 400)
            from ..bench.schema import BenchItem, read_jsonl, write_jsonl
            sid = body.get("scene_id")
            if sid not in scenes:
                return self._json({"error": f"no scene {sid!r}"}, 400)
            items = read_jsonl(items_path) if os.path.exists(items_path) else []
            by_id = {i.id: i for i in items}
            iid = body.get("id") or f"{sid}_v{len(items):04d}"
            text = (body.get("text") or "").strip()
            if not text:
                return self._json({"error": "empty text"}, 400)
            q = parse(text)
            it = by_id.get(iid) or BenchItem(id=iid, scene_id=sid,
                                             dataset=scenes[sid].dataset,
                                             text=text)
            it.text = text
            it.target_ids = [int(x) for x in (body.get("target_ids") or [])]
            it.frame = body.get("frame", "unspecified")
            it.frame_stated_in_text = bool(body.get("frame_stated", False))
            it.ambiguous = bool(body.get("ambiguous", False))
            it.ambiguity_kind = body.get("ambiguity_kind", "none")
            it.difficulty = body.get("difficulty", "medium")
            it.relation = q.primary_relation
            it.relation_type = q.relation_type()
            it.gold_parse = q.to_dict()
            it.annotator = body.get("annotator", "viewer")
            it.source = "viewer"
            vp = _viewpoint_from(body)
            it.viewpoint_mode = vp.mode
            it.viewpoint_index = vp.index
            it.viewpoint_position = (None if vp.position is None
                                     else list(map(float, vp.position)))
            it.viewpoint_landmark = vp.landmark
            problems = it.validate(scenes[sid])
            if iid not in by_id:
                items.append(it)
            write_jsonl(items, items_path)
            return self._json({"saved": it.id, "n_items": len(items),
                               "problems": problems})

    return Handler


def serve(scenes: Dict[str, Scene], host: str = "127.0.0.1", port: int = 8765,
          cfg: Optional[RelationConfig] = None,
          items_path: Optional[str] = None, open_browser: bool = True):
    cfg = cfg or RelationConfig.load()
    handler = make_handler(scenes, cfg, items_path)
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"viewer on {url}")
    print(f"  scenes: {', '.join(sorted(scenes))}")
    if items_path:
        print(f"  annotating into {items_path}")
    print("  ctrl-c to stop")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()

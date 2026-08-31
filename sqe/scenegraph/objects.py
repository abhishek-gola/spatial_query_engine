"""The scene data model: `Object3D`, `CameraTrajectory`, `Scene`, and the
on-disk cache.

Perception is expensive and the interesting part of this repo is not. So a scene
is built once into this structure, written to disk as `scene.json` +
`arrays.npz`, and every query after that is pure geometry on numpy arrays --
milliseconds, no GPU, no model loading. The viewer and the benchmark both read
the cache, never the raw dataset.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..categories import (has_intrinsic_front, is_container, is_room_fixed,
                          is_support_surface, normalize_label, prior)
from ..geom.obb import OBB, fit_obb
from ..geom.room import RoomStructure
from ..geom.transforms import (camera_forward, camera_center, horizontal,
                               normalize, se3_inverse)

SCHEMA_VERSION = 3


# --------------------------------------------------------------------------
# Objects
# --------------------------------------------------------------------------

@dataclass
class Object3D:
    id: int
    label: str
    obb: OBB
    points: Optional[np.ndarray] = None          # (N,3) world, voxel-reduced
    point_count: int = 0
    color: Tuple[float, float, float] = (0.6, 0.6, 0.6)
    label_scores: List[Tuple[str, float]] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None       # open-vocab feature, L2-normed

    # orientation, filled in by sqe.perception.orientation
    front: Optional[np.ndarray] = None
    front_confidence: float = 0.0
    front_method: str = "none"
    front_alternatives: List[Tuple[List[float], float]] = field(default_factory=list)

    # internal horizontal surfaces, e.g. the individual boards of a bookshelf
    levels: List[float] = field(default_factory=list)
    level_confidence: float = 0.0

    source: str = "unknown"                      # 'gt' | 'openvocab' | 'synthetic'
    meta: Dict = field(default_factory=dict)

    # -- convenience -------------------------------------------------------
    @property
    def canonical_label(self) -> str:
        return normalize_label(self.label)

    @property
    def center(self) -> np.ndarray:
        return self.obb.center

    @property
    def extent(self) -> np.ndarray:
        return self.obb.extent

    @property
    def height(self) -> float:
        return self.obb.height

    @property
    def volume(self) -> float:
        return self.obb.volume

    @property
    def top(self) -> float:
        return self.obb.top

    @property
    def bottom(self) -> float:
        return self.obb.bottom

    @property
    def max_dim(self) -> float:
        return float(np.max(self.extent))

    @property
    def diameter(self) -> float:
        return float(np.linalg.norm(self.extent))

    @property
    def has_intrinsic_front(self) -> bool:
        """Whether this *category* is the kind of thing that has a front.

        Separate from `front is not None`, which says whether we managed to
        estimate one. The distinction matters: a chair whose front we failed to
        estimate is a low-confidence case; a mug has no front to estimate and
        the intrinsic frame is simply not applicable.
        """
        return has_intrinsic_front(self.label)

    @property
    def is_support_surface(self) -> bool:
        return is_support_surface(self.label)

    @property
    def is_container(self) -> bool:
        return is_container(self.label)

    @property
    def is_room_fixed(self) -> bool:
        return is_room_fixed(self.label)

    def cloud(self) -> np.ndarray:
        """Points if we have them, box corners as a stand-in if we do not."""
        if self.points is not None and len(self.points):
            return self.points
        return self.obb.corners()

    def short(self) -> str:
        c = self.center
        return (f"#{self.id} {self.label} "
                f"({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) "
                f"{self.extent[0]:.2f}x{self.extent[1]:.2f}x{self.extent[2]:.2f}")

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "canonical_label": self.canonical_label,
            "obb": self.obb.to_dict(),
            "point_count": self.point_count,
            "color": list(self.color),
            "label_scores": [[l, float(s)] for l, s in self.label_scores],
            "front": None if self.front is None else np.asarray(self.front).tolist(),
            "front_confidence": float(self.front_confidence),
            "front_method": self.front_method,
            "front_alternatives": [[list(map(float, v)), float(s)]
                                   for v, s in self.front_alternatives],
            "levels": [float(z) for z in self.levels],
            "level_confidence": float(self.level_confidence),
            "source": self.source,
            "meta": _jsonable(self.meta),
        }

    @staticmethod
    def from_dict(d: dict) -> "Object3D":
        return Object3D(
            id=int(d["id"]),
            label=d["label"],
            obb=OBB.from_dict(d["obb"]),
            point_count=int(d.get("point_count", 0)),
            color=tuple(d.get("color", (0.6, 0.6, 0.6))),
            label_scores=[(l, float(s)) for l, s in d.get("label_scores", [])],
            front=None if d.get("front") is None else np.asarray(d["front"], float),
            front_confidence=float(d.get("front_confidence", 0.0)),
            front_method=d.get("front_method", "none"),
            front_alternatives=[(list(map(float, v)), float(s))
                                for v, s in d.get("front_alternatives", [])],
            levels=[float(z) for z in d.get("levels", [])],
            level_confidence=float(d.get("level_confidence", 0.0)),
            source=d.get("source", "unknown"),
            meta=d.get("meta", {}),
        )


# --------------------------------------------------------------------------
# Camera trajectory
# --------------------------------------------------------------------------

@dataclass
class CameraTrajectory:
    """Camera-to-world poses of the capture, plus intrinsics."""
    poses: np.ndarray                       # (T,4,4) camera-to-world
    K: np.ndarray                           # (3,3) or (T,3,3)
    image_size: Tuple[int, int] = (0, 0)    # (width, height)
    frame_ids: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return 0 if self.poses is None else len(self.poses)

    def K_at(self, i: int) -> np.ndarray:
        return self.K if self.K.ndim == 2 else self.K[i]

    @property
    def centers(self) -> np.ndarray:
        return self.poses[:, :3, 3]

    @property
    def forwards(self) -> np.ndarray:
        return self.poses[:, :3, 2]

    def mean_heading(self, up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
        """Average horizontal look direction over the capture.

        A plain vector mean, so a capture that panned a full circle correctly
        returns something near zero -- callers check for that instead of being
        handed a meaningless unit vector.
        """
        if len(self) == 0:
            return np.zeros(3)
        f = self.forwards - (self.forwards @ up)[:, None] * up
        m = f.mean(axis=0)
        return m if np.linalg.norm(m) < 1e-6 else normalize(m)

    def heading_concentration(self, up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> float:
        """0 when the capture looked everywhere, 1 when it stared one way."""
        if len(self) == 0:
            return 0.0
        f = self.forwards - (self.forwards @ up)[:, None] * up
        n = np.linalg.norm(f, axis=1, keepdims=True)
        f = f / np.maximum(n, 1e-9)
        return float(np.linalg.norm(f.mean(axis=0)))

    def nearest_index(self, point: np.ndarray) -> int:
        d = np.linalg.norm(self.centers - np.asarray(point, float), axis=1)
        return int(np.argmin(d))

    def best_view(self, obb: OBB, up: np.ndarray = np.array([0.0, 0.0, 1.0]),
                  target_angular_fraction: float = 0.18) -> int:
        """Index of the pose that shows `obb` best.

        Not simply the pose that fills the most of the frame. An earlier version
        scored angular size clipped at 1.0, which meant a frame with the camera
        pressed against -- or inside -- an object won, and the "best view" of an
        office chair came out as a close-up of the cables under the desk it was
        tucked beneath.

        That mattered well beyond rendering: `best_view` is the default
        egocentric viewpoint for queries, and a viewpoint inside the anchor makes
        its left and right meaningless.

        So: reject poses inside the box or nearer than its own radius, then
        prefer a distance at which the object subtends roughly
        `target_angular_fraction` of the view, with a log-normal falloff either
        side, times how squarely the camera faces it.
        """
        if len(self) == 0:
            return -1
        c = obb.center
        radius = max(0.05, float(np.linalg.norm(obb.half)))
        d = c - self.centers
        dist = np.linalg.norm(d, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            dirs = d / np.maximum(dist[:, None], 1e-9)
        on_axis = np.einsum("ij,ij->i", dirs, self.forwards)

        # too close: the camera is inside or touching the object
        inside = obb.contains(self.centers, pad=0.05)
        too_close = dist < 1.25 * radius

        desired = max(radius / max(target_angular_fraction, 1e-3), 0.5)
        ratio = np.maximum(dist, 1e-6) / desired
        fit = np.exp(-0.5 * (np.log(ratio) / 0.55) ** 2)

        score = np.where((on_axis > 0.25) & ~inside & ~too_close,
                         on_axis * fit, -1.0)
        if float(score.max()) <= 0.0:
            # nothing sees it properly; fall back to the nearest pose that is at
            # least outside the object
            fallback = np.where(inside | too_close, np.inf, dist)
            if np.all(np.isinf(fallback)):
                return -1
            return int(np.argmin(fallback))
        return int(np.argmax(score))

    def viewer_pose(self, index: int) -> np.ndarray:
        return self.poses[index]

    def to_arrays(self) -> dict:
        return {"traj_poses": self.poses.astype(np.float32),
                "traj_K": self.K.astype(np.float32)}


# --------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------

@dataclass
class Scene:
    scene_id: str
    objects: List[Object3D] = field(default_factory=list)
    room: Optional[RoomStructure] = None
    trajectory: Optional[CameraTrajectory] = None
    background: Optional[np.ndarray] = None       # (M,3) full scene cloud, reduced
    background_color: Optional[np.ndarray] = None  # (M,3) uint8
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    dataset: str = "unknown"
    meta: Dict = field(default_factory=dict)

    _by_id: Dict[int, Object3D] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.reindex()

    def reindex(self):
        self._by_id = {o.id: o for o in self.objects}

    def __len__(self) -> int:
        return len(self.objects)

    def by_id(self, oid: int) -> Optional[Object3D]:
        return self._by_id.get(int(oid))

    def labels(self) -> List[str]:
        return sorted({o.canonical_label for o in self.objects})

    def find(self, label: str, min_score: float = 0.6) -> List[Object3D]:
        """Objects whose label matches, best first. Closed-vocabulary helper."""
        from ..categories import label_matches
        scored = [(label_matches(label, o.label), o) for o in self.objects]
        hits = [(s, o) for s, o in scored if s >= min_score]
        hits.sort(key=lambda t: (-t[0], t[1].id))
        return [o for _, o in hits]

    def movable_objects(self) -> List[Object3D]:
        return [o for o in self.objects if not o.is_room_fixed]

    def scale(self) -> float:
        """Characteristic room size in metres, used to make thresholds relative."""
        if self.room is not None:
            return max(1.0, self.room.diagonal)
        if self.background is not None and len(self.background):
            e = self.background.max(axis=0) - self.background.min(axis=0)
            return max(1.0, float(np.linalg.norm(e[:2])))
        return 5.0

    def summary(self) -> str:
        lines = [f"scene {self.scene_id} [{self.dataset}] {len(self.objects)} objects"]
        if self.room is not None:
            r = self.room
            lines.append(f"  room {r.bounds.extent[0]:.2f} x {r.bounds.extent[1]:.2f}"
                         f" x {(r.ceiling_z or r.floor_z) - r.floor_z:.2f} m,"
                         f" manhattan conf {r.axis_confidence:.2f},"
                         f" forward '{r.forward_convention}' margin"
                         f" {r.forward_confidence:.2f}")
        if self.trajectory is not None and len(self.trajectory):
            lines.append(f"  trajectory {len(self.trajectory)} poses,"
                         f" heading concentration"
                         f" {self.trajectory.heading_concentration(self.up):.2f}")
        counts: Dict[str, int] = {}
        for o in self.objects:
            counts[o.canonical_label] = counts.get(o.canonical_label, 0) + 1
        top = sorted(counts.items(), key=lambda t: -t[1])[:14]
        lines.append("  labels: " + ", ".join(f"{k} x{v}" for k, v in top))
        nf = sum(1 for o in self.objects if o.front is not None)
        nfc = sum(1 for o in self.objects if o.has_intrinsic_front)
        lines.append(f"  fronts estimated {nf}/{nfc} of front-bearing categories")
        return "\n".join(lines)

    # -- cache -------------------------------------------------------------
    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        arrays: Dict[str, np.ndarray] = {}

        pts, offs = [], [0]
        embs, emb_ids = [], []
        for o in self.objects:
            p = o.points if o.points is not None else np.zeros((0, 3))
            pts.append(np.asarray(p, dtype=np.float32))
            offs.append(offs[-1] + len(p))
            if o.embedding is not None:
                embs.append(np.asarray(o.embedding, dtype=np.float32))
                emb_ids.append(o.id)
        arrays["obj_points"] = (np.concatenate(pts, axis=0) if pts
                                else np.zeros((0, 3), np.float32))
        arrays["obj_offsets"] = np.asarray(offs, dtype=np.int64)
        if embs:
            arrays["embeddings"] = np.stack(embs)
            arrays["embedding_ids"] = np.asarray(emb_ids, dtype=np.int64)
        if self.background is not None:
            arrays["background"] = np.asarray(self.background, dtype=np.float32)
        if self.background_color is not None:
            arrays["background_color"] = np.asarray(self.background_color, np.uint8)
        if self.trajectory is not None and len(self.trajectory):
            arrays.update(self.trajectory.to_arrays())

        np.savez_compressed(os.path.join(directory, "arrays.npz"), **arrays)
        meta = {
            "schema_version": SCHEMA_VERSION,
            "scene_id": self.scene_id,
            "dataset": self.dataset,
            "up": self.up.tolist(),
            "objects": [o.to_dict() for o in self.objects],
            "room": None if self.room is None else self.room.to_dict(),
            "trajectory": None if self.trajectory is None or not len(self.trajectory)
            else {"image_size": list(self.trajectory.image_size),
                  "frame_ids": list(self.trajectory.frame_ids)},
            "meta": _jsonable(self.meta),
        }
        with open(os.path.join(directory, "scene.json"), "w") as f:
            json.dump(meta, f, indent=1)

    @staticmethod
    def load(directory: str) -> "Scene":
        with open(os.path.join(directory, "scene.json")) as f:
            meta = json.load(f)
        if int(meta.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError(
                f"cache at {directory} was written by schema v"
                f"{meta.get('schema_version')} but this build expects v"
                f"{SCHEMA_VERSION}; rebuild it with `sqe build`")
        z = np.load(os.path.join(directory, "arrays.npz"))

        objs = [Object3D.from_dict(d) for d in meta["objects"]]
        offs = z["obj_offsets"]
        allpts = z["obj_points"]
        for i, o in enumerate(objs):
            a, b = int(offs[i]), int(offs[i + 1])
            o.points = allpts[a:b].astype(np.float64) if b > a else None
        if "embeddings" in z:
            emb, ids = z["embeddings"], z["embedding_ids"]
            lookup = {int(k): emb[j] for j, k in enumerate(ids)}
            for o in objs:
                if o.id in lookup:
                    o.embedding = lookup[o.id]

        traj = None
        if "traj_poses" in z:
            tmeta = meta.get("trajectory") or {}
            traj = CameraTrajectory(
                poses=z["traj_poses"].astype(np.float64),
                K=z["traj_K"].astype(np.float64),
                image_size=tuple(tmeta.get("image_size", (0, 0))),
                frame_ids=list(tmeta.get("frame_ids", [])),
            )

        return Scene(
            scene_id=meta["scene_id"],
            objects=objs,
            room=None if meta.get("room") is None
            else RoomStructure.from_dict(meta["room"]),
            trajectory=traj,
            background=z["background"].astype(np.float64) if "background" in z else None,
            background_color=z["background_color"] if "background_color" in z else None,
            up=np.asarray(meta.get("up", [0, 0, 1]), float),
            dataset=meta.get("dataset", "unknown"),
            meta=meta.get("meta", {}),
        )


def _jsonable(obj):
    """Recursively convert numpy types so `json.dump` stops complaining."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

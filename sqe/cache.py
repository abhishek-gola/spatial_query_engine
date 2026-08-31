"""Scene cache: build once, query forever.

Perception and room analysis take seconds; a query takes milliseconds. The cache
is the seam between them, and keeping it explicit is what makes the viewer and
the benchmark cheap to run repeatedly.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import List, Optional

from .scenegraph.objects import SCHEMA_VERSION, Scene

DEFAULT_CACHE = os.environ.get(
    "SQE_CACHE", os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "cache"))


def scene_key(dataset: str, scene_id: str, tag: str = "") -> str:
    base = f"{dataset}__{scene_id}"
    return f"{base}__{tag}" if tag else base


def scene_dir(cache_root: str, dataset: str, scene_id: str, tag: str = "") -> str:
    return os.path.join(cache_root, scene_key(dataset, scene_id, tag))


def exists(cache_root: str, dataset: str, scene_id: str, tag: str = "") -> bool:
    d = scene_dir(cache_root, dataset, scene_id, tag)
    return (os.path.exists(os.path.join(d, "scene.json"))
            and os.path.exists(os.path.join(d, "arrays.npz")))


def save(scene: Scene, cache_root: str, tag: str = "") -> str:
    d = scene_dir(cache_root, scene.dataset, scene.scene_id, tag)
    scene.save(d)
    return d


def load(cache_root: str, dataset: str, scene_id: str, tag: str = "") -> Scene:
    return Scene.load(scene_dir(cache_root, dataset, scene_id, tag))


def list_cached(cache_root: str) -> List[dict]:
    out = []
    if not os.path.isdir(cache_root):
        return out
    for name in sorted(os.listdir(cache_root)):
        p = os.path.join(cache_root, name, "scene.json")
        if not os.path.exists(p):
            continue
        try:
            with open(p) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"key": name, "scene_id": meta.get("scene_id"),
                    "dataset": meta.get("dataset"),
                    "n_objects": len(meta.get("objects", [])),
                    "schema_version": meta.get("schema_version"),
                    "stale": meta.get("schema_version") != SCHEMA_VERSION})
    return out

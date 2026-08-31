"""CLIP features for 3-D proposals: view selection, cropping, encoding.

The open-vocabulary label for a 3-D proposal comes from looking at it in the
frames that actually see it. Three things decide whether this works:

1. **View selection.** A proposal is scored per frame by how much of it is
   in-bounds *and* unoccluded, judged against the frame's own depth map. Cropping
   from a frame where the object is behind a wall produces a confident label for
   the wall.
2. **Cropping.** A tight crop of a keyboard is a grey rectangle. The crop is
   padded to give CLIP some context, and is squared up, because CLIP's
   preprocessing squashes non-square crops.
3. **Aggregation.** Per-view embeddings are averaged with weights from the
   visibility score, then re-normalised, so a marginal view cannot outvote a
   clear one.

Runs on MPS, CUDA or CPU. On an M-series Mac the cost is dominated by decoding
video frames, not by CLIP, which is why frames are decoded once in a single
forward pass over the file and shared across every proposal that needs them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..geom.transforms import project, se3_inverse, transform_points

DEFAULT_MODEL = "openai/clip-vit-base-patch16"

#: Prompt ensemble. A single "a photo of a {}" is noticeably worse on cropped
#: indoor objects, which are often partial and badly lit.
PROMPTS = (
    "a photo of a {}",
    "a photo of a {} in a room",
    "a cropped photo of a {}",
    "a close-up photo of a {}",
    "a blurry photo of a {}",
    "an indoor photo of a {}",
)


def pick_device(prefer: Optional[str] = None) -> str:
    import torch
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------
# view selection
# --------------------------------------------------------------------------

@dataclass
class ViewScore:
    frame_index: int
    score: float                    # visible fraction, occlusion-aware
    bbox: Tuple[int, int, int, int]  # x0, y0, x1, y1 in the RGB frame
    n_visible: int


def score_views(points: np.ndarray, poses: np.ndarray, Ks: np.ndarray,
                image_size: Tuple[int, int],
                depth_reader=None, depth_size: Tuple[int, int] = (256, 192),
                frame_stride: int = 1,
                max_points: int = 400,
                occlusion_tol: float = 0.15,
                min_visible: int = 25,
                seed: int = 0) -> List[ViewScore]:
    """Score how well each frame sees a point set.

    The visible fraction is computed against the frame's depth map, so a
    proposal hidden behind furniture scores low even though it projects into the
    image. Points are subsampled -- 400 is plenty to estimate a fraction and
    keeps the whole thing linear in the number of frames.
    """
    from ..geom.pointcloud import subsample
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return []
    pts = pts[subsample(pts, max_points, seed)]
    w, h = image_size
    dw, dh = depth_size
    out: List[ViewScore] = []

    for i in range(0, len(poses), max(1, frame_stride)):
        pose = poses[i]
        K = Ks[i] if np.ndim(Ks) == 3 else Ks
        cam = transform_points(se3_inverse(pose), pts)
        uv, z, valid = project(cam, K)
        inb = (valid & (uv[:, 0] >= 0) & (uv[:, 0] < w)
               & (uv[:, 1] >= 0) & (uv[:, 1] < h) & (z > 0.1))
        n_in = int(inb.sum())
        if n_in < min_visible:
            continue

        vis = inb.copy()
        if depth_reader is not None:
            d = depth_reader(i)
            if d is not None:
                sx, sy = dw / float(w), dh / float(h)
                du = np.clip((uv[:, 0] * sx).astype(np.int64), 0, dw - 1)
                dv = np.clip((uv[:, 1] * sy).astype(np.int64), 0, dh - 1)
                meas = d[dv, du]
                ok = meas > 0.05
                # a point much further than the measured surface is occluded
                vis = inb & (~ok | (z <= meas + occlusion_tol))
        n_vis = int(vis.sum())
        if n_vis < min_visible:
            continue
        sel = uv[vis]
        x0, y0 = sel.min(axis=0)
        x1, y1 = sel.max(axis=0)
        out.append(ViewScore(frame_index=int(i),
                             score=float(n_vis) / float(len(pts)),
                             bbox=(int(x0), int(y0), int(x1), int(y1)),
                             n_visible=n_vis))
    out.sort(key=lambda v: -v.score)
    return out


def crop_square(img: np.ndarray, bbox: Tuple[int, int, int, int],
                pad_frac: float = 0.25, min_side: int = 48) -> Optional[np.ndarray]:
    """Padded, squared crop. Returns None when the box is too small to be useful."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bbox
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    if max(bw, bh) < min_side * 0.5:
        return None
    side = int(max(bw, bh) * (1.0 + 2.0 * pad_frac))
    side = max(side, min_side)
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    sx = int(round(cx - side / 2.0))
    sy = int(round(cy - side / 2.0))
    sx = max(0, min(w - 1, sx))
    sy = max(0, min(h - 1, sy))
    ex = min(w, sx + side)
    ey = min(h, sy + side)
    crop = img[sy:ey, sx:ex]
    if crop.size == 0 or min(crop.shape[:2]) < 8:
        return None
    return crop


# --------------------------------------------------------------------------
# encoder
# --------------------------------------------------------------------------

class ClipEncoder:
    """Lazy CLIP wrapper. Text embeddings are cached and prompt-ensembled."""

    def __init__(self, model_name: str = DEFAULT_MODEL,
                 device: Optional[str] = None, batch_size: int = 64,
                 fp16: bool = True):
        self.model_name = model_name
        self.device = pick_device(device)
        self.batch_size = batch_size
        self.fp16 = fp16 and self.device != "cpu"
        self._model = None
        self._processor = None
        self._text_cache: Dict[str, np.ndarray] = {}

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:      # pragma: no cover
            raise ImportError(
                "the open-vocabulary backend needs torch + transformers "
                '(pip install -e ".[openvocab]")') from exc
        self._model = CLIPModel.from_pretrained(self.model_name)
        self._model.eval().to(self.device)
        if self.fp16:
            try:
                self._model.half()
            except Exception:
                self.fp16 = False
        self._processor = CLIPProcessor.from_pretrained(self.model_name)

    @property
    def dim(self) -> int:
        self._load()
        return int(self._model.config.projection_dim)

    def encode_images(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """(N, D) L2-normalised image embeddings. Input crops are BGR uint8."""
        import torch
        self._load()
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        out = []
        for start in range(0, len(crops), self.batch_size):
            batch = crops[start:start + self.batch_size]
            # ascontiguousarray, not a bare ::-1 slice: the reversed view has a
            # negative stride and torch.from_numpy refuses it
            rgb = [np.ascontiguousarray(c[:, :, ::-1]) for c in batch]
            inputs = self._processor(images=rgb, return_tensors="pt")
            pix = inputs["pixel_values"].to(self.device)
            if self.fp16:
                pix = pix.half()
            with torch.no_grad():
                f = _feature_tensor(
                    self._model.get_image_features(pixel_values=pix))
            f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            out.append(f.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    def encode_texts(self, texts: Sequence[str],
                     ensemble: bool = True) -> np.ndarray:
        """(N, D) L2-normalised text embeddings, prompt-ensembled and cached."""
        import torch
        self._load()
        todo = [t for t in texts if t not in self._text_cache]
        if todo:
            prompts: List[str] = []
            for t in todo:
                prompts.extend(p.format(t) for p in
                               (PROMPTS if ensemble else ("{}",)))
            n_p = len(PROMPTS) if ensemble else 1
            feats = []
            for start in range(0, len(prompts), self.batch_size):
                chunk = prompts[start:start + self.batch_size]
                inputs = self._processor(text=chunk, return_tensors="pt",
                                         padding=True, truncation=True)
                ids = inputs["input_ids"].to(self.device)
                att = inputs["attention_mask"].to(self.device)
                with torch.no_grad():
                    f = _feature_tensor(self._model.get_text_features(
                        input_ids=ids, attention_mask=att))
                f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                feats.append(f.float().cpu().numpy())
            allf = np.concatenate(feats, axis=0).reshape(len(todo), n_p, -1)
            allf = allf.mean(axis=1)
            allf /= np.maximum(np.linalg.norm(allf, axis=1, keepdims=True), 1e-8)
            for t, v in zip(todo, allf):
                self._text_cache[t] = v.astype(np.float32)
        return np.stack([self._text_cache[t] for t in texts])


def _feature_tensor(out):
    """Pull the projected embedding out of a CLIP feature call.

    transformers <5 returns a plain tensor from `get_image_features`; 5.x
    returns a `BaseModelOutputWithPooling` whose `pooler_output` is the
    projected embedding (512-d for ViT-B/16, matching `projection_dim`). Both
    are supported so the backend does not pin a transformers major version.
    """
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    if hasattr(out, "last_hidden_state"):
        raise RuntimeError(
            "CLIP returned only hidden states, not a projected embedding; "
            "this transformers version is not supported")
    return out


def aggregate_view_features(features: np.ndarray,
                            weights: Sequence[float]) -> np.ndarray:
    """Visibility-weighted mean of per-view embeddings, re-normalised."""
    if len(features) == 0:
        return np.zeros(0, np.float32)
    w = np.asarray(list(weights), dtype=np.float64)
    w = np.maximum(w, 1e-6)
    w /= w.sum()
    v = (features * w[:, None]).sum(axis=0)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-8 else v.astype(np.float32)


def classify(embeddings: np.ndarray, vocab: Sequence[str],
             encoder: ClipEncoder, top_k: int = 5,
             temperature: float = 100.0) -> List[List[Tuple[str, float]]]:
    """Rank a vocabulary against each embedding. Returns per-row (label, prob)."""
    if len(embeddings) == 0:
        return []
    text = encoder.encode_texts(list(vocab))
    logits = temperature * (embeddings @ text.T)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    out = []
    for row in probs:
        idx = np.argsort(-row)[:top_k]
        out.append([(vocab[int(i)], float(row[int(i)])) for i in idx])
    return out

"""ONNX Runtime inference: preprocess → pre-NMS ONNX → rotated NMS."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
from PIL import Image

try:
    from .postprocess import meta_to_finalize_kwargs, ort_pre_nms_to_detections
    from .preprocess import (
        canvas_size_from_meta,
        crop_or_pad,
        mean_std_from_meta,
        overlap_pixels_from_meta,
        preprocess_from_meta,
        preprocess_path,
        preprocess_rgb,
        scale_obb_to_original,
        slice_size_from_meta,
        sliding_window_grid,
        to_rgb_uint8,
        uses_sliding_window,
    )
except ImportError:  # copied next to model.onnx as top-level modules
    from postprocess import meta_to_finalize_kwargs, ort_pre_nms_to_detections
    from preprocess import (
        canvas_size_from_meta,
        crop_or_pad,
        mean_std_from_meta,
        overlap_pixels_from_meta,
        preprocess_from_meta,
        preprocess_path,
        preprocess_rgb,
        scale_obb_to_original,
        slice_size_from_meta,
        sliding_window_grid,
        to_rgb_uint8,
        uses_sliding_window,
    )

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_EXPORT_DIR = Path(__file__).resolve().parent
# Source of truth stays under export/; these are copied next to model.onnx.
_RUNTIME_SIDECARS = (
    "nms.py",
    "preprocess.py",
    "postprocess.py",
    "runtime.py",
    "ort_runtime.py",
    "requirements-runtime.txt",
)
_RUNTIME_SCRIPTS = (
    ("scripts/demo.py", "demo.py"),
    ("scripts/infer_onnx.py", "infer_onnx.py"),
)
_RUNTIME_ASSETS = (
    ("demo/planes_pleiades_neo.jpg", "demo/planes_pleiades_neo.jpg"),
)


def copy_runtime_sidecars(onnx_path: Union[str, Path]) -> List[Path]:
    """Copy the consumer detection stack next to the ONNX file.

    Source files stay in ``export/``. Copies are enough to run preprocess →
    ONNX Runtime → NMS without oriented-det or the rest of this repo.
    """
    dest_dir = Path(onnx_path).resolve().parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    to_copy = [(name, name) for name in _RUNTIME_SIDECARS]
    to_copy.extend(_RUNTIME_SCRIPTS)
    to_copy.extend(_RUNTIME_ASSETS)
    for src_rel, dest_rel in to_copy:
        src = _EXPORT_DIR / src_rel
        if not src.is_file():
            raise FileNotFoundError(f"Runtime sidecar missing: {src}")
        dest = dest_dir / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        written.append(dest)
        print(f"Copied {src_rel} → {dest}")
    return written


def load_export_meta(onnx_path: Union[str, Path], meta_path: Union[str, Path, None] = None) -> Dict[str, Any]:
    """Load ``*.export_meta.json`` next to the ONNX (or ``meta_path``)."""
    onnx_file = Path(onnx_path)
    if meta_path is not None:
        path = Path(meta_path)
    else:
        path = onnx_file.with_suffix(".export_meta.json")
        if not path.is_file():
            alt = onnx_file.with_suffix(".json")
            path = alt if alt.is_file() else path
    if not path.is_file():
        raise FileNotFoundError(f"Export meta not found next to {onnx_file} ({path})")
    return json.loads(path.read_text(encoding="utf-8"))


def merge_per_class_score_threshold(
    sidecar: Optional[Mapping[str, float]],
    override: Optional[Mapping[str, float]],
) -> Optional[Dict[str, float]]:
    """Merge an inference-time per-class score override into the sidecar map.

    ``override is None`` returns a copy of ``sidecar`` (or ``None`` when that
    map is missing or empty). A passed map updates one class at a time:
    omitted classes keep their sidecar floors. A key that matches an existing
    name case-insensitively replaces that entry and keeps the sidecar spelling
    (``Drone`` updates ``drone``). Names that are not already present are added.
    Neither input mapping is mutated.
    """
    merged: Dict[str, float] = {}
    if sidecar:
        merged = {str(key): float(value) for key, value in sidecar.items()}
    if override is None:
        return merged or None
    for key, value in override.items():
        name = str(key).strip()
        if not name:
            raise ValueError("per_class_score_threshold key must be a non-empty class name")
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"per_class_score_threshold[{name!r}] must be a float, got {value!r}"
            ) from exc
        target = name if name in merged else None
        if target is None:
            lowered = name.lower()
            for existing in merged:
                if existing.lower() == lowered:
                    target = existing
                    break
        if target is None:
            merged[name] = score
        else:
            merged[target] = score
    return merged or None


def detections_to_records(
    padded: np.ndarray,
    count: int,
    class_names: List[str],
) -> List[Dict[str, Any]]:
    """Convert padded ``[N, 7]`` (cx, cy, w, h, angle, score, label) to dicts."""
    n = max(0, int(count))
    rows = np.asarray(padded[:n], dtype=np.float32)
    names = list(class_names or [])
    out: List[Dict[str, Any]] = []
    for row in rows:
        lid = int(row[6])
        out.append(
            {
                "cx": float(row[0]),
                "cy": float(row[1]),
                "w": float(row[2]),
                "h": float(row[3]),
                "angle_rad": float(row[4]),
                "score": float(row[5]),
                "label": lid,
                "class_name": names[lid - 1] if 1 <= lid <= len(names) else str(lid),
            }
        )
    return out


def detect_array(
    images_nchw: np.ndarray,
    onnx_path: Union[str, Path],
    meta: Mapping[str, Any],
    *,
    orig_w: Optional[int] = None,
    orig_h: Optional[int] = None,
    score_threshold: Optional[float] = None,
    per_class_score_threshold: Optional[Mapping[str, float]] = None,
    nms_backend: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run pre-NMS ONNX + NMS on an already-normalized NCHW batch (batch=1).

    ``score_threshold`` replaces the global production floor only. It does not
    change ``per_class_score_threshold``. That argument, when passed, updates
    the sidecar map key by key (see :func:`merge_per_class_score_threshold`).
    ``None`` leaves the sidecar map in place. Neither override can recover
    scores the graph already dropped at its baked pre-NMS floor.
    """
    fk = meta_to_finalize_kwargs(dict(meta))
    if score_threshold is not None:
        fk["score_threshold"] = float(score_threshold)
    if per_class_score_threshold is not None:
        fk["per_class_score_threshold"] = merge_per_class_score_threshold(
            fk.get("per_class_score_threshold"),
            per_class_score_threshold,
        )
    if nms_backend is not None:
        fk["nms_backend"] = str(nms_backend)
    names = list(meta.get("output_names") or [])
    padded, count = ort_pre_nms_to_detections(images_nchw, str(onnx_path), names, fk)
    canvas_h, canvas_w = canvas_size_from_meta(meta)
    if orig_w is not None and orig_h is not None and count > 0:
        padded = padded.copy()
        padded[:count, :5] = scale_obb_to_original(
            padded[:count, :5], orig_w, orig_h, canvas_w, canvas_h
        )
    return detections_to_records(padded, count, list(meta.get("class_names") or []))


def _probe_hw(image: Union[str, Path, Image.Image, np.ndarray]) -> Optional[tuple]:
    """Return ``(height, width)``, or None when a path is not a readable image."""
    if isinstance(image, (str, Path)):
        try:
            with Image.open(image) as im:
                width, height = im.size
            return int(height), int(width)
        except OSError:
            return None
    if isinstance(image, Image.Image):
        width, height = image.size
        return int(height), int(width)
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] in (3, 4) and arr.shape[0] > 0 and arr.shape[1] > 0:
        return int(arr.shape[0]), int(arr.shape[1])
    return None


def _load_rgb(image: Union[str, Path, Image.Image, np.ndarray], *, bgr: bool) -> np.ndarray:
    if isinstance(image, (str, Path)):
        return np.asarray(Image.open(image).convert("RGB"), dtype=np.uint8)
    return to_rgb_uint8(image, bgr=bgr)


def nms_detection_records(
    records: List[Dict[str, Any]],
    meta: Mapping[str, Any],
    *,
    nms_backend: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Rotated NMS across windows. No extra per-image cap beyond the IoU filter."""
    if len(records) <= 1:
        return list(records)
    try:
        from .nms import rotated_nms
    except ImportError:
        from nms import rotated_nms

    fk = meta_to_finalize_kwargs(dict(meta))
    backend = str(nms_backend) if nms_backend is not None else fk.get("nms_backend")
    boxes = np.array(
        [[d["cx"], d["cy"], d["w"], d["h"], d["angle_rad"]] for d in records],
        dtype=np.float32,
    )
    scores = np.array([d["score"] for d in records], dtype=np.float32)
    labels = np.array([d["label"] for d in records], dtype=np.int64)
    keep = rotated_nms(
        boxes,
        scores,
        labels,
        iou_threshold=float(fk["final_nms_iou_threshold"]),
        max_detections=None,
        class_agnostic=bool(fk["nms_class_agnostic"]),
        backend=backend,
    )
    return [records[int(i)] for i in keep]


def _map_window_detection(
    det: Dict[str, Any],
    *,
    x0: int,
    y0: int,
    crop_w: int,
    crop_h: int,
    scale_x: float,
    scale_y: float,
) -> Optional[Dict[str, Any]]:
    """Canvas box → source pixels, then add the window origin.

    Centroids that land in the zero-pad band (image smaller than the slice on
    that side) are dropped. Overlap copies are kept; a later NMS merges them.
    """
    eps = 1e-3
    cx = float(det["cx"]) / scale_x
    cy = float(det["cy"]) / scale_y
    if cx < -eps or cy < -eps or cx >= float(crop_w) - eps or cy >= float(crop_h) - eps:
        return None
    shifted = dict(det)
    shifted["cx"] = cx + float(x0)
    shifted["cy"] = cy + float(y0)
    shifted["w"] = float(det["w"]) / scale_x
    shifted["h"] = float(det["h"]) / scale_y
    return shifted


def _detect_sliding_window(
    rgb: np.ndarray,
    onnx_path: Union[str, Path],
    meta: Mapping[str, Any],
    *,
    score_threshold: Optional[float] = None,
    per_class_score_threshold: Optional[Mapping[str, float]] = None,
    nms_backend: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Tile ``rgb``, run one ONNX forward per window, NMS the stitched boxes."""
    image_h, image_w = int(rgb.shape[0]), int(rgb.shape[1])
    canvas_h, canvas_w = canvas_size_from_meta(meta)
    slice_h, slice_w = slice_size_from_meta(meta)
    overlap = overlap_pixels_from_meta(meta)
    mean, std = mean_std_from_meta(meta)
    scale_x = float(canvas_w) / float(slice_w)
    scale_y = float(canvas_h) / float(slice_h)
    positions = sliding_window_grid(
        image_h, image_w, slice_h, slice_w, overlap_pixels=overlap
    )
    merged: List[Dict[str, Any]] = []
    for x0, y0 in positions:
        x1 = min(x0 + slice_w, image_w)
        y1 = min(y0 + slice_h, image_h)
        src_x0 = max(0, x0)
        src_y0 = max(0, y0)
        crop_w = x1 - src_x0
        crop_h = y1 - src_y0
        if crop_w <= 0 or crop_h <= 0:
            continue
        window = crop_or_pad(rgb, x0, y0, slice_w, slice_h, image_w, image_h)
        blob, _, _ = preprocess_rgb(window, canvas_h, canvas_w, mean, std)
        dets = detect_array(
            blob,
            onnx_path,
            meta,
            score_threshold=score_threshold,
            per_class_score_threshold=per_class_score_threshold,
            nms_backend=nms_backend,
        )
        for det in dets:
            shifted = _map_window_detection(
                det,
                x0=x0,
                y0=y0,
                crop_w=crop_w,
                crop_h=crop_h,
                scale_x=scale_x,
                scale_y=scale_y,
            )
            if shifted is not None:
                merged.append(shifted)
    return nms_detection_records(merged, meta, nms_backend=nms_backend)


def detect_image(
    image: Union[str, Path, Image.Image, np.ndarray],
    onnx_path: Union[str, Path],
    meta: Optional[Mapping[str, Any]] = None,
    *,
    bgr: bool = False,
    score_threshold: Optional[float] = None,
    per_class_score_threshold: Optional[Mapping[str, float]] = None,
    nms_backend: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Preprocess one image, run ONNX, return detections in original pixel coords.

    When ``resize_mode`` is ``fixed`` or ``crop`` and the image is larger than
    the slice (``window_size``, otherwise ``target_size``), the image is tiled.
    Each window is bilinear-resized onto the ONNX canvas. Overlap copies are
    merged with a second rotated NMS. ``pad`` and ``keep_ratio`` stay one
    full-image resize.

    ``score_threshold`` and ``per_class_score_threshold`` are forwarded to
    :func:`detect_array`.
    """
    onnx_file = Path(onnx_path)
    loaded = dict(meta) if meta is not None else load_export_meta(onnx_file)
    hw = _probe_hw(image)
    if hw is not None and uses_sliding_window(hw[0], hw[1], loaded):
        rgb = _load_rgb(image, bgr=bgr)
        return _detect_sliding_window(
            rgb,
            onnx_file,
            loaded,
            score_threshold=score_threshold,
            per_class_score_threshold=per_class_score_threshold,
            nms_backend=nms_backend,
        )
    if isinstance(image, (str, Path)):
        blob, orig_h, orig_w = preprocess_path(image, loaded)
    else:
        blob, orig_h, orig_w = preprocess_from_meta(image, loaded, bgr=bgr)
    return detect_array(
        blob,
        onnx_file,
        loaded,
        orig_w=orig_w,
        orig_h=orig_h,
        score_threshold=score_threshold,
        per_class_score_threshold=per_class_score_threshold,
        nms_backend=nms_backend,
    )


_DRAW_COLORS = (
    (255, 140, 0),
    (0, 200, 0),
    (40, 80, 255),
    (220, 220, 0),
    (220, 0, 220),
    (0, 200, 220),
)


def draw_detections(
    image: Union[Image.Image, np.ndarray],
    detections: List[Dict[str, Any]],
) -> Image.Image:
    """Draw rotated boxes + labels on a copy of ``image``."""
    from PIL import ImageDraw

    try:
        from .nms import rboxes_to_corners
    except ImportError:
        from nms import rboxes_to_corners

    if isinstance(image, Image.Image):
        vis = image.convert("RGB").copy()
    else:
        vis = Image.fromarray(np.asarray(image), mode="RGB")
    draw = ImageDraw.Draw(vis)
    for det in detections:
        w, h = float(det["w"]), float(det["h"])
        if w <= 0 or h <= 0:
            continue
        corners = rboxes_to_corners(
            np.array([[float(det["cx"]), float(det["cy"]), w, h, float(det["angle_rad"])]])
        )[0]
        pts = [(float(x), float(y)) for x, y in corners]
        lid = int(det["label"])
        color = _DRAW_COLORS[(lid - 1) % len(_DRAW_COLORS)]
        draw.line(pts + [pts[0]], fill=color, width=3)
        caption = f"{det.get('class_name', lid)} {float(det['score']):.2f}"
        tx, ty = pts[0]
        draw.text((max(0, tx), max(0, ty - 12)), caption, fill=color)
    return vis


def list_images(folder: Path) -> List[Path]:
    files = [
        p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    ]
    if not files:
        raise FileNotFoundError(f"No images in {folder}")
    return files

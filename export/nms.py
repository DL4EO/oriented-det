"""CPU rotated IoU and greedy NMS.

Source of truth: ``export/nms.py``. ``odet export onnx`` copies this file
with the rest of the consumer stack next to ``model.onnx``. No oriented-det imports.

IoU backends:
- ``python`` (default): Sutherland–Hodgman convex-quad clip (numpy only)
- ``shapely``: Shapely polygon intersection (``pip install shapely``)
- ``auto``: Shapely if installed, else ``python``
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

_EPS = 1e-8
_SHAPELY = None
_SHAPELY_TRIED = False


def shapely_available() -> bool:
    """True if the Shapely package can be imported."""
    global _SHAPELY, _SHAPELY_TRIED
    if not _SHAPELY_TRIED:
        _SHAPELY_TRIED = True
        try:
            from shapely.geometry import Polygon  # noqa: F401

            _SHAPELY = True
        except Exception:
            _SHAPELY = False
    return bool(_SHAPELY)


def resolve_nms_backend(backend: Optional[str] = None) -> str:
    """Return ``python`` or ``shapely``. Raises if Shapely was requested but missing."""
    raw = "python" if backend is None else str(backend).strip().lower()
    if raw in ("", "python", "numpy"):
        return "python"
    if raw == "auto":
        return "shapely" if shapely_available() else "python"
    if raw == "shapely":
        if not shapely_available():
            raise ImportError(
                "CPU NMS backend 'shapely' requires the shapely package. "
                "Install it with: pip install shapely"
            )
        return "shapely"
    raise ValueError(f"Unknown NMS backend {backend!r}; use python, shapely, or auto.")


def rboxes_to_corners(boxes: np.ndarray) -> np.ndarray:
    """Convert ``[N, 5]`` ``(cx, cy, w, h, angle)`` to ``[N, 4, 2]`` corners."""
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, 5)
    if arr.size == 0:
        return np.zeros((0, 4, 2), dtype=np.float64)
    cx, cy, w, h, ang = arr.T
    cos_a = np.cos(ang)
    sin_a = np.sin(ang)
    w2 = w * 0.5
    h2 = h * 0.5
    local_x = np.stack([-w2, w2, w2, -w2], axis=-1)
    local_y = np.stack([-h2, -h2, h2, h2], axis=-1)
    rx = local_x * cos_a[:, None] - local_y * sin_a[:, None]
    ry = local_x * sin_a[:, None] + local_y * cos_a[:, None]
    return np.stack([cx[:, None] + rx, cy[:, None] + ry], axis=-1)


def _aabbs_from_corners(corners: np.ndarray) -> np.ndarray:
    """Return ``[N, 4]`` ``(x_min, y_min, x_max, y_max)``."""
    return np.concatenate(
        [corners.min(axis=1), corners.max(axis=1)],
        axis=1,
    )


def _aabb_overlaps(a: Sequence[float], b: Sequence[float]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _cross(a: Sequence[float], b: Sequence[float], p: Sequence[float]) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _segment_intersection(
    p1: Sequence[float],
    p2: Sequence[float],
    p3: Sequence[float],
    p4: Sequence[float],
) -> Tuple[float, float]:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return (float(x2), float(y2))
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (float(px), float(py))


def _signed_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    return poly[::-1] if _signed_area(poly) < 0.0 else poly


def _sutherland_hodgman(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    clip_pts = [tuple(p) for p in _ensure_ccw(np.asarray(clipper, dtype=np.float64))]
    output = [tuple(p) for p in np.asarray(subject, dtype=np.float64)]
    if len(output) < 3 or len(clip_pts) < 3:
        return np.zeros((0, 2), dtype=np.float64)
    cp1 = clip_pts[-1]
    for cp2 in clip_pts:
        inp = output
        output = []
        if not inp:
            break
        s = inp[-1]
        for e in inp:
            inside_e = _cross(cp1, cp2, e) >= 0.0
            inside_s = _cross(cp1, cp2, s) >= 0.0
            if inside_e:
                if not inside_s:
                    output.append(_segment_intersection(s, e, cp1, cp2))
                output.append(e)
            elif inside_s:
                output.append(_segment_intersection(s, e, cp1, cp2))
            s = e
        cp1 = cp2
    if len(output) < 3:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(output, dtype=np.float64)


def _polygon_area(pts: np.ndarray) -> float:
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return abs(0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def _intersection_area(corners_a: np.ndarray, corners_b: np.ndarray, backend: str) -> float:
    if backend == "shapely":
        from shapely.geometry import Polygon

        pa = Polygon([(float(x), float(y)) for x, y in corners_a])
        pb = Polygon([(float(x), float(y)) for x, y in corners_b])
        if not pa.is_valid:
            pa = pa.buffer(0)
        if not pb.is_valid:
            pb = pb.buffer(0)
        if pa.is_empty or pb.is_empty:
            return 0.0
        inter = pa.intersection(pb)
        if inter.is_empty:
            return 0.0
        return float(inter.area)
    return _polygon_area(_sutherland_hodgman(corners_a, corners_b))


def rbox_iou(
    box_a: np.ndarray,
    box_b: np.ndarray,
    *,
    backend: Optional[str] = None,
) -> float:
    """IoU of two ``(cx, cy, w, h, angle)`` boxes."""
    impl = resolve_nms_backend(backend)
    a = np.asarray(box_a, dtype=np.float64).reshape(5)
    b = np.asarray(box_b, dtype=np.float64).reshape(5)
    if a[2] <= 0.0 or a[3] <= 0.0 or b[2] <= 0.0 or b[3] <= 0.0:
        return 0.0
    corners = rboxes_to_corners(np.stack([a, b], axis=0))
    aabbs = _aabbs_from_corners(corners)
    if not _aabb_overlaps(aabbs[0], aabbs[1]):
        return 0.0
    inter = _intersection_area(corners[0], corners[1], impl)
    if inter <= 0.0:
        return 0.0
    union = float(a[2] * a[3] + b[2] * b[3] - inter)
    if union <= _EPS:
        return 0.0
    return float(inter / union)


def rbox_iou_one_to_many(
    box: np.ndarray,
    boxes: np.ndarray,
    *,
    backend: Optional[str] = None,
) -> np.ndarray:
    """IoU of one rbox ``[5]`` against ``[N, 5]``."""
    impl = resolve_nms_backend(backend)
    others = np.asarray(boxes, dtype=np.float64)
    n = int(others.shape[0])
    out = np.zeros((n,), dtype=np.float64)
    if n == 0:
        return out
    a = np.asarray(box, dtype=np.float64).reshape(5)
    if a[2] <= 0.0 or a[3] <= 0.0:
        return out
    corners_all = rboxes_to_corners(others)
    corners_a = rboxes_to_corners(a)[0]
    aabb_a = _aabbs_from_corners(corners_a[None, ...])[0]
    aabbs = _aabbs_from_corners(corners_all)
    area_a = float(a[2] * a[3])
    for i in range(n):
        w, h = float(others[i, 2]), float(others[i, 3])
        if w <= 0.0 or h <= 0.0:
            continue
        if not _aabb_overlaps(aabb_a, aabbs[i]):
            continue
        inter = _intersection_area(corners_a, corners_all[i], impl)
        if inter <= 0.0:
            continue
        union = area_a + w * h - inter
        if union > _EPS:
            out[i] = inter / union
    return out


def max_pairwise_rbox_iou(boxes: np.ndarray, *, backend: Optional[str] = None) -> float:
    """Maximum rotated IoU over unique pairs. Empty / single box → 0."""
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return 0.0
    best = 0.0
    n = int(arr.shape[0])
    for i in range(n):
        if arr[i, 2] <= 0.0 or arr[i, 3] <= 0.0:
            continue
        ious = rbox_iou_one_to_many(arr[i], arr[i + 1 :], backend=backend)
        if ious.size:
            best = max(best, float(ious.max()))
    return best


def any_pair_iou_at_least(
    boxes: np.ndarray,
    threshold: float,
    *,
    backend: Optional[str] = None,
) -> bool:
    """True if any unique pair has rotated IoU >= ``threshold`` (early-exit)."""
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return False
    thr = float(threshold)
    n = int(arr.shape[0])
    for i in range(n):
        if arr[i, 2] <= 0.0 or arr[i, 3] <= 0.0:
            continue
        ious = rbox_iou_one_to_many(arr[i], arr[i + 1 :], backend=backend)
        if ious.size and float(ious.max()) >= thr:
            return True
    return False


def rotated_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    iou_threshold: float,
    max_detections: Optional[int],
    class_agnostic: bool,
    backend: Optional[str] = None,
) -> np.ndarray:
    """Greedy rotated NMS. Returns keep indices in score-descending order.

    When ``class_agnostic`` is True, overlapping boxes of any class suppress each
    other (one NMS over all classes). When False (class-aware), suppression only
    applies within the same label.

    ``backend`` is ``python`` (numpy clip), ``shapely`` (CPU Shapely IoU), or ``auto``.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    n = int(boxes.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    suppressed = np.zeros(n, dtype=bool)
    keep: list[int] = []
    cap = n if max_detections is None else int(max_detections)
    iou_thr = float(iou_threshold)
    for pos, idx in enumerate(order):
        idx = int(idx)
        if suppressed[idx]:
            continue
        keep.append(idx)
        if len(keep) >= cap:
            break
        ious = rbox_iou_one_to_many(boxes[idx], boxes, backend=backend)
        same_class = (
            np.ones(n, dtype=bool) if class_agnostic else labels == labels[idx]
        )
        later = np.zeros(n, dtype=bool)
        later[order[pos + 1 :]] = True
        suppressed |= (ious > iou_thr) & same_class & later
    return np.asarray(keep, dtype=np.int64)

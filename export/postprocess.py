"""Post-NMS detection finalization for ONNX export (numpy rotated NMS + score filter).

Consumer path: numpy + ONNX Runtime. No torch and no oriented-det.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from .nms import any_pair_iou_at_least, max_pairwise_rbox_iou, rotated_nms
    from .ort_runtime import get_ort_session
except ImportError:  # copied next to model.onnx as top-level modules
    from nms import any_pair_iou_at_least, max_pairwise_rbox_iou, rotated_nms
    from ort_runtime import get_ort_session

_MIN_BOX_SIZE = 1.0

__all__ = [
    "any_pair_iou_at_least",
    "build_class_id_to_name",
    "finalize_detections_numpy",
    "max_pairwise_rbox_iou",
    "meta_to_finalize_kwargs",
    "normalize_class_id_to_name",
    "normalize_finalize_kwargs",
    "ort_pre_nms_to_detections",
    "ort_run_pre_nms",
    "score_filter_numpy",
]


def normalize_class_id_to_name(
    class_id_to_name: Optional[Dict[Any, str]],
) -> Dict[int, str]:
    """Coerce map keys to int (JSON sidecars may stringify them)."""
    if not class_id_to_name:
        return {}
    out: Dict[int, str] = {}
    for k, v in class_id_to_name.items():
        out[int(k)] = str(v)
    return out


def normalize_finalize_kwargs(finalize_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of finalize kwargs safe after Keras save/load."""
    fk = dict(finalize_kwargs)
    fk["class_id_to_name"] = normalize_class_id_to_name(fk.get("class_id_to_name"))
    return fk


def _squeeze_pre_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    count: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    boxes = np.asarray(boxes)
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if boxes.ndim == 3:
        boxes = boxes[0]
    if scores.ndim == 2:
        scores = scores[0]
    if labels.ndim == 2:
        labels = labels[0]
    return boxes, scores, labels, int(np.asarray(count).reshape(-1)[0])


def _effective_score_threshold(
    class_name: str,
    global_threshold: float,
    per_class: Optional[Dict[str, float]],
) -> float:
    if not per_class:
        return float(global_threshold)
    if class_name in per_class:
        return float(per_class[class_name])
    cn_lower = str(class_name).lower()
    for k, v in per_class.items():
        if str(k).lower() == cn_lower:
            return float(v)
    return float(global_threshold)


def _score_keep_mask(
    scores: np.ndarray,
    labels: np.ndarray,
    score_threshold: float,
    per_class_score_threshold: Optional[Dict[str, float]],
    class_id_to_name: Dict[int, str],
) -> np.ndarray:
    """Boolean mask of boxes that meet the production score floor."""
    n = int(scores.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=bool)
    if not per_class_score_threshold and (score_threshold is None or float(score_threshold) <= 0.0):
        return np.ones(n, dtype=bool)
    if not per_class_score_threshold:
        return scores >= float(score_threshold)
    keep = np.zeros(n, dtype=bool)
    for i in range(n):
        lid = int(labels[i])
        cname = class_id_to_name.get(lid, f"class_{lid}")
        thr = _effective_score_threshold(cname, score_threshold, per_class_score_threshold)
        keep[i] = float(scores[i]) >= thr
    return keep


def finalize_detections_numpy(
    pre_nms_boxes: np.ndarray,
    pre_nms_scores: np.ndarray,
    pre_nms_labels: np.ndarray,
    pre_nms_count: int,
    *,
    nms_class_agnostic: bool,
    final_nms_iou_threshold: float,
    max_detections_per_image: Optional[int],
    final_nms_use_cpu: bool = True,
    score_threshold: float,
    per_class_score_threshold: Optional[Dict[str, float]],
    class_id_to_name: Dict[Union[int, str], str],
    max_output_slots: int,
    nms_backend: str = "python",
) -> Tuple[np.ndarray, int]:
    """Run rotated NMS + production score filter; return padded ``[max_output_slots, 7]``.

    Candidates below the production score floor are dropped **before** NMS. For greedy
    score-ordered NMS this matches filtering after NMS, and avoids O(n²) NMS on
    thousands of sub-threshold boxes.

    ``final_nms_use_cpu`` is accepted for meta compatibility; NMS is always CPU.
    ``nms_backend`` is ``python`` (numpy clip), ``shapely``, or ``auto``.
    """
    del final_nms_use_cpu
    class_id_to_name_i = normalize_class_id_to_name(class_id_to_name)

    out = np.zeros((max_output_slots, 7), dtype=np.float32)
    boxes, scores, labels, n = _squeeze_pre_nms(
        pre_nms_boxes, pre_nms_scores, pre_nms_labels, pre_nms_count
    )
    if n <= 0:
        return out, 0

    boxes = np.asarray(boxes[:n], dtype=np.float32)
    scores = np.asarray(scores[:n], dtype=np.float32)
    labels = np.asarray(labels[:n], dtype=np.int64)

    keep_pre = _score_keep_mask(
        scores, labels, score_threshold, per_class_score_threshold, class_id_to_name_i
    )
    keep_pre &= boxes[:, 2] >= _MIN_BOX_SIZE
    keep_pre &= boxes[:, 3] >= _MIN_BOX_SIZE
    keep_pre &= np.isfinite(boxes).all(axis=1)
    keep_pre &= np.isfinite(scores)
    if not bool(keep_pre.any()):
        return out, 0
    boxes = boxes[keep_pre]
    scores = scores[keep_pre]
    labels = labels[keep_pre]

    keep_idx = rotated_nms(
        boxes,
        scores,
        labels,
        iou_threshold=float(final_nms_iou_threshold),
        max_detections=max_detections_per_image,
        class_agnostic=bool(nms_class_agnostic),
        backend=nms_backend,
    )
    if keep_idx.size == 0:
        return out, 0

    boxes = boxes[keep_idx]
    scores = scores[keep_idx]
    labels = labels[keep_idx]

    keep_mask = _score_keep_mask(
        scores, labels, score_threshold, per_class_score_threshold, class_id_to_name_i
    )
    keep_mask &= boxes[:, 2] >= _MIN_BOX_SIZE
    keep_mask &= boxes[:, 3] >= _MIN_BOX_SIZE
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    labels = labels[keep_mask]

    m = min(int(boxes.shape[0]), int(max_output_slots))
    if m == 0:
        return out, 0

    det = np.column_stack(
        [
            boxes[:m],
            scores[:m].reshape(-1, 1),
            labels[:m].astype(np.float32).reshape(-1, 1),
        ]
    )
    out[:m] = det
    return out, m


def build_class_id_to_name(class_names: List[str]) -> Dict[int, str]:
    """Map 1-based foreground label id to class name."""
    return {i + 1: name for i, name in enumerate(class_names)}


def ort_run_pre_nms(
    images: "np.ndarray",
    onnx_path: str,
    ort_output_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """ONNX Runtime forward; return raw pre-NMS boxes/scores/labels/count."""
    if not ort_output_names:
        raise ValueError("ort_output_names is empty; export meta output_names is required.")
    onnx_file = Path(onnx_path)
    if not onnx_file.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    img = np.asarray(images, dtype=np.float32)
    if img.ndim == 3:
        img = img[np.newaxis, ...]
    sess = get_ort_session(str(onnx_file))
    input_name = sess.get_inputs()[0].name
    outs = sess.run(ort_output_names, {input_name: img})
    name_to_val = dict(zip(ort_output_names, outs))
    for required in ("pre_nms_boxes", "pre_nms_scores", "pre_nms_labels", "pre_nms_count"):
        if required not in name_to_val:
            raise KeyError(
                f"Missing ONNX output {required!r}; got {sorted(name_to_val)}. "
                "Re-export with faster_rcnn_pre_nms, oriented_rcnn_pre_nms, "
                "rotated_fcos_pre_nms, or rotated_rtmdet_pre_nms."
            )
    count = int(np.asarray(name_to_val["pre_nms_count"]).reshape(-1)[0])
    return (
        np.asarray(name_to_val["pre_nms_boxes"]),
        np.asarray(name_to_val["pre_nms_scores"]),
        np.asarray(name_to_val["pre_nms_labels"]),
        count,
    )


def score_filter_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    count: int,
    score_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep finite boxes with positive size and score >= threshold."""
    n = int(count)
    if n <= 0:
        empty_b = np.zeros((0, 5), dtype=np.float32)
        return empty_b, scores[:0], labels[:0]
    boxes = np.asarray(boxes[:n], dtype=np.float32)
    scores = np.asarray(scores[:n], dtype=np.float32)
    labels = np.asarray(labels[:n])
    keep = scores >= float(score_threshold)
    keep &= boxes[:, 2] > 0
    keep &= boxes[:, 3] > 0
    keep &= np.isfinite(boxes).all(axis=1)
    keep &= np.isfinite(scores)
    return boxes[keep], scores[keep], labels[keep]


def ort_pre_nms_to_detections(
    images: "np.ndarray",
    onnx_path: str,
    ort_output_names: List[str],
    finalize_kwargs: Dict[str, Any],
) -> Tuple["np.ndarray", int]:
    """ONNX Runtime forward + finalize (for TF ``numpy_function`` / Keras bundle)."""
    boxes, scores, labels, count = ort_run_pre_nms(images, onnx_path, ort_output_names)
    detections, num = finalize_detections_numpy(
        boxes,
        scores,
        labels,
        count,
        **normalize_finalize_kwargs(finalize_kwargs),
    )
    return detections, int(num)


def meta_to_finalize_kwargs(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Extract finalize_detections_numpy kwargs from export meta JSON."""
    prod = meta.get("production") or {}
    post = meta.get("postprocess") or {}
    class_names: List[str] = list(meta.get("class_names") or [])
    max_det = int(
        prod.get("max_detections_per_image")
        or post.get("max_detections")
        or meta.get("max_detections_per_image")
        or meta.get("max_detections")
        or 3000
    )
    nms_ag = prod.get("nms_class_agnostic")
    if nms_ag is None:
        nms_ag = post.get("nms_class_agnostic", False)
    return normalize_finalize_kwargs(
        {
            "nms_class_agnostic": bool(nms_ag),
            "final_nms_iou_threshold": float(
                prod.get("final_nms_iou_threshold")
                or post.get("nms_iou_threshold")
                or 0.1
            ),
            "max_detections_per_image": max_det,
            "final_nms_use_cpu": bool(
                prod.get("final_nms_use_cpu", post.get("final_nms_use_cpu", True))
            ),
            "score_threshold": float(
                prod.get("score_threshold") or post.get("score_threshold") or 0.05
            ),
            "per_class_score_threshold": prod.get("per_class_score_threshold"),
            "class_id_to_name": build_class_id_to_name(class_names),
            "max_output_slots": max_det,
            "nms_backend": str(
                post.get("nms_backend") or prod.get("nms_backend") or "python"
            ),
        }
    )

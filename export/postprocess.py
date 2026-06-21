"""Post-NMS detection finalization for TF export (exact CPU rotated NMS + score filter)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from export.ort_runtime import get_ort_device, get_ort_session
from oriented_det.models.faster_rcnn_inference import PreNmsDetections, apply_final_rotated_nms
from oriented_det.train.utils import effective_score_threshold_for_class_name


class _NmsConfigView:
    """Minimal attribute bag for :func:`apply_final_rotated_nms`."""

    def __init__(
        self,
        *,
        nms_class_agnostic: bool,
        final_nms_iou_threshold: float,
        max_detections_per_image: Optional[int],
        final_nms_use_cpu: bool,
    ) -> None:
        self.nms_class_agnostic = nms_class_agnostic
        self.final_nms_iou_threshold = final_nms_iou_threshold
        self.max_detections_per_image = max_detections_per_image
        self.final_nms_use_cpu = final_nms_use_cpu


def finalize_detections_numpy(
    pre_nms_boxes: np.ndarray,
    pre_nms_scores: np.ndarray,
    pre_nms_labels: np.ndarray,
    pre_nms_count: int,
    *,
    nms_class_agnostic: bool,
    final_nms_iou_threshold: float,
    max_detections_per_image: Optional[int],
    final_nms_use_cpu: bool,
    score_threshold: float,
    per_class_score_threshold: Optional[Dict[str, float]],
    class_id_to_name: Dict[int, str],
    max_output_slots: int,
) -> Tuple[np.ndarray, int]:
    """Run rotated NMS + production score filter; return padded ``[max_output_slots, 7]``."""
    if torch is None:
        raise RuntimeError("torch is required for finalize_detections_numpy")

    out = np.zeros((max_output_slots, 7), dtype=np.float32)
    n = int(pre_nms_count)
    if n <= 0:
        return out, 0

    boxes_t = torch.from_numpy(np.asarray(pre_nms_boxes[:n], dtype=np.float32))
    scores_t = torch.from_numpy(np.asarray(pre_nms_scores[:n], dtype=np.float32))
    labels_t = torch.from_numpy(np.asarray(pre_nms_labels[:n], dtype=np.int64))

    nms_view = _NmsConfigView(
        nms_class_agnostic=nms_class_agnostic,
        final_nms_iou_threshold=final_nms_iou_threshold,
        max_detections_per_image=max_detections_per_image,
        final_nms_use_cpu=final_nms_use_cpu,
    )
    final = apply_final_rotated_nms(nms_view, PreNmsDetections(boxes_t, scores_t, labels_t))

    if final.boxes.numel() == 0:
        return out, 0

    keep_mask = torch.ones(final.scores.shape[0], dtype=torch.bool)
    if per_class_score_threshold or score_threshold is not None:
        keep_list = []
        for i in range(int(final.scores.shape[0])):
            lid = int(final.labels[i].item())
            cname = class_id_to_name.get(lid, f"class_{lid}")
            thr = effective_score_threshold_for_class_name(
                cname, score_threshold, per_class_score_threshold
            )
            keep_list.append(float(final.scores[i].item()) >= thr)
        keep_mask = torch.tensor(keep_list, dtype=torch.bool)

    final_boxes = final.boxes[keep_mask]
    final_scores = final.scores[keep_mask]
    final_labels = final.labels[keep_mask]

    m = min(int(final_boxes.shape[0]), max_output_slots)
    if m == 0:
        return out, 0

    det = np.column_stack(
        [
            final_boxes[:m].cpu().numpy(),
            final_scores[:m].cpu().numpy().reshape(-1, 1),
            final_labels[:m].cpu().numpy().astype(np.float32).reshape(-1, 1),
        ]
    )
    out[:m] = det
    return out, m


def build_class_id_to_name(class_names: List[str]) -> Dict[int, str]:
    """Map 1-based foreground label id to class name."""
    return {i + 1: name for i, name in enumerate(class_names)}


def ort_pre_nms_to_detections(
    images: "np.ndarray",
    onnx_path: str,
    ort_output_names: List[str],
    finalize_kwargs: Dict[str, Any],
) -> Tuple["np.ndarray", int]:
    """ONNX Runtime forward + finalize (for TF ``numpy_function`` / Keras bundle)."""
    import numpy as np

    img = np.asarray(images, dtype=np.float32)
    if img.ndim == 3:
        img = img[np.newaxis, ...]
    sess = get_ort_session(onnx_path)
    input_name = sess.get_inputs()[0].name
    outs = sess.run(ort_output_names, {input_name: img})
    name_to_val = dict(zip(ort_output_names, outs))
    detections, num = finalize_detections_numpy(
        name_to_val["pre_nms_boxes"],
        name_to_val["pre_nms_scores"],
        name_to_val["pre_nms_labels"],
        int(np.asarray(name_to_val["pre_nms_count"]).reshape(-1)[0]),
        **finalize_kwargs,
    )
    return detections, int(num)


def meta_to_finalize_kwargs(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Extract finalize_detections_numpy kwargs from export meta JSON."""
    prod = meta.get("production") or {}
    class_names: List[str] = list(meta.get("class_names") or [])
    max_det = int(prod.get("max_detections_per_image") or meta.get("max_detections_per_image") or 3000)
    return {
        "nms_class_agnostic": bool(prod.get("nms_class_agnostic", False)),
        "final_nms_iou_threshold": float(prod.get("final_nms_iou_threshold", 0.1)),
        "max_detections_per_image": max_det,
        "final_nms_use_cpu": bool(prod.get("final_nms_use_cpu", True)),
        "score_threshold": float(prod.get("score_threshold", 0.05)),
        "per_class_score_threshold": prod.get("per_class_score_threshold"),
        "class_id_to_name": build_class_id_to_name(class_names),
        "max_output_slots": max_det,
    }

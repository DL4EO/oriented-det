"""Post-NMS detection finalization for TF export (exact CPU rotated NMS + score filter)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from export.ort_runtime import get_ort_session
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


def normalize_class_id_to_name(
    class_id_to_name: Optional[Dict[Any, str]],
) -> Dict[int, str]:
    """Coerce map keys to int (Keras JSON round-trip stringifies them)."""
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
    class_id_to_name: Dict[Union[int, str], str],
    max_output_slots: int,
) -> Tuple[np.ndarray, int]:
    """Run rotated NMS + production score filter; return padded ``[max_output_slots, 7]``.

    Candidates below the production score floor are dropped **before** NMS. For greedy
    score-ordered NMS this matches filtering after NMS, and avoids O(n²) exact CPU NMS
    on thousands of sub-threshold boxes (the dominant cost when ORT CUDA is ~tens of ms).
    """
    if torch is None:
        raise RuntimeError("torch is required for finalize_detections_numpy")

    class_id_to_name_i = normalize_class_id_to_name(class_id_to_name)

    out = np.zeros((max_output_slots, 7), dtype=np.float32)
    n = int(pre_nms_count)
    if n <= 0:
        return out, 0

    boxes_t = torch.from_numpy(np.asarray(pre_nms_boxes[:n], dtype=np.float32).copy())
    scores_t = torch.from_numpy(np.asarray(pre_nms_scores[:n], dtype=np.float32).copy())
    labels_t = torch.from_numpy(np.asarray(pre_nms_labels[:n], dtype=np.int64).copy())

    keep_pre = _score_keep_mask(
        scores_t, labels_t, score_threshold, per_class_score_threshold, class_id_to_name_i
    )
    if not bool(keep_pre.any()):
        return out, 0
    boxes_t = boxes_t[keep_pre]
    scores_t = scores_t[keep_pre]
    labels_t = labels_t[keep_pre]

    # GPU NMS when allowed — ORT CUDA is wasted if we always stay on CPU here.
    if (not final_nms_use_cpu) and torch.cuda.is_available():
        boxes_t = boxes_t.cuda(non_blocking=True)
        scores_t = scores_t.cuda(non_blocking=True)
        labels_t = labels_t.cuda(non_blocking=True)

    nms_view = _NmsConfigView(
        nms_class_agnostic=nms_class_agnostic,
        final_nms_iou_threshold=final_nms_iou_threshold,
        max_detections_per_image=max_detections_per_image,
        final_nms_use_cpu=final_nms_use_cpu,
    )
    final = apply_final_rotated_nms(nms_view, PreNmsDetections(boxes_t, scores_t, labels_t))

    if final.boxes.numel() == 0:
        return out, 0

    # Scores already pre-filtered; keep a cheap post-pass for API stability / per-class.
    keep_mask = _score_keep_mask(
        final.scores,
        final.labels,
        score_threshold,
        per_class_score_threshold,
        class_id_to_name_i,
    )
    final_boxes = final.boxes[keep_mask]
    final_scores = final.scores[keep_mask]
    final_labels = final.labels[keep_mask]

    m = min(int(final_boxes.shape[0]), max_output_slots)
    if m == 0:
        return out, 0

    det = np.column_stack(
        [
            final_boxes[:m].detach().cpu().numpy(),
            final_scores[:m].detach().cpu().numpy().reshape(-1, 1),
            final_labels[:m].detach().cpu().numpy().astype(np.float32).reshape(-1, 1),
        ]
    )
    out[:m] = det
    return out, m


def _score_keep_mask(
    scores: "torch.Tensor",
    labels: "torch.Tensor",
    score_threshold: float,
    per_class_score_threshold: Optional[Dict[str, float]],
    class_id_to_name: Dict[int, str],
) -> "torch.Tensor":
    """Boolean mask of boxes that meet the production score floor."""
    if torch is None:
        raise RuntimeError("torch is required")
    if not per_class_score_threshold and (score_threshold is None or float(score_threshold) <= 0.0):
        return torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
    if not per_class_score_threshold:
        return scores >= float(score_threshold)
    keep = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
    for i in range(int(scores.shape[0])):
        lid = int(labels[i].item())
        cname = class_id_to_name.get(lid, f"class_{lid}")
        thr = effective_score_threshold_for_class_name(
            cname, score_threshold, per_class_score_threshold
        )
        keep[i] = float(scores[i].item()) >= thr
    return keep


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
                "Re-export with faster_rcnn_pre_nms, oriented_rcnn_pre_nms, or rotated_fcos_pre_nms."
            )
    detections, num = finalize_detections_numpy(
        name_to_val["pre_nms_boxes"],
        name_to_val["pre_nms_scores"],
        name_to_val["pre_nms_labels"],
        int(np.asarray(name_to_val["pre_nms_count"]).reshape(-1)[0]),
        **normalize_finalize_kwargs(finalize_kwargs),
    )
    return detections, int(num)


def meta_to_finalize_kwargs(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Extract finalize_detections_numpy kwargs from export meta JSON."""
    prod = meta.get("production") or {}
    class_names: List[str] = list(meta.get("class_names") or [])
    max_det = int(prod.get("max_detections_per_image") or meta.get("max_detections_per_image") or 3000)
    return normalize_finalize_kwargs(
        {
            "nms_class_agnostic": bool(prod.get("nms_class_agnostic", False)),
            "final_nms_iou_threshold": float(prod.get("final_nms_iou_threshold", 0.1)),
            "max_detections_per_image": max_det,
            "final_nms_use_cpu": bool(prod.get("final_nms_use_cpu", True)),
            "score_threshold": float(prod.get("score_threshold", 0.05)),
            "per_class_score_threshold": prod.get("per_class_score_threshold"),
            "class_id_to_name": build_class_id_to_name(class_names),
            "max_output_slots": max_det,
        }
    )

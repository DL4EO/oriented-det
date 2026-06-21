"""Rotated IoU / NMS backend switch.

The first release is intentionally self-contained: no MMCV dependency and no new
custom CUDA kernels. The default path uses this repo's parallel tensor/GPU
sampling implementation. A CPU backend exists only for debugging/reference
checks. If profiling shows a large win later, add a new in-repo CUDA-kernel
backend behind this abstraction.

Callers choose behavior via env ``ORIENTED_DET_ROTATED_BACKEND``:
- ``gpu_sample`` (default): sampling-based parallel GPU implementation.
- ``auto``: alias for ``gpu_sample`` for convenience.
- ``cpu``: CPU polygon path for debugging/reference checks.

Sampling granularity for the GPU path (perfect squares such as ``9``, ``49``, ``1024``):

Geometry-based sizing (**default**): grid count from box ``w``/``h`` and aspect
ratio (see ``ops/README.md``).

- ``ORIENTED_DET_GPU_ORIENTED_IOU_TARGET_SPACING_PX`` (default ``2`` px).
- ``ORIENTED_DET_GPU_ORIENTED_IOU_MIN_SAMPLES`` / ``ORIENTED_DET_GPU_ORIENTED_IOU_MAX_SAMPLES``
  (defaults ``25`` … ``1024``).
- ``ORIENTED_DET_GPU_ORIENTED_IOU_SAMPLE_BY_MAX_SIDE``: **on by default**; set
  ``0`` / ``false`` / ``no`` / ``off`` for a flat ``100``-sample grid (debug only).
- ``ORIENTED_DET_GPU_NMS_IOU_SAMPLES`` — NMS floor / flat count when geometry off (default ``100``).
- ``ORIENTED_DET_GPU_NMS_IOU_SAMPLE_BY_MAX_SIDE`` — NMS geometry toggle (default on).
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional

try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    Tensor = None  # type: ignore


class RotatedBackend(str, Enum):
    AUTO = "auto"
    GPU_SAMPLE = "gpu_sample"
    CPU = "cpu"


def get_rotated_backend() -> RotatedBackend:
    raw = os.environ.get("ORIENTED_DET_ROTATED_BACKEND", "auto").strip().lower()
    try:
        return RotatedBackend(raw)
    except ValueError:
        return RotatedBackend.AUTO


def rotated_nms(
    boxes: Tensor,
    scores: Tensor,
    iou_threshold: float,
    max_detections: Optional[int] = None,
    *,
    force_cpu: bool = False,
) -> Tensor:
    """Class-wise caller should loop; this NMS runs on one pool of boxes/scores.

    Args:
        force_cpu: If True, always use exact polygon IoU NMS on CPU for this call
            (Shapely when installed, else Sutherland–Hodgman with a warning),
            ignoring ``ORIENTED_DET_ROTATED_BACKEND`` and the GPU sampling path.
    """
    if torch is None or boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device if boxes.numel() else "cpu")
    backend = RotatedBackend.CPU if force_cpu else get_rotated_backend()
    if backend == RotatedBackend.CPU:
        from ..geometry import RBox
        from .nms import oriented_nms
        from .utils import resolve_exact_polygon_iou_backend

        resolve_exact_polygon_iou_backend()
        rboxes = [RBox(*t.tolist()) for t in boxes.detach().cpu()]
        scores_list = [float(s) for s in scores.detach().cpu()]
        idx = oriented_nms(rboxes, scores_list, iou_threshold, max_detections)
        return torch.tensor(idx, dtype=torch.long, device=boxes.device)

    from .gpu_ops import oriented_nms_gpu

    return oriented_nms_gpu(
        boxes,
        scores,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
    )


def pairwise_rotated_iou(
    boxes1: Tensor,
    boxes2: Tensor,
) -> Tensor:
    """Return [N, M] IoU matrix."""
    if torch is None:
        raise RuntimeError("torch required")
    backend = get_rotated_backend()
    if backend == RotatedBackend.CPU:
        from ..geometry import RBox
        from .iou import batch_rbox_iou

        a = [RBox(*t.tolist()) for t in boxes1.detach().cpu()]
        b = [RBox(*t.tolist()) for t in boxes2.detach().cpu()]
        mat = batch_rbox_iou(a, b, device=boxes1.device)
        return torch.tensor(mat, dtype=torch.float32, device=boxes1.device)

    from .gpu_ops import oriented_box_iou_gpu

    return oriented_box_iou_gpu(boxes1, boxes2)


__all__ = [
    "RotatedBackend",
    "get_rotated_backend",
    "rotated_nms",
    "pairwise_rotated_iou",
]

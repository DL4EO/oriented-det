"""Ops module exposing oriented IoU and NMS helpers."""

from . import iou, nms, utils, gpu_ops, rotated_ops
from .iou import polygon_iou, qbox_iou, rbox_iou, batch_rbox_iou
from .nms import oriented_nms
from .gpu_ops import (
    oriented_box_iou_gpu,
    obb_to_xyxy_gpu,
    hbb_iou_gpu,
    oriented_box_hbb_iou_gpu,
    generate_oriented_anchors_gpu,
    match_anchors_to_gt_gpu,
    oriented_nms_gpu,
)
from .kfiou import (
    kfiou_loss,
    kfiou_loss_per_box,
    kfiou_overlap_ratio,
    mean_auxiliary_box_reg_loss,
    xy_wh_r_to_xy_sigma,
)
from .probiou import probiou_loss, probiou_loss_per_box
from .gaussian_angle import aspect_gated_angle_loss_per_box
from .diff_iou_rotated import diff_iou_rotated_2d, riou_loss_per_box

# Canonical OBB -> xyxy (HBB) conversion; works on CPU and GPU
obb_to_xyxy = obb_to_xyxy_gpu
rotated_nms = rotated_ops.rotated_nms
pairwise_rotated_iou = rotated_ops.pairwise_rotated_iou

__all__ = [
    "polygon_iou",
    "qbox_iou",
    "rbox_iou",
    "batch_rbox_iou",
    "oriented_nms",
    "iou",
    "nms",
    "utils",
    "gpu_ops",
    "rotated_ops",
    # GPU-accelerated functions
    "oriented_box_iou_gpu",
    "obb_to_xyxy_gpu",
    "hbb_iou_gpu",
    "oriented_box_hbb_iou_gpu",
    "generate_oriented_anchors_gpu",
    "match_anchors_to_gt_gpu",
    "oriented_nms_gpu",
    "obb_to_xyxy",
    "rotated_nms",
    "pairwise_rotated_iou",
    "xy_wh_r_to_xy_sigma",
    "kfiou_overlap_ratio",
    "kfiou_loss_per_box",
    "kfiou_loss",
    "mean_auxiliary_box_reg_loss",
    "probiou_loss_per_box",
    "probiou_loss",
    "aspect_gated_angle_loss_per_box",
    "diff_iou_rotated_2d",
    "riou_loss_per_box",
]

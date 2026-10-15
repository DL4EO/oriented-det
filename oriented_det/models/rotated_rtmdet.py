"""Rotated RTMDet: CSPNeXt + PAFPN single-stage oriented detector (clean-room).

Architecture (public RTMDet-R design):
- CSPNeXt-tiny + CSPNeXt-PAFPN (P3–P5)
- SepBN head (shared convs, per-level BN); no centerness
- Dynamic soft-label assigner (SimOTA-style cost: cls + rotated IoU)
- Quality Focal Loss + decoded rotated IoU box regression
- DistanceAnglePointCoder (l, t, r, b, θ) le90
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

from ..ops.diff_iou_rotated import riou_loss_per_box, diff_iou_rotated_2d
from ..ops.rotated_ops import rotated_nms
from .bbox_coder import DistanceAnglePointCoder
from .backbones.cspnext import build_cspnext_pafpn_backbone
from .faster_rcnn_inference import PreNmsDetections, pad_pre_nms_detections
from .oriented_rpn import norm_angle_le90
from .rotated_fcos import generate_fpn_points, Scale
from .utils import (
    prepare_targets,
    extract_backbone_features,
    derive_fpn_strides_from_grid,
    warn_if_fpn_strides_mismatch,
    tensor_to_rboxes,
    SigmoidFocalClassWeightsMixin,
)


# ---------------------------------------------------------------------------
# Quality Focal Loss
# ---------------------------------------------------------------------------

def quality_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float = 2.0,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Quality Focal Loss (sum reduction).

    Args:
        pred: [N, C] logits
        target: [N, C] continuous quality targets in [0, 1] (IoU for positives)
        beta: focusing parameter
        weight: optional [C] per-class scale (applied to all entries of that column)
    """
    pred_sig = pred.sigmoid()
    scale = (target - pred_sig).abs().pow(float(beta))
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * scale
    if weight is not None:
        loss = loss * weight.view(1, -1).to(device=loss.device, dtype=loss.dtype)
    return loss.sum()


# ---------------------------------------------------------------------------
# Dynamic soft-label assigner (SimOTA-style for OBBs)
# ---------------------------------------------------------------------------

def _points_in_obb(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Boolean [P, G]: point inside oriented box (local frame)."""
    if boxes.numel() == 0:
        return points.new_zeros((points.size(0), 0), dtype=torch.bool)
    dx = points[:, None, 0] - boxes[None, :, 0]
    dy = points[:, None, 1] - boxes[None, :, 1]
    cos_a = torch.cos(boxes[None, :, 4])
    sin_a = torch.sin(boxes[None, :, 4])
    local_x = dx * cos_a + dy * sin_a
    local_y = -dx * sin_a + dy * cos_a
    hw = boxes[None, :, 2] * 0.5
    hh = boxes[None, :, 3] * 0.5
    return (local_x.abs() <= hw) & (local_y.abs() <= hh)


def dynamic_soft_label_assign_single(
    points: torch.Tensor,
    cls_logits: torch.Tensor,
    decoded_boxes: torch.Tensor,
    gt_bboxes: torch.Tensor,
    gt_labels: torch.Tensor,
    num_classes: int,
    *,
    soft_center_radius: float = 3.0,
    topk: int = 13,
    iou_weight: float = 3.0,
    strides: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign soft labels for one image.

    Returns:
        labels [P] in {0..K-1 fg, K bg},
        soft_cls_target [P, K] quality targets,
        bbox_targets [P, 5] encoded (l,t,r,b,θ) absolute (caller may stride-norm),
        pos_mask [P] bool
    """
    device = points.device
    num_points = points.size(0)
    num_gts = gt_bboxes.size(0)
    bg = num_classes
    labels = points.new_full((num_points,), bg, dtype=torch.long)
    soft_target = points.new_zeros((num_points, num_classes))
    bbox_targets = points.new_zeros((num_points, 5))
    pos_mask = torch.zeros(num_points, dtype=torch.bool, device=device)

    if num_gts == 0 or num_points == 0:
        return labels, soft_target, bbox_targets, pos_mask

    # Prior: points inside GT OBBs (optionally soft center prior via stride)
    inside = _points_in_obb(points, gt_bboxes)  # [P, G]
    if strides is not None and soft_center_radius > 0:
        # Soft center: also require distance to GT center <= radius * stride
        dists = torch.sqrt(
            ((points[:, None, :] - gt_bboxes[None, :, :2]) ** 2).sum(dim=-1)
        )
        stride_col = strides[:, None].expand_as(dists)
        center_ok = dists <= (float(soft_center_radius) * stride_col)
        prior = inside & center_ok
    else:
        prior = inside

    if not prior.any():
        return labels, soft_target, bbox_targets, pos_mask

    # IoU only on prior pairs. Non-prior costs are discarded, and mosaic makes the
    # full point×GT matrix (plus a Python loop over boxes) the training bottleneck.
    with torch.no_grad():
        point_idx, gt_idx = prior.nonzero(as_tuple=True)
        iou = points.new_zeros((num_points, num_gts))
        pair_iou = diff_iou_rotated_2d(
            decoded_boxes.detach()[point_idx],
            gt_bboxes.detach()[gt_idx],
        )
        iou[point_idx, gt_idx] = pair_iou

        pred_sig = cls_logits.detach().sigmoid().clamp(1e-6, 1 - 1e-6)
        cls_index = (gt_labels.to(dtype=torch.long) - 1).clamp(0, num_classes - 1)
        valid_cls = (gt_labels >= 1) & (gt_labels <= num_classes)
        soft_y = iou[point_idx, gt_idx].clamp(0, 1)
        p = pred_sig[point_idx, cls_index[gt_idx]]
        pair_cls = -(soft_y * torch.log(p) + (1.0 - soft_y) * torch.log(1.0 - p))
        pair_cls = torch.where(valid_cls[gt_idx], pair_cls, torch.zeros_like(pair_cls))
        cls_cost = torch.zeros((num_points, num_gts), device=device, dtype=pred_sig.dtype)
        cls_cost[point_idx, gt_idx] = pair_cls

        iou_cost = -torch.log(iou.clamp(min=1e-6))
        cost = cls_cost + float(iou_weight) * iou_cost
        cost = cost.masked_fill(~prior, 1e6)

        # Dynamic k: per GT, top-k lowest-cost prior points. A point claimed by
        # several boxes keeps the lowest cost; ties keep the earlier GT.
        k = min(int(topk), num_points)
        top_cost, top_idx = torch.topk(cost, k, dim=0, largest=False)
        selected = torch.zeros_like(prior)
        selected.scatter_(0, top_idx, top_cost < 1e5)
        cost_selected = cost.masked_fill(~selected, float("inf"))
        best_cost, best_gt = cost_selected.min(dim=1)
        pos = torch.isfinite(best_cost)
        if not pos.any():
            return labels, soft_target, bbox_targets, pos_mask

        matched_gt = torch.where(pos, best_gt, torch.full_like(best_gt, -1))
        pos_inds = pos.nonzero(as_tuple=False).squeeze(1)
        gt_inds = matched_gt[pos_inds]
        labels[pos_inds] = (gt_labels[gt_inds] - 1).clamp(min=0, max=num_classes - 1)
        pos_mask[pos_inds] = True
        soft_target[pos_inds, labels[pos_inds]] = iou[pos_inds, gt_inds].clamp(0, 1).to(
            soft_target.dtype
        )

        # Encoded bbox targets via DistanceAnglePointCoder (absolute ltrb + angle)
        coder = DistanceAnglePointCoder(angle_version="le90")
        encoded = coder.encode(points[pos_inds], gt_bboxes[gt_inds])
        bbox_targets[pos_inds] = encoded

    return labels, soft_target, bbox_targets, pos_mask


# ---------------------------------------------------------------------------
# SepBN head
# ---------------------------------------------------------------------------

class _SepBNConv(nn.Module):
    """Shared Conv2d weights + per-level BatchNorm + SiLU."""

    def __init__(self, in_channels: int, out_channels: int, num_levels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03) for _ in range(num_levels)]
        )
        self.act = nn.SiLU(inplace=True)
        nn.init.normal_(self.conv.weight, std=0.01)

    def forward(self, x: torch.Tensor, level: int) -> torch.Tensor:
        return self.act(self.bns[level](self.conv(x)))


class RTMDetSepBNHead(nn.Module):
    """RTMDet-style sep-BN head: shared towers, per-level BN; cls + bbox(4) + angle."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        stacked_convs: int = 2,
        feat_channels: Optional[int] = None,
        strides: Sequence[float] = (8, 16, 32),
        norm_on_bbox: bool = True,
        scale_angle: bool = True,
        exp_on_reg: bool = True,
    ):
        if nn is None:
            raise RuntimeError("PyTorch is required for RTMDetSepBNHead.")
        super().__init__()
        self.num_classes = int(num_classes)
        self.stacked_convs = max(1, int(stacked_convs))
        self.feat_channels = int(feat_channels or in_channels)
        self.strides = list(strides)
        self.num_levels = len(self.strides)
        self.norm_on_bbox = bool(norm_on_bbox)
        self.exp_on_reg = bool(exp_on_reg)
        self.scale_angle = bool(scale_angle)

        self.cls_convs = nn.ModuleList(
            [
                _SepBNConv(
                    in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    self.num_levels,
                )
                for i in range(self.stacked_convs)
            ]
        )
        self.reg_convs = nn.ModuleList(
            [
                _SepBNConv(
                    in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    self.num_levels,
                )
                for i in range(self.stacked_convs)
            ]
        )
        self.rtm_cls = nn.Conv2d(self.feat_channels, self.num_classes, 1)
        self.rtm_reg = nn.Conv2d(self.feat_channels, 4, 1)
        self.rtm_angle = nn.Conv2d(self.feat_channels, 1, 1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.scale_angle_module = Scale(1.0) if scale_angle else None

        nn.init.normal_(self.rtm_cls.weight, std=0.01)
        bias_init = -math.log((1 - 0.01) / 0.01)
        nn.init.constant_(self.rtm_cls.bias, bias_init)
        nn.init.normal_(self.rtm_reg.weight, std=0.01)
        nn.init.constant_(self.rtm_reg.bias, 0)
        nn.init.normal_(self.rtm_angle.weight, std=0.01)
        nn.init.constant_(self.rtm_angle.bias, 0)

    def forward(
        self,
        features: List[torch.Tensor],
        strides: Optional[Sequence[float]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        cls_scores: List[torch.Tensor] = []
        bbox_preds: List[torch.Tensor] = []
        angle_preds: List[torch.Tensor] = []
        stride_list = list(strides) if strides is not None else list(self.strides)
        if len(features) != self.num_levels:
            raise ValueError(
                f"features length {len(features)} != num_levels {self.num_levels}"
            )
        if len(stride_list) != len(features):
            raise ValueError(
                f"strides length {len(stride_list)} != features length {len(features)}"
            )
        for lvl, (feat, scale, stride) in enumerate(zip(features, self.scales, stride_list)):
            cls_feat = feat
            for conv in self.cls_convs:
                cls_feat = conv(cls_feat, lvl)
            reg_feat = feat
            for conv in self.reg_convs:
                reg_feat = conv(reg_feat, lvl)

            cls_scores.append(self.rtm_cls(cls_feat))
            bbox = scale(self.rtm_reg(reg_feat)).float()
            if self.exp_on_reg:
                bbox = bbox.exp()
            if self.norm_on_bbox:
                # Predictions are stride-normalized distances; multiply at inference.
                if not self.training:
                    bbox = bbox * float(stride)
            bbox_preds.append(bbox)

            angle = self.rtm_angle(reg_feat)
            if self.scale_angle_module is not None:
                angle = self.scale_angle_module(angle).float()
            angle_preds.append(angle)
        return cls_scores, bbox_preds, angle_preds


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_rotated_rtmdet_loss(
    cls_scores: List[torch.Tensor],
    bbox_preds: List[torch.Tensor],
    angle_preds: List[torch.Tensor],
    points: List[torch.Tensor],
    strides: Sequence[float],
    gt_boxes: List[torch.Tensor],
    gt_labels: List[torch.Tensor],
    num_classes: int,
    *,
    bbox_coder: Optional[DistanceAnglePointCoder] = None,
    norm_on_bbox: bool = True,
    qfl_beta: float = 2.0,
    box_reg_weight: float = 2.0,
    angle_weight: float = 0.2,
    assign_topk: int = 13,
    soft_center_radius: float = 3.0,
    iou_weight: float = 3.0,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """QFL classification + decoded rIoU box reg (+ encoded angle L1)."""
    if bbox_coder is None:
        bbox_coder = DistanceAnglePointCoder(angle_version="le90")

    device = cls_scores[0].device
    num_levels = len(points)
    num_imgs = cls_scores[0].size(0)
    num_points_per_lvl = [p.size(0) for p in points]
    concat_points = torch.cat(points, dim=0)

    # Flatten predictions per image then assign
    loss_cls_sum = cls_scores[0].new_tensor(0.0)
    loss_box_sum = cls_scores[0].new_tensor(0.0)
    loss_angle_sum = cls_scores[0].new_tensor(0.0)
    total_pos = 0

    for img_idx in range(num_imgs):
        cls_i = torch.cat(
            [
                cls_scores[lvl][img_idx].permute(1, 2, 0).reshape(-1, num_classes)
                for lvl in range(num_levels)
            ],
            dim=0,
        )
        bbox_i = torch.cat(
            [
                bbox_preds[lvl][img_idx].permute(1, 2, 0).reshape(-1, 4)
                for lvl in range(num_levels)
            ],
            dim=0,
        )
        angle_i = torch.cat(
            [
                angle_preds[lvl][img_idx].permute(1, 2, 0).reshape(-1, 1)
                for lvl in range(num_levels)
            ],
            dim=0,
        )
        stride_t = torch.cat(
            [
                concat_points.new_full((num_points_per_lvl[lvl],), float(strides[lvl]))
                for lvl in range(num_levels)
            ],
            dim=0,
        )

        # Decode for assignment (absolute coords)
        if norm_on_bbox:
            decode_ltrb = bbox_i * stride_t.unsqueeze(-1)
        else:
            decode_ltrb = bbox_i
        decoded = bbox_coder.decode(
            concat_points, torch.cat([decode_ltrb, angle_i], dim=-1)
        )

        labels, soft_tgt, bbox_tgt, pos_mask = dynamic_soft_label_assign_single(
            concat_points,
            cls_i,
            decoded,
            gt_boxes[img_idx],
            gt_labels[img_idx],
            num_classes,
            soft_center_radius=soft_center_radius,
            topk=assign_topk,
            iou_weight=iou_weight,
            strides=stride_t,
        )

        # QFL over all points
        loss_cls_sum = loss_cls_sum + quality_focal_loss(
            cls_i, soft_tgt, beta=qfl_beta, weight=class_weights
        )

        num_pos = int(pos_mask.sum().item())
        total_pos += num_pos
        if num_pos == 0:
            loss_box_sum = loss_box_sum + bbox_i.sum() * 0.0
            loss_angle_sum = loss_angle_sum + angle_i.sum() * 0.0
            continue

        pos_pts = concat_points[pos_mask]
        pos_bbox = bbox_i[pos_mask]
        pos_angle = angle_i[pos_mask]
        pos_tgt = bbox_tgt[pos_mask]
        pos_stride = stride_t[pos_mask]

        # Targets: absolute ltrb; normalize if needed
        tgt_ltrb = pos_tgt[:, :4]
        tgt_angle = pos_tgt[:, 4:5]
        if norm_on_bbox:
            pred_ltrb_abs = pos_bbox * pos_stride.unsqueeze(-1)
            tgt_ltrb_norm = tgt_ltrb / pos_stride.unsqueeze(-1)
            # For angle L1 we keep radians
        else:
            pred_ltrb_abs = pos_bbox
            tgt_ltrb_norm = tgt_ltrb

        decoded_pred = bbox_coder.decode(
            pos_pts, torch.cat([pred_ltrb_abs, pos_angle], dim=-1)
        )
        decoded_tgt = bbox_coder.decode(
            pos_pts, torch.cat([tgt_ltrb, tgt_angle], dim=-1)
        ).detach()

        loss_box_sum = loss_box_sum + riou_loss_per_box(decoded_pred, decoded_tgt).sum()
        ang_err = norm_angle_le90(pos_angle.squeeze(-1) - tgt_angle.squeeze(-1)).abs()
        loss_angle_sum = loss_angle_sum + ang_err.sum()

        # Keep unused var referenced for graph stability when num_pos>0 but riou fails
        _ = tgt_ltrb_norm

    denom = max(float(total_pos), 1.0)
    out = {
        "loss_classifier": loss_cls_sum / denom,
        "loss_box_reg": float(box_reg_weight) * loss_box_sum / denom,
        "loss_angle": float(angle_weight) * loss_angle_sum / denom,
    }
    return out


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _rtmdet_level_topk_decode(
    cls_map: torch.Tensor,
    bbox_map: torch.Tensor,
    angle_map: torch.Tensor,
    pts: torch.Tensor,
    bbox_coder: DistanceAnglePointCoder,
    num_classes: int,
    score_threshold: float,
    nms_pre: int,
    min_bbox_size: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cls_flat = cls_map.permute(1, 2, 0).reshape(-1, num_classes)
    bbox_flat = bbox_map.permute(1, 2, 0).reshape(-1, 4)
    angle_flat = angle_map.permute(1, 2, 0).reshape(-1, 1)
    scores = torch.sigmoid(cls_flat)
    max_scores, fg_idx = scores.max(dim=1)
    n_loc = int(max_scores.shape[0])
    k = min(int(nms_pre), n_loc)
    topk_scores, topk_idx = max_scores.topk(k)
    labels = fg_idx[topk_idx] + 1
    decoded = bbox_coder.decode(
        pts[topk_idx],
        torch.cat([bbox_flat[topk_idx], angle_flat[topk_idx]], dim=-1),
    )
    valid = topk_scores >= float(score_threshold)
    valid = valid & (decoded[:, 2] > float(min_bbox_size)) & (
        decoded[:, 3] > float(min_bbox_size)
    )
    valid = valid & (decoded[:, 2] > 0) & (decoded[:, 3] > 0)
    valid = valid & torch.isfinite(decoded).all(dim=1) & torch.isfinite(topk_scores)
    return decoded, topk_scores, labels, valid


def rotated_rtmdet_decode_pre_nms(
    cls_per_level: Sequence[torch.Tensor],
    bbox_per_level: Sequence[torch.Tensor],
    angle_per_level: Sequence[torch.Tensor],
    points: Sequence[torch.Tensor],
    bbox_coder: DistanceAnglePointCoder,
    num_classes: int,
    score_threshold: float,
    nms_pre: int,
    min_bbox_size: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boxes_l: List[torch.Tensor] = []
    scores_l: List[torch.Tensor] = []
    labels_l: List[torch.Tensor] = []
    for cls_map, bbox_map, angle_map, pts in zip(
        cls_per_level, bbox_per_level, angle_per_level, points
    ):
        decoded, topk_scores, labels, valid = _rtmdet_level_topk_decode(
            cls_map,
            bbox_map,
            angle_map,
            pts,
            bbox_coder,
            num_classes,
            score_threshold,
            nms_pre,
            min_bbox_size,
        )
        boxes_l.append(decoded[valid])
        scores_l.append(topk_scores[valid])
        labels_l.append(labels[valid])
    device = cls_per_level[0].device
    if not boxes_l:
        return (
            torch.zeros((0, 5), dtype=torch.float32, device=device),
            torch.zeros((0,), dtype=torch.float32, device=device),
            torch.zeros((0,), dtype=torch.int64, device=device),
        )
    return (
        torch.cat(boxes_l, dim=0),
        torch.cat(scores_l, dim=0),
        torch.cat(labels_l, dim=0).to(dtype=torch.int64),
    )


def rotated_rtmdet_inference_pre_nms_padded(
    model: "RotatedRTMDet",
    images: torch.Tensor,
    max_candidates: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch-1 export path: padded pre-NMS boxes/scores/labels + count.

    Each FPN level always emits ``topk(min(nms_pre, H*W))`` rows (invalid rows
    get score 0 and ``w=h=0``) so ONNX keeps a fixed-length gather and does not
    need ``argsort`` / ``nonzero``. Final rotated NMS stays in Python.
    """
    if images.dim() != 4 or images.shape[0] != 1:
        raise ValueError(
            "rotated_rtmdet_inference_pre_nms_padded expects images [1, 3, H, W]."
        )

    device = images.device
    dtype = images.dtype
    feature_list = extract_backbone_features(
        model.backbone,
        [images[0]],
        use_checkpoint=False,
        training=False,
        include_pool_level=False,
    )
    feature_map_sizes = [(f.shape[2], f.shape[3]) for f in feature_list]
    fpn_strides_live = derive_fpn_strides_from_grid(
        (images.shape[2], images.shape[3]),
        feature_map_sizes,
        configured=model.fpn_strides,
    )
    cls_scores, bbox_preds, angle_preds = model.head(
        feature_list, strides=fpn_strides_live
    )
    points = generate_fpn_points(
        feature_map_sizes,
        fpn_strides_live,
        dtype=bbox_preds[0].dtype,
        device=device,
    )
    boxes_l: List[torch.Tensor] = []
    scores_l: List[torch.Tensor] = []
    labels_l: List[torch.Tensor] = []
    for cls_b, bbox_b, angle_b, pts in zip(cls_scores, bbox_preds, angle_preds, points):
        boxes_k, scores_k, labels_k, valid_k = _rtmdet_level_topk_decode(
            cls_b[0],
            bbox_b[0],
            angle_b[0],
            pts,
            model.bbox_coder,
            model.num_classes,
            model.score_threshold,
            model.nms_pre,
            model.min_bbox_size,
        )
        boxes_k = torch.where(valid_k.unsqueeze(-1), boxes_k, torch.zeros_like(boxes_k))
        scores_k = torch.where(valid_k, scores_k, torch.zeros_like(scores_k))
        labels_k = torch.where(valid_k, labels_k, torch.zeros_like(labels_k))
        boxes_l.append(boxes_k)
        scores_l.append(scores_k)
        labels_l.append(labels_k)

    raw_boxes = torch.cat(boxes_l, dim=0)
    raw_scores = torch.cat(scores_l, dim=0)
    raw_labels = torch.cat(labels_l, dim=0)
    pre_nms = PreNmsDetections(
        raw_boxes, raw_scores, raw_labels.to(dtype=torch.int64)
    )
    boxes, scores, labels, _ = pad_pre_nms_detections(
        pre_nms, max_candidates, device, dtype
    )
    # Live slots = all per-level top-k (invalids are score-0 / zero-size).
    # Avoid argsort/nonzero so the graph stays ONNX-exportable.
    live = min(int(raw_boxes.shape[0]), int(max_candidates))
    count = raw_scores.new_zeros((), dtype=torch.int64) + live
    return boxes, scores, labels, count


def _apply_rtmdet_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    iou_threshold: float,
    max_detections_per_image: int,
    final_nms_use_cpu: bool,
    class_agnostic: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        return boxes, scores, labels
    if class_agnostic:
        keep = rotated_nms(
            boxes, scores, iou_threshold=iou_threshold, force_cpu=final_nms_use_cpu
        )
        keep = keep[:max_detections_per_image]
        return boxes[keep], scores[keep], labels[keep]
    # Class-aware
    keep_all: List[torch.Tensor] = []
    for c in labels.unique():
        mask = labels == c
        idx = mask.nonzero(as_tuple=False).squeeze(1)
        k = rotated_nms(
            boxes[idx],
            scores[idx],
            iou_threshold=iou_threshold,
            force_cpu=final_nms_use_cpu,
        )
        keep_all.append(idx[k])
    if not keep_all:
        return boxes[:0], scores[:0], labels[:0]
    keep = torch.cat(keep_all)
    # Re-sort by score and cap
    order = scores[keep].argsort(descending=True)
    keep = keep[order][:max_detections_per_image]
    return boxes[keep], scores[keep], labels[keep]


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class RotatedRTMDet(SigmoidFocalClassWeightsMixin, nn.Module):
    """Rotated RTMDet detector (CSPNeXt + PAFPN, SepBN head)."""

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "cspnext_tiny",
        pretrained_backbone: bool = True,
        frozen_stages: int = -1,
        fpn_strides: Optional[List[float]] = None,
        stacked_convs: int = 2,
        norm_on_bbox: bool = True,
        scale_angle: bool = True,
        exp_on_reg: bool = True,
        qfl_beta: float = 2.0,
        box_reg_weight: float = 2.0,
        angle_weight: float = 0.2,
        assign_topk: int = 13,
        soft_center_radius: float = 3.0,
        assign_iou_weight: float = 3.0,
        score_threshold: float = 0.05,
        final_nms_iou_threshold: float = 0.1,
        max_detections_per_image: int = 2000,
        nms_pre: int = 2000,
        min_bbox_size: float = 0.0,
        final_nms_use_cpu: bool = False,
        nms_class_agnostic: bool = False,
        final_nms_iou_schedule_epochs: Optional[List[int]] = None,
        final_nms_iou_schedule_values: Optional[List[float]] = None,
        roi_class_weights: Optional[Union[Dict[str, float], torch.Tensor]] = None,
        **kwargs: Any,
    ):
        if nn is None:
            raise RuntimeError("PyTorch is required for RotatedRTMDet.")
        super().__init__()
        if kwargs:
            # Ignore unknown keys quietly (train.py may pass shared FCOS fields)
            pass
        self.num_classes = int(num_classes)
        self._init_sigmoid_focal_class_weights(roi_class_weights)
        self.fpn_strides = list(fpn_strides or [8, 16, 32])
        self.norm_on_bbox = bool(norm_on_bbox)
        self.qfl_beta = float(qfl_beta)
        self.box_reg_weight = float(box_reg_weight)
        self.angle_weight = float(angle_weight)
        self.assign_topk = int(assign_topk)
        self.soft_center_radius = float(soft_center_radius)
        self.assign_iou_weight = float(assign_iou_weight)
        self.score_threshold = float(score_threshold)
        self.final_nms_iou_threshold = float(final_nms_iou_threshold)
        self.max_detections_per_image = int(max_detections_per_image)
        self.nms_pre = int(nms_pre)
        self.min_bbox_size = float(min_bbox_size)
        self.final_nms_use_cpu = bool(final_nms_use_cpu)
        self.nms_class_agnostic = bool(nms_class_agnostic)
        self._final_nms_iou_schedule_epochs = final_nms_iou_schedule_epochs
        self._final_nms_iou_schedule_values = final_nms_iou_schedule_values

        self.backbone = build_cspnext_pafpn_backbone(
            backbone_name,
            pretrained=pretrained_backbone,
            frozen_stages=frozen_stages if frozen_stages is not None else -1,
        )
        in_channels = int(self.backbone.out_channels)
        self.head = RTMDetSepBNHead(
            in_channels=in_channels,
            num_classes=self.num_classes,
            stacked_convs=stacked_convs,
            strides=self.fpn_strides,
            norm_on_bbox=norm_on_bbox,
            scale_angle=scale_angle,
            exp_on_reg=exp_on_reg,
        )
        self.bbox_coder = DistanceAnglePointCoder(angle_version="le90")

    def set_final_nms_iou_for_epoch(self, epoch: int) -> None:
        if self._final_nms_iou_schedule_epochs is None or self._final_nms_iou_schedule_values is None:
            return
        if not self._final_nms_iou_schedule_epochs or not self._final_nms_iou_schedule_values:
            return
        idx = 0
        for boundary in self._final_nms_iou_schedule_epochs:
            if epoch < boundary:
                break
            idx += 1
        idx = min(idx, len(self._final_nms_iou_schedule_values) - 1)
        self.final_nms_iou_threshold = self._final_nms_iou_schedule_values[idx]

    def forward(
        self,
        images: Sequence[torch.Tensor],
        targets: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
        if not isinstance(images, (list, tuple)):
            images = [images]
        image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        feature_list = extract_backbone_features(
            self.backbone,
            images,
            use_checkpoint=False,
            training=self.training,
            include_pool_level=False,
        )
        feature_map_sizes = [(f.shape[2], f.shape[3]) for f in feature_list]
        fpn_strides_live = derive_fpn_strides_from_grid(
            image_sizes[0], feature_map_sizes, configured=self.fpn_strides
        )
        warn_if_fpn_strides_mismatch(self.fpn_strides, fpn_strides_live)

        device = images[0].device
        cls_scores, bbox_preds, angle_preds = self.head(
            feature_list, strides=fpn_strides_live
        )
        points = generate_fpn_points(
            feature_map_sizes,
            fpn_strides_live,
            dtype=bbox_preds[0].dtype,
            device=device,
        )

        if self.training:
            if targets is None:
                raise ValueError("Targets required during training.")
            gt_boxes_list, gt_labels_list, _, _ = prepare_targets(targets, device=device)
            return compute_rotated_rtmdet_loss(
                cls_scores=cls_scores,
                bbox_preds=bbox_preds,
                angle_preds=angle_preds,
                points=points,
                strides=fpn_strides_live,
                gt_boxes=gt_boxes_list,
                gt_labels=gt_labels_list,
                num_classes=self.num_classes,
                bbox_coder=self.bbox_coder,
                norm_on_bbox=self.norm_on_bbox,
                qfl_beta=self.qfl_beta,
                box_reg_weight=self.box_reg_weight,
                angle_weight=self.angle_weight,
                assign_topk=self.assign_topk,
                soft_center_radius=self.soft_center_radius,
                iou_weight=self.assign_iou_weight,
                class_weights=self.roi_class_weights_tensor,
            )

        return self._inference(cls_scores, bbox_preds, angle_preds, points, device)

    def _inference(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        angle_preds: List[torch.Tensor],
        points: List[torch.Tensor],
        device: torch.device,
    ) -> List[Dict[str, Any]]:
        num_imgs = cls_scores[0].size(0)
        outputs: List[Dict[str, Any]] = []
        for img_idx in range(num_imgs):
            boxes, scores, labels = rotated_rtmdet_decode_pre_nms(
                [c[img_idx] for c in cls_scores],
                [b[img_idx] for b in bbox_preds],
                [a[img_idx] for a in angle_preds],
                points,
                self.bbox_coder,
                self.num_classes,
                self.score_threshold,
                self.nms_pre,
                self.min_bbox_size,
            )
            if boxes.numel() == 0:
                outputs.append(
                    {
                        "rboxes": [],
                        "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                        "scores": torch.zeros((0,), dtype=torch.float32, device=device),
                    }
                )
                continue
            boxes, scores, labels = _apply_rtmdet_nms(
                boxes,
                scores,
                labels,
                iou_threshold=self.final_nms_iou_threshold,
                max_detections_per_image=self.max_detections_per_image,
                final_nms_use_cpu=self.final_nms_use_cpu,
                class_agnostic=self.nms_class_agnostic,
            )
            outputs.append(
                {
                    "rboxes": tensor_to_rboxes(boxes),
                    "labels": labels.to(dtype=torch.int64),
                    "scores": scores,
                }
            )
        return outputs


def build_rotated_rtmdet(
    model_cfg: Any,
    num_classes: int,
    *,
    roi_class_weights: Optional[Union[Dict[str, float], "torch.Tensor"]] = None,
) -> "RotatedRTMDet":
    """Construct RotatedRTMDet from a ``ModelConfig`` (or any object with the same fields)."""
    frozen = getattr(model_cfg, "frozen_stages", None)
    return RotatedRTMDet(
        num_classes=int(num_classes),
        backbone_name=getattr(model_cfg, "backbone", "cspnext_tiny"),
        pretrained_backbone=bool(getattr(model_cfg, "pretrained_backbone", True)),
        frozen_stages=-1 if frozen is None else int(frozen),
        fpn_strides=getattr(model_cfg, "fpn_strides", None),
        stacked_convs=int(getattr(model_cfg, "rtmdet_stacked_convs", 2)),
        norm_on_bbox=bool(getattr(model_cfg, "rtmdet_norm_on_bbox", True)),
        scale_angle=bool(getattr(model_cfg, "rtmdet_scale_angle", True)),
        exp_on_reg=bool(getattr(model_cfg, "rtmdet_exp_on_reg", True)),
        qfl_beta=float(getattr(model_cfg, "rtmdet_qfl_beta", 2.0)),
        box_reg_weight=float(getattr(model_cfg, "box_reg_weight", 2.0)),
        angle_weight=float(getattr(model_cfg, "rtmdet_angle_weight", 0.2)),
        assign_topk=int(getattr(model_cfg, "rtmdet_assign_topk", 13)),
        soft_center_radius=float(getattr(model_cfg, "rtmdet_soft_center_radius", 3.0)),
        assign_iou_weight=float(getattr(model_cfg, "rtmdet_assign_iou_weight", 3.0)),
        score_threshold=float(getattr(model_cfg, "inference_pre_nms_score_threshold", 0.05)),
        final_nms_iou_threshold=float(getattr(model_cfg, "final_nms_iou_threshold", 0.1)),
        max_detections_per_image=int(getattr(model_cfg, "max_detections_per_image", 2000)),
        nms_pre=int(getattr(model_cfg, "rtmdet_nms_pre", 2000)),
        final_nms_use_cpu=bool(getattr(model_cfg, "final_nms_use_cpu", False)),
        nms_class_agnostic=bool(getattr(model_cfg, "nms_class_agnostic", False)),
        final_nms_iou_schedule_epochs=getattr(model_cfg, "final_nms_iou_schedule_epochs", None),
        final_nms_iou_schedule_values=getattr(model_cfg, "final_nms_iou_schedule_values", None),
        roi_class_weights=roi_class_weights,
    )


__all__ = [
    "RotatedRTMDet",
    "build_rotated_rtmdet",
    "RTMDetSepBNHead",
    "quality_focal_loss",
    "dynamic_soft_label_assign_single",
    "compute_rotated_rtmdet_loss",
    "rotated_rtmdet_decode_pre_nms",
    "rotated_rtmdet_inference_pre_nms_padded",
]

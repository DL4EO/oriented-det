"""YOLO-OBB: C2f backbone, DFL box head, task-aligned assigner (clean-room).

Boxes stay le90 ``(cx, cy, w, h, θ)``. Losses: BCE on soft targets, ProbIoU,
distribution focal loss on the four side bins, and a small le90 angle L1.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

from ..ops.diff_iou_rotated import diff_iou_rotated_2d
from ..ops.probiou import probiou_loss_per_box
from ..ops.rotated_ops import rotated_nms
from .backbones.yolo_obb import Conv, build_yolo_obb_backbone
from .bbox_coder import DistanceAnglePointCoder
from .oriented_rpn import norm_angle_le90
from .rotated_fcos import Scale, generate_fpn_points
from .utils import (
    prepare_targets,
    extract_backbone_features,
    derive_fpn_strides_from_grid,
    warn_if_fpn_strides_mismatch,
    tensor_to_rboxes,
    SigmoidFocalClassWeightsMixin,
)


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


def dfl_expectation(pred: torch.Tensor, reg_max: int) -> torch.Tensor:
    """Softmax over bins → expected distance. ``pred`` is ``[N, 4 * reg_max]`` → ``[N, 4]``."""
    bins = pred.view(-1, 4, int(reg_max))
    prob = bins.softmax(dim=-1)
    proj = torch.arange(int(reg_max), device=pred.device, dtype=pred.dtype)
    return (prob * proj).sum(dim=-1)


def distribution_focal_loss(pred: torch.Tensor, target: torch.Tensor, reg_max: int) -> torch.Tensor:
    """Per-box DFL. ``pred`` ``[N, 4, reg_max]`` logits, ``target`` ``[N, 4]`` in stride units.

    Returns ``[N]`` (mean over the four sides).
    """
    n = pred.size(0)
    logits = pred.reshape(-1, int(reg_max))
    tgt = target.reshape(-1).clamp(min=0.0, max=float(reg_max) - 1.001)
    left = tgt.floor().long()
    right = (left + 1).clamp(max=int(reg_max) - 1)
    left = left.clamp(max=int(reg_max) - 1)
    weight_right = tgt - left.to(dtype=tgt.dtype)
    weight_left = 1.0 - weight_right
    loss_left = F.cross_entropy(logits, left, reduction="none")
    loss_right = F.cross_entropy(logits, right, reduction="none")
    per_side = loss_left * weight_left + loss_right * weight_right
    return per_side.view(n, 4).mean(dim=1)


def task_aligned_assign_single(
    points: torch.Tensor,
    cls_logits: torch.Tensor,
    decoded_boxes: torch.Tensor,
    gt_bboxes: torch.Tensor,
    gt_labels: torch.Tensor,
    num_classes: int,
    *,
    topk: int = 13,
    alpha: float = 0.5,
    beta: float = 6.0,
    reg_max: int = 16,
    strides: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Task-aligned assignment for one image.

    Returns:
        labels ``[P]`` in ``{0..K-1}`` foreground or ``K`` background,
        soft_cls_target ``[P, K]``,
        bbox_targets ``[P, 5]`` encoded ``(l, t, r, b, θ)`` in pixels,
        pos_mask ``[P]``
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

    inside = _points_in_obb(points, gt_bboxes)
    if not inside.any():
        return labels, soft_target, bbox_targets, pos_mask

    with torch.no_grad():
        point_idx, gt_idx = inside.nonzero(as_tuple=True)
        pair_iou = diff_iou_rotated_2d(
            decoded_boxes.detach()[point_idx],
            gt_bboxes.detach()[gt_idx],
        ).clamp(0, 1)
        # diff_iou returns fp32 even under AMP. points follow the head dtype (fp16).
        iou = torch.zeros((num_points, num_gts), device=device, dtype=torch.float32)
        iou[point_idx, gt_idx] = pair_iou

        pred_sig = cls_logits.detach().sigmoid().clamp(0, 1)
        cls_index = (gt_labels.to(dtype=torch.long) - 1).clamp(0, num_classes - 1)
        valid_cls = (gt_labels >= 1) & (gt_labels <= num_classes)
        scores = pred_sig[point_idx, cls_index[gt_idx]]
        pair_align = scores.pow(float(alpha)) * pair_iou.pow(float(beta))
        pair_align = torch.where(valid_cls[gt_idx], pair_align, torch.zeros_like(pair_align))
        align = torch.zeros((num_points, num_gts), device=device, dtype=torch.float32)
        align[point_idx, gt_idx] = pair_align.to(dtype=torch.float32)

        k = min(int(topk), num_points)
        top_align, top_idx = torch.topk(align, k, dim=0, largest=True)
        selected = torch.zeros_like(inside)
        selected.scatter_(0, top_idx, top_align > 0)
        iou_selected = iou.masked_fill(~selected, -1.0)
        best_iou, best_gt = iou_selected.max(dim=1)
        pos = best_iou > 0
        if not pos.any():
            return labels, soft_target, bbox_targets, pos_mask

        pos_inds = pos.nonzero(as_tuple=False).squeeze(1)
        gt_inds = best_gt[pos_inds]
        fg = (gt_labels[gt_inds] - 1).clamp(min=0, max=num_classes - 1)
        labels[pos_inds] = fg
        pos_mask[pos_inds] = True
        pos_align = align[pos_inds, gt_inds]
        norm = pos_align / pos_align.max().clamp(min=1e-6)
        soft_target[pos_inds, fg] = norm.to(soft_target.dtype)

        coder = DistanceAnglePointCoder(angle_version="le90")
        encoded = coder.encode(points[pos_inds], gt_bboxes[gt_inds])
        bbox_targets[pos_inds] = encoded.to(dtype=bbox_targets.dtype)
        _ = strides
        _ = reg_max

    return labels, soft_target, bbox_targets, pos_mask


class YoloOBBHead(nn.Module):
    """Per-level decoupled head: cls, DFL box bins, le90 angle."""

    def __init__(
        self,
        in_channels: Sequence[int],
        num_classes: int,
        *,
        reg_max: int = 16,
        strides: Sequence[float] = (8, 16, 32),
    ):
        if nn is None:
            raise RuntimeError("PyTorch is required for YoloOBBHead.")
        super().__init__()
        self.num_classes = int(num_classes)
        self.reg_max = int(reg_max)
        self.strides = [float(s) for s in strides]
        channels = [int(c) for c in in_channels]
        self.num_levels = len(channels)
        c0 = channels[0]
        reg_ch = max(16, c0 // 4, self.reg_max * 4)
        cls_ch = max(c0, min(self.num_classes, 100))
        ang_ch = max(c0 // 4, 16)

        def _tower(in_ch: int, hidden: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                Conv(in_ch, hidden, 3, 1),
                Conv(hidden, hidden, 3, 1),
                nn.Conv2d(hidden, out_ch, 1),
            )

        self.cls_heads = nn.ModuleList(_tower(c, cls_ch, self.num_classes) for c in channels)
        self.reg_heads = nn.ModuleList(_tower(c, reg_ch, 4 * self.reg_max) for c in channels)
        self.angle_heads = nn.ModuleList(_tower(c, ang_ch, 1) for c in channels)
        self.angle_scales = nn.ModuleList(Scale(1.0) for _ in channels)
        prior = math.log(0.01 / 0.99)
        for head in self.cls_heads:
            last = head[-1]
            nn.init.constant_(last.bias, prior)
            nn.init.normal_(last.weight, std=0.01)
        for head in list(self.reg_heads) + list(self.angle_heads):
            last = head[-1]
            nn.init.normal_(last.weight, std=0.01)
            nn.init.zeros_(last.bias)

    def forward(
        self, features: Sequence[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        cls_scores: List[torch.Tensor] = []
        bbox_preds: List[torch.Tensor] = []
        angle_preds: List[torch.Tensor] = []
        half_pi = math.pi / 2.0
        for level, feat in enumerate(features):
            cls_scores.append(self.cls_heads[level](feat))
            bbox_preds.append(self.reg_heads[level](feat))
            raw = self.angle_scales[level](self.angle_heads[level](feat))
            angle_preds.append(norm_angle_le90(torch.tanh(raw) * half_pi))
        return cls_scores, bbox_preds, angle_preds


def _decode_distances(
    bbox_bins: torch.Tensor,
    angle: torch.Tensor,
    points: torch.Tensor,
    strides: torch.Tensor,
    reg_max: int,
    bbox_coder: DistanceAnglePointCoder,
) -> torch.Tensor:
    """Bin logits ``[N, 4*reg_max]`` + angle ``[N, 1]`` → le90 boxes ``[N, 5]``."""
    dist = dfl_expectation(bbox_bins, reg_max) * strides.unsqueeze(-1)
    return bbox_coder.decode(points, torch.cat([dist, angle], dim=-1))


def compute_yolo_obb_loss(
    cls_scores: Sequence[torch.Tensor],
    bbox_preds: Sequence[torch.Tensor],
    angle_preds: Sequence[torch.Tensor],
    points: Sequence[torch.Tensor],
    strides: Sequence[float],
    gt_boxes: Sequence[torch.Tensor],
    gt_labels: Sequence[torch.Tensor],
    num_classes: int,
    *,
    reg_max: int,
    bbox_coder: DistanceAnglePointCoder,
    box_reg_weight: float,
    dfl_weight: float,
    angle_weight: float,
    assign_topk: int,
    tal_alpha: float,
    tal_beta: float,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """BCE + ProbIoU + DFL + le90 angle L1, divided by the positive count."""
    num_levels = len(points)
    num_imgs = cls_scores[0].size(0)
    num_points_per_lvl = [p.size(0) for p in points]
    concat_points = torch.cat(points, dim=0)
    loss_cls_sum = cls_scores[0].new_tensor(0.0)
    loss_box_sum = cls_scores[0].new_tensor(0.0)
    loss_dfl_sum = cls_scores[0].new_tensor(0.0)
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
        bins_i = torch.cat(
            [
                bbox_preds[lvl][img_idx].permute(1, 2, 0).reshape(-1, 4 * reg_max)
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
        decoded = _decode_distances(bins_i, angle_i, concat_points, stride_t, reg_max, bbox_coder)
        _labels, soft_tgt, bbox_tgt, pos_mask = task_aligned_assign_single(
            concat_points,
            cls_i,
            decoded,
            gt_boxes[img_idx],
            gt_labels[img_idx],
            num_classes,
            topk=assign_topk,
            alpha=tal_alpha,
            beta=tal_beta,
            reg_max=reg_max,
            strides=stride_t,
        )
        cls_loss = F.binary_cross_entropy_with_logits(cls_i, soft_tgt, reduction="none")
        if class_weights is not None:
            cls_loss = cls_loss * class_weights.view(1, -1).to(device=cls_loss.device, dtype=cls_loss.dtype)
        loss_cls_sum = loss_cls_sum + cls_loss.sum()

        num_pos = int(pos_mask.sum().item())
        total_pos += num_pos
        if num_pos == 0:
            loss_box_sum = loss_box_sum + bins_i.sum() * 0.0
            loss_dfl_sum = loss_dfl_sum + bins_i.sum() * 0.0
            loss_angle_sum = loss_angle_sum + angle_i.sum() * 0.0
            continue

        pos_bins = bins_i[pos_mask]
        pos_angle = angle_i[pos_mask]
        pos_pts = concat_points[pos_mask]
        pos_stride = stride_t[pos_mask]
        pos_tgt = bbox_tgt[pos_mask]
        tgt_ltrb = pos_tgt[:, :4]
        tgt_angle = pos_tgt[:, 4:5]
        pred_boxes = _decode_distances(pos_bins, pos_angle, pos_pts, pos_stride, reg_max, bbox_coder)
        tgt_boxes = bbox_coder.decode(pos_pts, pos_tgt).detach()
        loss_box_sum = loss_box_sum + probiou_loss_per_box(pred_boxes, tgt_boxes).sum()

        tgt_bins = (tgt_ltrb / pos_stride.unsqueeze(-1)).clamp(0, float(reg_max) - 1.001)
        loss_dfl_sum = loss_dfl_sum + distribution_focal_loss(
            pos_bins.view(-1, 4, reg_max), tgt_bins, reg_max
        ).sum()
        ang_err = norm_angle_le90(pos_angle.squeeze(-1) - tgt_angle.squeeze(-1)).abs()
        loss_angle_sum = loss_angle_sum + ang_err.sum()

    denom = max(float(total_pos), 1.0)
    return {
        "loss_classifier": loss_cls_sum / denom,
        "loss_box_reg": float(box_reg_weight) * loss_box_sum / denom,
        "loss_dfl": float(dfl_weight) * loss_dfl_sum / denom,
        "loss_angle": float(angle_weight) * loss_angle_sum / denom,
    }


def _level_topk_decode(
    cls_map: torch.Tensor,
    bbox_map: torch.Tensor,
    angle_map: torch.Tensor,
    pts: torch.Tensor,
    stride: float,
    bbox_coder: DistanceAnglePointCoder,
    num_classes: int,
    reg_max: int,
    score_threshold: float,
    nms_pre: int,
    min_bbox_size: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cls_flat = cls_map.permute(1, 2, 0).reshape(-1, num_classes)
    bins_flat = bbox_map.permute(1, 2, 0).reshape(-1, 4 * reg_max)
    angle_flat = angle_map.permute(1, 2, 0).reshape(-1, 1)
    scores = torch.sigmoid(cls_flat)
    max_scores, fg_idx = scores.max(dim=1)
    k = min(int(nms_pre), int(max_scores.shape[0]))
    topk_scores, topk_idx = max_scores.topk(k)
    strides = pts.new_full((k,), float(stride))
    decoded = _decode_distances(
        bins_flat[topk_idx],
        angle_flat[topk_idx],
        pts[topk_idx],
        strides,
        reg_max,
        bbox_coder,
    )
    labels = fg_idx[topk_idx] + 1
    valid = topk_scores >= float(score_threshold)
    valid = valid & (decoded[:, 2] > float(min_bbox_size)) & (decoded[:, 3] > float(min_bbox_size))
    valid = valid & (decoded[:, 2] > 0) & (decoded[:, 3] > 0)
    valid = valid & torch.isfinite(decoded).all(dim=1) & torch.isfinite(topk_scores)
    return decoded[valid], topk_scores[valid], labels[valid]


def yolo_obb_decode_pre_nms(
    cls_per_level: Sequence[torch.Tensor],
    bbox_per_level: Sequence[torch.Tensor],
    angle_per_level: Sequence[torch.Tensor],
    points: Sequence[torch.Tensor],
    strides: Sequence[float],
    bbox_coder: DistanceAnglePointCoder,
    num_classes: int,
    reg_max: int,
    score_threshold: float,
    nms_pre: int,
    min_bbox_size: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boxes_l: List[torch.Tensor] = []
    scores_l: List[torch.Tensor] = []
    labels_l: List[torch.Tensor] = []
    for cls_map, bbox_map, angle_map, pts, stride in zip(
        cls_per_level, bbox_per_level, angle_per_level, points, strides
    ):
        boxes, scores, labels = _level_topk_decode(
            cls_map,
            bbox_map,
            angle_map,
            pts,
            float(stride),
            bbox_coder,
            num_classes,
            reg_max,
            score_threshold,
            nms_pre,
            min_bbox_size,
        )
        boxes_l.append(boxes)
        scores_l.append(scores)
        labels_l.append(labels)
    device = cls_per_level[0].device
    if not boxes_l or all(b.numel() == 0 for b in boxes_l):
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


def _apply_yolo_nms(
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
        keep = rotated_nms(boxes, scores, iou_threshold=iou_threshold, force_cpu=final_nms_use_cpu)
        keep = keep[:max_detections_per_image]
        return boxes[keep], scores[keep], labels[keep]
    keep_all: List[torch.Tensor] = []
    for c in labels.unique():
        idx = (labels == c).nonzero(as_tuple=False).squeeze(1)
        k = rotated_nms(boxes[idx], scores[idx], iou_threshold=iou_threshold, force_cpu=final_nms_use_cpu)
        keep_all.append(idx[k])
    if not keep_all:
        return boxes[:0], scores[:0], labels[:0]
    keep = torch.cat(keep_all)
    order = scores[keep].argsort(descending=True)
    keep = keep[order][:max_detections_per_image]
    return boxes[keep], scores[keep], labels[keep]


class YoloOBB(SigmoidFocalClassWeightsMixin, nn.Module):
    """YOLO-OBB detector (``model_type: yolo_obb``)."""

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "yolo_obb_s",
        pretrained_backbone: bool = False,
        frozen_stages: int = -1,
        fpn_strides: Optional[List[float]] = None,
        reg_max: int = 16,
        box_reg_weight: float = 7.5,
        dfl_weight: float = 1.5,
        angle_weight: float = 0.2,
        assign_topk: int = 13,
        tal_alpha: float = 0.5,
        tal_beta: float = 6.0,
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
        force_horizontal: bool = False,
        **kwargs: Any,
    ):
        if nn is None:
            raise RuntimeError("PyTorch is required for YoloOBB.")
        super().__init__()
        if kwargs:
            pass
        self.num_classes = int(num_classes)
        self._init_sigmoid_focal_class_weights(roi_class_weights)
        self.fpn_strides = list(fpn_strides or [8, 16, 32])
        self.reg_max = int(reg_max)
        self.box_reg_weight = float(box_reg_weight)
        self.dfl_weight = float(dfl_weight)
        self.angle_weight = float(angle_weight)
        self.force_horizontal = bool(force_horizontal)
        self.assign_topk = int(assign_topk)
        self.tal_alpha = float(tal_alpha)
        self.tal_beta = float(tal_beta)
        self.score_threshold = float(score_threshold)
        self.final_nms_iou_threshold = float(final_nms_iou_threshold)
        self.max_detections_per_image = int(max_detections_per_image)
        self.nms_pre = int(nms_pre)
        self.min_bbox_size = float(min_bbox_size)
        self.final_nms_use_cpu = bool(final_nms_use_cpu)
        self.nms_class_agnostic = bool(nms_class_agnostic)
        self._final_nms_iou_schedule_epochs = final_nms_iou_schedule_epochs
        self._final_nms_iou_schedule_values = final_nms_iou_schedule_values

        self.backbone = build_yolo_obb_backbone(
            backbone_name,
            pretrained=pretrained_backbone,
            frozen_stages=frozen_stages if frozen_stages is not None else -1,
        )
        self.head = YoloOBBHead(
            self.backbone.out_channels_list,
            num_classes=self.num_classes,
            reg_max=self.reg_max,
            strides=self.fpn_strides,
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
        cls_scores, bbox_preds, angle_preds = self.head(feature_list)
        if self.force_horizontal:
            # Constant 0 is outside the graph. ProbIoU and DFL then cannot
            # train the angle head; angle_weight 0 alone does not do that.
            angle_preds = [pred.new_zeros(pred.shape) for pred in angle_preds]
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
            return compute_yolo_obb_loss(
                cls_scores,
                bbox_preds,
                angle_preds,
                points,
                fpn_strides_live,
                gt_boxes_list,
                gt_labels_list,
                self.num_classes,
                reg_max=self.reg_max,
                bbox_coder=self.bbox_coder,
                box_reg_weight=self.box_reg_weight,
                dfl_weight=self.dfl_weight,
                angle_weight=self.angle_weight,
                assign_topk=self.assign_topk,
                tal_alpha=self.tal_alpha,
                tal_beta=self.tal_beta,
                class_weights=self.roi_class_weights_tensor,
            )
        return self._inference(cls_scores, bbox_preds, angle_preds, points, fpn_strides_live, device)

    def _inference(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        angle_preds: List[torch.Tensor],
        points: List[torch.Tensor],
        strides: Sequence[float],
        device: torch.device,
    ) -> List[Dict[str, Any]]:
        num_imgs = cls_scores[0].size(0)
        outputs: List[Dict[str, Any]] = []
        for img_idx in range(num_imgs):
            boxes, scores, labels = yolo_obb_decode_pre_nms(
                [c[img_idx] for c in cls_scores],
                [b[img_idx] for b in bbox_preds],
                [a[img_idx] for a in angle_preds],
                points,
                strides,
                self.bbox_coder,
                self.num_classes,
                self.reg_max,
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
            boxes, scores, labels = _apply_yolo_nms(
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


def build_yolo_obb(
    model_cfg: Any,
    num_classes: int,
    *,
    roi_class_weights: Optional[Union[Dict[str, float], "torch.Tensor"]] = None,
) -> "YoloOBB":
    """Construct YoloOBB from a ``ModelConfig`` (or any object with the same fields)."""
    frozen = getattr(model_cfg, "frozen_stages", None)
    return YoloOBB(
        num_classes=int(num_classes),
        backbone_name=getattr(model_cfg, "backbone", "yolo_obb_s"),
        pretrained_backbone=bool(getattr(model_cfg, "pretrained_backbone", False)),
        frozen_stages=-1 if frozen is None else int(frozen),
        fpn_strides=getattr(model_cfg, "fpn_strides", None),
        reg_max=int(getattr(model_cfg, "yolo_obb_reg_max", 16)),
        box_reg_weight=float(getattr(model_cfg, "box_reg_weight", 7.5)),
        dfl_weight=float(getattr(model_cfg, "yolo_obb_dfl_weight", 1.5)),
        angle_weight=float(getattr(model_cfg, "yolo_obb_angle_weight", 0.2)),
        force_horizontal=bool(getattr(model_cfg, "yolo_obb_force_horizontal", False)),
        assign_topk=int(getattr(model_cfg, "yolo_obb_tal_topk", 13)),
        tal_alpha=float(getattr(model_cfg, "yolo_obb_tal_alpha", 0.5)),
        tal_beta=float(getattr(model_cfg, "yolo_obb_tal_beta", 6.0)),
        score_threshold=float(getattr(model_cfg, "inference_pre_nms_score_threshold", 0.05)),
        final_nms_iou_threshold=float(getattr(model_cfg, "final_nms_iou_threshold", 0.1)),
        max_detections_per_image=int(getattr(model_cfg, "max_detections_per_image", 2000)),
        nms_pre=int(getattr(model_cfg, "yolo_obb_nms_pre", 2000)),
        final_nms_use_cpu=bool(getattr(model_cfg, "final_nms_use_cpu", False)),
        nms_class_agnostic=bool(getattr(model_cfg, "nms_class_agnostic", False)),
        final_nms_iou_schedule_epochs=getattr(model_cfg, "final_nms_iou_schedule_epochs", None),
        final_nms_iou_schedule_values=getattr(model_cfg, "final_nms_iou_schedule_values", None),
        roi_class_weights=roi_class_weights,
    )


def _bare_state_key(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


def _is_last_cls_conv(bare: str) -> bool:
    """``head.cls_heads.<level>.2.weight|bias`` — the 1×1 classifier."""
    parts = bare.split(".")
    return (
        len(parts) == 5
        and parts[0] == "head"
        and parts[1] == "cls_heads"
        and parts[3] == "2"
        and parts[4] in {"weight", "bias"}
    )


def _cls_out_channels(state: Dict[str, Any]) -> Optional[int]:
    for key, value in state.items():
        if not _is_last_cls_conv(_bare_state_key(key)):
            continue
        if not _bare_state_key(key).endswith(".weight"):
            continue
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
    return None


def drop_yolo_obb_class_transfer_keys(
    model_state: Dict[str, Any],
    checkpoint_state: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """Drop the angle head and the last classifier when class counts differ.

    A same-shape checkpoint (DOTA resume, or COCO resume) is returned unchanged.
    Dropped layers stay at the model's initialization.
    """
    model_out = _cls_out_channels(model_state)
    ckpt_out = _cls_out_channels(checkpoint_state)
    if model_out is None or ckpt_out is None or model_out == ckpt_out:
        return dict(checkpoint_state), []
    kept: Dict[str, Any] = {}
    skipped: List[str] = []
    for key, value in checkpoint_state.items():
        bare = _bare_state_key(key)
        if bare.startswith("head.angle_heads.") or bare.startswith("head.angle_scales."):
            skipped.append(key)
            continue
        if _is_last_cls_conv(bare):
            skipped.append(key)
            continue
        kept[key] = value
    return kept, skipped


__all__ = [
    "YoloOBB",
    "YoloOBBHead",
    "build_yolo_obb",
    "dfl_expectation",
    "distribution_focal_loss",
    "drop_yolo_obb_class_transfer_keys",
    "task_aligned_assign_single",
    "compute_yolo_obb_loss",
]

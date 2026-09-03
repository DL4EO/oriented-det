"""Rotated FCOS: anchor-free single-stage oriented detector (MMRotate-style).

Uses DistanceAnglePointCoder (left, top, right, bottom, angle), center-in-OBB
assignment with per-level regress ranges, centerness, and L1 / KFIoU / decoded
rIoU box regression.
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

from ..ops.gaussian_angle import aspect_gated_angle_loss_per_box
from ..ops.diff_iou_rotated import riou_loss_per_box
from ..ops.kfiou import kfiou_loss_per_box
from ..ops.probiou import probiou_loss_per_box
from ..ops.rotated_ops import rotated_nms
from .bbox_coder import DistanceAnglePointCoder
from .faster_rcnn_inference import PreNmsDetections, pad_pre_nms_detections
from .oriented_rpn import norm_angle_le90
from .rotated_retinanet import sigmoid_focal_loss_sum
from .utils import (
    prepare_targets,
    setup_backbone,
    extract_backbone_features,
    derive_fpn_strides_from_grid,
    warn_if_fpn_strides_mismatch,
    tensor_to_rboxes,
    SigmoidFocalClassWeightsMixin,
)

INF = 1e8

DEFAULT_REGRESS_RANGES: Tuple[Tuple[float, float], ...] = (
    (-1.0, 64.0),
    (64.0, 128.0),
    (128.0, 256.0),
    (256.0, 512.0),
    (512.0, INF),
)


class Scale(nn.Module):
    """Learnable scalar multiplier (mmcv.cnn.Scale equivalent)."""

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def generate_fpn_points(
    feature_map_sizes: Sequence[Tuple[int, int]],
    strides: Sequence[float],
    dtype: torch.dtype,
    device: torch.device,
) -> List[torch.Tensor]:
    """Grid priors at pixel centers: (stride/2 + i*stride, stride/2 + j*stride).

    Returns one [H*W, 2] tensor per FPN level (x, y) in image coordinates.
    """
    points_per_level: List[torch.Tensor] = []
    for (h, w), stride in zip(feature_map_sizes, strides):
        shift_x = (torch.arange(w, device=device, dtype=dtype) + 0.5) * stride
        shift_y = (torch.arange(h, device=device, dtype=dtype) + 0.5) * stride
        # meshgrid indexing='xy' → xx varies along columns
        yy, xx = torch.meshgrid(shift_y, shift_x, indexing="ij")
        points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        points_per_level.append(points)
    return points_per_level


def centerness_target(pos_bbox_targets: torch.Tensor) -> torch.Tensor:
    """Centerness from (l, t, r, b): sqrt((min/max)_lr * (min/max)_tb)."""
    left_right = pos_bbox_targets[:, [0, 2]]
    top_bottom = pos_bbox_targets[:, [1, 3]]
    if pos_bbox_targets.numel() == 0:
        return pos_bbox_targets.new_zeros((0,))
    ctr = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0].clamp(min=1e-6)) * (
        top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0].clamp(min=1e-6)
    )
    return torch.sqrt(ctr.clamp(min=0.0))


def assign_fcos_targets_single(
    points: torch.Tensor,
    gt_bboxes: torch.Tensor,
    gt_labels: torch.Tensor,
    regress_ranges: torch.Tensor,
    num_points_per_lvl: Sequence[int],
    strides: Sequence[float],
    num_classes: int,
    center_sampling: bool = True,
    center_sample_radius: float = 1.5,
    gt_bboxes_ignore: Optional[torch.Tensor] = None,
    gt_bboxes_lookalike: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign labels / bbox / angle targets for one image (all FPN points).

    Args:
        points: [P, 2] concatenated FPN points
        gt_bboxes: [G, 5] (cx, cy, w, h, angle)
        gt_labels: [G] **1-indexed** class ids
        regress_ranges: [P, 2] per-point (min, max) regress range
        num_points_per_lvl: points per FPN level
        strides: stride per level
        num_classes: foreground class count K; background label = K (0-indexed scheme)
        gt_bboxes_ignore: optional [I, 5]
        gt_bboxes_lookalike: optional [L, 5] hard-negative lookalikes; non-fg points inside
            stay background ``K`` (not ignore ``-1``).

    Returns:
        labels [P] in {0..K-1 fg, K bg, -1 ignore},
        bbox_targets [P, 4] (l,t,r,b) absolute (not stride-normalized),
        angle_targets [P, 1],
        pos_mask [P] bool
    """
    device = points.device
    num_points = points.size(0)
    num_gts = gt_bboxes.size(0)
    bg = num_classes

    if num_gts == 0:
        labels = points.new_full((num_points,), bg, dtype=torch.long)
        bbox_targets = points.new_zeros((num_points, 4))
        angle_targets = points.new_zeros((num_points, 1))
        if gt_bboxes_ignore is not None and gt_bboxes_ignore.numel() > 0:
            ignore_mask = _points_inside_any_obb(points, gt_bboxes_ignore)
            labels[ignore_mask] = -1
        if gt_bboxes_lookalike is not None and gt_bboxes_lookalike.numel() > 0:
            look_mask = _points_inside_any_obb(points, gt_bboxes_lookalike)
            # Lookalike wins over ignore: force background (trainable negative).
            labels = torch.where(look_mask, torch.full_like(labels, bg), labels)
        return labels, bbox_targets, angle_targets, torch.zeros(num_points, dtype=torch.bool, device=device)

    areas = (gt_bboxes[:, 2] * gt_bboxes[:, 3]).unsqueeze(0).repeat(num_points, 1)
    rr = regress_ranges[:, None, :].expand(num_points, num_gts, 2)
    pts = points[:, None, :].expand(num_points, num_gts, 2)
    gts = gt_bboxes[None].expand(num_points, num_gts, 5)

    gt_ctr, gt_wh, gt_angle = torch.split(gts, [2, 2, 1], dim=2)
    cos_a, sin_a = torch.cos(gt_angle), torch.sin(gt_angle)
    rot = torch.cat([cos_a, sin_a, -sin_a, cos_a], dim=-1).reshape(num_points, num_gts, 2, 2)
    offset = torch.matmul(rot, (pts - gt_ctr).unsqueeze(-1)).squeeze(-1)
    w, h = gt_wh[..., 0], gt_wh[..., 1]
    ox, oy = offset[..., 0], offset[..., 1]
    left = w / 2 + ox
    right = w / 2 - ox
    top = h / 2 + oy
    bottom = h / 2 - oy
    bbox_targets_all = torch.stack((left, top, right, bottom), dim=-1)

    inside_gt = bbox_targets_all.min(dim=-1)[0] > 0
    if center_sampling:
        stride_buf = offset.new_zeros(offset.shape)
        lvl_begin = 0
        for lvl_idx, n_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + n_lvl
            stride_buf[lvl_begin:lvl_end] = float(strides[lvl_idx]) * center_sample_radius
            lvl_begin = lvl_end
        inside_center = (offset.abs() < stride_buf).all(dim=-1)
        inside_gt = inside_gt & inside_center

    max_regress = bbox_targets_all.max(dim=-1)[0]
    inside_range = (max_regress >= rr[..., 0]) & (max_regress <= rr[..., 1])

    areas = areas.clone()
    areas[~inside_gt] = INF
    areas[~inside_range] = INF
    min_area, min_inds = areas.min(dim=1)

    # Convert 1-indexed GT labels → 0-indexed class ids
    labels = gt_labels[min_inds] - 1
    labels[min_area == INF] = bg
    bbox_targets = bbox_targets_all[torch.arange(num_points, device=device), min_inds]
    angle_targets = gt_angle[torch.arange(num_points, device=device), min_inds]

    if gt_bboxes_ignore is not None and gt_bboxes_ignore.numel() > 0:
        ignore_mask = _points_inside_any_obb(points, gt_bboxes_ignore)
        # Only ignore background / unassigned points that fall in ignore regions
        labels = torch.where(
            ignore_mask & (labels == bg),
            torch.full_like(labels, -1),
            labels,
        )

    if gt_bboxes_lookalike is not None and gt_bboxes_lookalike.numel() > 0:
        look_mask = _points_inside_any_obb(points, gt_bboxes_lookalike)
        # Never override foreground; lookalike wins over ignore → background K.
        labels = torch.where(
            look_mask & (labels < 0),
            torch.full_like(labels, bg),
            labels,
        )
        labels = torch.where(
            look_mask & (labels == bg),
            torch.full_like(labels, bg),
            labels,
        )

    pos_mask = (labels >= 0) & (labels < bg)
    return labels, bbox_targets, angle_targets, pos_mask


def _points_inside_any_obb(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """True where point is inside at least one OBB (via distance test)."""
    if boxes.numel() == 0:
        return points.new_zeros((points.size(0),), dtype=torch.bool)
    num_points = points.size(0)
    num_boxes = boxes.size(0)
    pts = points[:, None, :].expand(num_points, num_boxes, 2)
    gts = boxes[None].expand(num_points, num_boxes, 5)
    gt_ctr, gt_wh, gt_angle = torch.split(gts, [2, 2, 1], dim=2)
    cos_a, sin_a = torch.cos(gt_angle), torch.sin(gt_angle)
    rot = torch.cat([cos_a, sin_a, -sin_a, cos_a], dim=-1).reshape(num_points, num_boxes, 2, 2)
    offset = torch.matmul(rot, (pts - gt_ctr).unsqueeze(-1)).squeeze(-1)
    w, h = gt_wh[..., 0], gt_wh[..., 1]
    ox, oy = offset[..., 0], offset[..., 1]
    ltrb = torch.stack((w / 2 + ox, h / 2 + oy, w / 2 - ox, h / 2 - oy), dim=-1)
    return (ltrb.min(dim=-1)[0] > 0).any(dim=1)


class ConvGNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.gn = nn.GroupNorm(32, out_channels)
        self.act = nn.ReLU(inplace=True)
        nn.init.normal_(self.conv.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class RotatedFCOSHead(nn.Module):
    """MMRotate-style Rotated FCOS head: cls / reg(4) / angle / centerness."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        stacked_convs: int = 4,
        feat_channels: Optional[int] = None,
        strides: Sequence[float] = (8, 16, 32, 64, 128),
        centerness_on_reg: bool = True,
        norm_on_bbox: bool = True,
        scale_angle: bool = True,
    ):
        if nn is None:
            raise RuntimeError("PyTorch is required for RotatedFCOSHead.")
        super().__init__()
        self.num_classes = num_classes
        self.stacked_convs = max(1, int(stacked_convs))
        self.feat_channels = int(feat_channels or in_channels)
        self.strides = list(strides)
        self.centerness_on_reg = centerness_on_reg
        self.norm_on_bbox = norm_on_bbox
        self.scale_angle = scale_angle

        self.cls_convs = nn.ModuleList(
            [
                ConvGNReLU(in_channels if i == 0 else self.feat_channels, self.feat_channels)
                for i in range(self.stacked_convs)
            ]
        )
        self.reg_convs = nn.ModuleList(
            [
                ConvGNReLU(in_channels if i == 0 else self.feat_channels, self.feat_channels)
                for i in range(self.stacked_convs)
            ]
        )
        self.conv_cls = nn.Conv2d(self.feat_channels, num_classes, 3, padding=1)
        self.conv_bbox = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.conv_angle = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.scale_angle_module = Scale(1.0) if scale_angle else None

        for m in (self.conv_bbox, self.conv_angle, self.conv_centerness):
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.conv_cls.weight, std=0.01)
        bias_init = -math.log((1 - 0.01) / 0.01)
        nn.init.constant_(self.conv_cls.bias, bias_init)

    def forward(
        self,
        features: List[torch.Tensor],
        strides: Optional[Sequence[float]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        cls_scores: List[torch.Tensor] = []
        bbox_preds: List[torch.Tensor] = []
        angle_preds: List[torch.Tensor] = []
        centernesses: List[torch.Tensor] = []
        stride_list = list(strides) if strides is not None else list(self.strides)
        if len(stride_list) != len(features):
            raise ValueError(
                f"strides length {len(stride_list)} != features length {len(features)}"
            )
        if len(self.scales) != len(features):
            raise ValueError(
                f"scales length {len(self.scales)} != features length {len(features)}"
            )
        for feat, scale, stride in zip(features, self.scales, stride_list):
            cls_feat = feat
            for conv in self.cls_convs:
                cls_feat = conv(cls_feat)
            reg_feat = feat
            for conv in self.reg_convs:
                reg_feat = conv(reg_feat)

            cls_scores.append(self.conv_cls(cls_feat))
            bbox = scale(self.conv_bbox(reg_feat)).float()
            if self.norm_on_bbox:
                bbox = bbox.clamp(min=0)
                if not self.training:
                    bbox = bbox * float(stride)
            else:
                bbox = bbox.exp()
            bbox_preds.append(bbox)

            angle = self.conv_angle(reg_feat)
            if self.scale_angle_module is not None:
                angle = self.scale_angle_module(angle).float()
            angle_preds.append(angle)

            ctr_feat = reg_feat if self.centerness_on_reg else cls_feat
            centernesses.append(self.conv_centerness(ctr_feat))
        return cls_scores, bbox_preds, angle_preds, centernesses


_FCOS_AUX_LOSS_TYPES = ("kfiou", "probiou")


def _normalize_fcos_box_reg_loss_type(box_reg_loss_type: str) -> str:
    lt = (box_reg_loss_type or "l1").strip().lower()
    if lt == "probiou":
        raise ValueError(
            f"Unsupported FCOS box_reg_loss_type={box_reg_loss_type!r}; "
            "use 'l1', 'kfiou', or 'riou' (decoded differentiable polygon IoU). "
            "probiou is aux-only."
        )
    if lt == "smooth_l1":
        warnings.warn(
            "RotatedFCOS does not implement Smooth L1; mapping box_reg_loss_type "
            "'smooth_l1' → 'l1' (encoded ltrb + le90 angle).",
            UserWarning,
            stacklevel=3,
        )
        lt = "l1"
    if lt not in ("l1", "kfiou", "riou"):
        raise ValueError(
            f"Unsupported box_reg_loss_type={box_reg_loss_type!r}; "
            "use 'l1', 'kfiou', or 'riou'."
        )
    return lt


def _normalize_fcos_aux_loss(
    aux_loss_type: Optional[str],
    aux_loss_weight: float,
) -> Tuple[Optional[str], float]:
    weight = float(aux_loss_weight or 0.0)
    if weight < 0.0:
        raise ValueError(f"aux_loss_weight must be >= 0, got {aux_loss_weight!r}.")
    if weight == 0.0:
        return None, 0.0
    if aux_loss_type is None or str(aux_loss_type).strip() == "":
        raise ValueError(
            "aux_loss_weight > 0 requires aux_loss_type in {'kfiou', 'probiou'}."
        )
    aux_lt = str(aux_loss_type).strip().lower()
    if aux_lt == "riou":
        raise ValueError(
            "Unsupported FCOS aux_loss_type='riou' (sampling IoU is not backprop-friendly); "
            "use 'kfiou' or 'probiou'."
        )
    if aux_lt not in _FCOS_AUX_LOSS_TYPES:
        raise ValueError(
            f"Unsupported FCOS aux_loss_type={aux_loss_type!r}; use 'kfiou' or 'probiou'."
        )
    return aux_lt, weight


def _normalize_fcos_aux_angle(
    aux_lt: Optional[str],
    aux_angle_weight: float,
    aux_angle_lambda: float,
) -> Tuple[float, float]:
    """Heading-term knobs. Ignored when decoded aux is off."""
    weight = float(aux_angle_weight if aux_angle_weight is not None else 1.0)
    lam = float(aux_angle_lambda if aux_angle_lambda is not None else 1.0)
    if aux_lt is None:
        return 0.0, 1.0
    if weight < 0.0:
        raise ValueError(f"aux_angle_weight must be >= 0, got {aux_angle_weight!r}.")
    if weight > 0.0 and lam <= 0.0:
        raise ValueError(f"aux_angle_lambda must be > 0 when aux_angle_weight > 0, got {aux_angle_lambda!r}.")
    return weight, lam


def _decode_fcos_pos_boxes(
    bbox_coder: DistanceAnglePointCoder,
    points: torch.Tensor,
    ltrb: torch.Tensor,
    angle: torch.Tensor,
    ltrb_tgt: torch.Tensor,
    angle_tgt: torch.Tensor,
    strides: torch.Tensor,
    norm_on_bbox: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    stride_col = strides.unsqueeze(-1)
    if norm_on_bbox:
        pred_ltrb = ltrb * stride_col
        tgt_ltrb = ltrb_tgt * stride_col
    else:
        pred_ltrb = ltrb
        tgt_ltrb = ltrb_tgt
    decoded_pred = bbox_coder.decode(points, torch.cat([pred_ltrb, angle], dim=-1))
    decoded_tgt = bbox_coder.decode(
        points, torch.cat([tgt_ltrb, angle_tgt], dim=-1)
    ).detach()
    return decoded_pred, decoded_tgt


def _fcos_decoded_aux_per_box(
    aux_loss_type: str,
    decoded_pred: torch.Tensor,
    decoded_tgt: torch.Tensor,
    *,
    aux_angle_weight: float = 1.0,
    aux_angle_lambda: float = 1.0,
) -> torch.Tensor:
    if aux_loss_type == "kfiou":
        gauss = kfiou_loss_per_box(decoded_pred, decoded_tgt)
    else:
        gauss = probiou_loss_per_box(decoded_pred, decoded_tgt, mode="l1")
    if float(aux_angle_weight) == 0.0:
        return gauss
    heading = aspect_gated_angle_loss_per_box(
        decoded_pred, decoded_tgt, lam=float(aux_angle_lambda)
    )
    return gauss + float(aux_angle_weight) * heading


def compute_rotated_fcos_loss(
    cls_scores: List[torch.Tensor],
    bbox_preds: List[torch.Tensor],
    angle_preds: List[torch.Tensor],
    centernesses: List[torch.Tensor],
    points: List[torch.Tensor],
    strides: Sequence[float],
    gt_boxes: List[torch.Tensor],
    gt_labels: List[torch.Tensor],
    gt_boxes_ignore: Optional[List[torch.Tensor]],
    num_classes: int,
    regress_ranges: Sequence[Tuple[float, float]],
    center_sampling: bool,
    center_sample_radius: float,
    norm_on_bbox: bool,
    focal_alpha: float,
    focal_gamma: float,
    box_reg_weight: float,
    angle_weight: float,
    box_reg_loss_type: str = "l1",
    bbox_coder: Optional[DistanceAnglePointCoder] = None,
    aux_loss_type: Optional[str] = None,
    aux_loss_weight: float = 0.0,
    aux_angle_weight: float = 1.0,
    aux_angle_lambda: float = 1.0,
    gt_boxes_lookalike: Optional[List[torch.Tensor]] = None,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Focal cls + box regression (L1 / KFIoU / decoded rIoU) + optional decoded aux + centerness BCE."""
    device = cls_scores[0].device
    dtype = bbox_preds[0].dtype
    num_levels = len(points)
    num_imgs = cls_scores[0].size(0)
    reg_lt = _normalize_fcos_box_reg_loss_type(box_reg_loss_type)
    aux_lt, aux_w = _normalize_fcos_aux_loss(aux_loss_type, aux_loss_weight)
    aux_ang_w, aux_ang_lam = _normalize_fcos_aux_angle(
        aux_lt, aux_angle_weight, aux_angle_lambda
    )
    if bbox_coder is None:
        bbox_coder = DistanceAnglePointCoder(angle_version="le90")

    expanded_ranges = []
    for i in range(num_levels):
        rr = points[i].new_tensor(regress_ranges[i], dtype=dtype)
        expanded_ranges.append(rr[None].expand(points[i].size(0), 2))
    concat_points = torch.cat(points, dim=0)
    concat_ranges = torch.cat(expanded_ranges, dim=0)
    num_points_per_lvl = [p.size(0) for p in points]

    label_list = []
    bbox_tgt_list = []
    angle_tgt_list = []
    for img_idx in range(num_imgs):
        ignore = None
        if gt_boxes_ignore is not None and img_idx < len(gt_boxes_ignore):
            ignore = gt_boxes_ignore[img_idx]
        lookalike = None
        if gt_boxes_lookalike is not None and img_idx < len(gt_boxes_lookalike):
            lookalike = gt_boxes_lookalike[img_idx]
        labels_i, bbox_i, angle_i, _ = assign_fcos_targets_single(
            points=concat_points,
            gt_bboxes=gt_boxes[img_idx],
            gt_labels=gt_labels[img_idx],
            regress_ranges=concat_ranges,
            num_points_per_lvl=num_points_per_lvl,
            strides=strides,
            num_classes=num_classes,
            center_sampling=center_sampling,
            center_sample_radius=center_sample_radius,
            gt_bboxes_ignore=ignore,
            gt_bboxes_lookalike=lookalike,
        )
        # Split per level then re-concat per-level across images later
        label_list.append(list(labels_i.split(num_points_per_lvl, 0)))
        bbox_split = list(bbox_i.split(num_points_per_lvl, 0))
        angle_split = list(angle_i.split(num_points_per_lvl, 0))
        if norm_on_bbox:
            for lvl in range(num_levels):
                bbox_split[lvl] = bbox_split[lvl] / float(strides[lvl])
        bbox_tgt_list.append(bbox_split)
        angle_tgt_list.append(angle_split)

    flatten_cls = []
    flatten_bbox = []
    flatten_angle = []
    flatten_ctr = []
    flatten_labels = []
    flatten_bbox_tgt = []
    flatten_angle_tgt = []
    flatten_points = []
    flatten_strides = []

    for lvl in range(num_levels):
        cls_lvl = cls_scores[lvl].permute(0, 2, 3, 1).reshape(-1, num_classes)
        bbox_lvl = bbox_preds[lvl].permute(0, 2, 3, 1).reshape(-1, 4)
        angle_lvl = angle_preds[lvl].permute(0, 2, 3, 1).reshape(-1, 1)
        ctr_lvl = centernesses[lvl].permute(0, 2, 3, 1).reshape(-1)
        labels_lvl = torch.cat([label_list[i][lvl] for i in range(num_imgs)], dim=0)
        bbox_t = torch.cat([bbox_tgt_list[i][lvl] for i in range(num_imgs)], dim=0)
        angle_t = torch.cat([angle_tgt_list[i][lvl] for i in range(num_imgs)], dim=0)
        pts_lvl = points[lvl].repeat(num_imgs, 1)
        stride_lvl = pts_lvl.new_full((pts_lvl.size(0),), float(strides[lvl]))

        flatten_cls.append(cls_lvl)
        flatten_bbox.append(bbox_lvl)
        flatten_angle.append(angle_lvl)
        flatten_ctr.append(ctr_lvl)
        flatten_labels.append(labels_lvl)
        flatten_bbox_tgt.append(bbox_t)
        flatten_angle_tgt.append(angle_t)
        flatten_points.append(pts_lvl)
        flatten_strides.append(stride_lvl)

    flatten_cls = torch.cat(flatten_cls)
    flatten_bbox = torch.cat(flatten_bbox)
    flatten_angle = torch.cat(flatten_angle)
    flatten_ctr = torch.cat(flatten_ctr)
    flatten_labels = torch.cat(flatten_labels)
    flatten_bbox_tgt = torch.cat(flatten_bbox_tgt)
    flatten_angle_tgt = torch.cat(flatten_angle_tgt)
    flatten_points = torch.cat(flatten_points)
    flatten_strides = torch.cat(flatten_strides)

    bg = num_classes
    valid = flatten_labels >= 0
    pos = (flatten_labels >= 0) & (flatten_labels < bg)
    num_pos = int(pos.sum().item())
    num_pos_t = max(float(num_pos), 1.0)

    # Classification (sigmoid focal on valid points)
    if valid.any():
        cls_logits = flatten_cls[valid]
        cls_lab = flatten_labels[valid]
        cls_targets = torch.zeros_like(cls_logits)
        fg = (cls_lab >= 0) & (cls_lab < bg)
        if fg.any():
            cls_targets[fg, cls_lab[fg].long()] = 1.0
        loss_cls = sigmoid_focal_loss_sum(
            cls_logits,
            cls_targets,
            alpha=focal_alpha,
            gamma=focal_gamma,
            class_weights=class_weights,
        ) / num_pos_t
    else:
        loss_cls = flatten_cls.sum() * 0.0

    if num_pos > 0:
        pos_bbox = flatten_bbox[pos]
        pos_angle = flatten_angle[pos]
        pos_ctr = flatten_ctr[pos]
        pos_bbox_t = flatten_bbox_tgt[pos]
        pos_angle_t = flatten_angle_tgt[pos]
        ctr_t = centerness_target(pos_bbox_t).detach()
        ctr_denorm = max(float(ctr_t.sum().item()), 1e-6)

        decoded_pred = None
        decoded_tgt = None
        if reg_lt in ("kfiou", "riou") or aux_lt is not None:
            decoded_pred, decoded_tgt = _decode_fcos_pos_boxes(
                bbox_coder,
                flatten_points[pos],
                pos_bbox,
                pos_angle,
                pos_bbox_t,
                pos_angle_t,
                flatten_strides[pos],
                norm_on_bbox,
            )
        if reg_lt == "kfiou":
            loss_i = kfiou_loss_per_box(decoded_pred, decoded_tgt)
            loss_box = float(box_reg_weight) * (loss_i * ctr_t).sum() / ctr_denorm
        elif reg_lt == "riou":
            loss_i = riou_loss_per_box(decoded_pred, decoded_tgt)
            loss_box = float(box_reg_weight) * (loss_i * ctr_t).sum() / ctr_denorm
        else:
            l1_ltrb = (pos_bbox - pos_bbox_t).abs().sum(dim=-1)
            ang_err = norm_angle_le90(
                pos_angle.squeeze(-1) - pos_angle_t.squeeze(-1)
            ).abs()
            reg_per = l1_ltrb + float(angle_weight) * ang_err
            loss_box = float(box_reg_weight) * (reg_per * ctr_t).sum() / ctr_denorm

        loss_ctr = F.binary_cross_entropy_with_logits(pos_ctr, ctr_t, reduction="sum") / num_pos_t
        if aux_lt is not None:
            aux_i = _fcos_decoded_aux_per_box(
                aux_lt,
                decoded_pred,
                decoded_tgt,
                aux_angle_weight=aux_ang_w,
                aux_angle_lambda=aux_ang_lam,
            )
            loss_box_aux = float(aux_w) * (aux_i * ctr_t).sum() / ctr_denorm
    else:
        loss_box = flatten_bbox.sum() * 0.0
        loss_ctr = flatten_ctr.sum() * 0.0
        if aux_lt is not None:
            loss_box_aux = flatten_bbox.sum() * 0.0

    out = {
        "loss_classifier": loss_cls,
        "loss_box_reg": loss_box,
        "loss_centerness": loss_ctr,
    }
    if aux_lt is not None:
        out["loss_box_reg_aux"] = loss_box_aux
    return out


class RotatedFCOS(SigmoidFocalClassWeightsMixin, nn.Module):
    """Full Rotated FCOS detector (ResNet-FPN P3–P7)."""

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "resnet50",
        pretrained_backbone: bool = True,
        trainable_layers: int = 5,
        returned_layers: Optional[List[int]] = None,
        fpn_strides: Optional[List[float]] = None,
        fpn_extra_level: bool = True,
        stacked_convs: int = 4,
        center_sampling: bool = True,
        center_sample_radius: float = 1.5,
        norm_on_bbox: bool = True,
        centerness_on_reg: bool = True,
        scale_angle: bool = True,
        regress_ranges: Optional[Sequence[Tuple[float, float]]] = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        box_reg_weight: float = 1.0,
        box_reg_loss_type: str = "l1",
        aux_loss_type: Optional[str] = None,
        aux_loss_weight: float = 0.0,
        aux_angle_weight: float = 1.0,
        aux_angle_lambda: float = 1.0,
        angle_weight: float = 1.0,
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
            raise RuntimeError("PyTorch is required for RotatedFCOS.")
        super().__init__()
        box_reg_loss_type = _normalize_fcos_box_reg_loss_type(box_reg_loss_type)
        aux_loss_type, aux_loss_weight = _normalize_fcos_aux_loss(
            aux_loss_type, aux_loss_weight
        )
        aux_angle_weight, aux_angle_lambda = _normalize_fcos_aux_angle(
            aux_loss_type, aux_angle_weight, aux_angle_lambda
        )
        if aux_loss_type is not None and box_reg_loss_type == aux_loss_type:
            raise ValueError(
                f"aux_loss_type={aux_loss_type!r} duplicates box_reg_loss_type; "
                "use L1 primary + decoded aux, or a different aux type."
            )
        self.num_classes = int(num_classes)
        self._init_sigmoid_focal_class_weights(roi_class_weights)
        self.fpn_extra_level = bool(fpn_extra_level)
        self.returned_layers = returned_layers or [2, 3, 4]
        self.fpn_strides = list(fpn_strides or [8, 16, 32, 64, 128])
        self.center_sampling = center_sampling
        self.center_sample_radius = float(center_sample_radius)
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.scale_angle = scale_angle
        self.regress_ranges = tuple(regress_ranges or DEFAULT_REGRESS_RANGES)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.box_reg_weight = float(box_reg_weight)
        self.box_reg_loss_type = box_reg_loss_type
        self.aux_loss_type = aux_loss_type
        self.aux_loss_weight = float(aux_loss_weight)
        self.aux_angle_weight = float(aux_angle_weight)
        self.aux_angle_lambda = float(aux_angle_lambda)
        self.angle_weight = float(angle_weight)
        self.score_threshold = float(score_threshold)
        self.final_nms_iou_threshold = float(final_nms_iou_threshold)
        self.max_detections_per_image = int(max_detections_per_image)
        self.nms_pre = int(nms_pre)
        self.min_bbox_size = float(min_bbox_size)
        self.final_nms_use_cpu = bool(final_nms_use_cpu)
        self.nms_class_agnostic = bool(nms_class_agnostic)
        self._final_nms_iou_schedule_epochs = final_nms_iou_schedule_epochs
        self._final_nms_iou_schedule_values = final_nms_iou_schedule_values

        self.backbone, in_channels = setup_backbone(
            backbone=None,
            backbone_name=backbone_name,
            pretrained_backbone=pretrained_backbone,
            trainable_layers=trainable_layers,
            returned_layers=self.returned_layers,
            use_p6p7_extra_levels=self.fpn_extra_level,
        )
        self.head = RotatedFCOSHead(
            in_channels=in_channels,
            num_classes=self.num_classes,
            stacked_convs=stacked_convs,
            strides=self.fpn_strides,
            centerness_on_reg=centerness_on_reg,
            norm_on_bbox=norm_on_bbox,
            scale_angle=scale_angle,
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
            include_pool_level=not self.fpn_extra_level,
        )
        feature_map_sizes = [(f.shape[2], f.shape[3]) for f in feature_list]
        fpn_strides_live = derive_fpn_strides_from_grid(
            image_sizes[0], feature_map_sizes, configured=self.fpn_strides
        )
        warn_if_fpn_strides_mismatch(self.fpn_strides, fpn_strides_live)

        images_tensor = torch.stack(images, dim=0)
        device = images_tensor.device
        cls_scores, bbox_preds, angle_preds, centernesses = self.head(
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
            gt_boxes_list, gt_labels_list, gt_boxes_ignore_list, gt_boxes_lookalike_list = prepare_targets(
                targets, device=device
            )
            return compute_rotated_fcos_loss(
                cls_scores=cls_scores,
                bbox_preds=bbox_preds,
                angle_preds=angle_preds,
                centernesses=centernesses,
                points=points,
                strides=fpn_strides_live,
                gt_boxes=gt_boxes_list,
                gt_labels=gt_labels_list,
                gt_boxes_ignore=gt_boxes_ignore_list,
                gt_boxes_lookalike=gt_boxes_lookalike_list,
                num_classes=self.num_classes,
                regress_ranges=self.regress_ranges,
                center_sampling=self.center_sampling,
                center_sample_radius=self.center_sample_radius,
                norm_on_bbox=self.norm_on_bbox,
                focal_alpha=self.focal_alpha,
                focal_gamma=self.focal_gamma,
                box_reg_weight=self.box_reg_weight,
                angle_weight=self.angle_weight,
                box_reg_loss_type=self.box_reg_loss_type,
                bbox_coder=self.bbox_coder,
                aux_loss_type=self.aux_loss_type,
                aux_loss_weight=self.aux_loss_weight,
                aux_angle_weight=self.aux_angle_weight,
                aux_angle_lambda=self.aux_angle_lambda,
                class_weights=self.roi_class_weights_tensor,
            )

        return self._inference(
            cls_scores,
            bbox_preds,
            angle_preds,
            centernesses,
            points,
            device,
        )

    def _inference(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        angle_preds: List[torch.Tensor],
        centernesses: List[torch.Tensor],
        points: List[torch.Tensor],
        device: torch.device,
    ) -> List[Dict[str, Any]]:
        num_imgs = cls_scores[0].size(0)
        outputs: List[Dict[str, Any]] = []
        for img_idx in range(num_imgs):
            pre_nms = rotated_fcos_decode_pre_nms(
                [c[img_idx] for c in cls_scores],
                [b[img_idx] for b in bbox_preds],
                [a[img_idx] for a in angle_preds],
                [ctr[img_idx] for ctr in centernesses],
                points,
                self.bbox_coder,
                self.num_classes,
                self.score_threshold,
                self.nms_pre,
                self.min_bbox_size,
            )
            boxes, scores, labels = pre_nms.boxes, pre_nms.scores, pre_nms.labels
            if boxes.numel() == 0:
                outputs.append(
                    {
                        "rboxes": [],
                        "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                        "scores": torch.zeros((0,), dtype=torch.float32, device=device),
                    }
                )
                continue
            boxes, scores, labels = _apply_fcos_nms(
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


def _fcos_level_topk_decode(
    cls_map: torch.Tensor,
    bbox_map: torch.Tensor,
    angle_map: torch.Tensor,
    ctr_map: torch.Tensor,
    pts: torch.Tensor,
    bbox_coder: DistanceAnglePointCoder,
    num_classes: int,
    score_threshold: float,
    nms_pre: int,
    min_bbox_size: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-level top-k decode. Returns ``(boxes, scores, labels, valid)`` of length ``k``."""
    cls_flat = cls_map.permute(1, 2, 0).reshape(-1, num_classes)
    bbox_flat = bbox_map.permute(1, 2, 0).reshape(-1, 4)
    angle_flat = angle_map.permute(1, 2, 0).reshape(-1, 1)
    ctr_flat = ctr_map.permute(1, 2, 0).reshape(-1)
    scores = torch.sigmoid(cls_flat) * torch.sigmoid(ctr_flat).unsqueeze(1)
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


def rotated_fcos_decode_pre_nms(
    cls_per_level: Sequence[torch.Tensor],
    bbox_per_level: Sequence[torch.Tensor],
    angle_per_level: Sequence[torch.Tensor],
    centerness_per_level: Sequence[torch.Tensor],
    points: Sequence[torch.Tensor],
    bbox_coder: DistanceAnglePointCoder,
    num_classes: int,
    score_threshold: float,
    nms_pre: int,
    min_bbox_size: float,
) -> PreNmsDetections:
    """Decode one image's FCOS heads to pre-NMS boxes (no NMS).

    Per-level ``topk(min(nms_pre, H*W))`` runs **before** the score floor so the
    gather stays in the ONNX graph at a compile-time ``k``. That is equivalent
    to filter-then-topk used by the previous eager path.
    """
    boxes_l: List[torch.Tensor] = []
    scores_l: List[torch.Tensor] = []
    labels_l: List[torch.Tensor] = []
    for cls_map, bbox_map, angle_map, ctr_map, pts in zip(
        cls_per_level,
        bbox_per_level,
        angle_per_level,
        centerness_per_level,
        points,
    ):
        decoded, topk_scores, labels, valid = _fcos_level_topk_decode(
            cls_map,
            bbox_map,
            angle_map,
            ctr_map,
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
        return PreNmsDetections(
            torch.zeros((0, 5), dtype=torch.float32, device=device),
            torch.zeros((0,), dtype=torch.float32, device=device),
            torch.zeros((0,), dtype=torch.int64, device=device),
        )
    return PreNmsDetections(
        torch.cat(boxes_l, dim=0),
        torch.cat(scores_l, dim=0),
        torch.cat(labels_l, dim=0).to(dtype=torch.int64),
    )


def rotated_fcos_inference_pre_nms_padded(
    model: "RotatedFCOS",
    images: torch.Tensor,
    max_candidates: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch-1 export path: padded pre-NMS boxes/scores/labels + count.

    Each FPN level always emits ``topk(min(nms_pre, H*W))`` rows (invalid rows
    get score 0 and ``w=h=0``) so ONNX keeps a fixed-length gather and does not
    need ``argsort`` / ``nonzero``.
    """
    if images.dim() != 4 or images.shape[0] != 1:
        raise ValueError("rotated_fcos_inference_pre_nms_padded expects images [1, 3, H, W].")

    device = images.device
    dtype = images.dtype
    img_list = [images[0]]
    image_sizes = [(images.shape[2], images.shape[3])]
    feature_list = extract_backbone_features(
        model.backbone,
        img_list,
        use_checkpoint=False,
        training=False,
        include_pool_level=not model.fpn_extra_level,
    )
    feature_map_sizes = [(f.shape[2], f.shape[3]) for f in feature_list]
    fpn_strides_live = derive_fpn_strides_from_grid(
        image_sizes[0], feature_map_sizes, configured=model.fpn_strides
    )
    cls_scores, bbox_preds, angle_preds, centernesses = model.head(
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
    for cls_b, bbox_b, angle_b, ctr_b, pts in zip(
        cls_scores, bbox_preds, angle_preds, centernesses, points
    ):
        boxes_k, scores_k, labels_k, valid_k = _fcos_level_topk_decode(
            cls_b[0],
            bbox_b[0],
            angle_b[0],
            ctr_b[0],
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


def _apply_fcos_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    iou_threshold: float,
    max_detections_per_image: int,
    final_nms_use_cpu: bool,
    class_agnostic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotated NMS + global top-k (eval path; not exported).

    Class-aware (default): NMS per label, then global top-k.
    Class-agnostic: one ``rotated_nms`` over all boxes, then score sort.
    """
    if boxes.numel() == 0:
        return boxes, scores, labels

    if class_agnostic:
        keep_indices = rotated_nms(
            boxes,
            scores,
            iou_threshold=iou_threshold,
            max_detections=max_detections_per_image,
            force_cpu=final_nms_use_cpu,
        )
        if len(keep_indices) > 0:
            _, order = scores[keep_indices].sort(descending=True)
            keep_indices = keep_indices[order]
        return boxes[keep_indices], scores[keep_indices], labels[keep_indices]

    keep_indices = []
    for cls_id in labels.unique():
        cls_mask = labels == cls_id
        cls_boxes = boxes[cls_mask]
        cls_scores_i = scores[cls_mask]
        cls_idx = cls_mask.nonzero(as_tuple=False).squeeze(1)
        if cls_boxes.size(0) == 0:
            continue
        kept = rotated_nms(
            cls_boxes,
            cls_scores_i,
            iou_threshold=iou_threshold,
            max_detections=None,
            force_cpu=final_nms_use_cpu,
        )
        keep_indices.append(cls_idx[kept])
    if keep_indices:
        keep = torch.cat(keep_indices)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
    else:
        boxes = boxes[:0]
        scores = scores[:0]
        labels = labels[:0]

    if boxes.size(0) > max_detections_per_image:
        _, topk = scores.topk(max_detections_per_image)
        boxes = boxes[topk]
        scores = scores[topk]
        labels = labels[topk]
    return boxes, scores, labels


def _apply_fcos_class_aware_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    iou_threshold: float,
    max_detections_per_image: int,
    final_nms_use_cpu: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward-compatible alias for class-aware ``_apply_fcos_nms``."""
    return _apply_fcos_nms(
        boxes,
        scores,
        labels,
        iou_threshold=iou_threshold,
        max_detections_per_image=max_detections_per_image,
        final_nms_use_cpu=final_nms_use_cpu,
        class_agnostic=False,
    )

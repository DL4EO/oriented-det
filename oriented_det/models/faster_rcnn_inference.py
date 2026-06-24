"""Shared Rotated Faster R-CNN inference helpers (PyTorch deploy + ONNX export)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore

from ..ops.rotated_ops import rotated_nms
from ..utils.logging import logger
from .bbox_coder import xyxy_to_obb
from .horizontal_roi_coder import decode_delta_xywh_th
from .oriented_rpn import generate_horizontal_proposals, generate_oriented_anchors
from .utils import extract_backbone_features, tensor_to_rboxes

if TYPE_CHECKING:
    from .oriented_rcnn import RotatedFasterRCNN


@dataclass
class PreNmsDetections:
    """Decoded oriented boxes before final rotated NMS."""

    boxes: torch.Tensor  # [N, 5] cx,cy,w,h,angle
    scores: torch.Tensor  # [N]
    labels: torch.Tensor  # [N] 1-indexed foreground


def _roi_candidates_from_head(
    model: "RotatedFasterRCNN",
    img_proposals: torch.Tensor,
    feature_list: List[torch.Tensor],
    image_sizes: List[Tuple[int, int]],
    img_idx: int,
    fpn_strides_live: List[int],
    roi_spatial_scales_live: List[float],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Run ROI align + head; return filtered proposals/scores/labels/box_regression or None."""
    if img_proposals.numel() == 0:
        return None

    num_proposals = len(img_proposals)
    device = img_proposals.device
    box_to_image_tensor = torch.full((num_proposals,), img_idx, dtype=torch.long, device=device)
    roi_features = model.roi_align(
        feature_maps=feature_list,
        boxes_xyxy=img_proposals,
        image_sizes=image_sizes,
        box_to_image=box_to_image_tensor,
        fpn_strides_override=fpn_strides_live,
        spatial_scales_override=roi_spatial_scales_live,
    )
    roi_features_flat = roi_features.view(roi_features.shape[0], -1)
    class_logits, box_regression = model.roi_head(roi_features_flat)

    class_probs = F.softmax(class_logits, dim=1)
    fg_probs = class_probs[:, 1:]
    score_stats = fg_probs.max(dim=1).values
    _tracing = bool(
        torch.onnx.is_in_onnx_export() if hasattr(torch.onnx, "is_in_onnx_export") else False
    )
    if len(score_stats) > 0 and not _tracing:
        logger.trace(
            "Score distribution: min={:.3f}, max={:.3f}, mean={:.3f}, "
            "median={:.3f}, >0.3: {}, >0.5: {}, >0.7: {}",
            score_stats.min().item(),
            score_stats.max().item(),
            score_stats.mean().item(),
            score_stats.median().item(),
            (score_stats > 0.3).sum().item(),
            (score_stats > 0.5).sum().item(),
            (score_stats > 0.7).sum().item(),
        )

    score_threshold = model.inference_pre_nms_score_threshold
    if model.roi_inference_top_class_only:
        max_scores, argmax_cls = fg_probs.max(dim=1)
        keep_prop = max_scores > score_threshold
        if keep_prop.any():
            proposal_indices = keep_prop.nonzero(as_tuple=True)[0]
            filtered_proposals = img_proposals[proposal_indices]
            filtered_scores = max_scores[proposal_indices]
            filtered_class_indices = argmax_cls[proposal_indices]
            filtered_labels = filtered_class_indices + 1
            filtered_box_regression = box_regression[proposal_indices]
        else:
            return None
    else:
        candidate_mask = fg_probs > score_threshold
        if candidate_mask.any():
            proposal_indices, class_indices = candidate_mask.nonzero(as_tuple=True)
            filtered_proposals = img_proposals[proposal_indices]
            filtered_scores = fg_probs[proposal_indices, class_indices]
            filtered_labels = class_indices + 1
            filtered_box_regression = box_regression[proposal_indices]
            filtered_class_indices = class_indices
        else:
            return None

    if not _tracing:
        total_candidates = int(fg_probs.numel())
        num_after_threshold = int(filtered_proposals.shape[0])
        _roi_mode = "top_class" if model.roi_inference_top_class_only else "multiclass"
        logger.trace(
            "ROI inference candidates: mode={}, pre_nms_thr={:.3f}, total={}, kept_after_thr={}",
            _roi_mode,
            score_threshold,
            total_candidates,
            num_after_threshold,
        )

    if model.roi_class_agnostic_regression:
        if filtered_box_regression.shape[1] != 5:
            raise RuntimeError(
                f"Expected class-agnostic regression with shape [N, 5], "
                f"but got shape {filtered_box_regression.shape}."
            )
        selected_regression = filtered_box_regression
    else:
        expected_size = model.num_classes * 5
        if filtered_box_regression.shape[1] != expected_size:
            raise RuntimeError(
                f"Expected class-specific regression with shape [N, {expected_size}], "
                f"but got shape {filtered_box_regression.shape}."
            )
        filtered_box_regression = filtered_box_regression.view(
            len(filtered_proposals), model.num_classes, 5
        )
        selected_regression = filtered_box_regression[
            torch.arange(len(filtered_proposals), device=filtered_proposals.device),
            filtered_class_indices,
        ]

    return filtered_proposals, filtered_scores, filtered_labels, selected_regression


def decode_roi_refinements(
    model: "RotatedFasterRCNN",
    filtered_proposals: torch.Tensor,
    filtered_scores: torch.Tensor,
    filtered_labels: torch.Tensor,
    selected_regression: torch.Tensor,
) -> PreNmsDetections:
    refined_boxes = decode_delta_xywh_th(
        filtered_proposals,
        selected_regression,
        means=model.target_means,
        stds=model.target_stds,
        norm_factor=model.roi_norm_factor,
        edge_swap=model.roi_edge_swap,
        proj_xy=getattr(model, "roi_proj_xy", False),
    )
    return PreNmsDetections(boxes=refined_boxes, scores=filtered_scores, labels=filtered_labels)


def apply_final_rotated_nms(
    model: "RotatedFasterRCNN",
    pre_nms: PreNmsDetections,
) -> PreNmsDetections:
    refined_boxes = pre_nms.boxes
    filtered_scores = pre_nms.scores
    filtered_labels = pre_nms.labels

    if refined_boxes.numel() == 0:
        return pre_nms

    if model.nms_class_agnostic:
        keep_indices = rotated_nms(
            refined_boxes,
            filtered_scores,
            model.final_nms_iou_threshold,
            model.max_detections_per_image,
            force_cpu=model.final_nms_use_cpu,
        )
        if len(keep_indices) > 0:
            k_scores = filtered_scores[keep_indices]
            _, order = k_scores.sort(descending=True)
            keep_indices = keep_indices[order]
        kept_per_class = {"agnostic": int(len(keep_indices))}
    else:
        keep_indices_per_class = []
        kept_per_class = {}
        unique_labels = filtered_labels.unique()
        for cls in unique_labels:
            cls_mask = filtered_labels == cls
            cls_indices = cls_mask.nonzero(as_tuple=True)[0]
            cls_keep = rotated_nms(
                refined_boxes[cls_mask],
                filtered_scores[cls_mask],
                model.final_nms_iou_threshold,
                model.max_detections_per_image,
                force_cpu=model.final_nms_use_cpu,
            )
            kept_per_class[int(cls.item())] = int(len(cls_keep))
            keep_indices_per_class.append(cls_indices[cls_keep])
        if keep_indices_per_class:
            keep_indices = torch.cat(keep_indices_per_class, dim=0)
            keep_scores = filtered_scores[keep_indices]
            _, sort_idx = keep_scores.sort(descending=True)
            keep_indices = keep_indices[sort_idx]
            if model.max_detections_per_image is not None:
                keep_indices = keep_indices[: model.max_detections_per_image]
        else:
            keep_indices = torch.zeros((0,), dtype=torch.long, device=filtered_scores.device)

    final_boxes = refined_boxes[keep_indices]
    final_scores = filtered_scores[keep_indices]
    final_labels = filtered_labels[keep_indices]
    logger.trace(
        "ROI inference kept after {} NMS: total={}, per_class={}",
        "class-agnostic" if model.nms_class_agnostic else "class-wise",
        int(len(keep_indices)),
        kept_per_class,
    )

    if len(final_boxes) > 0:
        min_size = 1.0
        valid_mask = (final_boxes[:, 2] >= min_size) & (final_boxes[:, 3] >= min_size)
        final_boxes = final_boxes[valid_mask]
        final_scores = final_scores[valid_mask]
        final_labels = final_labels[valid_mask]

    return PreNmsDetections(boxes=final_boxes, scores=final_scores, labels=final_labels)


def _pad_xyxy_proposals(
    proposals: torch.Tensor,
    max_count: int,
    image_size: Tuple[int, int],
) -> Tuple[torch.Tensor, int]:
    """Pad/truncate proposals to fixed length for traceable ROI align (ONNX export)."""
    device = proposals.device
    dtype = proposals.dtype
    img_h, img_w = image_size
    n = min(int(proposals.shape[0]), int(max_count))
    padded = torch.zeros((max_count, 4), dtype=dtype, device=device)
    if n > 0:
        padded[:n] = proposals[:n]
    if n < max_count:
        # Harmless 1x1 boxes at origin for padded slots (filtered by score later).
        padded[n:] = torch.tensor(
            [[0.0, 0.0, 1.0, 1.0]], dtype=dtype, device=device
        ).expand(max_count - n, 4)
    return padded, n


def faster_rcnn_roi_pre_nms(
    model: "RotatedFasterRCNN",
    img_proposals: torch.Tensor,
    feature_list: List[torch.Tensor],
    image_sizes: List[Tuple[int, int]],
    img_idx: int,
    fpn_strides_live: List[int],
    roi_spatial_scales_live: List[float],
    *,
    pad_proposals_to: Optional[int] = None,
) -> Optional[PreNmsDetections]:
    if pad_proposals_to is not None and pad_proposals_to > 0:
        img_proposals, _ = _pad_xyxy_proposals(
            img_proposals, pad_proposals_to, image_sizes[img_idx]
        )
    cand = _roi_candidates_from_head(
        model, img_proposals, feature_list, image_sizes, img_idx, fpn_strides_live, roi_spatial_scales_live
    )
    if cand is None:
        return None
    filtered_proposals, filtered_scores, filtered_labels, selected_regression = cand
    return decode_roi_refinements(
        model, filtered_proposals, filtered_scores, filtered_labels, selected_regression
    )


def faster_rcnn_inference(
    model: "RotatedFasterRCNN",
    images: Sequence[torch.Tensor],
    *,
    anchors: Optional[List[torch.Tensor]] = None,
    deterministic_rpn: bool = False,
) -> List[Dict[str, Any]]:
    """Full Rotated Faster R-CNN inference (decode + rotated NMS + RBox outputs)."""
    if not isinstance(images, (list, tuple)):
        images = [images]

    image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
    images_tensor = torch.stack(images, dim=0)

    feature_list = extract_backbone_features(
        model.backbone,
        images,
        use_checkpoint=False,
        training=False,
        include_pool_level=True,  # P6 for the RPN, matching RotatedFasterRCNN.forward
    )
    feature_map_sizes = [(f.shape[2], f.shape[3]) for f in feature_list]
    from .utils import derive_fpn_strides_from_grid

    fpn_strides_live = derive_fpn_strides_from_grid(image_sizes[0], feature_map_sizes)
    roi_spatial_scales_live = [1.0 / s for s in fpn_strides_live]

    objectness_logits, bbox_regression = model.rpn_head(feature_list)

    if anchors is None:
        img_h, img_w = image_sizes[0]
        anchors = generate_oriented_anchors(
            image_size=(img_h, img_w),
            feature_map_sizes=feature_map_sizes,
            anchor_scales=model.anchor_scales,
            anchor_ratios=model.anchor_ratios,
            anchor_angles=model.anchor_angles,
            stride_per_level=fpn_strides_live,
        )

    # No RPN score threshold (MMRotate keeps top-k by score only);
    # model.inference_pre_nms_score_threshold filters ROI-head scores downstream.
    proposals_xyxy = generate_horizontal_proposals(
        objectness_logits=objectness_logits,
        bbox_regression=bbox_regression,
        anchors=anchors,
        image_sizes=image_sizes,
        score_threshold=0.0,
        nms_threshold=model.rpn_nms_threshold,
        pre_nms_top_n=model.rpn_pre_nms_top_n,
        post_nms_top_n=model.rpn_post_nms_top_n,
        min_size=model.rpn_min_size,
        target_means=(0.0, 0.0, 0.0, 0.0),
        target_stds=(1.0, 1.0, 1.0, 1.0),
        deterministic=deterministic_rpn,
    )

    return_debug = getattr(model, "_return_anchors_proposals", False)
    all_anchors = None
    if return_debug:
        all_anchors = torch.cat(anchors, dim=0).detach().cpu()

    outputs: List[Dict[str, Any]] = []
    for img_idx, img_proposals in enumerate(proposals_xyxy):
        if len(img_proposals) == 0:
            out: Dict[str, Any] = {
                "rboxes": [],
                "labels": torch.zeros((0,), dtype=torch.int64, device=images_tensor.device),
                "scores": torch.zeros((0,), dtype=torch.float32, device=images_tensor.device),
            }
            if return_debug:
                out["anchors"] = all_anchors
                out["proposals"] = (
                    img_proposals.detach().cpu()
                    if img_proposals.numel() > 0
                    else torch.zeros((0, 5), dtype=torch.float32)
                )
            outputs.append(out)
            continue

        pre_nms = faster_rcnn_roi_pre_nms(
            model,
            img_proposals,
            feature_list,
            image_sizes,
            img_idx,
            fpn_strides_live,
            roi_spatial_scales_live,
        )
        if pre_nms is None:
            out = {
                "rboxes": [],
                "labels": torch.zeros((0,), dtype=torch.int64, device=images_tensor.device),
                "scores": torch.zeros((0,), dtype=torch.float32, device=images_tensor.device),
            }
        else:
            final = apply_final_rotated_nms(model, pre_nms)
            out = {
                "rboxes": tensor_to_rboxes(final.boxes),
                "labels": final.labels,
                "scores": final.scores,
            }
        if return_debug:
            out["anchors"] = all_anchors
            out["proposals"] = xyxy_to_obb(img_proposals.detach()).cpu()
        outputs.append(out)

    return outputs


def _pad_dim0_to_k(
    tensor: torch.Tensor,
    k: int,
    pad_shape_tail: Tuple[int, ...],
) -> torch.Tensor:
    """Pad or truncate ``tensor`` along dim 0 to length ``k`` (ONNX-safe ``Concat``, not ``Expand``).

    Slice assignment (``out[:n] = x``) lowers to ``Expand`` in ONNX and fails at runtime when
    the dynamic length is not 1 or ``k``. ``torch.cat`` with zero padding traces correctly.
    """
    truncated = tensor[:k]
    pad_n = k - truncated.shape[0]
    pad = torch.zeros((pad_n,) + pad_shape_tail, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([truncated, pad], dim=0)


def pad_pre_nms_detections(
    pre_nms: Optional[PreNmsDetections],
    max_candidates: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad pre-NMS detections to fixed length for ONNX export."""
    k = int(max_candidates)
    if pre_nms is None or pre_nms.boxes.numel() == 0:
        boxes = torch.zeros((k, 5), dtype=dtype, device=device)
        scores = torch.zeros((k,), dtype=dtype, device=device)
        labels = torch.zeros((k,), dtype=torch.int64, device=device)
        count = torch.zeros((), dtype=torch.int64, device=device)
        return boxes, scores, labels, count

    boxes = _pad_dim0_to_k(pre_nms.boxes, k, (5,))
    scores = _pad_dim0_to_k(pre_nms.scores, k, ())
    labels = _pad_dim0_to_k(pre_nms.labels, k, ())
    count = torch.minimum(
        torch.as_tensor(pre_nms.boxes.shape[0], dtype=torch.int64, device=device),
        torch.as_tensor(k, dtype=torch.int64, device=device),
    )
    return boxes, scores, labels, count


def faster_rcnn_inference_pre_nms_padded(
    model: "RotatedFasterRCNN",
    images: torch.Tensor,
    max_candidates: int,
    *,
    anchors: List[torch.Tensor],
    fpn_strides_live: List[int],
    deterministic_rpn: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch-1 export path: padded pre-NMS boxes/scores/labels + count."""
    if images.dim() != 4 or images.shape[0] != 1:
        raise ValueError("faster_rcnn_inference_pre_nms_padded expects images [1, 3, H, W].")

    device = images.device
    dtype = images.dtype
    img_list = [images[0]]
    image_sizes = [(images.shape[2], images.shape[3])]

    feature_list = extract_backbone_features(
        model.backbone,
        img_list,
        use_checkpoint=False,
        training=False,
        include_pool_level=True,  # P6 for the RPN, matching RotatedFasterRCNN.forward
    )
    roi_spatial_scales_live = [1.0 / s for s in fpn_strides_live]

    objectness_logits, bbox_regression = model.rpn_head(feature_list)
    proposals_xyxy = generate_horizontal_proposals(
        objectness_logits=objectness_logits,
        bbox_regression=bbox_regression,
        anchors=anchors,
        image_sizes=image_sizes,
        score_threshold=0.0,
        nms_threshold=model.rpn_nms_threshold,
        pre_nms_top_n=model.rpn_pre_nms_top_n,
        post_nms_top_n=model.rpn_post_nms_top_n,
        min_size=model.rpn_min_size,
        target_means=(0.0, 0.0, 0.0, 0.0),
        target_stds=(1.0, 1.0, 1.0, 1.0),
        deterministic=deterministic_rpn,
    )

    img_proposals = proposals_xyxy[0]
    pre_nms = faster_rcnn_roi_pre_nms(
        model,
        img_proposals,
        feature_list,
        image_sizes,
        0,
        fpn_strides_live,
        roi_spatial_scales_live,
        pad_proposals_to=max_candidates,
    )
    return pad_pre_nms_detections(pre_nms, max_candidates, device, dtype)


__all__ = [
    "PreNmsDetections",
    "apply_final_rotated_nms",
    "decode_roi_refinements",
    "faster_rcnn_inference",
    "faster_rcnn_inference_pre_nms_padded",
    "faster_rcnn_roi_pre_nms",
    "pad_pre_nms_detections",
]

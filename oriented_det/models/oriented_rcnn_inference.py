"""Shared Oriented R-CNN inference helpers (ONNX export pre-NMS path)."""

from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore

from .faster_rcnn_inference import PreNmsDetections, pad_pre_nms_detections
from .oriented_rpn import decode_oriented_boxes, generate_midpoint_proposals
from .utils import extract_backbone_features

if TYPE_CHECKING:
    from .oriented_rcnn import OrientedRCNN


def _pad_obb_proposals(
    proposals: torch.Tensor,
    max_count: int,
) -> Tuple[torch.Tensor, int]:
    """Pad/truncate oriented proposals to fixed length for traceable ROI align.

    Pad slots use zero-size OBBs ``(0,0,0,0,0)`` so candidates can be dropped with a
    dynamic positive-size check after the ROI head.

    Always ``cat([proposals[:k], zeros(k)])[:k]`` so ONNX keeps a pad path even when
    the trace-time RPN count already equals ``k`` (zeros dummy). ``int(shape[0])``
    control flow that skips padding breaks real images with fewer proposals
    (``/roi_align/Reshape`` size mismatch).
    """
    k = int(max_count)
    pad = proposals.new_zeros((k, 5))
    padded = torch.cat([proposals[:k], pad], dim=0)[:k]
    return padded, k


def _obb_positive_size_mask(boxes: torch.Tensor) -> torch.Tensor:
    """Boolean mask for non-pad oriented boxes (positive w and h)."""
    return (boxes[:, 2] > 0) & (boxes[:, 3] > 0)


def _roi_candidates_from_head(
    model: "OrientedRCNN",
    img_proposals: torch.Tensor,
    feature_list: List[torch.Tensor],
    image_sizes: List[Tuple[int, int]],
    img_idx: int,
    fpn_strides_live: List[int],
    roi_spatial_scales_live: List[float],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Run oriented ROI align + head; return filtered proposals/scores/labels/reg or None."""
    if img_proposals.numel() == 0:
        return None

    # Track proposals' dim0 (do not use len() — freezes trace-time size in ONNX).
    box_to_image_tensor = torch.zeros_like(img_proposals[:, 0], dtype=torch.long)
    if img_idx != 0:
        box_to_image_tensor = box_to_image_tensor + int(img_idx)
    roi_features = model.roi_align(
        feature_maps=feature_list,
        boxes=img_proposals,
        image_sizes=image_sizes,
        box_to_image=box_to_image_tensor,
        fpn_strides_override=fpn_strides_live,
        spatial_scales_override=roi_spatial_scales_live,
    )
    roi_features_flat = roi_features.view(roi_features.shape[0], -1)
    class_logits, box_regression = model.roi_head(roi_features_flat)

    class_probs = F.softmax(class_logits, dim=1)
    fg_probs = class_probs[:, 1:]
    score_threshold = model.inference_pre_nms_score_threshold
    real_prop = _obb_positive_size_mask(img_proposals)

    if model.roi_inference_top_class_only:
        max_scores, argmax_cls = fg_probs.max(dim=1)
        keep_prop = (max_scores > score_threshold) & real_prop
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
        candidate_mask = (fg_probs > score_threshold) & real_prop.unsqueeze(1)
        if candidate_mask.any():
            proposal_indices, class_indices = candidate_mask.nonzero(as_tuple=True)
            filtered_proposals = img_proposals[proposal_indices]
            filtered_scores = fg_probs[proposal_indices, class_indices]
            filtered_labels = class_indices + 1
            filtered_box_regression = box_regression[proposal_indices]
            filtered_class_indices = class_indices
        else:
            return None

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


def oriented_rcnn_roi_pre_nms(
    model: "OrientedRCNN",
    img_proposals: torch.Tensor,
    feature_list: List[torch.Tensor],
    image_sizes: List[Tuple[int, int]],
    img_idx: int,
    fpn_strides_live: List[int],
    roi_spatial_scales_live: List[float],
    *,
    pad_proposals_to: Optional[int] = None,
) -> Optional[PreNmsDetections]:
    """ROI align + decode through pre-NMS detections (no final rotated NMS)."""
    if pad_proposals_to is not None and pad_proposals_to > 0:
        img_proposals, _ = _pad_obb_proposals(img_proposals, pad_proposals_to)
        # Match OrientedRCNN.eval clamp of proposal sides (pads stay w=h=0).
        ih, iw = image_sizes[img_idx]
        max_side = float(max(iw, ih)) * 2.0
        img_proposals = img_proposals.clone()
        wh = img_proposals[:, 2:4]
        positive = wh > 0
        wh_clamped = wh.clamp(min=1.0, max=max_side)
        img_proposals[:, 2:4] = torch.where(positive, wh_clamped, wh)
    elif img_proposals.numel() > 0:
        ih, iw = image_sizes[img_idx]
        max_side = float(max(iw, ih)) * 2.0
        img_proposals = img_proposals.clone()
        img_proposals[:, 2] = img_proposals[:, 2].clamp(min=1.0, max=max_side)
        img_proposals[:, 3] = img_proposals[:, 3].clamp(min=1.0, max=max_side)

    cand = _roi_candidates_from_head(
        model,
        img_proposals,
        feature_list,
        image_sizes,
        img_idx,
        fpn_strides_live,
        roi_spatial_scales_live,
    )
    if cand is None:
        return None
    filtered_proposals, filtered_scores, filtered_labels, selected_regression = cand
    refined = decode_oriented_boxes(
        filtered_proposals,
        selected_regression,
        target_means=model.target_means,
        target_stds=model.target_stds,
        norm_factor=model.roi_norm_factor,
        edge_swap=model.roi_edge_swap,
        proj_xy=model.roi_proj_xy,
    )
    return PreNmsDetections(
        boxes=refined, scores=filtered_scores, labels=filtered_labels
    )


def oriented_rcnn_inference_pre_nms_padded(
    model: "OrientedRCNN",
    images: torch.Tensor,
    max_candidates: int,
    *,
    anchors: List[torch.Tensor],
    fpn_strides_live: List[int],
    deterministic_rpn: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch-1 export path: padded pre-NMS boxes/scores/labels + count."""
    if images.dim() != 4 or images.shape[0] != 1:
        raise ValueError("oriented_rcnn_inference_pre_nms_padded expects images [1, 3, H, W].")

    device = images.device
    dtype = images.dtype
    img_list = [images[0]]
    image_sizes = [(images.shape[2], images.shape[3])]

    feature_list = extract_backbone_features(
        model.backbone,
        img_list,
        use_checkpoint=False,
        training=False,
        include_pool_level=True,  # P6 for the RPN, matching OrientedRCNN.forward
    )
    roi_spatial_scales_live = [1.0 / s for s in fpn_strides_live]

    objectness_logits, bbox_regression = model.rpn_head(feature_list)
    proposals_list = generate_midpoint_proposals(
        objectness_logits,
        bbox_regression,
        anchors,
        image_sizes,
        score_threshold=0.0,
        nms_threshold=model.rpn_nms_threshold,
        pre_nms_top_n=model.rpn_pre_nms_top_n,
        post_nms_top_n=model.rpn_post_nms_top_n,
        min_size=0.0,
        deterministic=deterministic_rpn,
    )

    img_proposals = proposals_list[0]
    pre_nms = oriented_rcnn_roi_pre_nms(
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
    "oriented_rcnn_inference_pre_nms_padded",
    "oriented_rcnn_roi_pre_nms",
]

#!/usr/bin/env python
"""
Script to run inference on validation images and save predictions with metrics to a dated folder.
Results include:
- JSON file with predictions and metrics per image
- Visualization images with bounding boxes
- Configuration metadata (experiment_dir, checkpoint, etc.)
"""

import os

# Match tools/train.py: reduces allocator fragmentation on long inference runs (must be before torch import).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import json
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

try:  # Optional dependency; only required for image viz output.
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
import numpy as np
import torch
from tqdm import tqdm

# Optional: ORIENTED_DET_CUDNN_BENCHMARK=0 avoids cuDNN autotune picking unstable / huge-workspace algorithms.
_cudnn_bm = os.environ.get("ORIENTED_DET_CUDNN_BENCHMARK", "").lower()
if _cudnn_bm in ("0", "false", "no", "off"):
    torch.backends.cudnn.benchmark = False

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from oriented_det import OrientedRCNN, RotatedFasterRCNN, RotatedRetinaNet
from oriented_det.data import DOTADataset, DOTAAnnotation
from oriented_det.data import (
    Detection,
    GroundTruth,
    compute_gt_best_iou_alignment_metrics,
    compute_oriented_map,
    format_gt_best_iou_alignment_table_from_dict,
    format_mmrotate_class_metrics_table,
    gt_best_iou_alignment_metrics_to_dict,
)
from oriented_det.data.build import build_split_dataset, dataset_format_name
from oriented_det.geometry import RBox, normalize_le90
from oriented_det.train.config import (
    TrainingExperimentConfig,
    get_preprocessing_params,
    apply_inference_config_to_model,
    effective_eval_metric_thresholds,
    resolve_preds_final_nms_iou_threshold,
    resolve_inference_sliding_window_overlap_pixels,
)
from oriented_det.utils import tqdm_progress_stream

# For diagnostics: raw inference + threshold + NMS; pad/tile path when image size ≠ model input
from oriented_det.runtime.inference import (
    count_sliding_window_positions,
    resolve_window_batch_size,
    run_inference_auto,
)
from oriented_det.runtime.checkpoint import (
    infer_num_classes_from_checkpoint,
    load_model_from_checkpoint,
    resolve_inference_config_path,
)
from oriented_det.data.dota_classes import DOTA_V1_CLASSES
from oriented_det.data.dota import dota_label_path_for_image
from oriented_det.ops.iou import rbox_iou
from export.val_dataset import collect_split_images

# ImageNet normalization constants (fallback only; prefer config.preprocessing to match training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]   # RGB
# MMDetection/MMRotate default (same as oriented_det.train.config.PREPROCESSING_DEFAULT_*)
MMDET_MEAN = [123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0]
MMDET_STD = [58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0]


def _safe_div(numer: float, denom: float) -> float:
    return float(numer / denom) if denom else 0.0


def _rbox_centroid_in_tile_interior(
    rbox: RBox, image_width: int, image_height: int, margin_px: int
) -> bool:
    """True when the centroid lies in the tile interior [margin, W-margin] x [margin, H-margin].

    Matches deploy ``_filter_detections_by_image_margin``: objects in the outer overlap
    band (centroid in [0, margin) or (W-margin, W]) are excluded from metrics.
    """
    if margin_px <= 0 or image_width <= 0 or image_height <= 0:
        return True
    left = float(margin_px)
    top = float(margin_px)
    right = float(image_width - margin_px)
    bottom = float(image_height - margin_px)
    if right <= left or bottom <= top:
        return False
    cx = float(rbox.cx)
    cy = float(rbox.cy)
    return left <= cx <= right and top <= cy <= bottom


def _resolve_metrics_margin_pixels(
    margin_pixels: Optional[int],
    overlap_ratio: Optional[float],
    overlap_pixels: int,
    preprocessing: Dict[str, Any],
) -> int:
    """Resolve metrics margin (pixels). Default is overlap/2."""
    if margin_pixels is not None:
        return max(0, int(margin_pixels))
    if overlap_ratio is not None:
        ts = preprocessing.get("target_size", [1024, 1024])
        th, tw = (int(ts[0]), int(ts[1])) if isinstance(ts, (list, tuple)) else (1024, 1024)
        overlap_px = int(round(min(th, tw) * float(overlap_ratio)))
        return max(0, overlap_px // 2)
    return max(0, int(overlap_pixels) // 2)


def _map_metric_label(iou_thr: float) -> str:
    """COCO-style name for AP at a single IoU (e.g. 0.5 -> mAP50)."""
    return f"mAP{int(round(float(iou_thr) * 100))}"


def _compute_fbeta(precision: float, recall: float, beta: float = 1.0) -> float:
    beta_sq = beta * beta
    denom = (beta_sq * precision) + recall
    if denom <= 0:
        return 0.0
    return float((1.0 + beta_sq) * precision * recall / denom)


def _match_counts(detections: List[Detection], ground_truths: List[GroundTruth], iou_threshold: float) -> Dict[str, int]:
    """Greedy one-to-one matching per image and class, sorted by detection confidence."""
    if not detections and not ground_truths:
        return {"tp": 0, "fp": 0, "fn": 0}
    if not detections:
        return {"tp": 0, "fp": 0, "fn": len(ground_truths)}
    if not ground_truths:
        return {"tp": 0, "fp": len(detections), "fn": 0}

    gt_used = [False] * len(ground_truths)
    tp = 0
    fp = 0

    for det in sorted(detections, key=lambda d: float(d.score), reverse=True):
        best_iou = -1.0
        best_j = -1
        det_cid = int(det.class_id)
        det_cid_alt = det_cid - 1  # Some inference paths use 1-based class labels.
        det_cname = str(getattr(det, "class_name", "") or "")
        for j, gt in enumerate(ground_truths):
            if gt_used[j]:
                continue
            gt_cid = int(gt.class_id)
            gt_cname = str(getattr(gt, "class_name", "") or "")
            # Robust class matching for mixed 0-based/1-based class IDs.
            if not (
                det_cid == gt_cid
                or det_cid_alt == gt_cid
                or (det_cname and gt_cname and det_cname == gt_cname)
            ):
                continue
            try:
                iou_val = rbox_iou(det.rbox, gt.rbox)
            except Exception:
                continue
            if iou_val >= iou_threshold and iou_val > best_iou:
                best_iou = iou_val
                best_j = j
        if best_j >= 0:
            gt_used[best_j] = True
            tp += 1
        else:
            fp += 1

    fn = int(len(ground_truths) - sum(gt_used))
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn)}


def _per_image_precision_recall_f1_f2(counts: Dict[str, int]) -> tuple[float, float, float, float]:
    """P/R/F1/F2 for one image at a fixed score threshold.

    When there are no ground truths and no predictions above threshold (tp=fp=fn=0),
    treat as a correct empty prediction: precision=recall=F1=F2=1.0 so these tiles are
    not mislabeled as failures (e.g. for hard-tile oversampling on F1).
    """
    if counts["tp"] == 0 and counts["fp"] == 0 and counts["fn"] == 0:
        return 1.0, 1.0, 1.0, 1.0
    precision = _safe_div(counts["tp"], counts["tp"] + counts["fp"])
    recall = _safe_div(counts["tp"], counts["tp"] + counts["fn"])
    return (
        precision,
        recall,
        _compute_fbeta(precision, recall, beta=1.0),
        _compute_fbeta(precision, recall, beta=2.0),
    )


def _compute_confusion_matrix(
    detections_by_image: Dict[str, List[Detection]],
    ground_truths_by_image: Dict[str, List[GroundTruth]],
    class_names: List[str],
    score_threshold: float,
    iou_threshold: float,
) -> Dict[str, Any]:
    """Compute a GT-row / prediction-column confusion matrix.

    Matching is greedy by detection score and IoU-only. This exposes class confusions:
    a detection overlapping a GT with the wrong class increments row=GT class,
    column=predicted class. Unmatched detections go to the False Positive row;
    unmatched GTs go to the Missed column.
    """
    classes = list(class_names) if class_names else sorted(
        {
            str(getattr(x, "class_name", "") or "")
            for values in list(detections_by_image.values()) + list(ground_truths_by_image.values())
            for x in values
            if str(getattr(x, "class_name", "") or "")
        }
    )
    missed_col = "__missed__"
    fp_row = "__false_positive__"
    row_labels = classes + [fp_row]
    col_labels = classes + [missed_col]
    matrix = {row: {col: 0 for col in col_labels} for row in row_labels}

    def _name(obj: Any) -> str:
        cname = str(getattr(obj, "class_name", "") or "")
        if cname:
            return cname
        cid = int(getattr(obj, "class_id", -1))
        if 1 <= cid <= len(classes):
            return classes[cid - 1]
        if 0 <= cid < len(classes):
            return classes[cid]
        return f"class_{cid}"

    image_ids = sorted(set(detections_by_image.keys()) | set(ground_truths_by_image.keys()))
    for image_id in image_ids:
        dets = [
            d for d in detections_by_image.get(image_id, [])
            if float(getattr(d, "score", 0.0)) >= float(score_threshold)
        ]
        gts = list(ground_truths_by_image.get(image_id, []))
        gt_used = [False] * len(gts)

        for det in sorted(dets, key=lambda d: float(getattr(d, "score", 0.0)), reverse=True):
            best_iou = -1.0
            best_j = -1
            for j, gt in enumerate(gts):
                if gt_used[j]:
                    continue
                try:
                    iou_val = rbox_iou(det.rbox, gt.rbox)
                except Exception:
                    continue
                if iou_val >= iou_threshold and iou_val > best_iou:
                    best_iou = iou_val
                    best_j = j

            pred_name = _name(det)
            if pred_name not in col_labels:
                matrix.setdefault(fp_row, {col: 0 for col in col_labels})
                for row in matrix.values():
                    row.setdefault(pred_name, 0)
                col_labels.append(pred_name)

            if best_j >= 0:
                gt_used[best_j] = True
                gt_name = _name(gts[best_j])
                if gt_name not in matrix:
                    matrix[gt_name] = {col: 0 for col in col_labels}
                    row_labels.insert(max(0, len(row_labels) - 1), gt_name)
                matrix[gt_name][pred_name] += 1
            else:
                matrix[fp_row][pred_name] += 1

        for used, gt in zip(gt_used, gts):
            if used:
                continue
            gt_name = _name(gt)
            if gt_name not in matrix:
                matrix[gt_name] = {col: 0 for col in col_labels}
                row_labels.insert(max(0, len(row_labels) - 1), gt_name)
            matrix[gt_name][missed_col] += 1

    return {
        "threshold": float(score_threshold),
        "iou_threshold": float(iou_threshold),
        "row_labels": row_labels,
        "column_labels": col_labels,
        "matrix": matrix,
        "row_label_display": {fp_row: "False Positive"},
        "column_label_display": {missed_col: "Missed"},
        "convention": "Rows are ground-truth classes; columns are predicted classes. False Positive row contains unmatched detections; Missed column contains unmatched GTs.",
    }


def _plot_pr_curve(pr_curve: List[Dict[str, float]], out_path: str) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Warning: matplotlib unavailable; skipping PR curve plot: {e}")
        return None

    recalls = [float(p["recall"]) for p in pr_curve]
    precisions = [float(p["precision"]) for p in pr_curve]
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker="o", linewidth=1.2)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def _plot_threshold_metrics(pr_curve: List[Dict[str, float]], out_path: str, best_idx: int) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Warning: matplotlib unavailable; skipping threshold metrics plot: {e}")
        return None

    thresholds = [float(p["threshold"]) for p in pr_curve]
    precisions = [float(p["precision"]) for p in pr_curve]
    recalls = [float(p["recall"]) for p in pr_curve]
    f1_values = [float(p["f1"]) for p in pr_curve]

    plt.figure(figsize=(9, 6))
    plt.plot(thresholds, precisions, label="Precision", linewidth=1.5)
    plt.plot(thresholds, recalls, label="Recall", linewidth=1.5)
    plt.plot(thresholds, f1_values, label="F1", linewidth=1.8)
    if 0 <= best_idx < len(thresholds):
        bt = thresholds[best_idx]
        bf1 = f1_values[best_idx]
        plt.scatter([bt], [bf1], marker="o", s=60, label=f"Best F1 @ {bt:.3f}")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.title("Precision/Recall/F1 vs Threshold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def _write_model_analysis_md(
    output_dir: str,
    metadata: Dict[str, Any],
    diagnostics: Optional[Dict[str, Any]],
    analysis: Dict[str, Any],
    artifacts: Dict[str, Optional[str]],
) -> str:
    report_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"model_analysis_{report_stamp}.md")
    checkpoint = metadata.get("checkpoint", "")
    ckpt_mtime = None
    if checkpoint and os.path.exists(checkpoint):
        ckpt_mtime = datetime.fromtimestamp(os.path.getmtime(checkpoint)).isoformat()

    bt = analysis.get("best_threshold", {})
    lines = [
        "# Model Analysis Report",
        "",
        f"- Generated at: `{datetime.now().isoformat()}`",
        "",
        "## Model metadata",
        f"- Experiment dir: `{metadata.get('experiment_dir', '')}`",
        f"- Checkpoint: `{checkpoint}`",
        f"- Checkpoint modified: `{ckpt_mtime}`",
        f"- Config: `{metadata.get('config_file', '')}`",
        "",
        "## Source data",
        f"- Data root: `{metadata.get('data_root', '')}`",
        f"- Data split: `{metadata.get('data_split', '')}`",
        f"- Total images: `{metadata.get('total_images', 0)}`",
        f"- Total ground truth objects: `{metadata.get('total_ground_truth', 0)}`",
        f"- Total predictions: `{metadata.get('total_predictions', 0)}`",
        "",
        "## Evaluation setup",
        (
            f"- mAP / PR matching IoU (rotated boxes, VOC-style; **not** NMS IoU): "
            f"`{float(analysis.get('iou_threshold', 0.5)):.2f}`"
        ),
        (
            f"- NMS IoU (deduplication): `{float(diagnostics.get('final_nms_iou_threshold', 0.0)):.2f}`"
            if diagnostics and (
                diagnostics.get("final_nms_iou_threshold") is not None
            )
            else "- NMS IoU: (see model config / CLI)"
        ),
        f"- Threshold sweep: `{analysis.get('threshold_min', 0.0)}` to `{analysis.get('threshold_max', 1.0)}` step `{analysis.get('threshold_step', 0.05)}`",
        "",
        "## Key outcomes",
        f"- Best threshold (F1): `{bt.get('threshold', 0.0):.4f}`",
        f"- Precision at best threshold: `{bt.get('precision', 0.0):.4f}`",
        f"- Recall at best threshold: `{bt.get('recall', 0.0):.4f}`",
        f"- F1 at best threshold: `{bt.get('f1', 0.0):.4f}`",
        f"- F2 at best threshold: `{bt.get('f2', 0.0):.4f}`",
    ]
    if diagnostics and isinstance(diagnostics.get("mAP"), (int, float)):
        _miou = float(diagnostics.get("iou_threshold", analysis.get("iou_threshold", 0.5)))
        lines.append(
            f"- {_map_metric_label(_miou)}: `{diagnostics['mAP']:.4f}` ({float(diagnostics['mAP']) * 100:.2f}%)"
        )
        align_raw = diagnostics.get("gt_alignment_metrics")
        if isinstance(align_raw, dict) and align_raw.get("per_class"):
            lines.extend(
                [
                    "",
                    "## GT alignment (mean best IoU vs raw detections)",
                    "",
                    (
                        f"- Global mean best IoU (any class): "
                        f"`{float(align_raw.get('mean_best_iou_any', 0.0)):.4f}`"
                    ),
                    (
                        f"- Global mean best IoU (same class): "
                        f"`{float(align_raw.get('mean_best_iou_same_class', 0.0)):.4f}` "
                        f"(median `{float(align_raw.get('median_best_iou_same_class', 0.0)):.4f}`)"
                    ),
                    "",
                    "Per-class breakdown (each GT: max rotated IoU vs detections on the same image):",
                    "",
                ]
            )
            class_order = metadata.get("class_names")
            table_md = format_gt_best_iou_alignment_table_from_dict(
                align_raw,
                class_names=class_order if isinstance(class_order, list) else None,
                markdown=True,
            )
            if table_md:
                lines.append(table_md)
        class_metrics = diagnostics.get("class_metrics") or {}
        class_aps = diagnostics.get("class_aps") or {}
        if isinstance(class_metrics, dict) and class_metrics:
            lines.extend(
                [
                    "",
                    f"## Per-class metrics ({_map_metric_label(_miou)})",
                    "",
                    "| Class | gts | dets | recall | AP |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            class_order = metadata.get("class_names")
            if isinstance(class_order, list) and class_order:
                ordered = [c for c in class_order if c in class_metrics] + [
                    c for c in class_metrics.keys() if c not in class_order
                ]
            else:
                ordered = sorted(class_metrics.keys())
            for cname in ordered:
                row = class_metrics.get(cname) or {}
                try:
                    ap = float(row.get("ap", class_aps.get(cname, 0.0)))
                except Exception:
                    ap = 0.0
                lines.append(
                    f"| `{cname}` | {int(row.get('gts', 0))} | {int(row.get('dets', 0))} | "
                    f"{float(row.get('recall', 0.0)):.3f} | {ap:.4f} |"
                )
            if isinstance(diagnostics.get("mAP"), (int, float)):
                lines.append(
                    f"| **mAP** | | | | {float(diagnostics['mAP']):.4f} |"
                )
        elif isinstance(class_aps, dict) and class_aps:
            lines.extend(
                [
                    "",
                    f"## Per-class AP ({_map_metric_label(_miou)})",
                    "",
                    "| Class | AP |",
                    "| --- | ---: |",
                ]
            )
            class_order = metadata.get("class_names")
            if isinstance(class_order, list) and class_order:
                ordered = [c for c in class_order if c in class_aps] + [
                    c for c in class_aps.keys() if c not in class_order
                ]
            else:
                ordered = sorted(class_aps.keys())
            for cname in ordered:
                try:
                    ap = float(class_aps.get(cname, 0.0))
                except Exception:
                    ap = 0.0
                lines.append(f"| `{cname}` | {ap:.4f} |")

    per_class = analysis.get("best_threshold_per_class") or {}
    criterion = str(analysis.get("per_class_criterion", "f1") or "f1")
    if per_class:
        crit_u = criterion.upper()
        lines.extend(
            [
                "",
                f"## Per-class best thresholds (max {crit_u} over the same sweep)",
                "",
                "| Class | Threshold | Precision | Recall | F1 | TP | FP | FN |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for cname in sorted(per_class.keys()):
            row = per_class[cname]
            if not isinstance(row, dict):
                continue
            lines.append(
                f"| `{cname}` | {float(row.get('threshold', 0.0)):.4f} | "
                f"{float(row.get('precision', 0.0)):.4f} | {float(row.get('recall', 0.0)):.4f} | "
                f"{float(row.get('f1', 0.0)):.4f} | {int(row.get('tp', 0))} | "
                f"{int(row.get('fp', 0))} | {int(row.get('fn', 0))} |"
            )

    confusion = analysis.get("confusion_matrix") or {}
    if isinstance(confusion, dict) and confusion.get("matrix"):
        matrix = confusion.get("matrix") or {}
        row_labels = list(confusion.get("row_labels") or [])
        col_labels = list(confusion.get("column_labels") or [])
        row_display = confusion.get("row_label_display") or {}
        col_display = confusion.get("column_label_display") or {}
        lines.extend(
            [
                "",
                "## Confusion matrix",
                "",
                (
                    f"Computed at score threshold `{float(confusion.get('threshold', bt.get('threshold', 0.0))):.4f}` "
                    f"and IoU `{float(confusion.get('iou_threshold', analysis.get('iou_threshold', 0.5))):.2f}`."
                ),
                "",
                "Rows are ground-truth classes; columns are predicted classes. The `False Positive` row contains unmatched detections; the `Missed` column contains unmatched GTs.",
                "",
            ]
        )
        header = ["Actual \\ Predicted"] + [
            str(col_display.get(c, c)) for c in col_labels
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] + ["---:" for _ in col_labels]) + " |")
        for row_key in row_labels:
            row = matrix.get(row_key) or {}
            row_name = str(row_display.get(row_key, row_key))
            values = [str(int(row.get(col_key, 0))) for col_key in col_labels]
            lines.append("| " + " | ".join([f"`{row_name}`"] + values) + " |")

    note_lines = [
        "## Notes",
        "- Global threshold selected by maximizing F1; tie-breaks favor recall, then lower threshold.",
    ]
    if per_class:
        note_lines.append(
            "- Per-class table: best threshold per class maximizes F1 on the same threshold grid "
            "(see `best_threshold_per_class` in the analysis JSON)."
        )
    note_lines.extend(
        [
            "- Precision/recall are computed using class-aware IoU matching with one-to-one assignment.",
            "",
        ]
    )

    lines.extend(
        [
            "",
            "## Artifacts",
            f"- Predictions JSON: `predictions.json`",
            f"- Analysis JSON: `{os.path.basename(artifacts.get('analysis_json') or 'analysis_iou0.50.json')}`",
            f"- PR curve: `{os.path.basename(artifacts.get('pr_curve') or 'pr_curve.png')}`",
            f"- Threshold metrics: `{os.path.basename(artifacts.get('threshold_metrics') or 'threshold_metrics.png')}`",
            "",
            *note_lines,
        ]
    )
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    return report_path


def find_latest_experiment(model_type: str = 'oriented_rcnn') -> tuple:
    """
    Automatically find the latest experiment directory and best checkpoint.
    
    Args:
        model_type: Model type directory name (e.g., 'oriented_rcnn')
    
    Returns:
        tuple: (experiment_dir, checkpoint_path, config_path)
    """
    from oriented_det.train.utils import get_project_root

    runs_root = get_project_root() / "runs"
    model_dir = runs_root / model_type
    
    if not model_dir.exists():
        raise ValueError(f"Model directory not found: {model_dir}")
    
    # Find latest experiment directory (sorted by modification time)
    experiment_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()], 
                          key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not experiment_dirs:
        raise ValueError(f"No experiment directories found in {model_dir}")
    
    latest_dir = experiment_dirs[0]
    
    # Find best checkpoint, fallback to latest epoch checkpoint
    checkpoint_dir = latest_dir / 'checkpoints'
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoints directory not found: {checkpoint_dir}")
    
    # Prefer best checkpoint (checkpoint_best.pth or best_*.pth e.g. best_mAP_0.42.pth)
    legacy_best = checkpoint_dir / 'checkpoint_best.pth'
    best_ckpts = [legacy_best] if legacy_best.exists() else sorted(checkpoint_dir.glob('best_*.pth'))
    if best_ckpts:
        checkpoint = best_ckpts[0]
    else:
        epoch_ckpts = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pth'), 
                            key=lambda x: x.stat().st_mtime, reverse=True)
        if not epoch_ckpts:
            raise ValueError(f"No checkpoints found in {checkpoint_dir}")
        checkpoint = epoch_ckpts[0]
    
    # Find config file
    config_path = latest_dir / 'config.json'
    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")
    
    return str(latest_dir), str(checkpoint), str(config_path)


def resolve_experiment_paths(experiment_dir: str) -> tuple[str, str, str]:
    """
    Resolve (experiment_dir, checkpoint_path, config_path) for a specific experiment directory.

    This mirrors `find_latest_experiment`, but *does not* look at other runs. It is intended for
    CLI usage like:
      --experiment-dir runs/<model>/<timestamp>
    where the user expects config and checkpoint to come from that directory.
    """
    exp_dir = Path(experiment_dir)
    if not exp_dir.exists():
        raise ValueError(f"Experiment directory not found: {exp_dir}")

    checkpoint_dir = exp_dir / "checkpoints"
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoints directory not found: {checkpoint_dir}")

    legacy_best = checkpoint_dir / "checkpoint_best.pth"
    best_ckpts = [legacy_best] if legacy_best.exists() else sorted(checkpoint_dir.glob("best_*.pth"))
    if best_ckpts:
        checkpoint = best_ckpts[0]
    else:
        epoch_ckpts = sorted(
            checkpoint_dir.glob("checkpoint_epoch_*.pth"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if not epoch_ckpts:
            raise ValueError(f"No checkpoints found in {checkpoint_dir}")
        checkpoint = epoch_ckpts[0]

    config_path = exp_dir / "config.json"
    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")

    return str(exp_dir), str(checkpoint), str(config_path)



def rbox_to_array(rbox: RBox) -> np.ndarray:
    """Convert RBox to numpy array format (cx, cy, w, h, angle)."""
    return np.array([rbox.cx, rbox.cy, rbox.width, rbox.height, rbox.angle])


def load_gt_as_ground_truths(
    txt_path: Path,
    class_map: dict,
) -> list:
    """Load DOTA annotations as list of GroundTruth for mAP.

    Expectation: ``txt_path`` pairs with an image in ``images/`` (same stem). Polygon coordinates
    must already be in **that image's pixel space** (origin top-left, same width/height as the
    image). There is no remapping between a separate “global” label space and the file on disk.

    class_map: class_name -> class_id. Boxes are normalized to le90 like training.
    """
    if not Path(txt_path).exists():
        return []
    gt_list = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ann = DOTAAnnotation.from_line(line)
                cid = class_map.get(ann.class_name, -1)
                if cid < 0:
                    continue
                # Match training: normalize to le90 (width >= height, angle in [-π/2, π/2))
                rbox = normalize_le90(ann.rbox)
                gt_list.append(
                    GroundTruth(
                        rbox=rbox,
                        class_id=cid,
                        class_name=ann.class_name,
                        difficult=ann.difficult,
                    )
                )
            except Exception:
                continue
    return gt_list


def _annotations_to_ground_truths(annotations: List[Any], class_map: dict) -> list:
    """Convert list of DOTAAnnotation to list of GroundTruth for mAP (e.g. from Airbus CSV dataset)."""
    gt_list = []
    for ann in annotations:
        cid = class_map.get(ann.class_name, -1)
        if cid < 0:
            continue
        rbox = normalize_le90(ann.rbox)
        gt_list.append(
            GroundTruth(rbox=rbox, class_id=cid, class_name=ann.class_name, difficult=ann.difficult)
        )
    return gt_list


def draw_rotated_boxes(img: np.ndarray, rboxes: List[RBox], scores: np.ndarray, 
                       labels: np.ndarray, score_thr: float = 0.3, 
                       class_names: List[str] = None) -> np.ndarray:
    """
    Draw rotated bounding boxes on image
    
    Args:
        img: numpy array of image (RGB)
        rboxes: list of RBox objects
        scores: confidence scores
        labels: class labels
        score_thr: score threshold for visualization
        class_names: list of class names
    
    Returns:
        Image with drawn boxes (RGB)
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for visualization. Install `opencv-python`.")
    # Convert RGB to BGR for cv2 drawing functions
    img_show = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
    
    for rbox, score, label in zip(rboxes, scores, labels):
        if score < score_thr:
            continue
        
        # Convert RBox to polygon points
        polygon = rbox.to_polygon()
        points = np.array([[p.x, p.y] for p in polygon.points], dtype=np.int32)
        
        # Draw rotated rectangle (cv2 expects BGR, so green is (0, 255, 0) in BGR)
        color = (0, 255, 0)  # Green for detections (BGR format for cv2)
        cv2.polylines(img_show, [points], isClosed=True, color=color, thickness=2)
        
        # Add label and score (labels are 1-indexed from model; use 0-based index for class_names)
        idx = label - 1 if class_names and 1 <= label <= len(class_names) else None
        label_name = class_names[idx] if idx is not None else f"class_{label}"
        text = f"{label_name}: {score:.2f}"
        cv2.putText(img_show, text, (int(rbox.cx), int(rbox.cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Convert back to RGB for return
    return cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)


def load_dota_annotations(txt_file: str) -> tuple:
    """Load DOTA format annotations from txt file
    
    Returns:
        tuple: (rboxes array, class_names list)
    """
    from oriented_det.data import DOTAAnnotation
    
    rboxes = []
    class_names = []
    
    if not os.path.exists(txt_file):
        return np.array([]).reshape(0, 5), []
    
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ann = DOTAAnnotation.from_line(line)
                rboxes.append(ann.rbox)
                class_names.append(ann.class_name)
            except Exception as e:
                # Skip invalid lines
                continue
    
    if len(rboxes) == 0:
        return np.array([]).reshape(0, 5), []
    
    # Convert to array format
    rbox_array = np.array([rbox_to_array(rbox) for rbox in rboxes])
    return rbox_array, class_names


def _metric_maps_from_saved_results(
    results: List[Dict[str, Any]],
    metrics_margin_px: int,
    class_names: Optional[List[str]],
) -> Tuple[
    Dict[str, List[Detection]],
    Dict[str, List[GroundTruth]],
    Dict[str, str],
    List[float],
]:
    """Rebuild Detection/GroundTruth maps from predictions.json ``results`` for offline metrics."""
    all_detections: Dict[str, List[Detection]] = {}
    all_ground_truths: Dict[str, List[GroundTruth]] = {}
    image_name_by_id: Dict[str, str] = {}
    all_scores: List[float] = []

    for row in results:
        img_name = str(row.get("image_name") or "")
        image_id = Path(img_name).stem
        image_name_by_id[image_id] = img_name
        iw = int(row.get("image_width") or 0)
        ih = int(row.get("image_height") or 0)

        detections_for_metrics: List[Detection] = []
        for p in row.get("predictions") or []:
            bbox = p["bbox"]
            rbox = normalize_le90(
                RBox(cx=float(bbox[0]), cy=float(bbox[1]), width=float(bbox[2]), height=float(bbox[3]), angle=float(bbox[4]))
            )
            label = int(p["label"])
            cname = str(p.get("class_name") or "")
            if (not cname) and class_names and 1 <= label <= len(class_names):
                cname = str(class_names[label - 1])
            detections_for_metrics.append(
                Detection(rbox=rbox, score=float(p["score"]), class_id=label, class_name=cname, image_id=image_id)
            )
            all_scores.append(float(p["score"]))

        gt_for_metrics: List[GroundTruth] = []
        for g in row.get("ground_truths") or []:
            bbox = g["bbox"]
            rbox = normalize_le90(
                RBox(cx=float(bbox[0]), cy=float(bbox[1]), width=float(bbox[2]), height=float(bbox[3]), angle=float(bbox[4]))
            )
            cid = int(g.get("class_id", -1))
            cname = str(g.get("class_name") or "unknown")
            diff = int(g.get("difficult", 0))
            gt_for_metrics.append(
                GroundTruth(rbox=rbox, class_id=cid, class_name=cname, difficult=diff, image_id=image_id)
            )

        if metrics_margin_px > 0 and iw > 0 and ih > 0:
            detections_for_metrics = [
                d for d in detections_for_metrics
                if _rbox_centroid_in_tile_interior(d.rbox, iw, ih, metrics_margin_px)
            ]
            gt_for_metrics = [
                g for g in gt_for_metrics
                if _rbox_centroid_in_tile_interior(g.rbox, iw, ih, metrics_margin_px)
            ]

        all_detections[image_id] = detections_for_metrics
        all_ground_truths[image_id] = gt_for_metrics

    return all_detections, all_ground_truths, image_name_by_id, all_scores


def run_diagnostics_pipeline(
    *,
    experiment_dir: str,
    checkpoint_path: str,
    config_path: str,
    data_root: Any,
    data_split: str,
    class_names: List[str],
    score_threshold: float,
    per_cls_thr: Optional[Dict[str, float]],
    nms_threshold: float,
    nms_class_agnostic: bool,
    iou_threshold: float,
    pr_iou_threshold: Optional[float],
    pr_threshold_min: float,
    pr_threshold_max: float,
    pr_threshold_step: float,
    per_class_threshold_analysis: bool,
    resolved_metrics_margin_px: int,
    all_detections: Dict[str, List[Detection]],
    all_ground_truths: Dict[str, List[GroundTruth]],
    results: List[Dict[str, Any]],
    all_scores: List[float],
    image_name_by_id: Dict[str, str],
    sliding_window_positions_total: Optional[int],
    window_batch_effective: Optional[int],
    t_infer_sec: float,
    device: str,
    output_dir: str,
    tile_metrics_csv: Optional[str],
    use_exact_rotated_iou: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Per-image stats, oriented mAP, PR sweep, analysis artifacts (shared inference vs offline-from-json)."""
    total_raw = sum(r.get("stats", {}).get("num_raw", 0) for r in results)
    total_after_threshold = sum(r.get("stats", {}).get("num_after_threshold", 0) for r in results)
    total_after_nms = sum(r.get("stats", {}).get("num_after_nms", 0) for r in results)
    any_sliding_stats = any(r.get("stats", {}).get("sliding_window") for r in results)
    total_pred_json = sum(r.get("num_pred", 0) for r in results)
    diagnostics: Dict[str, Any] = {
        'score_threshold': score_threshold,
        'per_class_score_threshold': per_cls_thr,
        'final_nms_iou_threshold': nms_threshold,
        'nms_class_agnostic': bool(nms_class_agnostic),
        'iou_threshold': iou_threshold,
        'total_raw': total_raw,
        'total_after_threshold': total_after_threshold,
        'total_after_nms': total_after_nms,
        'sliding_window_eval': any_sliding_stats,
        'total_predictions': total_pred_json,
        'metrics_margin_pixels': int(resolved_metrics_margin_px),
    }
    if resolved_metrics_margin_px > 0:
        diagnostics['total_predictions_after_metrics_margin'] = int(sum(len(v) for v in all_detections.values()))
        diagnostics['total_ground_truth_after_metrics_margin'] = int(sum(len(v) for v in all_ground_truths.values()))
    if sliding_window_positions_total is not None:
        diagnostics['estimated_sliding_window_positions'] = sliding_window_positions_total
    if window_batch_effective is not None:
        diagnostics['oriented_det_window_batch_size'] = window_batch_effective
    if sliding_window_positions_total is not None and window_batch_effective is not None:
        diagnostics['estimated_model_batch_calls'] = max(
            1, (sliding_window_positions_total + window_batch_effective - 1) // window_batch_effective
        )
    diagnostics['inference_loop_seconds'] = float(t_infer_sec)
    if all_scores:
        arr = np.array(all_scores)
        diagnostics['score_min'] = float(np.min(arr))
        diagnostics['score_max'] = float(np.max(arr))
        diagnostics['score_mean'] = float(np.mean(arr))
        diagnostics['score_std'] = float(np.std(arr)) if len(arr) > 1 else 0.0
        diagnostics['score_percentiles'] = {
            'p25': float(np.percentile(arr, 25)),
            'p50': float(np.percentile(arr, 50)),
            'p75': float(np.percentile(arr, 75)),
            'p90': float(np.percentile(arr, 90)),
        }
        diagnostics['counts_at_thresholds'] = {t: int((arr >= t).sum()) for t in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]}
    else:
        diagnostics['score_min'] = diagnostics['score_max'] = diagnostics['score_mean'] = diagnostics['score_std'] = None
        diagnostics['score_percentiles'] = {}
        diagnostics['counts_at_thresholds'] = {}

    flat_dets = []
    for img_id, dets in all_detections.items():
        for d in dets:
            flat_dets.append((img_id, d))
    if flat_dets and all_ground_truths:
        import random
        from oriented_det.ops.iou import rbox_iou as _rbox_iou_diag

        max_iou = 0.0
        sample_size = min(4000, len(flat_dets))
        flat_sample = random.Random(0).sample(flat_dets, sample_size) if len(flat_dets) > sample_size else flat_dets
        for img_id, det in flat_sample:
            gts = all_ground_truths.get(img_id, [])
            for gt in gts:
                if gt.class_name != det.class_name:
                    continue
                try:
                    iou_val = _rbox_iou_diag(det.rbox, gt.rbox)
                    if iou_val > max_iou:
                        max_iou = iou_val
                except Exception:
                    continue
        diagnostics['sample_max_iou'] = max_iou
        diagnostics['sample_max_iou_num_checked'] = len(flat_sample)

    print("Computing GT alignment metrics (per-class mean best IoU)...", flush=True)
    try:
        align_device = None if use_exact_rotated_iou else torch.device(device)
        gt_alignment = compute_gt_best_iou_alignment_metrics(
            all_detections,
            all_ground_truths,
            class_names=class_names,
            use_exact_rotated_iou=use_exact_rotated_iou,
            device=align_device,
            show_progress=True,
            progress_stream=tqdm_progress_stream(),
        )
        diagnostics["gt_alignment_metrics"] = gt_best_iou_alignment_metrics_to_dict(gt_alignment)
        print(
            f"  Global mean best IoU (same class): {gt_alignment.mean_best_iou_same_class:.4f}, "
            f"median: {gt_alignment.median_best_iou_same_class:.4f}",
            flush=True,
        )
    except Exception as e:
        diagnostics["gt_alignment_metrics"] = None
        diagnostics["gt_alignment_metrics_error"] = str(e)
        print(f"\nWarning: GT alignment metrics failed: {e}\n", flush=True)

    try:
        device_obj = None if use_exact_rotated_iou else torch.device(device)
        mean_ap, class_aps, class_metrics = compute_oriented_map(
            detections=all_detections,
            ground_truths=all_ground_truths,
            iou_threshold=iou_threshold,
            class_names=class_names,
            show_progress=True,
            progress_stream=tqdm_progress_stream(),
            device=device_obj,
            use_exact_rotated_iou=use_exact_rotated_iou,
        )
        diagnostics['mAP'] = mean_ap
        diagnostics['class_aps'] = class_aps
        diagnostics['class_metrics'] = {
            name: {
                'gts': m.num_gts,
                'dets': m.num_dets,
                'recall': m.recall,
                'ap': m.ap,
            }
            for name, m in class_metrics.items()
        }
        _ml = _map_metric_label(iou_threshold)
        print("\n" + "=" * 60, flush=True)
        print(
            f"Final {_ml}: {mean_ap:.4f} ({mean_ap * 100:.2f}%)  "
            f"[rotated IoU ≥ {float(iou_threshold):.2f} for GT–det matching; "
            f"NMS IoU = {float(nms_threshold):.2f}]",
            flush=True,
        )
        if class_metrics:
            print("\nPer-class metrics:", flush=True)
            print("", flush=True)
            print(
                format_mmrotate_class_metrics_table(
                    class_metrics,
                    class_names=class_names,
                    mean_ap=mean_ap,
                ),
                flush=True,
            )
            print("", flush=True)
        align_dict = diagnostics.get("gt_alignment_metrics")
        if isinstance(align_dict, dict) and align_dict.get("per_class"):
            print("Per-class GT alignment (mean best IoU vs raw detections):", flush=True)
            print("", flush=True)
            print(
                format_gt_best_iou_alignment_table_from_dict(
                    align_dict,
                    class_names=class_names,
                ),
                flush=True,
            )
            print("", flush=True)
        print("=" * 60 + "\n", flush=True)
    except Exception as e:
        diagnostics['mAP'] = None
        diagnostics['class_aps'] = {}
        diagnostics['class_metrics'] = {}
        diagnostics['mAP_error'] = str(e)
        print(f"\nWarning: mAP computation failed: {e}\n", flush=True)

    threshold_step = max(float(pr_threshold_step), 1e-6)
    pr_iou = float(iou_threshold if pr_iou_threshold is None else pr_iou_threshold)
    threshold_min = max(0.0, min(1.0, float(pr_threshold_min)))
    threshold_max = max(0.0, min(1.0, float(pr_threshold_max)))
    if threshold_max < threshold_min:
        threshold_min, threshold_max = threshold_max, threshold_min
    thresholds = np.arange(threshold_min, threshold_max + threshold_step * 0.5, threshold_step)
    if len(thresholds) == 0:
        thresholds = np.array([score_threshold], dtype=float)
    thresholds = np.clip(thresholds, 0.0, 1.0)

    curve_points: List[Dict[str, float]] = []
    per_image_metrics: List[Dict[str, Any]] = []
    best_idx = -1
    best_key = (-1.0, -1.0, 0.0)

    print(
        f"PR/F1 global threshold sweep: {len(thresholds)} steps "
        f"([{threshold_min:.3f}, {threshold_max:.3f}], step={threshold_step:.4f})",
        flush=True,
    )
    for idx, thr in tqdm(
        enumerate(thresholds),
        total=len(thresholds),
        desc="PR/F1 global sweep",
        unit="thr",
        file=tqdm_progress_stream(),
    ):
        total_tp = total_fp = total_fn = 0
        for image_id, dets in all_detections.items():
            gts = all_ground_truths.get(image_id, [])
            dets_thr = [d for d in dets if float(d.score) >= float(thr)]
            counts = _match_counts(dets_thr, gts, iou_threshold=pr_iou)
            total_tp += counts["tp"]
            total_fp += counts["fp"]
            total_fn += counts["fn"]

        precision = _safe_div(total_tp, total_tp + total_fp)
        recall = _safe_div(total_tp, total_tp + total_fn)
        f1 = _compute_fbeta(precision, recall, beta=1.0)
        f2 = _compute_fbeta(precision, recall, beta=2.0)

        curve_points.append(
            {
                "threshold": float(thr),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "f2": f2,
                "tp": int(total_tp),
                "fp": int(total_fp),
                "fn": int(total_fn),
            }
        )
        key = (f1, recall, -float(thr))
        if key > best_key:
            best_key = key
            best_idx = idx

    if best_idx < 0:
        best_idx = 0
    best = curve_points[best_idx]

    best_threshold_per_class: Dict[str, Any] = {}
    if per_class_threshold_analysis and class_names:
        print(
            f"Per-class best threshold (max F1): {len(class_names)} classes × {len(thresholds)} thresholds",
            flush=True,
        )
        for cname in tqdm(
            class_names,
            desc="Per-class best thr (F1)",
            unit="class",
            file=tqdm_progress_stream(),
        ):
            best_f1_c = -1.0
            best_entry: Optional[Dict[str, Any]] = None
            c_short = cname if len(cname) <= 36 else cname[:33] + "..."
            for _, thr in tqdm(
                enumerate(thresholds),
                total=len(thresholds),
                leave=False,
                desc=f"  thr [{c_short}]",
                unit="thr",
                file=tqdm_progress_stream(),
            ):
                total_tp = total_fp = total_fn = 0
                for image_id, dets in all_detections.items():
                    gts = all_ground_truths.get(image_id, [])
                    dets_thr = [d for d in dets if d.class_name == cname and float(d.score) >= float(thr)]
                    gts_c = [g for g in gts if g.class_name == cname]
                    counts = _match_counts(dets_thr, gts_c, iou_threshold=pr_iou)
                    total_tp += counts["tp"]
                    total_fp += counts["fp"]
                    total_fn += counts["fn"]
                precision = _safe_div(total_tp, total_tp + total_fp)
                recall = _safe_div(total_tp, total_tp + total_fn)
                f1 = _compute_fbeta(precision, recall, beta=1.0)
                if f1 > best_f1_c:
                    best_f1_c = f1
                    best_entry = {
                        "threshold": float(thr),
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "tp": int(total_tp),
                        "fp": int(total_fp),
                        "fn": int(total_fn),
                    }
            if best_entry is not None:
                best_threshold_per_class[cname] = best_entry

    print(f"Per-image metrics at best global F1 threshold ({best['threshold']:.4f})...", flush=True)
    for image_id, dets in tqdm(
        all_detections.items(),
        desc="Per-image metrics",
        unit="img",
        total=len(all_detections),
        file=tqdm_progress_stream(),
    ):
        gts = all_ground_truths.get(image_id, [])
        dets_best = [d for d in dets if float(d.score) >= float(best["threshold"])]
        counts = _match_counts(dets_best, gts, iou_threshold=pr_iou)
        precision_i, recall_i, f1_i, f2_i = _per_image_precision_recall_f1_f2(counts)
        per_image_metrics.append(
            {
                "image_id": image_id,
                "image_name": image_name_by_id.get(image_id, image_id),
                "threshold": float(best["threshold"]),
                "tp": int(counts["tp"]),
                "fp": int(counts["fp"]),
                "fn": int(counts["fn"]),
                "precision": precision_i,
                "recall": recall_i,
                "f1": f1_i,
                "f2": f2_i,
            }
        )

    print("Computing confusion matrix at best global F1 threshold...", flush=True)
    confusion_matrix = _compute_confusion_matrix(
        detections_by_image=all_detections,
        ground_truths_by_image=all_ground_truths,
        class_names=class_names,
        score_threshold=float(best["threshold"]),
        iou_threshold=pr_iou,
    )

    pr_curve_path = os.path.join(output_dir, "pr_curve.png")
    threshold_metrics_path = os.path.join(output_dir, "threshold_metrics.png")
    print("Saving PR curve and threshold metric plots...", flush=True)
    saved_pr_curve = _plot_pr_curve(curve_points, pr_curve_path)
    saved_threshold_plot = _plot_threshold_metrics(curve_points, threshold_metrics_path, best_idx)
    analysis_path = os.path.join(output_dir, f"analysis_iou{pr_iou:.2f}.json")
    print(f"Writing analysis JSON ({os.path.basename(analysis_path)})...", flush=True)

    analysis: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "iou_threshold": float(pr_iou),
        "threshold_min": float(threshold_min),
        "threshold_max": float(threshold_max),
        "threshold_step": float(threshold_step),
        "score_threshold_global": float(score_threshold),
        "per_class_score_threshold_applied": per_cls_thr,
        "class_aps": dict((diagnostics or {}).get("class_aps") or {}),
        "class_metrics": dict((diagnostics or {}).get("class_metrics") or {}),
        "gt_alignment_metrics": dict((diagnostics or {}).get("gt_alignment_metrics") or {})
        if isinstance((diagnostics or {}).get("gt_alignment_metrics"), dict)
        else None,
        "best_threshold": {
            "threshold": float(best["threshold"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "f1": float(best["f1"]),
            "f2": float(best["f2"]),
            "tp": int(best["tp"]),
            "fp": int(best["fp"]),
            "fn": int(best["fn"]),
        },
        "best_threshold_per_class": best_threshold_per_class,
        "global_criterion": "f1",
        "per_class_criterion": "f1",
        "confusion_matrix": confusion_matrix,
        "pr_curve": curve_points,
        "per_image_metrics": per_image_metrics,
        "metadata": {
            "experiment_dir": experiment_dir,
            "checkpoint": checkpoint_path,
            "config_file": config_path,
            "data_root": str(data_root),
            "data_split": data_split,
            "total_images": len(results),
        },
        "artifacts": {
            "analysis_json": os.path.basename(analysis_path),
            "pr_curve": os.path.basename(saved_pr_curve) if saved_pr_curve else None,
            "threshold_metrics": os.path.basename(saved_threshold_plot) if saved_threshold_plot else None,
        },
    }
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)

    if tile_metrics_csv and per_image_metrics:
        print(f"Writing tile metrics CSV ({tile_metrics_csv})...", flush=True)
        csv_out = tile_metrics_csv
        if not os.path.isabs(csv_out):
            csv_out = os.path.join(output_dir, csv_out)
        parent = os.path.dirname(csv_out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fieldnames = list(per_image_metrics[0].keys())
        with open(csv_out, "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_image_metrics:
                writer.writerow(row)
        analysis["artifacts"]["tile_metrics_csv"] = os.path.basename(csv_out)

    print("Writing model analysis markdown report...", flush=True)
    report_path = _write_model_analysis_md(
        output_dir=output_dir,
        metadata={
            "experiment_dir": experiment_dir,
            "checkpoint": checkpoint_path,
            "config_file": config_path,
            "data_root": str(data_root),
            "data_split": data_split,
            "total_images": len(results),
            "total_predictions": sum(r['num_pred'] for r in results),
            "total_ground_truth": sum(r['num_gt'] for r in results),
        },
        diagnostics=diagnostics,
        analysis=analysis,
        artifacts={
            "analysis_json": analysis_path,
            "pr_curve": saved_pr_curve,
            "threshold_metrics": saved_threshold_plot,
        },
    )
    analysis["artifacts"]["model_analysis_md"] = os.path.basename(report_path)

    return diagnostics, analysis


def run_inference_and_save(experiment_dir: str, checkpoint_path: str, config_path: str,
                           data_root: Optional[str], output_dir: str = None,
                           save_visualizations: bool = False,
                           data_split: str = 'val',
                           val_dir: Optional[str] = None,
                           overlap_ratio: Optional[float] = None,
                           overlap_pixels: Optional[int] = None,
                           vis_score_threshold: float = 0.5,
                           score_threshold: Optional[float] = None,
                           per_class_score_threshold: Optional[Dict[str, float]] = None,
                           ignore_config_per_class_score_threshold: bool = False,
                           nms_threshold: Optional[float] = None,
                           nms_class_agnostic: Optional[bool] = None,
                           inference_pre_nms_score_threshold: Optional[float] = None,
                           rpn_pre_nms_top_n: Optional[int] = None,
                           rpn_post_nms_top_n: Optional[int] = None,
                           rpn_nms_threshold: Optional[float] = None,
                           max_detections_per_image: Optional[int] = None,
                           iou_threshold: Optional[float] = None,
                           metrics_margin_pixels: Optional[int] = None,
                           run_diagnostics: bool = True,
                           debug_coords: bool = False,
                           pr_threshold_min: float = 0.0,
                           pr_threshold_max: float = 1.0,
                           pr_threshold_step: float = 0.05,
                           pr_iou_threshold: Optional[float] = None,
                           per_class_threshold_analysis: bool = False,
                           tile_metrics_csv: Optional[str] = None) -> Dict[str, Any]:
    """
    Run inference on dataset, save ALL predictions, and run diagnostics (per-image stats + mAP).

    Args:
        experiment_dir: path to experiment directory
        checkpoint_path: path to checkpoint file
        config_path: path to config.json file
        data_root: path to data root directory (optional if config has dataset.data_root)
        output_dir: output directory (auto-generated if None)
        save_visualizations: whether to save visualization images (default: False)
        data_split: which data split to use ('train', 'val', or 'test') (default: 'val')
        vis_score_threshold: score threshold for visualization (default: 0.5).
        score_threshold: score threshold for diagnostics and mAP. If None, uses
            ``effective_eval_metric_thresholds`` (``production.*`` overrides ``evaluation.*`` when set).
        overlap_pixels: sliding-window overlap per axis. If None (and ``overlap_ratio`` is None),
            uses ``resolve_inference_sliding_window_overlap_pixels(config)`` (production.overlap_pixels
            or default 200).
        nms_threshold: IoU threshold for NMS in diagnostics. If None, uses
                       config.model.final_nms_iou_threshold.
        nms_class_agnostic: if set, override model.nms_class_agnostic at inference time.
        iou_threshold: Rotated IoU threshold for mAP / PR matching. If None, uses
                       evaluation.iou_threshold from config (default 0.5).
        metrics_margin_pixels: Edge-margin filter for metrics only. GT/detections whose
            centroid lies in the outer band ([0, margin) or (W-margin, W] per axis) are
            discarded; interior [margin, W-margin] is kept (same rule as deploy production.ignore_margin_pixels).
            If None, uses ``production.ignore_margin_pixels`` when set, else overlap/2.
        run_diagnostics: if True, compute per-image stats and mAP (default: True)

    Returns:
        Dictionary with metadata about the run (includes 'diagnostics' when run_diagnostics=True)
    """
    def _ts() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    t_run_start = time.perf_counter()
    print(f"[{_ts()}] run_inference_and_save: starting", flush=True)

    # Create output directory with timestamp
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join('predictions', timestamp)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if save_visualizations:
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
    
    print(f"[{_ts()}] Output directory: {output_dir}", flush=True)
    
    # Get device
    from oriented_det.utils import get_device
    device = str(get_device())
    print(f"[{_ts()}] Using device: {device}", flush=True)
    
    # Load model
    print(f"[{_ts()}] Loading model...", flush=True)
    t_load0 = time.perf_counter()
    resolved_config_path = resolve_inference_config_path(checkpoint_path, config_path)
    if resolved_config_path != Path(config_path):
        config_path = str(resolved_config_path)
        print(f"[{_ts()}] Using pretrained sidecar config: {config_path}", flush=True)
    model, config, class_names = load_model_from_checkpoint(checkpoint_path, config_path, device)
    print(
        f"[{_ts()}] Model loaded in {time.perf_counter() - t_load0:.1f}s",
        flush=True,
    )
    if overlap_ratio is None and overlap_pixels is None:
        overlap_pixels = resolve_inference_sliding_window_overlap_pixels(config)
        print(
            f"[{_ts()}] Resolved sliding-window overlap: {overlap_pixels}px/axis "
            "(production.overlap_pixels when set, else 200)",
            flush=True,
        )
    if inference_pre_nms_score_threshold is not None:
        if float(inference_pre_nms_score_threshold) < 0.0:
            raise ValueError("--inference-pre-nms-score-threshold must be >= 0 when set")
        if not hasattr(model, "inference_pre_nms_score_threshold"):
            raise AttributeError(
                f"Loaded model type {type(model).__name__} does not expose inference_pre_nms_score_threshold"
            )
        setattr(model, "inference_pre_nms_score_threshold", float(inference_pre_nms_score_threshold))
        print(
            f"Using CLI inference_pre_nms_score_threshold={float(inference_pre_nms_score_threshold)} "
            "(overrides config model.inference_pre_nms_score_threshold)"
        )
    if rpn_pre_nms_top_n is not None:
        if int(rpn_pre_nms_top_n) <= 0:
            raise ValueError("--rpn-pre-nms-top-n must be positive when set")
        if not hasattr(model, "rpn_pre_nms_top_n"):
            raise AttributeError(f"Loaded model type {type(model).__name__} does not expose rpn_pre_nms_top_n")
        setattr(model, "rpn_pre_nms_top_n", int(rpn_pre_nms_top_n))
        print(
            f"Using CLI rpn_pre_nms_top_n={int(rpn_pre_nms_top_n)} "
            "(overrides config model.rpn_pre_nms_top_n)"
        )
    if rpn_post_nms_top_n is not None:
        if int(rpn_post_nms_top_n) <= 0:
            raise ValueError("--rpn-post-nms-top-n must be positive when set")
        if not hasattr(model, "rpn_post_nms_top_n"):
            raise AttributeError(f"Loaded model type {type(model).__name__} does not expose rpn_post_nms_top_n")
        setattr(model, "rpn_post_nms_top_n", int(rpn_post_nms_top_n))
        print(
            f"Using CLI rpn_post_nms_top_n={int(rpn_post_nms_top_n)} "
            "(overrides config model.rpn_post_nms_top_n)"
        )
    if rpn_nms_threshold is not None:
        if not (0.0 <= float(rpn_nms_threshold) <= 1.0):
            raise ValueError("--rpn-nms-threshold must be in [0, 1] when set")
        if not hasattr(model, "rpn_nms_threshold"):
            raise AttributeError(f"Loaded model type {type(model).__name__} does not expose rpn_nms_threshold")
        setattr(model, "rpn_nms_threshold", float(rpn_nms_threshold))
        print(
            f"Using CLI rpn_nms_threshold={float(rpn_nms_threshold)} "
            "(overrides config model.rpn_nms_threshold)"
        )
    if max_detections_per_image is not None:
        if max_detections_per_image <= 0:
            raise ValueError("--max-detections-per-image must be positive when set")
        if not hasattr(model, "max_detections_per_image"):
            raise AttributeError(
                f"Loaded model type {type(model).__name__} does not expose max_detections_per_image"
            )
        setattr(model, "max_detections_per_image", int(max_detections_per_image))
        print(
            f"Using CLI max_detections_per_image={int(max_detections_per_image)} "
            "(overrides config model.max_detections_per_image)"
        )
    cfg_thr_sc, cfg_thr_pc, cfg_thr_iou = effective_eval_metric_thresholds(config)

    per_cls_thr: Optional[Dict[str, float]] = per_class_score_threshold
    if score_threshold is not None:
        print(f"Using score_threshold={score_threshold} from CLI (for mAP and diagnostics; overrides config)")
    else:
        score_threshold = cfg_thr_sc
        print(
            "Using score_threshold from config (production.score_threshold overrides "
            f"evaluation.score_threshold when set): {score_threshold}"
        )
    if ignore_config_per_class_score_threshold:
        if per_cls_thr is not None:
            print("Ignoring config per-class score thresholds; explicit per-class thresholds were also suppressed")
        elif cfg_thr_pc:
            print("Ignoring merged per-class score thresholds from config (evaluation + production)")
        per_cls_thr = None
    elif per_cls_thr is None:
        per_cls_thr = cfg_thr_pc
    if per_cls_thr:
        print(f"Using per-class score thresholds for {len(per_cls_thr)} class(es) (post-NMS filtering)")
    model_nms = (
        float(model.final_nms_iou_threshold)
        if hasattr(model, "final_nms_iou_threshold")
        else None
    )
    nms_threshold, nms_source = resolve_preds_final_nms_iou_threshold(
        config,
        cli_nms_threshold=nms_threshold,
        model_nms_threshold=model_nms,
    )
    if hasattr(model, "final_nms_iou_threshold"):
        setattr(model, "final_nms_iou_threshold", float(nms_threshold))
    print(f"Using final_nms_iou_threshold={nms_threshold} from {nms_source}")
    if nms_class_agnostic is not None:
        if not hasattr(model, "nms_class_agnostic"):
            raise AttributeError(
                f"Loaded model type {type(model).__name__} does not expose nms_class_agnostic"
            )
        setattr(model, "nms_class_agnostic", bool(nms_class_agnostic))
        print(
            f"Using CLI nms_class_agnostic={bool(nms_class_agnostic)} "
            "(overrides config model.nms_class_agnostic)"
        )
    if iou_threshold is None:
        iou_threshold = cfg_thr_iou
        print(
            f"Using iou_threshold from config (evaluation.iou_threshold): {iou_threshold} (mAP / PR matching)"
        )
    else:
        print(f"Using CLI iou_threshold={iou_threshold} (mAP / PR matching)")
    preprocessing = get_preprocessing_params(config)
    print(f"Preprocessing from config: resize_mode={preprocessing['resize_mode']}, target_size={preprocessing['target_size']}")
    if overlap_ratio is not None:
        print(f"Sliding-window overlap: ratio={overlap_ratio} (overrides --overlap-pixels)")
    else:
        opx = overlap_pixels if overlap_pixels is not None else resolve_inference_sliding_window_overlap_pixels(config)
        print(f"Sliding-window overlap: {opx} px per axis (stride = tile size − overlap)")
    margin_for_metrics = metrics_margin_pixels
    if margin_for_metrics is None:
        inf_m = getattr(getattr(config, "production", None), "ignore_margin_pixels", None)
        if inf_m is not None:
            margin_for_metrics = max(0, int(round(float(inf_m))))
            print(
                f"Using metrics_margin_pixels={margin_for_metrics} from production.ignore_margin_pixels",
                flush=True,
            )
    resolved_overlap_px = (
        overlap_pixels
        if overlap_pixels is not None
        else resolve_inference_sliding_window_overlap_pixels(config)
    )
    resolved_metrics_margin_px = _resolve_metrics_margin_pixels(
        margin_pixels=margin_for_metrics,
        overlap_ratio=overlap_ratio,
        overlap_pixels=resolved_overlap_px,
        preprocessing=preprocessing,
    )
    print(
        f"Metrics edge margin: {resolved_metrics_margin_px}px "
        "(GT/detections with centroid in outer [0,m) or (W-m,W] band are excluded from metrics)"
    )

    # Resolve data_root from config if not provided
    if not data_root and getattr(config, 'dataset', None) and getattr(config.dataset, 'data_root', None):
        data_root = str(config.dataset.data_root)
        print(f"Using data_root from config: {data_root}")
    if not data_root:
        raise ValueError("data_root is required. Provide --data-root or use a config that has dataset.data_root.")
    data_root = Path(data_root)
    
    dataset_format = dataset_format_name(getattr(config, "dataset", None))
    gt_by_image_path = None  # For Airbus / HRSC: path -> list of GroundTruth; for DOTA stays None
    label_dir = None
    same_folder = False

    if dataset_format in ("airbus_playground", "hrsc2016"):
        from dataclasses import replace

        ds_config = config.dataset
        if dataset_format == "airbus_playground":
            if not getattr(ds_config, 'annotations_file', None) or not getattr(ds_config, 'split_file', None):
                raise ValueError(
                    "Airbus Playground format requires dataset.annotations_file and dataset.split_file in config."
                )
            if data_split not in ("train", "val"):
                raise ValueError(f"Airbus dataset supports only 'train' or 'val' split, got '{data_split}'.")
        ds_cfg = replace(ds_config, data_root=data_root)
        native_dataset = build_split_dataset(ds_cfg, data_split, filter_empty_gt=False)
        class_map = {name: i for i, name in enumerate(class_names)} if class_names else {}
        split_images = []
        gt_by_image_path = {}
        for idx in range(len(native_dataset)):
            sample = native_dataset[idx]
            split_images.append(Path(sample.image_path))
            gt_by_image_path[Path(sample.image_path)] = _annotations_to_ground_truths(
                list(sample.annotations), class_map
            )
        if dataset_format == "airbus_playground":
            val_split_id = getattr(ds_config, "val_split_id", 0)
            print(
                f"Using Airbus Playground CSV dataset: {ds_config.annotations_file}, "
                f"{ds_config.split_file} (val_split_id={val_split_id})"
            )
        else:
            print(
                f"Using HRSC2016 dataset: {data_root} "
                f"(ImageSets split={getattr(native_dataset, 'split', data_split)})"
            )
        print(f"\nFound {len(split_images)} {data_split} images\n")
    else:
        if data_split == "val" and val_dir:
            print(f"Using override val dir: {val_dir}")
        split_images, label_dir, _ = collect_split_images(
            config,
            data_root,
            data_split=data_split,
            val_dir=Path(val_dir) if val_dir else None,
            filter_empty_gt=False,
        )
        same_folder = getattr(getattr(config, "dataset", None), "same_folder", False)
        print(f"\nFound {len(split_images)} {data_split} images\n")
    
    class_map = {name: i for i, name in enumerate(class_names)} if class_names else {}
    all_detections = {}
    all_ground_truths = {}
    all_scores = []
    image_name_by_id: Dict[str, str] = {}
    sliding_window_positions_total: Optional[int] = None
    window_batch_effective: Optional[int] = None

    # Full-val: cost is ~sum of sliding windows per image, not ~num images (unlike pre-tiled 1024 eval).
    if dataset_format != "airbus_playground" and len(split_images) > 0:
        from PIL import Image as PILImage

        # Large DOTA / satellite rasters exceed PIL's default pixel cap (same as tools/tile_dota.py).
        PILImage.MAX_IMAGE_PIXELS = None

        wh_list: List[Tuple[int, int]] = []
        for p in split_images:
            with PILImage.open(p) as im:
                w, h = im.size
            wh_list.append((h, w))
        per_image_wins = [
            count_sliding_window_positions(
                h, w, preprocessing, overlap_ratio=overlap_ratio, overlap_pixels=overlap_pixels,
            )
            for h, w in wh_list
        ]
        sliding_window_positions_total = int(sum(per_image_wins))
        max_wins_one = max(per_image_wins) if per_image_wins else 0
        # No sliding-window micro-batch is used when every image fits in one tile (inference is batch-1
        # run_inference_raw per image). Skip the GPU probe in that case — it only stresses VRAM/cuDNN
        # and does not reflect real work.
        if max_wins_one > 1:
            window_batch_effective = resolve_window_batch_size(model, device, preprocessing)
        else:
            window_batch_effective = 1
        nb = max(1, (sliding_window_positions_total + window_batch_effective - 1) // window_batch_effective)
        print(
            f"[{_ts()}] Full-image sliding-window estimate: {len(split_images)} images → "
            f"{sliding_window_positions_total} window positions (max {max_wins_one} on one image); "
            f"~{nb} model batch forwards (window batch = {window_batch_effective}; "
            f"env ORIENTED_DET_WINDOW_BATCH_SIZE or --window-batch-size to override). "
            f"Compare window count/stride to the other eval (inference time scales with this, not 593).",
            flush=True,
        )
    
    # Process all images
    results = []
    print(
        f"[{_ts()}] Running inference on {len(split_images)} image(s) (slowest phase for full-val)…",
        flush=True,
    )
    t_infer0 = time.perf_counter()
    for img_path in tqdm(
        split_images,
        desc=f"Processing {data_split} images",
        file=tqdm_progress_stream(),
    ):
        img_name = img_path.name
        image_id = img_path.stem
        image_name_by_id[image_id] = img_name
        txt_path = (
            (label_dir / f"{img_path.stem}.txt")
            if label_dir is not None
            else dota_label_path_for_image(img_path, same_folder=same_folder)
        )

        img = cv2.imread(str(img_path))
        img_height, img_width = (img.shape[:2] if img is not None else (0, 0))
        # Ground truth: from Airbus CSV (gt_by_image_path) or from DOTA label file (txt_path)
        if gt_by_image_path is not None:
            gt_list_for_image = gt_by_image_path.get(img_path, [])
            num_gt = len(gt_list_for_image)
            gt_entries = [
                {
                    "bbox": rbox_to_array(gt.rbox).tolist(),
                    "class_name": gt.class_name,
                    "class_id": int(gt.class_id),
                    "difficult": int(getattr(gt, "difficult", 0)),
                }
                for gt in gt_list_for_image
            ]
        else:
            try:
                if txt_path and txt_path.exists():
                    gt_rboxes, gt_class_names = load_dota_annotations(str(txt_path))
                else:
                    gt_rboxes = np.array([]).reshape(0, 5)
                    gt_class_names = []
                num_gt = len(gt_rboxes)
                gt_entries = [
                    {
                        "bbox": gt_rboxes[i].tolist(),
                        "class_name": gt_class_names[i] if i < len(gt_class_names) else "unknown",
                        "class_id": int(class_map.get(gt_class_names[i], -1)) if i < len(gt_class_names) else -1,
                        "difficult": 0,
                    }
                    for i in range(len(gt_rboxes))
                ]
            except Exception as e:
                print(f"Warning: Could not load GT for {img_name}: {e}")
                gt_rboxes = np.array([]).reshape(0, 5)
                gt_class_names = []
                num_gt = 0
                gt_entries = []

        # run_inference_auto: pad = whole-image train preprocess; fixed/crop tile if larger than canvas
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
            if img_rgb is None:
                raise ValueError(f"Could not load image {img_path}")
            detections, raw_output = run_inference_auto(
                image=img_rgb,
                model=model,
                device=device,
                preprocessing=preprocessing,
                score_threshold=score_threshold,
                nms_threshold=nms_threshold,
                overlap_ratio=overlap_ratio,
                overlap_pixels=overlap_pixels,
                window_batch_size=window_batch_effective,
                return_raw_output=True,
                per_class_score_threshold=per_cls_thr,
                class_names=class_names,
            )
        except Exception as e:
            print(f"Warning: Windowed inference failed for {img_name}: {e}")
            detections = []
            raw_output = None
        rboxes = [d["rbox"] for d in detections]
        pred_scores_list = [d["score"] for d in detections]
        pred_labels_list = [d["label"] for d in detections]
        if raw_output is None:
            raw_output = {
                "rboxes": rboxes,
                "scores": torch.tensor(pred_scores_list) if pred_scores_list else torch.tensor([]),
                "labels": torch.tensor(pred_labels_list, dtype=torch.int64) if pred_labels_list else torch.tensor([], dtype=torch.int64),
            }
        ts = preprocessing.get("target_size", [1024, 1024])
        th, tw = (int(ts[0]), int(ts[1])) if isinstance(ts, (list, tuple)) else (1024, 1024)
        used_sliding = not (img_height <= th and img_width <= tw)
        per_image_stats = (
            {
                "sliding_window": used_sliding,
                "num_after_nms": len(detections),
            }
            if run_diagnostics
            else {}
        )
        final_detections = [
            Detection(
                rbox=normalize_le90(d["rbox"]),
                score=d["score"],
                class_id=d["label"],
                class_name=class_names[d["label"] - 1] if class_names and 1 <= d["label"] <= len(class_names) else f"class_{d['label']}",
                image_id=image_id,
            )
            for d in detections
        ]

        all_scores.extend(pred_scores_list)

        # Diagnostics: attach detections for mAP
        if run_diagnostics:
            detections_for_metrics = [
                Detection(rbox=normalize_le90(d.rbox), score=d.score, class_id=d.class_id, class_name=d.class_name, image_id=image_id)
                for d in final_detections
            ]
            gt_for_metrics: List[GroundTruth]
            if gt_by_image_path is not None:
                gt_for_metrics = [
                    GroundTruth(rbox=gt.rbox, class_id=gt.class_id, class_name=gt.class_name, difficult=gt.difficult, image_id=image_id)
                    for gt in gt_list_for_image
                ]
            else:
                gt_list_raw = load_gt_as_ground_truths(txt_path, class_map)
                gt_for_metrics = [
                    GroundTruth(rbox=gt.rbox, class_id=gt.class_id, class_name=gt.class_name, difficult=gt.difficult, image_id=image_id)
                    for gt in gt_list_raw
                ]

            if resolved_metrics_margin_px > 0 and img_width > 0 and img_height > 0:
                detections_filtered = [
                    d for d in detections_for_metrics
                    if _rbox_centroid_in_tile_interior(
                        d.rbox, img_width, img_height, resolved_metrics_margin_px
                    )
                ]
                gt_filtered = [
                    g for g in gt_for_metrics
                    if _rbox_centroid_in_tile_interior(
                        g.rbox, img_width, img_height, resolved_metrics_margin_px
                    )
                ]
                if run_diagnostics and per_image_stats is not None:
                    per_image_stats["num_after_metrics_margin"] = int(len(detections_filtered))
                    per_image_stats["num_gt_after_metrics_margin"] = int(len(gt_filtered))
                all_detections[image_id] = detections_filtered
                all_ground_truths[image_id] = gt_filtered
            else:
                all_detections[image_id] = detections_for_metrics
                all_ground_truths[image_id] = gt_for_metrics

        # Convert RBox objects to array format for JSON (save ALL predictions)
        pred_boxes_array = [rbox_to_array(rbox).tolist() for rbox in rboxes]

        image_result = {
            'image_name': img_name,
            'image_path': os.path.relpath(img_path, data_root),
            'image_width': int(img_width),
            'image_height': int(img_height),
            'resize_mode': preprocessing.get('resize_mode', 'fixed'),
            'target_size': preprocessing.get('target_size', [1024, 1024]),
            'num_gt': int(num_gt),
            'num_pred': int(len(rboxes)),
            'predictions': [
                {
                    'bbox': pred_boxes_array[i],
                    'score': float(pred_scores_list[i]),
                    'label': int(pred_labels_list[i]),
                    'class_name': (class_names[pred_labels_list[i] - 1] if class_names and 1 <= pred_labels_list[i] <= len(class_names) else f'class_{pred_labels_list[i]}')
                }
                for i in range(len(rboxes))
            ],
            'ground_truths': gt_entries,
        }
        if run_diagnostics and per_image_stats:
            image_result['stats'] = per_image_stats
        results.append(image_result)
        
        # Visualization: use raw predictions with vis threshold
        if save_visualizations and img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            scores_np = np.array(pred_scores_list) if pred_scores_list else np.array([])
            labels_np = np.array(pred_labels_list) if pred_labels_list else np.array([], dtype=np.int64)
            img_with_boxes = draw_rotated_boxes(
                img_rgb, rboxes, scores_np, labels_np,
                score_thr=vis_score_threshold, class_names=class_names
            )
            vis_path = os.path.join(vis_dir, img_name)
            cv2.imwrite(vis_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
    
    t_infer_sec = time.perf_counter() - t_infer0
    print(
        f"[{_ts()}] Inference loop finished in {t_infer_sec:.1f}s "
        f"({len(split_images) / max(t_infer_sec, 1e-6):.2f} images/s).",
        flush=True,
    )
    
    # Optional: debug first-image det vs GT coordinate alignment and max IoU
    if debug_coords and run_diagnostics and all_detections and all_ground_truths:
        from oriented_det.ops import iou as iou_ops
        import math
        first_id = next(iter(all_detections.keys()))
        dets_first = all_detections.get(first_id, [])
        gts_first = all_ground_truths.get(first_id, [])
        first_result = next((r for r in results if Path(r.get("image_name", "")).stem == first_id), next(iter(results), {}))
        w, h = first_result.get("image_width", 0), first_result.get("image_height", 0)
        print("\n--- Debug coords (first image) ---")
        print(f"  image_id: {first_id}, size: {w} x {h}")
        print(f"  num detections (after threshold): {len(dets_first)}, num GTs: {len(gts_first)}")
        for i, d in enumerate(dets_first[:3]):
            r = d.rbox
            print(f"  det[{i}] cx={r.cx:.1f} cy={r.cy:.1f} w={r.width:.1f} h={r.height:.1f} angle_deg={math.degrees(r.angle):.1f}")
        for i, gt in enumerate(gts_first[:3]):
            r = gt.rbox
            print(f"  gt[{i}]  cx={r.cx:.1f} cy={r.cy:.1f} w={r.width:.1f} h={r.height:.1f} angle_deg={math.degrees(r.angle):.1f}")
        max_iou = 0.0
        for d in dets_first[:50]:
            for gt in gts_first:
                try:
                    v = iou_ops.rbox_iou(d.rbox, gt.rbox)
                    if v > max_iou:
                        max_iou = v
                except Exception:
                    pass
        print(f"  max IoU (first 50 dets vs all GTs): {max_iou:.4f}")

    # Aggregated diagnostics and mAP
    diagnostics = None
    analysis = None
    if run_diagnostics:
        diagnostics, analysis = run_diagnostics_pipeline(
            experiment_dir=experiment_dir,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            data_root=data_root,
            data_split=data_split,
            class_names=class_names,
            score_threshold=float(score_threshold),
            per_cls_thr=per_cls_thr,
            nms_threshold=float(nms_threshold),
            nms_class_agnostic=bool(getattr(model, "nms_class_agnostic", False)),
            iou_threshold=float(iou_threshold),
            pr_iou_threshold=pr_iou_threshold,
            pr_threshold_min=pr_threshold_min,
            pr_threshold_max=pr_threshold_max,
            pr_threshold_step=pr_threshold_step,
            per_class_threshold_analysis=per_class_threshold_analysis,
            resolved_metrics_margin_px=int(resolved_metrics_margin_px),
            all_detections=all_detections,
            all_ground_truths=all_ground_truths,
            results=results,
            all_scores=all_scores,
            image_name_by_id=image_name_by_id,
            sliding_window_positions_total=sliding_window_positions_total,
            window_batch_effective=window_batch_effective,
            t_infer_sec=float(t_infer_sec),
            device=device,
            output_dir=output_dir,
            tile_metrics_csv=tile_metrics_csv,
            use_exact_rotated_iou=True,
        )

    # Prepare metadata
    print(
        f"[{_ts()}] Assembling metadata and writing predictions.json (inference {t_infer_sec:.1f}s so far)…",
        flush=True,
    )
    t_at_json = time.perf_counter() - t_run_start
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'experiment_dir': experiment_dir,
        'checkpoint': checkpoint_path,
        'config_file': config_path,
        'data_root': str(data_root),
        'data_split': data_split,
        'filter_empty_gt': False,
        'filter_empty_gt_training_config': bool(
            getattr(getattr(config, "dataset", None), "filter_empty_gt", False)
        ),
        'device': device,
        'class_names': class_names,
        'score_threshold': score_threshold,
        'per_class_score_threshold': per_cls_thr,
        'nms_class_agnostic': bool(getattr(model, "nms_class_agnostic", False)),
        'total_images': len(results),
        'total_predictions': sum(r['num_pred'] for r in results),
        'total_ground_truth': sum(r['num_gt'] for r in results),
        'inference_loop_seconds': float(t_infer_sec),
        'run_wall_clock_seconds': float(t_at_json),
        # tools/app.py: prediction bboxes are in original image pixels (pad/tile path), not model-input space
        'bbox_coordinate_space': 'image_pixels',
        'metrics_margin_pixels': int(resolved_metrics_margin_px),
    }
    if sliding_window_positions_total is not None:
        metadata['estimated_sliding_window_positions'] = sliding_window_positions_total
    if window_batch_effective is not None:
        metadata['oriented_det_window_batch_size'] = window_batch_effective
    if overlap_ratio is not None:
        metadata['sliding_window_overlap_ratio'] = float(overlap_ratio)
    else:
        metadata['sliding_window_overlap_pixels'] = int(resolved_overlap_px)
    if diagnostics is not None:
        metadata['diagnostics'] = diagnostics
    if analysis is not None:
        metadata['best_threshold_f1'] = analysis.get("best_threshold", {})
        metadata['analysis_file'] = analysis.get("artifacts", {}).get("analysis_json", f"analysis_iou{iou_threshold:.2f}.json")
        metadata['pr_iou_threshold'] = analysis.get("iou_threshold", iou_threshold)
    
    # Save results to JSON
    output_data = {
        'metadata': metadata,
        'results': results
    }
    
    json_path = os.path.join(output_dir, 'predictions.json')
    print(f"Writing predictions.json ({len(results)} images)...", flush=True)
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE")
    print("="*60)
    print(f"Total images processed: {len(results)}")
    print(f"Total predictions: {metadata['total_predictions']}")
    print(f"Total ground truth objects: {metadata['total_ground_truth']}")
    if diagnostics is not None:
        print(
            f"\n--- Diagnostics (score_thr={score_threshold}, "
            f"NMS IoU={float(nms_threshold):.2f}, mAP/PR match IoU={float(iou_threshold):.2f}) ---"
        )
        if diagnostics.get("sliding_window_eval"):
            print("  Sliding-window: raw→threshold→NMS totals below count only single-crop images (not sliding); ignore that pipeline line.")
            print(f"  Total predictions (per image, post window-merge + NMS): {diagnostics.get('total_predictions', diagnostics['total_after_nms'])}")
        else:
            print(f"  Raw → after threshold → after NMS: {diagnostics['total_raw']} → {diagnostics['total_after_threshold']} → {diagnostics['total_after_nms']}")
        mm_px = int(diagnostics.get("metrics_margin_pixels", 0))
        if mm_px > 0:
            p_after = diagnostics.get("total_predictions_after_metrics_margin", "N/A")
            gt_after = diagnostics.get("total_ground_truth_after_metrics_margin", "N/A")
            print(
                f"  Metrics edge margin: {mm_px}px "
                f"(predictions for metrics: {p_after}, GT for metrics: {gt_after})"
            )
        if diagnostics.get('score_min') is not None:
            print(f"  Score distribution: min={diagnostics['score_min']:.4f} max={diagnostics['score_max']:.4f} mean={diagnostics['score_mean']:.4f}")
        if diagnostics.get('mAP') is not None:
            _ml = _map_metric_label(float(iou_threshold))
            print(
                f"  {_ml}: {diagnostics['mAP']:.4f} ({diagnostics['mAP'] * 100:.2f}%)  "
                f"(matching IoU ≥ {float(iou_threshold):.2f}; not NMS)"
            )
        else:
            print(f"  mAP: {diagnostics.get('mAP_error', 'N/A')}")
        align_dict = diagnostics.get("gt_alignment_metrics")
        if isinstance(align_dict, dict) and align_dict.get("mean_best_iou_same_class") is not None:
            print(
                f"  GT alignment mean best IoU (same class): "
                f"{float(align_dict['mean_best_iou_same_class']):.4f} "
                f"(median {float(align_dict.get('median_best_iou_same_class', 0.0)):.4f}; "
                "see per-class table in model_analysis / analysis JSON)"
            )
        if diagnostics.get('sample_max_iou') is not None:
            nchk = diagnostics.get('sample_max_iou_num_checked', "")
            suf = f" (random sample of {nchk} dets)" if nchk else ""
            print(f"  Sample max IoU (det vs GT, same class/image){suf}: {diagnostics['sample_max_iou']:.4f}")
            if diagnostics.get("sample_max_iou", 0) < float(iou_threshold):
                print(
                    f"  Note: sample max IoU < mAP matching threshold ({float(iou_threshold):.2f}); "
                    f"{_map_metric_label(float(iou_threshold))} can be 0. Try --iou-threshold 0.25 to sanity-check overlap."
                )
    if analysis is not None:
        best = analysis.get("best_threshold", {})
        print("\n--- Precision/Recall/F1 analysis ---")
        print(f"  IoU threshold (PR matching): {analysis.get('iou_threshold', iou_threshold):.2f}")
        print(f"  Best threshold (F1): {best.get('threshold', 0.0):.4f}")
        print(f"  Precision={best.get('precision', 0.0):.4f} Recall={best.get('recall', 0.0):.4f} F1={best.get('f1', 0.0):.4f} F2={best.get('f2', 0.0):.4f}")
        best_per_class = analysis.get("best_threshold_per_class") or {}
        if best_per_class:
            print("\n  Per-class best threshold (F1):")
            print("  " + "-" * 62)
            print(f"  {'Class':<32} {'Thr':>8} {'F1':>8} {'P':>8} {'R':>8}")
            print("  " + "-" * 62)
            ordered_classes = class_names if class_names else sorted(best_per_class.keys())
            for cname in ordered_classes:
                row = best_per_class.get(cname)
                if not row:
                    continue
                thr = float(row.get("threshold", 0.0))
                f1v = float(row.get("f1", 0.0))
                prv = float(row.get("precision", 0.0))
                rcv = float(row.get("recall", 0.0))
                cname_short = cname if len(cname) <= 32 else (cname[:29] + "...")
                print(f"  {cname_short:<32} {thr:>8.4f} {f1v:>8.4f} {prv:>8.4f} {rcv:>8.4f}")
            print("  " + "-" * 62)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - predictions.json (ALL predictions + per-image stats + diagnostics)")
    if analysis is not None:
        print(f"  - {analysis.get('artifacts', {}).get('analysis_json', f'analysis_iou{iou_threshold:.2f}.json')} (PR curve + best-threshold metrics)")
        print(f"  - pr_curve.png")
        print(f"  - threshold_metrics.png")
        print(f"  - model_analysis_<timestamp>.md")
    if save_visualizations:
        print(f"  - visualizations/ ({len(results)} images)")
    else:
        print(f"\nNote: Visualizations not saved (use --save-visualizations to enable)")

    t_end = time.perf_counter() - t_run_start
    print(
        f"[{_ts()}] run_inference_and_save: finished (total wall {t_end:.1f}s, inference {t_infer_sec:.1f}s).",
        flush=True,
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Run GPU inference to predictions.json; optionally compute mAP/PR (--no-diagnostics to skip) '
                    'or recompute metrics from an existing JSON (--metrics-from-json).'
    )
    parser.add_argument('--model-type', type=str, default='rotated_faster_rcnn',
                       help='Model type (used for auto-detection). Default: rotated_faster_rcnn')
    parser.add_argument('--experiment-dir', type=str, default=None,
                       help='Path to experiment directory (auto-detected if not provided)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (auto-detected if not provided)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.json file')
    parser.add_argument('--data-root', type=str, default=None,
                       help='Path to data root directory (optional if config has dataset.data_root)')
    parser.add_argument('--data-split', type=str, default='val', choices=['train', 'val', 'test'],
                       help='Which data split to use (train, val, or test). Default: val')
    parser.add_argument('--val-dir', type=str, default=None,
                       help='Override validation folder (for non-tiled DOTA val); when set, used instead of config dataset.val_tiles_dir')
    parser.add_argument(
        '--overlap-pixels',
        type=int,
        default=None,
        help='Sliding-window overlap in pixels per axis. When omitted, uses production.overlap_pixels '
             'from the experiment config when set, else 200 (same as tools/tile_dota.py). Ignored if '
             '--overlap-ratio is set.',
    )
    parser.add_argument(
        '--overlap-ratio',
        type=float,
        default=None,
        help='If set, window overlap as fraction of tile size [0,1); overrides --overlap-pixels.',
    )
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (auto-generated if not provided)')
    parser.add_argument('--save-visualizations', action='store_true',
                       help='Save visualization images (disabled by default to save disk space)')
    parser.add_argument('--vis-score-threshold', type=float, default=0.5,
                       help='Score threshold for visualization (default: 0.5).')
    parser.add_argument('--score-threshold', type=float, default=None,
                       help='Score threshold for diagnostics and mAP (overrides evaluation.score_threshold '
                            'in config when set; default: use evaluation.score_threshold or 0.5)')
    parser.add_argument('--no-per-class-score-thresholds', action='store_true',
                       help='Ignore evaluation.per_class_score_threshold from config and use the global score threshold only.')
    parser.add_argument(
        '--nms-threshold',
        type=float,
        default=None,
        help=(
            'Final detection NMS IoU for odet preds / eval-val. '
            'Default: evaluation.final_nms_iou_threshold when set, else production/model '
            '(after production patch), else 0.5. Deploy/image_demo keep production.*.'
        ),
    )
    parser.add_argument(
        '--nms-class-agnostic',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Override model.nms_class_agnostic at inference time '
             '(use --nms-class-agnostic or --no-nms-class-agnostic).',
    )
    parser.add_argument('--inference-pre-nms-score-threshold', type=float, default=None,
                       help='Override model.inference_pre_nms_score_threshold (filters boxes before final NMS).')
    parser.add_argument('--rpn-pre-nms-top-n', type=int, default=None,
                       help='Override model.rpn_pre_nms_top_n (RPN proposals kept before proposal NMS).')
    parser.add_argument('--rpn-post-nms-top-n', type=int, default=None,
                       help='Override model.rpn_post_nms_top_n (RPN proposals kept after proposal NMS).')
    parser.add_argument('--rpn-nms-threshold', type=float, default=None,
                       help='Override model.rpn_nms_threshold (IoU threshold for RPN proposal NMS).')
    parser.add_argument('--max-detections-per-image', type=int, default=None,
                       help='Override model.max_detections_per_image after loading the checkpoint.')
    parser.add_argument('--iou-threshold', type=float, default=None,
                       help='Rotated IoU for mAP / PR matching (default: evaluation.iou_threshold in config, else 0.5)')
    parser.add_argument(
        '--metrics-margin-pixels',
        type=int,
        default=None,
        help='Interior margin in pixels for metrics-only filtering. GT/predictions whose centroid '
             'falls inside [margin, W-margin]x[margin, H-margin] are discarded for metrics. '
             'Default: overlap/2.',
    )
    parser.add_argument('--pr-iou-threshold', type=float, default=None,
                       help='IoU threshold for precision/recall/F1/F2 matching (default: same as --iou-threshold)')
    parser.add_argument('--no-diagnostics', action='store_true',
                       help='Disable per-image stats and mAP (faster run)')
    parser.add_argument('--debug-coords', action='store_true',
                       help='Print first-image det/GT box coords and max IoU to verify coordinate alignment')
    parser.add_argument('--pr-threshold-min', type=float, default=0.0,
                       help='Minimum confidence threshold for PR/F1 sweep (default: 0.0)')
    parser.add_argument('--pr-threshold-max', type=float, default=1.0,
                       help='Maximum confidence threshold for PR/F1 sweep (default: 1.0)')
    parser.add_argument('--pr-threshold-step', type=float, default=0.05,
                       help='Threshold step for PR/F1 sweep (default: 0.05)')
    parser.add_argument('--per-class-threshold-analysis', action='store_true',
                       help='(Deprecated) Kept for backward compatibility; per-class thresholds are computed by default.')
    parser.add_argument('--no-per-class-threshold-analysis', action='store_true',
                       help='Disable per-class best-threshold computation (faster).')
    parser.add_argument('--tile-metrics-csv', type=str, default=None,
                       help='Write per-tile metrics CSV to this path (relative to output_dir if not absolute)')
    parser.add_argument(
        '--metrics-from-json',
        type=str,
        default=None,
        metavar='PATH',
        help='Skip inference: load PATH (predictions.json or its parent directory), rebuild GT/det maps, '
             'run mAP/PR/analysis with CLI thresholds; writes artifacts beside JSON and updates metadata.',
    )

    args = parser.parse_args()

    if args.metrics_from_json:
        from oriented_det.utils import get_device

        mj = args.metrics_from_json.rstrip(os.sep)
        json_path = os.path.join(mj, 'predictions.json') if os.path.isdir(mj) else mj
        if not os.path.isfile(json_path):
            raise SystemExit(f'predictions JSON not found: {json_path}')
        output_dir = args.output_dir or os.path.dirname(os.path.abspath(json_path))
        os.makedirs(output_dir, exist_ok=True)
        with open(json_path, 'r', encoding='utf-8') as f:
            bundle = json.load(f)
        meta = dict(bundle.get('metadata') or {})
        results = list(bundle.get('results') or [])
        class_names = list(meta.get('class_names') or [])

        config = None
        if args.config:
            from oriented_det.utils.config import _framework_config_roots  # local import

            raw = str(args.config)
            cfg_path = Path(raw)
            if not cfg_path.exists():
                rel = raw.replace("\\", "/").lstrip("/")
                if rel.startswith("configs/"):
                    rel = rel[len("configs/") :]
                for fw_root in _framework_config_roots():
                    candidate = (fw_root / rel).resolve()
                    if candidate.exists():
                        cfg_path = candidate
                        break
            config = TrainingExperimentConfig.load(cfg_path)
        else:
            cfg_meta = meta.get("config_file")
            if cfg_meta:
                cfg_path = Path(str(cfg_meta)).expanduser()
                if cfg_path.is_file():
                    config = TrainingExperimentConfig.load(cfg_path)

        mmargin = args.metrics_margin_pixels
        if mmargin is None:
            if config is not None:
                inf_m = getattr(getattr(config, "production", None), "ignore_margin_pixels", None)
                if inf_m is not None:
                    mmargin = int(inf_m)
            if mmargin is None:
                mmargin = int(meta.get('metrics_margin_pixels', 0))

        diag_prev = meta.get('diagnostics') if isinstance(meta.get('diagnostics'), dict) else {}

        cfg_thr_sc, cfg_thr_pc, cfg_thr_iou = (None, None, None)
        if config is not None:
            cfg_thr_sc, cfg_thr_pc, cfg_thr_iou = effective_eval_metric_thresholds(config)

        iou_thr = args.iou_threshold
        if iou_thr is None:
            if cfg_thr_iou is not None:
                iou_thr = cfg_thr_iou
            else:
                iou_thr = diag_prev.get('iou_threshold')
                if iou_thr is None:
                    iou_thr = meta.get('pr_iou_threshold', 0.5)
        iou_thr = float(iou_thr)

        score_thr = args.score_threshold
        if score_thr is None:
            if cfg_thr_sc is not None:
                score_thr = cfg_thr_sc
            else:
                score_thr = float(meta.get('score_threshold', 0.5))

        nms_thr = args.nms_threshold
        if nms_thr is None:
            if config is not None:
                nms_thr, _ = resolve_preds_final_nms_iou_threshold(config)
            else:
                v = diag_prev.get('final_nms_iou_threshold')
                nms_thr = float(v) if v is not None else 0.5

        per_cls = meta.get('per_class_score_threshold')
        if args.no_per_class_score_thresholds:
            per_cls = None
        elif cfg_thr_pc is not None and args.score_threshold is None:
            per_cls = cfg_thr_pc
        elif per_cls is not None and not isinstance(per_cls, dict):
            per_cls = None

        nms_ag = meta.get('nms_class_agnostic')
        if nms_ag is None:
            nms_ag = bool(diag_prev.get('nms_class_agnostic', False))

        all_detections, all_ground_truths, image_name_by_id, all_scores = _metric_maps_from_saved_results(
            results, int(mmargin), class_names,
        )
        device = str(get_device())
        use_exact_map = (
            bool(getattr(config.evaluation, "use_exact_rotated_iou", True))
            if config is not None
            else True
        )
        t_infer = float(meta.get('inference_loop_seconds', 0.0))
        sw_est = meta.get('estimated_sliding_window_positions')
        wb = meta.get('oriented_det_window_batch_size')

        diagnostics, analysis = run_diagnostics_pipeline(
            experiment_dir=str(meta.get('experiment_dir', '')),
            checkpoint_path=str(meta.get('checkpoint', '')),
            config_path=str(meta.get('config_file', '')),
            data_root=meta.get('data_root', ''),
            data_split=str(meta.get('data_split', 'val')),
            class_names=class_names,
            score_threshold=float(score_thr),
            per_cls_thr=per_cls,
            nms_threshold=float(nms_thr),
            nms_class_agnostic=bool(nms_ag),
            iou_threshold=iou_thr,
            pr_iou_threshold=args.pr_iou_threshold,
            pr_threshold_min=args.pr_threshold_min,
            pr_threshold_max=args.pr_threshold_max,
            pr_threshold_step=args.pr_threshold_step,
            per_class_threshold_analysis=(not args.no_per_class_threshold_analysis),
            resolved_metrics_margin_px=int(mmargin),
            all_detections=all_detections,
            all_ground_truths=all_ground_truths,
            results=results,
            all_scores=all_scores,
            image_name_by_id=image_name_by_id,
            sliding_window_positions_total=int(sw_est) if sw_est is not None else None,
            window_batch_effective=int(wb) if wb is not None else None,
            t_infer_sec=t_infer,
            device=device,
            output_dir=output_dir,
            tile_metrics_csv=args.tile_metrics_csv,
            use_exact_rotated_iou=use_exact_map,
        )

        meta['metrics_margin_pixels'] = int(mmargin)
        meta['diagnostics'] = diagnostics
        if analysis is not None:
            meta['best_threshold_f1'] = analysis.get('best_threshold', {})
            meta['analysis_file'] = analysis.get('artifacts', {}).get(
                'analysis_json', f'analysis_iou{iou_thr:.2f}.json'
            )
            meta['pr_iou_threshold'] = analysis.get('iou_threshold', iou_thr)
        bundle['metadata'] = meta
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(bundle, f, indent=2)
        print(f'Updated metrics in-place: {json_path}')
        return

    experiment_dir = args.experiment_dir
    checkpoint = args.checkpoint
    config_path = args.config

    # Resolve any missing paths.
    #
    # Important: if --experiment-dir is provided, users expect checkpoint/config resolution to be
    # constrained to that directory (NOT the latest run). Previously we filled missing checkpoint
    # or config by calling find_latest_experiment(), which could silently mix an explicit experiment
    # with a different run's config/ckpt.
    if experiment_dir is not None and (checkpoint is None or config_path is None):
        exp_dir_resolved, ckpt_resolved, cfg_resolved = resolve_experiment_paths(experiment_dir)
        if checkpoint is None:
            checkpoint = ckpt_resolved
            print(f"Resolved checkpoint from experiment dir: {checkpoint}")
        if config_path is None:
            config_path = cfg_resolved
            print(f"Resolved config from experiment dir: {config_path}")
        experiment_dir = exp_dir_resolved
    elif experiment_dir is None or checkpoint is None or config_path is None:
        print(f"Auto-detecting latest experiment for model type: {args.model_type}")
        detected_experiment_dir, detected_checkpoint, detected_config = find_latest_experiment(args.model_type)
        if experiment_dir is None:
            experiment_dir = detected_experiment_dir
            print(f"  Experiment directory: {experiment_dir}")
        if checkpoint is None:
            checkpoint = detected_checkpoint
            print(f"  Checkpoint: {checkpoint}")
        if config_path is None:
            config_path = detected_config
            print(f"  Config: {config_path}")

    run_inference_and_save(
        experiment_dir=experiment_dir,
        checkpoint_path=checkpoint,
        config_path=config_path,
        data_root=args.data_root,
        output_dir=args.output_dir,
        save_visualizations=args.save_visualizations,
        data_split=args.data_split,
        val_dir=args.val_dir,
        overlap_ratio=args.overlap_ratio,
        overlap_pixels=args.overlap_pixels,
        vis_score_threshold=args.vis_score_threshold,
        score_threshold=args.score_threshold,
        ignore_config_per_class_score_threshold=args.no_per_class_score_thresholds,
        nms_threshold=args.nms_threshold,
        nms_class_agnostic=args.nms_class_agnostic,
        inference_pre_nms_score_threshold=args.inference_pre_nms_score_threshold,
        rpn_pre_nms_top_n=args.rpn_pre_nms_top_n,
        rpn_post_nms_top_n=args.rpn_post_nms_top_n,
        rpn_nms_threshold=args.rpn_nms_threshold,
        max_detections_per_image=args.max_detections_per_image,
        iou_threshold=args.iou_threshold,
        metrics_margin_pixels=args.metrics_margin_pixels,
        run_diagnostics=not args.no_diagnostics,
        debug_coords=args.debug_coords,
        pr_threshold_min=args.pr_threshold_min,
        pr_threshold_max=args.pr_threshold_max,
        pr_threshold_step=args.pr_threshold_step,
        pr_iou_threshold=args.pr_iou_threshold,
        per_class_threshold_analysis=(not args.no_per_class_threshold_analysis),
        tile_metrics_csv=args.tile_metrics_csv,
    )


if __name__ == '__main__':
    main()

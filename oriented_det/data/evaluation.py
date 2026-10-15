"""Oriented mAP evaluation compatible with DOTA protocol.

Uses detection and ground-truth RBoxes exactly as provided (no coordinate rescaling).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import statistics
import sys

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

try:
    import torch
except ImportError:
    torch = None  # type: ignore

# Type hint for device - use Optional[torch.device] if torch is available
if torch is not None:
    DeviceType = Optional[torch.device]
else:
    DeviceType = Optional[object]  # Fallback when torch is not available

from ..geometry import RBox
from ..ops import iou
from ..ops.utils import resolve_exact_polygon_iou_backend


@dataclass(frozen=True)
class Detection:
    """Single detection result."""

    rbox: RBox
    score: float
    class_id: int
    class_name: str
    image_id: Optional[str] = None  # Required for correct mAP: match only to GTs from same image


@dataclass(frozen=True)
class GroundTruth:
    """Single ground truth annotation."""

    rbox: RBox
    class_id: int
    class_name: str
    difficult: int = 0
    image_id: Optional[str] = None  # Required for correct mAP: match only to detections from same image


@dataclass(frozen=True)
class ClassEvalMetrics:
    """Per-class detection stats (MMRotate-style eval table)."""

    num_gts: int
    num_dets: int
    recall: float
    ap: float


def ossdd_map_iou_thresholds() -> List[float]:
    """Rotated-IoU thresholds for the OSSDD paper mAP (0.10, 0.15, ..., 0.95)."""
    return [round(0.10 + 0.05 * i, 2) for i in range(18)]


def format_map_iou_span(thresholds: Sequence[float]) -> str:
    """Compact label for a uniform or arbitrary IoU list."""
    vals = [float(t) for t in thresholds]
    if len(vals) >= 2:
        step = vals[1] - vals[0]
        uniform = all(
            abs((vals[i + 1] - vals[i]) - step) < 1e-6 for i in range(len(vals) - 1)
        )
        if uniform:
            return f"{vals[0]:.2f} to {vals[-1]:.2f} step {step:.2f} ({len(vals)} values)"
    return ", ".join(f"{t:.2f}" for t in vals)


def average_precision_area(recalls: Sequence[float], precisions: Sequence[float]) -> float:
    """Area under the precision-recall curve (VOC 2010 all-point interpolation)."""
    if not recalls:
        return 0.0
    mrec = [0.0, *[float(r) for r in recalls], 1.0]
    mpre = [0.0, *[float(p) for p in precisions], 0.0]
    for i in range(len(mpre) - 1, 0, -1):
        if mpre[i - 1] < mpre[i]:
            mpre[i - 1] = mpre[i]
    ap = 0.0
    for i in range(len(mrec) - 1):
        delta = mrec[i + 1] - mrec[i]
        if delta:
            ap += delta * mpre[i + 1]
    return ap


def _greedy_tp_fp(
    sorted_dets: Sequence[Detection],
    ground_truths: Sequence[GroundTruth],
    candidates: Sequence[Sequence[Tuple[int, float]]],
    iou_threshold: float,
) -> Tuple[List[int], List[int]]:
    """Score-ordered greedy match. Difficult ground truths absorb a detection without counting."""
    gt_matched = [False] * len(ground_truths)
    tp: List[int] = []
    fp: List[int] = []
    for det_idx, _det in enumerate(sorted_dets):
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, det_iou in candidates[det_idx]:
            if gt_matched[gt_idx]:
                continue
            if det_iou > best_iou:
                best_iou = det_iou
                best_gt_idx = gt_idx
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            if ground_truths[best_gt_idx].difficult == 0:
                gt_matched[best_gt_idx] = True
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    return tp, fp


def _precision_recall_curve(
    tp: Sequence[int],
    fp: Sequence[int],
    num_positives: int,
) -> Tuple[List[float], List[float], float]:
    """Cumulative precision/recall in score order, and recall after the last detection."""
    tp_cumsum: List[int] = []
    fp_cumsum: List[int] = []
    cumsum_tp = 0
    cumsum_fp = 0
    for t, f in zip(tp, fp):
        cumsum_tp += int(t)
        cumsum_fp += int(f)
        tp_cumsum.append(cumsum_tp)
        fp_cumsum.append(cumsum_fp)
    if num_positives <= 0:
        return [], [], 0.0
    recalls = [t / num_positives for t in tp_cumsum]
    precisions = [
        tp_cumsum[i] / (tp_cumsum[i] + fp_cumsum[i]) if (tp_cumsum[i] + fp_cumsum[i]) > 0 else 0.0
        for i in range(len(tp_cumsum))
    ]
    final_recall = recalls[-1] if recalls else 0.0
    return recalls, precisions, final_recall


class APCalculator:
    """Calculate Average Precision (AP) for oriented object detection."""

    def __init__(
        self,
        iou_threshold: float = 0.5,
        class_names: Optional[Sequence[str]] = None,
        device: DeviceType = None,
        *,
        use_exact_rotated_iou: bool = True,
    ):
        if not 0 < iou_threshold <= 1:
            raise ValueError("IoU threshold must be in (0, 1]")
        self.iou_threshold = iou_threshold
        self.class_names = class_names or []
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)} if self.class_names else {}
        self.use_exact_rotated_iou = bool(use_exact_rotated_iou)
        self.device = None if self.use_exact_rotated_iou else device
        self._exact_iou_backend: Optional[str] = (
            resolve_exact_polygon_iou_backend() if self.use_exact_rotated_iou else None
        )
    
    def compute_ap(
        self,
        detections: Sequence[Detection],
        ground_truths: Sequence[GroundTruth],
        *,
        class_name: Optional[str] = None,
        show_progress: bool = False,
        progress_stream: Optional[Any] = None,
        max_iou_calculations: Optional[int] = None,
        min_box_size: float = 1.0,
    ) -> Tuple[float, ClassEvalMetrics]:
        """Compute Average Precision for a single class.
        
        Args:
            detections: List of detections
            ground_truths: List of ground truths
            class_name: Optional class name
            show_progress: Whether to show progress bar
            max_iou_calculations: Maximum number of IoU calculations to allow.
                                 If exceeded, will use approximate method or return 0.0.
            min_box_size: Minimum box size (width/height) in pixels to consider valid.
                         Boxes smaller than this will be filtered out (default: 1.0).
        """
        if class_name:
            detections = [d for d in detections if d.class_name == class_name]
            ground_truths = [gt for gt in ground_truths if gt.class_name == class_name]
        
        # Filter out degenerated boxes
        original_det_count = len(detections)
        original_gt_count = len(ground_truths)
        detections = [d for d in detections if d.rbox.is_valid(min_size=min_box_size)]
        ground_truths = [gt for gt in ground_truths if gt.rbox.is_valid(min_size=min_box_size)]
        
        if show_progress:
            filtered_dets = original_det_count - len(detections)
            filtered_gts = original_gt_count - len(ground_truths)
            if filtered_dets > 0 or filtered_gts > 0:
                filter_msg = f"  Filtered {filtered_dets} degenerated detections, {filtered_gts} degenerated ground truths"
                print(filter_msg)
        
        if not ground_truths:
            ap = 0.0 if detections else 1.0
            return ap, ClassEvalMetrics(
                num_gts=0,
                num_dets=len(detections),
                recall=0.0,
                ap=ap,
            )

        # Check if we should optimize due to too many calculations
        num_dets = len(detections)
        num_gts = len(ground_truths)
        total_calculations = num_dets * num_gts
        
        # Use batch IoU if available and calculations are too many
        use_batch_iou = max_iou_calculations and total_calculations > max_iou_calculations

        # Exact polygon mAP (final_nms_use_cpu): never use GPU sampling IoU
        from ..utils import is_gpu_device
        use_gpu = (
            not self.use_exact_rotated_iou
            and torch is not None
            and self.device is not None
            and is_gpu_device(self.device)
        )
        
        sorted_dets = sorted(detections, key=lambda d: d.score, reverse=True)
        tp_fp_from_batch = False
        tp: List[int] = []
        fp: List[int] = []
        
        if use_batch_iou:
            # Try using batch IoU computation which is much faster
            try:
                det_rboxes = [d.rbox for d in sorted_dets]
                gt_rboxes = [gt.rbox for gt in ground_truths]
                
                # Use chunking for very large matrices to avoid memory issues
                # Process detections in chunks to limit memory usage.
                # GPU IoU builds [chunk*S, num_gt] tensors with S ~= default oriented IoU samples
                # (typically 100; tier/env can raise it). When num_gt is large (e.g. 72k)
                # a fixed 5000-detect chunk can OOM. Cap chunk so chunk*S*num_gt <= ~500M elements.
                num_gt = len(ground_truths)
                max_elements = 500_000_000  # ~500M elements for bool tensor (~500 MB)
                samples_per_box = 100  # conservative match to oriented_box_iou_gpu default envelope
                adaptive_chunk = max(1, max_elements // (samples_per_box * max(1, num_gt)))
                CHUNK_SIZE = min(5000, adaptive_chunk)
                num_dets = len(sorted_dets)
                
                if show_progress and tqdm is not None:
                    if self.use_exact_rotated_iou:
                        iou_type = f"CPU exact ({self._exact_iou_backend})"
                    else:
                        iou_type = "GPU-accelerated" if use_gpu else "CPU"
                    print(f"    Using chunked batch IoU computation ({iou_type}) for {class_name} ({total_calculations:,} calculations)")
                    if num_dets > CHUNK_SIZE:
                        print(f"    Processing {num_dets:,} detections in chunks of {CHUNK_SIZE:,}")
                
                # Process detections in chunks
                gt_matched = [False] * len(ground_truths)
                tp = []
                fp = []
                
                # Process detections in chunks; tqdm to progress_stream so console sees it but train.log does not
                tqdm_file = progress_stream if progress_stream is not None else sys.stderr
                det_iter = (
                    tqdm(range(0, num_dets, CHUNK_SIZE), desc="    Processing chunks", unit="chunk", leave=False, file=tqdm_file)
                    if (show_progress and tqdm is not None and num_dets > CHUNK_SIZE)
                    else range(0, num_dets, CHUNK_SIZE)
                )

                for chunk_start in det_iter:
                    chunk_end = min(chunk_start + CHUNK_SIZE, num_dets)
                    chunk_dets = sorted_dets[chunk_start:chunk_end]
                    chunk_det_rboxes = det_rboxes[chunk_start:chunk_end]
                    
                    # Compute IoU matrix for this chunk
                    chunk_iou_matrix = None
                    try:
                        if use_gpu:
                            # Use GPU-accelerated IoU (much faster)
                            from ..ops.gpu_ops import oriented_box_iou_gpu
                            from ..ops.utils import rboxes_to_tensor
                            
                            # Convert RBox objects to tensors
                            chunk_det_tensors = rboxes_to_tensor(chunk_det_rboxes, device=self.device)
                            gt_tensors = rboxes_to_tensor(gt_rboxes, device=self.device)
                            
                            # Compute IoU matrix on GPU
                            iou_matrix_gpu = oriented_box_iou_gpu(chunk_det_tensors, gt_tensors)
                            
                            # Convert to CPU list of lists for compatibility
                            chunk_iou_matrix = iou_matrix_gpu.cpu().tolist()
                        else:
                            from ..ops.iou import batch_rbox_iou
                            chunk_iou_matrix = batch_rbox_iou(
                                chunk_det_rboxes,
                                gt_rboxes,
                                device=self.device,
                                intersection_backend=(
                                    self._exact_iou_backend
                                    if self.use_exact_rotated_iou
                                    else "auto"
                                ),
                            )
                    except Exception as chunk_e:
                        # If chunk batch fails, fall back to per-detection computation for this chunk
                        if show_progress:
                            print(f"    Chunk batch IoU failed, using per-detection for chunk {chunk_start}-{chunk_end}: {chunk_e}")
                        chunk_iou_matrix = None
                    
                    # Process each detection in this chunk
                    for local_idx, det in enumerate(chunk_dets):
                        det_idx = chunk_start + local_idx
                        best_iou = 0.0
                        best_gt_idx = -1
                        
                        if chunk_iou_matrix is not None:
                            # Use pre-computed IoU matrix
                            for gt_idx, gt in enumerate(ground_truths):
                                if gt_matched[gt_idx] or gt.class_name != det.class_name:
                                    continue
                                if det.image_id is not None and gt.image_id is not None and det.image_id != gt.image_id:
                                    continue
                                det_iou = chunk_iou_matrix[local_idx][gt_idx]
                                if det_iou > best_iou:
                                    best_iou = det_iou
                                    best_gt_idx = gt_idx
                        else:
                            # Fall back to per-detection IoU computation
                            for gt_idx, gt in enumerate(ground_truths):
                                if gt_matched[gt_idx] or gt.class_name != det.class_name:
                                    continue
                                if det.image_id is not None and gt.image_id is not None and det.image_id != gt.image_id:
                                    continue
                                try:
                                    from ..ops.iou import rbox_iou
                                    det_iou = rbox_iou(
                                        det.rbox,
                                        gt.rbox,
                                        intersection_backend=(
                                            self._exact_iou_backend
                                            if self.use_exact_rotated_iou
                                            else "auto"
                                        ),
                                    )
                                    if det_iou > best_iou:
                                        best_iou = det_iou
                                        best_gt_idx = gt_idx
                                except Exception:
                                    continue
                        
                        # Check if match is valid
                        if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                            gt = ground_truths[best_gt_idx]
                            if gt.difficult == 0:
                                gt_matched[best_gt_idx] = True
                                tp.append(1)
                                fp.append(0)
                            else:
                                tp.append(0)
                                fp.append(0)
                        else:
                            tp.append(0)
                            fp.append(1)
                    
                    # Clear chunk matrix from memory
                    del chunk_iou_matrix
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                
                tp_fp_from_batch = True
                
            except Exception as e:
                # Fall back to regular computation if batch fails
                if show_progress:
                    print(f"    Batch IoU failed for {class_name}, using regular method (will be slow): {e}")
                use_batch_iou = False
        
        if not tp_fp_from_batch:
            # Track which ground truths have been matched (standard per-detection path)
            gt_matched = [False] * len(ground_truths)
            tp = []
            fp = []
            
            # Use progress bar if requested and many detections; write to progress_stream so console sees it but train.log does not
            tqdm_file = progress_stream if progress_stream is not None else sys.stderr
            use_pbar = show_progress and tqdm is not None and len(sorted_dets) > 5000
            det_iter = tqdm(sorted_dets, desc=f"  {class_name or 'All'}", unit="det", leave=False, ncols=100, file=tqdm_file) if use_pbar else sorted_dets
            
            for det_idx, det in enumerate(det_iter):
                best_iou = 0.0
                best_gt_idx = -1
                
                # Find best matching ground truth (only from same image when image_id is set)
                for gt_idx, gt in enumerate(ground_truths):
                    if gt_matched[gt_idx] or gt.class_name != det.class_name:
                        continue
                    if det.image_id is not None and gt.image_id is not None and det.image_id != gt.image_id:
                        continue
                    try:
                        det_iou = iou.rbox_iou(
                            det.rbox,
                            gt.rbox,
                            intersection_backend=(
                                self._exact_iou_backend
                                if self.use_exact_rotated_iou
                                else "auto"
                            ),
                        )
                        if det_iou > best_iou:
                            best_iou = det_iou
                            best_gt_idx = gt_idx
                    except Exception:
                        # Skip problematic IoU calculations and continue with next GT
                        continue

                # Check if match is valid
                if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                    gt = ground_truths[best_gt_idx]
                    if gt.difficult == 0:  # Only count non-difficult as TP
                        gt_matched[best_gt_idx] = True
                        tp.append(1)
                        fp.append(0)
                    else:
                        tp.append(0)
                        fp.append(0)  # Difficult GTs don't count as FP either
                else:
                    tp.append(0)
                    fp.append(1)
        
        # Compute precision and recall
        tp_cumsum = []
        fp_cumsum = []
        cumsum_tp = 0
        cumsum_fp = 0
        
        for t, f in zip(tp, fp):
            cumsum_tp += t
            cumsum_fp += f
            tp_cumsum.append(cumsum_tp)
            fp_cumsum.append(cumsum_fp)
        
        num_positives = sum(1 for gt in ground_truths if gt.difficult == 0)
        num_dets = len(sorted_dets)
        if num_positives == 0:
            return 0.0, ClassEvalMetrics(
                num_gts=0,
                num_dets=num_dets,
                recall=0.0,
                ap=0.0,
            )

        recalls = [t / num_positives for t in tp_cumsum]
        precisions = [
            tp_cumsum[i] / (tp_cumsum[i] + fp_cumsum[i]) if (tp_cumsum[i] + fp_cumsum[i]) > 0 else 0.0
            for i in range(len(tp_cumsum))
        ]

        final_recall = recalls[-1] if recalls else 0.0

        # Compute AP using 11-point interpolation
        ap = 0.0
        for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            precisions_at_recall = [p for r, p in zip(recalls, precisions) if r >= t]
            if precisions_at_recall:
                ap += max(precisions_at_recall) / 11.0

        return ap, ClassEvalMetrics(
            num_gts=num_positives,
            num_dets=num_dets,
            recall=final_recall,
            ap=ap,
        )

    def _iou_matrix(
        self,
        det_rboxes: Sequence[Any],
        gt_rboxes: Sequence[Any],
    ) -> List[List[float]]:
        """Pairwise rotated IoU. One matrix is reused across IoU thresholds."""
        if not det_rboxes or not gt_rboxes:
            return [[] for _ in det_rboxes]
        from ..utils import is_gpu_device

        use_gpu = (
            not self.use_exact_rotated_iou
            and torch is not None
            and self.device is not None
            and is_gpu_device(self.device)
        )
        if use_gpu:
            from ..ops.gpu_ops import oriented_box_iou_gpu
            from ..ops.utils import rboxes_to_tensor

            det_tensors = rboxes_to_tensor(list(det_rboxes), device=self.device)
            gt_tensors = rboxes_to_tensor(list(gt_rboxes), device=self.device)
            return oriented_box_iou_gpu(det_tensors, gt_tensors).detach().cpu().tolist()
        from ..ops.iou import batch_rbox_iou

        return batch_rbox_iou(
            list(det_rboxes),
            list(gt_rboxes),
            device=self.device,
            intersection_backend=(
                self._exact_iou_backend if self.use_exact_rotated_iou else "auto"
            ),
        )

    def _iou_candidates(
        self,
        sorted_dets: Sequence[Detection],
        ground_truths: Sequence[GroundTruth],
    ) -> List[List[Tuple[int, float]]]:
        """Per detection, ``(gt_index, iou)`` for ground truths on the same image.

        A missing ``image_id`` matches every box, same as the single-threshold matcher.
        """
        dets_by_image: Dict[Optional[str], List[int]] = defaultdict(list)
        gts_by_image: Dict[Optional[str], List[int]] = defaultdict(list)
        for det_idx, det in enumerate(sorted_dets):
            dets_by_image[det.image_id].append(det_idx)
        for gt_idx, gt in enumerate(ground_truths):
            gts_by_image[gt.image_id].append(gt_idx)
        none_gts = gts_by_image.get(None, [])
        candidates: List[List[Tuple[int, float]]] = [[] for _ in sorted_dets]
        for image_id, det_idxs in dets_by_image.items():
            if image_id is None:
                gt_idxs = list(range(len(ground_truths)))
            else:
                gt_idxs = list(gts_by_image.get(image_id, [])) + list(none_gts)
            if not gt_idxs:
                continue
            matrix = self._iou_matrix(
                [sorted_dets[i].rbox for i in det_idxs],
                [ground_truths[j].rbox for j in gt_idxs],
            )
            for row, det_i in enumerate(det_idxs):
                candidates[det_i] = [
                    (gt_idxs[col], float(matrix[row][col]))
                    for col in range(len(gt_idxs))
                ]
        return candidates

    def compute_ap_over_thresholds(
        self,
        detections: Sequence[Detection],
        ground_truths: Sequence[GroundTruth],
        iou_thresholds: Sequence[float],
        *,
        class_name: Optional[str] = None,
        min_box_size: float = 1.0,
    ) -> Tuple[float, ClassEvalMetrics, Dict[float, float]]:
        """Mean of area-under-PR over ``iou_thresholds``. IoU is computed once.

        Recall on the returned metrics is at ``self.iou_threshold`` when that
        value is in the list, otherwise at the closest listed threshold.
        """
        thresholds = [float(t) for t in iou_thresholds]
        if not thresholds:
            raise ValueError("iou_thresholds must not be empty")
        if class_name:
            detections = [d for d in detections if d.class_name == class_name]
            ground_truths = [gt for gt in ground_truths if gt.class_name == class_name]
        detections = [d for d in detections if d.rbox.is_valid(min_size=min_box_size)]
        ground_truths = [gt for gt in ground_truths if gt.rbox.is_valid(min_size=min_box_size)]
        num_dets = len(detections)
        if not ground_truths:
            ap = 0.0 if detections else 1.0
            per_thr = {thr: ap for thr in thresholds}
            return ap, ClassEvalMetrics(
                num_gts=0,
                num_dets=num_dets,
                recall=0.0,
                ap=ap,
            ), per_thr

        sorted_dets = sorted(detections, key=lambda d: d.score, reverse=True)
        candidates = self._iou_candidates(sorted_dets, ground_truths)
        num_positives = sum(1 for gt in ground_truths if gt.difficult == 0)
        per_thr: Dict[float, float] = {}
        recall_at_ref = 0.0
        ref = min(thresholds, key=lambda t: abs(t - self.iou_threshold))
        for thr in thresholds:
            tp, fp = _greedy_tp_fp(sorted_dets, ground_truths, candidates, thr)
            if num_positives == 0:
                ap_thr = 0.0
                final_recall = 0.0
            else:
                recalls, precisions, final_recall = _precision_recall_curve(tp, fp, num_positives)
                ap_thr = average_precision_area(recalls, precisions)
            per_thr[thr] = ap_thr
            if thr == ref:
                recall_at_ref = final_recall
        mean_ap = sum(per_thr.values()) / len(per_thr)
        return mean_ap, ClassEvalMetrics(
            num_gts=num_positives,
            num_dets=num_dets,
            recall=recall_at_ref,
            ap=mean_ap,
        ), per_thr

    def compute_map(
        self,
        all_detections: Dict[str, List[Detection]],  # image_id -> detections
        all_ground_truths: Dict[str, List[GroundTruth]],  # image_id -> ground_truths
        show_progress: bool = True,
        progress_stream: Optional[Any] = None,
        max_iou_calculations_per_class: int = 10_000_000,  # 10M IoU calculations max per class
        min_box_size: float = 1.0,
        iou_thresholds: Optional[Sequence[float]] = None,
        iou_ap_out: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, ClassEvalMetrics]]:
        """Compute mAP across all classes.
        
        Args:
            all_detections: Dictionary mapping image_id -> list of Detection objects
            all_ground_truths: Dictionary mapping image_id -> list of GroundTruth objects
            show_progress: Whether to show progress bars
            max_iou_calculations_per_class: Maximum IoU calculations per class before optimization
            min_box_size: Minimum box size (width/height) in pixels to consider valid.
                         Boxes smaller than this will be filtered out (default: 1.0).
        
        Returns:
            ``(class_aps, class_metrics)`` — AP and MMRotate-style stats per class.
        """
        # Collect all unique class names
        all_classes = set()
        for dets in all_detections.values():
            all_classes.update(d.class_name for d in dets)
        for gts in all_ground_truths.values():
            all_classes.update(gt.class_name for gt in gts)
        
        if self.class_names:
            all_classes = all_classes.intersection(set(self.class_names))
        
        # Aggregate detections and ground truths by class
        class_aps: Dict[str, float] = {}
        class_metrics: Dict[str, ClassEvalMetrics] = {}
        thresholds = [float(t) for t in iou_thresholds] if iou_thresholds else None
        ap_sums: Dict[float, float] = {thr: 0.0 for thr in thresholds} if thresholds else {}
        sorted_classes = sorted(all_classes)
        num_classes = len(sorted_classes)
        
        # Use progress bar for classes; write to progress_stream so console sees it but train.log does not
        tqdm_file = progress_stream if progress_stream is not None else sys.stderr
        use_pbar = show_progress and tqdm is not None and num_classes > 1
        class_iter = tqdm(sorted_classes, desc="Computing AP per class", unit="class", ncols=120, file=tqdm_file) if use_pbar else sorted_classes
        
        for class_name in class_iter:
            try:
                class_detections = []
                class_ground_truths = []
                
                for image_id in set(all_detections.keys()) | set(all_ground_truths.keys()):
                    dets = all_detections.get(image_id, [])
                    gts = all_ground_truths.get(image_id, [])
                    
                    class_detections.extend([d for d in dets if d.class_name == class_name])
                    class_ground_truths.extend([gt for gt in gts if gt.class_name == class_name])
                
                # Log class statistics
                num_dets = len(class_detections)
                num_gts = len(class_ground_truths)
                
                # Compute AP for this class (show progress if many detections or ground truths)
                # Show progress if computation will be slow (many IoU calculations)
                total_iou_calculations = num_dets * num_gts
                show_det_progress = total_iou_calculations > 100000 or num_dets > 5000 or num_gts > 1000
                
                # Warn if too many calculations
                if total_iou_calculations > max_iou_calculations_per_class:
                    warning_msg = (f"  WARNING: {class_name} has {total_iou_calculations:,} IoU calculations "
                                  f"({num_dets:,} dets × {num_gts:,} GTs). This will be slow!")
                    if use_pbar and isinstance(class_iter, tqdm):
                        class_iter.write(warning_msg)
                    else:
                        print(warning_msg)
                
                # Update progress bar with class info
                if use_pbar and isinstance(class_iter, tqdm):
                    if show_det_progress:
                        class_iter.set_postfix_str(f"{class_name}: {num_dets} dets × {num_gts} GTs = {total_iou_calculations:,} IoUs")
                        class_iter.write(f"  Computing AP for {class_name}: {num_dets:,} dets, {num_gts:,} GTs ({total_iou_calculations:,} IoU calculations)")
                    else:
                        class_iter.set_postfix_str(f"{class_name}: {num_dets} dets, {num_gts} GTs")
                
                if thresholds:
                    ap, metrics, ap_at = self.compute_ap_over_thresholds(
                        class_detections,
                        class_ground_truths,
                        thresholds,
                        class_name=class_name,
                        min_box_size=min_box_size,
                    )
                    for thr, thr_ap in ap_at.items():
                        ap_sums[thr] = ap_sums.get(thr, 0.0) + float(thr_ap)
                else:
                    ap, metrics = self.compute_ap(
                        class_detections,
                        class_ground_truths,
                        class_name=class_name,
                        show_progress=show_det_progress,
                        progress_stream=progress_stream,
                        max_iou_calculations=max_iou_calculations_per_class,
                        min_box_size=min_box_size,
                    )
                class_aps[class_name] = ap
                class_metrics[class_name] = metrics
                
                # Update progress bar with current AP if using tqdm
                if use_pbar and isinstance(class_iter, tqdm):
                    class_iter.set_postfix_str(f"{class_name}: AP={ap:.4f}")
                    
            except Exception as e:
                # Log error and continue with other classes
                error_msg = f"Error computing AP for class '{class_name}': {e}"
                if use_pbar and isinstance(class_iter, tqdm):
                    class_iter.write(f"  ERROR: {error_msg}")
                else:
                    print(f"  ERROR: {error_msg}")
                import traceback
                traceback.print_exc()
                # Set AP to 0.0 for failed classes
                class_aps[class_name] = 0.0
                class_metrics[class_name] = ClassEvalMetrics(
                    num_gts=0,
                    num_dets=0,
                    recall=0.0,
                    ap=0.0,
                )
                if use_pbar and isinstance(class_iter, tqdm):
                    class_iter.set_postfix_str(f"{class_name}: ERROR")

        if iou_ap_out is not None and thresholds and class_aps:
            n_classes = len(class_aps)
            iou_ap_out["thresholds"] = list(thresholds)
            iou_ap_out["per_iou"] = [ap_sums[thr] / n_classes for thr in thresholds]

        return class_aps, class_metrics


def compute_oriented_map(
    detections: Dict[str, List[Detection]],
    ground_truths: Dict[str, List[GroundTruth]],
    *,
    iou_threshold: float = 0.5,
    class_names: Optional[Sequence[str]] = None,
    show_progress: bool = True,
    progress_stream: Optional[Any] = None,
    max_iou_calculations_per_class: int = 10_000_000,
    device: DeviceType = None,
    min_box_size: float = 1.0,
    use_exact_rotated_iou: bool = True,
    iou_thresholds: Optional[Sequence[float]] = None,
    iou_ap_out: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, float], Dict[str, ClassEvalMetrics]]:
    """Compute mean Average Precision (mAP) for oriented detection.
    
    Args:
        detections: Dictionary mapping image_id -> list of Detection objects
        ground_truths: Dictionary mapping image_id -> list of GroundTruth objects
        iou_threshold: IoU threshold for positive matches. Also the recall operating
            point when ``iou_thresholds`` is set.
        class_names: Optional list of class names to evaluate
        show_progress: Whether to show progress bars during computation
        progress_stream: If set (e.g. original stderr when stdout is teed to a log file),
            progress bars write here so they appear on console but not in the log file.
        max_iou_calculations_per_class: Maximum IoU calculations per class before using optimization
        device: Optional torch device for GPU acceleration. If None, uses CPU.
            Ignored when ``use_exact_rotated_iou`` is True (Shapely/python polygon IoU only).
        min_box_size: Minimum box size (width/height) in pixels to consider valid.
                     Boxes smaller than this will be filtered out (default: 1.0).
        use_exact_rotated_iou: When True, mAP matching uses exact polygon IoU on CPU
            (Shapely when installed); never ``oriented_box_iou_gpu``. When False, uses GPU
            sampling IoU (controlled by ``evaluation.val_exact_rotated_iou`` in training config).
        iou_thresholds: When set, mAP is the mean of area-under-PR at each rotated-IoU
            threshold. The pairwise IoU matrix is built once. ``None`` keeps the
            single-threshold 11-point AP at ``iou_threshold``.
        iou_ap_out: When set with ``iou_thresholds``, receives ``thresholds`` and
            ``per_iou`` (mean AP at each threshold).
    
    Returns:
        Tuple of (mAP, class_ap_dict, class_metrics) where mAP is the mean of all
        class APs and ``class_metrics`` holds per-class gts/dets/recall (MMRotate-style).
    """
    thresholds = [float(t) for t in iou_thresholds] if iou_thresholds else None
    if show_progress:
        if thresholds:
            backend = (
                resolve_exact_polygon_iou_backend()
                if use_exact_rotated_iou
                else "gpu-sampling"
            )
            print(
                "mAP: mean of area-under-PR over rotated IoU "
                f"{format_map_iou_span(thresholds)} "
                f"(backend={backend!r}; IoU matrix once)",
                flush=True,
            )
        elif use_exact_rotated_iou:
            backend = resolve_exact_polygon_iou_backend()
            print(
                f"mAP: exact rotated IoU on CPU (backend={backend!r}; "
                "GPU sampling IoU disabled)",
                flush=True,
            )
    calculator = APCalculator(
        iou_threshold=iou_threshold,
        class_names=class_names,
        device=device,
        use_exact_rotated_iou=use_exact_rotated_iou,
    )
    class_aps, class_metrics = calculator.compute_map(
        detections,
        ground_truths,
        show_progress=show_progress,
        progress_stream=progress_stream,
        max_iou_calculations_per_class=max_iou_calculations_per_class,
        min_box_size=min_box_size,
        iou_thresholds=thresholds,
        iou_ap_out=iou_ap_out,
    )

    if not class_aps:
        return 0.0, {}, {}

    mean_ap = sum(class_aps.values()) / len(class_aps)
    return mean_ap, class_aps, class_metrics


@dataclass(frozen=True)
class ClassGtBestIouMetrics:
    """Per-class GT alignment: max IoU vs raw detections (before score filter)."""

    num_gts: int
    mean_best_iou_any: float
    mean_best_iou_same_class: float
    median_best_iou_same_class: float


@dataclass(frozen=True)
class GtBestIouAlignmentMetrics:
    """Global and per-class mean best IoU (per-GT max over raw detections)."""

    num_gts: int
    mean_best_iou_any: float
    mean_best_iou_same_class: float
    median_best_iou_any: float
    median_best_iou_same_class: float
    per_class: Dict[str, ClassGtBestIouMetrics]


def detection_matches_ground_truth_class(det: Detection, gt: GroundTruth) -> bool:
    """True when ``det`` and ``gt`` refer to the same object class."""
    if det.class_id == gt.class_id:
        return True
    # Some inference paths use 1-based class labels.
    if det.class_id == gt.class_id + 1 or det.class_id - 1 == gt.class_id:
        return True
    if det.class_name and gt.class_name and det.class_name == gt.class_name:
        return True
    return False


def _aggregate_gt_best_iou_samples(
    best_any_by_class: Dict[str, List[float]],
    best_same_by_class: Dict[str, List[float]],
    *,
    class_names: Optional[Sequence[str]] = None,
) -> GtBestIouAlignmentMetrics:
    all_any: List[float] = []
    all_same: List[float] = []
    for vals in best_any_by_class.values():
        all_any.extend(vals)
    for vals in best_same_by_class.values():
        all_same.extend(vals)

    class_keys = set(best_any_by_class) | set(best_same_by_class)
    if class_names:
        ordered = [c for c in class_names if c in class_keys]
        ordered.extend(c for c in sorted(class_keys) if c not in ordered)
    else:
        ordered = sorted(class_keys)

    per_class: Dict[str, ClassGtBestIouMetrics] = {}
    for cname in ordered:
        vals_any = best_any_by_class.get(cname, [])
        vals_same = best_same_by_class.get(cname, [])
        per_class[cname] = ClassGtBestIouMetrics(
            num_gts=len(vals_any),
            mean_best_iou_any=statistics.fmean(vals_any) if vals_any else 0.0,
            mean_best_iou_same_class=statistics.fmean(vals_same) if vals_same else 0.0,
            median_best_iou_same_class=float(statistics.median(vals_same)) if vals_same else 0.0,
        )

    return GtBestIouAlignmentMetrics(
        num_gts=len(all_same),
        mean_best_iou_any=statistics.fmean(all_any) if all_any else 0.0,
        mean_best_iou_same_class=statistics.fmean(all_same) if all_same else 0.0,
        median_best_iou_any=float(statistics.median(all_any)) if all_any else 0.0,
        median_best_iou_same_class=float(statistics.median(all_same)) if all_same else 0.0,
        per_class=per_class,
    )


def compute_gt_best_iou_alignment_metrics(
    all_detections: Dict[str, List[Detection]],
    all_ground_truths: Dict[str, List[GroundTruth]],
    *,
    class_names: Optional[Sequence[str]] = None,
    skip_difficult: bool = False,
    use_exact_rotated_iou: bool = True,
    device: DeviceType = None,
    show_progress: bool = False,
    progress_stream: Optional[Any] = None,
) -> GtBestIouAlignmentMetrics:
    """Per-GT max rotated IoU vs raw detections (global + per-class means).

    For each non-skipped GT, ``best_any`` is the max IoU over all detections on the
    same image; ``best_same`` restricts to same-class detections. Uses exact polygon
    IoU on CPU when ``use_exact_rotated_iou`` is true (same as mAP matching).
    """
    best_any_by_class: Dict[str, List[float]] = defaultdict(list)
    best_same_by_class: Dict[str, List[float]] = defaultdict(list)

    image_ids = sorted(all_ground_truths.keys())
    tqdm_file = progress_stream if progress_stream is not None else sys.stderr
    use_pbar = show_progress and tqdm is not None and len(image_ids) > 200
    img_iter: Any = (
        tqdm(image_ids, desc="GT alignment IoU", unit="img", leave=False, file=tqdm_file)
        if use_pbar
        else image_ids
    )

    for image_id in img_iter:
        ground_truths = all_ground_truths.get(image_id, [])
        raw_detections = all_detections.get(image_id, [])
        if not ground_truths:
            continue

        raw_iou: Optional[list[list[float]]] = None
        if raw_detections:
            gt_rboxes = [g.rbox for g in ground_truths]
            if use_exact_rotated_iou:
                raw_iou = iou.batch_rbox_iou(
                    [d.rbox for d in raw_detections],
                    gt_rboxes,
                    intersection_backend=resolve_exact_polygon_iou_backend(),
                )
            else:
                if torch is None:
                    raw_iou = iou.batch_rbox_iou(
                        [d.rbox for d in raw_detections],
                        gt_rboxes,
                    )
                else:
                    from ..ops.gpu_ops import oriented_box_iou_gpu
                    from ..ops.utils import rboxes_to_tensor

                    dev = device
                    if dev is None:
                        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    ta = rboxes_to_tensor([d.rbox for d in raw_detections], device=dev)
                    tb = rboxes_to_tensor(gt_rboxes, device=dev)
                    raw_iou = oriented_box_iou_gpu(ta, tb).detach().cpu().tolist()

        for gt_idx, gt in enumerate(ground_truths):
            if skip_difficult and gt.difficult != 0:
                continue
            best_any = 0.0
            best_same = 0.0
            if raw_iou is not None:
                for det_idx, det in enumerate(raw_detections):
                    iou_val = raw_iou[det_idx][gt_idx]
                    if iou_val > best_any:
                        best_any = iou_val
                    if detection_matches_ground_truth_class(det, gt) and iou_val > best_same:
                        best_same = iou_val
            cname = gt.class_name or "unknown"
            best_any_by_class[cname].append(best_any)
            best_same_by_class[cname].append(best_same)

    return _aggregate_gt_best_iou_samples(
        best_any_by_class,
        best_same_by_class,
        class_names=class_names,
    )


def gt_best_iou_alignment_metrics_to_dict(
    metrics: GtBestIouAlignmentMetrics,
) -> Dict[str, Any]:
    """JSON-serializable view of :class:`GtBestIouAlignmentMetrics`."""
    return {
        "num_gts": metrics.num_gts,
        "mean_best_iou_any": metrics.mean_best_iou_any,
        "mean_best_iou_same_class": metrics.mean_best_iou_same_class,
        "median_best_iou_any": metrics.median_best_iou_any,
        "median_best_iou_same_class": metrics.median_best_iou_same_class,
        "per_class": {
            name: {
                "num_gts": row.num_gts,
                "mean_best_iou_any": row.mean_best_iou_any,
                "mean_best_iou_same_class": row.mean_best_iou_same_class,
                "median_best_iou_same_class": row.median_best_iou_same_class,
            }
            for name, row in metrics.per_class.items()
        },
    }


def format_gt_best_iou_alignment_table(
    metrics: GtBestIouAlignmentMetrics,
    class_names: Optional[Sequence[str]] = None,
    *,
    markdown: bool = False,
) -> str:
    """Format per-class mean best IoU (same class) table."""
    if not metrics.per_class:
        return ""

    if class_names:
        ordered = [c for c in class_names if c in metrics.per_class]
        ordered.extend(c for c in sorted(metrics.per_class) if c not in ordered)
    else:
        ordered = sorted(metrics.per_class.keys())

    rows = [(name, metrics.per_class[name]) for name in ordered]
    class_w = max(len("class"), *(len(name) for name, _ in rows))
    col_n = max(len("gts"), 3)
    col_any = max(len("mean_any"), 8)
    col_same = max(len("mean_same"), 9)
    col_med = max(len("med_same"), 8)

    if markdown:
        lines = [
            "| Class | gts | mean_any | mean_same | med_same |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name, row in rows:
            lines.append(
                f"| `{name}` | {row.num_gts} | {row.mean_best_iou_any:.4f} | "
                f"{row.mean_best_iou_same_class:.4f} | {row.median_best_iou_same_class:.4f} |"
            )
        lines.append(
            f"| **global** | {metrics.num_gts} | {metrics.mean_best_iou_any:.4f} | "
            f"{metrics.mean_best_iou_same_class:.4f} | {metrics.median_best_iou_same_class:.4f} |"
        )
        return "\n".join(lines)

    def _sep() -> str:
        return (
            f"|{'-' * (class_w + 2)}|{'-' * (col_n + 2)}:|"
            f"{'-' * (col_any + 2)}:|{'-' * (col_same + 2)}:|{'-' * (col_med + 2)}:|"
        )

    lines = [
        f"| {'class'.ljust(class_w)} | {'gts'.rjust(col_n)} | "
        f"{'mean_any'.rjust(col_any)} | {'mean_same'.rjust(col_same)} | "
        f"{'med_same'.rjust(col_med)} |",
        _sep(),
    ]
    for name, row in rows:
        lines.append(
            f"| {name.ljust(class_w)} | {row.num_gts:>{col_n}d} | "
            f"{row.mean_best_iou_any:>{col_any}.4f} | {row.mean_best_iou_same_class:>{col_same}.4f} | "
            f"{row.median_best_iou_same_class:>{col_med}.4f} |"
        )
    lines.append(
        f"| {'global'.ljust(class_w)} | {metrics.num_gts:>{col_n}d} | "
        f"{metrics.mean_best_iou_any:>{col_any}.4f} | {metrics.mean_best_iou_same_class:>{col_same}.4f} | "
        f"{metrics.median_best_iou_same_class:>{col_med}.4f} |"
    )
    return "\n".join(lines)


def format_gt_best_iou_alignment_table_from_dict(
    data: Dict[str, Any],
    class_names: Optional[Sequence[str]] = None,
    *,
    markdown: bool = False,
) -> str:
    """Format table from :func:`gt_best_iou_alignment_metrics_to_dict` output."""
    per_class_raw = data.get("per_class") or {}
    if not per_class_raw:
        return ""
    per_class = {
        name: ClassGtBestIouMetrics(
            num_gts=int(row.get("num_gts", 0)),
            mean_best_iou_any=float(row.get("mean_best_iou_any", 0.0)),
            mean_best_iou_same_class=float(row.get("mean_best_iou_same_class", 0.0)),
            median_best_iou_same_class=float(row.get("median_best_iou_same_class", 0.0)),
        )
        for name, row in per_class_raw.items()
    }
    metrics = GtBestIouAlignmentMetrics(
        num_gts=int(data.get("num_gts", 0)),
        mean_best_iou_any=float(data.get("mean_best_iou_any", 0.0)),
        mean_best_iou_same_class=float(data.get("mean_best_iou_same_class", 0.0)),
        median_best_iou_any=float(data.get("median_best_iou_any", 0.0)),
        median_best_iou_same_class=float(data.get("median_best_iou_same_class", 0.0)),
        per_class=per_class,
    )
    return format_gt_best_iou_alignment_table(metrics, class_names, markdown=markdown)


def format_mmrotate_class_metrics_table(
    class_metrics: Dict[str, ClassEvalMetrics],
    class_names: Optional[Sequence[str]] = None,
    *,
    mean_ap: Optional[float] = None,
) -> str:
    """Format per-class gts / dets / recall / ap table like MMRotate eval output."""
    if not class_metrics:
        return ""

    if class_names:
        ordered = [c for c in class_names if c in class_metrics]
        ordered.extend(c for c in sorted(class_metrics) if c not in ordered)
    else:
        ordered = sorted(class_metrics.keys())

    rows: List[Tuple[str, ClassEvalMetrics]] = [
        (name, class_metrics[name]) for name in ordered
    ]
    class_w = max(len("class"), *(len(name) for name, _ in rows))
    col_gts = max(len("gts"), 3)
    col_dets = max(len("dets"), 4)
    col_recall = max(len("recall"), 6)
    col_ap = max(len("ap"), 2)

    def _sep() -> str:
        return (
            f"|{'-' * (class_w + 2)}|{'-' * (col_gts + 2)}:|"
            f"{'-' * (col_dets + 2)}:|{'-' * (col_recall + 2)}:|{'-' * (col_ap + 2)}:|"
        )

    lines = [
        f"| {'class'.ljust(class_w)} | {'gts'.rjust(col_gts)} | "
        f"{'dets'.rjust(col_dets)} | {'recall'.rjust(col_recall)} | {'ap'.rjust(col_ap)} |",
        _sep(),
    ]
    for name, m in rows:
        lines.append(
            f"| {name.ljust(class_w)} | {m.num_gts:>{col_gts}d} | {m.num_dets:>{col_dets}d} | "
            f"{m.recall:>{col_recall}.3f} | {m.ap:>{col_ap}.3f} |"
        )
    if mean_ap is not None:
        lines.append(
            f"| {'mAP'.ljust(class_w)} | {'':>{col_gts}} | {'':>{col_dets}} | "
            f"{'':>{col_recall}} | {mean_ap:>{col_ap}.3f} |"
        )
    return "\n".join(lines)


__all__ = [
    "Detection",
    "GroundTruth",
    "ClassEvalMetrics",
    "ClassGtBestIouMetrics",
    "GtBestIouAlignmentMetrics",
    "APCalculator",
    "compute_oriented_map",
    "compute_gt_best_iou_alignment_metrics",
    "detection_matches_ground_truth_class",
    "format_gt_best_iou_alignment_table",
    "format_gt_best_iou_alignment_table_from_dict",
    "format_mmrotate_class_metrics_table",
    "gt_best_iou_alignment_metrics_to_dict",
]

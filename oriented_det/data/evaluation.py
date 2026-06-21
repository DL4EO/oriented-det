"""Oriented mAP evaluation compatible with DOTA protocol.

Uses detection and ground-truth RBoxes exactly as provided (no coordinate rescaling).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
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
    
    def compute_map(
        self,
        all_detections: Dict[str, List[Detection]],  # image_id -> detections
        all_ground_truths: Dict[str, List[GroundTruth]],  # image_id -> ground_truths
        show_progress: bool = True,
        progress_stream: Optional[Any] = None,
        max_iou_calculations_per_class: int = 10_000_000,  # 10M IoU calculations max per class
        min_box_size: float = 1.0,
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
) -> Tuple[float, Dict[str, float], Dict[str, ClassEvalMetrics]]:
    """Compute mean Average Precision (mAP) for oriented detection.
    
    Args:
        detections: Dictionary mapping image_id -> list of Detection objects
        ground_truths: Dictionary mapping image_id -> list of GroundTruth objects
        iou_threshold: IoU threshold for positive matches
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
            sampling IoU (controlled by ``evaluation.use_exact_rotated_iou`` in training config).
    
    Returns:
        Tuple of (mAP, class_ap_dict, class_metrics) where mAP is the mean of all
        class APs and ``class_metrics`` holds per-class gts/dets/recall (MMRotate-style).
    """
    if use_exact_rotated_iou and show_progress:
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
    )

    if not class_aps:
        return 0.0, {}, {}

    mean_ap = sum(class_aps.values()) / len(class_aps)
    return mean_ap, class_aps, class_metrics


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
    "APCalculator",
    "compute_oriented_map",
    "format_mmrotate_class_metrics_table",
]

#!/usr/bin/env python
"""
Gradio app to browse inference results (predictions mode) or explore a DOTA
dataset without predictions (dataset mode). Supports tiled and original
DOTA directory layouts.
"""

import os
import json
import argparse
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# Patch Gradio to handle schema errors
try:
    from gradio_client import utils as gradio_client_utils
    
    # Patch _json_schema_to_python_type to handle boolean schemas
    original_json_schema = gradio_client_utils._json_schema_to_python_type
    
    def patched_json_schema(schema, defs=None):
        """Patched version that handles boolean and invalid schemas"""
        if isinstance(schema, bool):
            return "Any"
        if not isinstance(schema, dict):
            return "Any"
        try:
            return original_json_schema(schema, defs)
        except Exception:
            return "Any"
    
    gradio_client_utils._json_schema_to_python_type = patched_json_schema
    
    # Also patch get_type
    if hasattr(gradio_client_utils, 'get_type'):
        original_get_type = gradio_client_utils.get_type
        
        def patched_get_type(schema):
            if isinstance(schema, bool):
                return "Any"
            if not isinstance(schema, dict):
                return "Any"
            try:
                return original_get_type(schema)
            except Exception:
                return "Any"
        
        gradio_client_utils.get_type = patched_get_type
except Exception:
    pass  # If patching fails, continue anyway

import gradio as gr
import cv2
import numpy as np
from PIL import Image

# Add project root to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from oriented_det.geometry import RBox
from oriented_det.data import DOTAAnnotation


# Image extensions to scan in dataset mode
_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def detect_data_root() -> str:
    """Detect the data root directory - placeholder for now."""
    # Try to detect from common locations or use environment variable
    data_root = os.environ.get('DOTA_DATA_ROOT', '/mnt/data/share/DOTA-v2.0')
    return data_root


def list_dota_images(data_root: str, tiles_dir: Optional[str] = None) -> List[Dict]:
    """Build a list of image entries by scanning a DOTA dataset directory.

    Supports both tiled layout (e.g. train/tiles_1024/images) and flat layout
    (e.g. train/images or train/ with images directly inside).

    Args:
        data_root: Root path of the dataset (e.g. /path/to/dota).
        tiles_dir: Optional subpath (e.g. 'train', 'val', 'train/tiles_1024').
            If None, scans data_root directly.

    Returns:
        List of dicts with keys: image_path (relative to data_root), and
        optionally image_width, image_height (left unset; filled lazily in get_images).
    """
    root = Path(data_root)
    base = root if not tiles_dir else root / tiles_dir
    images_dir = base / "images" if (base / "images").is_dir() else base
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    entries = []
    for path in sorted(images_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS:
            try:
                rel = path.relative_to(root)
            except ValueError:
                rel = path
            entries.append({
                "image_path": str(rel).replace(os.sep, '/'),
                "predictions": [],
            })
    return entries


def format_stats_markdown(metadata: Dict, result: Optional[Dict], class_names: Optional[List[str]] = None) -> str:
    """Format diagnostics (global) and per-image stats as markdown for the Stats tab."""
    class_names = class_names or []
    lines = []
    diag = metadata.get("diagnostics")
    if diag:
        lines.append("## Global diagnostics")
        if isinstance(diag.get('mAP'), (int, float)):
            lines.append(f"- **mAP** (IoU={diag.get('iou_threshold', 0.5)}): {diag['mAP']:.4f}")
        else:
            lines.append(f"- **mAP**: {diag.get('mAP_error', 'N/A')}")
        lines.append(f"- Pipeline: raw → after threshold → after NMS: **{diag.get('total_raw', 0)}** → **{diag.get('total_after_threshold', 0)}** → **{diag.get('total_after_nms', 0)}**")
        if diag.get('score_min') is not None:
            lines.append(f"- Score (all images): min={diag['score_min']:.4f} max={diag['score_max']:.4f} mean={diag['score_mean']:.4f}")
        if diag.get('class_aps'):
            lines.append("\n**Per-class AP:**")
            for cname, ap in sorted(diag["class_aps"].items()):
                lines.append(f"  - {cname}: {ap:.4f}")
        best_f1 = metadata.get("best_threshold_f1") or metadata.get("best_threshold_f2")
        if isinstance(best_f1, dict) and best_f1:
            lines.append("\n**Best threshold by F1:**")
            lines.append(
                f"  - threshold={best_f1.get('threshold', 0.0):.4f}, "
                f"precision={best_f1.get('precision', 0.0):.4f}, "
                f"recall={best_f1.get('recall', 0.0):.4f}, "
                f"F1={best_f1.get('f1', 0.0):.4f}"
            )
        lines.append("")
    if result and result.get("stats"):
        st = result["stats"]
        lines.append("## This image")
        lines.append(f"- Raw → after threshold → after NMS: **{st.get('num_raw', 0)}** → **{st.get('num_after_threshold', 0)}** → **{st.get('num_after_nms', 0)}**")
        if st.get("score_min") is not None:
            lines.append(f"- Score: min={st['score_min']:.4f} max={st['score_max']:.4f} mean={st['score_mean']:.4f}")
        def _class_label(k, names):
            ki = int(k) if isinstance(k, str) and str(k).lstrip("-").isdigit() else (k if isinstance(k, int) else -1)
            return names[ki] if isinstance(ki, int) and 0 <= ki < len(names) else f"class_{k}"

        if st.get("per_class_raw"):
            items = st["per_class_raw"].items()
            sorted_items = sorted(items, key=lambda kv: (0, kv[0]) if isinstance(kv[0], int) else (1, str(kv[0])))
            lines.append("\n**Per-class (raw):** " + ", ".join(f"{_class_label(k, class_names)}:{v}" for k, v in sorted_items))
        if st.get("per_class_final"):
            items = st["per_class_final"].items()
            sorted_items = sorted(items, key=lambda kv: (0, kv[0]) if isinstance(kv[0], int) else (1, str(kv[0])))
            lines.append("\n**Per-class (after NMS):** " + ", ".join(f"{_class_label(k, class_names)}:{v}" for k, v in sorted_items))
    if not lines:
        return (
            "No diagnostics in this predictions directory. "
            "`make preds` writes inference only; run **`make metrics`** on that output folder "
            "(or **`make metrics`**) to compute mAP / PR and per-image stats, then reopen the viewer."
        )
    return "\n".join(lines)


def load_dota_annotations(txt_file: str) -> Tuple[np.ndarray, List[str]]:
    """Load DOTA format annotations from txt file
    Returns: (boxes array, class_names list)"""
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
            except Exception:
                continue
    
    if len(rboxes) == 0:
        return np.array([]).reshape(0, 5), []
    
    # Convert to array format (cx, cy, w, h, angle)
    rbox_array = np.array([[rbox.cx, rbox.cy, rbox.width, rbox.height, rbox.angle] 
                          for rbox in rboxes])
    return rbox_array, class_names


def array_to_rbox(bbox_array: List[float]) -> RBox:
    """Convert array format [cx, cy, w, h, angle] to RBox."""
    return RBox(cx=bbox_array[0], cy=bbox_array[1], 
                width=bbox_array[2], height=bbox_array[3], 
                angle=bbox_array[4])


def _canonicalize_name(name: str) -> str:
    return str(name).strip().lower()


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _slider_index(value: Any, default: int = 0) -> int:
    """Coerce Gradio slider/state payloads; None fails Slider preprocess (min bound check)."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class SimpleViewer:
    """Unified viewer for inference results or dataset-only exploration."""

    def __init__(
        self,
        predictions_dir: Optional[str] = None,
        data_root: Optional[str] = None,
        results: Optional[List[Dict]] = None,
    ):
        self.current_idx = 0
        self.sorted_indices = []
        self.current_sort = "no order"
        self.last_thresh_signature = None
        self._gt_count_cache: Dict[int, int] = {}  # for dataset mode sort by object count

        if predictions_dir is not None:
            self._init_from_predictions(predictions_dir, data_root)
        elif results is not None and data_root is not None:
            self._init_from_dataset(results, data_root)
        else:
            raise ValueError(
                "Provide either predictions_dir (and optional data_root) "
                "or results and data_root for dataset mode."
            )

    def _init_from_predictions(self, predictions_dir: str, data_root: Optional[str]) -> None:
        self.has_predictions = True
        self.predictions_dir = predictions_dir
        self.json_path = os.path.join(predictions_dir, 'predictions.json')
        if not os.path.exists(self.json_path):
            raise ValueError(f"predictions.json not found in {predictions_dir}")
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        self.metadata = data['metadata']
        self.results = data['results']
        self.class_names = list(self.metadata.get("class_names") or [])
        best_meta = self.metadata.get("best_threshold_f1") or self.metadata.get("best_threshold_f2")
        best_thr = (
            best_meta.get("threshold")
            if isinstance(best_meta, dict)
            else None
        )
        if isinstance(best_thr, (int, float)):
            self.conf_threshold = float(best_thr)
        else:
            self.conf_threshold = _to_float(
                self.metadata.get('score_threshold', self.metadata.get('conf_threshold', 0.3)),
                0.3,
            )
        self.per_class_threshold_defaults = {
            cname: float(self.conf_threshold) for cname in self.class_names
        }
        self.best_threshold_per_class: Dict[str, Dict[str, Any]] = {}
        self._set_defaults_from_config_thresholds()
        if data_root is not None:
            self.data_root = data_root
        elif 'data_root' in self.metadata:
            self.data_root = self.metadata['data_root']
        else:
            self.data_root = detect_data_root()
        self.f1_scores = {}
        self.load_f1_scores()
        self._set_defaults_from_analysis_thresholds()
        if self.best_threshold_per_class:
            self.metadata["best_threshold_per_class"] = self.best_threshold_per_class
        self.sorted_indices = list(range(len(self.results)))
        print(f"Using data root: {self.data_root}")
        print(f"Loaded {len(self.results)} images from {predictions_dir}")

    def _init_from_dataset(self, results: List[Dict], data_root: str) -> None:
        self.has_predictions = False
        self.predictions_dir = None
        self.metadata = {}
        self.results = results
        self.conf_threshold = 0.3
        self.class_names = []
        self.per_class_threshold_defaults = {}
        self.best_threshold_per_class = {}
        self.data_root = data_root
        self.f1_scores = {}
        self.sorted_indices = list(range(len(self.results)))
        print(f"Using data root: {self.data_root}")
        print(f"Dataset mode: loaded {len(self.results)} images")

    def _set_defaults_from_config_thresholds(self) -> None:
        raw = self.metadata.get("per_class_score_threshold")
        if not isinstance(raw, dict):
            return
        by_key = {_canonicalize_name(k): _to_float(v, self.conf_threshold) for k, v in raw.items()}
        for cname in self.class_names:
            thr = by_key.get(_canonicalize_name(cname))
            if thr is not None:
                self.per_class_threshold_defaults[cname] = float(thr)

    def _set_defaults_from_analysis_thresholds(self) -> None:
        if not isinstance(self.best_threshold_per_class, dict):
            return
        by_key = {_canonicalize_name(k): v for k, v in self.best_threshold_per_class.items()}
        for cname in self.class_names:
            entry = by_key.get(_canonicalize_name(cname))
            if isinstance(entry, dict) and "threshold" in entry:
                self.per_class_threshold_defaults[cname] = _to_float(entry.get("threshold"), self.conf_threshold)
    
    def load_f1_scores(self):
        """Load F1 scores from analysis JSON if available"""
        analysis_file = self.metadata.get("analysis_file", "analysis_iou0.50.json")
        analysis_path = (
            analysis_file
            if os.path.isabs(str(analysis_file))
            else os.path.join(self.predictions_dir, str(analysis_file))
        )
        if os.path.exists(analysis_path):
            try:
                with open(analysis_path, 'r') as f:
                    analysis_data = json.load(f)
                per_image_metrics = analysis_data.get('per_image_metrics', [])
                for metric in per_image_metrics:
                    image_name = metric.get('image_name')
                    f1 = metric.get('f1', 0.0)
                    if image_name:
                        self.f1_scores[image_name] = float(f1)
                best_per_class = analysis_data.get("best_threshold_per_class") or {}
                if isinstance(best_per_class, dict):
                    self.best_threshold_per_class = best_per_class
                print(f"Loaded F1 scores for {len(self.f1_scores)} images from analysis JSON")
            except Exception as e:
                print(f"Warning: Could not load F1 scores from analysis JSON: {e}")

    def _effective_threshold_for_pred(
        self,
        pred: Dict[str, Any],
        threshold: float,
        use_per_class_thresholds: bool = False,
        per_class_thresholds: Optional[Dict[str, float]] = None,
    ) -> float:
        if not use_per_class_thresholds or not per_class_thresholds:
            return float(threshold)
        cls_name = str(pred.get("class_name", ""))
        if cls_name in per_class_thresholds:
            return float(per_class_thresholds[cls_name])
        key = _canonicalize_name(cls_name)
        if key in per_class_thresholds:
            return float(per_class_thresholds[key])
        return float(threshold)

    def get_detection_count(
        self,
        idx: int,
        threshold: float,
        use_per_class_thresholds: bool = False,
        per_class_thresholds: Optional[Dict[str, float]] = None,
    ) -> int:
        """Get number of detections for an image at given threshold."""
        result = self.results[idx]
        count = 0
        for pred in result.get('predictions', []):
            thr = self._effective_threshold_for_pred(
                pred,
                threshold,
                use_per_class_thresholds=use_per_class_thresholds,
                per_class_thresholds=per_class_thresholds,
            )
            if pred['score'] >= thr:
                count += 1
        return count

    def get_stats_markdown(self, display_idx: int) -> str:
        """Get stats markdown for the image at display_idx (after sorting)."""
        if display_idx < 0 or display_idx >= len(self.sorted_indices):
            return ""
        actual_idx = self.sorted_indices[display_idx]
        result = self.results[actual_idx]
        class_names = self.metadata.get("class_names") or []
        return format_stats_markdown(self.metadata, result, class_names)

    def get_image_label_markdown(self, display_idx: int) -> str:
        """Return markdown identifying the current image (path or name)."""
        if display_idx < 0 or display_idx >= len(self.sorted_indices):
            return ""
        actual_idx = self.sorted_indices[display_idx]
        result = self.results[actual_idx]
        rel_path = str(result.get("image_path", "")).replace(os.sep, "/")
        name = result.get("image_name")
        if rel_path:
            display = rel_path
        elif name:
            display = name
        else:
            return ""
        idx_str = f"{display_idx + 1} / {len(self.sorted_indices)}"
        return f"**Image {idx_str}:** `{display}`"

    def _get_gt_count(self, idx: int) -> int:
        """Get number of ground-truth objects for an image (for dataset mode sorting)."""
        if idx in self._gt_count_cache:
            return self._gt_count_cache[idx]
        result = self.results[idx]
        relative_path = result['image_path']
        img_path = os.path.join(self.data_root, relative_path) if not os.path.isabs(relative_path) else relative_path
        img_path_obj = Path(img_path)
        possible_label_dirs = [
            img_path_obj.parent.parent / 'labels',
            img_path_obj.parent.parent / 'labelTxt',
            img_path_obj.parent / 'labels',
        ]
        txt_path = None
        for label_dir in possible_label_dirs:
            potential_txt = label_dir / f"{img_path_obj.stem}.txt"
            if potential_txt.exists():
                txt_path = potential_txt
                break
        if txt_path is None:
            txt_path = img_path_obj.with_suffix('.txt')
        gt_boxes, _ = load_dota_annotations(str(txt_path))
        count = len(gt_boxes)
        self._gt_count_cache[idx] = count
        return count

    def apply_sorting(
        self,
        sort_mode: str,
        threshold: float,
        use_per_class_thresholds: bool = False,
        per_class_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        """Apply sorting to results and update sorted_indices."""
        self.current_sort = sort_mode
        self.last_thresh_signature = (
            float(threshold),
            bool(use_per_class_thresholds),
            tuple(sorted((str(k), float(v)) for k, v in (per_class_thresholds or {}).items())),
        )

        if sort_mode == "no order":
            self.sorted_indices = list(range(len(self.results)))
        elif not self.has_predictions:
            # Dataset mode: only no order and object count
            if sort_mode == "objs asc":
                indices_with_objs = [(idx, self._get_gt_count(idx)) for idx in range(len(self.results))]
                indices_with_objs.sort(key=lambda x: x[1])
                self.sorted_indices = [idx for idx, _ in indices_with_objs]
            elif sort_mode == "objs desc":
                indices_with_objs = [(idx, self._get_gt_count(idx)) for idx in range(len(self.results))]
                indices_with_objs.sort(key=lambda x: x[1], reverse=True)
                self.sorted_indices = [idx for idx, _ in indices_with_objs]
            else:
                self.sorted_indices = list(range(len(self.results)))
        else:
            # Predictions mode: F1 and det count
            if sort_mode == "F1 asc":
                indices_with_f1 = []
                for idx in range(len(self.results)):
                    result = self.results[idx]
                    image_name = result.get('image_name', Path(self.results[idx]['image_path']).stem)
                    f1 = self.f1_scores.get(image_name, 0.0)
                    indices_with_f1.append((idx, f1))
                indices_with_f1.sort(key=lambda x: x[1])
                self.sorted_indices = [idx for idx, _ in indices_with_f1]
            elif sort_mode == "F1 desc":
                indices_with_f1 = []
                for idx in range(len(self.results)):
                    result = self.results[idx]
                    image_name = result.get('image_name', Path(self.results[idx]['image_path']).stem)
                    f1 = self.f1_scores.get(image_name, 0.0)
                    indices_with_f1.append((idx, f1))
                indices_with_f1.sort(key=lambda x: x[1], reverse=True)
                self.sorted_indices = [idx for idx, _ in indices_with_f1]
            elif sort_mode == "dets asc":
                indices_with_dets = [
                    (
                        idx,
                        self.get_detection_count(
                            idx,
                            threshold,
                            use_per_class_thresholds=use_per_class_thresholds,
                            per_class_thresholds=per_class_thresholds,
                        ),
                    )
                    for idx in range(len(self.results))
                ]
                indices_with_dets.sort(key=lambda x: x[1])
                self.sorted_indices = [idx for idx, _ in indices_with_dets]
            elif sort_mode == "dets desc":
                indices_with_dets = [
                    (
                        idx,
                        self.get_detection_count(
                            idx,
                            threshold,
                            use_per_class_thresholds=use_per_class_thresholds,
                            per_class_thresholds=per_class_thresholds,
                        ),
                    )
                    for idx in range(len(self.results))
                ]
                indices_with_dets.sort(key=lambda x: x[1], reverse=True)
                self.sorted_indices = [idx for idx, _ in indices_with_dets]
            else:
                self.sorted_indices = list(range(len(self.results)))
    
    def get_images(
        self,
        idx: int,
        threshold: float,
        zoom_scale: float = 2.0,
        show_labels: bool = True,
        view_mode: str = "preds_gt",
        use_per_class_thresholds: bool = False,
        per_class_thresholds: Optional[Dict[str, float]] = None,
    ) -> Image.Image:
        """Render one view of the current image.

        view_mode:
            - ``image``: original image only
            - ``gt``: ground truth only (green boxes)
            - ``preds``: predictions only (red boxes)
            - ``preds_gt``: GT (green) then predictions (red) on top

        Args:
            idx: Displayed index (after sorting)
            threshold: Confidence threshold for predictions
            zoom_scale: Zoom scale factor
        """
        if idx < 0 or idx >= len(self.sorted_indices):
            idx = 0
        
        self.current_idx = idx
        # Map displayed index to actual result index
        actual_idx = self.sorted_indices[idx]
        result = self.results[actual_idx]
        
        # Load image
        relative_path = result['image_path']
        if os.path.isabs(relative_path):
            img_path = relative_path
        else:
            img_path = os.path.join(self.data_root, relative_path)
        
        img = cv2.imread(img_path)
        if img is None:
            return Image.new('RGB', (800, 600), color='red')
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width = img_rgb.shape[:2]
        
        # Scale from preprocessed (model) space to the image we're drawing on.
        # Use the actual loaded image dimensions so boxes always match the displayed image
        # (avoids mismatch when JSON has different image_width/height than the file on disk).
        # Match training/save_predictions: fixed resize to (target_h, target_w).
        resize_mode = result.get('resize_mode', 'fixed')
        target = result.get('target_size', [1024, 1024])
        if isinstance(target, (list, tuple)):
            target_h, target_w = int(target[0]), int(target[1])
        else:
            target_h = target_w = int(target)
        pad_offset_x = pad_offset_y = 0
        if resize_mode == 'pad':
            from oriented_det.data.preprocessing import build_spatial_meta_from_dims
            meta = build_spatial_meta_from_dims('pad', img_width, img_height, target)
            inv = 1.0 / max(meta.scale, 1e-8)
            scale_back_x = scale_back_y = inv
            pad_offset_x = meta.pad_left
            pad_offset_y = meta.pad_top
        elif resize_mode == 'crop':
            from oriented_det.data.preprocessing import build_spatial_meta_from_dims
            meta = build_spatial_meta_from_dims('crop', img_width, img_height, target)
            scale_back_x = scale_back_y = 1.0
            pad_offset_x = meta.pad_left - meta.crop_left
            pad_offset_y = meta.pad_top - meta.crop_top
        else:
            scale_back_x = img_width / max(target_w, 1)
            scale_back_y = img_height / max(target_h, 1)

        # save_predictions (pad/tile inference): bboxes are already in the loaded image's pixel space.
        if self.metadata.get('bbox_coordinate_space') == 'image_pixels':
            scale_back_x = scale_back_y = 1.0
            pad_offset_x = pad_offset_y = 0
        
        # Predictions: red (BGR); ground truth: green
        pred_color_bgr = (0, 0, 255)  # red in BGR
        gt_colors = {
            'car': (0, 255, 0),
            'truck': (0, 255, 0),
        }

        # Ground truth boxes (prefer GT serialized in predictions.json; fallback to label files)
        gt_items = result.get("ground_truths", [])
        if gt_items:
            gt_boxes = np.array([g.get("bbox", [0, 0, 0, 0, 0]) for g in gt_items], dtype=np.float32)
            gt_class_names = [g.get("class_name", "unknown") for g in gt_items]
        else:
            img_path_obj = Path(img_path)
            possible_label_dirs = [
                img_path_obj.parent.parent / 'labels',
                img_path_obj.parent.parent / 'labelTxt',
                img_path_obj.parent / 'labels',
            ]

            txt_path = None
            for label_dir in possible_label_dirs:
                potential_txt = label_dir / f"{img_path_obj.stem}.txt"
                if potential_txt.exists():
                    txt_path = potential_txt
                    break

            if txt_path is None:
                txt_path = img_path_obj.with_suffix('.txt')

            gt_boxes, gt_class_names = load_dota_annotations(str(txt_path))

        if view_mode == "image":
            out_rgb = img_rgb
        else:
            out_bgr = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR)
            pred_font = cv2.FONT_HERSHEY_SIMPLEX
            pred_font_scale = 0.4
            pred_font_thickness = 1
            pred_label_bg_bgr = (0, 0, 0)
            pred_label_fg_bgr = (255, 255, 255)
            gt_fill_alpha = 0.25
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.35
            font_thickness = 1
            label_bg_bgr = (0, 0, 0)
            label_fg_bgr = (255, 255, 255)

            draw_gt = view_mode in ("gt", "preds_gt")
            draw_preds = view_mode in ("preds", "preds_gt") and self.has_predictions

            if draw_gt:
                gt_draw_items = []
                gt_overlay = out_bgr.copy()
                for i, gt_box in enumerate(gt_boxes):
                    rbox = array_to_rbox(gt_box)
                    polygon = rbox.to_polygon()
                    points = np.array([list(p) for p in polygon.points], dtype=np.int32)
                    class_name = gt_class_names[i] if i < len(gt_class_names) else 'unknown'
                    color_rgb = gt_colors.get(class_name, (0, 255, 0))
                    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
                    gt_draw_items.append((points, color_bgr))
                    # Fill GT polygons lightly so overlaps with predictions are easier to inspect.
                    cv2.fillPoly(gt_overlay, [points], color_bgr)
                    if show_labels:
                        (text_w, text_h), _ = cv2.getTextSize(class_name, font, font_scale, font_thickness)
                        right_x = int(np.max(points[:, 0]))
                        center_y = int(np.mean(points[:, 1]))
                        tx = max(0, min(right_x + 4, img_width - text_w - 4))
                        ty = max(text_h + 2, min(center_y, img_height - 2))
                        cv2.rectangle(
                            out_bgr, (tx - 2, ty - text_h - 2), (tx + text_w + 2, ty + 2),
                            label_bg_bgr, -1,
                        )
                        cv2.putText(out_bgr, class_name, (tx, ty), font, font_scale, label_fg_bgr, font_thickness)
                out_bgr = cv2.addWeighted(gt_overlay, gt_fill_alpha, out_bgr, 1.0 - gt_fill_alpha, 0)
                for points, color_bgr in gt_draw_items:
                    cv2.polylines(out_bgr, [points], isClosed=True, color=color_bgr, thickness=2)

            if draw_preds:
                pred_draw_items = []
                for pred in result.get("predictions", []):
                    class_name = pred.get('class_name', 'car')
                    effective_thr = self._effective_threshold_for_pred(
                        pred,
                        threshold,
                        use_per_class_thresholds=use_per_class_thresholds,
                        per_class_thresholds=per_class_thresholds,
                    )
                    if pred['score'] < effective_thr:
                        continue
                    bbox = pred['bbox']
                    rbox = array_to_rbox(bbox)
                    rbox_scaled = RBox(
                        cx=(rbox.cx - pad_offset_x) * scale_back_x,
                        cy=(rbox.cy - pad_offset_y) * scale_back_y,
                        width=rbox.width * scale_back_x,
                        height=rbox.height * scale_back_y,
                        angle=rbox.angle,
                    )
                    polygon = rbox_scaled.to_polygon()
                    points = np.array([list(p) for p in polygon.points], dtype=np.int32)
                    pred_draw_items.append((points, pred_color_bgr))
                    if show_labels:
                        label_text = f"{class_name}: {float(pred['score']):.2f}"
                        (tw, th), _ = cv2.getTextSize(label_text, pred_font, pred_font_scale, pred_font_thickness)
                        right_x = int(np.max(points[:, 0]))
                        center_y = int(np.mean(points[:, 1]))
                        tx = max(0, min(right_x + 4, img_width - tw - 4))
                        ty = max(th + 2, min(center_y, img_height - 2))
                        cv2.rectangle(
                            out_bgr,
                            (tx - 2, ty - th - 2),
                            (tx + tw + 2, ty + 2),
                            pred_label_bg_bgr,
                            -1,
                        )
                        cv2.putText(
                            out_bgr,
                            label_text,
                            (tx, ty),
                            pred_font,
                            pred_font_scale,
                            pred_label_fg_bgr,
                            pred_font_thickness,
                        )
                for points, color_bgr in pred_draw_items:
                    cv2.polylines(out_bgr, [points], isClosed=True, color=color_bgr, thickness=2)

            out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

        def apply_zoom(img_array: np.ndarray, scale: float) -> Image.Image:
            if scale == 1.0:
                return Image.fromarray(img_array, "RGB")
            h, w = img_array.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            return Image.fromarray(resized, "RGB")

        return apply_zoom(out_rgb, zoom_scale)


def create_app(
    mode: str = "predictions",
    predictions_dir: Optional[str] = None,
    data_root: Optional[str] = None,
    tiles_dir: Optional[str] = None,
    threshold: Optional[float] = None,
):
    """Create Gradio app for predictions viewer or dataset explorer.

    Args:
        mode: "predictions" or "dataset"
        predictions_dir: Path to predictions directory (required when mode is "predictions")
        data_root: Path to data root (required for dataset mode; optional for predictions)
        tiles_dir: Subpath for dataset (e.g. "train", "train/tiles_1024") when mode is "dataset"
        threshold: Confidence threshold for predictions mode (default: 0.3)
    """
    if mode == "dataset":
        if not data_root:
            raise ValueError("Dataset mode requires --data-root")
        results = list_dota_images(data_root, tiles_dir)
        if not results:
            raise ValueError(f"No images found under {data_root}" + (f" / {tiles_dir}" if tiles_dir else ""))
        viewer = SimpleViewer(results=results, data_root=data_root)
    else:
        if not predictions_dir:
            raise ValueError("Predictions mode requires --predictions-dir")
        viewer = SimpleViewer(predictions_dir=predictions_dir, data_root=data_root)
        if threshold is not None:
            viewer.conf_threshold = threshold

    def _build_per_class_thresholds(values: Tuple[Any, ...]) -> Dict[str, float]:
        if not viewer.class_names:
            return {}
        return {
            cname: _to_float(v, viewer.conf_threshold)
            for cname, v in zip(viewer.class_names, values)
        }

    def _maybe_sort(
        sort_mode: str,
        threshold_val: float,
        use_per_class_val: bool = False,
        per_class_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        needs_resort = viewer.current_sort != sort_mode
        if viewer.has_predictions and sort_mode in ["dets asc", "dets desc"]:
            threshold_signature = (
                float(threshold_val),
                bool(use_per_class_val),
                tuple(sorted((str(k), float(v)) for k, v in (per_class_thresholds or {}).items())),
            )
            if viewer.last_thresh_signature != threshold_signature:
                needs_resort = True
        if needs_resort:
            viewer.apply_sorting(
                sort_mode,
                float(threshold_val),
                use_per_class_thresholds=use_per_class_val,
                per_class_thresholds=per_class_thresholds,
            )

    def refresh_prediction_tabs(
        idx: int,
        threshold_val: float,
        zoom_str: str,
        sort_mode: str,
        show_labels_val: bool,
        use_per_class_val: bool,
        *per_class_vals: Any,
    ):
        per_class_thresholds = _build_per_class_thresholds(per_class_vals)
        _maybe_sort(sort_mode, threshold_val, use_per_class_val, per_class_thresholds)
        zoom = float(zoom_str.replace("x", ""))
        i = int(idx)
        img_pg = viewer.get_images(
            i,
            float(threshold_val),
            zoom,
            show_labels_val,
            view_mode="preds_gt",
            use_per_class_thresholds=use_per_class_val,
            per_class_thresholds=per_class_thresholds,
        )
        img_p = viewer.get_images(
            i,
            float(threshold_val),
            zoom,
            show_labels_val,
            view_mode="preds",
            use_per_class_thresholds=use_per_class_val,
            per_class_thresholds=per_class_thresholds,
        )
        img_g = viewer.get_images(
            i,
            float(threshold_val),
            zoom,
            show_labels_val,
            view_mode="gt",
            use_per_class_thresholds=use_per_class_val,
            per_class_thresholds=per_class_thresholds,
        )
        img_i = viewer.get_images(
            i,
            float(threshold_val),
            zoom,
            show_labels_val,
            view_mode="image",
            use_per_class_thresholds=use_per_class_val,
            per_class_thresholds=per_class_thresholds,
        )
        stats_md = viewer.get_stats_markdown(i)
        image_label = viewer.get_image_label_markdown(i)
        return img_pg, img_p, img_g, img_i, stats_md, image_label

    def _tabs_and_slider(
        idx: int,
        threshold_val: float,
        zoom_str: str,
        sort_mode: str,
        show_labels_val: bool,
        use_per_class_val: bool,
        slider_idx: int,
        *per_class_vals: Any,
    ):
        """Same as ``refresh_prediction_tabs`` but order matches Gradio: 4 images, slider, then markdown."""
        img_pg, img_p, img_g, img_i, stats_md, image_label = refresh_prediction_tabs(
            idx,
            threshold_val,
            zoom_str,
            sort_mode,
            show_labels_val,
            use_per_class_val,
            *per_class_vals,
        )
        idx = _slider_index(slider_idx, 0)
        return img_pg, img_p, img_g, img_i, idx, idx, stats_md, image_label

    def refresh_dataset_tabs(idx: int, zoom_str: str, sort_mode: str):
        _maybe_sort(sort_mode, threshold_for_api)
        zoom = float(zoom_str.replace("x", ""))
        i = int(idx)
        img_g = viewer.get_images(i, threshold_for_api, zoom, show_labels=False, view_mode="gt")
        img_i = viewer.get_images(i, threshold_for_api, zoom, show_labels=False, view_mode="image")
        image_label = viewer.get_image_label_markdown(i)
        return img_g, img_i, image_label

    def build_download_image_predictions(
        idx: int,
        threshold_val: float,
        zoom_str: str,
        sort_mode: str,
        show_labels_val: bool,
        use_per_class_val: bool,
        active_view_mode: str,
        *per_class_vals: Any,
    ) -> str:
        per_class_thresholds = _build_per_class_thresholds(per_class_vals)
        _maybe_sort(sort_mode, threshold_val, use_per_class_val, per_class_thresholds)
        zoom = float(zoom_str.replace("x", ""))
        i = int(idx)
        view_mode = str(active_view_mode or "preds_gt")
        if view_mode not in {"preds_gt", "preds", "gt", "image"}:
            view_mode = "preds_gt"
        rendered = viewer.get_images(
            i,
            float(threshold_val),
            zoom,
            show_labels_val,
            view_mode=view_mode,
            use_per_class_thresholds=use_per_class_val,
            per_class_thresholds=per_class_thresholds,
        )
        with tempfile.NamedTemporaryFile(prefix="viewer_", suffix=".png", delete=False) as f:
            out_path = f.name
        rendered.save(out_path, format="PNG")
        return out_path

    def build_download_image_dataset(
        idx: int,
        zoom_str: str,
        sort_mode: str,
        active_view_mode: str,
    ) -> str:
        _maybe_sort(sort_mode, threshold_for_api)
        zoom = float(zoom_str.replace("x", ""))
        i = int(idx)
        view_mode = "gt" if str(active_view_mode) == "gt" else "image"
        rendered = viewer.get_images(i, threshold_for_api, zoom, show_labels=False, view_mode=view_mode)
        with tempfile.NamedTemporaryFile(prefix="viewer_", suffix=".png", delete=False) as f:
            out_path = f.name
        rendered.save(out_path, format="PNG")
        return out_path

    if viewer.has_predictions:
        initial_pg, initial_p, initial_g, initial_i, initial_stats, initial_image_label = refresh_prediction_tabs(
            0, viewer.conf_threshold, "2x", "no order", False, False
        )
    else:
        initial_g, initial_i, initial_image_label = refresh_dataset_tabs(0, "2x", "no order")
        initial_stats = ""

    threshold_for_api = viewer.conf_threshold

    app_title = "Dataset Explorer" if mode == "dataset" else "Object Detection Viewer"
    with gr.Blocks(title=app_title) as app:
        gr.Markdown("# Dataset Explorer" if mode == "dataset" else "# Object Detection Inference Viewer")
        image_label_display = gr.Markdown(value=initial_image_label)

        with gr.Tabs():
            if viewer.has_predictions:
                with gr.Tab(label="Predictions + Ground Truth"):
                    tab_preds_gt = gr.Image(value=initial_pg, label="Predictions (red) + ground truth (green)", height=800)
                with gr.Tab(label="Predictions Only"):
                    tab_preds = gr.Image(value=initial_p, label="Predictions only", height=800)
                with gr.Tab(label="Ground Truth Only"):
                    tab_gt = gr.Image(value=initial_g, label="Ground truth only", height=800)
                with gr.Tab(label="Original Image"):
                    tab_image = gr.Image(value=initial_i, label="Original image", height=800)
            else:
                with gr.Tab(label="Ground Truth Only"):
                    tab_gt = gr.Image(value=initial_g, label="Ground truth only", height=800)
                with gr.Tab(label="Original Image"):
                    tab_image = gr.Image(value=initial_i, label="Original image", height=800)

        active_view_mode = gr.State("preds_gt" if viewer.has_predictions else "gt")

        with gr.Row():
            first_btn = gr.Button("First")
            prev_btn = gr.Button("Previous")
            next_btn = gr.Button("Next")
            last_btn = gr.Button("Last")

        with gr.Row():
            image_idx = gr.Slider(
                minimum=0,
                maximum=max(0, len(viewer.results) - 1),
                value=0,
                step=1,
                label="Image Index",
            )
            image_idx_state = gr.State(0)
            zoom_scale = gr.Radio(
                choices=["1x", "2x", "4x"],
                value="2x",
                label="Zoom Scale",
                interactive=True,
            )
            if viewer.has_predictions:
                conf_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=viewer.conf_threshold,
                    step=0.01,
                    label="Confidence Threshold",
                )
                with gr.Column():
                    show_labels = gr.Checkbox(value=False, label="Show Labels")
                    use_per_class_thresholds = gr.Checkbox(
                        value=False,
                        label="Use per-class thresholds (from metrics/config)",
                    )
        per_class_sliders: List[gr.Slider] = []
        if viewer.has_predictions and viewer.class_names:
            with gr.Accordion("Per-class thresholds", open=False):
                gr.Markdown(
                    "Defaults are loaded from per-class metrics when available "
                    "(otherwise from config, then global threshold)."
                )
                for cname in viewer.class_names:
                    default_thr = _to_float(
                        viewer.per_class_threshold_defaults.get(cname, viewer.conf_threshold),
                        viewer.conf_threshold,
                    )
                    per_class_sliders.append(
                        gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=default_thr,
                            step=0.01,
                            label=f"{cname} threshold",
                        )
                    )
        with gr.Row():
            sort_choices = (
                ["no order", "objs asc", "objs desc"]
                if mode == "dataset"
                else ["no order", "F1 asc", "F1 desc", "dets asc", "dets desc"]
            )
            sort_mode = gr.Radio(
                choices=sort_choices,
                value="no order",
                label="Sort by",
                interactive=True,
            )

        with gr.Row():
            download_btn = gr.Button("Download Current Image")
            download_file = gr.File(label="Download (PNG)", interactive=False)

        if viewer.has_predictions:
            gr.Markdown("### Diagnostics")
            stats_display = gr.Markdown(value=initial_stats)
            tab_outputs_p = [tab_preds_gt, tab_preds, tab_gt, tab_image, stats_display, image_label_display]
            tab_outputs_p_nav = [
                tab_preds_gt, tab_preds, tab_gt, tab_image, image_idx, image_idx_state, stats_display, image_label_display
            ]

            def update_sorting_predictions(threshold_val, zoom_str, sort_mode_val, show_labels_val, use_pc_val, *pc_vals):
                return _tabs_and_slider(0, threshold_val, zoom_str, sort_mode_val, show_labels_val, use_pc_val, 0, *pc_vals)

            def nav_first_p(t, z, s, sl, upc, *pc_vals):
                return _tabs_and_slider(0, t, z, s, sl, upc, 0, *pc_vals)

            def nav_prev_p(cur, t, z, s, sl, upc, *pc_vals):
                new_idx = max(0, _slider_index(cur, 0) - 1)
                return _tabs_and_slider(new_idx, t, z, s, sl, upc, new_idx, *pc_vals)

            def nav_next_p(cur, t, z, s, sl, upc, *pc_vals):
                last = len(viewer.results) - 1
                new_idx = min(last, _slider_index(cur, 0) + 1)
                return _tabs_and_slider(new_idx, t, z, s, sl, upc, new_idx, *pc_vals)

            def nav_last_p(t, z, s, sl, upc, *pc_vals):
                last = len(viewer.results) - 1
                return _tabs_and_slider(last, t, z, s, sl, upc, last, *pc_vals)

            def on_image_idx_change(slider_val, t, z, s, sl, upc, *pc_vals):
                idx = _slider_index(slider_val, 0)
                tabs = refresh_prediction_tabs(idx, t, z, s, sl, upc, *pc_vals)
                return tabs + (idx,)

            prediction_inputs = [
                image_idx_state, conf_threshold, zoom_scale, sort_mode, show_labels, use_per_class_thresholds
            ] + per_class_sliders
            prediction_inputs_no_idx = [
                conf_threshold, zoom_scale, sort_mode, show_labels, use_per_class_thresholds
            ] + per_class_sliders

            first_btn.click(
                fn=nav_first_p,
                inputs=prediction_inputs_no_idx,
                outputs=tab_outputs_p_nav,
            )
            prev_btn.click(
                fn=nav_prev_p,
                inputs=prediction_inputs,
                outputs=tab_outputs_p_nav,
            )
            next_btn.click(
                fn=nav_next_p,
                inputs=prediction_inputs,
                outputs=tab_outputs_p_nav,
            )
            last_btn.click(
                fn=nav_last_p,
                inputs=prediction_inputs_no_idx,
                outputs=tab_outputs_p_nav,
            )
            image_idx.change(
                fn=on_image_idx_change,
                inputs=[image_idx] + prediction_inputs_no_idx,
                outputs=tab_outputs_p + [image_idx_state],
            )
            conf_threshold.change(
                fn=refresh_prediction_tabs,
                inputs=prediction_inputs,
                outputs=tab_outputs_p,
            )
            zoom_scale.change(
                fn=refresh_prediction_tabs,
                inputs=prediction_inputs,
                outputs=tab_outputs_p,
            )
            show_labels.change(
                fn=refresh_prediction_tabs,
                inputs=prediction_inputs,
                outputs=tab_outputs_p,
            )
            use_per_class_thresholds.change(
                fn=refresh_prediction_tabs,
                inputs=prediction_inputs,
                outputs=tab_outputs_p,
            )
            sort_mode.change(
                fn=update_sorting_predictions,
                inputs=prediction_inputs_no_idx,
                outputs=tab_outputs_p_nav,
            )
            for pc_slider in per_class_sliders:
                pc_slider.change(
                    fn=refresh_prediction_tabs,
                    inputs=prediction_inputs,
                    outputs=tab_outputs_p,
                )
            tab_preds_gt.select(fn=lambda: "preds_gt", outputs=active_view_mode)
            tab_preds.select(fn=lambda: "preds", outputs=active_view_mode)
            tab_gt.select(fn=lambda: "gt", outputs=active_view_mode)
            tab_image.select(fn=lambda: "image", outputs=active_view_mode)
            download_btn.click(
                fn=build_download_image_predictions,
                inputs=[image_idx_state, conf_threshold, zoom_scale, sort_mode, show_labels, use_per_class_thresholds, active_view_mode] + per_class_sliders,
                outputs=download_file,
            )
        else:
            tab_outputs_d = [tab_gt, tab_image, image_label_display]
            tab_outputs_d_nav = [tab_gt, tab_image, image_idx, image_idx_state, image_label_display]

            def update_sorting_dataset(zoom_str, sort_mode_val):
                img_g, img_i, image_label = refresh_dataset_tabs(0, zoom_str, sort_mode_val)
                return img_g, img_i, 0, 0, image_label

            def nav_first_d(z, s):
                img_g, img_i, image_label = refresh_dataset_tabs(0, z, s)
                return img_g, img_i, 0, 0, image_label

            def nav_prev_d(cur, z, s):
                new_idx = max(0, _slider_index(cur, 0) - 1)
                img_g, img_i, image_label = refresh_dataset_tabs(new_idx, z, s)
                return img_g, img_i, new_idx, new_idx, image_label

            def nav_next_d(cur, z, s):
                last = len(viewer.results) - 1
                new_idx = min(last, _slider_index(cur, 0) + 1)
                img_g, img_i, image_label = refresh_dataset_tabs(new_idx, z, s)
                return img_g, img_i, new_idx, new_idx, image_label

            def nav_last_d(z, s):
                last = len(viewer.results) - 1
                img_g, img_i, image_label = refresh_dataset_tabs(last, z, s)
                return img_g, img_i, last, last, image_label

            def on_image_idx_change_d(slider_val, z, s):
                idx = _slider_index(slider_val, 0)
                img_g, img_i, image_label = refresh_dataset_tabs(idx, z, s)
                return img_g, img_i, idx, image_label

            first_btn.click(
                fn=nav_first_d,
                inputs=[zoom_scale, sort_mode],
                outputs=tab_outputs_d_nav,
            )
            prev_btn.click(
                fn=nav_prev_d,
                inputs=[image_idx_state, zoom_scale, sort_mode],
                outputs=tab_outputs_d_nav,
            )
            next_btn.click(
                fn=nav_next_d,
                inputs=[image_idx_state, zoom_scale, sort_mode],
                outputs=tab_outputs_d_nav,
            )
            last_btn.click(
                fn=nav_last_d,
                inputs=[zoom_scale, sort_mode],
                outputs=tab_outputs_d_nav,
            )
            image_idx.change(
                fn=on_image_idx_change_d,
                inputs=[image_idx, zoom_scale, sort_mode],
                outputs=tab_outputs_d + [image_idx_state],
            )
            zoom_scale.change(
                fn=refresh_dataset_tabs,
                inputs=[image_idx_state, zoom_scale, sort_mode],
                outputs=tab_outputs_d,
            )
            sort_mode.change(
                fn=update_sorting_dataset,
                inputs=[zoom_scale, sort_mode],
                outputs=tab_outputs_d_nav,
            )
            tab_gt.select(fn=lambda: "gt", outputs=active_view_mode)
            tab_image.select(fn=lambda: "image", outputs=active_view_mode)
            download_btn.click(
                fn=build_download_image_dataset,
                inputs=[image_idx_state, zoom_scale, sort_mode, active_view_mode],
                outputs=download_file,
            )
    return app


def main():
    parser = argparse.ArgumentParser(
        description="Launch Gradio app: predictions viewer or dataset explorer."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["predictions", "dataset"],
        default="predictions",
        help="Mode: 'predictions' (browse inference results) or 'dataset' (explore dataset only)",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=None,
        help="Path to predictions directory (required when --mode predictions)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to data root (required when --mode dataset; optional for predictions)",
    )
    parser.add_argument(
        "--tiles-dir",
        type=str,
        default=None,
        help="Subpath under data-root for dataset mode (e.g. train, train/tiles_1024)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Confidence threshold for predictions mode (default: 0.3)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run on",
    )
    args = parser.parse_args()

    if args.mode == "predictions" and not args.predictions_dir:
        parser.error("--predictions-dir is required when --mode predictions")
    if args.mode == "dataset" and not args.data_root:
        parser.error("--data-root is required when --mode dataset")

    app = create_app(
        mode=args.mode,
        predictions_dir=args.predictions_dir,
        data_root=args.data_root,
        tiles_dir=args.tiles_dir,
        threshold=args.threshold,
    )
    app.launch(server_port=args.port, server_name="0.0.0.0")


if __name__ == '__main__':
    main()

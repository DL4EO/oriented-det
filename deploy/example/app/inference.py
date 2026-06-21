# Copyright 2022 Jeff Faudi DL4EO. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""oriented-det inference for the deploy Sanic service (GeoJSON output)."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import geojson
import numpy as np
import torch
from loguru import logger
from PIL import Image

# Framework repo root: deploy/example/app/inference.py -> parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from oriented_det.train.config import (
    effective_eval_metric_thresholds,
    get_preprocessing_params,
    resolve_inference_score_threshold,
    resolve_inference_sliding_window_overlap_pixels,
)
from oriented_det.runtime.inference import apply_nms_to_detections, get_model_size, run_inference_auto
from oriented_det.runtime.checkpoint import load_model_from_checkpoint


def _class_name(class_names, label: int) -> str:
    """Map model label (1-based foreground) to category string."""
    if class_names and 1 <= int(label) <= len(class_names):
        return class_names[int(label) - 1]
    return f"class_{label}"


def _compute_square_input_size_from_image(
    image_height: int,
    image_width: int,
    align: int,
) -> int:
    """Match deploy/ref.py: square side = ceil(max(h,w) / align) * align (typical align=32)."""
    if align < 1:
        align = 32
    max_dim = max(image_height, image_width)
    return int(np.ceil(max_dim / align) * align)


def _production_bool(cfg, field: str, default: bool) -> bool:
    """Read a boolean from ``production``; use *default* when the section or field is null."""
    inf = getattr(cfg, "production", None)
    if inf is None:
        return default
    v = getattr(inf, field, None)
    if v is None:
        return default
    return bool(v)


def _stick_to_model_input_size(cfg) -> bool:
    """Keep ``preprocessing.target_size``; large images use sliding windows."""
    return _production_bool(cfg, "stick_to_model_canvas", default=True)


def _use_first_image_input_size(cfg) -> bool:
    """First request may expand square canvas from image size (``resize_mode=fixed`` only)."""
    return _production_bool(cfg, "use_first_image_canvas", default=False)


def _adapt_canvas_to_first_image(cfg) -> bool:
    """Whether deploy may expand the inference canvas from the first image (resize_mode=fixed only)."""
    if _stick_to_model_input_size(cfg):
        return False
    return _use_first_image_input_size(cfg)


def _filter_detections_by_image_margin(
    detections: list,
    height: int,
    width: int,
    margin_px: float,
) -> list:
    """Drop detections whose oriented-box centroid is outside the interior [m, W-m] x [m, H-m]."""
    if margin_px <= 0 or not detections:
        return detections
    if 2 * margin_px >= width or 2 * margin_px >= height:
        logger.warning(
            "ignore_margin_pixels (%s px) too large for image size %dx%d; skipping margin filter",
            margin_px,
            width,
            height,
        )
        return detections
    lo_x, hi_x = margin_px, width - margin_px
    lo_y, hi_y = margin_px, height - margin_px
    kept = []
    for d in detections:
        r = d["rbox"]
        cx, cy = float(r.cx), float(r.cy)
        if lo_x <= cx <= hi_x and lo_y <= cy <= hi_y:
            kept.append(d)
    return kept


class InferenceEngine:
    """Load oriented-det checkpoint + config; run inference; return GeoJSON FeatureCollection."""

    def __init__(
        self,
        config_file: str | Path | None = None,
        weights_file: str | Path | None = None,
    ):
        app_dir = Path(__file__).resolve().parent
        self.config_path = Path(config_file) if config_file else app_dir / "config.json"
        self.weights_path = Path(weights_file) if weights_file else app_dir / "weights" / "model.pth"

        self._model = None
        self._config = None
        self._class_names: list | None = None
        self._preprocessing = None
        self._device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._score_threshold = 0.5
        self._per_class_threshold: dict | None = None
        self._nms_threshold = 0.5
        self._overlap_pixels: int = 200
        self._margin_px = 0.0
        self._dataset_overlap_px = 128
        self._load_lock = threading.Lock()
        self._loaded = False
        self._deploy_input_side: int | None = None  # square side used after first-image sizing (log/debug)

        logger.info(f"Repo root (PYTHONPATH): {_REPO_ROOT}")
        logger.info(f"Config path: {self.config_path}")
        logger.info(f"Weights path: {self.weights_path}")
        logger.info(f"Device: {self._device_str}")
        logger.info(
            "Deploy reads config.production.* for inference (score, NMS/decode on the loaded model, "
            "sliding-window overlap, edge margin, canvas flags); null canvas flags use deploy defaults "
            "(stick_to_model_canvas=true, use_first_image_canvas=false)."
        )

    def _build_preprocessing_for_deploy(
        self,
        cfg,
        image_height: int | None,
        image_width: int | None,
    ) -> dict:
        """Start from training config; optionally override fixed resize to match first image (ref.py style)."""
        prep = dict(get_preprocessing_params(cfg))
        mode = prep.get("resize_mode", "fixed")
        old_ts = prep.get("target_size", (1024, 1024))
        if isinstance(old_ts, (list, tuple)) and len(old_ts) >= 2:
            model_h, model_w = int(old_ts[0]), int(old_ts[1])
        else:
            model_h = model_w = int(old_ts) if isinstance(old_ts, (int, float)) else 1024
        model_max = max(model_h, model_w)

        if (
            not _adapt_canvas_to_first_image(cfg)
            or image_height is None
            or image_width is None
            or mode != "fixed"
        ):
            if image_height is not None and image_width is not None and mode != "fixed":
                logger.info(
                    "DEPLOY first-image canvas adapt applies to resize_mode=fixed only; "
                    "using config target_size for resize_mode=%s",
                    mode,
                )
            self._deploy_input_side = None
            if _stick_to_model_input_size(cfg) and mode == "fixed":
                logger.info(
                    "Deploy canvas: using config target_size=%s (model max side=%d); "
                    "images wider/taller than the canvas will use sliding windows.",
                    old_ts,
                    model_max,
                )
            return prep

        align = int(prep.get("pad_size_divisor", 32))
        computed_side = _compute_square_input_size_from_image(image_height, image_width, align)
        # Never shrink below training canvas: small tiles stay at model size (future zoom at native res).
        side = max(computed_side, model_max)
        self._deploy_input_side = side
        prep["target_size"] = (side, side)

        if side > model_max:
            logger.info(
                "Deploy canvas ADAPTED from first image %dx%d → square %dx%d (align=%d; config target_size=%s, "
                "model max side=%d). Single forward at this size — sliding window not used while canvas matches.",
                image_width,
                image_height,
                side,
                side,
                align,
                old_ts,
                model_max,
            )
        else:
            logger.info(
                "Deploy canvas: first image %dx%d fits within or below model canvas — using model square %dx%d "
                "(config target_size=%s; computed=%d; align=%d). Small images are not shrunk below model size.",
                image_width,
                image_height,
                side,
                side,
                old_ts,
                computed_side,
                align,
            )
        return prep

    def _merge_per_class_score_thresholds(self, cfg) -> dict | None:
        """evaluation.per_class_score_threshold → production.per_class_score_threshold (overrides)."""
        _, per_class, _ = effective_eval_metric_thresholds(cfg)
        return per_class

    def _load_model(self, image_height: int | None = None, image_width: int | None = None):
        # Double-checked locking: fast path avoids the lock after the first load.
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            if not self.config_path.is_file():
                raise FileNotFoundError(f"Missing config: {self.config_path}")
            if not self.weights_path.is_file():
                raise FileNotFoundError(f"Missing weights: {self.weights_path}")

            logger.info("Loading oriented-det model from checkpoint…")
            model, cfg, class_names = load_model_from_checkpoint(
                str(self.weights_path),
                str(self.config_path),
                device=self._device_str,
            )
            self._model = model
            self._config = cfg
            self._class_names = list(class_names) if class_names else []
            self._preprocessing = self._build_preprocessing_for_deploy(cfg, image_height, image_width)

            self._score_threshold = resolve_inference_score_threshold(cfg)
            self._per_class_threshold = self._merge_per_class_score_thresholds(cfg)

            if hasattr(self._model, "final_nms_iou_threshold"):
                self._nms_threshold = float(self._model.final_nms_iou_threshold)

            self._overlap_pixels = resolve_inference_sliding_window_overlap_pixels(cfg)

            self._apply_deploy_margin_from_config(cfg)

            self._loaded = True
            logger.info(f"Classes: {self._class_names}")
            if _stick_to_model_input_size(cfg):
                logger.info(
                    "Deploy input canvas: stick to config target_size (production.stick_to_model_canvas). "
                    "Large images use sliding windows."
                )
            elif _use_first_image_input_size(cfg):
                logger.info(
                    "Deploy input canvas: adapt on first request (production.use_first_image_canvas). "
                    "Fixed resize may expand the square canvas for large tiles; smaller tiles stay at "
                    "least the training canvas."
                )
            else:
                logger.info(
                    "Deploy input canvas: fixed from config (production.use_first_image_canvas=false)."
                )
            logger.info(
                f"Inference thresholds: score={self._score_threshold}, "
                f"nms={self._nms_threshold} (from loaded model after production.* decode patch), "
                f"per_class={self._per_class_threshold} "
                "(evaluation → production; class names match case-insensitively)"
            )
            logger.info(
                "Deploy edge margin: dataset.overlap=%s px, ignore_margin_pixels=%s px (centroid filter; "
                "0 disables)",
                self._dataset_overlap_px,
                self._margin_px,
            )

    def _apply_deploy_margin_from_config(self, cfg) -> None:
        """production.ignore_margin_pixels when set, else dataset.overlap/2 (0 = off)."""
        ds = getattr(cfg, "dataset", None)
        overlap = int(getattr(ds, "overlap", 128)) if ds is not None else 128
        if overlap % 2 != 0:
            logger.warning("dataset.overlap should be even; got %s", overlap)
        self._dataset_overlap_px = overlap
        inf = getattr(cfg, "production", None)
        if inf is not None and getattr(inf, "ignore_margin_pixels", None) is not None:
            self._margin_px = float(inf.ignore_margin_pixels)
        else:
            self._margin_px = overlap / 2.0

    def predict(self, images, resolution, version=0):
        if len(images.size()) != 4:
            logger.error(f"Expected NCHW batch; got shape {images.size()}")
            raise ValueError("batch must be 4D")
        out = []
        for i in range(images.shape[0]):
            single = images[i]
            if single.shape[0] == 3:
                # CHW -> HWC
                arr = single.permute(1, 2, 0).cpu().numpy()
            else:
                arr = single.cpu().numpy()
            arr = np.asarray(arr, dtype=np.uint8)
            out.append(self.predict_single(arr, resolution))
        return out

    def predict_single(self, image: np.ndarray, resolution: float):
        """Run detection on one RGB image (H,W,3 uint8); resolution = meters per pixel."""
        logger.info(f"predict_single shape={getattr(image, 'shape', None)} resolution={resolution}")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be HWC RGB")

        h, w = int(image.shape[0]), int(image.shape[1])
        self._load_model(image_height=h, image_width=w)

        pil_image = Image.fromarray(image)

        sh, sw = get_model_size(self._preprocessing)
        if h <= sh and w <= sw:
            logger.info(
                "Inference path: single forward (image %dx%d <= canvas %dx%d); sliding window not used",
                w,
                h,
                sw,
                sh,
            )
        else:
            win_margin = self._overlap_pixels / 2.0
            logger.info(
                "Inference path: sliding-window tiling (image %dx%d > canvas %dx%d, "
                "overlap_pixels=%s, per-window centroid margin=%s px)",
                w,
                h,
                sw,
                sh,
                self._overlap_pixels,
                win_margin,
            )

        detections = run_inference_auto(
            image=pil_image,
            model=self._model,
            device=self._device_str,
            preprocessing=self._preprocessing,
            score_threshold=self._score_threshold,
            nms_threshold=self._nms_threshold,
            overlap_ratio=None,
            overlap_pixels=self._overlap_pixels,
            per_class_score_threshold=self._per_class_threshold,
            class_names=self._class_names,
            window_margin_pixels=self._overlap_pixels / 2.0,
        )
        detections = apply_nms_to_detections(detections, iou_threshold=self._nms_threshold)

        before = len(detections)
        detections = _filter_detections_by_image_margin(
            detections, h, w, self._margin_px
        )
        if before != len(detections):
            logger.info(
                "Margin filter: %d -> %d detections (image %dx%d, ignore_margin_pixels=%s px)",
                before,
                len(detections),
                w,
                h,
                self._margin_px,
            )

        return self._detections_to_geojson(detections, resolution)

    def _detections_to_geojson(self, detections: list, resolution: float) -> dict:
        features = []
        for det in detections:
            rbox = det["rbox"]
            score = float(det["score"])
            label = int(det["label"])
            poly = rbox.to_polygon()
            ring = [[float(p[0]), float(p[1])] for p in poly.points]
            if ring:
                ring.append(ring[0])

            category = _class_name(self._class_names, label)
            props = {
                "category": category,
                "confidence": round(score, 4),
            }
            w_px = float(rbox.width)
            h_px = float(rbox.height)
            long_px = max(w_px, h_px)
            short_px = min(w_px, h_px)
            props["length"] = round(long_px * float(resolution), 2)
            props["width"] = round(short_px * float(resolution), 2)

            features.append(
                geojson.Feature(
                    geometry=geojson.Polygon([ring]),
                    properties=props,
                )
            )

        fc = geojson.FeatureCollection(features)
        return json.loads(geojson.dumps(fc))

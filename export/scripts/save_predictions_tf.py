#!/usr/bin/env python3
"""Run validation inference with the TF/Keras export bundle; write predictions.json for metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from export.ort_runtime import configure_ort_device, get_ort_device
from export.val_dataset import collect_split_images
from oriented_det.data import Detection, GroundTruth
from oriented_det.geometry import RBox, normalize_le90
from oriented_det.train.config import (
    TrainingExperimentConfig,
    effective_eval_metric_thresholds,
    get_preprocessing_params,
    resolve_inference_sliding_window_overlap_pixels,
)
from oriented_det.utils import tqdm_progress_stream
from oriented_det.runtime.inference import _preprocess_image_tensor_training_style, get_model_size
from oriented_det.runtime.checkpoint import load_model_from_checkpoint
from tools.save_predictions import (
    _annotations_to_ground_truths,
    _resolve_metrics_margin_pixels,
    _rbox_centroid_in_tile_interior,
    load_dota_annotations,
    load_gt_as_ground_truths,
    rbox_to_array,
    run_diagnostics_pipeline,
)


def _load_gt_entries(
    img_path: Path,
    label_dir: Optional[Path],
    gt_by_image_path: Optional[Dict[Path, list]],
    class_map: Dict[str, int],
) -> tuple[int, list, list]:
    if gt_by_image_path is not None:
        gt_list = gt_by_image_path.get(img_path, [])
        num_gt = len(gt_list)
        gt_entries = [
            {
                "bbox": rbox_to_array(gt.rbox).tolist(),
                "class_name": gt.class_name,
                "class_id": int(gt.class_id),
                "difficult": int(getattr(gt, "difficult", 0)),
            }
            for gt in gt_list
        ]
        return num_gt, gt_entries, gt_list

    txt_path = (label_dir / f"{img_path.stem}.txt") if label_dir is not None else None
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
        gt_list = load_gt_as_ground_truths(txt_path, class_map) if txt_path and txt_path.exists() else []
    except Exception as exc:
        print(f"Warning: Could not load GT for {img_path.name}: {exc}")
        num_gt = 0
        gt_entries = []
        gt_list = []
    return num_gt, gt_entries, gt_list


def infer_keras_on_image(
    keras_model,
    pil_image: Image.Image,
    preprocessing: dict,
) -> List[Dict[str, Any]]:
    """Run Keras detect bundle on one image; return list of {rbox, score, label}."""
    import tensorflow as tf

    image_width, image_height = pil_image.size
    slice_h, slice_w = get_model_size(preprocessing)
    if image_height > slice_h or image_width > slice_w:
        raise NotImplementedError(
            f"Image {image_width}x{image_height} exceeds model canvas {slice_w}x{slice_h}. "
            "TF export preds does not implement sliding-window tiling; use pre-tiled val images."
        )

    tensor = _preprocess_image_tensor_training_style(pil_image, preprocessing)
    batch = tf.constant(tensor.unsqueeze(0).numpy(), dtype=tf.float32)
    detections_t, num_t = keras_model(batch, training=False)
    n = int(num_t.numpy().reshape(-1)[0])
    if n <= 0:
        return []

    det = detections_t.numpy()[:n]
    from oriented_det.data.preprocessing import build_spatial_meta_from_dims, remap_detections_to_original

    mode = preprocessing.get("resize_mode", "fixed")
    ts = preprocessing.get("target_size", (slice_h, slice_w))
    meta = build_spatial_meta_from_dims(mode, image_width, image_height, ts)

    model_dets: List[Dict[str, Any]] = []
    for row in det:
        cx, cy, w, h, ang, score, label = [float(x) for x in row]
        rbox = normalize_le90(RBox(cx=cx, cy=cy, width=w, height=h, angle=ang))
        model_dets.append({"rbox": rbox, "score": score, "label": int(label)})
    return remap_detections_to_original(model_dets, meta)


def run_tf_inference_and_save(
    *,
    config_path: Path,
    detect_dir: Path,
    output_dir: Optional[Path] = None,
    data_root: Optional[Path] = None,
    data_split: str = "val",
    val_dir: Optional[Path] = None,
    run_diagnostics: bool = True,
    reference_checkpoint: Optional[Path] = None,
    ort_device: Optional[str] = None,
) -> Dict[str, Any]:
    # Configure ORT (and hide TF GPUs) *before* importing tensorflow / keras bundle.
    ort_providers = configure_ort_device(ort_device)
    from export.tf_serving_model import load_keras_detect_model

    config = TrainingExperimentConfig.load(config_path)
    class_names = list(config.class_names or [])
    preprocessing = get_preprocessing_params(config)

    if not data_root:
        if getattr(config, "dataset", None) and getattr(config.dataset, "data_root", None):
            data_root = Path(config.dataset.data_root)
        else:
            raise ValueError("data_root required (CLI or config.dataset.data_root).")
    data_root = Path(data_root)

    keras_path = detect_dir / "keras_model.keras"
    if not keras_path.is_file():
        raise FileNotFoundError(
            f"Missing Keras bundle: {keras_path} "
            "(run: odet export-tf --config ... --checkpoint ... --output-dir ...)."
        )
    keras_model = load_keras_detect_model(keras_path)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("odet_export") / "predictions" / timestamp
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_thr_sc, cfg_thr_pc, cfg_thr_iou = effective_eval_metric_thresholds(config)
    score_threshold = cfg_thr_sc
    per_cls_thr = cfg_thr_pc
    nms_threshold = float(
        getattr(getattr(config, "production", None), "final_nms_iou_threshold", None)
        or getattr(getattr(config, "model", None), "final_nms_iou_threshold", 0.1)
    )
    nms_class_agnostic = bool(
        getattr(getattr(config, "production", None), "nms_class_agnostic", False)
        or getattr(getattr(config, "model", None), "nms_class_agnostic", False)
    )
    iou_threshold = float(cfg_thr_iou)
    overlap_pixels = resolve_inference_sliding_window_overlap_pixels(config)
    resolved_metrics_margin_px = _resolve_metrics_margin_pixels(
        margin_pixels=getattr(getattr(config, "production", None), "ignore_margin_pixels", None),
        overlap_ratio=None,
        overlap_pixels=overlap_pixels,
        preprocessing=preprocessing,
    )

    split_images, label_dir, dataset_format = collect_split_images(
        config, data_root, data_split=data_split, val_dir=val_dir
    )
    print(
        f"TF export inference: {len(split_images)} {data_split} images → {output_dir} "
        f"(ort_device={get_ort_device()}, providers={ort_providers})"
    )

    class_map = {name: i for i, name in enumerate(class_names)} if class_names else {}
    gt_by_image_path = None
    if dataset_format == "airbus_playground":
        from oriented_det.data.airbus_playground import AirbusPlaygroundCSVDataset

        ds_config = config.dataset
        airbus_dataset = AirbusPlaygroundCSVDataset(
            data_root=data_root,
            split=data_split,
            annotations_file=ds_config.annotations_file,
            split_file=ds_config.split_file,
            val_split_id=getattr(ds_config, "val_split_id", 0),
            difficult_strategy=ds_config.difficult_strategy,
            allowed_classes=getattr(ds_config, "allowed_classes", None),
            ignore_labels=getattr(ds_config, "ignore_labels", None) or [],
            map_labels=getattr(ds_config, "map_labels", None) or {},
        )
        gt_by_image_path = {}
        for idx in range(len(airbus_dataset)):
            sample = airbus_dataset[idx]
            gt_by_image_path[Path(sample.image_path)] = _annotations_to_ground_truths(
                list(sample.annotations), class_map
            )

    results: List[Dict[str, Any]] = []
    all_detections: Dict[str, list] = {}
    all_ground_truths: Dict[str, list] = {}
    all_scores: List[float] = []
    image_name_by_id: Dict[str, str] = {}

    t0 = time.perf_counter()
    for img_path in tqdm(
        split_images,
        desc=f"TF export {data_split}",
        file=tqdm_progress_stream(),
    ):
        img_name = img_path.name
        image_id = img_path.stem
        image_name_by_id[image_id] = img_name

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Warning: skip unreadable {img_path}")
            continue
        img_h, img_w = img_bgr.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        num_gt, gt_entries, gt_list_raw = _load_gt_entries(
            img_path, label_dir, gt_by_image_path, class_map
        )

        try:
            det_dicts = infer_keras_on_image(keras_model, pil, preprocessing)
        except Exception as exc:
            print(f"Warning: inference failed for {img_name}: {exc}")
            det_dicts = []

        pred_scores = [d["score"] for d in det_dicts]
        pred_labels = [d["label"] for d in det_dicts]
        rboxes = [d["rbox"] for d in det_dicts]
        all_scores.extend(pred_scores)

        if run_diagnostics:
            dets_m = [
                Detection(
                    rbox=d["rbox"],
                    score=d["score"],
                    class_id=d["label"],
                    class_name=class_names[d["label"] - 1]
                    if class_names and 1 <= d["label"] <= len(class_names)
                    else f"class_{d['label']}",
                    image_id=image_id,
                )
                for d in det_dicts
            ]
            gt_m = [
                GroundTruth(
                    rbox=gt.rbox,
                    class_id=gt.class_id,
                    class_name=gt.class_name,
                    difficult=gt.difficult,
                    image_id=image_id,
                )
                for gt in gt_list_raw
            ]
            if resolved_metrics_margin_px > 0 and img_w > 0 and img_h > 0:
                dets_m = [
                    d
                    for d in dets_m
                    if _rbox_centroid_in_tile_interior(d.rbox, img_w, img_h, resolved_metrics_margin_px)
                ]
                gt_m = [
                    g
                    for g in gt_m
                    if _rbox_centroid_in_tile_interior(g.rbox, img_w, img_h, resolved_metrics_margin_px)
                ]
            all_detections[image_id] = dets_m
            all_ground_truths[image_id] = gt_m

        pred_boxes_array = [rbox_to_array(rb).tolist() for rb in rboxes]
        results.append(
            {
                "image_name": img_name,
                "image_path": os.path.relpath(img_path, data_root),
                "image_width": int(img_w),
                "image_height": int(img_h),
                "resize_mode": preprocessing.get("resize_mode", "fixed"),
                "target_size": preprocessing.get("target_size", [1024, 1024]),
                "num_gt": int(num_gt),
                "num_pred": int(len(rboxes)),
                "predictions": [
                    {
                        "bbox": pred_boxes_array[i],
                        "score": float(pred_scores[i]),
                        "label": int(pred_labels[i]),
                        "class_name": (
                            class_names[pred_labels[i] - 1]
                            if class_names and 1 <= pred_labels[i] <= len(class_names)
                            else f"class_{pred_labels[i]}"
                        ),
                    }
                    for i in range(len(rboxes))
                ],
                "ground_truths": gt_entries,
                "stats": {"inference_backend": "tensorflow_keras_export"},
            }
        )

    t_infer = time.perf_counter() - t0
    experiment_dir = str(config_path.parent)
    checkpoint_ref = str(reference_checkpoint or config_path)

    diagnostics = None
    analysis = None
    if run_diagnostics:
        diagnostics, analysis = run_diagnostics_pipeline(
            experiment_dir=experiment_dir,
            checkpoint_path=checkpoint_ref,
            config_path=str(config_path),
            data_root=data_root,
            data_split=data_split,
            class_names=class_names,
            score_threshold=float(score_threshold),
            per_cls_thr=per_cls_thr,
            nms_class_agnostic=nms_class_agnostic,
            iou_threshold=iou_threshold,
            pr_iou_threshold=None,
            pr_threshold_min=0.0,
            pr_threshold_max=1.0,
            pr_threshold_step=0.1,
            per_class_threshold_analysis=False,
            resolved_metrics_margin_px=int(resolved_metrics_margin_px),
            all_detections=all_detections,
            all_ground_truths=all_ground_truths,
            results=results,
            all_scores=all_scores,
            image_name_by_id=image_name_by_id,
            sliding_window_positions_total=None,
            window_batch_effective=None,
            t_infer_sec=float(t_infer),
            device="tensorflow",
            output_dir=str(output_dir),
            tile_metrics_csv=None,
        )

    meta_path = detect_dir / "export_meta.json"
    export_meta = {}
    if meta_path.is_file():
        export_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    metadata: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "inference_backend": "tensorflow_keras_export",
        "detect_bundle": str(detect_dir.resolve()),
        "keras_model": str(keras_path.resolve()),
        "export_meta_core_backend": export_meta.get("core_backend"),
        "experiment_dir": experiment_dir,
        "checkpoint": checkpoint_ref,
        "config_file": str(config_path),
        "pytorch_reference_checkpoint": str(reference_checkpoint) if reference_checkpoint else None,
        "data_root": str(data_root),
        "data_split": data_split,
        "device": "tensorflow",
        "ort_device": get_ort_device(),
        "ort_providers": ort_providers,
        "class_names": class_names,
        "score_threshold": score_threshold,
        "per_class_score_threshold": per_cls_thr,
        "nms_class_agnostic": nms_class_agnostic,
        "total_images": len(results),
        "total_predictions": sum(r["num_pred"] for r in results),
        "total_ground_truth": sum(r["num_gt"] for r in results),
        "inference_loop_seconds": float(t_infer),
        "bbox_coordinate_space": "image_pixels",
        "metrics_margin_pixels": int(resolved_metrics_margin_px),
        "preprocess_note": "Resize+ToTensor+Normalize (same as PyTorch make preds)",
    }
    if diagnostics is not None:
        metadata["diagnostics"] = diagnostics
    if analysis is not None:
        metadata["best_threshold_f2"] = analysis.get("best_threshold", {})
        metadata["analysis_file"] = analysis.get("artifacts", {}).get(
            "analysis_json", f"analysis_iou{iou_threshold:.2f}.json"
        )
        metadata["pr_iou_threshold"] = analysis.get("iou_threshold", iou_threshold)

    json_path = output_dir / "predictions.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "results": results}, f, indent=2)

    print(f"Wrote {json_path}")
    return {"output_dir": str(output_dir), "metadata": metadata}


def main() -> None:
    p = argparse.ArgumentParser(description="Val inference via TF/Keras export bundle.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--detect-dir", type=Path, required=True, help="Directory with keras_model.keras")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: ./odet_export/predictions/<timestamp>/",
    )
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--data-split", default="val", choices=("train", "val", "test"))
    p.add_argument("--val-dir", type=Path, default=None)
    p.add_argument(
        "--reference-checkpoint",
        type=Path,
        default=None,
        help="PyTorch .pth path recorded in metadata for comparison (default: deploy weights path only in meta).",
    )
    p.add_argument("--no-diagnostics", action="store_true", help="Skip mAP/PR (inference-only JSON).")
    p.add_argument(
        "--ort-device",
        default=None,
        choices=("cpu", "cuda", "auto"),
        help="ONNX Runtime EP for the exported ONNX core (default: cpu or ORIENTED_DET_ORT_DEVICE).",
    )
    args = p.parse_args()

    run_tf_inference_and_save(
        config_path=args.config,
        detect_dir=args.detect_dir,
        output_dir=args.output_dir,
        data_root=args.data_root,
        data_split=args.data_split,
        val_dir=args.val_dir,
        run_diagnostics=not args.no_diagnostics,
        reference_checkpoint=args.reference_checkpoint,
        ort_device=args.ort_device,
    )


if __name__ == "__main__":
    main()

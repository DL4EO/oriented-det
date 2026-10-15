#!/usr/bin/env python3
"""Val inference with the exported ONNX + Python NMS; write predictions.json for eval-val metrics."""

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

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from export.ort_runtime import configure_ort_device, get_ort_device  # noqa: E402
from export.runtime import detect_image  # noqa: E402
from export.val_dataset import collect_split_images  # noqa: E402
from oriented_det.geometry import RBox, normalize_le90  # noqa: E402
from oriented_det.train.config import (  # noqa: E402
    TrainingExperimentConfig,
    resolve_inference_sliding_window_overlap_pixels,
    resolve_preds_score_threshold,
)
from oriented_det.utils import tqdm_progress_stream  # noqa: E402
from tools.save_predictions import (  # noqa: E402
    _annotations_to_ground_truths,
    _resolve_metrics_margin_pixels,
    load_dota_annotations,
    load_gt_as_ground_truths,
    rbox_to_array,
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


def _det_to_rbox(det: Dict[str, Any]) -> RBox:
    return normalize_le90(
        RBox(
            cx=float(det["cx"]),
            cy=float(det["cy"]),
            width=float(det["w"]),
            height=float(det["h"]),
            angle=float(det["angle_rad"]),
        )
    )


def run_onnx_inference_and_save(
    *,
    config_path: Path,
    onnx_path: Path,
    output_dir: Optional[Path] = None,
    data_root: Optional[Path] = None,
    data_split: str = "val",
    val_dir: Optional[Path] = None,
    reference_checkpoint: Optional[Path] = None,
    ort_device: Optional[str] = None,
    score_threshold: Optional[float] = None,
    nms_backend: Optional[str] = None,
) -> Dict[str, Any]:
    configure_ort_device(ort_device)
    config = TrainingExperimentConfig.load(config_path)
    class_names = list(config.class_names or [])

    if not data_root:
        if getattr(config, "dataset", None) and getattr(config.dataset, "data_root", None):
            data_root = Path(config.dataset.data_root)
        else:
            raise ValueError("data_root required (CLI or config.dataset.data_root).")
    data_root = Path(data_root)

    onnx_file = Path(onnx_path)
    if not onnx_file.is_file():
        raise FileNotFoundError(f"Missing ONNX: {onnx_file}")

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("onnx_export") / "predictions" / timestamp
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    score_thr, score_src = resolve_preds_score_threshold(
        config, data_split=data_split, cli_score_threshold=score_threshold
    )
    nms_backend = nms_backend or "shapely"
    nms_class_agnostic = bool(
        getattr(getattr(config, "production", None), "nms_class_agnostic", False)
        or getattr(getattr(config, "model", None), "nms_class_agnostic", False)
    )
    overlap_pixels = resolve_inference_sliding_window_overlap_pixels(config)
    if overlap_pixels is None:
        overlap_pixels = 256
    preprocessing = {
        "resize_mode": getattr(config.preprocessing, "resize_mode", "fixed"),
        "target_size": list(getattr(config.preprocessing, "target_size", [1024, 1024])),
    }
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
        f"ONNX export inference: {len(split_images)} {data_split} images → {output_dir} "
        f"(score={score_thr:g} from {score_src}, nms_backend={nms_backend}, "
        f"ort_device={get_ort_device()})"
    )

    class_map = {name: i for i, name in enumerate(class_names)} if class_names else {}
    gt_by_image_path = None
    if dataset_format in ("airbus_playground", "hrsc2016"):
        from dataclasses import replace

        from oriented_det.data.build import build_split_dataset

        ds_cfg = replace(config.dataset, data_root=data_root)
        native_dataset = build_split_dataset(ds_cfg, data_split, filter_empty_gt=False)
        gt_by_image_path = {}
        for idx in tqdm(
            range(len(native_dataset)),
            desc="Loading annotations",
            file=tqdm_progress_stream(),
        ):
            sample = native_dataset[idx]
            gt_by_image_path[Path(sample.image_path)] = _annotations_to_ground_truths(
                list(sample.annotations), class_map
            )

    results: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for img_path in tqdm(
        split_images,
        desc=f"ONNX export {data_split}",
        file=tqdm_progress_stream(),
    ):
        img_name = img_path.name
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as exc:
            print(f"Warning: skip unreadable {img_path}: {exc}")
            continue
        img_w, img_h = pil.size
        num_gt, gt_entries, _gt_list = _load_gt_entries(
            img_path, label_dir, gt_by_image_path, class_map
        )
        try:
            dets = detect_image(
                pil,
                onnx_file,
                score_threshold=float(score_thr),
                nms_backend=nms_backend,
            )
        except Exception as exc:
            print(f"Warning: inference failed for {img_name}: {exc}")
            dets = []

        pred_rows = []
        for d in dets:
            rbox = _det_to_rbox(d)
            pred_rows.append(
                {
                    "bbox": rbox_to_array(rbox).tolist(),
                    "score": float(d["score"]),
                    "label": int(d["label"]),
                    "class_name": str(d.get("class_name") or f"class_{int(d['label'])}"),
                }
            )
        results.append(
            {
                "image_name": img_name,
                "image_path": os.path.relpath(img_path, data_root),
                "image_width": int(img_w),
                "image_height": int(img_h),
                "resize_mode": preprocessing.get("resize_mode", "fixed"),
                "target_size": preprocessing.get("target_size", [1024, 1024]),
                "num_gt": int(num_gt),
                "num_pred": int(len(pred_rows)),
                "predictions": pred_rows,
                "ground_truths": gt_entries,
                "stats": {"inference_backend": "onnx_export"},
            }
        )

    t_infer = time.perf_counter() - t0
    experiment_dir = str(config_path.parent)
    checkpoint_ref = str(reference_checkpoint or config_path)
    metadata: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "inference_backend": "onnx_export",
        "onnx_path": str(onnx_file.resolve()),
        "experiment_dir": experiment_dir,
        "checkpoint": checkpoint_ref,
        "config_file": str(config_path),
        "pytorch_reference_checkpoint": str(reference_checkpoint) if reference_checkpoint else None,
        "data_root": str(data_root),
        "data_split": data_split,
        "device": get_ort_device(),
        "ort_device": get_ort_device(),
        "class_names": class_names,
        "score_threshold": float(score_thr),
        "score_threshold_source": score_src,
        "per_class_score_threshold": None,
        "nms_class_agnostic": nms_class_agnostic,
        "nms_backend": nms_backend,
        "total_images": len(results),
        "total_predictions": sum(r["num_pred"] for r in results),
        "total_ground_truth": sum(r["num_gt"] for r in results),
        "inference_loop_seconds": float(t_infer),
        "bbox_coordinate_space": "image_pixels",
        "metrics_margin_pixels": int(resolved_metrics_margin_px),
        "sliding_window_overlap_pixels": int(overlap_pixels) if overlap_pixels is not None else None,
        "preprocess_note": "export.preprocess + onnx_export NMS (eval-val score floor)",
    }
    json_path = output_dir / "predictions.json"
    json_path.write_text(
        json.dumps({"metadata": metadata, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {json_path}")
    print(
        f"images={metadata['total_images']}  preds={metadata['total_predictions']}  "
        f"gt={metadata['total_ground_truth']}  {t_infer:.1f}s"
    )
    return {"output_dir": str(output_dir), "metadata": metadata}


def main() -> None:
    p = argparse.ArgumentParser(description="Val inference via exported ONNX + Python NMS.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--onnx", type=Path, required=True, help="Path to model.onnx.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: ./onnx_export/predictions/<timestamp>/",
    )
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--data-split", default="val", choices=("train", "val", "test"))
    p.add_argument("--val-dir", type=Path, default=None)
    p.add_argument("--reference-checkpoint", type=Path, default=None)
    p.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Override eval-val score floor (default: 0.05 preds protocol).",
    )
    p.add_argument(
        "--nms-backend",
        choices=("python", "shapely", "auto"),
        default="shapely",
        help="CPU NMS IoU backend (default shapely to match eval-val CPU NMS).",
    )
    p.add_argument(
        "--ort-device",
        default=None,
        choices=("cpu", "cuda", "auto"),
        help="ONNX Runtime EP (default: cpu or ORIENTED_DET_ORT_DEVICE).",
    )
    args = p.parse_args()
    run_onnx_inference_and_save(
        config_path=args.config,
        onnx_path=args.onnx,
        output_dir=args.output_dir,
        data_root=args.data_root,
        data_split=args.data_split,
        val_dir=args.val_dir,
        reference_checkpoint=args.reference_checkpoint,
        ort_device=args.ort_device,
        score_threshold=args.score_threshold,
        nms_backend=args.nms_backend,
    )


if __name__ == "__main__":
    main()

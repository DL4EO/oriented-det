#!/usr/bin/env python3
"""Build a DOTA v1.0 Task 1 zip from ``predictions.json``, a training run, or Hub zoo weights.

Usage:
    odet dota-submit --from-json predictions/<ts>/predictions.json --output-dir work_dirs/Task1
    odet dota-submit --checkpoint hf://oriented_rcnn_dota_le90_3x \\
        --test-dir /path/to/DOTA/test --output-dir work_dirs/Task1
    odet dota-submit --experiment-dir runs/oriented_rcnn/<id> --test-dir /path/to/DOTA/test \\
        --output-dir work_dirs/Task1

Official test labels are not public. Upload the zip to the DOTA v1.0 Task 1 server.
This path uses sliding-window inference on full images: last tiles flush to the
image edge (same as ``tile_dota.py``) and overlap copies are kept, then NMS
(ResultMerge-style). It is not MMRotate's on-disk pre-tile merge.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from oriented_det.data.dota_classes import DOTA_V1_CLASSES
from oriented_det.data.dota_task1 import write_task1_submission


def _resolve_predictions_json(from_json: Path) -> Path:
    if from_json.is_dir():
        candidate = from_json / "predictions.json"
        if not candidate.is_file():
            raise FileNotFoundError(f"No predictions.json in {from_json}")
        return candidate
    if not from_json.is_file():
        raise FileNotFoundError(f"predictions JSON not found: {from_json}")
    return from_json


def _predictions_json_from_run(meta: dict, output_dir: str | None = None) -> Path:
    """Locate ``predictions.json`` after ``run_inference_and_save``.

    Older metadata dicts omitted ``output_dir``; prefer ``predictions_json`` then
    ``output_dir`` then the directory we passed in.
    """
    raw = meta.get("predictions_json")
    if raw:
        path = Path(raw)
        if path.is_file():
            return path
    for key in ("output_dir", "output_directory"):
        raw = meta.get(key)
        if raw:
            path = Path(raw) / "predictions.json"
            if path.is_file():
                return path
    if output_dir:
        path = Path(output_dir) / "predictions.json"
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Inference did not write predictions.json "
        f"(meta keys={sorted(meta)}, output_dir={output_dir!r})"
    )


def _run_test_preds(
    *,
    experiment_dir: str | None,
    checkpoint: str | None,
    config_path: str | None,
    data_root: str | None,
    test_dir: str | None,
    output_dir: str | None,
    window_margin_pixels: float = 0.0,
) -> Path:
    from tools.save_predictions import resolve_preds_model_paths, run_inference_and_save

    exp_dir, ckpt, cfg = resolve_preds_model_paths(
        experiment_dir=experiment_dir,
        checkpoint=checkpoint,
        config_path=config_path,
        auto_detect=False,
    )
    # Hub sidecars often still have production.final_nms_iou_threshold 0.5; Task 1 / eval-val is 0.1.
    # window_margin_pixels=0 keeps overlap copies, then NMS (ResultMerge-style).
    meta = run_inference_and_save(
        experiment_dir=exp_dir,
        checkpoint_path=ckpt,
        config_path=cfg,
        data_root=data_root,
        output_dir=output_dir,
        data_split="test",
        test_dir=test_dir,
        nms_threshold=0.1,
        window_margin_pixels=window_margin_pixels,
        run_diagnostics=False,
    )
    return _predictions_json_from_run(meta, output_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Write DOTA v1.0 Task 1 txt files + zip for the official evaluation server."
    )
    parser.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="predictions.json or a directory that contains it.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=None,
        help="If --from-json is omitted, run odet-preds on unlabeled test images from this run.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Hub slug (hf://oriented_rcnn_dota_le90_3x) or .pth. Sidecar JSON is used if --config is omitted.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config JSON (optional with Hub slugs; pretrained sidecar is used).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="DOTA root (used when config has no dataset.data_root, or to override a Hub sidecar placeholder).",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="Unlabeled test images (test/ or test/images/). Default: data_root/test.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for Task1_*.txt (created). Zip is written beside it unless --zip is set.",
    )
    parser.add_argument(
        "--zip",
        dest="zip_path",
        type=Path,
        default=None,
        help="Zip path (default: <output-dir>.zip next to the txt directory).",
    )
    parser.add_argument(
        "--no-dummy",
        action="store_true",
        help="Do not write a dummy line for classes with zero detections (server may reject empty files).",
    )
    parser.add_argument(
        "--preds-output-dir",
        type=str,
        default=None,
        help="When running inference, write predictions.json here (default: predictions/<timestamp>/).",
    )
    parser.add_argument(
        "--window-margin-pixels",
        type=float,
        default=0.0,
        help="Per-window centroid margin before merge (default 0: keep overlap copies, then NMS). "
             "Pass a positive value (e.g. overlap/2) to drop the interior overlap band.",
    )
    args = parser.parse_args(argv)

    if args.from_json is None and not args.experiment_dir and not args.checkpoint:
        parser.error("Provide --from-json, --checkpoint (hf://<slug>), or --experiment-dir")

    if args.from_json is None and args.checkpoint and not args.experiment_dir:
        if not args.test_dir and not args.data_root:
            parser.error(
                "When using --checkpoint, pass --test-dir (e.g. /data/DOTA-v1.0/test) "
                "or --data-root (official test images under <root>/test)."
            )

    if args.from_json is not None:
        pred_json = _resolve_predictions_json(args.from_json)
    else:
        pred_json = _run_test_preds(
            experiment_dir=args.experiment_dir,
            checkpoint=args.checkpoint,
            config_path=args.config,
            data_root=args.data_root,
            test_dir=args.test_dir,
            output_dir=args.preds_output_dir,
            window_margin_pixels=args.window_margin_pixels,
        )
        print(f"Wrote predictions: {pred_json}")

    zip_path = write_task1_submission(
        pred_json,
        args.output_dir,
        class_names=DOTA_V1_CLASSES,
        dummy_if_empty=not args.no_dummy,
        zip_path=args.zip_path,
    )
    n_files = len(list(args.output_dir.glob("Task1_*.txt")))
    print(f"Wrote {n_files} Task1_*.txt under {args.output_dir}")
    print(f"Zip: {zip_path}")
    print("Upload to https://captain-whu.github.io/DOTA/evaluation.html (DOTA-v1.0 Task 1).")
    print(
        "Note: last tiles flush to the image edge; overlap copies kept, then NMS 0.1 "
        "(live sliding-window ResultMerge-style, not MMRotate on-disk pre-tile merge)."
    )


if __name__ == "__main__":
    main()

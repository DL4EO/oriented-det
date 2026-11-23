# Tests

Run the full test suite from the repository root:

```bash
python -m pytest tests/ -q
```

Or with the Makefile (`PYTHONPATH` set to the repo root):

```bash
make test
```

Run specific test files or directories:

```bash
pytest tests/test_geometry.py
pytest tests/test_iou.py tests/test_nms.py
make test TESTS=tests/test_geometry.py
```

With coverage:

```bash
pytest --cov=oriented_det tests/
```

CI runs `pytest tests/ export/tests/` on push/PR (see `.github/workflows/test.yml`). Export tests that need `onnx` / `onnxruntime` skip unless `oriented-det[export]` is installed (CI installs it).

## Test modules

- **test_geometry.py**, **test_geometry_transforms.py** — Geometry (poly, rbox, qbox, transforms)
- **test_iou.py**, **test_nms.py**, **test_ops_utils.py**, **test_gpu_ops.py**, **test_kfiou.py**, **test_probiou.py**, **test_ciou.py**, **test_exact_rotated_iou.py** — IoU/NMS and ops
- **test_dota.py**, **test_dota_tile_roots.py**, **test_dota_task1.py**, **test_tiling.py**, **test_tile_dota.py**, **test_train_flips.py**, **test_train_rotates.py**, **test_mosaic.py** — Data loading, unlabeled test discovery, Task 1 export, `odet tile-dota` JPEG/PNG, flips, rotate, and mosaic (cached tiles keep flipped boxes; a box is dropped when less than 20% of its area remains (more than 80% outside))
- **test_models.py**, **test_rpn.py**, **test_roi.py**, **test_bbox_coder.py**, **test_sigmoid_focal_class_weights.py**, **test_rotated_rtmdet.py** — Models and heads; FCOS/RetinaNet class-weighted sigmoid focal; RTMDet ignore and lookalike point labels
- **test_train.py**, **test_wizard.py**, **test_hard_tile_oversampling.py**, **test_class_tile_oversampling.py**, **test_grouped_ce.py**, **test_cosine_tail_scheduler.py**, **test_linear_lr_scheduler.py**, **test_optimizer_param_groups.py** — Training engine, wizard diagnostics, hard-tile / class-tile oversampling, and schedulers
- **test_score_thresholds.py**, **test_output_classes.py**, **test_evaluation.py** — Metrics, mAP, and `production.output_classes` allowlist
- **test_utils_config.py**, **test_utils_viz.py** — Config and visualization
- **test_config_behavior.py**, **test_config_model_wiring.py**, **test_training_config_strict.py** — JSON config strictness and wiring; DOTA 3× is 1× + 36 epochs
- **test_airbus_playground.py** — Airbus Playground CSV dataset
- **test_hrsc2016.py** — HRSC2016 XML loader, ImageSets splits, DOTA export
- **test_fair1m.py** — FAIR1M XML loader, holdout split, DOTA export, 1× recipes
- **test_coco_obb.py** — COCO polygon → le90 rbox (quads and n-gons), DOTA export
- **test_ssdd.py** — SSDD official 1/9 split, XML/COCO layouts, DOTA export, 1× recipe; skips HRSID_JPG; `python -m oriented_det.cli`
- **test_hrsid.py** — HRSID COCO loader, train/test, DOTA export, 1× recipes
- **test_pretrained_hub.py** — Hugging Face Hub manifest and download helpers
- **test_sliding_window_margin.py**, **test_metrics_margin_filter.py** — Inference margin helpers; last-tile flush to image edge; pad vs DOTA native sliding-window routing; window micro-batch default (8 GPU / no auto-probe)
- **test_deploy_generate_description.py** — Deploy script smoke
- **test_viewer_gradio_compat.py** — Gradio slider patch and viewer outline width (`oriented-det[viewer]`, including OpenCV; skipped if Gradio is missing)

Export tests live under [`export/tests/`](../export/tests/README.md) (`make export-test`).

See the [main README](../README.md) for installation and [docs/contributing.md](../docs/contributing.md) for contribution guidelines.

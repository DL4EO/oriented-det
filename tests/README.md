# Tests

Run the full test suite from the repository root:

```bash
pytest
```

Or with the Makefile:

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

Export tests (separate):

```bash
cd export && make test
```

CI runs `pytest tests/` on push/PR (see `.github/workflows/test.yml`).

## Test modules

- **test_geometry.py**, **test_geometry_transforms.py** — Geometry (poly, rbox, qbox, transforms)
- **test_iou.py**, **test_nms.py**, **test_ops_utils.py**, **test_gpu_ops.py**, **test_kfiou.py**, **test_probiou.py**, **test_exact_rotated_iou.py** — IoU/NMS and ops
- **test_dota.py**, **test_dota_tile_roots.py**, **test_tiling.py**, **test_transforms.py**, **test_train_flips.py** — Data loading and augmentation
- **test_models.py**, **test_rpn.py**, **test_roi.py**, **test_bbox_coder.py** — Models and heads
- **test_train.py**, **test_grouped_ce.py**, **test_cosine_tail_scheduler.py**, **test_optimizer_param_groups.py** — Training engine and schedulers
- **test_score_thresholds.py**, **test_evaluation.py** — Metrics and mAP
- **test_utils_config.py**, **test_utils_viz.py** — Config and visualization
- **test_config_behavior.py**, **test_config_model_wiring.py**, **test_training_config_strict.py** — JSON config strictness and wiring
- **test_airbus_playground.py** — Airbus Playground CSV dataset
- **test_pretrained_hub.py** — Hugging Face Hub manifest and download helpers
- **test_sliding_window_margin.py**, **test_metrics_margin_filter.py** — Inference margin helpers
- **test_deploy_generate_description.py** — Deploy script smoke

See the [main README](../README.md) for installation and [docs/contributing.md](../docs/contributing.md) for contribution guidelines.

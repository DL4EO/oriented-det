# Tutorial

An 11-lesson OrientedDet course, from a first detection to a deployable ONNX model. Lessons 01–04 are about boxes and detectors, 05–07 about data, configs, and evaluation, 08–10 fine-tune on new datasets and sensors, and 11 exports the result. Each lesson mixes short theory sections with runnable cells and ends with "what to try next". Every notebook runs from a **local checkout** or on **Kaggle** (Internet ON for pip and the Hugging Face Hub).

## Curriculum

| # | Notebook | You learn | Posts | Kaggle inputs | Trains |
|---|----------|-----------|-------|---------------|--------|
| 01 | [`01_first_inference.ipynb`](01_first_inference.ipynb) | Hub checkpoints and sidecars, `run_inference_auto`, reading a detection, oriented vs axis-aligned boxes, the `image-demo` CLI | [macOS pure Python](https://deeplearning.earth/posts/2026-06-25_oriented_object_detection_on_macos_in_pure_python/) · [v0.1.0](https://deeplearning.earth/posts/2026-06-22_oriented-det_v0_1_0_sovereign_oriented_object_detection_for_eo/) | none | no |
| 02 | [`02_boxes_angles_nms.ipynb`](02_boxes_angles_nms.ipynb) | `RBox`, the le90 convention, DOTA quads, flips and rotations, rotated IoU, rotated vs axis-aligned NMS | [v0.1.0](https://deeplearning.earth/posts/2026-06-22_oriented-det_v0_1_0_sovereign_oriented_object_detection_for_eo/) | none | no |
| 03 | [`03_probiou_box_losses.ipynb`](03_probiou_box_losses.ipynb) | Box regression losses: L1, decoded rIoU, ProbIoU, KFIoU; angle periodicity and square blindness; which recipe uses which loss | [ProbIoU](https://deeplearning.earth/posts/2026-07-10_rotated_faster_rcnn_probiou_dota/) · [v0.2.0](https://deeplearning.earth/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/) | none | no |
| 04 | [`04_detector_zoo_thresholds.ipynb`](04_detector_zoo_thresholds.ipynb) | Oriented R-CNN, Faster R-CNN, FCOS, RetinaNet on one image; score distributions and per-model deploy floors; speed and size; RTMDet Hub model | [Optical demo](https://deeplearning.earth/posts/2026-09-06_oriented_det_optical_satellite_demo/) · [FCOS vs ORCNN](https://deeplearning.earth/posts/2026-09-02_rotated_fcos_vs_oriented_rcnn_on_macos/) | none | no |
| 05 | [`05_dota_fifteen_classes.ipynb`](05_dota_fifteen_classes.ipynb) | The DOTA benchmark, class order, per-class Task 1 results, ground truth vs predictions, recall and confusions | [15 DOTA classes](https://deeplearning.earth/posts/2026-06-23_oriented_rcnn_detections_for_the_15_dota_classes/) · [Apache vs DOTA](https://deeplearning.earth/posts/2026-09-10_oriented_det_apache_license_versus_dota/) | none (sections 3–5 local only) | no |
| 06 | [`06_large_images_sentinel2.ipynb`](06_large_images_sentinel2.ipynb) | Sliding-window inference, window overlap, a zero-shot Sentinel-2 ship count with physical lengths | [Final pretrained model](https://deeplearning.earth/posts/2026-06-29_announcing_the_final_oriented_det_pretrained_model/) · [Sentinel-2 ships](https://deeplearning.earth/posts/2026-06-25_zero-shot_ship_detection_on_a_copernicus_sentinel-2_tile_with_oriented_rcnn/) | none | no |
| 07 | [`07_config_train_eval.ipynb`](07_config_train_eval.ipynb) | JSON recipes and `_base_` inheritance, patching and saving a recipe, `odet train` outputs, how oriented mAP is computed, `odet preds`, Task 1 vs local val | [DOTA lessons](https://deeplearning.earth/posts/2026-09-17_lessons_learned_on_dota_oriented_det_and_mmrotate_parity/) · [v0.3.0](https://deeplearning.earth/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/) | none (sections 5–6 local only) | no |
| 08 | [`08_hrsc2016_finetune.ipynb`](08_hrsc2016_finetune.ipynb) | A new dataset end to end: HRSC2016 statistics, `keep_ratio` resizing, evaluating the Hub 3× model, a training smoke from ImageNet weights | [HRSC2016](https://deeplearning.earth/posts/2026-09-13_hrsc2016_recipes_trains_and_results/) | [`guofeng/hrsc2016`](https://www.kaggle.com/datasets/guofeng/hrsc2016) | optional smoke |
| 09 | [`09_fair1m_finetune.ipynb`](09_fair1m_finetune.ipynb) | Data preparation: FAIR1M's 37 fine-grained classes, convert to DOTA, image-level holdout, `odet tile-dota`, fine-tune from a DOTA Hub checkpoint | [FAIR1M](https://deeplearning.earth/posts/2026-09-24_fair1m_fine_grained_oriented_detection/) | [`ollypowell/fair1m-…`](https://www.kaggle.com/datasets/ollypowell/fair1m-satellite-imagery-for-object-detection) | optional smoke |
| 10 | [`10_sar_ssdd_hrsid.ipynb`](10_sar_ssdd_hrsid.ipynb) | Optical to radar: SSDD and HRSID, the zero-shot DOTA baseline on SAR, fine-tuning with the published recipes | [v0.3.0](https://deeplearning.earth/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/) | an SSDD or HRSID dump (any dataset whose name contains `ssdd` or `hrsid`) | optional smoke |
| 11 | [`11_onnx_export.ipynb`](11_onnx_export.ipynb) | `python -m export onnx`, the exported runtime files, ONNX Runtime inference, parity with PyTorch | [v0.3.0](https://deeplearning.earth/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/) · [export docs](../docs/examples/export.md) | none | no |

Lessons 01–04 and 06 need nothing but Internet. Lessons 05 and 07 use the 36 DOTA val tiles in [`demo/dota_samples/`](../demo/dota_samples/) for their ground-truth sections; DOTA is academic-use only, so those sections run from a local checkout and skip on Kaggle.

### Next (not in this pass)

- **Airbus Playground** and **your own data**: the loaders and recipes exist (see [data.md](../docs/user-guide/data.md)), but the data is proprietary or user-supplied, so there is no notebook yet. Lesson 09 is the closest template: convert to DOTA, tile, fine-tune from a Hub checkpoint.
- **RTMDet hands-on lesson**: lesson 04 runs the Hub `rotated_rtmdet_dota_le90_3x` checkpoint next to the ResNet zoo; a dedicated RTMDet lesson (training and fine-tuning) is not written yet.

## Local vs Kaggle

Every notebook opens with the same two setup cells:

1. **Install**: an editable `pip install -e <repo>` when the notebook sits inside a checkout (`pyproject.toml` + `oriented_det/` found above it), otherwise `pip install "oriented-det>=0.3.1"`. Lesson 11 adds the `[export]` extras (`onnx`, `onnxruntime`).
2. **Environment and helpers**:
   - **Work dir**: `/kaggle/working` on Kaggle, else `$ORIENTED_DET_TUTORIAL_WORK`, else `work_dirs/` in the checkout (gitignored). `ORIENTED_DET_PROJECT_ROOT` points there, so `odet train` runs land in `work_dirs/runs/`.
   - **Hub weights**: `hf://…` slugs cache into `pretrained/` in a checkout; on a wheel install `ORIENTED_DET_PRETRAINED_DIR` defaults to the work dir. `hub_files(slug)` also fetches the checkpoint's sidecar `.json` (needed by the CLI and `load_model_from_checkpoint`).
   - **Demo files**: from the checkout `demo/`, else downloaded from GitHub raw into the work dir.
   - **Datasets**: scan `/kaggle/input` for a known name, else `HRSC_DATA_ROOT` / `FAIR1M_ROOT` / `SSDD_DATA_ROOT` / `HRSID_DATA_ROOT` / `DOTA_TILED_ROOT`.
   - **Device**: CUDA → MPS → CPU.
   - **Helpers used across lessons**: `hub_files`, `hub_model`, `detect`, `show`, `run` (echoes and runs an `odet` command).

Training cells default to **off** (`RUN_SMOKE = False` in 08 and 10, `RUN_TRAIN = False` in 09). The smokes are one epoch on a few dozen images, enough to exercise loading, augmentation, loss, and checkpointing, and they need a GPU. Full 1× / 3× recipes run from the CLI: `odet train --config configs/…`.

### Local

```bash
cd /path/to/oriented-det
jupyter lab tutorial/
```

### Kaggle

1. New notebook, **Internet ON**. A GPU is optional: recommended for 04, 08, and 10, required only for the training smokes.
2. Upload the `.ipynb`.
3. For lessons 08–10, **Add Input** with the dataset from the table above.
4. Run top to bottom.

## Published numbers (do not expect them from smokes)

| Dataset | Quote | Lesson |
|---------|------:|--------|
| DOTA Oriented R-CNN 1× | **76.73%** Task 1 mAP50 | 04, 05, 07 |
| HRSC2016 Oriented R-CNN 3× | **90.41%** held-out test mAP50 | 08 (reproduced on all 453 test images) |
| FAIR1M Faster R-CNN 1× | **36.70%** tiled-val mAP50 | 09 |
| SSDD Faster R-CNN 1× | **90.34%** held-out test mAP50 | 10 |
| HRSID Faster R-CNN 1× | **78.55%** held-out test mAP50 | 10 |

## Reading-only posts

Linked from the lessons where useful; not turned into notebooks:

- [Oriented-det is coming](https://deeplearning.earth/posts/2026-05-28_introducing_oriented-det_sovereign_oriented_object_detection_for_eo/)
- Release notes: [v0.1.0](https://deeplearning.earth/posts/2026-06-22_oriented-det_v0_1_0_sovereign_oriented_object_detection_for_eo/), [v0.1.1](https://deeplearning.earth/posts/2026-07-11_oriented-det_v0_1_1_prob_iou_mmrotate_parity_and_the_updated_zoo/), [v0.2.0](https://deeplearning.earth/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/), [v0.3.1](https://deeplearning.earth/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/)

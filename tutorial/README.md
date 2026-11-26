# Tutorial

An 11-lesson OrientedDet course for **v0.4.2**, numbered **101–111** so Kaggle notebooks can be titled *OrientedDet 101*, *OrientedDet 102*, …. The path is **use → labels/scenes → compare models → train/eval → geometry/losses → finetune → export**.

**Course map (Kaggle-friendly):** [`100_index.ipynb`](100_index.ipynb) · [live on Kaggle](https://www.kaggle.com/code/jeffaudi/orienteddet-tutorial-index) — curriculum table with live kernel links, staging, setup, and published mAP. Prefer that notebook on Kaggle; this README is the GitHub view.

**Live kernels:**

- [OrientedDet tutorial index](https://www.kaggle.com/code/jeffaudi/orienteddet-tutorial-index)
- [OrientedDet 101 — Your first oriented detections](https://www.kaggle.com/code/jeffaudi/orienteddet-101-your-first-oriented-detections)
- [OrientedDet 102 — The fifteen DOTA classes](https://www.kaggle.com/code/jeffaudi/orienteddet-102-the-fifteen-dota-classes)
- [OrientedDet 103 — Large images and sliding windows](https://www.kaggle.com/code/jeffaudi/orienteddet-103-large-images-and-sliding-windows)
- [OrientedDet 104 — Detector zoo](https://www.kaggle.com/code/jeffaudi/orienteddet-104-detector-zoo)
- [OrientedDet 105 — Recipes, training and evaluation](https://www.kaggle.com/code/jeffaudi/orienteddet-105-recipes-training-and-evaluation)
- [OrientedDet 106 — Boxes, angles and rotated NMS](https://www.kaggle.com/code/jeffaudi/orienteddet-106-boxes-angles-and-rotated-nms)
- [OrientedDet 107 — Box losses: IoU, ProbIoU, QFL](https://www.kaggle.com/code/jeffaudi/orienteddet-107-box-losses-iou-probiou-qfl)
- [OrientedDet 108 — HRSC2016: training a ship detector](https://www.kaggle.com/code/jeffaudi/orienteddet-108-hrsc2016-training-ship-detector)
- [OrientedDet 109 — FAIR1M: data prep and finetune](https://www.kaggle.com/code/jeffaudi/orienteddet-109-fair1m-data-prep-and-finetune)
- [OrientedDet 110 — SAR ships: SSDD and HRSID](https://www.kaggle.com/code/jeffaudi/orienteddet-110-sar-ships-ssdd-and-hrsid)
- [OrientedDet 111 — Export RTMDet to ONNX](https://www.kaggle.com/code/jeffaudi/orienteddet-111-export-rtmdet-to-onnx)

**Default inference detector:** Hub [`rotated_rtmdet_dota_le90_3x`](https://huggingface.co/DL4EO/oriented-det) (CSPNeXt-S, official Task 1 **77.60%**, ONNX-capable). Lesson 104 also runs **YOLO-OBB** (`yolo_obb_dota_le90_12x`, Task 1 **77.80%**). Finetune smokes (108–110) keep the published Oriented R-CNN / Faster R-CNN recipes — there is no RTMDet/YOLO-OBB Hub zoo for HRSC, FAIR1M, SSDD, or HRSID yet.

## Curriculum

| # | Notebook | You learn | Posts | Kaggle inputs | Trains |
|---|----------|-----------|-------|---------------|--------|
| 101 | [`101_first_inference.ipynb`](101_first_inference.ipynb) | Hub checkpoints and sidecars, `run_inference_auto`, reading a detection, oriented vs axis-aligned boxes, the `image-demo` CLI — starts with **RTMDet** | [macOS pure Python](https://deeplearning.earth/posts/2026-06-25_oriented_object_detection_on_macos_in_pure_python/) · [v0.1.0](https://deeplearning.earth/posts/2026-06-22_oriented-det_v0_1_0_sovereign_oriented_object_detection_for_eo/) | none | no |
| 102 | [`102_dota_fifteen_classes.ipynb`](102_dota_fifteen_classes.ipynb) | The DOTA benchmark, class order, per-class Task 1 results, public demo predictions, ground truth vs predictions on sample tiles | [15 DOTA classes](https://deeplearning.earth/posts/2026-06-23_oriented_rcnn_detections_for_the_15_dota_classes/) · [Apache vs DOTA](https://deeplearning.earth/posts/2026-09-10_oriented_det_apache_license_versus_dota/) | none (sparse-clones `demo/dota_samples/`) | no |
| 103 | [`103_large_images_sentinel2.ipynb`](103_large_images_sentinel2.ipynb) | Sliding-window inference, window overlap, a zero-shot Sentinel-2 ship count with physical lengths (RTMDet default) | [Final pretrained model](https://deeplearning.earth/posts/2026-06-29_announcing_the_final_oriented_det_pretrained_model/) · [Sentinel-2 ships](https://deeplearning.earth/posts/2026-06-25_zero-shot_ship_detection_on_a_copernicus_sentinel-2_tile_with_oriented_rcnn/) | none | no |
| 104 | [`104_detector_zoo_thresholds.ipynb`](104_detector_zoo_thresholds.ipynb) | Speed tier first (RTMDet + YOLO-OBB), then ResNet zoo; score distributions, deploy floors, size and latency | [Optical demo](https://deeplearning.earth/posts/2026-09-06_oriented_det_optical_satellite_demo/) · [FCOS vs ORCNN](https://deeplearning.earth/posts/2026-09-02_rotated_fcos_vs_oriented_rcnn_on_macos/) | none | no |
| 105 | [`105_config_train_eval.ipynb`](105_config_train_eval.ipynb) | JSON recipes and `_base_` inheritance, patching an RTMDet recipe, `odet train` outputs, oriented mAP, `val_*` vs `test_*` thresholds, Task 1 vs local val | [DOTA lessons](https://deeplearning.earth/posts/2026-09-17_lessons_learned_on_dota_oriented_det_and_mmrotate_parity/) · [v0.4.0](https://deeplearning.earth/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/) | none (sparse-clones `demo/dota_samples/`) | no |
| 106 | [`106_boxes_angles_nms.ipynb`](106_boxes_angles_nms.ipynb) | `RBox`, the le90 convention, DOTA quads, flips and rotations, rotated IoU, rotated vs axis-aligned NMS | [v0.1.0](https://deeplearning.earth/posts/2026-06-22_oriented-det_v0_1_0_sovereign_oriented_object_detection_for_eo/) | none | no |
| 107 | [`107_probiou_box_losses.ipynb`](107_probiou_box_losses.ipynb) | Box losses side by side: L1, decoded rIoU, ProbIoU, KFIoU; RTMDet QFL; YOLO-OBB DFL/TAL; which recipe uses which | [ProbIoU](https://deeplearning.earth/posts/2026-07-10_rotated_faster_rcnn_probiou_dota/) · [v0.2.0](https://deeplearning.earth/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/) | none | no |
| 108 | [`108_hrsc2016_finetune.ipynb`](108_hrsc2016_finetune.ipynb) | A new dataset end to end: HRSC2016 statistics, `keep_ratio` resizing, evaluating the Hub 3× model, a training smoke from ImageNet weights | [HRSC2016](https://deeplearning.earth/posts/2026-09-13_hrsc2016_recipes_trains_and_results/) | [`hrsc2016-complete`](https://www.kaggle.com/datasets/juangarciaanton/hrsc2016-complete) (preferred) or [`guofeng/hrsc2016`](https://www.kaggle.com/datasets/guofeng/hrsc2016) (zip of RARs) | optional smoke |
| 109 | [`109_fair1m_finetune.ipynb`](109_fair1m_finetune.ipynb) | Fine-grained data prep: FAIR1M's 37 classes, convert to DOTA, image-level holdout, `odet tile-dota`, fine-tune from a DOTA Hub checkpoint | [FAIR1M](https://deeplearning.earth/posts/2026-09-24_fair1m_fine_grained_oriented_detection/) | [`ollypowell/fair1m-…`](https://www.kaggle.com/datasets/ollypowell/fair1m-satellite-imagery-for-object-detection) | optional smoke |
| 110 | [`110_sar_ssdd_hrsid.ipynb`](110_sar_ssdd_hrsid.ipynb) | Optical to radar: SSDD and HRSID, zero-shot RTMDet DOTA baseline on SAR, fine-tuning with the published Faster R-CNN recipes | [v0.3.0](https://deeplearning.earth/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/) | SSDD auto-download (or Kaggle/`SSDD_DATA_ROOT`); HRSID dump optional | optional smoke |
| 111 | [`111_onnx_export.ipynb`](111_onnx_export.ipynb) | Export **RTMDet** (`rotated_rtmdet_pre_nms`), ONNX Runtime inference, parity with PyTorch | [v0.3.0](https://deeplearning.earth/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/) · [export docs](../docs/examples/export.md) | none | no |

Lessons 101, 103, 104, 106, and 107 need nothing but Internet. Lessons 102 and 105 use the 36 DOTA val tiles in [`demo/dota_samples/`](../demo/dota_samples/) for their ground-truth sections (academic-use only). Outside a checkout they sparse-clone just that folder from GitHub into the work dir (~12 MB; Internet ON). Lesson 102 also runs a multi-class prediction on the public `demo/demo.jpg` everywhere.

### Next (not in this pass)

- **Airbus Playground** and **your own data**: the loaders and recipes exist (see [data.md](../docs/user-guide/data.md)), but the data is proprietary or user-supplied, so there is no notebook yet. Lesson 109 is the closest template: convert to DOTA, tile, fine-tune from a Hub checkpoint.
- **RTMDet / YOLO-OBB finetune recipes** for HRSC, FAIR1M, SSDD, HRSID: not published yet; lessons 108–110 stay on the Oriented R-CNN / Faster R-CNN Hub numbers.

## Local vs Kaggle

Every notebook opens with the same two setup cells:

1. **Install**: an editable `pip install -e <repo>` when the notebook sits inside a checkout (`pyproject.toml` + `oriented_det/` found above it), otherwise `pip install "oriented-det>=0.4.2"`. Lesson 111 adds the `[export]` extras (`onnx`, `onnxruntime`, `onnxscript` — needed by PyTorch 2.5+ `torch.onnx.export`).
2. **Environment and helpers**:
   - **Work dir**: `/kaggle/working` on Kaggle, else `$ORIENTED_DET_TUTORIAL_WORK`, else `work_dirs/` in the checkout (gitignored). `ORIENTED_DET_PROJECT_ROOT` points there, so `odet train` runs land in `work_dirs/runs/`.
   - **Hub weights**: `hf://…` slugs cache into `pretrained/` in a checkout; on a wheel install `ORIENTED_DET_PRETRAINED_DIR` defaults to the work dir. `hub_files(slug)` also fetches the checkpoint's sidecar `.json` (needed by the CLI and `load_model_from_checkpoint`).
   - **Demo files**: from the checkout `demo/`, else downloaded from GitHub raw into the work dir.
   - **Datasets**: scan `/kaggle/input` for a known name, else `HRSC_DATA_ROOT` / `FAIR1M_ROOT` / `SSDD_DATA_ROOT` / `HRSID_DATA_ROOT` / `DOTA_TILED_ROOT`.
   - **Device**: CUDA → MPS → CPU.
   - **Helpers used across lessons**: `hub_files`, `hub_model`, `detect`, `show`, `run` (echoes and runs an `odet` command).

Training cells default to **off** (`RUN_SMOKE = False` in 108 and 110, `RUN_TRAIN = False` in 109). The smokes are one epoch on a few dozen images, enough to exercise loading, augmentation, loss, and checkpointing, and they need a GPU. Full 1× / 3× recipes run from the CLI: `odet train --config configs/…`.

### Local

```bash
cd /path/to/oriented-det
jupyter lab tutorial/
```

### Kaggle

Open the [tutorial index on Kaggle](https://www.kaggle.com/code/jeffaudi/orienteddet-tutorial-index) or the live kernels linked from [`100_index.ipynb`](100_index.ipynb) (Internet ON). A GPU is optional: recommended for 104, 108, and 110; required only for the training smokes. Lessons 108–110 use the dataset inputs listed in the curriculum table when those kernels are published.

## Published numbers (do not expect them from smokes)

| Dataset | Quote | Lesson |
|---------|------:|--------|
| DOTA RTMDet-S 3× | **77.60%** Task 1 mAP50 | 101, 102, 103, 104, 111 |
| DOTA YOLO-OBB-S 12× | **77.80%** Task 1 mAP50 | 104 |
| DOTA Oriented R-CNN 1× | **76.73%** Task 1 mAP50 | 104, 105 |
| HRSC2016 Oriented R-CNN 3× | **90.41%** held-out test mAP50 | 108 (reproduced on all 453 test images) |
| FAIR1M Faster R-CNN 1× | **36.70%** tiled-val mAP50 | 109 |
| SSDD Faster R-CNN 1× | **90.34%** held-out test mAP50 | 110 |
| HRSID Faster R-CNN 1× | **78.55%** held-out test mAP50 | 110 |

## Reading-only posts

Linked from the lessons where useful; not turned into notebooks:

- [Oriented-det is coming](https://deeplearning.earth/posts/2026-05-28_introducing_oriented-det_sovereign_oriented_object_detection_for_eo/)
- Release notes: [v0.1.0](https://deeplearning.earth/posts/2026-06-22_oriented-det_v0_1_0_sovereign_oriented_object_detection_for_eo/), [v0.1.1](https://deeplearning.earth/posts/2026-07-11_oriented-det_v0_1_1_prob_iou_mmrotate_parity_and_the_updated_zoo/), [v0.2.0](https://deeplearning.earth/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/), [v0.3.1](https://deeplearning.earth/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/), [v0.4.0 / RTMDet](https://deeplearning.earth/), [v0.4.2 / YOLO-OBB](https://deeplearning.earth/)

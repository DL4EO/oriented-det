# CLI script implementations

After `uv pip install -e .` (or `make install`), use the **`odet`** command for day-to-day work (`odet train`, `odet preds`, …). See [oriented_det/cli/README.md](../oriented_det/cli/README.md) and the [repository layout](../README.md#repository-layout) in the main README.

This directory holds the **Python modules** that implement `odet` subcommands and supports direct `python tools/...` invocation. Shared inference and collate logic lives in [`oriented_det/runtime/`](../oriented_det/runtime/).

**PyPI configs:** `sync_vendored_configs.py` copies manifest-listed files from repo `configs/` → `oriented_det/configs/`. Use `make sync-configs` / `make check-configs` (see [configs/README.md](../configs/README.md)).

**Running without `odet`:** From the repo root, `python tools/train.py --config …` is equivalent to `odet train --config …` for debugging. Prefer **`odet`** or **`make`** in docs and CI so paths stay consistent.

**Thin compatibility modules:** `tools/inference.py` and `tools/helpers.py` re-export `oriented_det.runtime` and emit a deprecation warning; new code should import from `oriented_det.runtime.inference` and `oriented_det.runtime.collate` directly.

## Quick Start with Makefile

From the **repository root**, the Makefile calls **`odet`** under the hood. Run `make help` for targets; defaults (`CONFIG`, DOTA paths) are at the top of the Makefile. Use **`make train-multi-gpu`** (not a raw `odet train` in a misconfigured shell) so `torchrun -m oriented_det.cli.train` and pip cuDNN on `LD_LIBRARY_PATH` are applied.

Common targets:

```bash
make install          # Install the package (required before running tools)
make help             # Show all available commands
make train            # Train with CONFIG (checkpoint.* in JSON only)
make train-multi-gpu  # Multi-GPU training (same CONFIG)
make eval-val           # `make preds` then `make metrics` on newest `predictions/<ts>/`
make preds              # Val inference → `predictions/<ts>/` (experiment `production.*`; no GPU mAP)
make metrics            # Offline mAP/PR from latest `predictions/*/` or `METRICS_PRED_DIR=...` (defaults from JSON metadata)
make viewer             # Gradio: browse latest tiled predictions under `predictions/` (run `make preds` first)
make demo               # image_demo on all top-level images in demo/ (latest exp; see demo/README.md)
make free-gpu         # Kill GPU processes to free memory
make test             # Run tests
make docs-serve       # Build and serve documentation
make clean            # Remove generated output files
```

For **tiling** and **inference** on single images, run the scripts directly (see Scripts below); there are no `make tile` / `make inference` targets.

## Scripts

### `image_demo.py`

Run inference on image(s) using oriented-det **config + checkpoint**. Loads model type, num_classes, preprocessing, and class names from the config. For registered pretrained weights, you can omit the config and pass only `hf://<slug>`; the checkpoint sidecar JSON is used automatically. If you do pass a config, a sidecar config beside the `.pth` is preferred only when the provided config is the checkpoint's manifest `source_recipe`; a different config is kept as-is. Detection **labels are 1-based** foreground ids (same convention as training `class_map`); overlay text uses `class_names[label - 1]`.

**Inference:** Same as `run_inference_auto`: **`pad`** always one training-style whole-image forward; **`fixed`/`crop`** one forward when the image fits the canvas, otherwise **`run_inference_sliding_window`**. Use `--zoom 2` or `--zoom 4` to upscale for inference only; detections are mapped back to the original image before visualization.

**Usage:**
```bash
# Single image (output path optional)
python tools/image_demo.py demo/demo.jpg configs/oriented_rcnn/dota_le90_1x.json runs/.../checkpoints/best.pth --out-file result.jpg

# Registered pretrained weights (sidecar config auto-resolved)
python tools/image_demo.py demo/demo.jpg hf://oriented_rcnn_dota_le90_3x --out-file result.jpg
python tools/image_demo.py demo/demo.jpg hf://oriented_rcnn_dota_le90_1x --out-file result.jpg

# All images in a directory (writes to demo/out by default)
python tools/image_demo.py demo configs/.../config.json runs/.../checkpoints/best.pth --out-dir demo/out

# Options
python tools/image_demo.py demo/demo.jpg config.json checkpoint.pth \
    --out-file result.jpg --device cuda:0 --score-thr 0.3 --nms-thr 0.5 \
    --classes ship --zoom 2 --overlap-pixels 200 --ignore-margin-pixels 100 \
    --window-batch-size 8 --json-per-image --json-batch demo/out/json
```

**Arguments:** `img` (file or directory), then either `checkpoint` alone (registered pretrained checkpoint with sidecar config) or `config` + `checkpoint`. Optional: `--out-file`, `--out-dir`, `--device`, `--score-thr`, `--nms-thr`, `--classes`, `--zoom`, `--overlap-pixels` (default from `production.overlap_pixels`, else 200), `--ignore-margin-pixels` (default from `production.ignore_margin_pixels`, else dataset overlap/2), `--overlap-ratio` (pad/tile path only; ratio overrides pixels), `--window-batch-size` (fixed sliding-window micro-batch; skips auto GPU probing), `--json-per-image` (rich JSON next to each visualization: rbox, polygon, run metadata), or `--json-batch [PATH]` (compact batch JSON for pipelines: per-image files plus combined `detections.json` when `img` is a directory).

Demo images: place test images in `demo/` (see `demo/README.md` for a sample image or using your own).

**Makefile:** `make demo` runs `tools/image_demo.py` on every top-level `*.jpg` / `*.jpeg` / `*.png` in `DEMO_DIR` (default `demo/`) with the latest `runs/` checkpoint and config; outputs default to `demo/out/`. Variables: `DEMO_DIR`, `IMAGE_DEMO_OUT_DIR`, `IMAGE_DEMO_DEVICE` (see top-level `Makefile`).

### Horizontal BB → oriented BB (`hbb_to_obb.py`, `filter_predictions_by_gt.py`, `generate_oriented_annotations.py`)

Convert **horizontal** ground-truth boxes to **oriented** boxes using model predictions from `odet image-demo --json-batch`:

1. Run inference on images (high recall: e.g. `--score-thr 0.05 --nms-thr 0.5`).
2. Match predictions to horizontal GT by oriented IoU; drop unmatched detections (false positives).
3. For each GT box: use the matched oriented prediction, or fall back to the horizontal box (0°).

**Supported GT formats** (`--gt-format`):

| Format | Layout | Notes |
|--------|--------|-------|
| `csv` | `annotations.csv` with `id`, `image_id`, `geometry`, `class` | Polygon corners (axis-aligned OK) |
| `yolo` | `images/` + `labels/*.txt` | `class cx cy w h` normalized |
| `dota` | `images/` + `labelTxt/*.txt` | Standard DOTA lines |

**Output** (`--output-format`): `csv` → `annotations_oriented.csv`; `dota` → per-image `.txt` in `labels_oriented/`.

```bash
# 1) Inference
odet image-demo /path/to/images hf://oriented_rcnn_dota_le90_3x \
  --classes plane --score-thr 0.05 --nms-thr 0.5 \
  --json-batch /path/to/predictions --out-dir /path/to/predictions/vis

# 2) Optional: filter only (inspect FP removal)
python tools/filter_predictions_by_gt.py \
  --gt-format csv --dataset-root /path/to/data --annotations annotations.csv \
  --detections-json /path/to/predictions/detections.json \
  --output-json /path/to/predictions/detections_gt_filtered.json \
  --ignore-class Truncated_airplane

# 3) Generate oriented annotations (CSV or DOTA)
python tools/generate_oriented_annotations.py \
  --gt-format yolo --dataset-root /path/to/data \
  --yolo-class-name 0=plane \
  --detections-json /path/to/predictions/detections.json \
  --output-format csv

# Compare raw prediction counts vs horizontal GT
python tools/compare_hbb_obb_counts.py \
  --gt-format csv --annotations annotations.csv \
  --detections-json /path/to/predictions/detections.json
```

Shared library: `tools/hbb_to_obb.py`.

### `train.py`

Complete training script for oriented object detection models. Uses JSON configuration files (similar to MMRotate) and supports multiple model types.

**TensorBoard:** validation images under `val/predictions` include the source file basename (with extension) in a label at the top-left, taken from the collate target `image_filename` (`oriented_det.runtime.collate`).

**Usage:**
```bash
# Basic training with config file
python tools/train.py --config configs/oriented_rcnn/dota_le90_1x.json

# Override batch size
python tools/train.py --config configs/oriented_rcnn/dota_le90_1x.json --batch-size 4

# Enable mixed precision training
python tools/train.py --config configs/oriented_rcnn/dota_le90_1x.json --use-amp

# Disable mixed precision training
python tools/train.py --config configs/oriented_rcnn/dota_le90_1x.json --no-amp
```

For **checkpoints**, edit the JSON `checkpoint` section: `load_from_checkpoint`, `load_from_experiment`, `discover_previous_run`, `resume_from_checkpoint_epoch`, etc. See `configs/config.schema.json` and `oriented_det/train/README.md`.

**Troubleshooting: `GET was unable to find an engine to execute this computation`**

This usually comes from **cuDNN** during convolution forward/backward (not from oriented-det code). Try in order:

1. **Disable cuDNN benchmark** (often fixes bad algorithm selection):
   ```bash
   ORIENTED_DET_CUDNN_BENCHMARK=0 python tools/train.py --config configs/.../config.json
   ```
2. **Disable AMP**: `python tools/train.py --config configs/.../config.json --no-amp`
3. **`LD_LIBRARY_PATH`**: an older system CUDA/cuDNN can shadow libraries shipped with your PyTorch install. Compare the variable in shells where training works vs fails; try `unset LD_LIBRARY_PATH` temporarily (common with conda + manual CUDA).
4. Align **PyTorch / CUDA / driver** using `python -m torch.utils.collect_env`.

**Troubleshooting: NCCL `ALLREDUCE` / watchdog timeout after “Computing final mAP…”**

Rank 0 runs **full mAP** inside `evaluate()` (can take many minutes with large val sets). Other ranks skip mAP and used to exit `train()` first and call `destroy_process_group()` while rank 0 was still computing → broken process group and a long NCCL timeout. **Fixed** in `oriented_det/train/engine.py` with a `dist.barrier()` after final mAP when using DDP. If you still see timeouts, increase `NCCL_TIMEOUT` or set `evaluation.compute_map_final` to `false` and run a separate single-GPU eval on the best checkpoint.

**Configuration:**
Configuration is loaded from JSON files in the `configs/` directory. Each config file includes:
- Dataset paths (`data_root`, `train_tiles_dir`, `val_tiles_dir`)
- Model configuration (`backbone`, `anchor_scales`, `anchor_ratios`, etc.)
- Training hyperparameters (`batch_size`, `learning_rate`, `num_epochs`, etc.)
- Loss configuration (class weighting, focal loss, etc.)
- Evaluation settings
- Checkpoint settings

See `configs/` directory for reference configs for different model types.

### `publish_checkpoint.py`

Prepare a training checkpoint for Hub distribution: strip optimizer state, save CPU tensors, append **SHA-256[:8]** to the filename (MMDet-style).

```bash
python tools/publish_checkpoint.py \\
  runs/oriented_rcnn/20260621-092802/checkpoints/best_mAP_0.82.pth \\
  pretrained/oriented_rcnn_r50_fpn_dota_le90_3x
# -> pretrained/oriented_rcnn_r50_fpn_dota_le90_3x-<hash8>.pth
```

Copy the experiment **`config.json`** and **`train.log`** beside the published weight as `<weight-stem>.json` and `<weight-stem>.log` (same hash stem as the `.pth`).

Update `oriented_det/pretrained/manifest.json` with the new `filename` and `sha256`, then upload to the Hub repo (`make upload-pretrained` uploads `.pth`, sidecar `.json`, and `.log`).

### `lr_finder.py`

Learning rate finder: runs a short training sweep with exponentially increasing learning rate, records loss vs LR, **stops early** when EMA-smoothed loss exceeds `stop_mult` × the best smoothed loss so far (fastai `LRFinder` / `stop_div`), then **truncates the recorded trace** after the first raw divergence (`loss > stop_mult ×` running minimum before that step) before computing suggestions. Heuristics use a trimmed trace (skip first `num_steps//10` and last 5 points, like `Learner.lr_find`). Prints **valley** (default primary), **steep**, **minimum** (LR at min loss ÷ 10), and **slide** when they succeed; primary order: valley → steep → minimum → slide, else geometric mid of the sweep. Single-GPU only. Uses the same config and data as training.

**Usage:**
```bash
# Basic run (uses config's batch size and AMP)
python tools/lr_finder.py --config configs/oriented_rcnn/dota_le90_1x.json

# Custom sweep length and save plot
python tools/lr_finder.py --config configs/.../config.json --num-steps 150 --output lr_finder.png

# Disable AMP for the sweep; restore model state after (so you can train without reloading)
python tools/lr_finder.py --config configs/.../config.json --no-amp --restore
```

**Options:**
- `--config` — Path to training config JSON (required)
- `--batch-size` — Override batch size
- `--num-steps` — Number of steps in the LR sweep (default: 100)
- `--start-lr` / `--end-lr` — LR range (default: 1e-7 to 10.0)
- `--no-amp` — Disable mixed precision for the sweep
- `--restore` — Restore model weights after the sweep
- `--output` — Save loss-vs-LR plot (e.g. `lr_finder.png`; marks divergence cut when used)
- `--no-early-stop` — Run all `num-steps` even if loss explodes (truncation for suggestions still applies unless you also raise `--stop-mult` very high)
- `--stop-mult K` — Early stop and divergence cut when loss (or smoothed loss for early stop) exceeds `K` × best-so-far (default: 4)
- `--smooth-beta` — EMA for smoothed loss during early stop, `0 ≤ beta < 1` (default: 0.98)

Use the printed **primary** LR (or another heuristic) in your config’s `training.learning_rate`; the tool also prints values scaled to the config batch size when the sweep used a different `--batch-size`. Scale further by world size / gradient accumulation as in `train.py` if needed. The model is updated during the sweep; for clean training, start from scratch or load a checkpoint.

From the repo root you can run: `make lr-finder` or `make lr-finder CONFIG=configs/.../config.json` (optional: `OUTPUT=lr_finder.png`).

### `train.py` (continued)

**Features:**
- Config-based training (all parameters in JSON files)
- Supports multiple model types (Rotated Faster R-CNN, Oriented R-CNN, Rotated RetinaNet, Rotated FCOS)
- Command-line parameter overrides (batch size, AMP, etc.)
- Load DOTA dataset (automatically discovers classes)
- Automatic checkpointing and experiment management
- Learning rate scheduling with warmup
- Mixed precision training (configurable)
- Gradient accumulation
- Class distribution analysis
- Automatic resume from last experiment

### `save_predictions.py`

Runs inference on a dataset split, writes `predictions.json`, and (when diagnostics are enabled) produces:

- `analysis_iouXX.json`: PR/F1 sweep + **per-class AP** (`class_aps`) and **MMRotate-style stats** (`class_metrics`: gts, dets, recall, ap per class) + **`gt_alignment_metrics`** (global and per-class mean best IoU vs raw detections)
- `model_analysis_<timestamp>.md`: human-readable report including a **per-class gts / dets / recall / AP** table and a **GT alignment (mean best IoU)** section

**Note:** This tool requires a properly formatted DOTA dataset with tiled images:
```
dota_root/
  train/
    tiles_1024/
      images/
        P0001_0_0.png
        ...
      labels/
        P0001_0_0.txt
        ...
  val/
    tiles_1024/
      images/
        ...
      labels/
        ...
```

See `tools/tile_dota.py` to create tiles from large DOTA images.

### `oriented_det.runtime.inference`

Run inference with trained models.

**Usage:**
```bash
# Basic inference
python -m oriented_det.runtime.inference image.jpg \
    --checkpoint checkpoints/best.pth \
    --output detections.png

# With custom thresholds
python -m oriented_det.runtime.inference image.jpg \
    --checkpoint checkpoints/best.pth \
    --score-threshold 0.7 \
    --nms-threshold 0.5 \
    --output detections.png

# Specify model type and classes
python -m oriented_det.runtime.inference image.jpg \
    --checkpoint checkpoints/best.pth \
    --model-type oriented_rcnn \
    --num-classes 15 \
    --class-names plane ship vehicle ... \
    --output detections.png

# With config: pad/tile when image size ≠ model input (e.g. large 2048×2048 or small 512×512 vs 1024×1024)
python -m oriented_det.runtime.inference large_image.png --checkpoint best.pth --config runs/.../config.json --output out.png
# Optional: --overlap-pixels 200 (default) or --overlap-ratio 0.2
# Speed: use batched sliding-window inference (more GPU utilization)
ORIENTED_DET_WINDOW_BATCH_SIZE=16 python -m oriented_det.runtime.inference large_image.png --checkpoint best.pth --config runs/.../config.json --output out.png
```

**Features:**
- Load trained models from checkpoints
- Preprocess images
- Run inference
- Apply NMS to filter detections
- Visualize results with labels
- Print detection summaries
- **Padded-canvas / sliding-window inference:** With `--config`, inference **always** uses pad/tile windows to the model input (`preprocessing.target_size`): zero-pad small images, tile large ones; no full-image resize and no rescaling of box coordinates afterward. Before merge, each window drops detections whose centroid lies in the overlap band (**margin = overlap / 2** per axis; interior window sides only — full-image borders are exempt). Detections are merged and NMS runs in original image coordinates. Use **`--overlap-pixels`** (default **200** per axis, aligned with `tile_dota.py`) or **`--overlap-ratio`** in `[0,1)` (if set, overrides pixel overlap) when multiple tiles are used.
- **Sliding-window batching:** Sliding-window inference batches multiple windows per forward. **Default: auto** on CUDA/MPS (one-time binary search for the largest safe batch; cached) unless you set a positive int via `--window-batch-size` or `ORIENTED_DET_WINDOW_BATCH_SIZE=8` (or `auto` explicitly). CPU defaults to 4.

### `save_predictions.py`

Run inference on a validation (or train) split, save predictions to JSON, and optionally compute mAP in the same process (`--no-diagnostics` skips GPU-side metrics). Use **`make eval-val`** for inference plus offline metrics in one step, or after **`make preds`**, run **`make metrics`** (or `python tools/save_predictions.py --metrics-from-json path/to/dir`) to recompute mAP/PR with different `--iou-threshold`, `--metrics-margin-pixels`, PR sweep steps, etc., **without** re-running inference (metrics rebuild GT/det maps from `predictions.json`). Uses `oriented_det.runtime.inference.run_inference_auto`: **`resize_mode: pad`** (HRSC2016) always runs one training-style whole-image forward (scale long edge to `target_size`, then pad). **`fixed` / `crop`** (DOTA) use a single training-style forward when the image fits the canvas, and padded sliding windows when it is larger. Supports **override of the validation folder** for non-tiled DOTA val (e.g. to compare with literature mAP on full-size images). Model construction reads `model.fpn_returned_layers`, `model.fpn_strides`, and `model.trainable_layers` / `frozen_stages` from the experiment `config.json` so the backbone matches training (required when FPN does not use all ResNet stages, e.g. `[1,2,3]` without C5).

**GPU / cuDNN:** If every val image fits in one model tile, the script **does not** run the sliding-window batch probe (inference is already one forward per image). When some images need multiple tiles, the one-time probe may print `Plan failed with an OutOfMemoryError` **warnings** from cuDNN v8 trying convolution algorithms—that is usually the planner discarding a bad plan, not a failed run. If inference stalls or fragments memory after a probe, set `ORIENTED_DET_CUDNN_BENCHMARK=0` or a fixed `ORIENTED_DET_WINDOW_BATCH_SIZE` (see `oriented_det/runtime/inference.py`).

**Label / image coordinates:** For layouts with `images/` and `labels/` (or `labelTxt/`), each `basename.txt` must describe objects in **the same coordinate system** as `basename.jpg` (or `.png`)—pixel coords on that raster, top-left origin. The tool does not rescale or reproject labels to match the image.

**Usage:**
```bash
# Default: use config's val_tiles_dir, auto-detect latest experiment
python tools/save_predictions.py

# Override val folder (e.g. non-tiled DOTA val with full-size images)
python tools/save_predictions.py --config runs/.../config.json --val-dir /path/to/dota/val
# Pad/tile windows always; mAP compares predictions to labels in the images' pixel space (labels must match each image file).

# Overlap ratio between windows when the image needs multiple tiles (default 0.2)
python tools/save_predictions.py --config runs/.../config.json --val-dir /path/to/non-tiled/val --overlap-pixels 200
python tools/save_predictions.py --config runs/.../config.json --val-dir /path/to/non-tiled/val --overlap-ratio 0.25
```

**Options:** `--val-dir` overrides `config.dataset.val_tiles_dir` so you can point to a non-tiled validation folder (e.g. DOTA full-size images in `images/` and `labels/` or `labelTxt/`) while the config still references the tiled val. **`--no-diagnostics`** skips mAP/PR/analysis (inference-only JSON). **`--metrics-from-json PATH`** loads an existing `predictions.json` (PATH may be the file or its directory), recomputes metrics from stored boxes/scores/GTs, writes `analysis_*.json` / plots beside it, and refreshes `metadata.diagnostics`. Sliding-window overlap: **`--overlap-pixels`** (omit to use `production.overlap_pixels` from the experiment config when set, else **200**) or **`--overlap-ratio`** in `[0,1)` (if set, overrides pixels). Metrics edge margin: `--metrics-margin-pixels` discards GT/detections whose centroids fall in the outer overlap band (`[0, margin)` or `(W-margin, W]` per axis); keeps the tile interior `[margin, W-margin]` for **metrics only** (mAP/PR/per-image metrics), same rule as deploy `MARGIN`; when omitted, uses **`production.ignore_margin_pixels`** from the experiment config when set, else **overlap/2** (or the value stored in metadata when re-running metrics unless overridden on the CLI). If `--nms-threshold` is omitted, the tool uses `config.model.final_nms_iou_threshold` (fallback: 0.5; legacy `nms_threshold` is migrated on load). If `--iou-threshold` is omitted, mAP / PR matching uses **`effective_eval_metric_thresholds`** so **`production.iou_threshold`** overrides **`evaluation.iou_threshold`** when set, same merge order as validation during training—this is **rotated GT–det IoU**, not NMS IoU. If `--score-threshold` is omitted, the post-NMS filter for `run_inference_auto` and diagnostics uses the same helper so **`production.score_threshold`** wins when set. Logs print **`mAP50`-style** names (e.g. `Final mAP50: …`) and spell out NMS IoU separately so it is not confused with DOTA’s 0.1 NMS.

**Inference-time knob overrides (without editing config.json):** You can override the model’s proposal / prefilter behavior for evaluation-only runs:
- `--inference-pre-nms-score-threshold`: overrides `model.inference_pre_nms_score_threshold` (prefilter before final/merge NMS)
- `--rpn-pre-nms-top-n`: overrides `model.rpn_pre_nms_top_n`
- `--rpn-post-nms-top-n`: overrides `model.rpn_post_nms_top_n`
- `--rpn-nms-threshold`: overrides `model.rpn_nms_threshold` (proposal NMS)
- `--nms-class-agnostic` / `--no-nms-class-agnostic`: overrides `model.nms_class_agnostic`

**Progress (tqdm):** Bars use `oriented_det.utils.tqdm_progress_stream()` so they can render on the real terminal when stderr is piped (e.g. `2>&1 | tee preds.log`), like training’s `progress_stream`—log files stay line-based without `\\r` spam; mAP/PR use the same stream.

**Precision-recall and F1 analysis:** `save_predictions.py` computes a dataset-level precision-recall curve by sweeping score thresholds, selects the best **global** threshold using **F1**, and writes:
- `analysis_iou0.50.json` (threshold curve + best-threshold block + confusion matrix + per-image metrics including precision/recall/F1/F2 + **`gt_alignment_metrics`**)
- `pr_curve.png` (precision vs recall)
- `threshold_metrics.png` (precision/recall/F1 vs threshold)
- `model_analysis_<timestamp>.md` (timestamped model report with model/date/data/metrics, **per-class gts/dets/recall/AP** table like MMRotate, **per-class mean best IoU (GT alignment)** table, optional **per-class best-threshold table** (F1), confusion matrix, and artifact references)

The confusion matrix is computed at the best global F1 threshold. Rows are GT classes and columns are predicted classes; the extra `False Positive` row contains unmatched detections, and the extra `Missed` column contains unmatched GTs.

Sweep controls:
- `--pr-threshold-min` (default `0.0`)
- `--pr-threshold-max` (default `1.0`)
- `--pr-threshold-step` (default `0.05`)
- `--pr-iou-threshold` (default: uses `--iou-threshold`; lets PR/F1 matching IoU differ from mAP IoU)

**Per-class score thresholds:** For `save_predictions.py`, merged **`evaluation.*`** + **`production.*`** thresholds come from **`effective_eval_metric_thresholds`** (same rule as training-side val when production fields are set). `predictions.json` metadata records the applied `score_threshold` and `per_class_score_threshold`.

**Per-class best threshold analysis:** With `--per-class-threshold-analysis`, the analysis JSON also includes `best_threshold_per_class` (best F1 per class over the same threshold grid). This is expensive on large datasets.

**Per-tile metrics CSV:** Pass `--tile-metrics-csv path.csv` (relative paths are resolved under the run output directory). Writes one row per image (columns match `per_image_metrics` in the analysis JSON), for joining with training oversampling (`dataset.tile_metrics_csv` in config). Tiles with **no ground truth and no predictions** above the chosen global threshold get **precision = recall = F1 = F2 = 1.0** (correct empty image), so they are not treated as low-F1 failures. Training still ignores `tp=fp=fn=0` rows as hard tiles when reusing older CSVs that stored F1=0 for those cases.

**Hard-tile oversampling in training:** From the repo root, `make train-preds` runs the train split (latest experiment) and writes `tile_metrics.csv` under `<latest_exp>/train_tile_eval/` by default (override with `SAVE_TRAIN_PRED_OUT=/path`). Equivalent manual command: `python tools/save_predictions.py --data-split train --tile-metrics-csv tile_metrics.csv ...`. Then set in your training config: `dataset.tile_metrics_csv` to that CSV path, plus optional `hard_tile_metric_column` (default `f1`), `hard_tile_threshold` (default `0.8`), `hard_tile_oversample_factor` (default `2.0`). Single-GPU training uses `WeightedRandomSampler`; multi-GPU expands the dataset index list so `DistributedSampler` sees more draws of hard tiles. Optional **`dataset.drop_easy_empty_tiles: true`** (requires the CSV) removes train tiles with `tp=fp=fn=0` before `max_train_samples` and oversampling, so you can keep `filter_empty_gt: false` in the loader: easy empty tiles leave the epoch, empty tiles with false positives stay and can be oversampled. Tiles with no CSV row are kept. Validation is unchanged.

**`make preds` / `make metrics` (Makefile):** The root `Makefile` does **not** pass score, IoU, overlap, or NMS overrides. **`make preds`** runs `tools/save_predictions.py` with **`--no-diagnostics`** only; thresholds and tiling come from the experiment **`config.json`** (**`production.*`** for deploy-style inference; **`evaluation.*`** is for training-time validation). For tiled DOTA, **all tiles** under the split roots are included (**`dataset.filter_empty_gt` is not applied** at inference; training may still drop empty tiles). **`make metrics`** runs **`--metrics-from-json`** with no extra flags: mAP/PR reuse **`predictions.json`** metadata (the values written at inference time). To re-run metrics with different thresholds, call `python tools/save_predictions.py --metrics-from-json <dir> --score-threshold …` yourself.


### `app.py`

Gradio app to browse **predictions** from `save_predictions.py`, explore a DOTA dataset, or **edit OBB CSV annotations** with an optional read-only reference overlay.

The app includes a small Gradio 6.8 compatibility patch so cleared/null slider payloads fall back to the slider default instead of crashing during slider preprocessing.

Predictions mode (requires a predictions directory from `save_predictions.py`):
```bash
python tools/app.py --mode predictions --predictions-dir predictions/20250101_120000
# Optional: --data-root /path/to/dota --threshold 0.3 --port 7860
```

Dataset mode (browse DOTA labels without predictions):
```bash
python tools/app.py --mode dataset --data-root /path/to/dota --tiles-dir train/tiles_1024
```

CSV annotation editor (finalize HBB→OBB conversions or manual QA):
```bash
python tools/app.py --mode csv \
  --data-root /path/to/dataset \
  --annotations-csv /path/to/dataset/annotations_oriented.csv \
  --reference-csv /path/to/dataset/annotations.csv
```

- **Green** = read-only reference (`--reference-csv`, e.g. original horizontal GT).
- **Red** = editable layer (`--annotations-csv`); yellow outline = selected box.
- Edit **class**, **cx/cy/width/height/angle** fields and click **Apply changes**.
- **Add box**, **Delete selected**, **Copy reference → editable**, then **Save CSV** to write the editable file.

Related pipeline tools: `generate_oriented_annotations.py`, `filter_predictions_by_gt.py`, `compare_hbb_obb_counts.py` (see `hbb_to_obb.py`).


### `dataset_stats.py`

Dataset sanity checks, statistics (class distribution, annotations per image, image dimensions), and per-channel normalization mean/std for the dataset defined by your training config. Use the output mean/std in `preprocessing.normalize_mean` and `preprocessing.normalize_std`; values are in [0, 1] scale (after ToTensor).

**Usage:**
```bash
# Full run (sanity checks + stats + normalization)
python tools/dataset_stats.py --config configs/.../config.json

# Stats and sanity only (no normalization; faster)
python tools/dataset_stats.py --config path/to/config.json --stats-only

# Normalization only (legacy behavior)
python tools/dataset_stats.py --config path/to/config.json --normalization-only

# Quick run on a subset
python tools/dataset_stats.py --config path/to/config.json --max-samples 500

# Use validation split
python tools/dataset_stats.py --config path/to/config.json --split val
```

**Sanity checks:** Missing images, failed image loads, empty annotations, duplicate paths. **Stats:** Class counts, annotations per image (min/max/mean), image width/height (min/max/mean). **Output:** Prints normalization `normalize_mean` and `normalize_std` and a JSON snippet for `preprocessing` when normalization is run.

### `preview_augmentation.py`

Preview **training-time** augmentations from an experiment config before a long run. Loads the train (or val) split, applies the same collate path as `tools/train.py` (resize, optional random flips, optional random rotate, optional Albumentations), and writes comparison grids with oriented boxes drawn on each panel.

Use this to sanity-check `augmentation.json` / recipe overrides, flip / rotate settings in `preprocessing`, and `enable_albumentation` without starting training.

**Usage:**
```bash
# Recipe or resolved run config
python tools/preview_augmentation.py --config configs/oriented_rcnn/dota_le90_1x.json

# More tiles and random variants per tile
python tools/preview_augmentation.py --config runs/oriented_rcnn/20260616-030231/config.json --num-images 4 --variants 6

# Fixed dataset indices; Albumentations column without flips
python tools/preview_augmentation.py --config configs/.../config.json --indices 0,12,48 --include-albumentations-only

# Custom output directory
python tools/preview_augmentation.py --config configs/.../config.json --output-dir /tmp/aug_preview
```

**Options:**
- `--config` — Training recipe or resolved `config.json` (required)
- `--output-dir` — Where to write PNGs (default: `previews/augmentation/<config_stem>/` under the repo root)
- `--split` — `train` or `val` (default: `train`)
- `--num-images` — Random tiles to preview (default: `3`)
- `--variants` — Random augmented variants per tile (default: `4`)
- `--seed` — RNG seed for tile and augmentation sampling (default: `42`)
- `--indices` — Comma-separated dataset indices (overrides `--num-images` random sampling)
- `--include-albumentations-only` — Add a column with Albumentations only (no random flips)

**Output:** For each tile, one row PNG (`00042_<stem>.png`) with panels:
1. **baseline** — resize only, no augmentation
2. **albumentations only** — when `--include-albumentations-only` and `enable_albumentation` is true
3. **train aug #N** — full training collate (flips + random rotate + Albumentations when enabled)

Also writes `grid_all.png` (all rows stacked) and `meta.json` (config path, indices, seed, variants).

**Config loading:** Uses lenient parsing when the config contains unknown keys (e.g. older run configs); unknown section fields are dropped so preview still runs. Supports DOTA tiled datasets, Airbus Playground, and HRSC2016 (`dataset.format`).

**Typical workflow:** Run `dataset_stats.py` for normalization and class balance, then `preview_augmentation.py` to verify augmentations look reasonable on real tiles.

### `generate_airbus_playground_csv.py`

Build annotations and split CSVs under an Airbus Playground export root. By default both files are **dated as a pair** (`annotations_YYYYMMDD.csv` and `split_YYYYMMDD.csv`, UTC) so reruns on another day do not overwrite a previous snapshot; pass **`--annotations-file`** / **`--split-file`** for fixed names (if only one is dated, the other is inferred from the same date). Splits are **integer fold ids** (default **10** folds): groups `(dataset_id, zone_id, image_id)` are shuffled with `--seed` then assigned `0 .. num_splits-1` in round-robin order. Fold **0** is the conventional validation fold; rotate validation by setting **`dataset.val_split_id`** in your training JSON. After generation, set **`dataset.annotations_file`** and **`dataset.split_file`** in your training config to the printed filenames.

**Airbus Playground** (CSV generation → `make wizard` → `make lr-finder`): [Data guide](../docs/user-guide/data.md#airbus-playground).

**Usage:**
```bash
python tools/generate_airbus_playground_csv.py --data-root /path/to/playground_export
python tools/generate_airbus_playground_csv.py --data-root /path/to/export --seed 42
python tools/generate_airbus_playground_csv.py --data-root /path/to/export --ignore-label Confuser --map-label taxi=car
```

### `playground_to_dota.py`

Export Airbus Playground export folders to DOTA-format directories (images + `.txt` labels). Converts polygon annotations to DOTA OBB format. Optionally splits output into train/val using a **group-level ratio** (`--val-ratio`) or an existing **`split.csv`** from `generate_airbus_playground_csv.py`. For CSVs with **integer fold ids**, pass **`--val-split-id`** (default `0`) to choose which fold is written under `val/`.

**Usage:**
```bash
# Single output dir (no split)
python tools/playground_to_dota.py --data-root /path/to/playground_export --output-dir /path/to/dota_out

# Train/val split by image group
python tools/playground_to_dota.py --data-root /path/to/export --output-dir /path/to/dota_out --val-ratio 0.2 --seed 42

# Use existing split CSV (integer folds: val fold 0 unless --val-split-id is set)
python tools/playground_to_dota.py --data-root /path/to/export --output-dir /path/to/dota_out --split-file /path/to/export/split.csv
python tools/playground_to_dota.py --data-root /path/to/export --output-dir /path/to/dota_out --split-file /path/to/export/split.csv --val-split-id 2

# Label options
python tools/playground_to_dota.py --data-root /path/to/export --output-dir /path/to/dota_out --ignore-label Confuser --map-label "taxi=car"

# Dry run (report stats only)
python tools/playground_to_dota.py --data-root /path/to/export --output-dir /path/to/dota_out --dry-run
```

### `hrsc_to_dota.py`

Export official **HRSC2016** XML/BMP splits to DOTA-format PNG + `.txt` folders. Native training uses `dataset.format: hrsc2016` and does **not** require this step; use it when you want DOTA loaders or `odet tile-dota`.

**Usage:**
```bash
odet hrsc-to-dota --data-root /path/to/HRSC2016 --output-dir /path/to/HRSC2016-dota
python tools/hrsc_to_dota.py --data-root /path/to/HRSC2016 --output-dir /tmp/hrsc_dota --splits trainval,test
```

See [Data guide — HRSC2016](../docs/user-guide/data.md#hrsc2016).

### `tile_dota.py`

Tile large DOTA format images into smaller patches for training.

**Usage:**
```bash
# Basic tiling (creates 1024x1024 tiles with 200px overlap by default)
python tools/tile_dota.py /path/to/dota/train

# Custom tile size and overlap
python tools/tile_dota.py /path/to/dota/train \
    --tile-size 512 \
    --overlap 128

# Adjust minimum overlap ratio for keeping objects
python tools/tile_dota.py /path/to/dota/train \
    --min-overlap 0.5

# Overwrite existing tiles
python tools/tile_dota.py /path/to/dota/train --overwrite

# Legacy: last row/column may extend past the image (zero-padded tiles)
python tools/tile_dota.py /path/to/dota/train --pad-edge-tiles
```

**Features:**
- Tile large aerial/satellite images into manageable patches
- Process oriented bounding box annotations (DOTA format)
- Configurable tile size and overlap
- Minimum overlap ratio filtering (default **0.7**, aligned with MMRotate `iof_thr`; keeps objects with >= 70% area inside the tile)
- Automatic padding for images smaller than tile size
- By default, last row/column of tiles align on the image edge (no right/bottom zero-padding); use `--pad-edge-tiles` for the old stride-only edge behavior
- Computes minimum rotated rectangles for truncated objects
- Outputs official DOTA annotation format (comma-separated)

**Input Structure:**
```
data_dir/
  images/
    P0001.png
    P0002.png
    ...
  labels/
    P0001.txt
    P0002.txt
    ...
```

**Output Structure:**
```
data_dir/tiles_{size}/
  images/
    P0001_0_0.png
    P0001_960_0.png
    P0002_0_0.png
    ...
  labels/
    P0001_0_0.txt
    P0001_960_0.txt
    P0002_0_0.txt
    ...
```

**Note:** This tool is particularly useful when:
- Working with large DOTA v1.0 or v2.0 images (typically 800-20000 pixels)
- Training models that require fixed input sizes
- Need to control memory usage during training
- Want to apply sliding window inference

### `dota_labels_to_comma.py`

Convert DOTA label `.txt` files from space-separated to official comma-separated format in place.

**Usage:**
```bash
# Convert all .txt in a labels folder
python tools/dota_labels_to_comma.py /path/to/labels

# Dataset root (uses labels/ or labelTxt/ if present)
python tools/dota_labels_to_comma.py /path/to/dataset

# Dry run (no writes)
python tools/dota_labels_to_comma.py /path/to/labels --dry-run

# Backup originals as .txt.bak
python tools/dota_labels_to_comma.py /path/to/labels --backup
```

Accepts both space- and comma-separated input; writes official format. Keeps metadata lines (e.g. `imagesource:`, `gsd`) and empty lines unchanged.

## Requirements

All tools require the core dependencies:
```bash
uv pip install torch torchvision Pillow
```

For training and inference, ensure you have:
- PyTorch with CUDA support (optional, for GPU acceleration)
- Properly formatted DOTA dataset (for training)

## Customization

These tools are designed to be starting points for real-world use. You can customize them for:
- Different datasets (modify data loading)
- Custom augmentation (add transforms)
- Different model architectures (modify model creation)
- Custom evaluation metrics (add to training loop)

### `measure_sampled_riou_error.py`

Compare **sampling-based GPU rIoU** (`oriented_box_iou_gpu`) against **exact Shapely polygon IoU** on stratified synthetic pairs (tiny/small/medium/large squares, elongated ships, thin slivers). Reports absolute / relative error percentiles overall, by category, and by exact-IoU bin — use it to tune geometry defaults in `oriented_det/ops/gpu_ops.py` (`target_spacing_px`, `min_samples`, `max_samples`, `min_points_short`).

**Requires:** Shapely (core dependency).

```bash
# Default geometry (2 px spacing, 25…1024 samples), 3000 pairs
python tools/measure_sampled_riou_error.py

# More pairs + CSV export
python tools/measure_sampled_riou_error.py --pairs 5000 --seed 0 --csv /tmp/riou_error.csv

# Sweep target spacing to tune further
python tools/measure_sampled_riou_error.py --pairs 2000 --sweep-spacing 1.5 2 3 4

# Override caps (must stay perfect squares for min/max)
python tools/measure_sampled_riou_error.py --target-spacing 2 --min-samples 25 --max-samples 1024
```

**Output columns (CSV):** category, box sizes, exact/sampled IoU, errors, grid size, geometry params.

See [oriented_det/ops/README.md](../oriented_det/ops/README.md#geometry-based-riou-sampling--rationale--metrics) for design rationale and benchmark tables (spacing 2 px, min 25, max 1024).

## Tips

1. **Preview augmentations**: Run `preview_augmentation.py` on your config to verify flips, random rotate, and Albumentations before a long training run
2. **Test on small dataset**: Before full training, test on a subset of your data
3. **Monitor training**: Check checkpoint directory for saved models
4. **Adjust hyperparameters**: Learning rate, batch size, and thresholds may need tuning for your data


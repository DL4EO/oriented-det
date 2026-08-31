# Tools

This page provides links to practical scripts and tools for real-world use.

## Available Tools

After `uv pip install -e .`, use the **`odet`** CLI (`odet --help`). Script implementations live under `tools/`; see [tools/README.md](https://github.com/DL4EO/oriented-det/blob/main/tools/README.md) for full CLI reference.

### `odet` command reference

| Command | Role |
|---------|------|
| `train` | Config-based training (`tools/train.py`) |
| `train-multi-gpu` | `torchrun` wrapper for multi-GPU training |
| `preds` | Validation inference → `predictions/<timestamp>/` |
| `metrics` | Same as `preds` with diagnostics / mAP (alias) |
| `lr-finder` | Learning-rate sweep on a training config |
| `stats` | Dataset class / tile statistics |
| `tile-dota` | Tile large DOTA images into train patches |
| `image-demo` | Single-image or folder inference with sliding window |
| `viewer` | Gradio app for browsing predictions |
| `playground-csv` | Build Airbus Playground split CSV |
| `playground-to-dota` | Export Playground annotations to DOTA layout |
| `hrsc-to-dota` | Export HRSC2016 XML/BMP to DOTA PNG + labels |
| `export-onnx` | ONNX export (`export.scripts.export_onnx`) |
| `export-tf` | ONNX + Keras detect bundle (`export.scripts.export_tf`; needs `[export]`) |
| `export-detect` | Keras bundle from existing ONNX (`export.scripts.build_faster_rcnn_savedmodel`) |
| `export-preds` | Val inference via Keras bundle (`export.scripts.save_predictions_tf`) |
| `pretrained` | `list` / `download` Hub checkpoints (`hf://` slugs) |
| `labels-to-comma` | Convert DOTA label files to comma-separated format |
| `free-gpu` | Kill GPU processes (dev utility) |

Common commands:

### Training

**`tools/train.py`** - Complete config-based training pipeline:

```bash
odet train --config configs/oriented_rcnn/dota_le90_1x.json --batch-size 4
```

Features:
- Config-based training (JSON files)
- **Nested config inheritance** via `_base_` field
  - Factorize common settings into base configs (`configs/_base_/`)
  - Inherit from multiple base configs (datasets, schedules, models)
  - Override specific fields as needed
- Supports multiple model types (Rotated Faster R-CNN, Oriented R-CNN, Rotated RetinaNet, Rotated FCOS)
- Command-line parameter overrides
- Dataset loading
- Model initialization
- Training loop with checkpointing
- Evaluation

**Config inheritance example:**

```json
{
  "_base_": [
    "../_base_/models/oriented_rcnn_r50.json",
    "../_base_/schedules/1x.json"
  ],
  "checkpoint": {
    "load_from_checkpoint": null
  }
}
```

See the [Training Guide](../user-guide/training.md#nested-config-inheritance) for detailed documentation on config inheritance.

### Inference

For config + checkpoint inference on one or more images, prefer **`odet image-demo`** or **`tools/image_demo.py`** (reads experiment JSON; sliding window for DOTA `fixed`/`crop` when needed; **pad** is whole-image). Library API: `oriented_det.runtime.inference.run_inference_auto`.

The optional module CLI `python -m oriented_det.runtime.inference` supports legacy `--model-type oriented_rcnn|rotated_retinanet` only; production workflows use JSON configs and `odet image-demo`.

## Code Snippets

### Image Tiling

```python
from oriented_det.data import ImageTiler, visualize_tiles
from pathlib import Path

# Create tiler
tiler = ImageTiler(
    tile_size=1024,
    overlap=0.2,
    min_overlap_ratio=0.3
)

# Generate tiles
tiles = tiler.generate_tiles(image_width=4000, image_height=4000)

# Visualize
visualize_tiles(
    image_path=Path("large_image.png"),
    image_width=4000,
    image_height=4000,
    tiles=tiles,
    rboxes=annotations,
    output_path=Path("tiles_vis.png")
)
```

### Data Augmentation

```python
from oriented_det.data import apply_random_train_flips, apply_random_train_rotate
from oriented_det.geometry import RBox

rboxes = [RBox(100, 200, 50, 30, 0.5)]
image, rboxes = apply_random_train_flips(
    image, rboxes, image_width=512, image_height=512,
)
image, rboxes = apply_random_train_rotate(
    image, rboxes, image_width=512, image_height=512,
)
```

### Using RBoxes with the Model

```python
import torch
from oriented_det.geometry import RBox

# Your dataset provides RBoxes - use the targets format directly
targets = [{
    "rboxes": [
        RBox(100, 200, 50, 30, 0.5),
        RBox(300, 400, 80, 40, 0.2),
    ],
    "labels": torch.tensor([1, 2]),
}]

# Use with model (targets format expected by OrientedRCNN and RotatedRetinaNet)
model.train()
loss_dict = model([image], targets)
```

## Detailed Tool Documentation

### Gradio Visualization App (`app.py`)

The Gradio app provides an interactive web interface to browse and visualize inference results.

**Launching the app:**

```bash
# Auto-detect latest predictions/ under the repo root (run make preds first)
odet viewer
# or
python tools/app.py

# Published Hub eval (reports only — use predictions/ for viewer JSON)
make viewer VIEWER_PRED_DIR=predictions/20260627_082942 DOTA_DATA_ROOT=/path/to/DOTA-v1.0-tiled

# Local scratch run
python tools/app.py --predictions-dir predictions/20260602_120000 --data-root /path/to/DOTA
```

**Features:**
- **Image browsing**: Navigate through all images with predictions
- **Threshold adjustment**: Dynamically adjust confidence threshold to see how it affects detections
- **F1 score display**: Shows per-class F1 scores if analysis JSON is available
- **Sorting options**: Sort by F1 score, detection count, or original order
- **Zoom controls**: Zoom in/out on images for detailed inspection
- **Real-time filtering**: See how different thresholds affect detections instantly

**Command-line arguments:**
- `--predictions-dir`: Path to predictions directory (auto-detects latest if not specified)
- `--data-root`: Path to DOTA dataset root (for loading original images)
- `--port`: Port number for Gradio server (default: 7860)
- `--share`: Create public Gradio link

**Predictions JSON format:**

The app expects predictions in JSON format:
```json
{
  "metadata": {
    "experiment_dir": "runs/oriented_rcnn/20260616-030231",
    "checkpoint": "checkpoints/best.pth",
    "model_type": "oriented_rcnn"
  },
  "predictions": {
    "image_001.png": {
      "rboxes": [[cx, cy, w, h, angle], ...],
      "scores": [0.95, 0.87, ...],
      "labels": [1, 2, ...],
      "class_names": ["plane", "ship", ...]
    }
  }
}
```

**Use cases:**
- Visual inspection of model predictions
- Finding failure cases and edge cases
- Adjusting confidence thresholds interactively
- Comparing predictions across different models

### Save Predictions (`save_predictions.py` / `odet preds`)

Run validation inference and write `predictions/<timestamp>/predictions.json` at the **repository root** (not under `runs/`).

**Basic usage:**

```bash
make preds
make metrics

odet preds --experiment-dir runs/oriented_rcnn/<timestamp> --data-split val --no-diagnostics
odet preds --metrics-from-json predictions/<timestamp>
```

**Common flags** (`odet preds --help`):

- `--experiment-dir`, `--checkpoint`, `--config` — resolve model from a training run
- `--model-type` — `rotated_faster_rcnn`, `oriented_rcnn`, `rotated_retinanet`, or `rotated_fcos`
- `--data-root`, `--data-split` — dataset layout
- `--output-dir` — default: `predictions/<YYYYMMDD_HHMMSS>/`
- `--no-diagnostics` / `--metrics-from-json` — inference-only vs offline mAP

**Output structure:**

```
predictions/20260602_120000/
├── predictions.json
├── analysis_iou0.50.json
└── visualizations/   # with --save-visualizations
```

Thresholds and sliding-window overlap default from experiment **`production.*`** and **`evaluation.*`** unless overridden on the CLI. See [tools/README.md](https://github.com/DL4EO/oriented-det/blob/main/tools/README.md) for the full flag list.

### Tile DOTA (`tile_dota.py`)

Tile large DOTA format images into smaller patches with configurable tile size and overlap.

**Basic usage:**

```bash
# Tile images with default settings (1024x1024 tiles, 200px overlap; last row/column flush to image edge)
python tools/tile_dota.py /path/to/dota/train

# Custom tile size and overlap
python tools/tile_dota.py /path/to/dota/train --tile-size 512 --overlap 128

# Adjust minimum overlap ratio for keeping objects
python tools/tile_dota.py /path/to/dota/train --min-overlap 0.5

# Overwrite existing tiles
python tools/tile_dota.py /path/to/dota/train --overwrite

# Legacy: stride-only grid (last tiles may extend past the image with zero padding)
python tools/tile_dota.py /path/to/dota/train --pad-edge-tiles
```

**Input structure:** The script expects a directory containing `images/` and `labels/` subdirectories:

```
data_dir/
  images/
    image001.png
    image002.png
    ...
  labels/
    image001.txt
    image002.txt
    ...
```

**Command-line arguments:**
- `data_dir` (positional): Root directory containing `images/` and `labels/` subdirectories
- `--tile-size`: Tile width and height in pixels (default: 1024)
- `--overlap`: Overlap between adjacent tiles in **pixels** (default: 200)
- `--min-overlap`: Minimum overlap ratio (0.0–1.0) to keep objects that cross tile boundaries (default: **0.7**, MMRotate `iof_thr`)
- `--overwrite`: Overwrite existing tile files

**Output format:** Tiles are written to `data_dir/tiles_{size}/`:

```
data_dir/tiles_1024/
  images/
    image001_0_0.png
    image001_960_0.png
    ...
  labels/
    image001_0_0.txt
    image001_960_0.txt
    ...
```

**Tile naming convention:**
- Format: `{original_name}_{x_start}_{y_start}.png`
- Example: `P0001_0_0.png` (tile starting at x=0, y=0)

**Use cases:**
- Preparing large DOTA images for training (GPU memory constraints)
- Creating validation/test tiles from full-size images
- Data augmentation through tiling
- Processing very large satellite/aerial images

**Best practices:**
- Use `--overlap 128` or similar (e.g., ~10–20% of tile size) to avoid cutting objects at tile boundaries
- Default `--min-overlap 0.7` matches MMRotate; use a lower value (e.g. `0.3`) to keep more truncated objects
- Use `--tile-size 1024` for most models (balance between context and memory)
- Ensure tiles are large enough for your smallest objects

## Tutorials

More detailed tutorials are available in the [training](../examples/training.md) and [inference](../examples/inference.md) examples.


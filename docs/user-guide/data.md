# Data Module

The data module provides dataset loading, image tiling, data augmentation, and evaluation utilities.

## DOTA Dataset

### Label file format (official DOTA: comma-separated)

We **produce and use** the [official DOTA format](https://captain-whu.github.io/DOTA/dataset.html): **comma-separated** lines:

```text
x1, y1, x2, y2, x3, y3, x4, y4, category, difficult
```

- **Writing**: `format_dota_line()`, `DOTAAnnotation.to_line()`, `tools/tile_dota.py`, and `tools/playground_to_dota.py` all output this format.
- **Reading**: `DOTAAnnotation.from_line()` accepts **both** comma-separated (official) and space-separated (legacy). The dataset loader uses this too.


### Loading DOTA Dataset

The DOTA loader supports two modes for discovering annotation files:

#### Mode 1: Pattern Matching (Default)

```python
from oriented_det.data import DOTADataset

dataset = DOTADataset(
    root_dir="/path/to/dota",
    split="train",
    allowed_classes=["plane", "ship", "vehicle"],
    difficult_strategy="drop"
)
```

#### Mode 2: Split File (Official DOTA Convention)

```python
dataset = DOTADataset(
    root_dir="/path/to/dota",
    split="train",
    split_file="train.txt",  # Lists image names, one per line
    difficult_strategy="drop"
)
```

#### Mode 3: Separate Folders for Splits

If your dataset has train, val, and test in separate folders:

```
data_root/
├── train/
│   ├── labelTxt/
│   └── images/
├── val/
│   ├── labelTxt/
│   └── images/
└── test/
    ├── labelTxt/
    └── images/
```

You can specify custom `label_dir` and `image_dir` paths:

```python
# Training set
train_dataset = DOTADataset(
    root_dir="/path/to/data_root",  # Base directory (not used when label_dir/image_dir specified)
    split="train",
    label_dir="/path/to/data_root/train/labelTxt",
    image_dir="/path/to/data_root/train/images",
    difficult_strategy="drop"
)

# Validation set
val_dataset = DOTADataset(
    root_dir="/path/to/data_root",
    split="val",
    label_dir="/path/to/data_root/val/labelTxt",
    image_dir="/path/to/data_root/val/images",
    difficult_strategy="drop"
)

# Or using build_dota_loader
from oriented_det.data import build_dota_loader

train_loader = build_dota_loader(
    root_dir="/path/to/data_root",
    split="train",
    label_dir="/path/to/data_root/train/labelTxt",
    image_dir="/path/to/data_root/train/images",
    batch_size=4,
    shuffle=True
)
```

#### Mode 4: Images and annotations in the same folder

You can keep images (`.jpg`/`.png`) and DOTA annotation files (`.txt`) in the **same directory**. Pass that directory as both `label_dir` and `image_dir`:

```python
# Single folder per split: images and .txt labels together
train_dataset = DOTADataset(
    root_dir="/path/to/train",
    split="train",
    label_dir="/path/to/train",
    image_dir="/path/to/train",
    difficult_strategy="drop"
)
```

For **config-based training**, set `dataset.same_folder: true` in your config so that `train_tiles_dir` and `val_tiles_dir` are used as the single folder for each split (no `images/` or `labels/` subdirectories):

```json
{
  "dataset": {
    "data_root": "/path/to/dota",
    "train_tiles_dir": "/path/to/dota/train",
    "val_tiles_dir": "/path/to/dota/val",
    "same_folder": true,
    "difficult_strategy": "drop"
  }
}
```

Annotation files must match image base names (e.g. `P0001.png` with `P0001.txt`, or `P0001_train.txt` with `P0001_train.png` when using split-suffix pattern matching).

**Detailed Loading Examples:**

**Example 1: Standard DOTA structure with pattern matching**

```python
from oriented_det.data import DOTADataset

# Standard DOTA structure:
# /path/to/dota/
#   ├── train/
#   │   ├── images/
#   │   └── labelTxt/
#   ├── val/
#   │   ├── images/
#   │   └── labelTxt/
#   └── test/
#       ├── images/
#       └── labelTxt/

dataset = DOTADataset(
    root_dir="/path/to/dota",
    split="train",
    difficult_strategy="drop",
    allowed_classes=["plane", "ship", "small-vehicle"]  # Optional: filter classes
)
```

**Example 2: Using split file (official DOTA convention)**

```python
# DOTA structure with split files:
# /path/to/dota/
#   ├── train.txt  # Lists image names: P0001.png, P0002.png, ...
#   ├── val.txt
#   ├── images/
#   └── labelTxt/

dataset = DOTADataset(
    root_dir="/path/to/dota",
    split="train",
    split_file="train.txt",  # Explicit split file
    difficult_strategy="drop"
)
```

**Example 3: Custom directory structure**

```python
# Custom structure with separate folders:
# /mnt/data/
#   ├── dota_train/
#   │   ├── images/
#   │   └── labels/
#   └── dota_val/
#       ├── images/
#       └── labels/

train_dataset = DOTADataset(
    root_dir="/mnt/data",  # Not used when label_dir/image_dir specified
    split="train",
    label_dir="/mnt/data/dota_train/labels",
    image_dir="/mnt/data/dota_train/images",
    difficult_strategy="drop"
)
```

**Edge Case Handling:**

- **Missing annotation files**: Dataset skips images without corresponding annotation files
- **Malformed annotations**: Lines that can't be parsed are skipped with a warning
- **Empty tiles**: By default, tiles with no ground-truth objects are included (empty target list). Set **`dataset.filter_empty_gt: true`** in the training config to drop them at dataset init (after `difficult_strategy`, `allowed_classes`, and `ignore_labels`), matching MMRotate `DOTADataset` (`filter_empty_gt=True`). The DOTA pretrain recipes [`dota_le90_1x.json`](https://github.com/DL4EO/oriented-det/blob/main/configs/oriented_rcnn/dota_le90_1x.json) and [`dota_le90_3x.json`](https://github.com/DL4EO/oriented-det/blob/main/configs/oriented_rcnn/dota_le90_3x.json) enable this. To keep empty tiles that the model still false-positives on, leave `filter_empty_gt` false and set **`dataset.drop_easy_empty_tiles: true`** with **`dataset.tile_metrics_csv`** from `make train-preds`: training then drops only vacuous tiles (`tp=fp=fn=0`) and can oversample hard empties.
- **Difficult objects**: Use `dataset.difficult_strategy`:
  - `drop`: remove difficult objects at read-time (never reach training/eval targets)
  - `ignore`: keep difficult objects but treat them as “don’t care” (MMRotate/MMDet style)
  - `keep`: treat difficult objects as normal GT

**Performance Considerations:**

- **Mode 1 (Pattern matching)**: Fastest, good for standard DOTA structure
- **Mode 2 (Split file)**: Slightly slower due to file reading, but more explicit
- **Mode 3 (Separate folders)**: Most flexible, similar performance to Mode 1

**Best Practices:**

1. For MMRotate parity, use `difficult_strategy="ignore"` (difficult objects should not contribute to loss)
2. Filter classes early with `allowed_classes` if you only need specific classes
3. Use `split_file` for explicit control over train/val/test splits
4. Set `label_dir` and `image_dir` for non-standard directory structures

### DOTA Polygon Format

DOTA uses 8 coordinates to define quadrilaterals:

```
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult
```

**Important Notes:**
- Corners are ordered sequentially around the polygon perimeter
- The loader converts polygons to `QBox` (which normalizes point order) and then to `RBox`
- `QBox` ensures counter-clockwise orientation and orders points starting from top-most

The loader automatically:
1. Parses polygons from annotation files
2. Converts to QBox (normalizes point order, ensures counter-clockwise)
3. Converts to RBox (computes center, dimensions, angle)

### Using with PyTorch DataLoader

```python
from oriented_det.data import build_dota_loader

loader = build_dota_loader(
    root_dir="/path/to/dota",
    split="train",
    batch_size=4,
    shuffle=True,
    num_workers=4
)
```

### Filtering

```python
# Filter by class
sample = sample.filter_by_class(allowed_classes=["plane", "ship"])

# Drop difficult annotations on an already-loaded sample
sample = sample.filter_by_class(drop_difficult=True)
```

## Airbus Playground

CSV + split-file datasets (`dataset.format: airbus_playground` in [Configuration](configuration.md)):

- `annotations_file` — object CSV from Playground export
- `split_file` — fold or train/val column (`val_split_id` for integer folds)
- `train_includes_val` — when `true`, train on all folds; `val_split_id` fold is still used for validation/monitoring only (DOTA `train_tiles_dirs` trainval parity)
- `ignore_labels`, `map_labels` — filter and rename classes
- **Lookalike confusers** (hard negatives, not a semantic class) — see [Lookalike confusers](#lookalike-confusers) below

Keep dataset JSON in your own config tree and inherit `@odet:configs/_base_/...` fragments. Prep tools: `odet playground-csv`, `odet playground-to-dota` (see [tools/README.md](https://github.com/DL4EO/oriented-det/blob/main/tools/README.md)).

## Lookalike confusers

`lookalike` is a **reserved class name**. It never enters `class_map` / `num_classes`. Boxes mapped onto it are trained as hard negatives: overlapping non-positive RPN/ROI/RetinaNet/FCOS samples are forced to **background** (not ignore), and two-stage heads preferentially sample those negatives.

Recipe (Airbus Playground CSV or any loader that applies `map_labels`):

```json
"dataset": {
  "map_labels": { "Confuser": "lookalike" },
  "ignore_labels": [],
  "lookalike_labels": null
}
```

Notes:

- CSV / DOTA labels must still contain the boxes. If they were dropped with `--ignore-label Confuser` / `ignore_labels`, regenerate or clear that filter.
- Optional `dataset.lookalike_labels` adds extra aliases treated the same way; `"lookalike"` is always included. Example without renaming: `"lookalike_labels": ["Confuser"]`.
- Lookalike wins over `ignore_labels` and is kept even when not in `allowed_classes`. Lookalike-only tiles are **not** dropped by `filter_empty_gt`.
- At eval, lookalikes are not positives; a real-class detection on that region counts as an FP (desired). Do not put them on `rboxes_ignore` at eval.

## HRSC2016

Native XML loader (`dataset.format: hrsc2016`).

### Download

The original paper site (`escience.cn`) is often offline. Use one of these copies of the **official 2016 release** (1,061 images: 436 train / 181 val / 444 test):

| Source | Size | Notes |
|--------|------|--------|
| [IEEE DataPort](https://ieee-dataport.org/documents/hrsc2016) | ~3.5 GB (`HRSC2016_dataset.zip`) | Free IEEE account; closest hosted archive |
| [Baidu AI Studio](https://aistudio.baidu.com/datasetdetail/54106) | — | Link used by [MMRotate](https://github.com/open-mmlab/mmrotate/blob/main/tools/data/hrsc/README.md); needs a Baidu account |
| [Kaggle `guofeng/hrsc2016`](https://www.kaggle.com/datasets/guofeng/hrsc2016) | — | Same layout; `kaggle datasets download -d guofeng/hrsc2016` |

Do **not** use **HRSC2016-MS** (a later multi-scale variant). After unzip, point `dataset.data_root` at the folder that contains `FullDataSet/` and `ImageSets/` (a wrapping `HRSC2016/` directory is also accepted).

Paper: Liu, Yuan, Weng, Yang, *A High Resolution Optical Satellite Image Dataset for Ship Recognition and Some New Baselines*, ICPRAM 2017. [DOI](https://doi.org/10.5220/0006120603240331).

Official layout:

```text
HRSC2016/
  FullDataSet/AllImages/*.bmp
  FullDataSet/Annotations/*.xml
  ImageSets/{train,val,test,trainval}.txt
```

- Single class: **`ship`** (fine-grained `Class_ID` values are ignored).
- XML `mbox_cx/cy/w/h/ang` uses **radians**; boxes are converted through the same polygon → RBox path as DOTA (**le90**).
- Default ImageSets mapping (MMRotate): train → **`trainval`**, val → **`test`**. Override with `dataset.train_split` / `dataset.val_split`.
- Oriented R-CNN, Faster R-CNN, and FCOS 1×/3× use **`keep_ratio`** (long edge 800) + `pad_size_divisor` 32. FCOS HRSC 1×/3× and Oriented R-CNN / Faster R-CNN 3× enable random rotate at p=0.5 **±20°**. Two-stage 1× recipes leave rotate off. Oriented R-CNN uses Smooth L1 + ProbIoU aux; Faster R-CNN keeps ProbIoU main + Smooth L1 aux. `make eval-val` / `odet preds` use the same whole-image path for `pad` / `keep_ratio` (no native sliding windows). DOTA eval-val stays on `fixed` pre-tiled rasters.
- HRSC / DOTA NMS split: train `model` and eval-val `evaluation.final_nms_iou_threshold` **0.1** (MMRotate test parity); deploy `production.final_nms_iou_threshold` **0.3**. Two-stage HRSC keeps max **2000** dets/image and score **0.05**.
- Optional DOTA export (for `odet tile-dota`): `odet hrsc-to-dota --data-root /path/to/HRSC2016 --output-dir /path/to/HRSC2016-dota`.

```json
{
  "dataset": {
    "format": "hrsc2016",
    "data_root": "/path/to/data/HRSC2016",
    "train_split": "trainval",
    "val_split": "test"
  }
}
```

Recipes: [`configs/oriented_rcnn/hrsc2016_le90_1x.json`](https://github.com/DL4EO/oriented-det/blob/main/configs/oriented_rcnn/hrsc2016_le90_1x.json), [`configs/oriented_rcnn/hrsc2016_le90_3x.json`](https://github.com/DL4EO/oriented-det/blob/main/configs/oriented_rcnn/hrsc2016_le90_3x.json) (36 epochs, milestones 24/33, ±20° rotate), [`configs/rotated_faster_rcnn/hrsc2016_le90_1x.json`](https://github.com/DL4EO/oriented-det/blob/main/configs/rotated_faster_rcnn/hrsc2016_le90_1x.json), [`configs/rotated_faster_rcnn/hrsc2016_le90_3x.json`](https://github.com/DL4EO/oriented-det/blob/main/configs/rotated_faster_rcnn/hrsc2016_le90_3x.json), [`configs/rotated_fcos/hrsc2016_le90_1x.json`](https://github.com/DL4EO/oriented-det/blob/main/configs/rotated_fcos/hrsc2016_le90_1x.json), [`configs/rotated_fcos/hrsc2016_le90_3x.json`](https://github.com/DL4EO/oriented-det/blob/main/configs/rotated_fcos/hrsc2016_le90_3x.json).

## Image Tiling

Split large images into overlapping patches:

```python
from oriented_det.data import ImageTiler

# Create tiler
tiler = ImageTiler(
    tile_size=1024,
    overlap=0.2,  # 20% overlap
    min_box_area=64,  # Filter small boxes
    min_overlap_ratio=0.3,  # Keep box if >= 30% overlaps tile
    edge_handling="clip"  # "clip", "ignore", or "keep"
)

# Generate tiles
tiles = tiler.generate_tiles(image_width=4000, image_height=4000)

# Process each tile
for tiled_sample in tiler.tile_image(
    image_path=Path("large_image.png"),
    image_width=4000,
    image_height=4000,
    rboxes=annotations,
    class_names=classes
):
    # Process tiled_sample
    process_tile(tiled_sample)
```

### Visualizing Tiles

```python
from oriented_det.data import visualize_tiles

visualize_tiles(
    image_path=Path("large_image.png"),
    image_width=4000,
    image_height=4000,
    tiles=tiles,
    rboxes=annotations,
    class_names=classes,
    output_path=Path("tiles_vis.png")
)
```

## Data Augmentation

Training collate applies geometric augs after spatial resize: random flips (`preprocessing.enable_flip_*`, MMRotate `RRandomFlip`) then optional random rotate (`enable_random_rotate`, `random_rotate_prob`, `random_rotate_angle_range` in degrees — MMRotate `PolyRandomRotate`, `auto_bound=False`). Val and inference do not flip or rotate. FCOS HRSC 1×/3× and Oriented R-CNN / Faster R-CNN HRSC 3× use p=0.5 ±20°; two-stage 1× and DOTA leave rotate off.

### 1. Geometric Transforms (Oriented Bounding Box Aware)

Image+box helpers in `oriented_det.data` (`flips.py` / `rotates.py`). Boxes are re-normalized to le90. `geometry.transforms.rotate` is rbox-only math (y-up); do not use it on PIL images without negating the angle.

```python
import math
from oriented_det.data import (
    apply_flip_to_image,
    apply_flip_to_rboxes,
    apply_random_train_flips,
    apply_random_train_rotate,
    apply_rotate_to_image,
    apply_rotate_to_rboxes,
)

# Same path as training collate
image, rboxes = apply_random_train_flips(
    image, rboxes, image_width=512, image_height=512,
    enable_horizontal=True, enable_vertical=True, enable_diagonal=True,
)
image, rboxes = apply_random_train_rotate(
    image, rboxes, image_width=512, image_height=512,
    prob=0.5, angle_range_deg=180.0,
)

# Fixed flip / rotate (PIL visual CCW)
image = apply_flip_to_image(image, "horizontal")
rboxes = apply_flip_to_rboxes(rboxes, "horizontal", image_width=512, image_height=512)
image = apply_rotate_to_image(image, 90.0)
rboxes = apply_rotate_to_rboxes(rboxes, math.radians(90.0), image_width=512, image_height=512)
```

### 2. Albumentations (Non-Geometric Only)

For additional augmentation options, you can use albumentations with non-geometric transforms. **Note:** Only non-geometric augmentations (color, contrast, blur, noise, etc.) are supported because albumentations does not support oriented bounding boxes.

```python
from oriented_det.data import create_albumentations_augmentation

# Create default augmentation pipeline
aug = create_albumentations_augmentation(
    brightness_limit=0.2,
    contrast_limit=0.2,
    gamma_limit=(80, 120),
    gauss_noise_var_limit=(10.0, 50.0),
    blur_limit=3,
    clahe_clip_limit=4.0,
    p_brightness_contrast=0.5,
    p_gamma=0.3,
    p_noise=0.2,
    p_blur=0.2,
    p_clahe=0.3,
)

# Apply to PIL Image
augmented_image = aug(image)  # Returns PIL Image
```

**Supported Non-Geometric Augmentations:**
- **Color/Contrast**: RandomBrightnessContrast, RandomGamma, CLAHE
- **Noise**: GaussNoise
- **Blur**: GaussianBlur

**Not Supported (Geometric):**
- Rotation, Scaling, Translation, Affine transforms (use `apply_rotate_to_*` / `apply_random_train_rotate`)
- Any transform that would require updating oriented bounding box coordinates

You can also create custom albumentations pipelines:

```python
import albumentations as A
from oriented_det.data import AlbumentationsTransform

# Create custom non-geometric augmentation
custom_aug = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.3),
    A.GaussNoise(p=0.2),
    A.GaussianBlur(p=0.2),
])

transform = AlbumentationsTransform(custom_aug)
augmented_image = transform(image)
```

## Evaluation

### Oriented mAP

Compute mean Average Precision (mAP) for oriented detection using `compute_oriented_map()`. This function implements the DOTA evaluation protocol for oriented bounding boxes.

**Basic usage:**

```python
from oriented_det.data import Detection, GroundTruth, compute_oriented_map
from oriented_det.geometry import RBox

# Prepare detections (from model predictions)
detections = {
    "img1": [
        Detection(
            rbox=RBox(100, 200, 50, 30, 0.5),
            score=0.9,
            class_id=0,
            class_name="plane"
        ),
        Detection(
            rbox=RBox(300, 400, 80, 40, 0.2),
            score=0.85,
            class_id=1,
            class_name="ship"
        ),
    ],
    "img2": [
        Detection(
            rbox=RBox(150, 250, 60, 35, 0.1),
            score=0.75,
            class_id=0,
            class_name="plane"
        ),
    ],
}

# Prepare ground truths (from dataset)
ground_truths = {
    "img1": [
        GroundTruth(
            rbox=RBox(100, 200, 50, 30, 0.5),
            class_id=0,
            class_name="plane",
            difficult=0
        ),
        GroundTruth(
            rbox=RBox(310, 410, 75, 45, 0.25),
            class_id=1,
            class_name="ship",
            difficult=0
        ),
    ],
    "img2": [
        GroundTruth(
            rbox=RBox(150, 250, 60, 35, 0.1),
            class_id=0,
            class_name="plane",
            difficult=0
        ),
    ],
}

# Compute mAP
mean_ap, class_aps = compute_oriented_map(
    detections,
    ground_truths,
    iou_threshold=0.5
)

print(f"mAP: {mean_ap:.4f}")
print("Per-class AP:")
for class_name, ap in class_aps.items():
    print(f"  {class_name}: {ap:.4f}")
```

**Advanced usage:**

```python
import torch
from oriented_det.data import compute_oriented_map

# Use GPU acceleration for large evaluations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean_ap, class_aps = compute_oriented_map(
    detections,
    ground_truths,
    iou_threshold=0.5,  # IoU threshold for positive matches
    class_names=["plane", "ship", "small-vehicle"],  # Evaluate specific classes
    show_progress=True,  # Show progress bars
    max_iou_calculations_per_class=10_000_000,  # Optimization threshold
    device=device,  # GPU acceleration
)
```

**Parameters:**

- `detections`: Dictionary mapping `image_id -> List[Detection]`
  - Each `Detection` has: `rbox` (RBox), `score` (float), `class_id` (int), `class_name` (str)
- `ground_truths`: Dictionary mapping `image_id -> List[GroundTruth]`
  - Each `GroundTruth` has: `rbox` (RBox), `class_id` (int), `class_name` (str), `difficult` (int)
- `iou_threshold`: IoU threshold for positive matches (default: 0.5)
- `class_names`: Optional list of class names to evaluate (default: all classes)
- `show_progress`: Whether to show progress bars (default: True)
- `max_iou_calculations_per_class`: Maximum IoU calculations before using optimization (default: 10M)
- `device`: Optional torch device for GPU acceleration

**Returns:**

- `mean_ap`: Mean Average Precision across all classes (float)
- `class_aps`: Dictionary mapping `class_name -> AP` for each class

**Integration with model evaluation:**

```python
from oriented_det.data import Detection, GroundTruth, compute_oriented_map
from oriented_det.geometry import RBox

# After running inference
model.eval()
detections = {}
ground_truths = {}

for image_id, (image, target) in enumerate(val_loader):
    with torch.no_grad():
        outputs = model([image])
    
    # Convert model outputs to Detection objects
    output = outputs[0]
    detections[image_id] = [
        Detection(
            rbox=rbox,
            score=float(score),
            class_id=int(label),
            class_name=class_names[int(label) - 1]  # 1-indexed labels
        )
        for rbox, score, label in zip(
            output["rboxes"],
            output["scores"],
            output["labels"]
        )
    ]
    
    # Convert ground truth to GroundTruth objects
    ground_truths[image_id] = [
        GroundTruth(
            rbox=rbox,
            class_id=int(label),
            class_name=class_names[int(label) - 1],
            difficult=0
        )
        for rbox, label in zip(target["rboxes"], target["labels"])
    ]

# Compute mAP
mean_ap, class_aps = compute_oriented_map(
    detections,
    ground_truths,
    iou_threshold=0.5
)
```

**Performance Notes:**

- For large evaluations (>10M IoU calculations per class), the function automatically uses batch IoU computation
- GPU acceleration is available when `device` is set to a CUDA device
- Progress bars show computation status for each class
- Difficult objects are included in evaluation (set `difficult=1` in GroundTruth to mark them)

**DOTA Protocol Compatibility:**

- Uses oriented IoU for matching (not axis-aligned IoU)
- Follows DOTA evaluation protocol
- Supports multiple IoU thresholds (call multiple times with different thresholds)
- Handles class-agnostic evaluation (set `class_names=None`)

## See Also

- [API Reference](../api/data.md) - Complete API documentation
- [Models Guide](models.md) - Using data with models


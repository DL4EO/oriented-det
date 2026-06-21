# Data API Reference

::: oriented_det.data
    options:
      show_root_heading: true
      show_root_toc_entry: true
      show_source: true

## Important Notes

### DOTA Dataset Loading Modes

The DOTA loader supports three modes for organizing your dataset:

1. **Pattern matching** (default): Matches annotation files by split pattern (e.g., `*_train.txt`)
2. **Split file** (official DOTA convention): Uses a file listing image names (e.g., `train.txt`)
3. **Separate folders**: Train/val/test in different directories

### DOTA Polygon Format

- DOTA uses 8 coordinates: `x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult`
- Corners are ordered sequentially around the polygon perimeter
- The loader converts polygons to `QBox` (which normalizes point order) and then to `RBox`
- `QBox` ensures counter-clockwise orientation and orders points starting from top-most

### Data Augmentation

OrientedDet supports two types of data augmentation:

1. **Geometric Transforms** (Oriented Bounding Box Aware): `HorizontalFlip`, `VerticalFlip`, `Rotate`, `Compose`
   - These transforms modify both the image and oriented bounding boxes
   - Angle information is preserved correctly

2. **Albumentations** (Non-Geometric Only): `create_albumentations_augmentation`, `AlbumentationsTransform`
   - Only non-geometric augmentations are supported (color, contrast, blur, noise, etc.)
   - **Note:** Albumentations does not support oriented bounding boxes, so only non-geometric transforms can be used

## Examples

### DOTA Dataset Loading

```python
from oriented_det.data import DOTADataset, build_dota_loader

# Mode 1: Pattern matching (backward compatible)
dataset = DOTADataset(
    root_dir="/path/to/dota",
    split="train",
    allowed_classes=["plane", "ship", "vehicle"],
    difficult_strategy="drop"
)

# Mode 2: Using split file (official DOTA convention)
dataset = DOTADataset(
    root_dir="/path/to/dota",
    split="train",
    split_file="train.txt",  # Lists image names, one per line
    allowed_classes=["plane", "ship", "vehicle"],
    difficult_strategy="drop"
)

# Mode 3: Separate folders for each split
train_dataset = DOTADataset(
    root_dir="/path/to/data_root",
    split="train",
    label_dir="/path/to/data_root/train/labelTxt",
    image_dir="/path/to/data_root/train/images",
    difficult_strategy="drop"
)

# Or use PyTorch DataLoader
loader = build_dota_loader(
    root_dir="/path/to/dota",
    split="train",
    split_file="train.txt",  # Optional
    batch_size=4,
    shuffle=True
)
```

### Image Tiling

```python
from oriented_det.data import ImageTiler, visualize_tiles
from pathlib import Path

# Configure tiler with overlap and filtering options
tiler = ImageTiler(
    tile_size=1024,
    overlap=0.2,  # 20% overlap between tiles
    min_box_area=64,  # Filter boxes < 64 pixels²
    min_overlap_ratio=0.3,  # Keep box only if >= 30% overlaps tile
    edge_handling="clip"  # "clip", "ignore", or "keep"
)

# Generate tiles and process
for tiled_sample in tiler.tile_image(
    image_path=Path("large_image.png"),
    image_width=4000,
    image_height=4000,
    rboxes=annotations,
    class_names=classes
):
    # Process each tile
    process_tile(tiled_sample)

# Visualize tiles for debugging
tiles = tiler.generate_tiles(4000, 4000)
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

### Data Augmentation

```python
from oriented_det.data import HorizontalFlip, Rotate, Compose, create_albumentations_augmentation

# Geometric transforms (oriented bounding box aware)
aug = Compose([
    HorizontalFlip(p=0.5),
    Rotate(degrees=90, p=0.3),
])

augmented_image, augmented_boxes = aug(image, rboxes, image_width, image_height)

# Albumentations (non-geometric only)
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

augmented_image = aug(image)  # Returns PIL Image
```

### Oriented mAP Evaluation

```python
from oriented_det.data import Detection, GroundTruth, compute_oriented_map
from oriented_det.geometry import RBox

detections = {
    "img1": [Detection(rbox=..., score=0.9, class_id=0, class_name="plane")],
}
ground_truths = {
    "img1": [GroundTruth(rbox=..., class_id=0, class_name="plane")],
}

mean_ap, class_aps = compute_oriented_map(
    detections, ground_truths, iou_threshold=0.5
)
```


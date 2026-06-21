# Quick Start

This guide will get you up and running with OrientedDet in minutes.

## Basic Geometry Operations

Start with the core geometry primitives:

```python
from oriented_det.geometry import RBox, Polygon, QBox, transforms
from math import radians

# Create a rotated bounding box
rbox = RBox(cx=100, cy=200, width=50, height=30, angle=radians(45))

# Convert to different formats
qbox = transforms.rbox_to_qbox(rbox)
polygon = transforms.rbox_to_polygon(rbox)

# Get properties
print(f"Area: {rbox.area}")
print(f"Aspect ratio: {rbox.aspect_ratio}")
print(f"Corners: {rbox.corners()}")
```

## IoU and NMS

Compute overlap and perform non-maximum suppression:

```python
from oriented_det.geometry import RBox
from oriented_det.ops import iou, nms

boxes = [
    RBox(0, 0, 10, 10, 0),
    RBox(5, 5, 10, 10, 0),
    RBox(100, 100, 10, 10, 0),
]
scores = [0.9, 0.8, 0.6]

overlap = iou.rbox_iou(boxes[0], boxes[1])
keep = nms.oriented_nms(boxes, scores, iou_threshold=0.5)
```

## Loading DOTA Dataset

```python
from oriented_det.data import DOTADataset, build_dota_loader

dataset = DOTADataset(
    root_dir="/path/to/dota",
    split="train",
    difficult_strategy="drop",
)

loader = build_dota_loader(
    root_dir="/path/to/dota",
    split="train",
    batch_size=4,
    shuffle=True,
)
```

## Train with the CLI (recommended)

From the repository root after `uv pip install -e .` (see [Installation](installation.md)):

```bash
odet train --config configs/rotated_faster_rcnn/dota_le90_3x.json
```

Or use `make train` (same default config). Set paths in `configs/_base_/datasets/dota_le90.json` for your DOTA layout. See [Configuration](../user-guide/configuration.md) for every JSON field.

## Programmatic training (optional)

For custom loops, you can call the training engine directly. Production workflows use JSON configs and `odet train`:

```python
import torch
from oriented_det import OrientedRCNN
from oriented_det.data import build_dota_loader
from oriented_det.train import train, CheckpointManager
from torch.optim import AdamW

model = OrientedRCNN(backbone_name="resnet50", num_classes=15)
train_loader = build_dota_loader("/path/to/dota", split="train", batch_size=4)
optimizer = AdamW(model.parameters(), lr=1e-4)
checkpoint_manager = CheckpointManager("checkpoints/", best_metric="mAP")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=12,
    checkpoint_manager=checkpoint_manager,
    use_amp=True,
)
```

## Inference and evaluation

After training, run validation predictions and offline metrics:

```bash
make preds      # writes predictions/<timestamp>/predictions.json
make metrics    # mAP/PR from that JSON
make viewer     # Gradio browser (optional)
```

For a single demo image:

```bash
odet image-demo demo/ runs/<model>/<timestamp>/config.json runs/.../checkpoints/checkpoint_best.pth --out-dir demo/out
```

Large images are handled with **sliding-window** inference inside `save_predictions.py` when they exceed the model canvas (`production.overlap_pixels` in config).

## Next Steps

- [Configuration](../user-guide/configuration.md) — JSON recipes and `_base_` inheritance
- [Training](../user-guide/training.md) — schedulers, checkpoints, AMP
- [Tools](tools.md) — `odet` commands and Makefile targets
- [API Reference](../api/geometry.md)

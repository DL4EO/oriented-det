# Models Module

The models module provides **true oriented object detection** models that predict rotated bounding boxes with rotation angles.

## Overview

OrientedDet provides four baseline detectors (see [Configuration](configuration.md#model_type)):

- **`OrientedRCNN`** — Two-stage ([Xie et al., ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Xie_Oriented_R-CNN_for_Object_Detection_ICCV_2021_paper.html)): horizontal RPN with midpoint-offset (6-param) proposals + oriented RoIAlign + oriented ROI head. Reference: MMRotate `oriented_rcnn_r50_fpn_1x_dota_le90`.
- **`RotatedFasterRCNN`** — Two-stage (Ren et al., NeurIPS 2015 baseline via [MMRotate](https://github.com/open-mmlab/mmrotate), Zhou et al., ACM MM 2022): horizontal RPN + horizontal RoIAlign + rotated ROI head. Reference: `rotated_faster_rcnn_r50_fpn_1x_dota_le90`.
- **`RotatedRetinaNet`** — Single-stage ([Lin et al., ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf) focal loss): separate cls/reg subnets, oriented anchors. Reference: `rotated_retinanet_obb_r50_fpn_1x_dota_le90`.
- **`RotatedFCOS`** — Anchor-free single-stage: distance-angle coder, centerness, L1 / KFIoU / decoded rIoU box loss, optional KFIoU aux. Reference: `rotated_fcos_r50_fpn_1x_dota_le90`.

All detectors:

- Predict oriented boxes with five parameters (cx, cy, w, h, angle)
- Preserve angle information through training and inference
- Use oriented IoU for matching and oriented NMS for post-processing

Load **OrientedDet checkpoints** from `pretrained/` or Hugging Face Hub (`odet pretrained download <slug>`). Recommended DOTA slugs: `oriented_rcnn_dota_le90_3x` (79.40% eval-val mAP50), `rotated_faster_rcnn_dota_le90_3x` (83.46%), `rotated_fcos_dota_le90_3x` (82.32%), and `rotated_retinanet_dota_le90_3x` (71.52%). HRSC2016: `oriented_rcnn_hrsc2016_le90_3x` (90.41%), `rotated_faster_rcnn_hrsc2016_le90_3x` (88.77%), `rotated_fcos_hrsc2016_le90_3x` (88.34%). See [pretrained/README.md](https://github.com/DL4EO/oriented-det/blob/main/pretrained/README.md).

## Training vs inference paths (two-stage models)

`RotatedFasterRCNN` and `OrientedRCNN` use **different code paths** in training versus eval, deploy, and ONNX export:

| Mode | Code path | Notes |
|------|-----------|--------|
| **Training** (`model.train()`) | Model `forward` → eager `horizontal_roi_align` (per-FPN loop) | Never calls `faster_rcnn_inference.py`; `torch.onnx.is_in_onnx_export()` is false |
| **Eval / deploy** | `faster_rcnn_inference.faster_rcnn_inference()` | Shared by PyTorch inference and `odet preds` |

`RotatedRetinaNet` is single-stage: training and inference both go through the same head + decode + rotated NMS path.

More detail (RoIAlign ONNX masking, regression tests): [oriented_det/models/README.md](https://github.com/DL4EO/oriented-det/blob/main/oriented_det/models/README.md) on GitHub.

## Important Notes

**True oriented detection** — Output RBoxes include the predicted angle (not `angle=0`). Angle information is preserved throughout the pipeline.

## Input Formats

Models accept targets in two formats:

### RBox Format (Recommended for Oriented Detection)

```python
from oriented_det.geometry import RBox

targets = [{
    "rboxes": [
        RBox(cx=100, cy=200, width=50, height=30, angle=0.5),
        RBox(cx=300, cy=400, width=80, height=40, angle=0.2),
    ],
    "labels": torch.tensor([1, 2]),
}]
```

### Standard Format (Also Supported)

```python
targets = [{
    "boxes": torch.tensor([[x1, y1, x2, y2], ...]),  # Converted to oriented (angle=0)
    "labels": torch.tensor([1, 2, ...]),
}]
```

## Output Format

Models return predictions with oriented boxes:

```python
outputs = [{
    "rboxes": [RBox(...), ...],  # Oriented boxes with predicted angles
    "boxes": torch.tensor([[cx, cy, w, h, angle], ...]),  # [N, 5] tensor
    "labels": torch.tensor([...]),  # Class labels (1-indexed)
    "scores": torch.tensor([...]),  # Confidence scores
}]
```

**✅ OrientedRCNN predicts rotation angles** - Output RBoxes include the predicted angle, not just `angle=0`.

## OrientedRCNN

**OrientedRCNN** is a two-stage detector with a **horizontal RPN** and **midpoint-offset** oriented proposals (not the same RPN as Rotated Faster R-CNN):

- Predicts oriented boxes with five parameters (cx, cy, w, h, angle)
- Oriented ROI align and class-agnostic ROI regression
- Config: `model_type: oriented_rcnn` — see [Configuration](configuration.md#model_type)
- ✅ Preserves angle information throughout training and inference
- ✅ Uses oriented IoU for matching and oriented NMS for post-processing

### Initialization

```python
from oriented_det import OrientedRCNN

# Typical DOTA configuration
model = OrientedRCNN(
    num_classes=15,
    backbone_name="resnet50",
    anchor_scales=[8],
    anchor_ratios=[0.5, 1.0, 2.0],
    target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
    target_stds=(0.1, 0.1, 0.2, 0.2, 0.1),
)

# Expanded configuration (for better coverage)
model = OrientedRCNN(
    num_classes=15,
    anchor_scales=[8],  # Single scale is sufficient (don't use multiple scales!)
    anchor_ratios=[0.125, 0.5, 1.0, 2.0, 8.0],  # Expanded ratios for extreme aspect ratios
)
```

**Note:** Multiple scales (e.g., `[8, 16, 32]`) are **redundant** with FPN strides. The single scale `[8]` is sufficient because higher FPN levels have larger receptive fields.

### Training

```python
from oriented_det.geometry import RBox

model.train()
images = [torch.randn(3, 512, 512)]
targets = [{
    "rboxes": [RBox(cx=150, cy=150, width=100, height=50, angle=0.5)],
    "labels": torch.tensor([1]),
}]

loss_dict = model(images, targets)
# Returns: {"loss_objectness", "loss_rpn_box_reg", "loss_classifier", "loss_box_reg"}
```

### Inference

```python
model.eval()
with torch.no_grad():
    outputs = model(images)
    rboxes = outputs[0]["rboxes"]  # List of RBox with predicted angles
    boxes = outputs[0]["boxes"]     # Tensor [N, 5] with [cx, cy, w, h, angle]
    labels = outputs[0]["labels"]   # Class labels
    scores = outputs[0]["scores"]    # Confidence scores
```

### Memory-Efficient Training

The `OrientedRCNN` model uses a **memory-efficient ROI align** implementation:

```python
model = OrientedRCNN(
    num_classes=15,
    backbone_name="resnet50",
    # Memory optimization parameters
    roi_chunk_size=32,        # Process ROIs in chunks (default: 32)
    roi_use_checkpoint=False,  # Enable gradient checkpointing for more savings
)
```

- **`roi_chunk_size`**: Number of ROIs to process in parallel. Lower values use less memory but are slower. Recommended: 16-64 for typical GPU memory (8-24GB).
- **`roi_use_checkpoint`**: If True, uses gradient checkpointing to trade compute for memory (~2x less memory, ~30% slower backward pass).

### Advanced Configuration

OrientedRCNN supports advanced loss configurations for handling class imbalance and hard examples.

#### Focal Loss

Focal loss is designed to address class imbalance and focus learning on hard examples. It down-weights easy examples and focuses on hard negatives.

**When to use focal loss:**
- Severe class imbalance (e.g., many background boxes, few object boxes)
- Hard examples are being overwhelmed by easy examples
- Standard cross-entropy loss is not converging well

**Basic focal loss:**

```python
from oriented_det import OrientedRCNN

model = OrientedRCNN(
    num_classes=15,
    roi_loss_type="focal",        # Use focal loss instead of cross-entropy
    roi_focal_alpha=1.0,          # Class weighting factor (default: 1.0)
    roi_focal_gamma=2.0,          # Focusing parameter (default: 2.0)
)
```

**Focal loss parameters:**
- **`roi_focal_alpha`** (default: 1.0): Class weighting factor. Can be used to balance positive/negative examples.
  - `alpha=1.0`: Equal weight for all classes
  - `alpha<1.0`: Down-weight positive examples
  - `alpha>1.0`: Up-weight positive examples
  
- **`roi_focal_gamma`** (default: 2.0): Focusing parameter that controls how much to focus on hard examples.
  - `gamma=0`: Equivalent to cross-entropy loss
  - `gamma>0`: Focuses more on hard examples
  - `gamma=2.0`: Standard value used in literature (good starting point)
  - Higher gamma: More focus on hard examples, less on easy ones

**Example: Adjusting focal loss for hard examples:**

```python
# Focus more on hard examples
model = OrientedRCNN(
    num_classes=15,
    roi_loss_type="focal",
    roi_focal_gamma=3.0,  # Stronger focus on hard examples
)

# Less focus on hard examples (closer to cross-entropy)
model = OrientedRCNN(
    num_classes=15,
    roi_loss_type="focal",
    roi_focal_gamma=1.0,  # Moderate focus
)
```

#### Class Weights

Class weights allow you to assign different importance to different classes, useful for handling imbalanced datasets.

**When to use class weights:**
- Some classes are much more common than others
- You want to prioritize detection of rare classes
- You have domain knowledge about class importance

**Using class weights with dictionary:**

```python
from oriented_det import OrientedRCNN

# Define class weights as dictionary
class_weights = {
    "plane": 2.0,           # Up-weight rare but important class
    "ship": 1.5,            # Slightly up-weight
    "small-vehicle": 0.8,   # Down-weight common class
    "large-vehicle": 1.0,   # Default weight
    # ... other classes
}

model = OrientedRCNN(
    num_classes=15,
    roi_class_weights=class_weights,  # Pass dictionary
)

# Must call set_class_weights() before training with class mapping
class_map = {
    "plane": 1,
    "ship": 2,
    "small-vehicle": 3,
    "large-vehicle": 4,
    # ... map all classes to IDs (1-indexed)
}
model.set_class_weights(class_map)
```

**Using class weights with tensor:**

```python
import torch
from oriented_det import OrientedRCNN

# Create weights tensor: [num_classes + 1] (background + object classes)
# Index 0 = background, indices 1-N = object classes
weights = torch.tensor([
    1.0,   # Background (index 0)
    2.0,   # Class 1 (plane)
    1.5,   # Class 2 (ship)
    0.8,   # Class 3 (small-vehicle)
    1.0,   # Class 4 (large-vehicle)
    # ... weights for all 15 classes
])

model = OrientedRCNN(
    num_classes=15,
    roi_class_weights=weights,  # Pass tensor directly
)
# No need to call set_class_weights() when using tensor
```

**Combining focal loss and class weights:**

```python
from oriented_det import OrientedRCNN

# Define class weights
class_weights = {
    "plane": 2.0,
    "ship": 1.5,
    "small-vehicle": 0.8,
}

# Create model with both focal loss and class weights
model = OrientedRCNN(
    num_classes=15,
    roi_loss_type="focal",
    roi_focal_gamma=2.0,
    roi_class_weights=class_weights,
)

# Set class weights before training
class_map = {"plane": 1, "ship": 2, "small-vehicle": 3, ...}
model.set_class_weights(class_map)
```

`RotatedFCOS` and `RotatedRetinaNet` accept the same `roi_class_weights` dict (or the `[C+1]` schedule tensor). `set_class_weights(class_map)` maps names onto one-hot columns `0..C-1`; missing names stay `1.0`. `background_weight` is ignored (no background class). Config: `loss.loss_type: focal_weighted` plus `loss.class_weight_*`. Recipe default `focal` stays unweighted.

**Best practices:**
- Start with default cross-entropy loss and standard training
- If class imbalance is severe, try focal loss with `gamma=2.0`
- If specific classes need more attention, add class weights
- Combine both only if needed (can be too aggressive)
- Monitor per-class performance to validate improvements

**Comparison: Cross-Entropy vs Focal Loss**

| Aspect | Cross-Entropy | Focal Loss |
|--------|--------------|------------|
| **Use case** | Balanced datasets, standard training | Class imbalance, hard examples |
| **Convergence** | Faster on easy examples | Slower but more stable |
| **Hard examples** | May be overwhelmed | Focused learning |
| **Parameters** | None | `alpha`, `gamma` |
| **Performance** | Good for balanced data | Better for imbalanced data |

## RotatedFasterRCNN

**RotatedFasterRCNN** (`RotatedFasterRCNN` in code) is the MMRotate-style two-stage detector:

- Oriented RPN proposals and oriented ROI align/head
- DOTA recipes: `configs/rotated_faster_rcnn/dota_le90_1x.json` and `configs/rotated_faster_rcnn/dota_le90_3x.json` (`model_type: rotated_faster_rcnn`)
- Key `model.*` knobs: `rpn_min_size`, standard RPN/ROI IoU thresholds, `add_gt_as_proposals`

```python
from oriented_det import RotatedFasterRCNN

model = RotatedFasterRCNN(
    num_classes=15,
    backbone_name="resnet50",
    pretrained_backbone=True,
)
```

Training is via `tools/train.py` — not a separate API from Oriented R-CNN beyond `model_type`.

## RotatedRetinaNet

**RotatedRetinaNet** is a complete Rotated RetinaNet implementation that performs **true oriented object detection**:

- ✅ Predicts oriented bounding boxes with 5 parameters (cx, cy, w, h, angle)
- ✅ Uses oriented anchors with rotation angles
- ✅ Uses focal loss for classification and smooth L1 for regression
- ✅ Preserves angle information throughout training and inference
- ✅ Uses oriented IoU for matching and oriented NMS for post-processing

### Initialization

```python
from oriented_det import RotatedRetinaNet

model = RotatedRetinaNet(
    num_classes=15,
    backbone_name="resnet50",
    pretrained_backbone=True
)
```

### Training and Inference

Usage is identical to `OrientedRCNN`:

```python
from oriented_det.geometry import RBox

# Training
model.train()
images = [torch.randn(3, 512, 512)]
targets = [{
    "rboxes": [RBox(cx=150, cy=150, width=100, height=50, angle=0.5)],
    "labels": torch.tensor([1]),
}]
loss_dict = model(images, targets)

# Inference
model.eval()
with torch.no_grad():
    outputs = model(images)
    rboxes = outputs[0]["rboxes"]  # List of RBox with predicted angles
    labels = outputs[0]["labels"]   # Class labels
    scores = outputs[0]["scores"]  # Confidence scores
```

## Using RBoxes Directly

OrientedRCNN accepts RBoxes directly in the target format - no conversion needed:

```python
from oriented_det.geometry import RBox

# Your dataset provides RBoxes
targets = [{
    "rboxes": [
        RBox(cx=100, cy=200, width=50, height=30, angle=0.5),
        RBox(cx=300, cy=400, width=80, height=40, angle=0.2),
    ],
    "labels": torch.tensor([1, 2]),
}]

# Use directly with model
model.train()
loss_dict = model([image], targets)
```

## Backbones

### ResNet-FPN

```python
from oriented_det.models.backbones import build_resnet_fpn_backbone

backbone = build_resnet_fpn_backbone(
    backbone_name="resnet50",
    pretrained=True
)
```

Available backbones: `"resnet18"`, `"resnet34"`, `"resnet50"`, `"resnet101"`, `"resnet152"`

### Backbone Customization

#### Using Pretrained Weights

```python
from oriented_det import OrientedRCNN

# Use pretrained ResNet50 backbone
model = OrientedRCNN(
    num_classes=15,
    backbone_name="resnet50",
    pretrained_backbone=True,  # Load ImageNet pretrained weights
)
```

#### Freezing Backbone Layers

Control how many backbone layers remain trainable:

```python
from oriented_det import OrientedRCNN

# Train only the last 3 layers (default)
model = OrientedRCNN(
    num_classes=15,
    backbone_name="resnet50",
    pretrained_backbone=True,
    trainable_layers=3,  # Last 3 layers trainable
)

# Train all layers (for training from scratch)
model = OrientedRCNN(
    num_classes=15,
    backbone_name="resnet50",
    pretrained_backbone=False,
    trainable_layers=5,  # All layers trainable (conv1 + layer1-4)
)
```

**Trainable layers for ResNet:**
- `trainable_layers=1`: Only layer4 trainable
- `trainable_layers=2`: layer3 + layer4 trainable
- `trainable_layers=3`: layer2 + layer3 + layer4 trainable (default)
- `trainable_layers=4`: layer1 + layer2 + layer3 + layer4 trainable
- `trainable_layers=5`: conv1 + layer1-4 trainable (all layers)

#### Using Custom Backbone

You can provide your own backbone module:

```python
from oriented_det import OrientedRCNN
from oriented_det.models.backbones import build_resnet_fpn_backbone

# Build custom backbone
custom_backbone = build_resnet_fpn_backbone(
    backbone_name="resnet101",
    pretrained=True,
    trainable_layers=3,
)

# Use with model
model = OrientedRCNN(
    num_classes=15,
    backbone=custom_backbone,  # Pass custom backbone
)
```

#### FPN Configuration

The FPN (Feature Pyramid Network) is automatically created with the backbone. FPN configuration is handled by torchvision:

**FPN Output Levels:**
- P3: Stride 8 (1/8 of input size)
- P4: Stride 16 (1/16 of input size)
- P5: Stride 32 (1/32 of input size)
- P6: Stride 64 (1/64 of input size) - optional
- P7: Stride 128 (1/128 of input size) - optional

**FPN Channels:**
- Default: 256 channels per level
- All levels have the same number of channels

**Accessing FPN Features:**

```python
import torch
from oriented_det.models.backbones import build_resnet_fpn_backbone

backbone = build_resnet_fpn_backbone("resnet50", pretrained=True)
backbone.eval()

# Forward pass
x = torch.randn(1, 3, 800, 800)
features = backbone(x)

# Features is an OrderedDict with keys "0", "1", "2", "3", ...
# Each value is a tensor of shape [batch, 256, H, W]
print(features.keys())  # dict_keys(['0', '1', '2', '3'])
print(features['0'].shape)  # torch.Size([1, 256, 100, 100])  # P3
print(features['1'].shape)  # torch.Size([1, 256, 50, 50])    # P4
print(features['2'].shape)  # torch.Size([1, 256, 25, 25])    # P5
```

#### Creating Custom Backbone

For advanced use cases, you can create a custom backbone:

```python
import torch
from torch import nn
from collections import OrderedDict
from oriented_det import OrientedRCNN

class CustomBackbone(nn.Module):
    """Custom backbone that returns FPN-like features."""
    
    def __init__(self):
        super().__init__()
        # Your custom architecture here
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.out_channels = 256
    
    def forward(self, x):
        # Return OrderedDict with feature maps at different scales
        f1 = self.conv1(x)  # 1/2 scale
        f2 = self.conv2(f1)  # 1/4 scale
        f3 = self.conv3(f2)  # 1/8 scale
        
        return OrderedDict({
            "0": f3,  # P3 equivalent (1/8)
            "1": f2,  # P4 equivalent (1/4)
            "2": f1,  # P5 equivalent (1/2)
        })

# Use custom backbone
custom_backbone = CustomBackbone()
model = OrientedRCNN(
    num_classes=15,
    backbone=custom_backbone,
)
```

**Requirements for custom backbone:**
- Must return `OrderedDict[str, torch.Tensor]` in forward pass
- Keys should be string numbers ("0", "1", "2", ...)
- Values should be feature maps at different scales
- Must have `out_channels` attribute indicating feature dimension
- Feature maps should be in order from highest resolution to lowest

#### Backbone Utilities

```python
from oriented_det.models.backbones.utils import (
    freeze_layers,
    count_trainable_parameters,
)

# Freeze specific layers
backbone = build_resnet_fpn_backbone("resnet50", pretrained=True)
freeze_layers(backbone.body, trainable_layers=2)  # Keep last 2 layers trainable

# Count trainable parameters
num_params = count_trainable_parameters(backbone)
print(f"Trainable parameters: {num_params:,}")
```

#### Best Practices

1. **Use pretrained weights**: Always use `pretrained_backbone=True` when available
2. **Freeze early layers**: For fine-tuning, freeze early layers (`trainable_layers=3`)
3. **Train all layers**: For training from scratch, set `trainable_layers=5`
4. **Match backbone to task**: ResNet50 is a good default, ResNet101 for better accuracy
5. **Custom backbones**: Only create custom backbones if you have specific requirements

## Understanding Loss Components

The Oriented R-CNN model produces **four distinct loss components**:

1. **`loss_objectness`** (RPN): Binary classification (object vs. background)
   - Expected: 0.5-0.7 early training → 0.1-0.3 converged
   - Should decrease steadily

2. **`loss_rpn_box_reg`** (RPN): Box regression for refining anchors into proposals
   - Expected: 0.8-1.5 early training → 0.3-0.6 converged
   - Operates on predefined anchors

3. **`loss_classifier`** (ROI): Multi-class classification for object classes
   - Expected: 0.5-1.0 early training → 0.2-0.4 converged
   - May be slower to converge than box regression

4. **`loss_box_reg`** (ROI): Box regression for refining proposals into final boxes
   - Expected: 0.8-1.5 early training → 0.2-0.5 converged
   - Operates on proposals from RPN stage

**Warning Signs:**
- All losses remain high (>1.0) after 5+ epochs → Check learning rate, data loading
- Model predicts only one class → Check class imbalance, disable curriculum learning
- Losses oscillating wildly → Reduce learning rate, check gradient clipping

## See Also

- [API Reference](../api/models.md) - Complete API documentation
- [Training Guide](training.md) - Training models
- [Data Guide](data.md) - Loading datasets


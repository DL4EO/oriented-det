# Utils API Reference

::: oriented_det.utils
    options:
      show_root_heading: true
      show_root_toc_entry: true
      show_source: true

## Important Notes

### Configuration Management

The `oriented_det.utils` package provides configuration helpers:

- `load_config(path_or_dict, overrides=...)` reads JSON/YAML configs, applies dotted-key overrides, and returns an immutable `FrozenConfig`
- **Nested config inheritance** via `_base_` field (MMRotate-style)
  - Supports single or multiple base configs
  - Recursive loading with circular dependency detection
  - Relative path resolution (relative to config file's directory)
  - Deep merging of nested dictionaries
- Configs are immutable to prevent accidental modifications
- Supports nested access with dot notation

### Visualization

The visualization utilities provide lightweight helpers for debugging:

- `viz.draw_boxes(image, boxes, specs=...)` overlays polygons, `QBox`, or `RBox` detections onto a Pillow image or NumPy array
- Pillow is optional; when it is not installed the helper raises a clear error
- Supports custom colors, line widths, and drawing specs

## Examples

### Configuration Loading

```python
from oriented_det.utils import load_config

# From file
cfg = load_config("configs/retinanet.yaml", overrides=["trainer.epochs=24"])

# From dict
cfg = load_config({"model": {"num_classes": 15}})

# Access with dot notation
print(cfg.model.num_classes)  # 15

# With nested config inheritance
# config.json:
# {
#   "_base_": [
#     "../_base_/models/oriented_rcnn_r50.json",
#     "../_base_/schedules/1x.json"
#   ],
#   "training": {"learning_rate": 0.01}
# }
cfg = load_config("config.json")
# Automatically loads and merges base configs, then applies overrides
```

### Visualization

```python
from oriented_det.utils import viz
from oriented_det.geometry import RBox
from PIL import Image

image = Image.open("image.jpg")

# Draw RBoxes
rboxes = [RBox(100, 200, 50, 30, 0.5)]
result = viz.draw_boxes(image, rboxes)

# Save
result.save("output.jpg")
```

### Custom Drawing Specs

```python
from oriented_det.utils import viz, DrawingSpec

# Custom drawing specs
specs = [
    DrawingSpec(outline=(255, 0, 0), width=3),  # Red, thick
    DrawingSpec(outline=(0, 255, 0), width=2),  # Green, medium
]

result = viz.draw_boxes(image, rboxes, specs=specs)
```


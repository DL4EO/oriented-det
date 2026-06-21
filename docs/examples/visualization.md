# Visualization Tool

This tool demonstrates how to create and visualize geometric primitives.

## Running the Tool

```bash
python tools/visualize_boxes.py
```

## Code Overview

The visualization script shows:

1. **Creating geometric primitives**:
   - RBox (rotated bounding box)
   - QBox (quadrilateral box)
   - Polygon

2. **Converting between formats**:
   - RBox → QBox → Polygon
   - Round-trip conversions

3. **Visualizing on images**:
   - Drawing boxes on images
   - Custom colors and styles

## Key Concepts

### Creating RBoxes

```python
from oriented_det.geometry import RBox
from math import radians

rbox = RBox(
    cx=256,      # Center X
    cy=256,      # Center Y
    width=100,   # Width
    height=50,   # Height
    angle=radians(45)  # Rotation angle
)
```

### Converting Formats

```python
from oriented_det.geometry import transforms

# RBox to QBox
qbox = transforms.rbox_to_qbox(rbox)

# QBox to Polygon
polygon = transforms.qbox_to_polygon(qbox)

# Round-trip
recovered = transforms.polygon_to_rbox(polygon)
```

### Drawing

```python
from oriented_det.utils import viz
from PIL import Image

image = Image.new("RGB", (512, 512), "white")
result = viz.draw_boxes(image, [rbox])
result.save("output.png")
```

## See Also

- [Geometry Guide](../user-guide/geometry.md) - Detailed geometry documentation
- [API Reference](../api/geometry.md) - Complete API


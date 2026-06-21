# Geometry API Reference

::: oriented_det.geometry
    options:
      show_root_heading: true
      show_root_toc_entry: true
      show_source: true

## Important Notes

### Angle Convention

The geometry module uses a **full circle** angle representation by default:

- **Range**: Angles are normalized to `(-π, π]` interval
- **Direction**: Positive angles rotate counter-clockwise (mathematical convention)
- **Origin**: Top-left corner of the image (x-axis right, y-axis down)
- **Normalization**: Angles are automatically normalized using `atan2(sin, cos)` to handle periodicity

### le90 Convention

For DOTA compatibility, use the `normalize_le90()` function to convert to the "long edge 90°" convention:

- **Range**: `[-π/2, π/2)`
- **Constraint**: Width is always >= height (dimensions are swapped if needed)
- **Usage**: Ensures boxes with the same orientation have the same representation

```python
from oriented_det.geometry import RBox, normalize_le90
import math

rbox = RBox(100, 100, 50, 100, math.pi)  # width < height, angle = π
normalized = normalize_le90(rbox)
# normalized.width = 100, normalized.height = 50, normalized.angle ≈ -π/2
```

### Round-trip Conversions

All conversions between `Polygon`, `QBox`, and `RBox` are designed to be **lossless** within floating-point precision:

```python
from oriented_det.geometry import RBox, transforms
import math

rbox = RBox(10.0, -3.0, 5.0, 2.0, math.radians(30))
qbox = transforms.rbox_to_qbox(rbox)
recovered = transforms.qbox_to_rbox(qbox)

assert abs(recovered.cx - rbox.cx) < 1e-6
assert abs(recovered.angle - rbox.angle) < 1e-6
```

### DOTA Polygon Format

When working with DOTA annotations, polygons use 8 coordinates:
```
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult
```

The loader automatically:
1. Parses polygons from annotation files
2. Converts to `QBox` (normalizes point order, ensures counter-clockwise)
3. Converts to `RBox` (computes center, dimensions, angle)

## Examples

### Creating Geometric Primitives

```python
from oriented_det.geometry import Polygon, QBox, RBox
from math import radians

# Polygon from points
poly = Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])

# RBox with rotation
rbox = RBox(cx=512, cy=384, width=120, height=48, angle=radians(20))

# QBox from 4 points
qbox = QBox([(0, 0), (4, 0), (4, 2), (0, 2)])
```

### Geometric Operations

```python
# Translation
translated = poly.translate(dx=10, dy=20)

# Rotation around origin
rotated = poly.rotate(radians=0.5, origin=(0, 0))

# Ensure counter-clockwise orientation
ccw_poly = poly.ensure_orientation(clockwise=False)
```

### Conversions

```python
from oriented_det.geometry import transforms

# RBox ↔ QBox
qbox = transforms.rbox_to_qbox(rbox)
rbox = transforms.qbox_to_rbox(qbox)

# RBox ↔ Polygon
polygon = transforms.rbox_to_polygon(rbox)
rbox = transforms.polygon_to_rbox(polygon)

# QBox ↔ Polygon
polygon = transforms.qbox_to_polygon(qbox)
qbox = transforms.polygon_to_qbox(polygon)
```

### Transformations

```python
from oriented_det.geometry.transforms import flip_horizontal, rotate_rbox

# Flip horizontally (mirror across vertical axis)
flipped = flip_horizontal(rbox, image_width=1024)

# Rotate around center
rotated = rotate_rbox(rbox, angle=radians(90), origin=(512, 512))
```


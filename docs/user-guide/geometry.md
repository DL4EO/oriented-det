# Geometry Module

The geometry module provides core primitives for working with rotated and oriented bounding boxes.

## Overview

OrientedDet provides three main geometric primitives:

- **`Polygon`** - Immutable polygon with validation and geometric operations
- **`QBox`** - Quadrilateral bounding box (4 corner points)
- **`RBox`** - Rotated bounding box (center, width, height, angle)

## Polygon

A `Polygon` is an immutable collection of points that represents a closed shape.

### Creating Polygons

```python
from oriented_det.geometry import Polygon

# From points
poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

# Using class method for rectangles
rect = Polygon.rectangle(cx=50, cy=50, width=100, height=80)
```

### Properties

```python
area = poly.area                    # Area of the polygon
centroid = poly.centroid            # (cx, cy) tuple
bounds = poly.bounds                # (x_min, y_min, x_max, y_max)
is_clockwise = poly.is_clockwise    # Orientation check
```

### Operations

```python
# Translation
translated = poly.translate(dx=10, dy=20)

# Rotation
rotated = poly.rotate(radians=0.5, origin=(0, 0))

# Ensure orientation
ccw_poly = poly.ensure_orientation(clockwise=False)
```

## RBox (Rotated Bounding Box)

An `RBox` represents a rotated rectangle using center coordinates, dimensions, and angle.

### Creating RBoxes

```python
from oriented_det.geometry import RBox
from math import radians

# Direct creation
rbox = RBox(cx=100, cy=200, width=50, height=30, angle=radians(45))

# From polygon
rbox = RBox.from_polygon(polygon)

# From QBox
rbox = RBox.from_qbox(qbox)
```

### Properties

```python
area = rbox.area                    # width * height
aspect_ratio = rbox.aspect_ratio    # width / height
corners = rbox.corners()            # 4 corner points
```

### Conversions

```python
polygon = rbox.to_polygon()         # Convert to Polygon
qbox = rbox.to_qbox()               # Convert to QBox
```

## QBox (Quadrilateral Box)

A `QBox` represents a quadrilateral using 4 corner points.

### Creating QBoxes

```python
from oriented_det.geometry import QBox

# From 4 points
qbox = QBox([
    (0, 0),
    (10, 0),
    (10, 10),
    (0, 10)
])
```

### Properties

```python
center = qbox.center                # (cx, cy) tuple
width = qbox.width                  # Average width
height = qbox.height                # Average height
angle = qbox.angle                  # Orientation angle
edges = qbox.edges                  # Edge lengths
```

## Transforms

The `transforms` module provides conversion functions between all formats.

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

### Round-trip Conversions

All conversions are designed to be lossless (within floating-point precision):

```python
rbox = RBox(100, 200, 50, 30, 0.5)
qbox = transforms.rbox_to_qbox(rbox)
recovered = transforms.qbox_to_rbox(qbox)

assert abs(recovered.cx - rbox.cx) < 1e-6
assert abs(recovered.angle - rbox.angle) < 1e-6
```

## DOTA Polygon Format

When working with DOTA dataset annotations, polygons use 8 coordinates:

```
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult
```

The corners are ordered sequentially around the polygon perimeter. The loader converts:
1. Polygon → QBox (normalizes point order, ensures counter-clockwise)
2. QBox → RBox (computes center, dimensions, angle)

## Best Practices

1. **Use RBox for rotated rectangles** - Most efficient representation
2. **Use Polygon for arbitrary shapes** - Most flexible
3. **Use QBox for quadrilateral conversions** - Intermediate format
4. **Leverage transforms** - Centralized conversion logic
5. **Check orientation** - Use `ensure_orientation()` when needed

## See Also

- [API Reference](../api/geometry.md) - Complete API documentation
- [Operations Guide](operations.md) - Using geometry with IoU/NMS


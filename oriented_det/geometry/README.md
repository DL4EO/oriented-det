# geometry

Rotated-detection primitives: polygons, quadrilateral boxes (`QBox`), rotated boxes (`RBox`), and coordinate transforms.

| Module | Role |
|--------|------|
| `poly.py` | `Polygon` — DOTA eight-point format, area, transforms |
| `qbox.py` | `QBox` — four corner points |
| `rbox.py` | `RBox` — center, size, angle; le90 normalization |
| `transforms.py` | Conversions polygon ↔ qbox ↔ rbox ↔ axis-aligned box |

User guide: [Geometry](../../docs/user-guide/geometry.md). API: [Geometry API](../../docs/api/geometry.md).

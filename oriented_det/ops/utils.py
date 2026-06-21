"""Utility helpers shared across ops modules."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple
import math

from ..geometry import Polygon, QBox, RBox

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # Optional Shapely for polygon intersections.
    from shapely.geometry import Polygon as ShapelyPolygon
    SHAPELY_AVAILABLE = True
except Exception:  # pragma: no cover - exercised when shapely is missing.
    ShapelyPolygon = None  # type: ignore[assignment]
    SHAPELY_AVAILABLE = False

# torchvision does not provide rotated box IoU or NMS; tests use these for backend validation
TORCH_BOX_IOU_ROTATED = None
TORCH_NMS_ROTATED = None

Point = Tuple[float, float]


def _as_point(value: Sequence[float]) -> Point:
    if len(value) != 2:
        raise ValueError("Point sequence must have exactly 2 elements.")
    x, y = float(value[0]), float(value[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Point coordinates must be finite.")
    return (x, y)


def as_polygon(value) -> Polygon:
    """Return a counter-clockwise polygon from multiple representations."""
    if isinstance(value, Polygon):
        poly = value
    elif isinstance(value, RBox):
        poly = value.to_polygon()
    elif isinstance(value, QBox):
        poly = value.to_polygon()
    else:
        poly = Polygon(value)
    return poly if not poly.is_clockwise else poly.ensure_orientation(clockwise=False)


def as_rbox(value) -> RBox:
    """Return a normalized RBox from various sources."""
    if isinstance(value, RBox):
        return value
    if isinstance(value, QBox):
        return RBox.from_qbox(value)
    if isinstance(value, Polygon):
        if len(value) != 4:
            raise ValueError("Only quadrilateral polygons can be converted to RBox.")
        return RBox.from_polygon(value)
    seq = tuple(value)  # type: ignore[arg-type]
    if len(seq) != 5:
        raise ValueError("Iterable RBox representation must have 5 elements.")
    cx, cy, w, h, angle = map(float, seq)
    return RBox(cx, cy, w, h, angle)


def torch_backend_available() -> bool:
    """Check if PyTorch is available for tensor operations."""
    return torch is not None


def rboxes_to_tensor(boxes, *, device=None):
    if torch is None:
        return None
    if hasattr(boxes, "shape"):
        return torch.as_tensor(boxes, dtype=torch.float32, device=device)
    data = []
    for box in boxes:
        rb = as_rbox(box)
        data.append([rb.cx, rb.cy, rb.width, rb.height, rb.angle])
    return torch.tensor(data, dtype=torch.float32, device=device)


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segment_intersection(p1: Point, p2: Point, p3: Point, p4: Point) -> Point:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if math.isclose(denom, 0.0, abs_tol=1e-12):
        return p2
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (px, py)


def sutherland_hodgman(subject: Iterable[Sequence[float]], clip: Iterable[Sequence[float]]) -> List[Point]:
    """Clip a polygon using the Sutherland-Hodgman algorithm."""
    subject_pts = [_as_point(pt) for pt in subject]
    clip_pts = [_as_point(pt) for pt in clip]
    if len(subject_pts) < 3 or len(clip_pts) < 3:
        return []

    output = subject_pts
    cp1 = clip_pts[-1]
    for cp2 in clip_pts:
        input_list = output
        output = []
        if not input_list:
            break
        s = input_list[-1]
        for e in input_list:
            inside_e = _cross(cp1, cp2, e) >= 0
            inside_s = _cross(cp1, cp2, s) >= 0
            if inside_e:
                if not inside_s:
                    output.append(_segment_intersection(s, e, cp1, cp2))
                output.append(e)
            elif inside_s:
                output.append(_segment_intersection(s, e, cp1, cp2))
            s = e
        cp1 = cp2

    return output


def polygon_area(points: Iterable[Sequence[float]]) -> float:
    pts = [_as_point(pt) for pt in points]
    if len(pts) < 3:
        return 0.0
    area = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


def _aabb_overlaps(
    aabb_a: Tuple[float, float, float, float], 
    aabb_b: Tuple[float, float, float, float]
) -> bool:
    """Check if two axis-aligned bounding boxes overlap.
    
    This is a fast pre-filter before computing expensive rotated IoU.
    If AABBs don't overlap, the rotated boxes cannot overlap either.
    
    Args:
        aabb_a: (x_min, y_min, x_max, y_max) for box A
        aabb_b: (x_min, y_min, x_max, y_max) for box B
    
    Returns:
        True if boxes overlap, False otherwise
    """
    x_min_a, y_min_a, x_max_a, y_max_a = aabb_a
    x_min_b, y_min_b, x_max_b, y_max_b = aabb_b
    
    # No overlap if one box is completely to the left/right/above/below the other
    return not (x_max_a < x_min_b or x_max_b < x_min_a or 
                y_max_a < y_min_b or y_max_b < y_min_a)


_EXACT_IOU_FALLBACK_WARNED = False


def resolve_exact_polygon_iou_backend(*, warn_if_fallback: bool = True) -> str:
    """Backend string for exact polygon IoU/NMS (``final_nms_use_cpu`` / mAP exact mode).

    Prefers ``shapely``. Falls back to ``python`` (Sutherland–Hodgman) with a prominent
    warning when Shapely is not installed.
    """
    global _EXACT_IOU_FALLBACK_WARNED
    if SHAPELY_AVAILABLE:
        return "shapely"
    if warn_if_fallback and not _EXACT_IOU_FALLBACK_WARNED:
        _EXACT_IOU_FALLBACK_WARNED = True
        print(
            "\n"
            + "=" * 72
            + "\n"
            "WARNING: Shapely is not installed. Exact polygon IoU/NMS will use the\n"
            "built-in Sutherland–Hodgman clipper instead of Shapely. Thin or elongated\n"
            "OBBs (ships, large-vehicle, harbor) may disagree with MMRotate-style eval.\n"
            "Reinstall oriented-det (shapely is a core dependency).\n"
            + "=" * 72
            + "\n",
            flush=True,
        )
    return "python"


def _should_use_shapely(backend: str) -> bool:
    """Check if Shapely backend should be used for intersection calculations."""
    backend = backend.lower()
    if backend not in {"auto", "python", "shapely"}:
        raise ValueError("intersection_backend must be 'auto', 'python', or 'shapely'.")
    if backend == "python":
        return False
    if not SHAPELY_AVAILABLE:
        if backend == "shapely":
            raise RuntimeError(
                "Shapely backend requested but unavailable. Reinstall oriented-det (shapely is a core dependency)."
            )
        return False
    return True


def polygon_intersection_area(
    poly_a: Polygon, 
    poly_b: Polygon, 
    *, 
    backend: str = "auto"
) -> float:
    """Compute the intersection area between two polygons.
    
    Args:
        poly_a: First polygon
        poly_b: Second polygon
        backend: Backend to use ('auto', 'python', or 'shapely')
            - 'auto': Use Shapely if available, otherwise use Python implementation
            - 'python': Use the Sutherland-Hodgman algorithm (default fallback)
            - 'shapely': Use Shapely library (requires shapely package)
    
    Returns:
        Intersection area as a float
    """
    if _should_use_shapely(backend):
        # Convert to Shapely polygons
        shapely_poly_a = ShapelyPolygon(poly_a.points)
        shapely_poly_b = ShapelyPolygon(poly_b.points)
        intersection = shapely_poly_a.intersection(shapely_poly_b)
        if intersection.is_empty:
            return 0.0
        return float(intersection.area)
    
    # Fallback to Python implementation (Sutherland-Hodgman)
    inter_points = sutherland_hodgman(poly_a.points, poly_b.points)
    if len(inter_points) < 3:
        return 0.0
    return polygon_area(inter_points)


__all__ = [
    "as_polygon",
    "as_rbox",
    "polygon_area",
    "polygon_intersection_area",
    "resolve_exact_polygon_iou_backend",
    "sutherland_hodgman",
    "torch_backend_available",
    "rboxes_to_tensor",
    "SHAPELY_AVAILABLE",
    "_aabb_overlaps",  # Internal utility for AABB pre-filtering
]

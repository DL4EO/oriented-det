import math

from oriented_det.geometry import Polygon, QBox, RBox, normalize_le90, transforms


def test_polygon_area_and_centroid():
    poly = Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
    assert math.isclose(poly.area, 2.0)
    assert poly.is_clockwise is False
    assert poly.bounds == (0, 0, 2, 1)
    cx, cy = poly.centroid
    assert math.isclose(cx, 1.0)
    assert math.isclose(cy, 0.5)


def test_polygon_rotation():
    rect = Polygon.rectangle(0, 0, 2, 4)
    rotated = rect.rotate(math.pi / 2)
    bounds = rotated.bounds
    assert math.isclose(bounds[0], -2.0, abs_tol=1e-9)
    assert math.isclose(bounds[2], 2.0, abs_tol=1e-9)


def test_rbox_qbox_round_trip():
    rbox = RBox(10.0, -3.0, 5.0, 2.0, math.radians(30))
    qbox = transforms.rbox_to_qbox(rbox)
    recovered = transforms.qbox_to_rbox(qbox)
    assert math.isclose(recovered.cx, rbox.cx, abs_tol=1e-6)
    assert math.isclose(recovered.cy, rbox.cy, abs_tol=1e-6)
    assert math.isclose(recovered.width, rbox.width, rel_tol=1e-6)
    assert math.isclose(recovered.height, rbox.height, rel_tol=1e-6)
    assert math.isclose(recovered.angle, rbox.angle, abs_tol=1e-6)


def test_points_to_rbox_agrees_with_polygon_conversion():
    qbox = QBox([(0, 0), (4, 1), (3, 5), (-1, 4)])
    rbox_from_q = RBox.from_qbox(qbox)
    rbox_from_poly = transforms.polygon_to_rbox(qbox.to_polygon())
    assert math.isclose(rbox_from_q.cx, rbox_from_poly.cx, abs_tol=1e-6)
    assert math.isclose(rbox_from_q.cy, rbox_from_poly.cy, abs_tol=1e-6)
    assert math.isclose(rbox_from_q.width, rbox_from_poly.width, abs_tol=1e-6)
    assert math.isclose(rbox_from_q.height, rbox_from_poly.height, abs_tol=1e-6)
    assert math.isclose(rbox_from_q.angle, rbox_from_poly.angle, abs_tol=1e-6)


def test_polygon_validation():
    """Test polygon validation and error handling."""
    import pytest
    
    # Too few points
    with pytest.raises(ValueError):
        Polygon([(0, 0), (1, 1)])
    
    # Degenerate polygon (zero area)
    with pytest.raises(ValueError):
        Polygon([(0, 0), (1, 0), (2, 0), (1, 0)])


def test_rbox_validation():
    """Test RBox validation."""
    import pytest
    
    # Invalid dimensions
    with pytest.raises(ValueError):
        RBox(0, 0, -1, 1, 0)
    
    with pytest.raises(ValueError):
        RBox(0, 0, 1, 0, 0)


def test_qbox_properties():
    """Test QBox properties."""
    qbox = QBox([(0, 0), (4, 0), (4, 2), (0, 2)])
    assert math.isclose(qbox.width, 4.0)
    assert math.isclose(qbox.height, 2.0)
    assert len(qbox.edges) == 4


def test_normalize_le90_is_idempotent():
    """normalize_le90(normalize_le90(x)) must equal normalize_le90(x)."""
    cases = [
        RBox(0, 0, 10, 4, 0.0),
        RBox(1, -2, 4, 10, math.pi / 3),
        RBox(5, 5, 7, 7, -3.5 * math.pi),
        RBox(-3, 8, 12, 3, 2.1 * math.pi),
    ]
    for box in cases:
        once = normalize_le90(box)
        twice = normalize_le90(once)
        assert math.isclose(twice.cx, once.cx, abs_tol=1e-12)
        assert math.isclose(twice.cy, once.cy, abs_tol=1e-12)
        assert math.isclose(twice.width, once.width, abs_tol=1e-12)
        assert math.isclose(twice.height, once.height, abs_tol=1e-12)
        assert math.isclose(twice.angle, once.angle, abs_tol=1e-12)


def test_normalize_le90_angle_boundaries():
    """Angle output always stays in [-pi/2, pi/2) near boundaries."""
    eps = 1e-9
    angles = [
        -math.pi / 2,
        -math.pi / 2 + eps,
        -math.pi / 2 - eps,
        0.0,
        math.pi / 2 - eps,
        math.pi / 2,
        math.pi,
        -math.pi,
    ]
    for a in angles:
        box = RBox(0, 0, 6, 2, a)
        n = normalize_le90(box)
        assert -math.pi / 2 <= n.angle < math.pi / 2
        assert n.width >= n.height


def test_normalize_le90_square_box_stability():
    """Square boxes stay valid and stable under repeated normalization."""
    box = RBox(2.0, -1.0, 5.0, 5.0, 7.25)
    n1 = normalize_le90(box)
    n2 = normalize_le90(n1)
    assert math.isclose(n1.width, n1.height, abs_tol=1e-12)
    assert -math.pi / 2 <= n1.angle < math.pi / 2
    assert math.isclose(n2.width, n1.width, abs_tol=1e-12)
    assert math.isclose(n2.height, n1.height, abs_tol=1e-12)
    assert math.isclose(n2.angle, n1.angle, abs_tol=1e-12)


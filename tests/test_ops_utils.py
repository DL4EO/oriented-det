"""Tests for low-level ops utility kernels and conversions."""

import math

import pytest

from oriented_det.geometry import Polygon, QBox, RBox
from oriented_det.ops.utils import (
    _aabb_overlaps,
    as_polygon,
    as_rbox,
    polygon_intersection_area,
    sutherland_hodgman,
)


def test_sutherland_hodgman_disjoint_returns_empty():
    subject = [(0, 0), (2, 0), (2, 2), (0, 2)]
    clip = [(10, 10), (12, 10), (12, 12), (10, 12)]
    inter = sutherland_hodgman(subject, clip)
    assert inter == []


def test_sutherland_hodgman_partial_overlap_has_expected_area():
    subject = [(0, 0), (4, 0), (4, 4), (0, 4)]
    clip = [(2, 1), (5, 1), (5, 3), (2, 3)]
    inter = sutherland_hodgman(subject, clip)
    # overlap rectangle: x in [2,4], y in [1,3] => area 4
    poly = Polygon(inter)
    assert math.isclose(poly.area, 4.0, rel_tol=1e-9)


def test_sutherland_hodgman_containment_returns_subject_area():
    subject = [(1, 1), (2, 1), (2, 2), (1, 2)]
    clip = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
    inter = sutherland_hodgman(subject, clip)
    poly = Polygon(inter)
    assert math.isclose(poly.area, 1.0, rel_tol=1e-9)


def test_polygon_intersection_area_orientation_invariant():
    poly_a_ccw = Polygon([(0, 0), (4, 0), (4, 3), (0, 3)])
    # Same polygon but clockwise input order.
    poly_a_cw = Polygon([(0, 0), (0, 3), (4, 3), (4, 0)])
    poly_b = Polygon([(2, 1), (5, 1), (5, 4), (2, 4)])

    inter_ccw = polygon_intersection_area(poly_a_ccw, poly_b, backend="python")
    inter_cw = polygon_intersection_area(poly_a_cw, poly_b, backend="python")
    assert math.isclose(inter_ccw, inter_cw, rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(inter_ccw, 4.0, rel_tol=1e-9)


def test_aabb_overlaps_touching_edges_and_separated():
    # Touching at a vertical edge counts as overlap in current implementation.
    assert _aabb_overlaps((0, 0, 1, 1), (1, 0, 2, 1))
    # Touching at a corner also counts as overlap.
    assert _aabb_overlaps((0, 0, 1, 1), (1, 1, 2, 2))
    # Strictly separated should be False.
    assert not _aabb_overlaps((0, 0, 1, 1), (1.000001, 0, 2, 1))


def test_as_polygon_returns_ccw_for_clockwise_input():
    cw = [(0, 0), (0, 2), (2, 2), (2, 0)]  # clockwise
    poly = as_polygon(cw)
    assert not poly.is_clockwise


def test_as_rbox_accepts_sequence_and_qbox():
    rb_from_seq = as_rbox((10, 20, 8, 4, 0.2))
    assert isinstance(rb_from_seq, RBox)
    assert math.isclose(rb_from_seq.cx, 10.0)
    assert math.isclose(rb_from_seq.cy, 20.0)

    qbox = QBox([(0, 0), (4, 0), (4, 2), (0, 2)])
    rb_from_q = as_rbox(qbox)
    assert isinstance(rb_from_q, RBox)
    assert math.isclose(rb_from_q.width, 4.0, rel_tol=1e-6)
    assert math.isclose(rb_from_q.height, 2.0, rel_tol=1e-6)


def test_as_rbox_invalid_sequence_raises():
    with pytest.raises(ValueError):
        as_rbox((1, 2, 3, 4))

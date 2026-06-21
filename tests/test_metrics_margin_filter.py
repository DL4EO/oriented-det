"""Tests for tiled-val metrics margin filtering (tools/save_predictions)."""

from oriented_det.geometry import RBox
from tools.save_predictions import _rbox_centroid_in_tile_interior  # metrics helpers stay in save_predictions CLI module


def test_centroid_in_tile_interior_center_kept():
    r = RBox(cx=512.0, cy=512.0, width=10.0, height=10.0, angle=0.0)
    assert _rbox_centroid_in_tile_interior(r, 1024, 1024, 128) is True


def test_centroid_in_edge_margin_band_excluded():
    r = RBox(cx=64.0, cy=512.0, width=10.0, height=10.0, angle=0.0)
    assert _rbox_centroid_in_tile_interior(r, 1024, 1024, 128) is False
    r2 = RBox(cx=1000.0, cy=1000.0, width=10.0, height=10.0, angle=0.0)
    assert _rbox_centroid_in_tile_interior(r2, 1024, 1024, 128) is False


def test_zero_margin_keeps_all():
    r = RBox(cx=1.0, cy=1.0, width=10.0, height=10.0, angle=0.0)
    assert _rbox_centroid_in_tile_interior(r, 1024, 1024, 0) is True

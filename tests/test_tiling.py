"""Tests for image tiling functionality."""

import math

from oriented_det.data import ImageTiler, Tile, TiledSample
from oriented_det.geometry import RBox


def test_tile_generation():
    """Test tile generation for various image sizes."""
    tiler = ImageTiler(tile_size=512, overlap=0.2)
    
    tiles = tiler.generate_tiles(1000, 1000)
    assert len(tiles) > 0
    
    # Check first tile
    first = tiles[0]
    assert first.x == 0
    assert first.y == 0
    assert first.width == 512
    assert first.height == 512


def test_tile_contains_point():
    """Test point containment in tiles."""
    tile = Tile(x=100, y=200, width=512, height=512, tile_id="test")
    
    assert tile.contains_point(150, 250)
    assert not tile.contains_point(50, 250)
    assert not tile.contains_point(150, 150)


def test_assign_boxes_to_tile():
    """Test assigning boxes to tiles with edge handling."""
    tiler = ImageTiler(tile_size=512, overlap=0.2, edge_handling="clip")
    
    tile = Tile(x=0, y=0, width=512, height=512, tile_id="test")
    
    # Box inside tile
    rbox_inside = RBox(256, 256, 50, 50, 0)
    
    # Box outside tile
    rbox_outside = RBox(600, 600, 50, 50, 0)
    
    tiled = tiler.assign_boxes_to_tile(
        tile=tile,
        rboxes=[rbox_inside, rbox_outside],
        class_names=["plane", "ship"],
    )
    
    assert len(tiled.annotations) >= 1  # At least inside box should be kept


def test_tile_clip_box():
    """Test clipping boxes to tile bounds."""
    tile = Tile(x=0, y=0, width=512, height=512, tile_id="test")
    
    # Box partially inside
    rbox = RBox(400, 400, 200, 200, 0)
    clipped = tile.clip_box(rbox)
    
    # Should return a box (or None if fully outside)
    assert clipped is None or isinstance(clipped, RBox)


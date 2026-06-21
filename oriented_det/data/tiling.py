"""Efficient image tiling with edge handling for large aerial images."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import math

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

from ..geometry import Polygon, RBox, transforms


@dataclass(frozen=True)
class Tile:
    """Represents a single tile/patch from a larger image."""

    x: int  # Left coordinate in original image
    y: int  # Top coordinate in original image
    width: int
    height: int
    tile_id: str
    
    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Returns (x, y, x + width, y + height)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def contains_point(self, px: float, py: float) -> bool:
        """Check if a point lies within this tile."""
        return self.x <= px < self.x + self.width and self.y <= py < self.y + self.height
    
    def clip_box(self, rbox: RBox) -> Optional[RBox]:
        """Clip an RBox to this tile's bounds, returning None if fully outside."""
        poly = rbox.to_polygon()
        tile_poly = Polygon.rectangle(
            self.x + self.width / 2.0,
            self.y + self.height / 2.0,
            self.width,
            self.height
        )
        
        # Quick AABB check
        rbox_bounds = poly.bounds
        tile_bounds = tile_poly.bounds
        
        if (rbox_bounds[2] < tile_bounds[0] or rbox_bounds[0] > tile_bounds[2] or
            rbox_bounds[3] < tile_bounds[1] or rbox_bounds[1] > tile_bounds[3]):
            return None
        
        # Check if any corner of rbox is inside tile
        corners = rbox.corners()
        has_inside = any(self.contains_point(x, y) for x, y in corners)
        
        if not has_inside:
            return None
        
        # For simplicity, return the original box if any part overlaps
        # More precise clipping would require polygon intersection
        return rbox


@dataclass(frozen=True)
class TiledSample:
    """A sample with its annotations clipped to a specific tile."""

    tile: Tile
    image_path: Path
    annotations: Tuple[RBox, ...]  # RBoxes clipped to tile coordinates
    class_names: Tuple[str, ...]
    difficult_flags: Tuple[int, ...]


class ImageTiler:
    """Efficient tiler for splitting large images into overlapping patches.
    
    The tiler supports configurable overlap and minimum overlap thresholds
    for keeping annotations that cross tile boundaries.
    """

    def __init__(
        self,
        tile_size: int = 1024,
        overlap: float = 0.2,
        *,
        min_box_area: float = 16.0,
        min_overlap_ratio: float = 0.0,
        edge_handling: str = "clip",  # "clip", "ignore", "keep"
    ):
        """Initialize ImageTiler.
        
        Args:
            tile_size: Size of each tile (width and height in pixels)
            overlap: Overlap ratio between adjacent tiles [0, 1).
                    overlap=0.2 means 20% overlap, so stride = tile_size * 0.8
            min_box_area: Minimum area (in pixels²) for a box to be kept in a tile.
                         Boxes smaller than this are filtered out.
            min_overlap_ratio: Minimum ratio of box area that must overlap with tile
                              to keep the box. Range [0, 1].
                              - 0.0: Keep box if any part overlaps (default)
                              - 0.5: Keep box only if >= 50% of its area is in tile
                              - 1.0: Keep box only if fully inside tile
            edge_handling: Strategy for boxes crossing tile boundaries:
                          - "clip": Keep boxes that overlap tile (default)
                          - "ignore": Only keep boxes fully inside tile
                          - "keep": Same as "clip" (kept for backward compatibility)
        
        Example:
            >>> tiler = ImageTiler(
            ...     tile_size=1024,
            ...     overlap=0.2,  # 20% overlap between tiles
            ...     min_box_area=64,  # Filter boxes < 64 pixels²
            ...     min_overlap_ratio=0.3,  # Keep box if >= 30% overlaps tile
            ... )
        """
        if tile_size <= 0:
            raise ValueError("Tile size must be positive.")
        if not 0 <= overlap < 1:
            raise ValueError("Overlap must be in [0, 1).")
        if not 0 <= min_overlap_ratio <= 1:
            raise ValueError("min_overlap_ratio must be in [0, 1].")
        if edge_handling not in {"clip", "ignore", "keep"}:
            raise ValueError(f"Unknown edge_handling: {edge_handling}")
        
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_box_area = min_box_area
        self.min_overlap_ratio = min_overlap_ratio
        self.edge_handling = edge_handling
        self.stride = int(tile_size * (1 - overlap))
    
    def generate_tiles(
        self,
        image_width: int,
        image_height: int,
        *,
        prefix: str = "tile"
    ) -> List[Tile]:
        """Generate tile grid for an image of given dimensions."""
        tiles = []
        tile_id = 0
        
        y = 0
        while y < image_height:
            x = 0
            while x < image_width:
                width = min(self.tile_size, image_width - x)
                height = min(self.tile_size, image_height - y)
                
                tiles.append(Tile(
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    tile_id=f"{prefix}_{tile_id:04d}"
                ))
                
                tile_id += 1
                x += self.stride
                if x >= image_width:
                    break
            
            y += self.stride
            if y >= image_height:
                break
        
        return tiles
    
    def assign_boxes_to_tile(
        self,
        tile: Tile,
        rboxes: Sequence[RBox],
        class_names: Sequence[str],
        difficult_flags: Optional[Sequence[int]] = None,
    ) -> TiledSample:
        """Assign boxes to a tile, applying edge handling and overlap thresholds.
        
        Args:
            tile: Target tile
            rboxes: Sequence of RBox annotations
            class_names: Class names corresponding to rboxes
            difficult_flags: Optional difficulty flags
        
        Returns:
            TiledSample with boxes assigned to the tile
        """
        if difficult_flags is None:
            difficult_flags = [0] * len(rboxes)
        
        if len(class_names) != len(rboxes):
            raise ValueError("class_names and rboxes must have same length.")
        if len(difficult_flags) != len(rboxes):
            raise ValueError("difficult_flags and rboxes must have same length.")
        
        assigned_boxes = []
        assigned_classes = []
        assigned_difficult = []
        
        # Create tile polygon for overlap calculation
        tile_poly = Polygon.rectangle(
            tile.x + tile.width / 2.0,
            tile.y + tile.height / 2.0,
            tile.width,
            tile.height
        )
        
        for rbox, cls_name, diff in zip(rboxes, class_names, difficult_flags):
            rbox_poly = rbox.to_polygon()
            
            # Check overlap based on edge handling strategy
            if self.edge_handling == "ignore":
                # Only keep boxes fully inside tile
                corners = rbox.corners()
                if not all(tile.contains_point(x, y) for x, y in corners):
                    continue
                clipped = rbox
            else:  # "clip" or "keep"
                # Check if box overlaps tile
                clipped = tile.clip_box(rbox)
                if clipped is None:
                    continue
                
                # Check minimum overlap ratio if specified
                if self.min_overlap_ratio > 0:
                    # Compute intersection area
                    from ..ops.utils import polygon_intersection_area
                    intersection_area = polygon_intersection_area(rbox_poly, tile_poly)
                    box_area = rbox.area
                    
                    if box_area > 0:
                        overlap_ratio = intersection_area / box_area
                        if overlap_ratio < self.min_overlap_ratio:
                            continue
            
            # Translate to tile-local coordinates
            local_rbox = RBox(
                cx=clipped.cx - tile.x,
                cy=clipped.cy - tile.y,
                width=clipped.width,
                height=clipped.height,
                angle=clipped.angle
            )
            
            # Filter by minimum area
            if local_rbox.area >= self.min_box_area:
                assigned_boxes.append(local_rbox)
                assigned_classes.append(cls_name)
                assigned_difficult.append(diff)
        
        return TiledSample(
            tile=tile,
            image_path=Path(""),  # Will be set by caller
            annotations=tuple(assigned_boxes),
            class_names=tuple(assigned_classes),
            difficult_flags=tuple(assigned_difficult)
        )
    
    def tile_image(
        self,
        image_path: Path,
        image_width: int,
        image_height: int,
        rboxes: Sequence[RBox],
        class_names: Sequence[str],
        difficult_flags: Optional[Sequence[int]] = None,
    ) -> Iterator[TiledSample]:
        """Generate tiled samples from an image and its annotations."""
        tiles = self.generate_tiles(image_width, image_height)
        
        for tile in tiles:
            tiled = self.assign_boxes_to_tile(
                tile=tile,
                rboxes=rboxes,
                class_names=class_names,
                difficult_flags=difficult_flags,
            )
            # Set image path
            tiled = TiledSample(
                tile=tiled.tile,
                image_path=image_path,
                annotations=tiled.annotations,
                class_names=tiled.class_names,
                difficult_flags=tiled.difficult_flags,
            )
            yield tiled


def visualize_tiles(
    image_path: Path,
    image_width: int,
    image_height: int,
    tiles: Sequence[Tile],
    rboxes: Optional[Sequence[RBox]] = None,
    class_names: Optional[Sequence[str]] = None,
    *,
    output_path: Optional[Path] = None,
    tile_color: Tuple[int, int, int] = (255, 0, 0),
    box_color: Tuple[int, int, int] = (0, 255, 0),
    tile_width: int = 2,
    box_width: int = 2,
) -> any:  # type: ignore
    """Visualize tiles and annotations for debugging tiling operations.
    
    This function helps debug tiling by drawing:
    - Tile boundaries in red (by default)
    - Annotations in green (by default)
    - Tile IDs as labels
    
    Args:
        image_path: Path to the image file
        image_width: Width of the image
        image_height: Height of the image
        tiles: Sequence of Tile objects to visualize
        rboxes: Optional sequence of RBox annotations
        class_names: Optional class names for annotations
        output_path: Optional path to save visualization
        tile_color: RGB color for tile boundaries (default: red)
        box_color: RGB color for annotation boxes (default: green)
        tile_width: Line width for tile boundaries
        box_width: Line width for annotation boxes
    
    Returns:
        PIL Image with visualization (or numpy array if PIL unavailable)
    
    Example:
        >>> tiler = ImageTiler(tile_size=512, overlap=0.2)
        >>> tiles = tiler.generate_tiles(2000, 2000)
        >>> visualize_tiles(
        ...     image_path=Path("image.png"),
        ...     image_width=2000,
        ...     image_height=2000,
        ...     tiles=tiles,
        ...     rboxes=annotations,
        ...     output_path=Path("tiles_vis.png"),
        ... )
    """
    try:
        from PIL import Image
        from ..utils import viz
    except ImportError:
        raise RuntimeError("PIL/Pillow is required for tile visualization.")
    
    # Load or create image
    if image_path.exists():
        image = Image.open(image_path).convert("RGB")
    else:
        # Create blank image for visualization
        image = Image.new("RGB", (image_width, image_height), color="white")
    
    # Draw tile boundaries
    tile_polygons = []
    tile_specs = []
    for tile in tiles:
        tile_poly = Polygon.rectangle(
            tile.x + tile.width / 2.0,
            tile.y + tile.height / 2.0,
            tile.width,
            tile.height
        )
        tile_polygons.append(tile_poly.points)
        tile_specs.append(viz.DrawingSpec(outline=tile_color, width=tile_width))
    
    # Draw tiles
    image = viz.draw_polygons(
        image,
        tile_polygons,
        specs=tile_specs,
    )
    
    # Draw annotations if provided
    if rboxes:
        annotation_polygons = [rbox.to_polygon().points for rbox in rboxes]
        annotation_specs = [
            viz.DrawingSpec(outline=box_color, width=box_width)
            for _ in annotation_polygons
        ]
        image = viz.draw_polygons(
            image,
            annotation_polygons,
            specs=annotation_specs,
        )
    
    if output_path:
        image.save(output_path)
    
    return image


__all__ = [
    "Tile",
    "TiledSample",
    "ImageTiler",
    "visualize_tiles",
]

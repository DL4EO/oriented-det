#!/usr/bin/env python3
"""
DOTA Image Tiling Tool

This script tiles large DOTA format images into smaller patches with configurable
tile size and overlap. It processes both images and their corresponding annotations,
computing intersections and creating minimum rotated rectangles for truncated objects.

The tiler handles:
- Large aerial/satellite images with varying dimensions
- Oriented bounding box annotations in DOTA format
- Configurable tile size and overlap
- Minimum overlap ratio filtering (keeps objects with >= X% overlap with tile)
- Automatic padding for images smaller than tile size
- By default, last row/column of tiles are aligned on the image edge (no right/bottom zero-padding).
  Use ``--pad-edge-tiles`` for the legacy behavior (stride-only grid; last tiles may extend past the image).

Output Format:
- Tiled images: {original_name}_{x_start}_{y_start}.png
- Tiled annotations: {original_name}_{x_start}_{y_start}.txt (official DOTA: comma-separated)
"""

import argparse
import os
import sys
import itertools
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# Ensure oriented_det is importable when run as script (e.g. python tools/tile_dota.py ...)
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from oriented_det.data import format_dota_line
import PIL.Image
from tqdm import tqdm
from shapely import geometry, affinity


def read_annotations(path: Path) -> List[Tuple[List[List[float]], str, str]]:
    """
    Read DOTA format annotation file.
    
    Accepts official DOTA comma-separated or space-separated input.
    Output is always official DOTA format (comma-separated).
    When parsing, exactly 10 fields are expected (single-token category).
    
    Args:
        path: Path to annotation .txt file
        
    Returns:
        List of (coordinates, label, difficulty) tuples
        coordinates: [[[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x1, y1]]]
    """
    with open(path) as file:
        annotations = file.read()

    records = []
    for line in annotations.split('\n'):
        # Skip metadata lines
        if 'imagesource:' in line or 'gsd' in line or not line.strip():
            continue

        try:
            # Support official DOTA (comma-separated) or space-separated
            if "," in line:
                parts = [p.strip() for p in line.split(",")]
            else:
                parts = line.split()
            if len(parts) < 10:
                continue
            x1, y1, x2, y2, x3, y3, x4, y4, label, difficulty = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6], parts[7], parts[8], parts[9]
            coordinates = [[[float(x1), float(y1)],
                           [float(x2), float(y2)],
                           [float(x3), float(y3)],
                           [float(x4), float(y4)],
                           [float(x1), float(y1)]]]
            record = (coordinates, label.strip(), difficulty)
            records.append(record)
        except Exception as e:
            print(f"Error reading file: {path}")
            raise e

    return records


def coords_in_tile(
    coords: List[List[float]], 
    x_start: int, 
    y_start: int, 
    width: int, 
    height: int, 
    truncated_percent: float
) -> Optional[List[float]]:
    """
    Check if annotation should be kept in tile and compute its intersection.
    
    Args:
        coords: Polygon coordinates [[x1, y1], [x2, y2], ...]
        x_start: Tile X start position
        y_start: Tile Y start position
        width: Tile width
        height: Tile height
        truncated_percent: Minimum overlap ratio (0.0-1.0) to keep object
        
    Returns:
        Flattened coordinates [x1, y1, x2, y2, x3, y3, x4, y4] of minimum rotated
        rectangle if object overlaps enough, None otherwise
    """
    # Convert coords into Shapely Polygon
    poly_shape = geometry.Polygon(coords)
    
    # Shift Polygon by x_start and y_start (to tile coordinates)
    poly_shape = affinity.translate(poly_shape, xoff=-x_start, yoff=-y_start)
    
    # Tile shape (bounding box)
    cell_shape = geometry.box(0, 0, width, height)

    # Compute intersection
    intersect = poly_shape.intersection(cell_shape)
    
    # Filter by overlap ratio
    if intersect.is_empty or intersect.area < truncated_percent * poly_shape.area:
        return None
    
    # Get oriented bounding box (minimum rotated rectangle)
    intersect = intersect.minimum_rotated_rectangle
    
    # Pop last element to keep only 4 vertices
    coords = intersect.exterior.coords[0:4]
    
    # Flatten the array of arrays into a single array
    coords = list(itertools.chain(*coords))
    return coords


def tile_dota_images(
    data_dir: Path,
    tile_width: int = 1024,
    tile_height: int = 1024,
    tile_overlap: int = 200,
    truncated_percent: float = 0.7,
    overwrite_files: bool = False,
    no_no_data: bool = True,
):
    """
    Tile DOTA images and annotations.
    
    Args:
        data_dir: Root directory containing 'images/' and 'labels/' subdirectories
        tile_width: Width of each tile in pixels
        tile_height: Height of each tile in pixels
        tile_overlap: Overlap between adjacent tiles in pixels
        truncated_percent: Minimum overlap ratio to keep objects (0.0-1.0)
        overwrite_files: If True, overwrite existing tiles
        no_no_data: If True (default), last row/column of tile windows are shifted so the
            right/bottom edges align with the image (no zero-padding on those edges).
            If False, only the stride grid is used; last tiles may extend past the image
            and are zero-padded (legacy).
    """
    # To avoid DecompressionBombError for large images
    PIL.Image.MAX_IMAGE_PIXELS = None

    # Get image list
    img_list = list((data_dir / 'images').glob('*.png'))
    if not img_list:
        print(f"Warning: No PNG images found in {data_dir / 'images'}")
        return
    
    print(f"Found {len(img_list)} images to process")

    # Create output directories
    tiles_path = data_dir / f'tiles_{tile_width}'
    (tiles_path / 'images').mkdir(parents=True, exist_ok=True)
    (tiles_path / 'labels').mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {tiles_path}")
    print(f"Tile size: {tile_width}x{tile_height}")
    print(f"Tile overlap: {tile_overlap} pixels")
    print(f"Minimum overlap ratio: {truncated_percent:.1%}")
    print(f"Flush last row/column to image edge (no right/bottom pad on edge tiles): {no_no_data}")
    print()

    # Process each image
    for img_path in tqdm(img_list, desc="Tiling images"):
        # Open image
        pil_img = PIL.Image.open(str(img_path), mode='r')
        pil_img = pil_img.convert('RGB')
        image_width, image_height = pil_img.size
        np_img = np.array(pil_img, dtype=np.uint8)

        # Pad image if smaller than tile size
        if image_width < tile_width or image_height < tile_height:
            new_width = max(image_width, tile_width)
            new_height = max(image_height, tile_height)
            new_img = np.zeros(shape=(new_height, new_width, 3), dtype=np.uint8)
            new_img[0:image_height, 0:image_width, :] = np_img[0:image_height, 0:image_width, :]
            np_img, image_width, image_height = new_img, new_width, new_height

        # Get annotations for image
        label_path = str(img_path).replace('images', 'labels').replace('.png', '.txt')
        if not os.path.exists(label_path):
            print(f"Warning: No label file found for {img_path.name}")
            img_labels = []
        else:
            img_labels = read_annotations(label_path)

        # Calculate stride (distance between tile starts)
        x_stride = tile_width - tile_overlap
        y_stride = tile_height - tile_overlap

        # Generate tile positions to avoid duplicates
        # All tiles must be exactly tile_width x tile_height
        # Edge tiles may extend beyond image boundary (will be padded with zeros)
        x_positions = []
        if image_width <= tile_width:
            # Image fits in one tile - still create full-size tile
            x_positions = [(0, tile_width)]
        else:
            x = 0
            while x < image_width:
                x_start = x
                # Always create full-size tiles (may extend beyond image)
                x_end = x_start + tile_width
                x_positions.append((x_start, x_end))
                x += x_stride
                # Stop when we've covered the entire image
                if x >= image_width:
                    break
            # Remove duplicate positions
            x_positions = list(dict.fromkeys(x_positions))  # Preserves order

            # Optionally make last column end exactly on image edge (no zero-padding)
            if no_no_data and image_width > tile_width:
                new_last = (max(0, image_width - tile_width), image_width)
                x_positions[-1] = new_last
                x_positions = list(dict.fromkeys(x_positions))

        y_positions = []
        if image_height <= tile_height:
            # Image fits in one tile - still create full-size tile
            y_positions = [(0, tile_height)]
        else:
            y = 0
            while y < image_height:
                y_start = y
                # Always create full-size tiles (may extend beyond image)
                y_end = y_start + tile_height
                y_positions.append((y_start, y_end))
                y += y_stride
                # Stop when we've covered the entire image
                if y >= image_height:
                    break
            # Remove duplicate positions
            y_positions = list(dict.fromkeys(y_positions))  # Preserves order

            # Optionally make last row end exactly on image edge (no zero-padding)
            if no_no_data and image_height > tile_height:
                new_last = (max(0, image_height - tile_height), image_height)
                y_positions[-1] = new_last
                y_positions = list(dict.fromkeys(y_positions))

        # Cut each tile
        for x_start, x_end in x_positions:
            for y_start, y_end in y_positions:
                
                # Generate tile filename
                tile_id = img_path.stem + "_" + str(x_start) + "_" + str(y_start) + img_path.suffix
                save_tile_path = tiles_path / 'images' / tile_id
                
                # Save tile image if needed
                if overwrite_files or not save_tile_path.exists():
                    # Always create full-size tile (1024x1024)
                    cut_tile = np.zeros(shape=(tile_height, tile_width, 3), dtype=np.uint8)
                    # Extract image portion, clamping to image boundaries
                    # Tiles may extend beyond image, which will be padded with zeros
                    img_y_start = max(0, y_start)
                    img_y_end = min(image_height, y_end)
                    img_x_start = max(0, x_start)
                    img_x_end = min(image_width, x_end)
                    
                    # Calculate offsets in the tile for the extracted portion
                    tile_y_offset = img_y_start - y_start
                    tile_x_offset = img_x_start - x_start
                    tile_y_size = img_y_end - img_y_start
                    tile_x_size = img_x_end - img_x_start
                    
                    # Copy image portion to tile (rest remains zeros)
                    if tile_y_size > 0 and tile_x_size > 0:
                        cut_tile[tile_y_offset:tile_y_offset+tile_y_size, 
                                tile_x_offset:tile_x_offset+tile_x_size, :] = \
                            np_img[img_y_start:img_y_end, img_x_start:img_x_end, :]
                    
                    cut_tile_img = PIL.Image.fromarray(cut_tile, "RGB")
                    cut_tile_img.save(save_tile_path)

                # Process annotations for this tile
                rows = []
                for (coordinates, label, difficulty) in img_labels:
                    intersection = coords_in_tile(
                        coordinates[0], x_start, y_start, 
                        tile_width, tile_height, truncated_percent
                    )
                    if intersection is not None:
                        rows.append((*intersection, label, difficulty))

                # Save tile annotations (official DOTA format: comma-separated)
                save_label_path = (tiles_path / 'labels' / tile_id).with_suffix('.txt')
                with open(save_label_path, 'w') as out:
                    for row in rows:
                        out.write(format_dota_line(*row) + '\n')

    print(f"\nTiling complete! Output saved to: {tiles_path}")
    
    # Print statistics
    num_tile_images = len(list((tiles_path / 'images').glob('*.png')))
    num_tile_labels = len(list((tiles_path / 'labels').glob('*.txt')))
    print(f"Generated {num_tile_images} tile images and {num_tile_labels} label files")


def main():
    parser = argparse.ArgumentParser(
        description="Tile DOTA format images and annotations into smaller patches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python tile_dota.py /path/to/dota/train

  # Custom tile size and overlap
  python tile_dota.py /path/to/dota/train --tile-size 512 --overlap 128

  # Adjust minimum overlap ratio for keeping objects
  python tile_dota.py /path/to/dota/train --min-overlap 0.5

  # Overwrite existing tiles
  python tile_dota.py /path/to/dota/train --overwrite

  # Legacy: allow zero-padding on right/bottom of last row/column of tiles
  python tile_dota.py /path/to/dota/train --pad-edge-tiles

Input Structure:
  data_dir/
    images/
      image001.png
      image002.png
      ...
    labels/
      image001.txt
      image002.txt
      ...

Output Structure:
  data_dir/tiles_{size}/
    images/
      image001_0_0.png
      image001_960_0.png
      ...
    labels/
      image001_0_0.txt
      image001_960_0.txt
      ...
        """
    )
    
    parser.add_argument(
        'data_dir',
        type=Path,
        help='Root directory containing images/ and labels/ subdirectories'
    )
    parser.add_argument(
        '--tile-size',
        type=int,
        default=1024,
        help='Tile width and height in pixels (default: 1024)'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        default=200,
        help='Overlap between adjacent tiles in pixels (default: 200)'
    )
    parser.add_argument(
        '--min-overlap',
        type=float,
        default=0.7,
        help='Minimum overlap ratio (0.0-1.0) to keep objects (default: 0.7, MMRotate iof_thr)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing tile files'
    )
    parser.add_argument(
        '--pad-edge-tiles',
        action='store_true',
        help='Allow zero-padding on the right/bottom of the last row/column of tiles '
             '(legacy stride-only grid). Default aligns those tiles on the image edge.',
    )
    parser.add_argument(
        '--no-no-data',
        action='store_true',
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()
    
    # Validate arguments
    if not args.data_dir.exists():
        parser.error(f"Data directory does not exist: {args.data_dir}")
    
    if not (args.data_dir / 'images').exists():
        parser.error(f"Images directory not found: {args.data_dir / 'images'}")
    
    if not (args.data_dir / 'labels').exists():
        print(f"Warning: Labels directory not found: {args.data_dir / 'labels'}")
        print("Proceeding with image tiling only...")
    
    if args.tile_size <= 0:
        parser.error("Tile size must be positive")
    
    if args.overlap < 0 or args.overlap >= args.tile_size:
        parser.error("Overlap must be >= 0 and < tile size")
    
    if args.min_overlap < 0.0 or args.min_overlap > 1.0:
        parser.error("Min overlap ratio must be between 0.0 and 1.0")
    
    # Flush edge tiles by default; --pad-edge-tiles restores legacy padding.
    no_no_data = not args.pad_edge_tiles

    # Run tiling
    tile_dota_images(
        data_dir=args.data_dir,
        tile_width=args.tile_size,
        tile_height=args.tile_size,
        tile_overlap=args.overlap,
        truncated_percent=args.min_overlap,
        overwrite_files=args.overwrite,
        no_no_data=no_no_data,
    )


if __name__ == '__main__':
    main()


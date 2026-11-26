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
- Optional ``--resize-rate`` (scale image + boxes before tiling; MMRotate MS rates 0.5 / 1.0 / 1.5)
- Minimum overlap ratio filtering (keeps objects with >= X% overlap with tile)
- Automatic padding for images smaller than tile size
- By default, last row/column of tiles are aligned on the image edge (no right/bottom zero-padding).
  Use ``--pad-edge-tiles`` for the legacy behavior (stride-only grid; last tiles may extend past the image).

Output Format:
- Tiled images: {original_name}_{x_start}_{y_start}.{png|jpg}
  Default ``auto``: JPEG sources → JPEG tiles, PNG (and TIFF/BMP) → PNG tiles.
  Override with ``--output-format png|jpg``.
- Tiled annotations: {original_name}_{x_start}_{y_start}.txt (official DOTA: comma-separated)
- Default folder: ``tiles_{size}`` at rate 1.0, else ``tiles_{size}_r{rate}``; or ``--output-dir``.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
import sys
from math import ceil
from pathlib import Path
from typing import List, Tuple, Optional

_SOURCE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
_JPEG_SUFFIXES = frozenset({".jpg", ".jpeg"})

import numpy as np

# Ensure oriented_det is importable when run as script (e.g. python tools/tile_dota.py ...)
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from oriented_det.data import format_dota_line
import PIL.Image
from tqdm import tqdm
from shapely import geometry, affinity


def compute_tile_starts(
    length: int,
    tile_size: int,
    tile_overlap: int,
    *,
    flush_to_edge: bool = True,
) -> List[int]:
    """Return tile origin coordinates along one image axis.

    Default ``flush_to_edge=True`` matches MMRotate ``img_split`` /
    Ultralytics ``get_windows``: ``ceil`` window count, last start snapped so the
    window lies inside the image (no right/bottom zero-pad on large images).

    ``flush_to_edge=False`` is the legacy stride walk: the last window may extend
    past ``length`` and is zero-padded when cropped (``--pad-edge-tiles``).

    Small / equal axes (``length <= tile_size``): a single start at ``0``. Callers
    pad the source canvas to ``tile_size`` before cropping so the emitted tile is
    always full size (zero fill; MMRotate uses RGB pad ``[104,116,124]`` on the
    crop instead — same layout, different fill).
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if tile_overlap < 0 or tile_overlap >= tile_size:
        raise ValueError(
            f"tile_overlap must be >= 0 and < tile_size, got {tile_overlap}"
        )
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")

    if length <= tile_size:
        return [0]

    stride = tile_size - tile_overlap
    if flush_to_edge:
        # MMRotate / Ultralytics: n = ceil((L - size) / step + 1), snap last.
        n = int(ceil((length - tile_size) / stride + 1))
        starts = [stride * i for i in range(n)]
        if len(starts) > 1 and starts[-1] + tile_size > length:
            starts[-1] = length - tile_size
        # Dedupe while preserving order (snap can collide with previous start).
        return list(dict.fromkeys(starts))

    # Legacy: keep walking by stride until the start covers the axis; last crop
    # may extend past ``length``.
    starts: List[int] = []
    x = 0
    while x < length:
        starts.append(x)
        x += stride
        if x >= length:
            break
    return list(dict.fromkeys(starts))


def resolve_tile_output_suffix(source_suffix: str, output_format: str = "auto") -> str:
    """Return ``.png`` or ``.jpg`` for tile files.

    ``auto`` keeps JPEG as JPEG and everything else as PNG (TIFF/BMP included).
    """
    fmt = (output_format or "auto").strip().lower()
    if fmt in {"jpg", "jpeg"}:
        return ".jpg"
    if fmt == "png":
        return ".png"
    if fmt != "auto":
        raise ValueError(
            f"Unsupported output_format={output_format!r}; expected auto, png, or jpg"
        )
    if str(source_suffix).lower() in _JPEG_SUFFIXES:
        return ".jpg"
    return ".png"


def collect_tile_source_images(image_dir: Path) -> List[Path]:
    """Sorted unique images under ``images/`` (png/jpg/jpeg/tif/tiff/bmp)."""
    image_dir = Path(image_dir)
    files: List[Path] = []
    for suffix in _SOURCE_SUFFIXES:
        files.extend(sorted(image_dir.glob(f"*{suffix}")))
        files.extend(sorted(image_dir.glob(f"*{suffix.upper()}")))
    seen: set[Path] = set()
    ordered: List[Path] = []
    for path in files:
        resolved = path.resolve()
        if path.is_file() and resolved not in seen:
            seen.add(resolved)
            ordered.append(path)
    return ordered


def _save_tile_image(
    image: PIL.Image.Image,
    path: Path,
    *,
    jpeg_quality: int = 95,
) -> None:
    if path.suffix.lower() in _JPEG_SUFFIXES:
        image.save(path, format="JPEG", quality=int(jpeg_quality), subsampling=0)
        return
    image.save(path)


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


def format_resize_rate(rate: float) -> str:
    """Stable dirname fragment for a resize rate (``0.5``, ``1``, ``1.5``)."""
    return f"{float(rate):g}"


def default_tiles_dirname(tile_width: int, resize_rate: float = 1.0) -> str:
    """Default output folder under ``data_dir`` (SS keeps ``tiles_{size}``)."""
    if abs(float(resize_rate) - 1.0) < 1e-9:
        return f"tiles_{tile_width}"
    return f"tiles_{tile_width}_r{format_resize_rate(resize_rate)}"


def scale_annotation_records(
    records: List[Tuple[List[List[float]], str, str]],
    resize_rate: float,
) -> List[Tuple[List[List[float]], str, str]]:
    """Scale DOTA polygon vertices by ``resize_rate`` (labels/difficulty unchanged)."""
    if abs(float(resize_rate) - 1.0) < 1e-9:
        return records
    rate = float(resize_rate)
    scaled: List[Tuple[List[List[float]], str, str]] = []
    for coordinates, label, difficulty in records:
        scaled_ring = [[float(x) * rate, float(y) * rate] for x, y in coordinates[0]]
        scaled.append(([scaled_ring], label, difficulty))
    return scaled


def tile_dota_images(
    data_dir: Path,
    tile_width: int = 1024,
    tile_height: int = 1024,
    tile_overlap: int = 200,
    truncated_percent: float = 0.7,
    overwrite_files: bool = False,
    no_no_data: bool = True,
    output_format: str = "auto",
    jpeg_quality: int = 95,
    resize_rate: float = 1.0,
    tiles_dirname: Optional[str] = None,
    output_dir: Optional[Path] = None,
    clean_output: bool = False,
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
        output_format: ``auto`` (JPEG→jpg, else png), or force ``png`` / ``jpg``.
        jpeg_quality: JPEG quality 1–100 when writing JPEG tiles (default 95).
        resize_rate: Scale each source image and its annotations by this factor before
            tiling (MMRotate / Ultralytics multiscale rate). ``1.0`` is single-scale.
            Rate ``0.5`` then tile 1024 is equivalent to cropping 2048 px then resizing
            to 1024; rate ``1.5`` equates to cropping ~682 px then resizing to 1024.
        tiles_dirname: Subfolder name under ``data_dir`` when ``output_dir`` is unset.
            Default: ``tiles_{size}`` at rate 1.0, else ``tiles_{size}_r{rate}``.
        output_dir: If set, write ``images/`` and ``labels/`` directly here (flat tile
            root). Useful for merging train+val into one multiscale tree per rate.
        clean_output: If True, delete ``images/`` and ``labels/`` under the tile root
            before writing. Use on the first split of a relaunch so orphan tiles from a
            prior window grid do not remain; do **not** pass it on the second split when
            merging train+val into the same ``output_dir``.
    """
    # To avoid DecompressionBombError for large images
    PIL.Image.MAX_IMAGE_PIXELS = None

    if resize_rate <= 0:
        raise ValueError(f"resize_rate must be positive, got {resize_rate}")

    image_dir = data_dir / "images"
    label_dir = data_dir / "labels"
    img_list = collect_tile_source_images(image_dir)
    if not img_list:
        print(f"Warning: No images found in {image_dir} (png/jpg/jpeg/tif/tiff/bmp)")
        return
    
    print(f"Found {len(img_list)} images to process")

    # Create output directories
    if output_dir is not None:
        tiles_path = Path(output_dir)
    else:
        dirname = tiles_dirname or default_tiles_dirname(tile_width, resize_rate)
        tiles_path = data_dir / dirname
    if clean_output and tiles_path.exists():
        for sub in ("images", "labels"):
            target = tiles_path / sub
            if target.exists():
                print(f"Cleaning output: removing {target}")
                shutil.rmtree(target)
    (tiles_path / 'images').mkdir(parents=True, exist_ok=True)
    (tiles_path / 'labels').mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {tiles_path}")
    print(f"Tile size: {tile_width}x{tile_height}")
    print(f"Tile overlap: {tile_overlap} pixels")
    print(f"Resize rate: {resize_rate:g}")
    print(f"Minimum overlap ratio: {truncated_percent:.1%}")
    print(f"Flush last row/column to image edge (no right/bottom pad on edge tiles): {no_no_data}")
    print(f"Clean output before write: {clean_output}")
    print(f"Output format: {output_format} (JPEG quality {jpeg_quality})")
    print()

    # Process each image
    for img_path in tqdm(img_list, desc="Tiling images"):
        out_suffix = resolve_tile_output_suffix(img_path.suffix, output_format)
        # Open image
        pil_img = PIL.Image.open(str(img_path), mode='r')
        pil_img = pil_img.convert('RGB')
        if abs(float(resize_rate) - 1.0) >= 1e-9:
            src_w, src_h = pil_img.size
            new_w = max(1, int(round(src_w * float(resize_rate))))
            new_h = max(1, int(round(src_h * float(resize_rate))))
            # Bilinear matches typical DOTA MS prep; boxes are scaled in annotation space.
            pil_img = pil_img.resize((new_w, new_h), resample=PIL.Image.BILINEAR)
        image_width, image_height = pil_img.size
        np_img = np.array(pil_img, dtype=np.uint8)

        # Pad image if smaller than tile size (top-left content, zero fill).
        # After resize_rate, a short side can fall below 1024 (common at rate 0.5).
        # MMRotate/Ultralytics instead crop a (1024/rate) window on the original and
        # pad that crop; train-time resize to 1024 yields the same content scale.
        if image_width < tile_width or image_height < tile_height:
            new_width = max(image_width, tile_width)
            new_height = max(image_height, tile_height)
            new_img = np.zeros(shape=(new_height, new_width, 3), dtype=np.uint8)
            new_img[0:image_height, 0:image_width, :] = np_img[0:image_height, 0:image_width, :]
            np_img, image_width, image_height = new_img, new_width, new_height

        # Get annotations for image
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.is_file():
            print(f"Warning: No label file found for {img_path.name}")
            img_labels = []
        else:
            img_labels = scale_annotation_records(
                read_annotations(label_path), resize_rate
            )

        # Window starts: MMRotate/Ultralytics ceil+snap when no_no_data (default).
        x_starts = compute_tile_starts(
            image_width, tile_width, tile_overlap, flush_to_edge=no_no_data
        )
        y_starts = compute_tile_starts(
            image_height, tile_height, tile_overlap, flush_to_edge=no_no_data
        )
        x_positions = [(x, x + tile_width) for x in x_starts]
        y_positions = [(y, y + tile_height) for y in y_starts]

        # Cut each tile
        for x_start, x_end in x_positions:
            for y_start, y_end in y_positions:
                
                # Generate tile filename
                tile_id = f"{img_path.stem}_{x_start}_{y_start}{out_suffix}"
                save_tile_path = tiles_path / "images" / tile_id
                
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
                    _save_tile_image(
                        cut_tile_img, save_tile_path, jpeg_quality=jpeg_quality
                    )

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
    num_tile_images = len(collect_tile_source_images(tiles_path / "images"))
    num_tile_labels = len(list((tiles_path / "labels").glob("*.txt")))
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

  # Multiscale train tiles (MMRotate-style): resize then tile 1024 / overlap 500
  python tile_dota.py /path/to/dota/train --overlap 500 --resize-rate 0.5
  python tile_dota.py /path/to/dota/train --overlap 500 --resize-rate 1.0 \\
      --tiles-dirname tiles_1024_r1
  python tile_dota.py /path/to/dota/train --overlap 500 --resize-rate 1.5

  # Merge train+val into one flat root per rate (Hub pretrain convention).
  # --clean-output on the first split only so a relaunch drops orphan tiles.
  python tile_dota.py /path/to/dota/train --overlap 500 --resize-rate 0.5 \\
      --output-dir /path/to/data/DOTA-v1.0-tiled-ms/r0.5 --clean-output
  python tile_dota.py /path/to/dota/val --overlap 500 --resize-rate 0.5 \\
      --output-dir /path/to/data/DOTA-v1.0-tiled-ms/r0.5

  # Overwrite existing tiles
  python tile_dota.py /path/to/dota/train --overwrite

  # Force JPEG tiles even from PNG sources (or png from JPEG)
  python tile_dota.py /path/to/dota/train --output-format jpg --jpeg-quality 95

  # Legacy: allow zero-padding on right/bottom of last row/column of tiles
  python tile_dota.py /path/to/dota/train --pad-edge-tiles

Input Structure:
  data_dir/
    images/
      image001.png
      image002.jpg
      ...
    labels/
      image001.txt
      image002.txt
      ...

Output Structure:
  data_dir/tiles_{size}/            # resize-rate 1.0 (default)
  data_dir/tiles_{size}_r{rate}/    # resize-rate != 1.0
  # or --output-dir <root>/ with images/ + labels/ directly
    images/
      image001_0_0.png
      image002_0_0.jpg
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
        '--resize-rate',
        type=float,
        default=1.0,
        help='Scale source image + annotations before tiling (default: 1.0). '
             'MMRotate MS rates are typically 0.5 / 1.0 / 1.5 with --overlap 500.',
    )
    parser.add_argument(
        '--tiles-dirname',
        type=str,
        default=None,
        help='Output subfolder under data_dir when --output-dir is unset '
             '(default: tiles_{size} at rate 1.0, else tiles_{size}_r{rate}). '
             'Use this for rate 1.0 MS trees so they do not collide with SS tiles_1024.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Write images/ and labels/ directly to this directory (flat tile root). '
             'Overrides --tiles-dirname. Merge train+val by pointing both runs here.',
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
        '--clean-output',
        action='store_true',
        help='Delete images/ and labels/ under the tile output root before writing. '
             'Use when relaunching so orphan tiles from a prior grid are removed. '
             'When merging train+val into one --output-dir, pass this only on the first split.',
    )
    parser.add_argument(
        '--output-format',
        type=str,
        default='auto',
        choices=('auto', 'png', 'jpg', 'jpeg'),
        help='Tile image format: auto (JPEG→jpg, else png), or force png/jpg (default: auto)',
    )
    parser.add_argument(
        '--jpeg-quality',
        type=int,
        default=95,
        help='JPEG quality 1-100 when writing JPEG tiles (default: 95)',
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

    if args.resize_rate <= 0:
        parser.error("Resize rate must be positive")

    if args.jpeg_quality < 1 or args.jpeg_quality > 100:
        parser.error("JPEG quality must be between 1 and 100")
    
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
        output_format=args.output_format,
        jpeg_quality=args.jpeg_quality,
        resize_rate=args.resize_rate,
        tiles_dirname=args.tiles_dirname,
        output_dir=args.output_dir,
        clean_output=args.clean_output,
    )


if __name__ == '__main__':
    main()


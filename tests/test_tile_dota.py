"""Tests for tools/tile_dota.py JPEG/PNG source and output formats."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from PIL import Image

_TILE_PATH = Path(__file__).resolve().parents[1] / "tools" / "tile_dota.py"
_spec = importlib.util.spec_from_file_location("tile_dota", _TILE_PATH)
assert _spec and _spec.loader
_tile_dota = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tile_dota)


def test_resolve_tile_output_suffix_auto_and_force():
    resolve = _tile_dota.resolve_tile_output_suffix
    assert resolve(".jpg", "auto") == ".jpg"
    assert resolve(".jpeg", "auto") == ".jpg"
    assert resolve(".png", "auto") == ".png"
    assert resolve(".tif", "auto") == ".png"
    assert resolve(".jpg", "png") == ".png"
    assert resolve(".png", "jpg") == ".jpg"


def _write_mini_split(root: Path, *, suffix: str) -> Path:
    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (48, 48), color=(10, 20, 30)).save(images / f"img{suffix}")
    labels.joinpath("img.txt").write_text(
        "5, 5, 20, 5, 20, 20, 5, 20, plane, 0\n",
        encoding="utf-8",
    )
    return root


def test_tile_dota_keeps_jpeg_and_png(tmp_path: Path):
    jpeg_root = _write_mini_split(tmp_path / "jpeg", suffix=".jpg")
    png_root = _write_mini_split(tmp_path / "png", suffix=".png")

    _tile_dota.tile_dota_images(
        jpeg_root, tile_width=32, tile_height=32, tile_overlap=8
    )
    _tile_dota.tile_dota_images(
        png_root, tile_width=32, tile_height=32, tile_overlap=8
    )

    jpeg_tiles = list((jpeg_root / "tiles_32" / "images").glob("*.jpg"))
    png_tiles = list((png_root / "tiles_32" / "images").glob("*.png"))
    assert jpeg_tiles
    assert png_tiles
    assert not list((jpeg_root / "tiles_32" / "images").glob("*.png"))
    assert not list((png_root / "tiles_32" / "images").glob("*.jpg"))


def test_tile_dota_output_format_override(tmp_path: Path):
    root = _write_mini_split(tmp_path / "jpg", suffix=".jpg")
    _tile_dota.tile_dota_images(
        root,
        tile_width=32,
        tile_height=32,
        tile_overlap=8,
        output_format="png",
    )
    tiles = list((root / "tiles_32" / "images").glob("*.png"))
    assert tiles
    assert not list((root / "tiles_32" / "images").glob("*.jpg"))


def test_default_tiles_dirname_ss_vs_ms():
    assert _tile_dota.default_tiles_dirname(1024, 1.0) == "tiles_1024"
    assert _tile_dota.default_tiles_dirname(1024, 0.5) == "tiles_1024_r0.5"
    assert _tile_dota.default_tiles_dirname(1024, 1.5) == "tiles_1024_r1.5"


def test_tile_dota_resize_rate_scales_boxes_and_dirname(tmp_path: Path):
    root = _write_mini_split(tmp_path / "png", suffix=".png")
    # 48x48 source at rate 0.5 → 24x24; pad to 32x32 tile.
    _tile_dota.tile_dota_images(
        root,
        tile_width=32,
        tile_height=32,
        tile_overlap=8,
        resize_rate=0.5,
    )
    tiles_root = root / "tiles_32_r0.5"
    tile_imgs = list((tiles_root / "images").glob("*.png"))
    assert tile_imgs
    label_files = list((tiles_root / "labels").glob("*.txt"))
    assert label_files
    label = label_files[0].read_text(encoding="utf-8").strip()
    assert label
    parts = [p.strip() for p in label.split(",")]
    # Original box [5,5]–[20,20] → half → [2.5,2.5]–[10,10]; still inside the tile.
    xs = [float(parts[i]) for i in range(0, 8, 2)]
    ys = [float(parts[i]) for i in range(1, 8, 2)]
    assert min(xs) < 5.0
    assert max(xs) < 15.0
    assert min(ys) < 5.0
    assert max(ys) < 15.0


def test_tile_dota_output_dir_flat_root(tmp_path: Path):
    root = _write_mini_split(tmp_path / "src", suffix=".png")
    out = tmp_path / "flat_ms"
    _tile_dota.tile_dota_images(
        root,
        tile_width=32,
        tile_height=32,
        tile_overlap=8,
        resize_rate=1.0,
        output_dir=out,
    )
    assert list((out / "images").glob("*.png"))
    assert list((out / "labels").glob("*.txt"))
    assert not (root / "tiles_32").exists()


def test_compute_tile_starts_small_equal_and_large():
    starts = _tile_dota.compute_tile_starts
    # Smaller or equal → single origin (caller pads canvas to tile_size).
    assert starts(800, 1024, 500, flush_to_edge=True) == [0]
    assert starts(1024, 1024, 500, flush_to_edge=True) == [0]
    # Larger: MMRotate/Ultralytics ceil count + snap last inside the image.
    assert starts(1025, 1024, 500, flush_to_edge=True) == [0, 1]
    assert starts(1500, 1024, 500, flush_to_edge=True) == [0, 476]
    assert starts(3000, 1024, 500, flush_to_edge=True) == [0, 524, 1048, 1572, 1976]
    # No window may extend past the axis when flush_to_edge.
    for length in (1500, 3000, 4000, 10000):
        for s in starts(length, 1024, 500, flush_to_edge=True):
            assert s >= 0
            assert s + 1024 <= length
    # Legacy stride walk keeps an overshooting last window.
    legacy = starts(1500, 1024, 500, flush_to_edge=False)
    assert legacy[0] == 0
    assert legacy[-1] + 1024 > 1500


def _write_sized_split(
    root: Path, *, width: int, height: int, box: str = "5, 5, 20, 5, 20, 20, 5, 20, plane, 0\n"
) -> Path:
    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(images / "img.png")
    labels.joinpath("img.txt").write_text(box, encoding="utf-8")
    return root


def test_tile_dota_image_smaller_than_tile_pads_one_tile(tmp_path: Path):
    root = _write_sized_split(tmp_path / "small", width=40, height=30)
    _tile_dota.tile_dota_images(
        root, tile_width=64, tile_height=64, tile_overlap=16
    )
    tiles = list((root / "tiles_64" / "images").glob("*.png"))
    assert len(tiles) == 1
    assert Image.open(tiles[0]).size == (64, 64)
    # Content top-left; right/bottom are zero pad.
    arr = __import__("numpy").array(Image.open(tiles[0]))
    assert arr[0, 0, 0] == 10
    assert arr[63, 63, 0] == 0


def test_tile_dota_image_larger_flush_no_overshoot(tmp_path: Path):
    # 100x48 with tile 32 / overlap 8 → stride 24; flush last inside.
    root = _write_sized_split(tmp_path / "big", width=100, height=48)
    _tile_dota.tile_dota_images(
        root, tile_width=32, tile_height=32, tile_overlap=8
    )
    names = sorted(p.stem for p in (root / "tiles_32" / "images").glob("*.png"))
    # x starts: ceil((100-32)/24+1)=ceil(3.833)=4 → [0,24,48,68] (last snapped)
    # y starts: ceil((48-32)/24+1)=ceil(1.666)=2 → [0,16]
    xs = sorted({int(n.split("_")[1]) for n in names})
    ys = sorted({int(n.split("_")[2]) for n in names})
    assert xs == [0, 24, 48, 68]
    assert ys == [0, 16]
    assert max(xs) + 32 == 100
    assert max(ys) + 32 == 48
    for p in (root / "tiles_32" / "images").glob("*.png"):
        assert Image.open(p).size == (32, 32)


def test_tile_dota_clean_output_removes_orphans(tmp_path: Path):
    root = _write_sized_split(tmp_path / "src", width=40, height=40)
    out = tmp_path / "flat"
    orphan_img = out / "images" / "orphan_999_999.png"
    orphan_lbl = out / "labels" / "orphan_999_999.txt"
    orphan_img.parent.mkdir(parents=True)
    orphan_lbl.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), color=(1, 2, 3)).save(orphan_img)
    orphan_lbl.write_text("x\n", encoding="utf-8")

    _tile_dota.tile_dota_images(
        root,
        tile_width=32,
        tile_height=32,
        tile_overlap=8,
        output_dir=out,
        clean_output=True,
    )
    assert not orphan_img.exists()
    assert not orphan_lbl.exists()
    assert list((out / "images").glob("*.png"))


def test_tile_dota_resize_rate_small_and_large(tmp_path: Path):
    # rate 0.5 on 80x80 → 40x40 < tile 64 → one padded tile.
    small = _write_sized_split(tmp_path / "ms_small", width=80, height=80)
    _tile_dota.tile_dota_images(
        small, tile_width=64, tile_height=64, tile_overlap=16, resize_rate=0.5
    )
    assert len(list((small / "tiles_64_r0.5" / "images").glob("*.png"))) == 1

    # rate 1.5 on 80x48 → 120x72; tile 64 / overlap 16 → multiple x, flush last.
    large = _write_sized_split(tmp_path / "ms_large", width=80, height=48)
    _tile_dota.tile_dota_images(
        large, tile_width=64, tile_height=64, tile_overlap=16, resize_rate=1.5
    )
    tiles = list((large / "tiles_64_r1.5" / "images").glob("*.png"))
    assert len(tiles) >= 2
    xs = [int(p.stem.split("_")[1]) for p in tiles]
    # Resized width 120; last start must satisfy start+64 <= 120.
    assert max(xs) + 64 <= 120

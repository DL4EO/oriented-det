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

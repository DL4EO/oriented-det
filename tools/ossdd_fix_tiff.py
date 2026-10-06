#!/usr/bin/env python3
"""Make OSSDD 700×700 train chips readable by Pillow.

Those GeoTIFFs are uncompressed float32 with photometric interpretation
34892 (DNG LinearRaw), which Pillow rejects. Val and test chips are already
photometric 1 (MinIsBlack). This rewrites only that tag to 1. Pixel bytes
stay in place.

Usage:
    python tools/ossdd_fix_tiff.py --data-root /path/to/data/OSSDD --dry-run
    python tools/ossdd_fix_tiff.py --data-root /path/to/data/OSSDD
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from oriented_det.utils import tqdm_progress_stream
from tools.ossdd_to_dota import find_ossdd_root

_PHOTOMETRIC_LINEAR_RAW = 34892
_PHOTOMETRIC_MINISBLACK = 1
_TAG_WIDTH = 256
_TAG_LENGTH = 257
_TAG_PHOTOMETRIC = 262
_TYPE_SHORT = 3


def _inline_short(endian: str, raw: bytes) -> int:
    return struct.unpack_from(endian + "H", raw)[0]


def _pack_inline_short(endian: str, value: int) -> bytes:
    return struct.pack(endian + "H", value) + b"\x00\x00"


def linearraw_tag_offset(data: bytes) -> int | None:
    """Byte offset of the inline photometric SHORT, or None if this file is left alone.

    Only 700×700 chips tagged 34892 are converted. Other sizes and tags are skipped.
    """
    if len(data) < 8 or data[:2] not in (b"II", b"MM"):
        return None
    endian = "<" if data[:2] == b"II" else ">"
    magic = struct.unpack_from(endian + "H", data, 2)[0]
    if magic != 42:
        return None
    ifd = struct.unpack_from(endian + "I", data, 4)[0]
    if ifd < 0 or ifd + 2 > len(data):
        return None
    count = struct.unpack_from(endian + "H", data, ifd)[0]
    pos = ifd + 2
    width = height = None
    photo_at: int | None = None
    photo = None
    for _ in range(count):
        if pos + 12 > len(data):
            return None
        tag, typ, nvals = struct.unpack_from(endian + "HHI", data, pos)
        raw = data[pos + 8 : pos + 12]
        if typ == _TYPE_SHORT and nvals == 1:
            value = _inline_short(endian, raw)
            if tag == _TAG_WIDTH:
                width = value
            elif tag == _TAG_LENGTH:
                height = value
            elif tag == _TAG_PHOTOMETRIC:
                photo = value
                photo_at = pos + 8
        pos += 12
    if width != 700 or height != 700 or photo != _PHOTOMETRIC_LINEAR_RAW or photo_at is None:
        return None
    return photo_at


def fix_linearraw_tiff(path: Path, *, dry_run: bool = False) -> str:
    """Return ``fixed``, ``skipped``, or ``dry-run``."""
    data = path.read_bytes()
    offset = linearraw_tag_offset(data)
    if offset is None:
        return "skipped"
    if dry_run:
        return "dry-run"
    endian = "<" if data[:2] == b"II" else ">"
    patched = bytearray(data)
    patched[offset : offset + 4] = _pack_inline_short(endian, _PHOTOMETRIC_MINISBLACK)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(patched)
    temporary.replace(path)
    return "fixed"


def fix_ossdd_tiffs(data_root: Path, *, dry_run: bool = False) -> dict[str, int]:
    """Rewrite photometric 34892 → 1 on 700×700 chips under ``Chip_VV`` and ``Chip_VH``."""
    root = find_ossdd_root(data_root)
    paths: list[Path] = []
    for folder_name in ("Chip_VV", "Chip_VH"):
        folder = root / folder_name
        if not folder.is_dir():
            continue
        paths.extend(sorted(folder.glob("*.tif")))
        paths.extend(sorted(folder.glob("*.tiff")))
    counts = {"fixed": 0, "skipped": 0, "dry-run": 0}
    desc = "Checking 700×700 chips" if dry_run else "Fixing 700×700 chips"
    for path in tqdm(paths, desc=desc, file=tqdm_progress_stream()):
        counts[fix_linearraw_tiff(path, dry_run=dry_run)] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    counts = fix_ossdd_tiffs(args.data_root, dry_run=args.dry_run)
    action = "would rewrite" if args.dry_run else "rewrote"
    print(
        f"{action} {counts['dry-run'] if args.dry_run else counts['fixed']} "
        f"700×700 chips (skipped {counts['skipped']})"
    )


if __name__ == "__main__":
    main()

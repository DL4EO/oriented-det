#!/usr/bin/env python3
"""Publish a checkpoint for Hub distribution (MMDet-style content hash suffix).

Strips optimizer state, saves CPU tensors, appends the first 8 hex chars of SHA-256.

Example::

    python tools/publish_checkpoint.py \\
        runs/rotated_retinanet/20260611-101135/checkpoints/best_mAP_0.70.pth \\
        pretrained/rotated_retinanet_r50_fpn_dota_le90_1x
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_checkpoint(
    in_file: Path,
    out_stem: Path,
    *,
    save_keys: Optional[Sequence[str]] = None,
) -> Path:
    """Write a published checkpoint; return final path with ``-{hash8}.pth`` suffix."""
    in_file = in_file.expanduser().resolve()
    if not in_file.is_file():
        raise FileNotFoundError(f"Input checkpoint not found: {in_file}")

    checkpoint = torch.load(in_file, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected dict checkpoint, got {type(checkpoint).__name__}")

    if save_keys is not None:
        keys = list(save_keys)
        for key in list(checkpoint):
            if key not in keys:
                checkpoint.pop(key, None)
    else:
        for key in ("optimizer", "ema_state_dict", "lr_scheduler"):
            checkpoint.pop(key, None)

    out_stem = out_stem.expanduser().resolve()
    if out_stem.suffix == ".pth":
        out_stem = out_stem.with_suffix("")

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_stem.parent / f"{out_stem.name}.pth"
    torch.save(checkpoint, temp_path, _use_new_zipfile_serialization=False)

    sha = _sha256_file(temp_path)
    final_path = out_stem.parent / f"{out_stem.name}-{sha[:8]}.pth"
    if final_path.exists():
        final_path.unlink()
    temp_path.rename(final_path)
    return final_path


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Publish a checkpoint with SHA256[:8] suffix.")
    parser.add_argument("in_file", type=Path, help="Training checkpoint (.pth)")
    parser.add_argument(
        "out_stem",
        type=Path,
        help="Output path without hash suffix (e.g. pretrained/foo_bar)",
    )
    parser.add_argument(
        "--save-keys",
        nargs="+",
        default=None,
        help="Keep only these top-level keys (default: drop optimizer only)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    final = publish_checkpoint(args.in_file, args.out_stem, save_keys=args.save_keys)
    sha = _sha256_file(final)
    print(f"Published: {final}")
    print(f"sha256: {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

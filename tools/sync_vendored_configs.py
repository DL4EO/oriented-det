#!/usr/bin/env python3
"""Sync repo-root configs/ into oriented_det/configs/ for PyPI packaging.

Source of truth: configs/ (repository root).
Vendored copy: oriented_det/configs/ (listed in vendored_manifest.txt).

Usage:
    python tools/sync_vendored_configs.py          # copy files
    python tools/sync_vendored_configs.py --check  # exit 1 if vendored copy differs
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_manifest(manifest_path: Path) -> list[str]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    paths: list[str] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        paths.append(line.replace("\\", "/"))
    if not paths:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return paths


def sync_vendored_configs(*, check: bool = False) -> int:
    root = _repo_root()
    src_root = root / "configs"
    dst_root = root / "oriented_det" / "configs"
    manifest_path = dst_root / "vendored_manifest.txt"
    rel_paths = _read_manifest(manifest_path)

    errors: list[str] = []
    updated = 0

    for rel in rel_paths:
        src = src_root / rel
        dst = dst_root / rel
        if not src.is_file():
            errors.append(f"source missing: configs/{rel}")
            continue
        if check:
            if not dst.is_file():
                errors.append(f"vendored missing: oriented_det/configs/{rel} (run: make sync-configs)")
            elif src.read_bytes() != dst.read_bytes():
                errors.append(
                    f"out of sync: {rel} (edit configs/, then: make sync-configs)"
                )
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_file() and dst.read_bytes() == src.read_bytes():
            continue
        shutil.copy2(src, dst)
        updated += 1

    if errors:
        print("Vendored config check failed:", file=sys.stderr)
        for msg in errors:
            print(f"  - {msg}", file=sys.stderr)
        return 1

    if check:
        print(f"OK: {len(rel_paths)} vendored config file(s) match configs/")
        return 0

    print(f"Synced {updated} file(s) into oriented_det/configs/ ({len(rel_paths)} in manifest)")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify vendored files match configs/; do not write.",
    )
    args = parser.parse_args(argv)
    raise SystemExit(sync_vendored_configs(check=args.check))


if __name__ == "__main__":
    main()

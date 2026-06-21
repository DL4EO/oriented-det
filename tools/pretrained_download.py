#!/usr/bin/env python3
"""Download OrientedDet pretrained checkpoints from Hugging Face Hub."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download OrientedDet pretrained checkpoints from Hugging Face Hub."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="Download one or more registered checkpoints")
    dl.add_argument(
        "assets",
        nargs="+",
        help="Asset name or .pth filename (see `list`)",
    )

    sub.add_parser("list", help="List registered pretrained assets")

    args = parser.parse_args()

    from oriented_det.pretrained import download_asset, list_assets, load_manifest

    if args.command == "list":
        manifest = load_manifest()
        repo_id = manifest.get("repo_id", "")
        print(f"repo_id: {repo_id}")
        print(f"revision: {manifest.get('revision', 'main')}")
        print("assets:")
        for name, remote in sorted(list_assets().items()):
            print(f"  {name}  ->  {remote}")
        return

    for asset in args.assets:
        path = download_asset(asset)
        print(f"Downloaded {asset} -> {path}")


if __name__ == "__main__":
    try:
        main()
    except (ImportError, KeyError, RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1) from e

#!/usr/bin/env python3
"""Convert DOTA label files from space-separated to official comma-separated format.

Reads all .txt annotation files in a folder (or in labels/ under a dataset root),
parses each line (accepts both space- and comma-separated input), and writes
back in official DOTA format: x1, y1, x2, y2, x3, y3, x4, y4, category, difficult.

Usage:
  # Convert all .txt in a labels folder
  python tools/dota_labels_to_comma.py /path/to/labels

  # Convert labels under a dataset root (looks for labels/ or labelTxt/)
  python tools/dota_labels_to_comma.py /path/to/dataset

  # Dry run (print what would be done)
  python tools/dota_labels_to_comma.py /path/to/labels --dry-run

  # Backup original files as .txt.bak before overwriting
  python tools/dota_labels_to_comma.py /path/to/labels --backup
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure oriented_det is importable when run as script
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from oriented_det.data import DOTAAnnotation


def _is_metadata_or_empty(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if "imagesource:" in line.lower() or "gsd" in line.lower():
        return True
    return False


def convert_file(path: Path, *, dry_run: bool = False, backup: bool = False) -> int:
    """Convert one DOTA label file to comma-separated. Returns number of annotation lines converted."""
    text = path.read_text(encoding="utf-8")
    lines_in = text.splitlines()
    output_lines: list[str] = []
    count = 0
    for raw in lines_in:
        if _is_metadata_or_empty(raw):
            output_lines.append(raw)  # Keep metadata/empty as-is
            continue
        try:
            ann = DOTAAnnotation.from_line(raw)
            output_lines.append(ann.to_line())
            count += 1
        except Exception as e:
            output_lines.append(raw)  # Keep unparseable lines unchanged
            if not dry_run:
                print(f"  Warning: could not parse line in {path}: {e}", file=sys.stderr)
    if not dry_run:
        if backup:
            path.with_suffix(path.suffix + ".bak").write_text(text, encoding="utf-8")
        path.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")
    return count


def find_label_txts(root: Path) -> list[Path]:
    """Find all .txt files under root. If root has labels/ or labelTxt/, use that dir."""
    root = root.resolve()
    if not root.is_dir():
        return []
    labels_dir = root / "labels"
    label_txt_dir = root / "labelTxt"
    if labels_dir.is_dir():
        return sorted(labels_dir.glob("*.txt"))
    if label_txt_dir.is_dir():
        return sorted(label_txt_dir.glob("*.txt"))
    return sorted(root.glob("*.txt"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert DOTA label .txt files from space-separated to official comma-separated format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing .txt label files, or dataset root (will use labels/ or labelTxt/ if present).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report which files would be converted; do not write.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Backup each file as .txt.bak before overwriting.",
    )
    args = parser.parse_args()
    folder = args.folder.resolve()
    if not folder.exists():
        print(f"Error: folder does not exist: {folder}", file=sys.stderr)
        return 1
    if not folder.is_dir():
        print(f"Error: not a directory: {folder}", file=sys.stderr)
        return 1
    txt_files = find_label_txts(folder)
    if not txt_files:
        print(f"No .txt files found in {folder} (or in {folder}/labels, {folder}/labelTxt).", file=sys.stderr)
        return 1
    print(f"Found {len(txt_files)} .txt file(s).")
    if args.dry_run:
        print("Dry run: no files will be modified.")
    total = 0
    for path in txt_files:
        n = convert_file(path, dry_run=args.dry_run, backup=args.backup)
        total += n
        if args.dry_run and n:
            print(f"  Would convert: {path} ({n} annotation lines)")
    if not args.dry_run:
        print(f"Converted {len(txt_files)} file(s), {total} annotation lines -> comma-separated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""``odet`` unified command-line interface."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Maps subcommand name -> (module path, argv[0] for the target script)
_COMMANDS: Dict[str, Tuple[str, str]] = {
    "train": ("tools.train", "odet-train"),
    "train-multi-gpu": ("tools.train_multi_gpu", "odet-train-multi-gpu"),
    "preds": ("tools.save_predictions", "odet-preds"),
    "metrics": ("tools.save_predictions", "odet-metrics"),
    "lr-finder": ("tools.lr_finder", "odet-lr-finder"),
    "stats": ("tools.dataset_stats", "odet-stats"),
    "tile-dota": ("tools.tile_dota", "odet-tile-dota"),
    "image-demo": ("tools.image_demo", "odet-image-demo"),
    "viewer": ("tools.app", "odet-viewer"),
    "playground-csv": ("tools.generate_airbus_playground_csv", "odet-playground-csv"),
    "playground-to-dota": ("tools.playground_to_dota", "odet-playground-to-dota"),
    "export-onnx": ("export.scripts.export_onnx", "odet-export-onnx"),
    "export-tf": ("export.scripts.export_tf", "odet-export-tf"),
    "export-detect": ("export.scripts.build_faster_rcnn_savedmodel", "odet-export-detect"),
    "export-preds": ("export.scripts.save_predictions_tf", "odet-export-preds"),
    "labels-to-comma": ("tools.dota_labels_to_comma", "odet-labels-to-comma"),
    "free-gpu": ("tools.free_gpu", "odet-free-gpu"),
    "pretrained": ("tools.pretrained_download", "odet-pretrained"),
}


def _print_help() -> None:
    print("Usage: odet <command> [options]")
    print("")
    print("Commands:")
    for name in sorted(_COMMANDS):
        print(f"  {name}")
    print("")
    print("Examples:")
    print("  odet train --config configs/oriented_rcnn/dota_le90_1x.json")
    print("  odet train-multi-gpu --config configs/oriented_rcnn/dota_le90_1x.json")
    print("  odet preds --experiment-dir runs/oriented_rcnn/<id>")
    print("  odet playground-csv --data-root /path/to/export")
    print("  odet export-tf --config path/to/config.json --checkpoint path/to/model.pth")
    print("  odet export-tf --mode oriented_rcnn_pre_nms --config ... --checkpoint ...")
    print("  odet export-tf --mode rotated_fcos_pre_nms --config ... --checkpoint ...")


def _invoke(module_path: str, prog: str, args: List[str]) -> None:
    # Editable checkout: repo root on sys.path. Wheel install: packages already importable.
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        if module_path.startswith("export."):
            raise SystemExit(
                f"Failed to import {module_path}: {e}\n"
                'For TF/ONNX export install extras: pip install "oriented-det[export]"'
            ) from e
        raise
    if not hasattr(mod, "main"):
        raise SystemExit(f"{module_path} has no main()")
    sys.argv = [prog] + args
    mod.main()


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        _print_help()
        return
    cmd, rest = argv[0], argv[1:]
    if cmd in ("-h", "--help"):
        _print_help()
        return
    if cmd not in _COMMANDS:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        _print_help()
        raise SystemExit(2)
    module_path, prog = _COMMANDS[cmd]
    _invoke(module_path, prog, rest)


if __name__ == "__main__":
    main()

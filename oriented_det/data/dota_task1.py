"""DOTA v1.0 Task 1 (OBB) submission files from ``predictions.json``.

Official server format (space-separated, one line per detection)::

    image_id confidence x1 y1 x2 y2 x3 y3 x4 y4

One ``Task1_{class}.txt`` per DOTA v1.0 class. Empty class files are rejected by
the server; write a dummy low-score box when a class has no detections.
"""

from __future__ import annotations

import json
import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from oriented_det.data.dota_classes import DOTA_V1_CLASSES
from oriented_det.geometry.rbox import RBox

TASK1_DUMMY_IMAGE_ID = "P0000"
TASK1_DUMMY_SCORE = 1e-6
TASK1_DUMMY_POLY = (0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0)


def task1_filename(class_name: str) -> str:
    return f"Task1_{class_name}.txt"


def format_task1_line(image_id: str, score: float, poly: Sequence[float]) -> str:
    if len(poly) != 8:
        raise ValueError(f"Task 1 polygon must have 8 coordinates, got {len(poly)}")
    coords = " ".join(f"{float(v):.2f}" for v in poly)
    return f"{image_id} {float(score):.4f} {coords}"


def dummy_task1_line() -> str:
    return format_task1_line(TASK1_DUMMY_IMAGE_ID, TASK1_DUMMY_SCORE, TASK1_DUMMY_POLY)


def rbox_to_task1_poly(bbox: Sequence[float]) -> tuple[float, ...]:
    """Convert ``[cx, cy, w, h, angle]`` to 8 corner coordinates."""
    if len(bbox) < 5:
        raise ValueError(f"bbox must be [cx, cy, w, h, angle], got {bbox!r}")
    rbox = RBox(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4]))
    corners = rbox.corners()
    out: list[float] = []
    for pt in corners:
        out.extend((float(pt[0]), float(pt[1])))
    return tuple(out)


def _image_id_from_result(entry: Mapping[str, Any]) -> str:
    name = entry.get("image_name") or entry.get("image_path") or ""
    return Path(str(name)).stem


def _class_name_from_pred(pred: Mapping[str, Any], class_names: Sequence[str]) -> Optional[str]:
    name = pred.get("class_name")
    if isinstance(name, str) and name:
        return name
    label = pred.get("label")
    try:
        idx = int(label)
    except (TypeError, ValueError):
        return None
    if 1 <= idx <= len(class_names):
        return class_names[idx - 1]
    return None


def predictions_to_task1_lines(
    results: Iterable[Mapping[str, Any]],
    *,
    class_names: Sequence[str] = DOTA_V1_CLASSES,
    dummy_if_empty: bool = True,
) -> Dict[str, list[str]]:
    """Group Task 1 lines by official class name (all ``class_names`` keys present)."""
    lines_by_class: Dict[str, list[str]] = {c: [] for c in class_names}
    known = set(class_names)
    skipped_classes: set[str] = set()

    for entry in results:
        image_id = _image_id_from_result(entry)
        if not image_id:
            continue
        for pred in entry.get("predictions") or []:
            class_name = _class_name_from_pred(pred, class_names)
            if class_name is None:
                continue
            if class_name not in known:
                skipped_classes.add(class_name)
                continue
            bbox = pred.get("bbox")
            if not bbox:
                continue
            try:
                poly = rbox_to_task1_poly(bbox)
            except (TypeError, ValueError):
                continue
            score = float(pred.get("score", 0.0))
            lines_by_class[class_name].append(format_task1_line(image_id, score, poly))

    if skipped_classes:
        warnings.warn(
            "Skipped Task 1 detections with unknown class names: "
            + ", ".join(sorted(skipped_classes)),
            stacklevel=2,
        )

    if dummy_if_empty:
        dummy = dummy_task1_line()
        for class_name, lines in lines_by_class.items():
            if not lines:
                lines.append(dummy)

    return lines_by_class


def load_predictions_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        path = path / "predictions.json"
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "results" not in payload:
        raise ValueError(f"{path} is not a predictions.json object with a 'results' list")
    return payload


def predictions_json_to_task1(
    payload: Mapping[str, Any] | str | Path,
    *,
    class_names: Sequence[str] = DOTA_V1_CLASSES,
    dummy_if_empty: bool = True,
) -> Dict[str, list[str]]:
    if not isinstance(payload, Mapping):
        payload = load_predictions_json(payload)
    results = payload.get("results") or []
    meta_names = payload.get("metadata") or {}
    names = meta_names.get("class_names") if isinstance(meta_names, Mapping) else None
    if names and not class_names:
        class_names = list(names)
    return predictions_to_task1_lines(
        results, class_names=class_names, dummy_if_empty=dummy_if_empty
    )


def write_task1_dir(
    out_dir: str | Path,
    lines_by_class: Mapping[str, Sequence[str]],
    *,
    class_names: Sequence[str] = DOTA_V1_CLASSES,
) -> Path:
    """Write ``Task1_*.txt`` under ``out_dir``. Returns the directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for class_name in class_names:
        dest = out_dir / task1_filename(class_name)
        lines = list(lines_by_class.get(class_name) or [])
        text = "\n".join(lines)
        if text:
            text += "\n"
        dest.write_text(text, encoding="utf-8")
    return out_dir


def zip_task1_dir(out_dir: str | Path, zip_path: str | Path | None = None) -> Path:
    """Zip ``Task1_*.txt`` at the archive root (no nested folder)."""
    out_dir = Path(out_dir)
    if zip_path is None:
        zip_path = out_dir.parent / f"{out_dir.name}.zip"
    zip_path = Path(zip_path)
    files = sorted(out_dir.glob("Task1_*.txt"))
    if not files:
        raise FileNotFoundError(f"No Task1_*.txt files in {out_dir}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for txt in files:
            zf.write(txt, arcname=txt.name)
    return zip_path


def write_task1_submission(
    predictions: Mapping[str, Any] | str | Path,
    out_dir: str | Path,
    *,
    class_names: Sequence[str] = DOTA_V1_CLASSES,
    dummy_if_empty: bool = True,
    zip_path: str | Path | None = None,
) -> Path:
    """Write Task 1 txt files and a zip. Returns the zip path."""
    lines = predictions_json_to_task1(
        predictions, class_names=class_names, dummy_if_empty=dummy_if_empty
    )
    write_task1_dir(out_dir, lines, class_names=class_names)
    return zip_task1_dir(out_dir, zip_path=zip_path)

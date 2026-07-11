"""Load horizontal GT boxes and convert to oriented boxes via model predictions.

Supports GT in CSV (polygon rows), YOLO horizontal labels, or DOTA label files.
"""

from __future__ import annotations

import ast
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from oriented_det import RBox
from oriented_det.data.dota import DOTAAnnotation, format_dota_line
from oriented_det.ops.iou import batch_rbox_iou

GtFormat = Literal["csv", "yolo", "dota"]


@dataclass(frozen=True)
class GtBox:
    ann_id: str
    image_id: str
    rbox: RBox
    class_name: str
    difficult: int = 0


def _normalize_class_key(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _should_skip_class(class_name: str, ignore_classes: Optional[set[str]]) -> bool:
    if not ignore_classes:
        return False
    key = _normalize_class_key(class_name)
    return key in ignore_classes or class_name.strip() in ignore_classes


def load_gt_from_csv(
    annotations_csv: Path,
    *,
    ignore_classes: Optional[set[str]] = None,
) -> Dict[str, List[GtBox]]:
    """Load GT from CSV with columns ``id``, ``image_id``, ``geometry``, ``class``.

    ``geometry`` is a Python literal list of ``(x, y)`` polygon corners (axis-aligned or not).
    """
    by_image: Dict[str, List[GtBox]] = defaultdict(list)
    skipped = 0
    with annotations_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            class_name = row["class"]
            if _should_skip_class(class_name, ignore_classes):
                skipped += 1
                continue
            pts = ast.literal_eval(row["geometry"])
            if pts and pts[0] == pts[-1]:
                pts = pts[:-1]
            by_image[row["image_id"]].append(
                GtBox(
                    ann_id=str(row["id"]),
                    image_id=row["image_id"],
                    rbox=RBox.from_points(pts),
                    class_name=class_name,
                )
            )
    total = sum(len(v) for v in by_image.values())
    print(f"GT (CSV): {total} boxes on {len(by_image)} images")
    if skipped:
        print(f"GT (CSV): skipped {skipped} ignored-class annotations")
    return by_image


def load_gt_from_yolo(
    dataset_root: Path,
    *,
    images_dir: str = "images",
    labels_dir: str = "labels",
    class_names: Optional[Dict[int, str]] = None,
    default_class_name: str = "object",
    image_glob: str = "*.jpg",
    ignore_classes: Optional[set[str]] = None,
) -> Dict[str, List[GtBox]]:
    """Load horizontal YOLO labels (``class cx cy w h``, normalized)."""
    from PIL import Image

    img_dir = dataset_root / images_dir
    lbl_dir = dataset_root / labels_dir
    by_image: Dict[str, List[GtBox]] = {}

    for img_path in sorted(img_dir.glob(image_glob)):
        for ext in (".txt",):
            label_path = lbl_dir / f"{img_path.stem}{ext}"
            if not label_path.exists():
                continue
            with Image.open(img_path) as im:
                img_w, img_h = im.size
            boxes: List[GtBox] = []
            for line_no, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                cx_n, cy_n, w_n, h_n = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                class_name = (
                    class_names.get(cls_id, default_class_name)
                    if class_names is not None
                    else default_class_name
                )
                if _should_skip_class(class_name, ignore_classes):
                    continue
                boxes.append(
                    GtBox(
                        ann_id=str(line_no),
                        image_id=img_path.name,
                        rbox=RBox(
                            float(cx_n) * img_w,
                            float(cy_n) * img_h,
                            float(w_n) * img_w,
                            float(h_n) * img_h,
                            0.0,
                        ),
                        class_name=class_name,
                    )
                )
            if boxes:
                by_image[img_path.name] = boxes

    total = sum(len(v) for v in by_image.values())
    print(f"GT (YOLO): {total} boxes on {len(by_image)} images")
    return by_image


def load_gt_from_dota(
    dataset_root: Path,
    *,
    images_dir: str = "images",
    labels_dir: str = "labelTxt",
    image_glob: str = "*.png",
    ignore_classes: Optional[set[str]] = None,
) -> Dict[str, List[GtBox]]:
    """Load horizontal or oriented DOTA ``.txt`` labels (one file per image stem)."""
    img_dir = dataset_root / images_dir
    lbl_root = dataset_root / labels_dir
    by_image: Dict[str, List[GtBox]] = {}

    image_paths = sorted(img_dir.glob(image_glob))
    if not image_paths:
        image_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.jpeg"))

    for img_path in image_paths:
        label_path = lbl_root / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        boxes: List[GtBox] = []
        for line_no, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = line.strip()
            if not line:
                continue
            ann = DOTAAnnotation.from_line(line)
            if _should_skip_class(ann.class_name, ignore_classes):
                continue
            boxes.append(
                GtBox(
                    ann_id=str(line_no),
                    image_id=img_path.name,
                    rbox=ann.rbox,
                    class_name=ann.class_name,
                    difficult=ann.difficult,
                )
            )
        if boxes:
            by_image[img_path.name] = boxes

    total = sum(len(v) for v in by_image.values())
    print(f"GT (DOTA): {total} boxes on {len(by_image)} images")
    return by_image


def load_gt_boxes(
    gt_format: GtFormat,
    *,
    dataset_root: Optional[Path] = None,
    annotations_path: Optional[Path] = None,
    images_dir: str = "images",
    labels_dir: Optional[str] = None,
    ignore_classes: Optional[Iterable[str]] = None,
    yolo_class_names: Optional[Dict[int, str]] = None,
    default_class_name: str = "object",
) -> Dict[str, List[GtBox]]:
    ignore_set = {_normalize_class_key(c) for c in ignore_classes} if ignore_classes else None
    if gt_format == "csv":
        if annotations_path is None:
            raise ValueError("annotations_path is required for gt_format=csv")
        return load_gt_from_csv(annotations_path, ignore_classes=ignore_set)
    if dataset_root is None:
        raise ValueError("dataset_root is required for gt_format=yolo or dota")
    if gt_format == "yolo":
        return load_gt_from_yolo(
            dataset_root,
            images_dir=images_dir,
            labels_dir=labels_dir or "labels",
            class_names=yolo_class_names,
            default_class_name=default_class_name,
            ignore_classes=ignore_set,
        )
    return load_gt_from_dota(
        dataset_root,
        images_dir=images_dir,
        labels_dir=labels_dir or "labelTxt",
        ignore_classes=ignore_set,
    )


def pred_to_rbox(pred: dict) -> RBox:
    if "bbox" in pred:
        cx, cy, w, h, angle = pred["bbox"]
        return RBox(float(cx), float(cy), float(w), float(h), float(angle))
    rbox = pred["rbox"]
    return RBox(
        float(rbox["cx"]),
        float(rbox["cy"]),
        float(rbox["width"]),
        float(rbox["height"]),
        float(rbox["angle"]),
    )


def normalize_predictions(row: dict) -> List[dict]:
    if "predictions" in row:
        return list(row.get("predictions", []))
    if "detections" in row:
        return list(row.get("detections", []))
    return []


def image_name_from_row(row: dict) -> str:
    if row.get("image_name"):
        return str(row["image_name"])
    source = row.get("source_image") or row.get("image_path") or ""
    return Path(source).name


def greedy_match(
    predictions: Sequence[dict],
    gt_boxes: Sequence[GtBox],
    iou_threshold: float,
) -> Tuple[List[dict], List[int], List[int]]:
    if not predictions or not gt_boxes:
        return [], [], list(range(len(predictions)))

    pred_rboxes = [pred_to_rbox(p) for p in predictions]
    gt_rboxes = [g.rbox for g in gt_boxes]
    iou_matrix = batch_rbox_iou(pred_rboxes, gt_rboxes, intersection_backend="python")

    pairs: List[Tuple[float, int, int]] = []
    for pi in range(len(predictions)):
        for gi in range(len(gt_boxes)):
            iou = float(iou_matrix[pi][gi])
            if iou >= iou_threshold:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True)

    pred_taken = [False] * len(predictions)
    gt_taken = [False] * len(gt_boxes)
    matched_preds: List[dict] = []
    matched_gt: List[int] = []

    for iou, pi, gi in pairs:
        if pred_taken[pi] or gt_taken[gi]:
            continue
        pred_taken[pi] = True
        gt_taken[gi] = True
        out = dict(predictions[pi])
        out["match_iou"] = round(iou, 4)
        out["gt_ann_id"] = gt_boxes[gi].ann_id
        matched_preds.append(out)
        matched_gt.append(gi)

    unmatched_pred = [i for i, taken in enumerate(pred_taken) if not taken]
    return matched_preds, matched_gt, unmatched_pred


def filter_predictions(
    detections_json: Path,
    gt_by_image: Dict[str, List[GtBox]],
    iou_threshold: float,
) -> Dict[str, Any]:
    payload = json.loads(detections_json.read_text(encoding="utf-8"))
    results_out: List[dict] = []
    stats = Counter()

    for row in payload.get("results", []):
        image_name = image_name_from_row(row)
        preds = normalize_predictions(row)
        gt_boxes = gt_by_image.get(image_name, [])
        matched, matched_gt_idx, unmatched_pred = greedy_match(preds, gt_boxes, iou_threshold)

        stats["raw_predictions"] += len(preds)
        stats["gt_boxes"] += len(gt_boxes)
        stats["kept_after_gt_filter"] += len(matched)
        stats["false_positives_removed"] += len(unmatched_pred)
        stats["missed_gt"] += len(gt_boxes) - len(matched_gt_idx)

        results_out.append(
            {
                **{k: v for k, v in row.items() if k not in {"predictions", "num_pred"}},
                "num_gt": len(gt_boxes),
                "num_pred_raw": len(preds),
                "num_pred": len(matched),
                "num_false_positives_removed": len(unmatched_pred),
                "num_missed_gt": len(gt_boxes) - len(matched_gt_idx),
                "predictions": matched,
            }
        )

    return {
        "filter": {
            "match_iou_threshold": iou_threshold,
            "source_detections": str(detections_json.resolve()),
        },
        "stats": dict(stats),
        "num_images": len(results_out),
        "total_predictions_raw": stats["raw_predictions"],
        "total_predictions": stats["kept_after_gt_filter"],
        "total_gt": stats["gt_boxes"],
        "total_false_positives_removed": stats["false_positives_removed"],
        "total_missed_gt": stats["missed_gt"],
        "results": results_out,
    }


@dataclass(frozen=True)
class OrientedAnn:
    ann_id: str
    image_id: str
    class_name: str
    rbox: RBox
    source: str
    difficult: int = 0
    score: Optional[float] = None
    match_iou: Optional[float] = None


def rbox_from_bbox(bbox: Sequence[float]) -> RBox:
    cx, cy, w, h, angle = bbox
    return RBox(float(cx), float(cy), float(w), float(h), float(angle))


def polygon_string(rbox: RBox) -> str:
    pts = list(rbox.to_polygon().points)
    closed = pts + [pts[0]]
    return str([(int(round(x)), int(round(y))) for x, y in closed])


def matched_by_gt_id(filtered: Dict[str, Any]) -> Dict[str, Dict[str, dict]]:
    out: Dict[str, Dict[str, dict]] = {}
    for row in filtered.get("results", []):
        image_name = image_name_from_row(row)
        mapping: Dict[str, dict] = {}
        for pred in row.get("predictions", []):
            gt_id = pred.get("gt_ann_id")
            if gt_id is not None:
                mapping[str(gt_id)] = pred
        out[image_name] = mapping
    return out


def build_oriented_annotations(
    gt_by_image: Dict[str, List[GtBox]],
    filtered: Dict[str, Any],
    *,
    model_source_label: str = "model",
) -> List[OrientedAnn]:
    matched = matched_by_gt_id(filtered)
    rows: List[OrientedAnn] = []
    oriented_count = 0
    fallback_count = 0

    for image_id in sorted(gt_by_image):
        for gt in gt_by_image[image_id]:
            pred = matched.get(image_id, {}).get(gt.ann_id)
            if pred is not None:
                rows.append(
                    OrientedAnn(
                        ann_id=gt.ann_id,
                        image_id=image_id,
                        class_name=gt.class_name,
                        rbox=rbox_from_bbox(pred["bbox"]),
                        source=model_source_label,
                        difficult=gt.difficult,
                        score=float(pred.get("score", 0.0)),
                        match_iou=float(pred.get("match_iou", 0.0)),
                    )
                )
                oriented_count += 1
            else:
                rows.append(
                    OrientedAnn(
                        ann_id=gt.ann_id,
                        image_id=image_id,
                        class_name=gt.class_name,
                        rbox=gt.rbox,
                        source="hbb_fallback",
                        difficult=gt.difficult,
                    )
                )
                fallback_count += 1

    print(f"Oriented from model: {oriented_count}")
    print(f"HBB fallback: {fallback_count}")
    print(f"Total annotations: {len(rows)}")
    return rows


def write_oriented_csv(path: Path, rows: Iterable[OrientedAnn]) -> None:
    fieldnames = [
        "id",
        "image_id",
        "class",
        "geometry",
        "cx",
        "cy",
        "width",
        "height",
        "angle_deg",
        "source",
        "score",
        "match_iou",
        "difficult",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            rb = row.rbox
            writer.writerow(
                {
                    "id": row.ann_id,
                    "image_id": row.image_id,
                    "class": row.class_name,
                    "geometry": polygon_string(rb),
                    "cx": f"{rb.cx:.3f}",
                    "cy": f"{rb.cy:.3f}",
                    "width": f"{rb.width:.3f}",
                    "height": f"{rb.height:.3f}",
                    "angle_deg": f"{math.degrees(rb.angle):.3f}",
                    "source": row.source,
                    "score": "" if row.score is None else f"{row.score:.6f}",
                    "match_iou": "" if row.match_iou is None else f"{row.match_iou:.4f}",
                    "difficult": row.difficult,
                }
            )
    print(f"Wrote CSV: {path}")


def write_oriented_dota_labels(output_dir: Path, rows: Iterable[OrientedAnn]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_image: Dict[str, List[OrientedAnn]] = defaultdict(list)
    for row in rows:
        by_image[row.image_id].append(row)

    for image_id, anns in sorted(by_image.items()):
        stem = Path(image_id).stem
        out_path = output_dir / f"{stem}.txt"
        lines = []
        for ann in anns:
            pts = ann.rbox.to_polygon().points
            if len(pts) != 4:
                continue
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = pts
            lines.append(
                format_dota_line(
                    x1, y1, x2, y2, x3, y3, x4, y4, ann.class_name, ann.difficult
                )
            )
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"Wrote DOTA labels: {output_dir} ({len(by_image)} files)")


def add_gt_cli_args(parser: Any) -> None:
    parser.add_argument(
        "--gt-format",
        choices=("csv", "yolo", "dota"),
        required=True,
        help="Horizontal GT format: csv (polygon CSV), yolo (norm cx cy w h), or dota (labelTxt).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset root (required for yolo/dota).",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Annotations CSV path (gt-format=csv).",
    )
    parser.add_argument("--images-dir", default="images", help="Images subdirectory (yolo/dota).")
    parser.add_argument(
        "--labels-dir",
        default=None,
        help="Labels subdirectory (default: labels for yolo, labelTxt for dota).",
    )
    parser.add_argument(
        "--ignore-class",
        action="append",
        default=None,
        metavar="NAME",
        help="Skip GT boxes with this class (repeatable, case-insensitive).",
    )
    parser.add_argument(
        "--yolo-class-name",
        action="append",
        default=None,
        metavar="ID=NAME",
        help="YOLO class id to name map, e.g. 0=plane (repeatable).",
    )
    parser.add_argument(
        "--default-class-name",
        default="object",
        help="Class name for YOLO labels when --yolo-class-name is not set.",
    )


def parse_yolo_class_names(specs: Optional[Sequence[str]]) -> Optional[Dict[int, str]]:
    if not specs:
        return None
    mapping: Dict[int, str] = {}
    for spec in specs:
        cid, _, name = spec.partition("=")
        mapping[int(cid.strip())] = name.strip()
    return mapping


def resolve_gt_from_args(args: Any) -> Dict[str, List[GtBox]]:
    annotations_path = args.annotations
    if args.gt_format == "csv" and annotations_path is None and args.dataset_root is not None:
        annotations_path = args.dataset_root / "annotations.csv"
    return load_gt_boxes(
        args.gt_format,
        dataset_root=args.dataset_root,
        annotations_path=annotations_path,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        ignore_classes=args.ignore_class,
        yolo_class_names=parse_yolo_class_names(args.yolo_class_name),
        default_class_name=args.default_class_name,
    )

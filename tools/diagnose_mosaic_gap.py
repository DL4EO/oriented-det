"""Compare the current COCO mosaic with the Ultralytics v8 mosaic.

Same four photos each time. No HSV, no flip, no GPU. The matched pass pins
the mosaic center at (640, 640) and keeps the affine at scale 1, center
translation. The training pass samples each pipeline's own center, scale, and
translate. A third pass forces scale 0.5 with a large translate so a dezoom
still keeps the shift.

Writes ``runs/coco_gap/mosaic_report.json`` and contact sheets.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics.data.augment import Mosaic, RandomPerspective
from ultralytics.utils.instance import Instances

from oriented_det.data.coco import CocoDetectionDataset
from oriented_det.data.mosaic import (
    MosaicSample,
    _paste_patch,
    apply_mosaic,
    apply_mosaic_affine,
    resize_long_side,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/coco_gap"
DATA = Path("/path/to/data/coco")
SIZE = 640
N_GROUPS = 16
PHOTO = 1
GRAY = 3


class _Dataset:
    cache = "ram"


def _long(box) -> float:
    if hasattr(box, "width"):
        return float(max(box.width, box.height))
    x1, y1, x2, y2 = box
    return float(max(x2 - x1, y2 - y1))


def _rgb_mask(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.stack([mask, mask, mask], axis=-1), mode="RGB")


def _codes(image: Image.Image) -> np.ndarray:
    return np.asarray(image)[:, :, 0]


def _ours_tiles(samples):
    images = []
    boxes = []
    masks = []
    for sample in samples:
        raw = Image.open(sample.image_path).convert("RGB")
        rboxes = [ann.rbox for ann in sample.annotations]
        image, scaled = resize_long_side(raw, rboxes, SIZE)
        images.append(image)
        boxes.append(list(scaled))
        mask = np.full(image.size[::-1], PHOTO, np.uint8)
        masks.append(mask)
    return images, boxes, masks


def _paste_mask(masks, center: tuple[int, int]) -> np.ndarray:
    canvas = Image.new("RGB", (SIZE * 2, SIZE * 2), (GRAY, GRAY, GRAY))
    for loc, mask in zip(("top_left", "top_right", "bottom_left", "bottom_right"), masks):
        _paste_patch(canvas, _rgb_mask(mask), loc, center, (SIZE, SIZE))
    return _codes(canvas)


def _affine_mask(mask: np.ndarray, scale: float, tx: float, ty: float) -> np.ndarray:
    height, width = mask.shape
    factor = max(float(scale), 1e-3)
    coeff = (
        1.0 / factor,
        0.0,
        width / 2.0 - float(tx) / factor,
        0.0,
        1.0 / factor,
        height / 2.0 - float(ty) / factor,
    )
    warped = Image.fromarray(mask, mode="L").transform(
        (SIZE, SIZE),
        Image.AFFINE,
        coeff,
        resample=Image.NEAREST,
        fillcolor=GRAY,
    )
    return np.asarray(warped)


def _ours(samples, *, center: tuple[int, int], scale: float, tx: float, ty: float):
    images, boxes, masks = _ours_tiles(samples)
    labels = [[1] * len(group) for group in boxes]
    mixed = apply_mosaic(
        MosaicSample(images[0], boxes[0], labels[0]),
        [MosaicSample(images[i], boxes[i], labels[i]) for i in range(1, 4)],
        img_scale=(SIZE, SIZE),
        center_xy=center,
        pad_val=114,
        rotate_prob=0.0,
        min_visible=0.2,
    )
    out = apply_mosaic_affine(
        mixed,
        (SIZE, SIZE),
        scale,
        tx,
        ty,
        pad_val=114,
        min_visible=0.2,
    )
    mask = _affine_mask(_paste_mask(masks, center), scale, tx, ty)
    return out.image, list(out.rboxes), mask


def _ultra_resize(image: Image.Image) -> tuple[Image.Image, float, float]:
    width, height = image.size
    ratio = SIZE / max(height, width, 1)
    resized_w = min(math.ceil(width * ratio), SIZE)
    resized_h = min(math.ceil(height * ratio), SIZE)
    if (resized_w, resized_h) != (width, height):
        image = image.resize((resized_w, resized_h), Image.BILINEAR)
    return image, resized_w / width, resized_h / height


def _ultra_labels(samples):
    labels = []
    for sample in samples:
        raw = Image.open(sample.image_path).convert("RGB")
        resized, sx, sy = _ultra_resize(raw)
        xyxy = []
        for ann in sample.annotations:
            box = ann.rbox
            xyxy.append(
                [
                    (box.cx - box.width / 2.0) * sx,
                    (box.cy - box.height / 2.0) * sy,
                    (box.cx + box.width / 2.0) * sx,
                    (box.cy + box.height / 2.0) * sy,
                ]
            )
        array = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        mask = np.ones(resized.size[::-1], np.uint8) * PHOTO
        labels.append(
            {
                "img": np.asarray(resized),
                "im_file": str(sample.image_path),
                "ori_shape": (raw.size[1], raw.size[0]),
                "resized_shape": resized.size[::-1],
                "cls": np.zeros((len(array), 1), dtype=np.float32),
                "instances": Instances(array, bbox_format="xyxy", normalized=False),
                "mask": mask,
            }
        )
    primary = labels[0]
    primary["mix_labels"] = labels[1:]
    return primary


def _ultra_mask_from_layout(labels, layout) -> np.ndarray:
    canvas = np.full((SIZE * 2, SIZE * 2), GRAY, np.uint8)
    patches = [labels] + labels["mix_labels"]
    for item, patch in zip(layout, patches):
        mask = patch["mask"]
        x1a, y1a, x2a, y2a = item["x1a"], item["y1a"], item["x2a"], item["y2a"]
        x1b, y1b, x2b, y2b = item["x1b"], item["y1b"], item["x2b"], item["y2b"]
        if x2a > x1a and y2a > y1a:
            canvas[y1a:y2a, x1a:x2a] = mask[y1b:y2b, x1b:x2b]
    return canvas


def _ultra_layout(labels, center: tuple[int, int] | None) -> list[dict]:
    side = SIZE
    if center is None:
        yc, xc = (int(random.uniform(side // 2, side * 2 - side // 2)) for _ in range(2))
    else:
        xc, yc = center
    layout = []
    for index in range(4):
        patch = labels if index == 0 else labels["mix_labels"][index - 1]
        height, width = patch["resized_shape"]
        if index == 0:
            x1a, y1a, x2a, y2a = max(xc - width, 0), max(yc - height, 0), xc, yc
            x1b, y1b, x2b, y2b = width - (x2a - x1a), height - (y2a - y1a), width, height
        elif index == 1:
            x1a, y1a, x2a, y2a = xc, max(yc - height, 0), min(xc + width, side * 2), yc
            x1b, y1b, x2b, y2b = 0, height - (y2a - y1a), min(width, x2a - x1a), height
        elif index == 2:
            x1a, y1a, x2a, y2a = max(xc - width, 0), yc, xc, min(side * 2, yc + height)
            x1b, y1b, x2b, y2b = width - (x2a - x1a), 0, width, min(y2a - y1a, height)
        else:
            x1a, y1a, x2a, y2a = xc, yc, min(xc + width, side * 2), min(side * 2, yc + height)
            x1b, y1b, x2b, y2b = 0, 0, min(width, x2a - x1a), min(y2a - y1a, height)
        layout.append(
            {
                "labels_patch": patch,
                "x1a": x1a,
                "y1a": y1a,
                "x2a": x2a,
                "y2a": y2a,
                "x1b": x1b,
                "y1b": y1b,
                "x2b": x2b,
                "y2b": y2b,
                "padw": x1a - x1b,
                "padh": y1a - y1b,
                "img_shape": (height, width),
            }
        )
    return layout


def _run_ultra(samples, *, center: tuple[int, int] | None, scale: float | None, tx: float | None, ty: float | None):
    labels = _ultra_labels(samples)
    mosaic = Mosaic(_Dataset(), imgsz=SIZE, p=1.0, n=4)
    layout = _ultra_layout(labels, center)
    params = {"layout": layout}
    mask1280 = _ultra_mask_from_layout(labels, layout)
    labels = mosaic.apply_image(labels, params)
    labels = mosaic.apply_instances(labels, params)
    if scale is None:
        affine = RandomPerspective(
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            size=(SIZE, SIZE),
        )
        affine_params = affine.get_params(labels)
    else:
        affine = RandomPerspective(
            degrees=0.0,
            translate=0.0,
            scale=(1.0, 1.0),
            shear=0.0,
            perspective=0.0,
            size=(SIZE, SIZE),
        )
        matrix = np.eye(3, dtype=np.float32)
        matrix[0, 0] = float(scale)
        matrix[1, 1] = float(scale)
        matrix[0, 2] = float(tx) - float(scale) * (labels["img"].shape[1] / 2.0)
        matrix[1, 2] = float(ty) - float(scale) * (labels["img"].shape[0] / 2.0)
        affine_params = {
            "M": matrix,
            "scale": float(scale),
            "orig_shape": labels["img"].shape[:2],
            "size": (SIZE, SIZE),
        }
    labels = affine.apply_image(labels, affine_params)
    mask = cv2.warpAffine(
        mask1280,
        affine_params["M"][:2],
        dsize=(SIZE, SIZE),
        flags=cv2.INTER_NEAREST,
        borderValue=GRAY,
    )
    labels = affine.apply_instances(labels, affine_params)
    xyxy = labels["instances"].bboxes
    longs = [_long(box) for box in xyxy]
    return Image.fromarray(labels["img"]), longs, mask, float(affine_params["scale"]), affine_params["M"]


def _photo_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask == PHOTO)
    if xs.size == 0:
        return float("nan"), float("nan")
    return float(xs.mean()), float(ys.mean())


def _stats(longs: list[float], mask: np.ndarray) -> dict:
    values = np.asarray(longs, dtype=np.float64)
    cx, cy = _photo_centroid(mask)
    return {
        "boxes": int(values.size),
        "long_side_median": float(np.median(values)) if values.size else 0.0,
        "long_side_mean": float(values.mean()) if values.size else 0.0,
        "frac_long_side_under_32": float((values < 32).mean()) if values.size else 0.0,
        "frac_photo": float((mask == PHOTO).mean()),
        "frac_gray": float((mask == GRAY).mean()),
        "photo_centroid_x": cx,
        "photo_centroid_y": cy,
    }


def _pool(rows: list[dict]) -> dict:
    def avg(key: str) -> float:
        values = [row[key] for row in rows if not (isinstance(row[key], float) and math.isnan(row[key]))]
        return float(np.mean(values)) if values else float("nan")

    return {
        "mosaics": len(rows),
        "boxes_mean": avg("boxes"),
        "long_side_median_mean": avg("long_side_median"),
        "long_side_mean_mean": avg("long_side_mean"),
        "frac_long_side_under_32_mean": avg("frac_long_side_under_32"),
        "frac_photo_mean": avg("frac_photo"),
        "frac_gray_mean": avg("frac_gray"),
        "photo_centroid_x_mean": avg("photo_centroid_x"),
        "photo_centroid_y_mean": avg("photo_centroid_y"),
    }


def _sheet(ours: Image.Image, ultra: Image.Image, ours_mask: np.ndarray, ultra_mask: np.ndarray, title: str) -> Image.Image:
    def paint(mask: np.ndarray) -> Image.Image:
        color = np.zeros((*mask.shape, 3), np.uint8)
        color[mask == PHOTO] = (40, 40, 40)
        color[mask == GRAY] = (114, 114, 114)
        return Image.fromarray(color)

    gap = 8
    canvas = Image.new("RGB", (SIZE * 2 + gap, SIZE * 2 + gap + 24), (20, 20, 20))
    canvas.paste(ours, (0, 24))
    canvas.paste(ultra, (SIZE + gap, 24))
    canvas.paste(paint(ours_mask), (0, SIZE + gap + 24))
    canvas.paste(paint(ultra_mask), (SIZE + gap, SIZE + gap + 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), title, fill=(255, 220, 0))
    draw.text((8, 28), "ours", fill=(255, 220, 0))
    draw.text((SIZE + gap + 8, 28), "ultralytics", fill=(255, 220, 0))
    return canvas


def main() -> None:
    dataset = CocoDetectionDataset(DATA, split="train", filter_empty_gt=True)
    groups = []
    bucket = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if not sample.annotations:
            continue
        bucket.append(sample)
        if len(bucket) == 4:
            groups.append(bucket)
            bucket = []
        if len(groups) == N_GROUPS:
            break
    if len(groups) < N_GROUPS:
        raise RuntimeError(f"Need {N_GROUPS} mosaics, found {len(groups)}")

    matched_ours = []
    matched_ultra = []
    train_ours = []
    train_ultra = []
    dezoom_ours = []
    dezoom_ultra = []
    first = {}
    for group_index, group in enumerate(groups):
        center = (SIZE, SIZE)
        ours_image, ours_boxes, ours_mask = _ours(group, center=center, scale=1.0, tx=320.0, ty=320.0)
        ultra_image, ultra_longs, ultra_mask, _, _ = _run_ultra(
            group, center=center, scale=1.0, tx=320.0, ty=320.0
        )
        matched_ours.append(_stats([_long(box) for box in ours_boxes], ours_mask))
        matched_ultra.append(_stats(ultra_longs, ultra_mask))
        if group_index == 0:
            first["matched"] = (ours_image, ultra_image, ours_mask, ultra_mask)

        random.seed(1000 + group_index)
        center = (
            int(random.uniform(0.5, 1.5) * SIZE),
            int(random.uniform(0.5, 1.5) * SIZE),
        )
        scale = random.uniform(0.5, 1.5)
        tx = random.uniform(0.4, 0.6) * SIZE
        ty = random.uniform(0.4, 0.6) * SIZE
        ours_image, ours_boxes, ours_mask = _ours(group, center=center, scale=scale, tx=tx, ty=ty)
        ultra_image, ultra_longs, ultra_mask, _, _ = _run_ultra(
            group, center=center, scale=scale, tx=tx, ty=ty
        )
        train_ours.append(_stats([_long(box) for box in ours_boxes], ours_mask))
        train_ultra.append(_stats(ultra_longs, ultra_mask))
        if group_index == 0:
            first["train"] = (ours_image, ultra_image, ours_mask, ultra_mask)

        # Forced dezoom with a large shift: content should sit left of center.
        ours_image, ours_boxes, ours_mask = _ours(group, center=center, scale=0.5, tx=256.0, ty=320.0)
        ultra_image, ultra_longs, ultra_mask, _, _ = _run_ultra(
            group, center=center, scale=0.5, tx=256.0, ty=320.0
        )
        dezoom_ours.append(_stats([_long(box) for box in ours_boxes], ours_mask))
        dezoom_ultra.append(_stats(ultra_longs, ultra_mask))
        if group_index == 0:
            first["dezoom"] = (ours_image, ultra_image, ours_mask, ultra_mask)
        print(f"mosaic {group_index + 1}/{N_GROUPS}", flush=True)

    report = {
        "groups": N_GROUPS,
        "images": N_GROUPS * 4,
        "matched_center_scale1": {"ours": _pool(matched_ours), "ultralytics": _pool(matched_ultra)},
        "same_random_draw": {"ours": _pool(train_ours), "ultralytics": _pool(train_ultra)},
        "dezoom_scale_0_5_tx_256": {
            "ours": _pool(dezoom_ours),
            "ultralytics": _pool(dezoom_ultra),
            "note": "scale 0.5 and tx=256 (40% of 640). A centered dezoom would put the photo centroid near 320. Keeping the shift moves it left.",
        },
        "mask_note": "Photo pixels are code 1. Mosaic fill and affine fill are code 3. No letterbox black remains on this path.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    _sheet(*first["matched"], "matched: center 640, scale 1").save(OUT / "mosaic_matched.png")
    _sheet(*first["train"], "same random draw: center, scale, translate").save(OUT / "mosaic_train.png")
    _sheet(*first["dezoom"], "dezoom scale 0.5, tx=256: shift kept").save(OUT / "mosaic_dezoom.png")
    (OUT / "mosaic_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

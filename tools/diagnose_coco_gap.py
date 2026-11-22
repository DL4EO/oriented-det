"""CPU diagnosis of the COCO YOLO-OBB run versus Ultralytics small.

Does not touch the GPU. The live training process keeps the device.

Writes ``runs/coco_gap/report.json`` and ``runs/coco_gap/pixels.png``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from oriented_det.data.coco import CocoDetectionDataset
from oriented_det.data.evaluation import Detection, GroundTruth, compute_oriented_map
from oriented_det.data.preprocessing import apply_pad_mode
from oriented_det.data.transforms import create_albumentations_augmentation
from oriented_det.geometry.rbox import RBox
from oriented_det.models.rotated_fcos import generate_fpn_points
from oriented_det.models.utils import derive_fpn_strides_from_grid, extract_backbone_features
from oriented_det.models.yolo_obb import (
    _decode_horizontal,
    build_yolo_obb,
    compute_yolo_obb_loss,
    obb_to_horizontal_xywh,
    task_aligned_assign_single,
)
from oriented_det.runtime.collate import create_collate_fn
from oriented_det.train.config import TrainingExperimentConfig
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.tal import TaskAlignedAssigner

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/yolo_obb/20261027-055950"
CKPT = RUN / "checkpoints/best_mAP_0.23.pth"
CONFIG = ROOT / "configs/yolo_obb/coco_300e_s.json"
OUT = ROOT / "runs/coco_gap"
N_IMAGES = 128
IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
MEAN = [0.470113, 0.446900, 0.407686]
STD = [0.276232, 0.271676, 0.286569]


def _visual_xywh(box: RBox) -> list[float]:
    pts = box.to_polygon().points
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))]


def _xyxy_from_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def _load_model(class_names: list[str]) -> torch.nn.Module:
    cfg = TrainingExperimentConfig.load(CONFIG)
    model = build_yolo_obb(cfg.model, num_classes=len(class_names))
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=True)
    model.score_threshold = 0.001
    model.final_nms_iou_threshold = 0.7
    model.max_detections_per_image = 300
    model.final_nms_use_cpu = True
    model.eval()
    return model


def _val_collate(class_names: list[str]):
    class_map = {name: i + 1 for i, name in enumerate(class_names)}
    return create_collate_fn(
        class_map,
        augmentation=None,
        normalize=True,
        resize_mode="pad",
        resize_to=(640, 640),
        pad_size_divisor=32,
        enable_flip_horizontal=False,
        enable_flip_vertical=False,
        enable_flip_diagonal=False,
        normalize_mean=MEAN,
        normalize_std=STD,
        enable_mosaic=False,
        random_scale=0.0,
        random_translate=0.0,
    )


def _letterbox_gray(image: Image.Image, size: int = 640, color: int = 114) -> Image.Image:
    w, h = image.size
    scale = size / max(w, h, 1)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = image.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (color, color, color))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def _tensor_to_pil(tensor: torch.Tensor, mean: list[float] | None, std: list[float] | None) -> Image.Image:
    image = tensor.detach().float().cpu()
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean).view(3, 1, 1)
        std_t = torch.tensor(std).view(3, 1, 1)
        image = image * std_t + mean_t
    image = image.clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(image)


def measure_pixels(sample) -> dict:
    raw = Image.open(sample.image_path).convert("RGB")
    ours_pil, _ = apply_pad_mode(raw, [], (640, 640)).image, None
    ours_pil = apply_pad_mode(raw, [], (640, 640)).image
    ultra = _letterbox_gray(raw)
    hsv = create_albumentations_augmentation(
        p_brightness_contrast=0,
        p_gamma=0,
        p_noise=0,
        p_blur=0,
        p_clahe=0,
        p_hsv=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )
    hsv_out = hsv(raw.copy())
    import numpy as np
    from torchvision.transforms.functional import normalize, to_tensor

    ours_t = normalize(to_tensor(ours_pil), MEAN, STD)
    ultra_t = to_tensor(ultra)
    hsv_max = max(max(pixel) for pixel in hsv_out.getdata())
    sheet = Image.new("RGB", (640 * 2 + 16, 640), (32, 32, 32))
    sheet.paste(ours_pil, (0, 0))
    sheet.paste(ultra, (656, 0))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), "ours pad 0", fill=(255, 255, 0))
    draw.text((664, 8), "ultralytics pad 114", fill=(255, 255, 0))
    OUT.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT / "pixels.png")
    _tensor_to_pil(ours_t, MEAN, STD).save(OUT / "pixels_ours_denorm.png")
    _tensor_to_pil(ultra_t, None, None).save(OUT / "pixels_ultra.png")
    border = ours_t[:, :2, :].mean().item()
    return {
        "hsv_runs_on_uint8_before_normalize": isinstance(hsv_out, Image.Image) and hsv_out.mode == "RGB",
        "hsv_sample_pixel_max": int(hsv_max),
        "ours_normalized_mean": [float(x) for x in ours_t.mean(dim=(1, 2))],
        "ours_normalized_std": [float(x) for x in ours_t.std(dim=(1, 2))],
        "ultra_div255_mean": [float(x) for x in ultra_t.mean(dim=(1, 2))],
        "ultra_div255_std": [float(x) for x in ultra_t.std(dim=(1, 2))],
        "ours_corner_normalized": float(border),
        "ultra_corner_div255": float(ultra_t[:, 0, 0].mean().item()),
        "pad_value_ours": 0,
        "pad_value_ultra": 114,
        "numpy": np.__version__,
    }


def _forward_raw(model, image: torch.Tensor):
    image_sizes = [(int(image.shape[-2]), int(image.shape[-1]))]
    features = extract_backbone_features(
        model.backbone,
        [image],
        use_checkpoint=False,
        training=False,
        include_pool_level=False,
    )
    sizes = [(f.shape[2], f.shape[3]) for f in features]
    strides = derive_fpn_strides_from_grid(image_sizes[0], sizes, configured=model.fpn_strides)
    cls_scores, bbox_preds, angle_preds = model.head(features)
    if model.force_horizontal:
        angle_preds = [pred.new_zeros(pred.shape) for pred in angle_preds]
    points = generate_fpn_points(sizes, strides, dtype=bbox_preds[0].dtype, device=image.device)
    return cls_scores, bbox_preds, angle_preds, points, strides


def _flat(levels: list[torch.Tensor], img_idx: int = 0) -> torch.Tensor:
    return torch.cat(
        [level[img_idx].permute(1, 2, 0).reshape(-1, level.shape[1]) for level in levels],
        dim=0,
    )


def measure_assign(model, images, targets) -> dict:
    image = images[0]
    cls_scores, bbox_preds, angle_preds, points, strides = _forward_raw(model, image)
    gt = obb_to_horizontal_xywh(targets[0]["rboxes"])
    labels = targets[0]["labels"]
    concat_points = torch.cat(points, dim=0)
    num_per = [p.size(0) for p in points]
    stride_t = torch.cat(
        [concat_points.new_full((n,), float(s)) for n, s in zip(num_per, strides)],
        dim=0,
    )
    cls_i = _flat(cls_scores)
    bins_i = _flat(bbox_preds)
    decoded = _decode_horizontal(bins_i, concat_points, stride_t, model.reg_max)
    _, soft, _, pos = task_aligned_assign_single(
        concat_points,
        cls_i,
        decoded,
        gt,
        labels,
        model.num_classes,
        topk=model.assign_topk,
        alpha=model.tal_alpha,
        beta=model.tal_beta,
        reg_max=model.reg_max,
        strides=stride_t,
    )
    our_loss = compute_yolo_obb_loss(
        cls_scores,
        bbox_preds,
        angle_preds,
        points,
        strides,
        [gt],
        [labels],
        model.num_classes,
        reg_max=model.reg_max,
        bbox_coder=model.bbox_coder,
        box_reg_weight=model.box_reg_weight,
        dfl_weight=model.dfl_weight,
        angle_weight=model.angle_weight,
        assign_topk=model.assign_topk,
        tal_alpha=model.tal_alpha,
        tal_beta=model.tal_beta,
        use_ciou=True,
    )
    pd_xyxy = _xyxy_from_cxcywh(decoded.detach())
    gt_xyxy = _xyxy_from_cxcywh(gt.detach())
    n_gt = int(gt_xyxy.shape[0])
    assigner = TaskAlignedAssigner(
        topk=int(model.assign_topk),
        num_classes=int(model.num_classes),
        alpha=float(model.tal_alpha),
        beta=float(model.tal_beta),
    )
    gt_labels = (labels.view(1, n_gt, 1) - 1).clamp(min=0).float()
    mask_gt = torch.ones(1, n_gt, 1)
    _, _, target_scores, fg_mask, _ = assigner(
        cls_i.detach().sigmoid().unsqueeze(0),
        pd_xyxy.unsqueeze(0),
        concat_points,
        gt_labels,
        gt_xyxy.unsqueeze(0),
        mask_gt,
    )
    ours_idx = set(pos.nonzero(as_tuple=False).flatten().tolist())
    ultra_idx = set(fg_mask[0].nonzero(as_tuple=False).flatten().tolist())
    inter = len(ours_idx & ultra_idx)
    union = len(ours_idx | ultra_idx) or 1
    stride_anchor = (concat_points / stride_t.unsqueeze(-1)).unsqueeze(0)
    pred_xyxy_stride = (pd_xyxy / stride_t.unsqueeze(-1)).unsqueeze(0)
    _, target_bboxes, target_scores, fg_mask, _ = assigner(
        cls_i.detach().sigmoid().unsqueeze(0),
        pd_xyxy.unsqueeze(0),
        concat_points,
        gt_labels,
        gt_xyxy.unsqueeze(0),
        mask_gt,
    )
    target_sum = target_scores.sum().clamp(min=1)
    bce = F.binary_cross_entropy_with_logits(cls_i.unsqueeze(0), target_scores.float(), reduction="sum")
    box_term, dfl_term = BboxLoss(model.reg_max)(
        bins_i.unsqueeze(0),
        pred_xyxy_stride,
        stride_anchor,
        target_bboxes / stride_t.view(1, -1, 1),
        target_scores.float(),
        target_sum,
        fg_mask,
        torch.tensor([640.0, 640.0]),
        stride_t,
    )
    batch = 1.0
    ultra = {
        "loss_box": float(box_term * 7.5 * batch),
        "loss_cls": float(bce / target_sum * 0.5 * batch),
        "loss_dfl": float(dfl_term * 1.5 * batch),
        "positives": len(ultra_idx),
        "soft_target_sum": float(target_scores.sum()),
    }
    return {
        "gt_boxes": n_gt,
        "ours_positives": len(ours_idx),
        "ultra_positives_topk13_ciou": ultra["positives"],
        "positive_jaccard": inter / union,
        "ours_soft_target_sum": float(soft.sum()),
        "ultra_soft_target_sum": ultra["soft_target_sum"],
        "ours_loss_box": float(our_loss["loss_box_reg"]),
        "ours_loss_cls": float(our_loss["loss_classifier"]),
        "ours_loss_dfl": float(our_loss["loss_dfl"]),
        "ours_loss_angle": float(our_loss["loss_angle"]),
        "ultra_loss_box": ultra["loss_box"],
        "ultra_loss_cls": ultra["loss_cls"],
        "ultra_loss_dfl": ultra["loss_dfl"],
    }


def measure_map(model, dataset, class_names: list[str], collate) -> dict:
    detections: dict[str, list[Detection]] = {}
    grounds: dict[str, list[GroundTruth]] = {}
    coco_images = []
    coco_anns = []
    coco_dets = []
    ann_id = 1
    for index in range(N_IMAGES):
        sample = dataset[index]
        images, targets = collate([sample])
        image_id = str(targets[0]["image_id"])
        with torch.no_grad():
            output = model([images[0]])[0]
        dets: list[Detection] = []
        for rbox, score, label in zip(output["rboxes"], output["scores"], output["labels"]):
            label_i = int(label)
            name = class_names[label_i - 1]
            score_f = float(score)
            dets.append(Detection(rbox=rbox, score=score_f, class_id=label_i, class_name=name, image_id=image_id))
            xywh = _visual_xywh(rbox)
            coco_dets.append(
                {
                    "image_id": index + 1,
                    "category_id": label_i,
                    "bbox": xywh,
                    "score": score_f,
                }
            )
        gts: list[GroundTruth] = []
        boxes = targets[0]["rboxes"]
        labels = targets[0]["labels"]
        for row, label in zip(boxes, labels):
            label_i = int(label)
            name = class_names[label_i - 1]
            rbox = RBox(float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]))
            gts.append(GroundTruth(rbox=rbox, class_id=label_i, class_name=name, image_id=image_id))
            coco_anns.append(
                {
                    "id": ann_id,
                    "image_id": index + 1,
                    "category_id": label_i,
                    "bbox": _visual_xywh(rbox),
                    "area": float(row[2] * row[3]),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
        detections[image_id] = dets
        grounds[image_id] = gts
        coco_images.append({"id": index + 1, "width": 640, "height": 640, "file_name": image_id})
        if (index + 1) % 16 == 0:
            print(f"  inferred {index + 1}/{N_IMAGES}", flush=True)
    ours_map, _, _ = compute_oriented_map(
        detections,
        grounds,
        iou_threshold=0.5,
        class_names=class_names,
        show_progress=False,
        use_exact_rotated_iou=True,
        iou_thresholds=IOU_THRESHOLDS,
    )
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": coco_images,
        "annotations": coco_anns,
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(class_names)],
    }
    coco_gt.createIndex()
    if coco_dets:
        coco_dt = coco_gt.loadRes(coco_dets)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        coco_map = float(evaluator.stats[0])
        coco_map50 = float(evaluator.stats[1])
    else:
        coco_map = 0.0
        coco_map50 = 0.0
    return {
        "images": N_IMAGES,
        "checkpoint": str(CKPT),
        "in_train_map_50_95": float(ours_map),
        "pycocotools_map_50_95": coco_map,
        "pycocotools_map50": coco_map50,
        "delta_pycocotools_minus_in_train": coco_map - float(ours_map),
    }


def measure_arch() -> dict:
    cfg = TrainingExperimentConfig.load(CONFIG)
    model = build_yolo_obb(cfg.model, num_classes=80)
    params = sum(p.numel() for p in model.parameters())
    log = (RUN / "train.log").read_text(errors="replace")
    wd_line = next((line.strip() for line in log.splitlines() if "Weight decay groups" in line), "")
    return {
        "parameters": int(params),
        "published_yolov8s_parameters": 11_200_000,
        "weight_decay_log_line": wd_line,
        "backbone": "yolo_obb_s C2f depth 0.33 width 0.50",
    }


def main() -> None:
    random.seed(0)
    torch.manual_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = CocoDetectionDataset("/path/to/data/coco", split="val", filter_empty_gt=False)
    names = dataset.get_class_names()
    print("loading checkpoint", flush=True)
    model = _load_model(names)
    collate = _val_collate(names)
    print("pixels", flush=True)
    pixels = measure_pixels(dataset[0])
    print("assign", flush=True)
    images, targets = collate([dataset[0]])
    # Skip an empty first image so the assigner has boxes.
    chosen = 0
    for chosen in range(20):
        images, targets = collate([dataset[chosen]])
        if int(targets[0]["rboxes"].shape[0]) > 0:
            break
    assign = measure_assign(model, images, targets)
    assign["image_index"] = chosen
    print("map", flush=True)
    scored = measure_map(model, dataset, names, collate)
    report = {
        "metric": scored,
        "assign": assign,
        "pixels": pixels,
        "architecture": measure_arch(),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

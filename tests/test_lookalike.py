"""Tests for reserved lookalike hard-negative routing."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from oriented_det.data.dota import DOTAAnnotation, DOTASample
from oriented_det.data.lookalike import (
    LOOKALIKE_CLASS_NAME,
    filter_semantic_class_names,
    is_lookalike_class_name,
    resolve_lookalike_label_set,
)
from oriented_det.models.oriented_roi import (
    compute_oriented_roi_loss,
    match_oriented_proposals_to_gt,
)
from oriented_det.models.oriented_rpn import match_oriented_anchors_to_gt
from oriented_det.models.rotated_fcos import assign_fcos_targets_single
from oriented_det.models.utils import (
    force_lookalike_to_background,
    prepare_targets,
    sample_fg_bg_indices,
)
from oriented_det.runtime.collate import create_collate_fn
from oriented_det.train.config import TrainingExperimentConfig


def test_lookalike_reserved_name_always_included():
    assert LOOKALIKE_CLASS_NAME == "lookalike"
    assert resolve_lookalike_label_set(None) == {"lookalike"}
    assert resolve_lookalike_label_set(["Confuser"]) == {"lookalike", "Confuser"}
    assert is_lookalike_class_name("lookalike")
    assert not is_lookalike_class_name("ship")
    assert filter_semantic_class_names(["ship", "lookalike", "car"]) == ["car", "ship"]
    # Aliases must not enter the semantic class map either.
    assert filter_semantic_class_names(
        ["ship", "Confuser"], lookalike_labels=["Confuser"]
    ) == ["ship"]


def test_dataset_config_accepts_lookalike_labels(tmp_path: Path):
    p = tmp_path / "cfg.json"
    p.write_text(
        json.dumps(
            {
                "model_type": "rotated_faster_rcnn",
                "dataset": {
                    "data_root": str(tmp_path),
                    "lookalike_labels": ["hard_neg"],
                    "map_labels": {"Confuser": "lookalike"},
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = TrainingExperimentConfig.load(p)
    assert cfg.dataset.lookalike_labels == ["hard_neg"]
    assert cfg.dataset.map_labels == {"Confuser": "lookalike"}
    # Reserved token must never become a class_map entry from get_class_names filtering.
    names = filter_semantic_class_names(
        ["ship", "lookalike", "hard_neg"],
        lookalike_labels=cfg.dataset.lookalike_labels,
    )
    assert "lookalike" not in names
    assert "hard_neg" not in names
    assert names == ["ship"]


def _ann(class_name: str, cx: float, cy: float, w: float = 8.0, h: float = 8.0, difficult: int = 0) -> DOTAAnnotation:
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy - h / 2
    x3, y3 = cx + w / 2, cy + h / 2
    x4, y4 = cx - w / 2, cy + h / 2
    return DOTAAnnotation.from_line(
        f"{x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4} {class_name} {int(difficult)}"
    )


def test_dota_filter_keeps_lookalike_over_ignore_and_allowed():
    anns = (
        _ann("ship", 10, 10),
        _ann("lookalike", 20, 20),
        _ann("junk", 30, 30),
    )
    sample = DOTASample(image_path=Path("x.png"), width=64, height=64, annotations=anns)
    filtered = sample.filter_by_class(
        allowed_classes=["ship"],
        ignore_labels=["lookalike", "junk"],
    )
    names = [a.class_name for a in filtered.annotations]
    assert "ship" in names
    assert "lookalike" in names
    assert "junk" not in names


def test_airbus_map_confuser_to_lookalike_excluded_from_class_names(tmp_path: Path):
    pytest.importorskip("shapely")
    from oriented_det.data.airbus_playground import (
        AirbusPlaygroundCSVDataset,
        generate_airbus_playground_csvs,
    )

    dataset_id = "dataset-L"
    zone_id = "zone-1"
    image_id = "image-1"
    for tile_id, label in [("tile-1", "car"), ("tile-2", "Confuser")]:
        img_path = tmp_path / dataset_id / "samples" / zone_id / image_id / f"{tile_id}.jpg"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color=(128, 128, 128)).save(img_path)
        label_path = tmp_path / dataset_id / "labels" / zone_id / f"{tile_id}.json"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[10, 10], [20, 10], [20, 18], [10, 18], [10, 10]]],
                    },
                    "properties": {"tags": [label]},
                }
            ],
        }
        label_path.write_text(json.dumps(payload), encoding="utf-8")

    generate_airbus_playground_csvs(tmp_path, num_splits=2, seed=1)

    with (tmp_path / "annotations.csv").open("r", encoding="utf-8", newline="") as f:
        raw = [r["class_name"] for r in csv.DictReader(f)]
    assert "Confuser" in raw

    ds_all = AirbusPlaygroundCSVDataset(
        data_root=tmp_path,
        split="train",
        annotations_file="annotations.csv",
        split_file="split.csv",
        map_labels={"Confuser": "lookalike"},
        train_includes_val=True,
    )
    kept = set()
    for sample in ds_all:
        for ann in sample.annotations:
            kept.add(ann.class_name)
    assert "lookalike" in kept
    assert "lookalike" not in ds_all.get_class_names()
    assert "car" in ds_all.get_class_names()


def test_collate_routes_lookalike_and_applies_shared_geometry(tmp_path: Path, monkeypatch):
    import oriented_det.data.flips as flips_mod

    img_path = tmp_path / "tile.png"
    Image.new("RGB", (64, 64), color=(40, 40, 40)).save(img_path)
    sample = DOTASample(
        image_path=img_path,
        width=64,
        height=64,
        annotations=(
            _ann("ship", 16, 16),
            _ann("lookalike", 48, 48),
        ),
    )
    collate = create_collate_fn(
        {"ship": 1},
        normalize=False,
        resize_mode="fixed",
        resize_to=(64, 64),
        pad_size_divisor=1,
        enable_flip_horizontal=True,
        enable_flip_vertical=False,
        enable_flip_diagonal=False,
        enable_random_rotate=False,
        lookalike_labels=None,
    )
    # Force the single enabled flip mode (horizontal) so both lists share geometry.
    monkeypatch.setattr(flips_mod.random, "random", lambda: 0.0)
    _, targets = collate([sample])
    tgt = targets[0]
    assert tgt["labels"].tolist() == [1]
    assert tgt["rboxes"].shape == (1, 5)
    assert tgt["rboxes_lookalike"].shape == (1, 5)
    # Horizontal flip of cx=16 on W=64 → cx≈48; lookalike cx=48 → ≈16.
    assert abs(float(tgt["rboxes"][0, 0]) - 48.0) < 1e-3
    assert abs(float(tgt["rboxes_lookalike"][0, 0]) - 16.0) < 1e-3


def test_prepare_targets_returns_lookalike_list():
    targets = [
        {
            "rboxes": torch.zeros((1, 5)),
            "labels": torch.tensor([1]),
            "rboxes_lookalike": torch.ones((2, 5)),
        }
    ]
    boxes, labels, ignore, lookalike = prepare_targets(targets)
    assert boxes[0].shape == (1, 5)
    assert labels[0].tolist() == [1]
    assert ignore[0].shape == (0, 5)
    assert lookalike[0].shape == (2, 5)


def test_rpn_lookalike_forces_background_not_ignore():
    anchors = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0, 0.0],
            [50.0, 50.0, 20.0, 20.0, 0.0],
            [200.0, 200.0, 10.0, 10.0, 0.0],
        ]
    )
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])
    look = torch.tensor([[50.0, 50.0, 25.0, 25.0, 0.0]])
    labels, matched = match_oriented_anchors_to_gt(
        anchors,
        gt,
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.3,
        gt_boxes_lookalike=look,
        lookalike_iou_threshold=0.3,
    )
    assert labels[0].item() == 1
    assert labels[1].item() == 0  # lookalike overlap → background, not -1
    assert matched[0].item() == 0


def test_roi_lookalike_does_not_override_positive():
    proposals = torch.tensor(
        [
            [0.0, 0.0, 20.0, 20.0, 0.0],
            [50.0, 50.0, 20.0, 20.0, 0.0],
        ]
    )
    gt = torch.tensor([[0.0, 0.0, 20.0, 20.0, 0.0]])
    gt_labels = torch.tensor([3], dtype=torch.int64)
    # Overlapping lookalike on the positive proposal region should not demote it.
    look = torch.tensor([[0.0, 0.0, 20.0, 20.0, 0.0]])
    labels, _, _ = match_oriented_proposals_to_gt(
        proposals,
        gt,
        gt_labels,
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.3,
        gt_boxes_lookalike=look,
        lookalike_iou_threshold=0.3,
    )
    assert labels[0].item() == 3


def test_roi_lookalike_wins_over_ignore():
    proposals = torch.tensor([[50.0, 50.0, 20.0, 20.0, 0.0]])
    gt = torch.zeros((0, 5))
    gt_labels = torch.tensor([], dtype=torch.int64)
    ignore = torch.tensor([[50.0, 50.0, 25.0, 25.0, 0.0]])
    look = torch.tensor([[50.0, 50.0, 25.0, 25.0, 0.0]])
    labels, _, _ = match_oriented_proposals_to_gt(
        proposals,
        gt,
        gt_labels,
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.3,
        gt_boxes_ignore=ignore,
        ignore_iou_threshold=0.3,
        gt_boxes_lookalike=look,
        lookalike_iou_threshold=0.3,
    )
    assert labels[0].item() == 0


def test_roi_loss_runs_on_lookalike_only_image():
    n = 8
    num_classes = 1
    proposals = torch.zeros((n, 5))
    for i in range(n):
        proposals[i] = torch.tensor([10.0 + i * 5.0, 10.0, 12.0, 12.0, 0.0])
    look = torch.tensor([[15.0, 10.0, 20.0, 20.0, 0.0]])
    class_logits = torch.zeros((n, num_classes + 1))
    box_regression = torch.zeros((n, num_classes * 5))
    losses = compute_oriented_roi_loss(
        class_logits=class_logits,
        box_regression=box_regression,
        proposals=proposals,
        gt_boxes=torch.zeros((0, 5)),
        gt_labels=torch.tensor([], dtype=torch.int64),
        gt_boxes_lookalike=look,
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.3,
        batch_size_per_image=4,
        num_classes=num_classes,
    )
    assert "loss_classifier" in losses
    assert torch.isfinite(losses["loss_classifier"])


def test_fcos_lookalike_points_stay_background_not_ignore():
    points = torch.tensor([[50.0, 50.0], [0.0, 0.0], [200.0, 200.0]])
    gt = torch.zeros((0, 5))
    gt_labels = torch.tensor([], dtype=torch.int64)
    ignore = torch.tensor([[50.0, 50.0, 30.0, 30.0, 0.0]])
    look = torch.tensor([[50.0, 50.0, 30.0, 30.0, 0.0]])
    ranges = torch.tensor([[0.0, 1e8]]).expand(3, 2)
    labels, _, _, pos = assign_fcos_targets_single(
        points=points,
        gt_bboxes=gt,
        gt_labels=gt_labels,
        regress_ranges=ranges,
        num_points_per_lvl=[3],
        strides=[8.0],
        num_classes=2,
        center_sampling=False,
        gt_bboxes_ignore=ignore,
        gt_bboxes_lookalike=look,
    )
    assert labels[0].item() == 2  # background K, not -1
    assert not pos[0]


def test_sample_prefers_lookalike_negatives():
    labels = torch.zeros(20, dtype=torch.int64)
    labels[0] = 1
    boxes = torch.zeros((20, 5))
    for i in range(20):
        boxes[i] = torch.tensor([float(i * 100), 0.0, 10.0, 10.0, 0.0])
    look = torch.tensor([[100.0, 0.0, 15.0, 15.0, 0.0]])  # overlaps index 1
    torch.manual_seed(0)
    sampled_fg, sampled_bg = sample_fg_bg_indices(
        labels,
        num_fg=1,
        num_bg=4,
        device=torch.device("cpu"),
        boxes=boxes,
        gt_boxes_lookalike=look,
        lookalike_iou_threshold=0.1,
        prefer_lookalike_bg_fraction=0.5,
    )
    assert 0 in sampled_fg.tolist()
    assert 1 in sampled_bg.tolist()


def test_force_lookalike_helper_protects_positives():
    labels = torch.tensor([1, -1, 0], dtype=torch.int64)
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0, 0.0],
            [50.0, 50.0, 10.0, 10.0, 0.0],
            [50.0, 50.0, 10.0, 10.0, 0.0],
        ]
    )
    look = torch.tensor([[50.0, 50.0, 20.0, 20.0, 0.0]])
    forced = force_lookalike_to_background(
        labels, boxes, look, iou_threshold=0.2, positive_mask=(labels > 0)
    )
    assert labels[0].item() == 1
    assert labels[1].item() == 0
    assert labels[2].item() == 0
    assert forced[1] and forced[2]
    assert not forced[0]


def test_collate_routes_difficult_to_rboxes_ignore_not_lookalike(tmp_path: Path):
    img_path = tmp_path / "tile.png"
    Image.new("RGB", (64, 64), color=(40, 40, 40)).save(img_path)
    sample = DOTASample(
        image_path=img_path,
        width=64,
        height=64,
        annotations=(
            _ann("car", 16, 16, difficult=0),
            _ann("car", 48, 48, difficult=1),
        ),
    )
    collate = create_collate_fn(
        {"car": 1},
        normalize=False,
        resize_mode="fixed",
        resize_to=(64, 64),
        pad_size_divisor=1,
        enable_flip_horizontal=False,
        enable_flip_vertical=False,
        enable_flip_diagonal=False,
        enable_random_rotate=False,
        difficult_strategy="ignore",
        lookalike_labels=None,
    )
    _, targets = collate([sample])
    tgt = targets[0]
    assert tgt["labels"].tolist() == [1]
    assert tgt["rboxes"].shape == (1, 5)
    assert tgt["rboxes_ignore"].shape == (1, 5)
    assert tgt["labels_ignore"].tolist() == [1]
    assert tgt["rboxes_lookalike"].shape == (0, 5)


def test_fcos_ignore_points_get_minus_one_not_bg():
    points = torch.tensor([[50.0, 50.0], [0.0, 0.0], [200.0, 200.0]])
    gt = torch.zeros((0, 5))
    gt_labels = torch.tensor([], dtype=torch.int64)
    ignore = torch.tensor([[50.0, 50.0, 30.0, 30.0, 0.0]])
    ranges = torch.tensor([[0.0, 1e8]]).expand(3, 2)
    labels, _, _, pos = assign_fcos_targets_single(
        points=points,
        gt_bboxes=gt,
        gt_labels=gt_labels,
        regress_ranges=ranges,
        num_points_per_lvl=[3],
        strides=[8.0],
        num_classes=2,
        center_sampling=False,
        gt_bboxes_ignore=ignore,
    )
    assert labels[0].item() == -1
    assert labels[1].item() == 2  # background
    assert not pos[0]


def test_fcos_ignore_does_not_override_foreground():
    points = torch.tensor([[50.0, 50.0]])
    gt = torch.tensor([[50.0, 50.0, 40.0, 40.0, 0.0]])
    gt_labels = torch.tensor([1], dtype=torch.int64)  # 1-indexed class
    ignore = torch.tensor([[50.0, 50.0, 40.0, 40.0, 0.0]])
    ranges = torch.tensor([[0.0, 1e8]])
    labels, _, _, pos = assign_fcos_targets_single(
        points=points,
        gt_bboxes=gt,
        gt_labels=gt_labels,
        regress_ranges=ranges,
        num_points_per_lvl=[1],
        strides=[8.0],
        num_classes=2,
        center_sampling=False,
        gt_bboxes_ignore=ignore,
    )
    assert labels[0].item() == 0  # FG class 0, not -1
    assert pos[0]


def test_rpn_ignore_marks_non_positive_as_minus_one():
    anchors = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0, 0.0],
            [50.0, 50.0, 20.0, 20.0, 0.0],
            [200.0, 200.0, 10.0, 10.0, 0.0],
        ]
    )
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])
    ignore = torch.tensor([[50.0, 50.0, 25.0, 25.0, 0.0]])
    labels, matched = match_oriented_anchors_to_gt(
        anchors,
        gt,
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.3,
        gt_boxes_ignore=ignore,
        ignore_iou_threshold=0.3,
    )
    assert labels[0].item() == 1
    assert labels[1].item() == -1  # ignore overlap → don't-care, not background
    assert matched[0].item() == 0


def test_dataset_config_accepts_difficult_tags(tmp_path: Path):
    from oriented_det.train.config import TrainingExperimentConfig

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "model_type": "rotated_fcos",
                "dataset": {
                    "data_root": str(tmp_path),
                    "format": "airbus_playground",
                    "annotations_file": "annotations.csv",
                    "split_file": "split.csv",
                    "difficult_tags": ["Partially Hidden"],
                    "difficult_strategy": "ignore",
                    "map_labels": {"car, van and pickup": "car"},
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = TrainingExperimentConfig.load(cfg_path)
    assert cfg.dataset.difficult_tags == ["Partially Hidden"]
    assert cfg.dataset.difficult_strategy == "ignore"
    assert cfg.dataset.map_labels["car, van and pickup"] == "car"

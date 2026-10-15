"""COCO horizontal pretrain: bbox loader, angle isolation, 80-to-15 transfer."""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

from oriented_det.data.build import build_split_dataset
from oriented_det.geometry import RBox
from oriented_det.data.coco import annotation_from_coco_bbox, coco_category_names
from oriented_det.train.config import DatasetConfig, TrainingExperimentConfig


def test_xywh_to_horizontal_le90():
    ann = annotation_from_coco_bbox(10, 20, 30, 40, "person")
    assert ann.rbox.cx == pytest.approx(25.0)
    assert ann.rbox.cy == pytest.approx(40.0)
    assert ann.rbox.width == pytest.approx(30.0)
    assert ann.rbox.height == pytest.approx(40.0)
    assert ann.rbox.angle == pytest.approx(0.0)


def test_category_id_gaps_do_not_leave_label_holes():
    id_to_name, names = coco_category_names(
        [
            {"id": 12, "name": "stop sign"},
            {"id": 1, "name": "person"},
            {"id": 3, "name": "car"},
        ]
    )
    assert names == ["person", "car", "stop sign"]
    assert id_to_name == {1: "person", 3: "car", 12: "stop sign"}
    labels = {cid: names.index(name) + 1 for cid, name in id_to_name.items()}
    assert labels == {1: 1, 3: 2, 12: 3}


def _write_coco_root(root: Path) -> None:
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("PIL required")
    for folder in ("train2017", "val2017"):
        (root / folder).mkdir(parents=True)
        Image.new("RGB", (80, 60), (1, 2, 3)).save(root / folder / "a.jpg")
        Image.new("RGB", (80, 60), (4, 5, 6)).save(root / folder / "empty.jpg")
    payload = {
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 3, "name": "car"},
        ],
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 80, "height": 60},
            {"id": 2, "file_name": "empty.jpg", "width": 80, "height": 60},
        ],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "category_id": 3,
                "iscrowd": 0,
                "bbox": [10, 20, 30, 40],
                "segmentation": [[0, 0, 70, 0, 70, 10, 0, 50]],
            },
            {
                "id": 11,
                "image_id": 1,
                "category_id": 1,
                "iscrowd": 1,
                "bbox": [0, 0, 5, 5],
            },
        ],
    }
    ann = root / "annotations"
    ann.mkdir()
    text = json.dumps(payload)
    (ann / "instances_train2017.json").write_text(text, encoding="utf-8")
    (ann / "instances_val2017.json").write_text(text, encoding="utf-8")


def test_coco_loader_keeps_empty_images_and_uses_bbox(tmp_path: Path):
    _write_coco_root(tmp_path)
    cfg = DatasetConfig(data_root=tmp_path, format="coco", filter_empty_gt=False)
    train = build_split_dataset(cfg, "train")
    assert train.get_class_names() == ["person", "car"]
    assert len(train) == 2
    labeled = next(sample for sample in train if sample.annotations)
    assert len(labeled.annotations) == 1
    ann = labeled.annotations[0]
    assert ann.class_name == "car"
    assert ann.rbox.angle == pytest.approx(0.0)
    assert ann.rbox.width == pytest.approx(30.0)
    assert ann.rbox.height == pytest.approx(40.0)
    filtered = build_split_dataset(cfg, "train", filter_empty_gt=True)
    assert len(filtered) == 1


def test_force_horizontal_blocks_angle_gradient():
    from oriented_det.models import YoloOBB

    image = torch.rand(3, 256, 256)
    target = {
        "rboxes": [RBox(128, 128, 48, 24, 0.4)],
        "labels": torch.tensor([1], dtype=torch.int64),
    }

    def _angle_grad_sum(model: torch.nn.Module) -> float:
        total = 0.0
        saw = False
        for name, param in model.named_parameters():
            if "angle_head" not in name and "angle_scale" not in name:
                continue
            if param.grad is None:
                continue
            saw = True
            total += float(param.grad.detach().abs().sum())
        return total if saw else 0.0

    held = YoloOBB(
        num_classes=2,
        backbone_name="yolo_obb_n",
        pretrained_backbone=False,
        force_horizontal=True,
        angle_weight=0.0,
    )
    held.train()
    losses = held([image], [target])
    sum(losses.values()).backward()
    assert _angle_grad_sum(held) == 0.0
    assert any(
        param.grad is not None and float(param.grad.abs().sum()) > 0
        for name, param in held.named_parameters()
        if name.startswith("head.reg_heads")
    )

    open_angle = YoloOBB(
        num_classes=2,
        backbone_name="yolo_obb_n",
        pretrained_backbone=False,
        force_horizontal=False,
        angle_weight=0.0,
    )
    open_angle.train()
    sum(open_angle([image], [target]).values()).backward()
    assert _angle_grad_sum(open_angle) > 0.0


def test_coco_checkpoint_replaces_classifier_and_angle_head(tmp_path: Path):
    from oriented_det.models import YoloOBB
    from oriented_det.models.yolo_obb import drop_yolo_obb_class_transfer_keys
    from oriented_det.train.utils import CheckpointManager

    src = YoloOBB(num_classes=80, backbone_name="yolo_obb_s", pretrained_backbone=False)
    dst = YoloOBB(num_classes=15, backbone_name="yolo_obb_s", pretrained_backbone=False)
    src_sd = {key: value.detach().clone() for key, value in src.state_dict().items()}
    for value in src_sd.values():
        if value.is_floating_point():
            value.fill_(4)
    before = {key: value.detach().clone() for key, value in dst.state_dict().items()}
    filtered, skipped = drop_yolo_obb_class_transfer_keys(dst.state_dict(), src_sd)
    assert any(key.endswith("head.angle_heads.0.2.weight") for key in skipped)
    assert any(key.endswith("head.cls_heads.0.2.weight") for key in skipped)
    assert "head.cls_heads.0.0.conv.weight" not in skipped
    assert "head.reg_heads.0.2.weight" not in skipped

    path = tmp_path / "coco.pth"
    torch.save({"model_state_dict": src_sd}, path)
    CheckpointManager(tmp_path).load(path, dst, strict=False)
    loaded = dst.state_dict()
    assert torch.allclose(loaded["head.reg_heads.0.2.weight"], torch.full_like(loaded["head.reg_heads.0.2.weight"], 4))
    assert torch.allclose(
        loaded["head.cls_heads.0.0.conv.weight"],
        torch.full_like(loaded["head.cls_heads.0.0.conv.weight"], 4),
    )
    assert torch.equal(loaded["head.angle_heads.0.2.weight"], before["head.angle_heads.0.2.weight"])
    assert torch.equal(loaded["head.angle_scales.0.scale"], before["head.angle_scales.0.scale"])
    assert torch.equal(loaded["head.cls_heads.0.2.weight"], before["head.cls_heads.0.2.weight"])
    assert loaded["head.cls_heads.0.2.weight"].shape[0] == 15

    same = YoloOBB(num_classes=15, backbone_name="yolo_obb_n", pretrained_backbone=False)
    same_sd = {key: value.detach().clone() for key, value in same.state_dict().items()}
    same_sd["head.angle_scales.0.scale"].fill_(4)
    fresh = YoloOBB(num_classes=15, backbone_name="yolo_obb_n", pretrained_backbone=False)
    kept, same_skipped = drop_yolo_obb_class_transfer_keys(fresh.state_dict(), same_sd)
    assert same_skipped == []
    fresh.load_state_dict(kept, strict=False)
    assert float(fresh.state_dict()["head.angle_scales.0.scale"]) == pytest.approx(4.0)


def test_committed_coco_and_dota_recipes():
    root = Path(__file__).resolve().parents[1]
    coco = TrainingExperimentConfig.load(root / "configs/yolo_obb/coco_300e_s.json")
    dota = TrainingExperimentConfig.load(root / "configs/yolo_obb/dota_le90_1x_s.json")
    finetune = TrainingExperimentConfig.load(root / "configs/yolo_obb/dota_le90_6x_s_coco.json")

    assert coco.dataset.format == "coco"
    assert coco.data_loader.batch_size == 16
    assert coco.training.num_epochs == 300
    assert coco.training.optimizer == "sgd"
    assert coco.training.learning_rate == pytest.approx(0.0025)
    assert coco.training.lr_scheduler_cosine_eta_min == pytest.approx(2.5e-5)
    assert coco.training.lr_warmup_steps == 22179
    assert coco.training.use_amp is True
    assert coco.model.yolo_obb_force_horizontal is True
    assert list(coco.preprocessing.target_size) == [640, 640]
    assert coco.preprocessing.enable_mosaic is True
    assert coco.preprocessing.enable_random_rotate is False
    assert coco.preprocessing.mosaic_disable_last_n_epochs == 10
    assert coco.evaluation.val_map_every_n_epochs == 50
    assert list(coco.preprocessing.normalize_mean) == list(dota.preprocessing.normalize_mean)
    assert list(coco.preprocessing.normalize_std) == list(dota.preprocessing.normalize_std)

    assert finetune.training.num_epochs == 72
    assert finetune.training.lr_warmup_steps == 1000
    assert finetune.model.yolo_obb_angle_weight == pytest.approx(0.2)
    assert finetune.model.yolo_obb_force_horizontal is False
    assert finetune.preprocessing.mosaic_disable_last_n_epochs == 0
    assert list(finetune.preprocessing.target_size) == [1024, 1024]
    assert finetune.checkpoint.load_from_checkpoint is not None

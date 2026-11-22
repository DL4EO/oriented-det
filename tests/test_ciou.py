"""CIoU for horizontal boxes, and the YOLO-OBB switch under force_horizontal."""

import pytest

pytest.importorskip("torch")

import torch

from oriented_det.ops.ciou import ciou_loss_per_box


def test_identical_boxes_are_near_zero():
    boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.float32, requires_grad=True)
    loss = ciou_loss_per_box(boxes, boxes.detach())
    assert loss.shape == (1,)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    loss.backward()
    assert boxes.grad is not None


def test_disjoint_same_aspect_matches_diou_term():
    pred = torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.float32)
    target = torch.tensor([[3.0, 0.0, 2.0, 2.0]], dtype=torch.float32)
    loss = ciou_loss_per_box(pred, target)
    # iou = 0, rho^2 = 9, enclosing diagonal^2 = 29
    assert loss.item() == pytest.approx(1.0 + 9.0 / 29.0, rel=1e-5)


def test_aspect_mismatch_raises_loss_above_plain_iou():
    pred = torch.tensor([[0.0, 0.0, 4.0, 4.0]], dtype=torch.float32, requires_grad=True)
    square = torch.tensor([[0.0, 0.0, 4.0, 4.0]], dtype=torch.float32)
    wide = torch.tensor([[0.0, 0.0, 8.0, 2.0]], dtype=torch.float32)
    same = ciou_loss_per_box(pred, square)
    mismatch = ciou_loss_per_box(pred, wide)
    assert same.item() < 1e-6
    assert mismatch.item() > 0.5
    mismatch.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert float(pred.grad[0, 2].abs() + pred.grad[0, 3].abs()) > 0


def test_ignores_angle_column():
    pred = torch.tensor([[1.0, 2.0, 4.0, 6.0, 0.3]], dtype=torch.float32)
    target = torch.tensor([[1.0, 2.0, 4.0, 6.0, 1.2]], dtype=torch.float32)
    assert ciou_loss_per_box(pred, target).item() == pytest.approx(0.0, abs=1e-6)


def _peaked_bins(reg_max: int, left: int, top: int, right: int, bottom: int) -> torch.Tensor:
    bins = torch.zeros(1, 4 * reg_max, 1, 1)
    for side, bin_id in enumerate((left, top, right, bottom)):
        bins[0, side * reg_max + bin_id, 0, 0] = 40.0
    return bins


def _horizontal_loss_kwargs(bins: torch.Tensor, gt_boxes: torch.Tensor) -> dict:
    from oriented_det.models.bbox_coder import DistanceAnglePointCoder

    reg_max = 16
    cls = torch.zeros(1, 1, 1, 1)
    cls[0, 0, 0, 0] = 10.0
    return dict(
        cls_scores=[cls],
        bbox_preds=[bins],
        angle_preds=[torch.zeros(1, 1, 1, 1)],
        points=[torch.tensor([[32.0, 32.0]])],
        strides=[8.0],
        gt_boxes=[gt_boxes],
        gt_labels=[torch.tensor([1], dtype=torch.int64)],
        num_classes=1,
        reg_max=reg_max,
        bbox_coder=DistanceAnglePointCoder(angle_version="le90"),
        box_reg_weight=7.5,
        dfl_weight=1.5,
        angle_weight=0.2,
        assign_topk=13,
        tal_alpha=0.5,
        tal_beta=6.0,
        use_ciou=True,
    )


def test_horizontal_ciou_rejects_le90_transpose():
    """A tall le90 box must train the image-axis rectangle, not the swapped (w, h)."""
    import oriented_det.models.yolo_obb as yolo_mod

    # Visual 16×64 at (32, 32). le90 stores the long side as width with angle −π/2.
    gt = torch.tensor([[32.0, 32.0, 64.0, 16.0, -torch.pi / 2]])
    # Stride 8. Image-axis distances: left/right = 8 px → bin 1, top/bottom = 32 px → bin 4.
    correct = yolo_mod.compute_yolo_obb_loss(
        **_horizontal_loss_kwargs(_peaked_bins(16, 1, 4, 1, 4), gt)
    )
    transposed = yolo_mod.compute_yolo_obb_loss(
        **_horizontal_loss_kwargs(_peaked_bins(16, 4, 1, 4, 1), gt)
    )
    correct_box = correct["loss_box_reg"].item() + correct["loss_dfl"].item()
    transposed_box = transposed["loss_box_reg"].item() + transposed["loss_dfl"].item()
    assert correct["loss_box_reg"].item() == pytest.approx(0.0, abs=1e-3)
    assert transposed["loss_box_reg"].item() > 1.0
    assert correct_box < transposed_box


def test_horizontal_loss_scales_with_batch_size():
    import oriented_det.models.yolo_obb as yolo_mod

    gt = torch.tensor([[32.0, 32.0, 64.0, 16.0, -torch.pi / 2]])
    # Near-perfect overlap so the soft-target sum exceeds 1 once two images are batched.
    # Below that, the sum is clamped to 1 and the batch factor is not a pure 2×.
    bins = _peaked_bins(16, 1, 4, 1, 4)
    one = yolo_mod.compute_yolo_obb_loss(**_horizontal_loss_kwargs(bins, gt))
    kwargs = _horizontal_loss_kwargs(bins, gt)
    kwargs["cls_scores"] = [kwargs["cls_scores"][0].repeat(2, 1, 1, 1)]
    kwargs["bbox_preds"] = [kwargs["bbox_preds"][0].repeat(2, 1, 1, 1)]
    kwargs["angle_preds"] = [kwargs["angle_preds"][0].repeat(2, 1, 1, 1)]
    kwargs["gt_boxes"] = [gt, gt.clone()]
    kwargs["gt_labels"] = [kwargs["gt_labels"][0], kwargs["gt_labels"][0].clone()]
    two = yolo_mod.compute_yolo_obb_loss(**kwargs)
    for key in ("loss_classifier", "loss_box_reg", "loss_dfl", "loss_angle"):
        assert two[key].item() == pytest.approx(2.0 * one[key].item(), rel=1e-5)


def test_force_horizontal_selects_ciou(monkeypatch):
    import oriented_det.models.yolo_obb as yolo_mod
    from oriented_det.models.bbox_coder import DistanceAnglePointCoder

    calls = {"ciou": 0, "probiou": 0}
    real_ciou = yolo_mod.ciou_loss_per_box
    real_probiou = yolo_mod.probiou_loss_per_box

    def _ciou(*args, **kwargs):
        calls["ciou"] += 1
        return real_ciou(*args, **kwargs)

    def _probiou(*args, **kwargs):
        calls["probiou"] += 1
        return real_probiou(*args, **kwargs)

    monkeypatch.setattr(yolo_mod, "ciou_loss_per_box", _ciou)
    monkeypatch.setattr(yolo_mod, "probiou_loss_per_box", _probiou)

    reg_max = 16
    bins = torch.zeros(1, 4 * reg_max, 1, 1)
    for side in range(4):
        bins[0, side * reg_max + 1, 0, 0] = 20.0
    cls = torch.zeros(1, 1, 1, 1)
    cls[0, 0, 0, 0] = 10.0
    angle = torch.zeros(1, 1, 1, 1)
    points = torch.tensor([[16.0, 16.0]])
    gt_boxes = torch.tensor([[16.0, 16.0, 16.0, 16.0, 0.0]])
    gt_labels = torch.tensor([1], dtype=torch.int64)
    coder = DistanceAnglePointCoder(angle_version="le90")
    kwargs = dict(
        cls_scores=[cls],
        bbox_preds=[bins],
        angle_preds=[angle],
        points=[points],
        strides=[8.0],
        gt_boxes=[gt_boxes],
        gt_labels=[gt_labels],
        num_classes=1,
        reg_max=reg_max,
        bbox_coder=coder,
        box_reg_weight=7.5,
        dfl_weight=1.5,
        angle_weight=0.2,
        assign_topk=13,
        tal_alpha=0.5,
        tal_beta=6.0,
    )

    held = yolo_mod.compute_yolo_obb_loss(**kwargs, use_ciou=True)
    assert calls == {"ciou": 1, "probiou": 0}
    assert torch.isfinite(held["loss_box_reg"])
    assert held["loss_box_reg"].item() == pytest.approx(0.0, abs=1e-3)

    open_angle = yolo_mod.compute_yolo_obb_loss(**kwargs, use_ciou=False)
    assert calls == {"ciou": 1, "probiou": 1}
    assert torch.isfinite(open_angle["loss_box_reg"])
    assert open_angle["loss_box_reg"].item() > held["loss_box_reg"].item()

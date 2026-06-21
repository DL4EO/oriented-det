"""Tests for GPU-accelerated operations (optional, requires GPU)."""

import pytest

pytest.importorskip("torch")

import torch

from oriented_det.geometry import RBox
from oriented_det.models.oriented_rpn import generate_oriented_anchors
from oriented_det.ops import iou, nms
from oriented_det.ops.gpu_ops import (
    match_anchors_to_gt_gpu,
    oriented_box_hbb_iou_gpu,
    oriented_box_iou_gpu,
    oriented_nms_gpu,
)


@pytest.fixture
def device():
    """Get device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dummy_boxes():
    """Create dummy boxes for testing."""
    return torch.tensor([
        [64.0, 64.0, 32.0, 16.0, 0.0],
        [128.0, 128.0, 32.0, 16.0, 0.0],
        [192.0, 192.0, 32.0, 16.0, 0.0],
    ])


class TestGPUAnchorGeneration:
    """Tests for GPU-accelerated anchor generation."""
    
    def test_gpu_anchor_generation(self, device):
        """Test GPU anchor generation."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        anchors = generate_oriented_anchors(
            image_size=(128, 128),
            feature_map_sizes=[(32, 32)],
            anchor_scales=[8],
            anchor_ratios=[0.5, 1.0, 2.0],
            anchor_angles=[-torch.pi / 2],
            stride_per_level=[4],
            device=device,
        )
        
        assert len(anchors) == 1
        assert anchors[0].device.type == "cuda"
        assert anchors[0].shape[1] == 5
    
    def test_gpu_cpu_consistency(self):
        """Test that GPU and CPU anchor generation produce consistent results."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Generate on CPU
        anchors_cpu = generate_oriented_anchors(
            image_size=(64, 64),
            feature_map_sizes=[(16, 16)],
            anchor_scales=[8],
            anchor_ratios=[1.0],
            anchor_angles=[0],
            stride_per_level=[4],
            device=torch.device("cpu"),
        )
        
        # Generate on GPU
        anchors_gpu = generate_oriented_anchors(
            image_size=(64, 64),
            feature_map_sizes=[(16, 16)],
            anchor_scales=[8],
            anchor_ratios=[1.0],
            anchor_angles=[0],
            stride_per_level=[4],
            device=torch.device("cuda"),
        )
        
        # Move GPU anchors to CPU for comparison
        anchors_gpu_cpu = [a.cpu() for a in anchors_gpu]
        
        # Should produce same results
        assert len(anchors_cpu) == len(anchors_gpu_cpu)
        for cpu_anchors, gpu_anchors in zip(anchors_cpu, anchors_gpu_cpu):
            assert torch.allclose(cpu_anchors, gpu_anchors, atol=1e-5)


class TestGPUIoU:
    """Tests for GPU-accelerated IoU computation."""
    
    def test_gpu_iou_basic(self, device, dummy_boxes):
        """Test basic GPU IoU computation."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        boxes_a = dummy_boxes.to(device)
        boxes_b = dummy_boxes.to(device)
        
        # Convert to RBox format
        rboxes_a = [RBox(*b.tolist()) for b in boxes_a.cpu()]
        rboxes_b = [RBox(*b.tolist()) for b in boxes_b.cpu()]
        
        # Test batch IoU (should use GPU if available)
        iou_matrix = iou.batch_rbox_iou(rboxes_a, rboxes_b)
        
        assert len(iou_matrix) == len(rboxes_a)
        assert all(len(row) == len(rboxes_b) for row in iou_matrix)
        
        # Diagonal should be 1.0 (identical boxes)
        for i in range(len(iou_matrix)):
            assert abs(iou_matrix[i][i] - 1.0) < 1e-6
    
    def test_gpu_iou_large_batch(self, device):
        """Test GPU IoU with large batch."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create many boxes
        num_boxes = 1000
        boxes = torch.randn(num_boxes, 5).to(device)
        boxes[:, 2:4] = torch.abs(boxes[:, 2:4]) + 1.0  # Ensure positive width/height
        
        rboxes = [RBox(*b.cpu().tolist()) for b in boxes]
        
        # Compute batch IoU (should be fast on GPU)
        iou_matrix = iou.batch_rbox_iou(rboxes, rboxes)
        
        assert len(iou_matrix) == num_boxes
        assert all(len(row) == num_boxes for row in iou_matrix)


class TestGPUNMS:
    """Tests for GPU-accelerated NMS."""
    
    def test_gpu_nms_basic(self, device, dummy_boxes):
        """Test basic GPU NMS."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        boxes = dummy_boxes.to(device)
        scores = torch.tensor([0.9, 0.8, 0.7]).to(device)
        
        rboxes = [RBox(*b.cpu().tolist()) for b in boxes]
        
        # Test NMS (should use GPU if available)
        keep = nms.oriented_nms(rboxes, scores.cpu().tolist(), iou_threshold=0.3)
        
        assert isinstance(keep, list)
        assert len(keep) <= len(rboxes)
    
    def test_gpu_nms_large_batch(self, device):
        """Test GPU NMS with large batch."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create many overlapping boxes
        num_boxes = 500
        boxes = torch.randn(num_boxes, 5).to(device)
        boxes[:, 0:2] = boxes[:, 0:2] * 10 + 64  # Centers around (64, 64)
        boxes[:, 2:4] = torch.abs(boxes[:, 2:4]) + 10.0  # Positive size
        boxes[:, 4] = boxes[:, 4] * 0.1  # Small angles
        
        scores = torch.rand(num_boxes).to(device)
        
        rboxes = [RBox(*b.cpu().tolist()) for b in boxes]
        
        # NMS should handle large batches efficiently
        keep = nms.oriented_nms(rboxes, scores.cpu().tolist(), iou_threshold=0.3)
        
        assert isinstance(keep, list)
        assert len(keep) <= num_boxes


class TestGPUFallback:
    """Tests for CPU fallback when GPU unavailable."""
    
    def test_cpu_fallback_anchor_generation(self):
        """Test that anchor generation falls back to CPU when GPU unavailable."""
        # Force CPU device
        anchors = generate_oriented_anchors(
            image_size=(64, 64),
            feature_map_sizes=[(16, 16)],
            anchor_scales=[8],
            anchor_ratios=[1.0],
            anchor_angles=[0],
            stride_per_level=[4],
            device=torch.device("cpu"),
        )
        
        assert len(anchors) == 1
        assert anchors[0].device.type == "cpu"
    
    def test_cpu_fallback_iou(self):
        """Test that IoU computation works on CPU."""
        boxes_a = [
            RBox(64, 64, 32, 16, 0),
            RBox(128, 128, 32, 16, 0),
        ]
        boxes_b = [
            RBox(64, 64, 32, 16, 0),
            RBox(200, 200, 32, 16, 0),
        ]
        
        # Should work on CPU
        iou_matrix = iou.batch_rbox_iou(boxes_a, boxes_b)
        
        assert len(iou_matrix) == len(boxes_a)
        assert all(len(row) == len(boxes_b) for row in iou_matrix)
    
    def test_cpu_fallback_nms(self):
        """Test that NMS works on CPU."""
        boxes = [
            RBox(64, 64, 32, 16, 0),
            RBox(65, 65, 32, 16, 0),  # Overlapping
            RBox(200, 200, 32, 16, 0),  # Far away
        ]
        scores = [0.9, 0.8, 0.7]
        
        # Should work on CPU
        keep = nms.oriented_nms(boxes, scores, iou_threshold=0.3)
        
        assert isinstance(keep, list)
        assert len(keep) <= len(boxes)


class TestGPUParity:
    """Deterministic parity checks against CPU references."""

    def test_oriented_box_iou_gpu_parity_with_python_backend(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        n, m = 10, 12
        boxes1 = torch.empty(n, 5)
        boxes2 = torch.empty(m, 5)

        boxes1[:, 0:2] = torch.rand(n, 2) * 128.0
        boxes1[:, 2:4] = torch.rand(n, 2) * 40.0 + 4.0
        boxes1[:, 4] = (torch.rand(n) - 0.5) * torch.pi
        boxes2[:, 0:2] = torch.rand(m, 2) * 128.0
        boxes2[:, 2:4] = torch.rand(m, 2) * 40.0 + 4.0
        boxes2[:, 4] = (torch.rand(m) - 0.5) * torch.pi

        gpu_iou = oriented_box_iou_gpu(boxes1.cuda(), boxes2.cuda(), num_samples=49).cpu()

        rboxes1 = [RBox(*b.tolist()) for b in boxes1]
        rboxes2 = [RBox(*b.tolist()) for b in boxes2]
        cpu_iou = torch.tensor(
            iou.batch_rbox_iou(rboxes1, rboxes2, intersection_backend="python"),
            dtype=torch.float32,
        )

        # Sampling-based GPU IoU should be close to CPU polygon IoU for small batches.
        assert torch.allclose(gpu_iou, cpu_iou, atol=0.12, rtol=0.2)

    def test_oriented_box_hbb_iou_gpu_parity_with_manual_hbb_iou(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(43)
        n, m = 8, 9
        boxes1 = torch.empty(n, 5)
        boxes2 = torch.empty(m, 5)
        boxes1[:, 0:2] = torch.rand(n, 2) * 100.0
        boxes1[:, 2:4] = torch.rand(n, 2) * 30.0 + 2.0
        boxes1[:, 4] = (torch.rand(n) - 0.5) * torch.pi
        boxes2[:, 0:2] = torch.rand(m, 2) * 100.0
        boxes2[:, 2:4] = torch.rand(m, 2) * 30.0 + 2.0
        boxes2[:, 4] = (torch.rand(m) - 0.5) * torch.pi

        gpu_hbb_iou = oriented_box_hbb_iou_gpu(boxes1.cuda(), boxes2.cuda()).cpu()

        def to_xyxy(obb: torch.Tensor) -> torch.Tensor:
            cx, cy, w, h, a = obb.unbind(dim=1)
            cos_a = torch.cos(a)
            sin_a = torch.sin(a)
            dx = torch.stack([-w / 2, w / 2, w / 2, -w / 2], dim=1)
            dy = torch.stack([-h / 2, -h / 2, h / 2, h / 2], dim=1)
            x = cx.unsqueeze(1) + dx * cos_a.unsqueeze(1) - dy * sin_a.unsqueeze(1)
            y = cy.unsqueeze(1) + dx * sin_a.unsqueeze(1) + dy * cos_a.unsqueeze(1)
            x1 = x.min(dim=1).values
            y1 = y.min(dim=1).values
            x2 = x.max(dim=1).values
            y2 = y.max(dim=1).values
            return torch.stack([x1, y1, x2, y2], dim=1)

        b1 = to_xyxy(boxes1)
        b2 = to_xyxy(boxes2)
        ix1 = torch.maximum(b1[:, None, 0], b2[None, :, 0])
        iy1 = torch.maximum(b1[:, None, 1], b2[None, :, 1])
        ix2 = torch.minimum(b1[:, None, 2], b2[None, :, 2])
        iy2 = torch.minimum(b1[:, None, 3], b2[None, :, 3])
        inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
        area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
        area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
        cpu_hbb_iou = inter / (area1[:, None] + area2[None, :] - inter + 1e-8)

        assert torch.allclose(gpu_hbb_iou, cpu_hbb_iou, atol=1e-5, rtol=1e-5)

    def test_match_anchors_to_gt_gpu_parity_with_reference_logic(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        anchors = torch.tensor(
            [
                [0.0, 0.0, 4.0, 2.0, 0.0],
                [1.0, 0.0, 4.0, 2.0, 0.0],
                [10.0, 0.0, 4.0, 2.0, 0.0],
                [50.0, 50.0, 4.0, 2.0, 0.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        gt = torch.tensor(
            [
                [0.0, 0.0, 4.0, 2.0, 0.0],
                [10.0, 0.0, 4.0, 2.0, 0.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        pos_thr, neg_thr = 0.5, 0.3

        labels_gpu, match_gpu = match_anchors_to_gt_gpu(
            anchors,
            gt,
            positive_iou_threshold=pos_thr,
            negative_iou_threshold=neg_thr,
            use_hbb_for_assignment=False,
        )

        # CPU reference with the same matching policy.
        anchors_cpu = [RBox(*a.tolist()) for a in anchors.cpu()]
        gt_cpu = [RBox(*g.tolist()) for g in gt.cpu()]
        mat = torch.tensor(iou.batch_rbox_iou(anchors_cpu, gt_cpu, intersection_backend="python"))
        n, m = mat.shape
        labels_ref = torch.full((n,), -1, dtype=torch.long)
        match_ref = torch.full((n,), -1, dtype=torch.long)

        max_iou_per_anchor, best_gt_per_anchor = mat.max(dim=1)
        positive_mask = max_iou_per_anchor >= pos_thr
        labels_ref[positive_mask] = 1
        match_ref[positive_mask] = best_gt_per_anchor[positive_mask]

        max_iou_per_gt, best_anchor_per_gt = mat.max(dim=0)
        best_anchor_mask = torch.zeros(n, dtype=torch.bool)
        for gt_idx in range(m):
            if max_iou_per_gt[gt_idx] > 0:
                a_idx = best_anchor_per_gt[gt_idx]
                labels_ref[a_idx] = 1
                match_ref[a_idx] = gt_idx
                best_anchor_mask[a_idx] = True

        negative_mask = (max_iou_per_anchor < neg_thr) & (~best_anchor_mask)
        labels_ref[negative_mask] = 0

        assert torch.equal(labels_gpu.cpu(), labels_ref)
        assert torch.equal(match_gpu.cpu(), match_ref)

    def test_oriented_nms_gpu_parity_on_clear_case(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        boxes = torch.tensor(
            [
                [0.0, 0.0, 4.0, 2.0, 0.0],
                [0.2, 0.1, 4.0, 2.0, 0.0],  # overlaps strongly with first
                [20.0, 20.0, 4.0, 2.0, 0.0],  # disjoint
            ],
            dtype=torch.float32,
            device="cuda",
        )
        scores = torch.tensor([0.95, 0.9, 0.7], dtype=torch.float32, device="cuda")

        keep_gpu = oriented_nms_gpu(boxes, scores, iou_threshold=0.3).cpu().tolist()
        keep_cpu = nms.oriented_nms(
            [RBox(*b.tolist()) for b in boxes.cpu()],
            scores.cpu().tolist(),
            iou_threshold=0.3,
            backend="python",
        )

        assert keep_gpu == keep_cpu

    def test_rotated_nms_force_cpu_matches_polygon_while_env_gpu(self, monkeypatch):
        """final_nms_use_cpu path: ignore ORIENTED_DET_ROTATED_BACKEND for this call."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        monkeypatch.setenv("ORIENTED_DET_ROTATED_BACKEND", "gpu_sample")
        from oriented_det.ops.rotated_ops import rotated_nms

        boxes = torch.tensor(
            [
                [0.0, 0.0, 4.0, 2.0, 0.0],
                [0.2, 0.1, 4.0, 2.0, 0.0],
                [20.0, 20.0, 4.0, 2.0, 0.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        scores = torch.tensor([0.95, 0.9, 0.7], dtype=torch.float32, device="cuda")
        keep_force = rotated_nms(boxes, scores, 0.3, force_cpu=True).cpu().tolist()
        from oriented_det.ops.utils import resolve_exact_polygon_iou_backend

        keep_ref = nms.oriented_nms(
            [RBox(*b.tolist()) for b in boxes.cpu()],
            scores.cpu().tolist(),
            iou_threshold=0.3,
        )
        assert keep_force == keep_ref
        assert resolve_exact_polygon_iou_backend(warn_if_fallback=False) in (
            "shapely",
            "python",
        )


class TestGeometrySampling:
    """Geometry-based oriented IoU sample count (CPU-side resolver)."""

    def test_small_square_uses_few_samples(self):
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])
        from oriented_det.ops.gpu_ops import resolve_oriented_iou_sample_count

        assert resolve_oriented_iou_sample_count(boxes, boxes) == 25

    def test_elongated_box_uses_many_samples(self):
        boxes = torch.tensor([[0.0, 0.0, 5.0, 100.0, 0.0]])
        from oriented_det.ops.gpu_ops import resolve_oriented_iou_sample_count

        assert resolve_oriented_iou_sample_count(boxes, boxes) == 1024

    def test_extreme_elongated_capped_at_1024(self):
        boxes = torch.tensor([[0.0, 0.0, 5.0, 2000.0, 0.0]])
        from oriented_det.ops.gpu_ops import resolve_oriented_iou_sample_count

        assert resolve_oriented_iou_sample_count(boxes, boxes) == 1024

    def test_batch_uses_max_requirement(self):
        small = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])
        ship = torch.tensor([[0.0, 0.0, 5.0, 100.0, 0.0]])
        from oriented_det.ops.gpu_ops import resolve_oriented_iou_sample_count

        assert resolve_oriented_iou_sample_count(small, ship) == 1024

    def test_flat_mode_when_geometry_disabled(self, monkeypatch):
        monkeypatch.setenv("ORIENTED_DET_GPU_ORIENTED_IOU_SAMPLE_BY_MAX_SIDE", "0")
        boxes = torch.tensor([[0.0, 0.0, 5.0, 100.0, 0.0]])
        from oriented_det.ops.gpu_ops import resolve_oriented_iou_sample_count

        assert resolve_oriented_iou_sample_count(boxes, boxes) == 100


class TestGPUSamplingEnv:
    """ORIENTED_DET_GPU_* sampling env validation (GPU path)."""

    def test_nms_iou_samples_env_invalid(self, monkeypatch):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        monkeypatch.setenv("ORIENTED_DET_GPU_NMS_IOU_SAMPLES", "37")
        boxes = torch.ones(2, 5, device="cuda")
        scores = torch.ones(2, device="cuda")
        with pytest.raises(ValueError, match="perfect square"):
            oriented_nms_gpu(boxes, scores, iou_threshold=0.5)

    def test_oriented_iou_env_invalid_when_default_used(self, monkeypatch):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        monkeypatch.setenv("ORIENTED_DET_GPU_ORIENTED_IOU_MIN_SAMPLES", "37")
        boxes1 = torch.tensor([[0.0, 0.0, 4.0, 2.0, 0.0]], device="cuda")
        boxes2 = torch.tensor([[1.0, 0.0, 4.0, 2.0, 0.0]], device="cuda")
        with pytest.raises(ValueError, match="perfect square"):
            oriented_box_iou_gpu(boxes1, boxes2)

    def test_oriented_iou_explicit_samples_skips_env(self, monkeypatch):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        monkeypatch.setenv("ORIENTED_DET_GPU_ORIENTED_IOU_MIN_SAMPLES", "37")
        boxes1 = torch.tensor([[0.0, 0.0, 4.0, 2.0, 0.0]], device="cuda")
        boxes2 = torch.tensor([[1.0, 0.0, 4.0, 2.0, 0.0]], device="cuda")
        out = oriented_box_iou_gpu(boxes1, boxes2, num_samples=25)
        assert out.shape == (1, 1)

    def test_oriented_nms_empty_when_max_candidates_zero(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        boxes = torch.ones(3, 5, device="cuda")
        scores = torch.tensor([0.9, 0.8, 0.7], device="cuda")
        keep = oriented_nms_gpu(boxes, scores, iou_threshold=0.5, max_detections=0)
        assert keep.numel() == 0

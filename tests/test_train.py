"""Tests for training engine and utilities."""

import pytest

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch not available")

pytest.importorskip("torchvision")

from oriented_det.geometry import RBox
from oriented_det.models import OrientedRCNN


def test_metric_tracker():
    """Test MetricTracker functionality."""
    from oriented_det.train import MetricTracker
    
    tracker = MetricTracker()
    
    tracker.update({"loss": 1.0, "accuracy": 0.9})
    tracker.update({"loss": 0.8, "accuracy": 0.95})
    
    assert tracker.get_average("loss") == pytest.approx(0.9)
    assert tracker.get_latest("accuracy") == 0.95
    
    summary = tracker.get_summary()
    assert "loss" in summary
    assert "accuracy" in summary


def test_checkpoint_manager(tmp_path):
    """Test CheckpointManager save/load."""
    from oriented_det.train import CheckpointManager
    
    if torch is None or nn is None:
        pytest.skip("PyTorch not available")
    
    model = nn.Linear(10, 5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    manager = CheckpointManager(tmp_path / "checkpoints")
    
    # Save checkpoint
    path = manager.save(model, optimizer, epoch=0, metrics={"loss": 1.0})
    assert path.exists()
    
    # Load checkpoint
    new_model = nn.Linear(10, 5)
    new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)
    
    checkpoint = manager.load(path, new_model, new_optimizer)
    assert checkpoint["epoch"] == 0
    assert checkpoint["metrics"]["loss"] == 1.0


def test_train_one_epoch_dummy():
    """Test train_one_epoch with dummy data."""
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        pytest.skip("PyTorch not available")
    
    from oriented_det.train import train_one_epoch, MetricTracker
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, images, targets=None):
            if self.training:
                # Return dummy loss dict
                x = torch.randn(1, 10)
                out = self.linear(x)
                return {"loss": (out ** 2).mean()}
            return {"rboxes": []}
    
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Create dummy dataset
    dataset = TensorDataset(torch.randn(10, 10))
    loader = DataLoader(dataset, batch_size=2)
    
    # Mock batch format
    def collate_fn(batch):
        images = [item[0] for item in batch]
        targets = [{"labels": torch.tensor([0])} for _ in batch]
        return images, targets
    
    loader.collate_fn = collate_fn
    
    device = torch.device("cpu")
    tracker = MetricTracker()
    
    metrics = train_one_epoch(
        model=model,
        data_loader=loader,
        optimizer=optimizer,
        device=device,
        metric_tracker=tracker,
    )
    
    assert "total_loss" in metrics
    assert metrics["epoch"] == 0


class TestTrainingLoop:
    """Tests for full training loop."""
    
    def test_train_one_epoch_with_oriented_rcnn(self):
        """Test train_one_epoch with OrientedRCNN."""
        from oriented_det.train import train_one_epoch, MetricTracker
        
        model = OrientedRCNN(
            num_classes=2,
            backbone_name="resnet18",
            pretrained_backbone=False,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Create dummy dataset
        images = [torch.rand(3, 128, 128) for _ in range(4)]
        targets = [
            {"rboxes": [RBox(64, 64, 32, 16, 0.0)], "labels": torch.tensor([1])}
            for _ in range(4)
        ]
        
        dataset = list(zip(images, targets))
        loader = DataLoader(dataset, batch_size=2)
        
        def collate_fn(batch):
            imgs = [item[0] for item in batch]
            targs = [item[1] for item in batch]
            return imgs, targs
        
        loader.collate_fn = collate_fn
        
        device = torch.device("cpu")
        tracker = MetricTracker()
        
        model.train()
        metrics = train_one_epoch(
            model=model,
            data_loader=loader,
            optimizer=optimizer,
            device=device,
            metric_tracker=tracker,
        )
        
        assert "total_loss" in metrics
        assert metrics["epoch"] == 0
    
    def test_train_one_epoch_gradient_accumulation(self):
        """Test train_one_epoch with gradient accumulation."""
        from oriented_det.train import train_one_epoch, MetricTracker
        
        model = OrientedRCNN(
            num_classes=2,
            backbone_name="resnet18",
            pretrained_backbone=False,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        images = [torch.rand(3, 128, 128) for _ in range(4)]
        targets = [
            {"rboxes": [RBox(64, 64, 32, 16, 0.0)], "labels": torch.tensor([1])}
            for _ in range(4)
        ]
        
        dataset = list(zip(images, targets))
        loader = DataLoader(dataset, batch_size=2)
        
        def collate_fn(batch):
            imgs = [item[0] for item in batch]
            targs = [item[1] for item in batch]
            return imgs, targs
        
        loader.collate_fn = collate_fn
        
        device = torch.device("cpu")
        tracker = MetricTracker()
        
        model.train()
        metrics = train_one_epoch(
            model=model,
            data_loader=loader,
            optimizer=optimizer,
            device=device,
            metric_tracker=tracker,
            gradient_accumulation_steps=2,
        )
        
        assert "total_loss" in metrics
    
    def test_train_one_epoch_mixed_precision(self):
        """Test train_one_epoch with mixed precision."""
        from oriented_det.train import train_one_epoch, MetricTracker
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision test")
        
        # Skip if GPU memory is insufficient (common in CI environments)
        try:
            model = OrientedRCNN(
                num_classes=2,
                backbone_name="resnet18",
                pretrained_backbone=False,
            ).cuda()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            
            images = [torch.rand(3, 128, 128).cuda() for _ in range(2)]
            targets = [
                {"rboxes": [RBox(64, 64, 32, 16, 0.0)], "labels": torch.tensor([1])}
                for _ in range(2)
            ]
            
            dataset = list(zip(images, targets))
            loader = DataLoader(dataset, batch_size=2)
            
            def collate_fn(batch):
                imgs = [item[0] for item in batch]
                targs = [item[1] for item in batch]
                return imgs, targs
            
            loader.collate_fn = collate_fn
            
            device = torch.device("cuda")
            tracker = MetricTracker()
            
            model.train()
            metrics = train_one_epoch(
                model=model,
                data_loader=loader,
                optimizer=optimizer,
                device=device,
                metric_tracker=tracker,
                use_amp=True,
            )
            
            assert "total_loss" in metrics
        except torch.cuda.OutOfMemoryError:
            pytest.skip("GPU out of memory - skipping mixed precision test")


class TestEvaluationLoop:
    """Tests for evaluation loop."""
    
    def test_evaluate_with_oriented_rcnn(self):
        """Test evaluation with OrientedRCNN."""
        from oriented_det.train import evaluate
        
        model = OrientedRCNN(
            num_classes=2,
            backbone_name="resnet18",
            pretrained_backbone=False,
        )
        
        images = [torch.rand(3, 128, 128) for _ in range(2)]
        targets = [
            {"rboxes": [RBox(64, 64, 32, 16, 0.0)], "labels": torch.tensor([1])}
            for _ in range(2)
        ]
        
        dataset = list(zip(images, targets))
        loader = DataLoader(dataset, batch_size=2)
        
        def collate_fn(batch):
            imgs = [item[0] for item in batch]
            targs = [item[1] for item in batch]
            return imgs, targs
        
        loader.collate_fn = collate_fn
        
        device = torch.device("cpu")
        
        model.eval()
        metrics = evaluate(
            model=model,
            data_loader=loader,
            device=device,
        )
        
        assert isinstance(metrics, dict)
    
    def test_evaluate_empty_dataset(self):
        """Test evaluation with empty dataset."""
        from oriented_det.train import evaluate
        
        model = OrientedRCNN(
            num_classes=2,
            backbone_name="resnet18",
            pretrained_backbone=False,
        )
        
        empty_dataset = []
        loader = DataLoader(empty_dataset, batch_size=2)
        
        device = torch.device("cpu")
        
        model.eval()
        metrics = evaluate(
            model=model,
            data_loader=loader,
            device=device,
        )
        
        # Should handle empty dataset gracefully
        assert isinstance(metrics, dict)


class TestEdgeCases:
    """Tests for edge cases in training."""
    
    def test_empty_dataset(self):
        """Test training with empty dataset."""
        from oriented_det.train import train_one_epoch, MetricTracker
        
        model = OrientedRCNN(
            num_classes=2,
            backbone_name="resnet18",
            pretrained_backbone=False,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        empty_dataset = []
        loader = DataLoader(empty_dataset, batch_size=2)
        
        device = torch.device("cpu")
        tracker = MetricTracker()
        
        model.train()
        # Should handle empty dataset gracefully
        metrics = train_one_epoch(
            model=model,
            data_loader=loader,
            optimizer=optimizer,
            device=device,
            metric_tracker=tracker,
        )
        
        assert isinstance(metrics, dict)
    
    def test_single_batch(self):
        """Test training with single batch."""
        from oriented_det.train import train_one_epoch, MetricTracker
        
        model = OrientedRCNN(
            num_classes=2,
            backbone_name="resnet18",
            pretrained_backbone=False,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        images = [torch.rand(3, 128, 128)]
        targets = [{"rboxes": [RBox(64, 64, 32, 16, 0.0)], "labels": torch.tensor([1])}]
        
        dataset = list(zip(images, targets))
        loader = DataLoader(dataset, batch_size=1)
        
        def collate_fn(batch):
            imgs = [item[0] for item in batch]
            targs = [item[1] for item in batch]
            return imgs, targs
        
        loader.collate_fn = collate_fn
        
        device = torch.device("cpu")
        tracker = MetricTracker()
        
        model.train()
        metrics = train_one_epoch(
            model=model,
            data_loader=loader,
            optimizer=optimizer,
            device=device,
            metric_tracker=tracker,
        )
        
        assert "total_loss" in metrics


def test_map_delta_ignores_skipped_previous_epoch():
    """mAP deltas must compare to the last computed mAP, not sentinel -1."""
    from oriented_det.train.engine import (
        _format_validation_metrics,
        _snapshot_val_metrics_for_comparison,
    )

    prior = None
    for _ in range(3):
        prior = _snapshot_val_metrics_for_comparison({"mAP": -1.0, "accuracy": 0.4}, prior)

    current = {"mAP": 0.6344, "accuracy": 0.5}
    out_first = _format_validation_metrics(current, prior)
    assert "mAP: 0.6344" in out_first
    assert "↑1.6344" not in out_first
    assert "  ↑" not in out_first.split("mAP:")[1].split("\n")[0]

    prior = _snapshot_val_metrics_for_comparison(current, prior)
    prior = _snapshot_val_metrics_for_comparison({"mAP": -1.0, "accuracy": 0.55}, prior)
    assert prior["mAP"] == pytest.approx(0.6344)

    out_second = _format_validation_metrics({"mAP": 0.70, "accuracy": 0.56}, prior)
    assert "mAP: 0.7000" in out_second
    assert "↑0.0656" in out_second


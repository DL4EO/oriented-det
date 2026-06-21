"""Example: Training oriented detection models on DOTA dataset.

This script demonstrates how to:
1. Load DOTA dataset
2. Set up model and optimizer
3. Train with the training engine
4. Evaluate on validation set
5. Save checkpoints
"""

import os
# Set CUDA memory allocation configuration before importing PyTorch
# This helps with CUDA out of memory errors by using expandable segments
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
from pathlib import Path
import sys

try:
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader
except ImportError as e:
    print(f"Required dependencies not installed: {e}")
    print("Please install: pip install torch")
    sys.exit(1)

from oriented_det import OrientedRCNN, RotatedRetinaNet
from oriented_det.data import build_dota_loader, DOTADataset
from oriented_det.train import train, CheckpointManager, MetricTracker
from oriented_det.train.utils import collate_dota_samples

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None  # type: ignore


def create_model(model_type: str, num_classes: int, backbone: str = "resnet50", pretrained: bool = False):
    """Create a detection model.
    
    Args:
        model_type: "oriented_rcnn" or "rotated_retinanet"
        num_classes: Number of detection classes
        backbone: Backbone name
        pretrained: Whether to use pretrained backbone
    """
    if model_type == "oriented_rcnn":
        return OrientedRCNN(
            num_classes=num_classes,
            backbone_name=backbone,
            pretrained_backbone=pretrained,
        )
    elif model_type == "rotated_retinanet":
        return RotatedRetinaNet(
            num_classes=num_classes,
            backbone_name=backbone,
            pretrained_backbone=pretrained,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_data_loaders(
    data_root: Path,
    batch_size: int = 4,
    num_workers: int = 4,
    allowed_classes: list = None,
    difficult_strategy: str = "drop",
):
    """Create training and validation data loaders.
    
    Args:
        data_root: Root directory of DOTA dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        allowed_classes: Optional list of allowed class names
        difficult_strategy: drop | ignore | keep (DOTA difficult flag handling)
    """
    # Note: This is a simplified example. In practice, you'd need to:
    # 1. Load images properly (DOTADataset returns paths, not tensors)
    # 2. Apply transforms/augmentation
    # 3. Create a proper collate function that loads images
    
    train_loader = build_dota_loader(
        root_dir=data_root,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        allowed_classes=allowed_classes,
        difficult_strategy=difficult_strategy,
    )
    
    val_loader = build_dota_loader(
        root_dir=data_root,
        split="val",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        allowed_classes=allowed_classes,
        difficult_strategy=difficult_strategy,
    )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train oriented detection model on DOTA")
    parser.add_argument("data_root", type=Path, help="Root directory of DOTA dataset")
    parser.add_argument("--model-type", choices=["oriented_rcnn", "rotated_retinanet"], default="oriented_rcnn",
                       help="Model type to train")
    parser.add_argument("--backbone", default="resnet50", help="Backbone architecture")
    parser.add_argument("--num-classes", type=int, default=15, help="Number of classes")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", default=None,
                       help="Device to train on")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"),
                       help="Directory to save checkpoints")
    parser.add_argument("--resume", type=Path, help="Path to checkpoint to resume from")
    parser.add_argument("--use-amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--gradient-accumulation", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--log-dir", type=Path, help="Directory for TensorBoard logs (enables TensorBoard if provided)")
    
    args = parser.parse_args()

    if args.device is None:
        from oriented_det.utils import get_device
        args.device = str(get_device())

    # Create model
    print(f"Creating {args.model_type} model with {args.num_classes} classes...")
    model = create_model(args.model_type, args.num_classes, args.backbone, pretrained=True)
    model.to(args.device)
    
    # Create data loaders
    print(f"Loading DOTA dataset from {args.data_root}...")
    try:
        train_loader, val_loader = create_data_loaders(
            args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nNote: This example requires a properly formatted DOTA dataset.")
        print("The dataset should have the following structure:")
        print("  data_root/")
        print("    labelTxt/  (annotation files)")
        print("    images/    (image files)")
        sys.exit(1)
    
    # Create optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0001,
    )
    
    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=8,
        gamma=0.1,
    )
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager(
        args.checkpoint_dir,
        best_metric="total_loss",  # In practice, use mAP
        higher_is_better=False,
    )
    
    # Setup TensorBoard writer if log_dir is provided
    writer = None
    if args.log_dir:
        if not TENSORBOARD_AVAILABLE:
            print("Warning: TensorBoard is not available. Install with: pip install tensorboard")
        else:
            args.log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(args.log_dir))
            print(f"TensorBoard logging enabled. Logs saved to: {args.log_dir}")
            print(f"View logs with: tensorboard --logdir {args.log_dir}")
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume and args.resume.exists():
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint_manager.load(args.resume, model, optimizer)
        start_epoch = checkpoint_manager.load(args.resume, model, optimizer).get("epoch", 0) + 1
    
    # Training configuration
    print("\nTraining configuration:")
    print(f"  Model: {args.model_type}")
    print(f"  Backbone: {args.backbone}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Device: {args.device}")
    print(f"  Mixed precision: {args.use_amp}")
    print(f"  Gradient accumulation: {args.gradient_accumulation}")
    print()
    
    # Train
    try:
        history = train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=torch.device(args.device),
            num_epochs=args.num_epochs,
            val_loader=val_loader,
            lr_scheduler=lr_scheduler,
            checkpoint_manager=checkpoint_manager,
            use_amp=args.use_amp,
            gradient_accumulation_steps=args.gradient_accumulation,
            max_grad_norm=1.0,
            start_epoch=start_epoch,
            writer=writer,
        )
        
        print("\nTraining completed!")
        print(f"Checkpoints saved to: {args.checkpoint_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print(f"Checkpoints saved to: {args.checkpoint_dir}")
    except Exception as e:
        print(f"\nTraining error: {e}")
        raise


if __name__ == "__main__":
    main()

"""Helper functions for training notebooks."""

from pathlib import Path
from typing import Any, Optional, Callable
import random
from PIL import Image
import torch
from torchvision import transforms as T
from torch.nn import functional as F
from oriented_det.data import create_albumentations_augmentation
from oriented_det.data.flips import apply_random_train_flips
from oriented_det.data.preprocessing import apply_spatial_preprocess
from oriented_det.geometry.rbox import normalize_le90

# ImageNet normalization constants (for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]   # RGB

# DOTA dataset normalization constants
# Computed from DOTA-v2.0 training tiles (02-EDA-DOTA-1024.ipynb)
# These are on [0, 1] scale (after ToTensor conversion)
DOTA_MEAN = [0.245, 0.248, 0.235]  # RGB
DOTA_STD = [0.22, 0.217, 0.208]   # RGB

# MMDetection/MMRotate normalization constants on [0, 1] scale
MMDET_MEAN = [123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0]  # RGB
MMDET_STD = [58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0]  # RGB


class _CollateFn:
    """Picklable collate callable for DataLoader (supports num_workers > 0 on macOS)."""

    def __init__(
        self,
        class_map: dict,
        augmentation: Optional[Any],
        normalize: bool,
        resize_mode: str,
        resize_to: tuple[int, int],
        pad_size_divisor: int,
        enable_train_flip: bool,
        enable_flip_horizontal: Optional[bool],
        enable_flip_vertical: Optional[bool],
        enable_flip_diagonal: Optional[bool],
        normalize_mean: Optional[list[float]],
        normalize_std: Optional[list[float]],
        difficult_strategy: str = "drop",
        random_crop: bool = True,
    ):
        self.class_map = class_map
        self.augmentation = augmentation
        self.normalize = normalize
        self.resize_mode = (resize_mode or "fixed").strip().lower()
        self.resize_to = resize_to
        self.pad_size_divisor = pad_size_divisor
        self.enable_train_flip = enable_train_flip
        self.enable_flip_horizontal = enable_flip_horizontal
        self.enable_flip_vertical = enable_flip_vertical
        self.enable_flip_diagonal = enable_flip_diagonal
        mean = normalize_mean if normalize_mean is not None else MMDET_MEAN
        std = normalize_std if normalize_std is not None else MMDET_STD
        self.normalize_transform = T.Normalize(mean=mean, std=std) if normalize else None
        self._to_tensor = T.ToTensor()
        ds = (difficult_strategy or "drop").strip().lower()
        if ds not in {"drop", "ignore", "keep"}:
            raise ValueError(f"Invalid difficult_strategy={difficult_strategy!r}; expected 'drop', 'ignore', or 'keep'.")
        self.difficult_strategy = ds
        self.random_crop = random_crop

    def __call__(self, batch):
        """Collate batch: converts DOTASample to model input (images, targets)."""
        images = []
        targets = []
        for sample in batch:
            image_path = sample.image_path
            try:
                pil_image = Image.open(image_path).convert("RGB")
                if self.augmentation is not None:
                    pil_image = self.augmentation(pil_image)
                target_h, target_w = self.resize_to
                raw_rboxes = [ann.rbox for ann in sample.annotations]
                spatial = apply_spatial_preprocess(
                    pil_image,
                    raw_rboxes,
                    self.resize_mode,
                    self.resize_to,
                    random_crop=self.random_crop and self.resize_mode == "crop",
                )
                pil_image = spatial.image
                scaled_rbox_list = spatial.rboxes

                rboxes = []
                labels = []
                rboxes_ignore = []
                labels_ignore = []
                for ann, scaled_rbox in zip(sample.annotations, scaled_rbox_list):
                    class_id = self.class_map.get(ann.class_name, 0)
                    if class_id <= 0:
                        continue
                    nrb = normalize_le90(scaled_rbox)
                    is_diff = int(getattr(ann, "difficult", 0)) == 1
                    if self.difficult_strategy == "ignore" and is_diff:
                        rboxes_ignore.append(nrb)
                        labels_ignore.append(class_id)
                    else:
                        # "drop": difficult annotations should already be removed by the dataset reader.
                        # "keep": treat difficult as normal GT.
                        rboxes.append(nrb)
                        labels.append(class_id)

                flip_h = self.enable_flip_horizontal if self.enable_flip_horizontal is not None else self.enable_train_flip
                flip_v = self.enable_flip_vertical if self.enable_flip_vertical is not None else self.enable_train_flip
                flip_d = self.enable_flip_diagonal if self.enable_flip_diagonal is not None else False
                if flip_h or flip_v or flip_d:
                    pil_image, rboxes = apply_random_train_flips(
                        pil_image,
                        rboxes,
                        image_width=float(target_w),
                        image_height=float(target_h),
                        enable_horizontal=bool(flip_h),
                        enable_vertical=bool(flip_v),
                        enable_diagonal=bool(flip_d),
                    )

                image_tensor = self._to_tensor(pil_image)
                content_size = spatial.meta.content_size
                if self.pad_size_divisor > 1:
                    _, img_h, img_w = image_tensor.shape
                    pad_h = (self.pad_size_divisor - (img_h % self.pad_size_divisor)) % self.pad_size_divisor
                    pad_w = (self.pad_size_divisor - (img_w % self.pad_size_divisor)) % self.pad_size_divisor
                    if pad_h or pad_w:
                        image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), value=0.0)
                        content_size = (target_h, target_w)
                if self.normalize_transform is not None:
                    image_tensor = self.normalize_transform(image_tensor)
                images.append(image_tensor)
            except Exception as e:
                raise RuntimeError(f"Failed to load image from {image_path}: {e}") from e

            if rboxes:
                rboxes_tensor = torch.tensor(
                    [[rb.cx, rb.cy, rb.width, rb.height, rb.angle] for rb in rboxes],
                    dtype=torch.float32,
                )
            else:
                rboxes_tensor = torch.zeros((0, 5), dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64) if labels else torch.tensor([], dtype=torch.int64)
            target = {
                "rboxes": rboxes_tensor,
                "labels": labels_tensor,
                "image_id": sample.image_path.stem,
                "image_filename": Path(sample.image_path).name,
            }
            if self.difficult_strategy == "ignore":
                if rboxes_ignore:
                    target["rboxes_ignore"] = torch.tensor(
                        [[rb.cx, rb.cy, rb.width, rb.height, rb.angle] for rb in rboxes_ignore],
                        dtype=torch.float32,
                    )
                    target["labels_ignore"] = torch.tensor(labels_ignore, dtype=torch.int64)
                else:
                    target["rboxes_ignore"] = torch.zeros((0, 5), dtype=torch.float32)
                    target["labels_ignore"] = torch.tensor([], dtype=torch.int64)
            if content_size is not None:
                target["content_size"] = content_size
            targets.append(target)
        return images, targets


def create_collate_fn(
    class_map: dict,
    augmentation: Optional[Any] = None,
    normalize: bool = True,
    resize_mode: str = "fixed",
    resize_to: tuple[int, int] = (1024, 1024),
    pad_size_divisor: int = 32,
    enable_train_flip: bool = False,
    enable_flip_horizontal: Optional[bool] = None,
    enable_flip_vertical: Optional[bool] = None,
    enable_flip_diagonal: Optional[bool] = None,
    normalize_mean: Optional[list[float]] = None,
    normalize_std: Optional[list[float]] = None,
    difficult_strategy: str = "drop",
    random_crop: bool = True,
) -> Callable:
    """Create a collate function with consistent class mapping and optional augmentation.
    
    This function preserves oriented boxes (RBoxes) with their angles for the complete
    OrientedRCNN model.
    
    Args:
        class_map: Dictionary mapping class names to class IDs
        augmentation: Optional albumentations transform to apply to training images.
            Should be None for validation (no augmentation).
        normalize: If True, apply MMDetection/MMRotate normalization.
        resize_mode: ``fixed`` (stretch), ``pad`` (uniform scale by large edge + pad), or
            ``crop`` (native-res crop/pad to canvas).
        resize_to: Target canvas (height, width). MMRotate DOTA baseline uses 1024x1024.
        pad_size_divisor: Pad image tensors to a multiple of this divisor.
        enable_train_flip: If True, apply random train flips (deprecated: use enable_flip_horizontal/vertical).
        enable_flip_horizontal: If True, allow random horizontal flip (training). None = use enable_train_flip.
        enable_flip_vertical: If True, allow random vertical flip (training). None = use enable_train_flip.
        enable_flip_diagonal: If True, allow random diagonal flip (MMRotate ``RRandomFlip``). Default False.
    
    Returns:
        Collate function that can be used with DataLoader (picklable for multiprocessing).
    """
    return _CollateFn(
        class_map=class_map,
        augmentation=augmentation,
        normalize=normalize,
        resize_mode=resize_mode,
        resize_to=resize_to,
        pad_size_divisor=pad_size_divisor,
        enable_train_flip=enable_train_flip,
        enable_flip_horizontal=enable_flip_horizontal,
        enable_flip_vertical=enable_flip_vertical,
        enable_flip_diagonal=enable_flip_diagonal,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        difficult_strategy=difficult_strategy,
        random_crop=random_crop,
    )


def create_train_augmentation(
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    gamma_limit: tuple[int, int] = (80, 120),
    gauss_noise_var_limit: tuple[float, float] = (10.0, 50.0),
    blur_limit: int = 3,
    clahe_clip_limit: float = 4.0,
    p_brightness_contrast: float = 0.5,
    p_gamma: float = 0.3,
    p_noise: float = 0.2,
    p_blur: float = 0.2,
    p_clahe: float = 0.3,
) -> Any:
    """Create albumentations augmentation for training (non-geometric only).
    
    Note: We only use non-geometric augmentations because albumentations doesn't 
    support oriented bounding boxes.
    
    Args:
        brightness_limit: Brightness adjustment limit
        contrast_limit: Contrast adjustment limit
        gamma_limit: Gamma correction range
        gauss_noise_var_limit: Gaussian noise variance range
        blur_limit: Blur kernel size limit
        clahe_clip_limit: CLAHE clip limit
        p_brightness_contrast: Probability of applying brightness/contrast
        p_gamma: Probability of applying gamma correction
        p_noise: Probability of applying noise
        p_blur: Probability of applying blur
        p_clahe: Probability of applying CLAHE
    
    Returns:
        Albumentations transform
    """
    return create_albumentations_augmentation(
        brightness_limit=brightness_limit,
        contrast_limit=contrast_limit,
        gamma_limit=gamma_limit,
        gauss_noise_var_limit=gauss_noise_var_limit,
        blur_limit=blur_limit,
        clahe_clip_limit=clahe_clip_limit,
        p_brightness_contrast=p_brightness_contrast,
        p_gamma=p_gamma,
        p_noise=p_noise,
        p_blur=p_blur,
        p_clahe=p_clahe,
    )


def normalize_image(image_tensor: torch.Tensor) -> torch.Tensor:
    """Apply MMRotate-compatible normalization to an image tensor.
    
    Args:
        image_tensor: Image tensor of shape (C, H, W) or (B, C, H, W) in [0, 1] range.
    
    Returns:
        Normalized image tensor with same shape.
    """
    normalize = T.Normalize(mean=MMDET_MEAN, std=MMDET_STD)
    if image_tensor.dim() == 3:
        return normalize(image_tensor)
    elif image_tensor.dim() == 4:
        return torch.stack([normalize(img) for img in image_tensor])
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {image_tensor.dim()}D")


def denormalize_image(image_tensor: torch.Tensor) -> torch.Tensor:
    """Reverse MMRotate-compatible normalization for visualization.
    
    Args:
        image_tensor: Normalized image tensor of shape (C, H, W) or (B, C, H, W).
    
    Returns:
        Denormalized image tensor in [0, 1] range.
    """
    mean = torch.tensor(MMDET_MEAN).view(3, 1, 1)
    std = torch.tensor(MMDET_STD).view(3, 1, 1)
    
    if image_tensor.dim() == 3:
        device = image_tensor.device
        mean = mean.to(device)
        std = std.to(device)
        return (image_tensor * std + mean).clamp(0, 1)
    elif image_tensor.dim() == 4:
        device = image_tensor.device
        mean = mean.to(device).unsqueeze(0)
        std = std.to(device).unsqueeze(0)
        return (image_tensor * std + mean).clamp(0, 1)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {image_tensor.dim()}D")


def check_directories(train_tiles_dir, val_tiles_dir):
    """Check if dataset directories exist and print status.

    Args:
        train_tiles_dir: Path or list of paths to training tile directories
        val_tiles_dir: Path or list of paths to validation tile directories
    """
    from pathlib import Path

    def _as_roots(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [Path(p) for p in value]
        return [Path(value)]

    train_roots = _as_roots(train_tiles_dir)
    val_roots = _as_roots(val_tiles_dir)

    for label, roots in ("Train", train_roots), ("Val", val_roots):
        if not roots:
            print(f"{label} tiles: (not configured)")
            continue
        print(f"{label} tiles ({len(roots)} root(s)):")
        for root in roots:
            print(f"  {root}")
            print(f"    Images exist: {(root / 'images').exists()}")
            print(f"    Labels exist: {(root / 'labels').exists()}")
        print()


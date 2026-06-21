"""Efficient data augmentation transforms for oriented detection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
import random

from ..geometry import Polygon, RBox, transforms

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    A = None  # type: ignore
    ToTensorV2 = None  # type: ignore


@dataclass
class OrientedTransform:
    """Base class for transforms that modify both image and oriented boxes."""

    def apply_to_image(self, image) -> any:  # type: ignore
        """Apply transform to image. Returns transformed image."""
        raise NotImplementedError
    
    def apply_to_rbox(self, rbox: RBox, image_width: int, image_height: int) -> RBox:
        """Apply transform to RBox. Returns transformed RBox."""
        raise NotImplementedError


class HorizontalFlip(OrientedTransform):
    """Horizontal flip transform for oriented boxes."""

    def __init__(self, p: float = 0.5):
        if not 0 <= p <= 1:
            raise ValueError("Probability p must be in [0, 1]")
        self.p = p
    
    def apply_to_image(self, image) -> any:  # type: ignore
        if Image is None:
            raise RuntimeError("PIL/Pillow is required for image transforms.")
        from PIL import Image as PILImage
        if isinstance(image, PILImage.Image):
            return image.transpose(PILImage.FLIP_LEFT_RIGHT)
        return image
    
    def apply_to_rbox(self, rbox: RBox, image_width: int, image_height: int) -> RBox:
        """Flip RBox horizontally."""
        return transforms.flip_horizontal(rbox, float(image_width))
    
    def __call__(self, image, rboxes: Sequence[RBox], image_width: int, image_height: int) -> Tuple[any, List[RBox]]:  # type: ignore
        if random.random() < self.p:
            flipped_image = self.apply_to_image(image)
            flipped_boxes = [self.apply_to_rbox(box, image_width, image_height) for box in rboxes]
            return flipped_image, flipped_boxes
        return image, list(rboxes)


class VerticalFlip(OrientedTransform):
    """Vertical flip transform for oriented boxes."""

    def __init__(self, p: float = 0.5):
        if not 0 <= p <= 1:
            raise ValueError("Probability p must be in [0, 1]")
        self.p = p
    
    def apply_to_image(self, image) -> any:  # type: ignore
        if Image is None:
            raise RuntimeError("PIL/Pillow is required for image transforms.")
        from PIL import Image as PILImage
        if isinstance(image, PILImage.Image):
            return image.transpose(PILImage.FLIP_TOP_BOTTOM)
        return image
    
    def apply_to_rbox(self, rbox: RBox, image_width: int, image_height: int) -> RBox:
        """Flip RBox vertically."""
        return transforms.flip_vertical(rbox, float(image_height))
    
    def __call__(self, image, rboxes: Sequence[RBox], image_width: int, image_height: int) -> Tuple[any, List[RBox]]:  # type: ignore
        if random.random() < self.p:
            flipped_image = self.apply_to_image(image)
            flipped_boxes = [self.apply_to_rbox(box, image_width, image_height) for box in rboxes]
            return flipped_image, flipped_boxes
        return image, list(rboxes)


class DiagonalFlip(OrientedTransform):
    """Diagonal flip (MMRotate ``direction='diagonal'``, le90 angle π − θ)."""

    def __init__(self, p: float = 0.5):
        if not 0 <= p <= 1:
            raise ValueError("Probability p must be in [0, 1]")
        self.p = p

    def apply_to_image(self, image) -> any:  # type: ignore
        if Image is None:
            raise RuntimeError("PIL/Pillow is required for image transforms.")
        from PIL import Image as PILImage
        if isinstance(image, PILImage.Image):
            return image.transpose(PILImage.FLIP_LEFT_RIGHT).transpose(PILImage.FLIP_TOP_BOTTOM)
        return image

    def apply_to_rbox(self, rbox: RBox, image_width: int, image_height: int) -> RBox:
        return transforms.flip_diagonal(rbox, float(image_width), float(image_height))

    def __call__(self, image, rboxes: Sequence[RBox], image_width: int, image_height: int) -> Tuple[any, List[RBox]]:  # type: ignore
        if random.random() < self.p:
            flipped_image = self.apply_to_image(image)
            flipped_boxes = [self.apply_to_rbox(box, image_width, image_height) for box in rboxes]
            return flipped_image, flipped_boxes
        return image, list(rboxes)


class Rotate(OrientedTransform):
    """Rotation transform for oriented boxes."""

    def __init__(self, degrees: float, p: float = 1.0):
        if not 0 <= p <= 1:
            raise ValueError("Probability p must be in [0, 1]")
        self.degrees = degrees
        self.radians = math.radians(degrees)
        self.p = p
    
    def apply_to_image(self, image) -> any:  # type: ignore
        if Image is None:
            raise RuntimeError("PIL/Pillow is required for image transforms.")
        from PIL import Image as PILImage
        if isinstance(image, PILImage.Image):
            return image.rotate(self.degrees, expand=False)
        return image
    
    def apply_to_rbox(self, rbox: RBox, image_width: int, image_height: int) -> RBox:
        """Rotate RBox around image center."""
        center_x, center_y = image_width / 2.0, image_height / 2.0
        
        # Translate to origin
        dx = rbox.cx - center_x
        dy = rbox.cy - center_y
        
        # Rotate
        cos_a = math.cos(self.radians)
        sin_a = math.sin(self.radians)
        new_dx = dx * cos_a - dy * sin_a
        new_dy = dx * sin_a + dy * cos_a
        
        # Translate back
        new_cx = center_x + new_dx
        new_cy = center_y + new_dy
        new_angle = rbox.angle + self.radians
        
        return RBox(new_cx, new_cy, rbox.width, rbox.height, new_angle)
    
    def __call__(self, image, rboxes: Sequence[RBox], image_width: int, image_height: int) -> Tuple[any, List[RBox]]:  # type: ignore
        if random.random() < self.p:
            rotated_image = self.apply_to_image(image)
            rotated_boxes = [self.apply_to_rbox(box, image_width, image_height) for box in rboxes]
            return rotated_image, rotated_boxes
        return image, list(rboxes)


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: Sequence[OrientedTransform]):
        self.transforms = transforms
    
    def __call__(self, image, rboxes: Sequence[RBox], image_width: int, image_height: int) -> Tuple[any, List[RBox]]:  # type: ignore
        current_image = image
        current_boxes = list(rboxes)
        
        for transform in self.transforms:
            current_image, current_boxes = transform(
                current_image, current_boxes, image_width, image_height
            )
        
        return current_image, current_boxes


class AlbumentationsTransform:
    """Wrapper for albumentations transforms that only apply non-geometric augmentations.
    
    This transform applies albumentations augmentations that do not modify the spatial
    layout of the image (no rotation, scaling, translation, etc.) since albumentations
    does not support oriented bounding boxes. Only color, contrast, blur, noise, and
    similar augmentations are applied.
    
    Args:
        transform: An albumentations Compose transform containing only non-geometric
            augmentations. The transform should work on numpy arrays (H, W, C).
        p: Probability of applying the transform (default: 1.0)
    
    Example:
        >>> import albumentations as A
        >>> from oriented_det.data import AlbumentationsTransform
        >>> 
        >>> # Create non-geometric augmentation pipeline
        >>> aug = A.Compose([
        ...     A.RandomBrightnessContrast(p=0.5),
        ...     A.RandomGamma(p=0.3),
        ...     A.GaussNoise(p=0.2),
        ...     A.GaussianBlur(p=0.2),
        ...     A.CLAHE(p=0.3),
        ... ])
        >>> 
        >>> transform = AlbumentationsTransform(aug)
        >>> augmented_image = transform(image)  # PIL Image or numpy array
    """
    
    def __init__(self, transform, p: float = 1.0):
        if A is None:
            raise RuntimeError("albumentations is required. Install with: pip install albumentations")
        if not 0 <= p <= 1:
            raise ValueError("Probability p must be in [0, 1]")
        self.transform = transform
        self.p = p
    
    def __call__(self, image) -> any:  # type: ignore
        """Apply albumentations transform to image.
        
        Args:
            image: PIL Image or numpy array (H, W, C) in RGB format
        
        Returns:
            Transformed image in the same format as input
        """
        if random.random() >= self.p:
            return image
        
        # Convert PIL Image to numpy array if needed
        is_pil = False
        if Image is not None and isinstance(image, Image.Image):
            is_pil = True
            import numpy as np
            # Convert to RGB if RGBA or other format
            if image.mode != "RGB":
                image = image.convert("RGB")
            image_array = np.array(image)
        else:
            import numpy as np
            image_array = np.asarray(image)
            # Handle RGBA numpy arrays (4 channels -> 3 channels)
            if image_array.ndim == 3 and image_array.shape[2] == 4:
                # Drop alpha channel
                image_array = image_array[:, :, :3]
        
        # Ensure image is in (H, W, C) format with uint8 dtype
        if image_array.dtype != np.uint8:
            # Normalize to [0, 255] if in [0, 1] range
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)
        
        # Apply albumentations transform
        # Note: albumentations expects numpy array in (H, W, C) format
        transformed = self.transform(image=image_array)["image"]
        
        # Convert back to PIL Image if input was PIL
        if is_pil:
            return Image.fromarray(transformed, "RGB")
        
        return transformed


def create_albumentations_augmentation(
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    gamma_limit: tuple = (80, 120),
    gauss_noise_var_limit: tuple = (10.0, 50.0),
    blur_limit: int = 3,
    clahe_clip_limit: float = 4.0,
    p_brightness_contrast: float = 0.5,
    p_gamma: float = 0.3,
    p_noise: float = 0.2,
    p_blur: float = 0.2,
    p_clahe: float = 0.3,
) -> AlbumentationsTransform:
    """Create a default albumentations augmentation pipeline with non-geometric transforms.
    
    This function creates a sensible default augmentation pipeline using only
    non-geometric transforms that are safe for oriented bounding boxes.
    
    Args:
        brightness_limit: Range for brightness adjustment (-limit to +limit)
        contrast_limit: Range for contrast adjustment (-limit to +limit)
        gamma_limit: Range for gamma correction (as percentage, e.g., (80, 120))
        gauss_noise_var_limit: Variance range for Gaussian noise (will be converted to std_range internally)
        blur_limit: Maximum kernel size for Gaussian blur (must be odd)
        clahe_clip_limit: Clip limit for CLAHE (Contrast Limited Adaptive Histogram Equalization)
        p_brightness_contrast: Probability of applying brightness/contrast
        p_gamma: Probability of applying gamma correction
        p_noise: Probability of adding Gaussian noise
        p_blur: Probability of applying Gaussian blur
        p_clahe: Probability of applying CLAHE
    
    Returns:
        AlbumentationsTransform instance with the configured augmentation pipeline
    
    Example:
        >>> from oriented_det.data import create_albumentations_augmentation
        >>> 
        >>> # Create default augmentation
        >>> aug = create_albumentations_augmentation()
        >>> 
        >>> # Apply to image
        >>> augmented_image = aug(image)
    """
    if A is None:
        raise RuntimeError("albumentations is required. Install with: pip install albumentations")
    
    # Ensure blur_limit is odd
    if blur_limit % 2 == 0:
        blur_limit += 1
    
    transforms = []
    
    # Color and contrast augmentations
    if p_brightness_contrast > 0:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=p_brightness_contrast,
            )
        )
    
    if p_gamma > 0:
        transforms.append(
            A.RandomGamma(gamma_limit=gamma_limit, p=p_gamma)
        )
    
    if p_clahe > 0:
        transforms.append(
            A.CLAHE(clip_limit=clahe_clip_limit, p=p_clahe)
        )
    
    # Noise and blur augmentations
    if p_noise > 0:
        # Convert variance to normalized standard deviation: std = sqrt(var) / 255
        # std_range expects values in [0, 1] as fraction of max value (255 for uint8)
        std_range = (
            math.sqrt(gauss_noise_var_limit[0]) / 255.0,
            math.sqrt(gauss_noise_var_limit[1]) / 255.0
        )
        transforms.append(
            A.GaussNoise(std_range=std_range, p=p_noise)
        )
    
    if p_blur > 0:
        transforms.append(
            A.GaussianBlur(blur_limit=(3, blur_limit), p=p_blur)
        )
    
    # Compose all transforms
    aug_pipeline = A.Compose(transforms)
    
    return AlbumentationsTransform(aug_pipeline, p=1.0)


__all__ = [
    "OrientedTransform",
    "HorizontalFlip",
    "VerticalFlip",
    "DiagonalFlip",
    "Rotate",
    "Compose",
    "AlbumentationsTransform",
    "create_albumentations_augmentation",
]

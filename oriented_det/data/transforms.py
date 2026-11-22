"""Non-geometric (photometric) augmentations for oriented detection.

Geometric flips and rotates live in ``flips.py`` and ``rotates.py``. Albumentations
does not support oriented boxes, so this module only wraps color / noise / blur.
"""

from __future__ import annotations

import math
import random

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

try:
    import albumentations as A
except ImportError:
    A = None  # type: ignore


class RandomHSVGain(A.ImageOnlyTransform if A is not None else object):  # type: ignore[misc]
    """Ultralytics / MMYOLO HSV jitter: multiply H, S, V by ``1 + U(-1, 1) * gain``.

    One gain triple per image, applied through a uint8 LUT on the OpenCV HSV image
    (``augment_hsv`` / ``YOLOv5HSVRandomAug``). Hue wraps modulo 180, saturation and
    value clip to 255. This is not ``A.HueSaturationValue``, which adds a shift: fed the
    reference fractions ``0.7`` / ``0.4`` that shift pushes most pixels to the clip rails.

    Args:
        hgain: Hue gain fraction (Ultralytics ``hsv_h``, default 0.015).
        sgain: Saturation gain fraction (``hsv_s``, default 0.7).
        vgain: Value gain fraction (``hsv_v``, default 0.4).
        p: Probability of applying the jitter.
    """

    def __init__(self, hgain: float = 0.015, sgain: float = 0.7, vgain: float = 0.4, p: float = 1.0):
        if A is None:
            raise RuntimeError(
                "albumentations is an optional extra. Install with: "
                'pip install "oriented-det[albumentations]"'
            )
        super().__init__(p=p)
        self.hgain = float(hgain)
        self.sgain = float(sgain)
        self.vgain = float(vgain)

    def get_params(self) -> dict:
        import numpy as np

        rng = getattr(self, "random_generator", np.random)
        gains = rng.uniform(-1.0, 1.0, 3) * np.array([self.hgain, self.sgain, self.vgain]) + 1.0
        return {"gains": gains}

    def apply(self, img, gains=None, **params):  # type: ignore[override]
        import cv2
        import numpy as np

        if gains is None:
            return img
        dtype = img.dtype
        hue, sat, val = cv2.split(cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2HSV))
        x = np.arange(0, 256, dtype=np.float64)
        lut_hue = ((x * gains[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * gains[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * gains[2], 0, 255).astype(dtype)
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        return cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)

    def get_transform_init_args_names(self):
        return ("hgain", "sgain", "vgain")


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
            raise RuntimeError(
                "albumentations is an optional extra. Install with: "
                'pip install "oriented-det[albumentations]"'
            )
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
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    p_hsv: float = 0.0,
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
        hsv_h: Hue gain fraction (Ultralytics ``hsv_h``): hue is multiplied by
            ``1 ± U(0, hsv_h)``. 0.015 moves the OpenCV hue index by at most 2.7 of 180.
        hsv_s: Saturation gain fraction (Ultralytics ``hsv_s``): ``× (1 ± U(0, hsv_s))``.
        hsv_v: Value gain fraction (Ultralytics ``hsv_v``): ``× (1 ± U(0, hsv_v))``.
        p_hsv: Probability of HSV jitter. 0 leaves other recipes unchanged.
    
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
        raise RuntimeError(
            "albumentations is an optional extra. Install with: "
            'pip install "oriented-det[albumentations]"'
        )
    
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

    if p_hsv > 0:
        transforms.append(
            RandomHSVGain(hgain=hsv_h, sgain=hsv_s, vgain=hsv_v, p=p_hsv)
        )
    
    # Compose all transforms
    aug_pipeline = A.Compose(transforms)
    
    return AlbumentationsTransform(aug_pipeline, p=1.0)


__all__ = [
    "AlbumentationsTransform",
    "RandomHSVGain",
    "create_albumentations_augmentation",
]

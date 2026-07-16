"""
src/augmentations_v2.py
GPU-optimized augmentation pipelines using Albumentations.
API-compatible with Albumentations >=1.4.
ElasticTransform removed — too slow for CPU-side loading.
All fast GPU-friendly transforms only.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_seg_train_transform(img_size=512):
    """Fast + effective augmentation for segmentation training."""
    if isinstance(img_size, (list, tuple)):
        h, w = img_size
    else:
        h = w = img_size

    return A.Compose([
        A.Resize(h, w),

        # Geometric — fast, no elastic/grid distortion
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.8, 1.2),
            rotate=(-30, 30),
            border_mode=0,
            p=0.7,
        ),

        # Flips
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),

        # Intensity — critical for ultrasound domain generalization
        A.OneOf([
            A.CLAHE(clip_limit=4.0, p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=1.0,
            ),
            A.RandomGamma(gamma_limit=(70, 130), p=1.0),
        ], p=0.5),

        # Noise — simulate ultrasound speckle
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=0.3),

        # Cutout — forces model to learn context
        A.CoarseDropout(
            num_holes_range=(2, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill=0,
            p=0.3,
        ),

        # Normalize with ImageNet stats
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_seg_val_transform(img_size=512):
    """Validation transform: resize + normalize only."""
    if isinstance(img_size, (list, tuple)):
        h, w = img_size
    else:
        h = w = img_size

    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_cls_train_transform(img_size=384):
    """Fast + effective augmentation for classification training."""
    return A.Compose([
        A.Resize(img_size, img_size),

        # Geometric
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.85, 1.15),
            rotate=(-20, 20),
            border_mode=0,
            p=0.6,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),

        # Intensity
        A.OneOf([
            A.CLAHE(clip_limit=4.0, p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.25, p=1.0,
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=0.5),

        # Noise
        A.GaussNoise(p=0.2),

        # Cutout
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(8, 24),
            hole_width_range=(8, 24),
            fill=0,
            p=0.3,
        ),

        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_cls_val_transform(img_size=384):
    """Validation transform for classification."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

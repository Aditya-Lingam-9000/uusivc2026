"""
src/augmentations_v2.py
Competition-grade augmentation pipelines using Albumentations.
Separate pipelines for segmentation vs classification tasks.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_seg_train_transform(img_size=512):
    """Heavy augmentation for segmentation training."""
    if isinstance(img_size, (list, tuple)):
        h, w = img_size
    else:
        h = w = img_size

    return A.Compose([
        A.Resize(h, w),

        # Geometric transforms
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.2, rotate_limit=30,
            border_mode=0, p=0.7,
        ),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
        ], p=0.4),

        # Flips
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),

        # Intensity transforms (critical for ultrasound domain generalization)
        A.OneOf([
            A.CLAHE(clip_limit=4.0, p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=1.0,
            ),
            A.RandomGamma(gamma_limit=(70, 130), p=1.0),
        ], p=0.5),

        # Noise (simulate ultrasound speckle)
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=0.3),

        # Cutout (forces model to learn context)
        A.CoarseDropout(
            max_holes=8, max_height=32, max_width=32,
            min_holes=2, min_height=8, min_width=8,
            fill_value=0, p=0.3,
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
    """Heavy augmentation for classification training."""
    return A.Compose([
        A.Resize(img_size, img_size),

        # Geometric
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15, rotate_limit=20,
            border_mode=0, p=0.6,
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
        A.GaussNoise(var_limit=(10, 50), p=0.2),

        # Cutout
        A.CoarseDropout(
            max_holes=6, max_height=24, max_width=24,
            fill_value=0, p=0.3,
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

"""
src/augmentations_v2.py
UUSIVC 2026 — Competition-Grade Augmentation Pipelines (v2)

Uses albumentations for all augmentations. Albumentations:
  - Is 2-4× faster than torchvision
  - Supports consistent joint transforms for image + mask pairs
  - Has specialized ultrasound-domain transforms

Install on Kaggle: !pip install albumentations --quiet

Usage:
    from src.augmentations_v2 import get_seg_transforms, get_cls_transforms
    train_tf = get_seg_transforms(cfg, mode='train')  # returns albumentations Compose
    val_tf   = get_seg_transforms(cfg, mode='val')
"""

import cv2
import numpy as np
import torch

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("[augmentations_v2] WARNING: albumentations not installed. Using fallback.")

import torchvision.transforms as T
from PIL import Image


# ─────────────────────────────────────────────────────────────
#  ImageNet normalization constants
# ─────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────
#  Segmentation Transforms
# ─────────────────────────────────────────────────────────────
def get_seg_transforms(cfg: dict, mode: str = "train"):
    """
    Returns albumentations Compose pipeline for segmentation.

    mode="train" → Full heavy augmentation
    mode="val"   → Resize + normalize only

    Usage with image+mask:
        aug = transform(image=np_img_hwc, mask=np_mask_hw)
        img_tensor  = aug["image"]    # (C, H, W) tensor
        mask_tensor = aug["mask"]     # (H, W) tensor → unsqueeze(0) for (1, H, W)
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return _fallback_transforms(mode)

    h = cfg.get("img_size", 512)
    w = cfg.get("img_size", 512)

    if mode == "train":
        return A.Compose([
            # ── Geometric ──────────────────────────────────────────
            A.Resize(h, w, always_apply=True),
            A.HorizontalFlip(p=cfg.get("aug_hflip_p", 0.5)),
            A.VerticalFlip(p=cfg.get("aug_vflip_p", 0.3)),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=cfg.get("aug_rotate_limit", 30),
                border_mode=cv2.BORDER_REFLECT_101,
                p=cfg.get("aug_shift_scale_rotate_p", 0.7),
            ),
            A.OneOf([
                A.ElasticTransform(
                    alpha=120, sigma=6,
                    border_mode=cv2.BORDER_REFLECT_101, p=1.0
                ),
                A.GridDistortion(
                    num_steps=5, distort_limit=0.3,
                    border_mode=cv2.BORDER_REFLECT_101, p=1.0
                ),
                A.OpticalDistortion(
                    distort_limit=0.05, shift_limit=0.05,
                    border_mode=cv2.BORDER_REFLECT_101, p=1.0
                ),
            ], p=cfg.get("aug_elastic_p", 0.3)),

            # ── Intensity (ultrasound-specific) ────────────────────
            A.OneOf([
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=1.0
                ),
                A.RandomGamma(gamma_limit=(70, 130), p=1.0),
            ], p=cfg.get("aug_brightness_p", 0.5)),

            # ── Noise (simulates ultrasound speckle) ───────────────
            A.GaussNoise(
                var_limit=(10.0, 50.0),
                mean=0,
                p=cfg.get("aug_gauss_noise_p", 0.3)
            ),

            # ── Dropout (randomly masks patches) ───────────────────
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill_value=0,
                p=cfg.get("aug_coarse_dropout_p", 0.3),
            ),

            # ── Normalize + Tensor ─────────────────────────────────
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:  # val / test
        return A.Compose([
            A.Resize(h, w, always_apply=True),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])


def get_seg_transforms_ceus(cfg: dict, mode: str = "train"):
    """
    CEUS-specific segmentation transforms (256×512 native size).
    Preserves aspect ratio — does NOT force square crop.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return _fallback_transforms(mode)

    h = cfg.get("ceus_seg_h", 256)
    w = cfg.get("ceus_seg_w", 512)

    if mode == "train":
        return A.Compose([
            A.Resize(h, w, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101, p=0.5
            ),
            A.OneOf([
                A.CLAHE(clip_limit=3.0, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            ], p=0.5),
            A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(h, w, always_apply=True),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])


def get_seg_transforms_video(cfg: dict, mode: str = "train"):
    """
    Video segmentation frame transforms (256×256 native size for cardiac).
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return _fallback_transforms(mode)

    h = cfg.get("video_seg_h", 256)
    w = cfg.get("video_seg_w", 256)

    if mode == "train":
        return A.Compose([
            A.Resize(h, w, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT_101, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(h, w, always_apply=True),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])


# ─────────────────────────────────────────────────────────────
#  Classification Transforms
# ─────────────────────────────────────────────────────────────
def get_cls_transforms(cfg: dict, mode: str = "train"):
    """
    Classification augmentation pipeline.
    Returns albumentations Compose pipeline.

    Usage:
        aug = transform(image=np_img_hwc)
        img_tensor = aug["image"]   # (3, H, W)
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return _fallback_transforms(mode)

    h = cfg.get("img_size", 512)
    w = cfg.get("img_size", 512)

    if mode == "train":
        return A.Compose([
            A.Resize(h, w, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.15, scale_limit=0.2, rotate_limit=30,
                border_mode=cv2.BORDER_REFLECT_101, p=0.6
            ),
            A.OneOf([
                A.CLAHE(clip_limit=4.0, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                A.RandomGamma(gamma_limit=(70, 130), p=1.0),
            ], p=0.6),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 6),
                hole_height_range=(8, 40),
                hole_width_range=(8, 40),
                fill_value=0, p=0.3
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(h, w, always_apply=True),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])


# ─────────────────────────────────────────────────────────────
#  Numpy / PIL helpers for dataset use
# ─────────────────────────────────────────────────────────────
def apply_seg_transform(transform, image_np: np.ndarray, mask_np: np.ndarray | None):
    """
    Apply albumentations transform to image+mask pair.
    Args:
        transform  : albumentations Compose
        image_np   : (H, W, C) uint8 numpy array
        mask_np    : (H, W) uint8 numpy array or None

    Returns:
        img_tensor  : (C, H, W) float32 tensor
        mask_tensor : (1, H, W) float32 tensor or None
    """
    if mask_np is not None:
        aug = transform(image=image_np, mask=mask_np)
        img_t  = aug["image"]                              # (C, H, W)
        mask_t = aug["mask"].float().unsqueeze(0) / 255.0  # (1, H, W)
        mask_t = mask_t.clamp(0, 1)
    else:
        aug    = transform(image=image_np)
        img_t  = aug["image"]
        mask_t = None
    return img_t, mask_t


def apply_cls_transform(transform, image_np: np.ndarray):
    """
    Apply albumentations transform to a classification image.
    Args:
        transform : albumentations Compose
        image_np  : (H, W, C) uint8 numpy array
    Returns:
        img_tensor : (C, H, W) float32 tensor
    """
    aug = transform(image=image_np)
    return aug["image"]


# ─────────────────────────────────────────────────────────────
#  Mixup & CutMix (applied at batch level in trainer)
# ─────────────────────────────────────────────────────────────
def mixup_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2):
    """
    MixUp augmentation — blends two random images and their labels.
    images : (B, C, H, W)
    labels : (B,) long or (B, num_classes) one-hot
    Returns: mixed_images, (labels_a, labels_b, lam)
    """
    if alpha <= 0:
        return images, (labels, labels, 1.0)
    lam = np.random.beta(alpha, alpha)
    B = images.size(0)
    idx = torch.randperm(B, device=images.device)
    mixed = lam * images + (1 - lam) * images[idx]
    return mixed, (labels, labels[idx], lam)


def cutmix_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0):
    """
    CutMix augmentation — replaces a rectangular region with another image.
    images : (B, C, H, W)
    labels : (B,) long
    Returns: mixed_images, (labels_a, labels_b, lam)
    """
    if alpha <= 0:
        return images, (labels, labels, 1.0)
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = images.shape
    idx = torch.randperm(B, device=images.device)

    cut_ratio = (1.0 - lam) ** 0.5
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    return mixed, (labels, labels[idx], lam)


def mixup_criterion(loss_fn, logits, labels_a, labels_b, lam):
    """Mixed loss = λ * loss(a) + (1-λ) * loss(b)"""
    return lam * loss_fn(logits, labels_a) + (1 - lam) * loss_fn(logits, labels_b)


# ─────────────────────────────────────────────────────────────
#  Fallback transforms (if albumentations not installed)
# ─────────────────────────────────────────────────────────────
def _fallback_transforms(mode: str):
    """Minimal torchvision fallback — install albumentations for best results."""
    print("[augmentations_v2] Using fallback torchvision transforms (install albumentations!)")
    if mode == "train":
        return T.Compose([
            T.Resize((512, 512)),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

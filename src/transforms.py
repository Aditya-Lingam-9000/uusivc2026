"""
src/transforms.py
Image augmentation transforms for UUSIVC 2026.

Note: These transforms apply ONLY to image tasks (image_cls, image_seg).
CEUS and video tasks (.npy) have their own normalization inside dataset.py.
"""

import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch
import numpy as np
import random

# ─────────────────────────────────────────────
#  Standard image transforms (classification)
# ─────────────────────────────────────────────
IMAGE_SIZE = 256  # resize target — based on EDA median ~550x716, but 256 is practical

def get_train_transforms():
    """For image_cls training images."""
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

def get_val_transforms():
    """For image_cls val/test images — no augmentation."""
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────
#  Segmentation transforms (joint image+mask)
# ─────────────────────────────────────────────
class SegTrainTransform:
    """
    Joint transform for image + mask (image_seg task).
    Applies the same spatial augmentations to both.
    """
    def __init__(self, size=IMAGE_SIZE):
        self.size = size

    def __call__(self, img, mask):
        # Resize both
        img  = TF.resize(img,  [self.size, self.size])
        mask = TF.resize(mask, [self.size, self.size],
                         interpolation=T.InterpolationMode.NEAREST)

        # Random horizontal flip (same for both)
        if random.random() > 0.5:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)

        # Random vertical flip
        if random.random() > 0.2:
            img  = TF.vflip(img)
            mask = TF.vflip(mask)

        # Color jitter on image only
        img = T.ColorJitter(brightness=0.2, contrast=0.2)(img)

        # To tensor + normalize image
        img  = TF.to_tensor(img)
        img  = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # Mask: to float tensor (already (1,H,W) from load_mask_png)
        # If mask is PIL, convert:
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(np.array(mask), dtype=torch.float32).unsqueeze(0)
            mask = TF.resize(mask, [self.size, self.size])
            mask = (mask > 0.5).float()

        return img, mask


class SegValTransform:
    """Val/test transform for image_seg — only resize, no augmentation."""
    def __init__(self, size=IMAGE_SIZE):
        self.size = size

    def __call__(self, img, mask=None):
        img = TF.resize(img, [self.size, self.size])
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        if mask is not None and isinstance(mask, torch.Tensor):
            import torch.nn.functional as F
            mask = F.interpolate(mask.unsqueeze(0), size=(self.size, self.size),
                                 mode='nearest').squeeze(0)
        return img, mask

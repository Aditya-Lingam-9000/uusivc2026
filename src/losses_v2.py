"""
src/losses_v2.py
Competition-grade loss functions.

Segmentation: Compound Dice + Focal + Boundary Loss
Classification: Focal Loss with label smoothing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


# ──────────────────────────────────────────────────────────────
#  Dice Loss
# ──────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


# ──────────────────────────────────────────────────────────────
#  Focal Loss (for segmentation — binary)
# ──────────────────────────────────────────────────────────────
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma

        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_weight * bce
        return loss.mean()


# ──────────────────────────────────────────────────────────────
#  Boundary Loss (distance-transform based)
# ──────────────────────────────────────────────────────────────
class BoundaryLoss(nn.Module):
    """
    Distance-transform based boundary loss.
    Penalizes predictions that are far from GT boundaries.
    Based on Kervadec et al. "Boundary loss for highly unbalanced segmentation" (2019).
    """
    def forward(self, logits, dist_maps):
        """
        logits: (B, 1, H, W) — raw model output
        dist_maps: (B, 1, H, W) — precomputed signed distance transforms
        """
        probs = torch.sigmoid(logits)
        # Inner product of probs with distance map
        # Positive dist = outside GT, negative dist = inside GT
        loss = (probs * dist_maps).mean()
        return loss


def compute_dist_map(mask_np):
    """
    Compute signed distance transform from a binary mask.
    Positive values outside GT, negative inside GT.
    mask_np: (H, W) numpy array with values {0, 1}
    Returns: (H, W) float32 numpy array
    """
    mask_bool = mask_np.astype(bool)
    if mask_bool.any() and (~mask_bool).any():
        pos_dist = distance_transform_edt(~mask_bool)   # distance outside GT
        neg_dist = distance_transform_edt(mask_bool)     # distance inside GT
        dist = pos_dist - neg_dist
    elif mask_bool.all():
        dist = -distance_transform_edt(mask_bool)
    else:
        dist = distance_transform_edt(~mask_bool)

    # Normalize to [-1, 1] range
    max_val = max(abs(dist.max()), abs(dist.min()), 1.0)
    dist = dist / max_val
    return dist.astype(np.float32)


# ──────────────────────────────────────────────────────────────
#  Compound Segmentation Loss
# ──────────────────────────────────────────────────────────────
class CompoundSegLoss(nn.Module):
    """
    Compound loss: w1*Dice + w2*Focal + w3*Boundary
    The boundary component is optional and activated when dist_maps are provided.
    """
    def __init__(self, cfg):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = BinaryFocalLoss(
            gamma=cfg["focal_gamma"],
            alpha=cfg["focal_alpha"],
        )
        self.boundary = BoundaryLoss()
        self.w_dice = cfg["dice_weight"]
        self.w_focal = cfg["focal_weight"]
        self.w_boundary = cfg["boundary_weight"]
        self.use_boundary = cfg["seg_loss"] == "dice_focal_boundary"

    def forward(self, logits, targets, dist_maps=None):
        loss_dice = self.dice(logits, targets)
        loss_focal = self.focal(logits, targets)
        total = self.w_dice * loss_dice + self.w_focal * loss_focal

        loss_boundary = torch.tensor(0.0, device=logits.device)
        if self.use_boundary and dist_maps is not None:
            loss_boundary = self.boundary(logits, dist_maps)
            total = total + self.w_boundary * loss_boundary

        return total, {
            "dice": loss_dice.item(),
            "focal": loss_focal.item(),
            "boundary": loss_boundary.item(),
        }


# ──────────────────────────────────────────────────────────────
#  Focal Loss (for classification — multi-class)
# ──────────────────────────────────────────────────────────────
class FocalCELoss(nn.Module):
    """Focal loss for multi-class classification with label smoothing."""
    def __init__(self, gamma=2.0, label_smoothing=0.1, weight=None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=label_smoothing,
            reduction='none',
        )

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()


def build_cls_loss(cfg, class_weights=None):
    """Build classification loss."""
    weight = class_weights.to(torch.float32) if class_weights is not None else None
    if cfg["cls_loss"] == "focal":
        return FocalCELoss(
            gamma=cfg["cls_focal_gamma"],
            label_smoothing=cfg["label_smoothing"],
            weight=weight,
        )
    else:
        return nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=cfg["label_smoothing"],
        )

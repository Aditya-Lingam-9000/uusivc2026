"""
src/losses_v2.py
UUSIVC 2026 — Competition-Grade Loss Functions (v2)

Losses:
  DiceLoss             — Soft Dice (segmentation)
  FocalLoss            — Focal loss for binary seg (hard examples)
  BoundaryLoss         — Distance-transform boundary loss (NSD boost)
  CompoundSegLoss      — Dice + Focal + Boundary (the v2 combined seg loss)
  FocalCELoss          — Focal CrossEntropy for classification (handles imbalance)

Usage:
    from src.losses_v2 import CompoundSegLoss, FocalCELoss
    seg_loss = CompoundSegLoss(CFG).to(device)
    cls_loss = FocalCELoss(gamma=2.0, class_weights=weights).to(device)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────
#  1. Soft Dice Loss
# ─────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    """
    Soft Dice Loss. Works directly on logits (applies sigmoid internally).
    logits  : (B, 1, H, W) raw model output
    targets : (B, 1, H, W) float binary {0, 1}
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        flat_p = probs.view(-1)
        flat_t = targets.view(-1)
        intersection = (flat_p * flat_t).sum()
        return 1.0 - (2.0 * intersection + self.smooth) / (
            flat_p.sum() + flat_t.sum() + self.smooth
        )


# ─────────────────────────────────────────────────────────────
#  2. Binary Focal Loss (for segmentation)
# ─────────────────────────────────────────────────────────────
class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for binary segmentation: forces the model to focus on
    hard-to-predict pixels (boundary regions).

    FL(p) = -α(1-p)^γ log(p)   for positive pixels
    FL(p) = -(1-α)p^γ log(1-p) for negative pixels

    gamma=2 reduces easy-example loss contribution by 99%, making the
    model heavily focus on boundary regions that are hard to classify.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE loss per-pixel
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t   = probs * targets + (1 - probs) * (1 - targets)   # p_t for each pixel
        alpha = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha * (1 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ─────────────────────────────────────────────────────────────
#  3. Boundary Loss (distance-transform based)
# ─────────────────────────────────────────────────────────────
class BoundaryLoss(nn.Module):
    """
    Soft Boundary Loss using distance transform of ground-truth boundary.

    This penalizes predictions that are far from the true boundary.
    Directly optimizes for NSD improvement.

    Implementation:
      1. Compute binary boundary of target mask (morphological erosion)
      2. Compute distance transform from boundary → weighting map
      3. BCE weighted by inverted distance (closer = higher weight)

    Reference: "Boundary loss for highly unbalanced segmentation"
               Kervadec et al., MIDL 2019
    """
    def __init__(self, theta0: float = 3.0, theta: float = 5.0):
        super().__init__()
        self.theta0 = theta0   # inner erosion kernel size
        self.theta  = theta    # boundary dilation for distance weighting

    @staticmethod
    def _get_boundary(mask: torch.Tensor) -> torch.Tensor:
        """Extract boundary pixels using morphological erosion."""
        # mask: (B, 1, H, W) float {0, 1}
        kernel = torch.ones(1, 1, 3, 3, device=mask.device)
        eroded = F.conv2d(mask, kernel, padding=1)
        # Boundary = original - fully surrounded pixels
        boundary = mask - (eroded == 9).float()
        return boundary.clamp(0, 1)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Compute boundary weight map from target
        boundary = self._get_boundary(targets)  # (B, 1, H, W)

        # Dilate boundary to create a weighting zone around edges
        kernel_size = max(3, int(self.theta) * 2 + 1)
        pad = kernel_size // 2
        weight_kernel = torch.ones(1, 1, kernel_size, kernel_size, device=boundary.device)
        weight_map = F.conv2d(boundary, weight_kernel, padding=pad).clamp(0, 1)

        # BCE weighted heavily near boundaries
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        # Increase weight near boundary by theta0
        weights = 1.0 + (self.theta0 - 1.0) * weight_map
        return (weights * bce).mean()


# ─────────────────────────────────────────────────────────────
#  4. Compound Segmentation Loss (Dice + Focal + Boundary)
# ─────────────────────────────────────────────────────────────
class CompoundSegLoss(nn.Module):
    """
    Competition-grade combined segmentation loss.

    Final loss = w_dice * Dice + w_focal * Focal + w_boundary * Boundary

    Default weights from config optimize for Score = 0.7*DSC + 0.3*NSD:
      - Dice loss      → directly optimizes DSC
      - Focal loss     → handles class imbalance + focuses on hard pixels
      - Boundary loss  → directly improves NSD (boundary precision)
    """
    def __init__(self, cfg: dict):
        super().__init__()
        self.w_dice     = cfg.get("seg_dice_weight",     0.4)
        self.w_focal    = cfg.get("seg_focal_weight",    0.3)
        self.w_boundary = cfg.get("seg_boundary_weight", 0.3)
        self.dice     = DiceLoss(smooth=1.0)
        self.focal    = BinaryFocalLoss(gamma=2.0, alpha=0.25)
        self.boundary = BoundaryLoss(theta0=3.0, theta=5.0)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        l_dice     = self.dice(logits, targets)
        l_focal    = self.focal(logits, targets)
        l_boundary = self.boundary(logits, targets)
        total = (self.w_dice * l_dice
                + self.w_focal * l_focal
                + self.w_boundary * l_boundary)
        return total, {
            "dice":     l_dice.item(),
            "focal":    l_focal.item(),
            "boundary": l_boundary.item(),
        }


# ─────────────────────────────────────────────────────────────
#  5. Focal Cross-Entropy Loss (for classification)
# ─────────────────────────────────────────────────────────────
class FocalCELoss(nn.Module):
    """
    Focal variant of CrossEntropyLoss for multi-class classification.

    Applies per-sample focal weighting: (1-p_t)^gamma
    Supports class_weights for additional imbalance correction.
    Includes label_smoothing to prevent overconfidence.

    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)
    """
    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma           = gamma
        self.class_weights   = class_weights
        self.label_smoothing = label_smoothing
        self.reduction       = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Smooth targets
        num_classes = logits.size(-1)
        ce = F.cross_entropy(
            logits, targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)
        # p_t: probability assigned to the correct class
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        # Focal weight
        focal_weight = (1.0 - p_t) ** self.gamma

        loss = focal_weight * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ─────────────────────────────────────────────────────────────
#  6. Organ-specific class weights (for classification)
# ─────────────────────────────────────────────────────────────
# Counts from EDA: (class0_count, class1_count)
_ORGAN_COUNTS = {
    # image_cls
    "Appendix":    (174, 300),
    "Breast":      (1343, 817),
    "Liver":       (207, 446),
    "Prostate":    (208, 547),
    # ceus_cls
    "BreastCEUS":  (82,  38),
    "LiverCEUS":   (24,  107),
    "ProstateCEUS":(98,  35),
    "ThyroidCEUS": (63,  57),
}

def get_organ_cls_weights(organ: str, device="cpu") -> torch.Tensor | None:
    """
    Returns a 2-element class weight tensor for use in FocalCELoss.
    Formula: w_i = total / (n_classes × count_i)
    Higher weight = rarer class → more loss contribution.
    """
    counts = _ORGAN_COUNTS.get(organ)
    if counts is None:
        return None
    c0, c1 = counts
    total = c0 + c1
    w0 = total / (2.0 * max(c0, 1))
    w1 = total / (2.0 * max(c1, 1))
    return torch.tensor([w0, w1], dtype=torch.float32, device=device)


def build_cls_losses(organs: list, device: str = "cpu") -> dict:
    """
    Builds a FocalCELoss per organ with appropriate class weights.
    Returns dict: {organ_name: FocalCELoss}
    """
    losses = {}
    for organ in organs:
        weights = get_organ_cls_weights(organ, device)
        losses[organ] = FocalCELoss(
            gamma=2.0,
            class_weights=weights,
            label_smoothing=0.1,
        ).to(device)
    return losses

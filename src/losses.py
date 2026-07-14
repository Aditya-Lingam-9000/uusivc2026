"""
src/losses.py
Loss functions for UUSIVC 2026.

Classification : CrossEntropyLoss with per-class weights (handles imbalance)
Segmentation   : BCEWithLogitsLoss + Dice Loss (combined)

Class weights from EDA:
  Liver CEUS  : [0.18, 0.82]  (severe imbalance 24 vs 107)
  Prostate CLS: [0.26, 0.74]
  All others  : roughly balanced, use uniform weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
#  Dice Loss  (for segmentation)
# ─────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits  : (B, 1, H, W) — raw model output (before sigmoid)
        targets : (B, 1, H, W) — binary float {0.0, 1.0}
        """
        probs = torch.sigmoid(logits)
        probs_flat   = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice


# ─────────────────────────────────────────────────────────────
#  Combined Seg Loss = BCE + Dice
# ─────────────────────────────────────────────────────────────
class SegLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None):
        super().__init__()
        self.bce_w  = bce_weight
        self.dice_w = dice_weight
        pw = torch.tensor([pos_weight]) if pos_weight else None
        self.bce  = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return self.bce_w * self.bce(logits, targets) + \
               self.dice_w * self.dice(logits, targets)


# ─────────────────────────────────────────────────────────────
#  Class-weighted CrossEntropy  (for classification)
# ─────────────────────────────────────────────────────────────
# Weights derived from EDA class counts (class_1_count / total)
# Higher weight on minority class.
_ORGAN_CLASS_WEIGHTS = {
    # (count_class0, count_class1)
    "image_cls/Breast":        (75,   75),
    "image_cls/Liver":         (37,   66),
    "image_cls/Appendix":      (200,  341),   # private+public combined
    "image_cls/BUSI":          (437,  210),
    "image_cls/BUS-BRA":       (1268, 607),
    "image_cls/Fatty-Liver":   (170,  380),
    "ceus_cls/Breast":         (82,   38),
    "ceus_cls/Liver":          (24,   107),
    "ceus_cls/Prostate":       (98,   35),
    "ceus_cls/Thyroid":        (63,   57),
}

def get_class_weights(task: str, organ: str, device="cpu") -> torch.Tensor | None:
    """
    Returns a 2-element weight tensor for CrossEntropyLoss, or None if balanced.
    Formula: weight_i = total / (n_classes * count_i)
    """
    key = f"{task}/{organ}"
    counts = _ORGAN_CLASS_WEIGHTS.get(key)
    if counts is None:
        return None   # unknown → no weighting
    c0, c1 = counts
    total = c0 + c1
    w0 = total / (2.0 * c0)
    w1 = total / (2.0 * c1)
    return torch.tensor([w0, w1], dtype=torch.float32, device=device)


def build_cls_loss(organ: str, task: str, device="cpu") -> nn.Module:
    """
    Returns a CrossEntropyLoss with appropriate class weights for an organ.
    """
    weights = get_class_weights(task, organ, device)
    return nn.CrossEntropyLoss(weight=weights)


def build_seg_loss(pos_weight: float | None = None) -> nn.Module:
    """
    Returns combined BCE+Dice loss for segmentation.
    pos_weight > 1 boosts positive (lesion) class — useful for small lesions.
    """
    return SegLoss(bce_weight=0.5, dice_weight=0.5, pos_weight=pos_weight)

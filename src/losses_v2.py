import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Tensor of shape (C,) or a scalar
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (B, C) logits
        # targets: (B,) class indices
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, float):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: (B, C, H, W) logits (or (B, 1, H, W))
        # targets: (B, C, H, W) probabilities or {0,1} masks
        inputs = torch.sigmoid(inputs)
        
        # Flatten
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        
        intersection = (inputs * targets).sum(dim=1)
        union = inputs.sum(dim=1) + targets.sum(dim=1)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class SegLossV2(nn.Module):
    def __init__(self, w_dice=0.4, w_focal=0.4, w_bce=0.2, pos_weight=None):
        super().__init__()
        self.w_dice = w_dice
        self.w_focal = w_focal
        self.w_bce = w_bce
        
        self.dice_loss = DiceLoss()
        
        if pos_weight is not None:
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()
            
    def forward(self, inputs, targets):
        # targets shape matching inputs
        loss_dice = self.dice_loss(inputs, targets) if self.w_dice > 0 else 0
        loss_bce = self.bce_loss(inputs, targets) if self.w_bce > 0 else 0
        
        # Binary Focal Loss
        if self.w_focal > 0:
            bce_none = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-bce_none)
            loss_focal = (((1 - pt) ** 2) * bce_none).mean()
        else:
            loss_focal = 0
            
        return self.w_dice * loss_dice + self.w_bce * loss_bce + self.w_focal * loss_focal

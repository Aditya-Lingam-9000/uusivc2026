import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassBalancedFocalLoss(nn.Module):
    """
    Focal loss to handle severe class imbalance, specifically targeted at 
    Liver (4.46:1) and Prostate (0.36:1) classification tasks.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Inputs: (Batch, NumClasses), Targets: (Batch)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


class BoundaryLoss(nn.Module):
    """
    Approximation of Hausdorff/Boundary loss to maximize the NSD (Normalized Surface Dice) metric.
    Penalizes fuzzy boundaries in ultrasound segmentation.
    """
    def __init__(self, theta0=3, theta=5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, target):
        # pred, target: (B, C, H, W) where C=1
        pred = torch.sigmoid(pred)
        
        # Simple edge detection via Sobel-like finite differences
        pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        
        target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        
        # Penalize differences in boundaries
        loss_dx = F.mse_loss(pred_dx, target_dx)
        loss_dy = F.mse_loss(pred_dy, target_dy)
        
        return loss_dx + loss_dy


class TemporalConsistencyLoss(nn.Module):
    """
    Penalizes large variations in segmentation masks between adjacent frames 
    in CEUS and Cardiac video tasks, smoothing predictions over time.
    """
    def __init__(self, weight=0.5):
        super().__init__()
        self.weight = weight

    def forward(self, preds):
        # preds: (B, T, C, H, W)
        if preds.size(1) < 2:
            return torch.tensor(0.0, device=preds.device)
            
        preds = torch.sigmoid(preds)
        
        # Calculate L2 difference between frame t and t+1
        diff = preds[:, 1:] - preds[:, :-1]
        consistency_loss = torch.mean(diff ** 2)
        
        return consistency_loss * self.weight


class UniversalLoss(nn.Module):
    def __init__(self, lambda_seg=1.0, lambda_bnd=0.5, lambda_cls=1.0, lambda_temp=0.1):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_bnd = lambda_bnd
        self.lambda_cls = lambda_cls
        self.lambda_temp = lambda_temp
        
        self.cls_loss_fn = ClassBalancedFocalLoss()
        self.bnd_loss_fn = BoundaryLoss()
        self.temp_loss_fn = TemporalConsistencyLoss()

    def forward(self, cls_preds, cls_targets, seg_preds, seg_targets, is_video=False):
        """
        Calculates a joint loss for universal multi-task learning.
        cls_preds: (B, NumClasses), cls_targets: (B)
        seg_preds: (B, 1, H, W) or (B, T, 1, H, W) for videos
        seg_targets: (B, 1, H, W) or (B, T, 1, H, W)
        """
        total_loss = 0.0
        
        # 1. Classification Loss
        if cls_targets is not None and cls_targets.numel() > 0:
            cls_loss = self.cls_loss_fn(cls_preds, cls_targets)
            total_loss += self.lambda_cls * cls_loss
            
        # 2. Segmentation Loss (Dice + BCE + Boundary)
        if seg_targets is not None and seg_targets.numel() > 0:
            if is_video:
                B, T, C, H, W = seg_preds.shape
                # Flatten time into batch for standard 2D losses
                s_preds = seg_preds.view(B * T, C, H, W)
                s_targs = seg_targets.view(B * T, C, H, W)
            else:
                s_preds = seg_preds
                s_targs = seg_targets

            bce_loss = F.binary_cross_entropy_with_logits(s_preds, s_targs)
            
            # Simple Dice Loss
            pred_sig = torch.sigmoid(s_preds)
            intersection = (pred_sig * s_targs).sum(dim=(2, 3))
            union = pred_sig.sum(dim=(2, 3)) + s_targs.sum(dim=(2, 3))
            dice_loss = 1 - (2. * intersection / (union + 1e-5)).mean()
            
            seg_loss = bce_loss + dice_loss
            total_loss += self.lambda_seg * seg_loss
            
            # Boundary Loss (for NSD)
            bnd_loss = self.bnd_loss_fn(s_preds, s_targs)
            total_loss += self.lambda_bnd * bnd_loss
            
            # 3. Temporal Consistency Loss (Videos Only)
            if is_video:
                temp_loss = self.temp_loss_fn(seg_preds)
                total_loss += self.lambda_temp * temp_loss
                
        return total_loss

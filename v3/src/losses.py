import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassBalancedFocalLoss(nn.Module):
    """
    Focal loss to handle class imbalance across multi-organ classification tasks.
    """
    def __init__(self, alpha=1.0, gamma=1.5, reduction='mean'):
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
    def __init__(self, init_weight=0.0):
        super().__init__()
        # Dynamic Task Weighting via Homoscedastic Uncertainty
        self.log_var_cls = nn.Parameter(torch.tensor(init_weight))
        self.log_var_seg = nn.Parameter(torch.tensor(init_weight))
        self.log_var_bnd = nn.Parameter(torch.tensor(init_weight))
        self.log_var_temp = nn.Parameter(torch.tensor(init_weight))
        
        self.cls_loss_fn = ClassBalancedFocalLoss()
        self.bnd_loss_fn = BoundaryLoss()
        self.temp_loss_fn = TemporalConsistencyLoss()
        
        # Massive penalty for missing tiny tumors (Class Imbalance)
        self.register_buffer('pos_weight', torch.tensor([10.0]))

    def forward(self, cls_preds, cls_targets, seg_preds, seg_targets):
        """
        Calculates a joint loss for universal multi-task learning.
        cls_preds: (B, NumClasses), cls_targets: (B)
        seg_preds: (B, T, 1, H, W) for all tasks
        seg_targets: (B, T, 1, H, W) for all tasks (dummy tasks padded with -1.0)
        """
        total_loss = 0.0
        
        # 1. Classification Loss
        # Only calculate classification loss if there are valid targets (not -1)
        if cls_targets is not None and cls_targets.numel() > 0 and (cls_targets >= 0).any():
            valid_mask = cls_targets >= 0
            cls_loss = self.cls_loss_fn(cls_preds[valid_mask], cls_targets[valid_mask])
            # Multiplied by 5.0x to balance classification gradients against 50,176-pixel segmentation gradients
            total_loss += (5.0 * cls_loss * torch.exp(-self.log_var_cls) + self.log_var_cls)
            
        # 2. Segmentation Loss (Dice + BCE + Boundary)
        if seg_targets is not None and seg_targets.numel() > 0:
            B, T, C, H, W = seg_preds.shape
            
            # Flatten time into batch for standard 2D losses
            s_preds = seg_preds.view(B * T, C, H, W)
            s_targs = seg_targets.view(B * T, C, H, W)
            
            # Mask out dummy padded frames (-1.0)
            valid_mask = (s_targs != -1.0).float()
            
            # If there's at least one valid pixel to segment
            if valid_mask.sum() > 0:
                # Compute BCE loss (unreduced) with massive foreground penalty
                bce_loss = F.binary_cross_entropy_with_logits(
                    s_preds, 
                    s_targs.clamp(min=0.0), 
                    pos_weight=self.pos_weight,
                    reduction='none'
                )
                bce_loss = (bce_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
                
                # Compute Dice Loss
                pred_sig = torch.sigmoid(s_preds) * valid_mask
                targs_clean = s_targs.clamp(min=0.0) * valid_mask
                
                intersection = (pred_sig * targs_clean).sum(dim=(2, 3))
                union = pred_sig.sum(dim=(2, 3)) + targs_clean.sum(dim=(2, 3))
                
                # Only average over batches/frames that actually had a target
                frame_has_target = valid_mask.sum(dim=(1, 2, 3)) > 0
                if frame_has_target.any():
                    dice_loss = 1 - (2. * intersection[frame_has_target] / (union[frame_has_target] + 1e-5)).mean()
                    
                    # Boundary Loss (for NSD)
                    bnd_loss = self.bnd_loss_fn(s_preds[frame_has_target], targs_clean[frame_has_target])
                    total_loss += (bnd_loss * torch.exp(-self.log_var_bnd) + self.log_var_bnd)
                else:
                    dice_loss = 0.0
                    
                seg_loss = bce_loss + dice_loss
                total_loss += (seg_loss * torch.exp(-self.log_var_seg) + self.log_var_seg)
            
            # 3. Temporal Consistency Loss (Videos Only)
            if T > 1:
                temp_loss = self.temp_loss_fn(seg_preds)
                total_loss += (temp_loss * torch.exp(-self.log_var_temp) + self.log_var_temp)
                
        return total_loss

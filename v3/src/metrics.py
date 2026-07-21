import torch
import torch.nn.functional as F

def compute_accuracy(preds, targets):
    """
    preds: (B, NumClasses) logits
    targets: (B,) class indices
    """
    if targets.numel() == 0 or not (targets >= 0).any():
        return 0.0, 0
        
    valid_mask = targets >= 0
    valid_preds = preds[valid_mask]
    valid_targets = targets[valid_mask]
    
    predicted_classes = torch.argmax(valid_preds, dim=1)
    correct = (predicted_classes == valid_targets).sum().item()
    total = valid_targets.size(0)
    
    return correct, total

def compute_dice(preds, targets):
    """
    preds: (B, T, 1, H, W) logits
    targets: (B, T, 1, H, W) with -1.0 as dummy padding
    """
    if targets.numel() == 0:
        return 0.0, 0
        
    valid_mask = (targets != -1.0).float()
    
    if valid_mask.sum() == 0:
        return 0.0, 0
        
    pred_sig = torch.sigmoid(preds) * valid_mask
    
    # Binarize predictions for strict Dice calculation
    pred_bin = (pred_sig > 0.5).float()
    targs_clean = targets.clamp(min=0.0) * valid_mask
    
    # Sum over Spatial and Channel dims (C, H, W) -> dims (2, 3, 4)
    intersection = (pred_bin * targs_clean).sum(dim=(2, 3, 4)) # Shape: (B, T)
    union = pred_bin.sum(dim=(2, 3, 4)) + targs_clean.sum(dim=(2, 3, 4)) # Shape: (B, T)
    
    # A frame is valid if it has any non-padding pixels
    frame_has_target = valid_mask.sum(dim=(2, 3, 4)) > 0 # Shape: (B, T)
    
    if not frame_has_target.any():
        return 0.0, 0
        
    # Calculate dice per frame
    dice_per_frame = (2. * intersection) / (union + 1e-5)
    
    dice = dice_per_frame[frame_has_target].sum().item()
    total_frames = frame_has_target.sum().item()
    
    return dice, total_frames

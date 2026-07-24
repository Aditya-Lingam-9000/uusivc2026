import torch
import torch.nn as nn
import torch.nn.functional as F

# Per-task calibrated positive weights based on typical foreground ratio in each organ's masks.
# Formula: (1 - fg_ratio) / fg_ratio.  Higher = rarer foreground = harder task.
TASK_POS_WEIGHTS = {
    # Segmentation organs
    'Cardiac':       3.0,   # ~25% FG — easy, penalise false negatives lightly
    'CardiacCH':     3.0,
    'CAMUS':         3.0,
    'Fetal_Head':    4.0,   # ~20% FG
    'Fetal_HC':      4.0,
    'Kidney':        7.0,   # ~12% FG
    'KidneyUS':      7.0,
    'Breast':       12.0,   # ~8%  FG
    'BreastCEUS':   12.0,
    'BUS-BRA':      12.0,
    'BUSI':         12.0,
    'BUSIS':        12.0,
    'UDIAT':        12.0,
    'Breast_luminal':12.0,
    'Liver':        80.0,   # ~1.5% FG — very rare, must amplify
    'LiverCEUS':    80.0,
    'Fatty-Liver':  20.0,   # whole-liver task, larger FG
    'Thyroid':      70.0,   # ~1.5% FG
    'ThyroidCEUS':  70.0,
    'DDTI':         70.0,
    'Appendix':     15.0,   # ~6% FG
    'Prostate':     10.0,
    'ProstateCEUS': 10.0,
}

class ClassBalancedFocalLoss(nn.Module):
    """
    Focal loss with label smoothing to handle class imbalance and prevent
    classifier collapse (overconfident constant predictions).
    """
    def __init__(self, alpha=1.0, gamma=1.5, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (Batch, NumClasses), targets: (Batch,)
        # Label smoothing prevents collapse by keeping gradient alive even when confident
        ce_loss = F.cross_entropy(inputs, targets, reduction='none',
                                  label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


class BoundaryLoss(nn.Module):
    """
    Sobel-based boundary loss that penalises boundary discrepancies for NSD metric.
    Computationally cheap alternative to full distance-transform during training.
    """
    def __init__(self, theta0=3, theta=5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, target):
        # pred, target: (B, C, H, W) where C=1
        pred = torch.sigmoid(pred)

        # Sobel-like first-order edge detection on both axes
        pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])

        target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

        loss_dx = F.mse_loss(pred_dx, target_dx)
        loss_dy = F.mse_loss(pred_dy, target_dy)

        return loss_dx + loss_dy


class TemporalConsistencyLoss(nn.Module):
    """
    Penalises large variations in segmentation masks between adjacent frames
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

        # L2 difference between frame t and t+1
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

        # Label smoothing 0.1 prevents classifier death
        self.cls_loss_fn = ClassBalancedFocalLoss(alpha=1.0, gamma=1.5, label_smoothing=0.1)
        self.bnd_loss_fn = BoundaryLoss()
        self.temp_loss_fn = TemporalConsistencyLoss()

        # Default pos_weight (per-sample overrides from organ_name below)
        self.register_buffer('default_pos_weight', torch.tensor([10.0]))

    def forward(self, cls_preds, cls_targets, seg_preds, seg_targets, organ_names=None):
        """
        Calculates a joint loss for universal multi-task learning.
        cls_preds:   (B, NumClasses), cls_targets: (B,)
        seg_preds:   (B, T, 1, H, W) for all tasks
        seg_targets: (B, T, 1, H, W) for all tasks (dummy tasks padded with -1.0)
        organ_names: list[str] of length B, used for per-task pos_weight lookup
        """
        total_loss = 0.0

        # ------------------------------------------------------------------ #
        # 1. Classification Loss                                              #
        # ------------------------------------------------------------------ #
        if cls_targets is not None and cls_targets.numel() > 0 and (cls_targets >= 0).any():
            valid_mask = cls_targets >= 0
            cls_loss = self.cls_loss_fn(cls_preds[valid_mask], cls_targets[valid_mask])
            # 5× multiplier balances gradient against segmentation (still far from 5000:1 raw)
            total_loss += (5.0 * cls_loss * torch.exp(-self.log_var_cls) + self.log_var_cls)

        # ------------------------------------------------------------------ #
        # 2. Segmentation Loss (per-instance Dice + per-task BCE + Boundary) #
        # ------------------------------------------------------------------ #
        if seg_targets is not None and seg_targets.numel() > 0:
            B, T, C, H, W = seg_preds.shape

            # Flatten time into batch for standard 2D losses
            s_preds = seg_preds.view(B * T, C, H, W)
            s_targs = seg_targets.view(B * T, C, H, W)

            # Mask out dummy padded frames (-1.0)
            valid_pixel_mask = (s_targs != -1.0).float()

            if valid_pixel_mask.sum() > 0:

                # ---- Per-frame BCE with per-task pos_weight ----
                if organ_names is not None:
                    # Build per-sample pos_weight tensor (B,) -> expand to (B*T,)
                    pw_per_b = []
                    for name in organ_names:
                        pw_per_b.append(TASK_POS_WEIGHTS.get(name, 10.0))
                    pw_tensor = torch.tensor(pw_per_b, device=seg_preds.device,
                                            dtype=seg_preds.dtype)  # (B,)
                    pw_tensor = pw_tensor.repeat_interleave(T)  # (B*T,)
                    # Compute BCE per pixel then reweight by pos_weight per frame
                    bce_unreduced = F.binary_cross_entropy_with_logits(
                        s_preds, s_targs.clamp(min=0.0), reduction='none'
                    )  # (B*T, C, H, W)
                    # Create pos_weight expanded mask: foreground pixels get pw, else 1.0
                    fg_mask = (s_targs.clamp(min=0.0) * valid_pixel_mask)  # (B*T, C, H, W)
                    pw_map = 1.0 + fg_mask * (pw_tensor.view(-1, 1, 1, 1) - 1.0)
                    bce_unreduced = bce_unreduced * pw_map
                else:
                    bce_unreduced = F.binary_cross_entropy_with_logits(
                        s_preds, s_targs.clamp(min=0.0),
                        pos_weight=self.default_pos_weight, reduction='none'
                    )
                bce_loss = (bce_unreduced * valid_pixel_mask).sum() / (valid_pixel_mask.sum() + 1e-8)

                # ---- Per-instance Dice (equal weight per frame, not per pixel) ----
                pred_sig = torch.sigmoid(s_preds) * valid_pixel_mask
                targs_clean = s_targs.clamp(min=0.0) * valid_pixel_mask

                intersection = (pred_sig * targs_clean).sum(dim=(1, 2, 3))   # (B*T,)
                union = pred_sig.sum(dim=(1, 2, 3)) + targs_clean.sum(dim=(1, 2, 3))  # (B*T,)

                frame_has_target = valid_pixel_mask.sum(dim=(1, 2, 3)) > 0  # (B*T,)
                if frame_has_target.any():
                    # Per-instance Dice: every sample frame gets equal weight
                    per_frame_dice = 2.0 * intersection / (union + 1e-5)  # (B*T,)
                    per_frame_dice_loss = 1.0 - per_frame_dice
                    dice_loss = per_frame_dice_loss[frame_has_target].mean()

                    # Boundary Loss (for NSD alignment)
                    bnd_loss = self.bnd_loss_fn(
                        s_preds[frame_has_target],
                        targs_clean[frame_has_target]
                    )
                    total_loss += (bnd_loss * torch.exp(-self.log_var_bnd) + self.log_var_bnd)
                else:
                    dice_loss = torch.tensor(0.0, device=seg_preds.device)

                seg_loss = bce_loss + dice_loss
                total_loss += (seg_loss * torch.exp(-self.log_var_seg) + self.log_var_seg)

            # ---- Temporal Consistency Loss (Videos Only) ----
            if T > 1:
                temp_loss = self.temp_loss_fn(seg_preds)
                total_loss += (temp_loss * torch.exp(-self.log_var_temp) + self.log_var_temp)

        return total_loss

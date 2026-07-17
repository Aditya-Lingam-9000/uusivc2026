"""
src/models_v2.py
UUSIVC 2026 — Competition-Grade Models (v2)

Replaces ResNet-50 + basic U-Net with:
  - EfficientNet-B5 encoder (pre-trained on ImageNet)
  - UNet++ decoder (dense nested skip connections)
  - Separate classification head for image_cls and ceus_cls
  - All via segmentation_models_pytorch (SMP)

Install on Kaggle:
    !pip install segmentation-models-pytorch timm --quiet

Models:
  build_seg_model(cfg)   → SMP UnetPlusPlus for image_seg / ceus_seg / video_seg
  build_cls_model(cfg)   → EfficientNet-B5 + MLP head for image_cls
  build_ceus_cls_model(cfg) → Temporal-pooled EfficientNet for ceus_cls
  EMA                    → Exponential Moving Average wrapper

Usage:
    from src.models_v2 import build_seg_model, build_cls_model, EMA
    seg_model  = build_seg_model(cfg).cuda()
    cls_model  = build_cls_model(cfg).cuda()
    ema        = EMA(seg_model, decay=cfg["ema_decay"])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("[models_v2] WARNING: segmentation_models_pytorch not installed.")

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("[models_v2] WARNING: timm not installed.")


# ─────────────────────────────────────────────────────────────
#  1. Segmentation Model (UNet++ with EfficientNet-B5)
# ─────────────────────────────────────────────────────────────
class SegModelV2(nn.Module):
    """
    SMP-based segmentation model.
    Default: UnetPlusPlus decoder + EfficientNet-B5 encoder

    Why UNet++:
      - Dense nested skip pathways reduce semantic gap between encoder/decoder
      - Better multi-scale feature fusion → better boundary delineation (NSD ↑)
      - 5-10% better than standard UNet on most medical benchmarks

    Why EfficientNet-B5:
      - 30M params (vs 25M ResNet-50) but 10× more efficient
      - Compound scaling balances depth, width, resolution
      - Better ImageNet features → faster convergence on medical data
    """
    def __init__(self, cfg: dict):
        super().__init__()
        if not SMP_AVAILABLE:
            raise ImportError("Install segmentation_models_pytorch: pip install segmentation-models-pytorch")

        decoder_type = cfg.get("seg_decoder", "UnetPlusPlus")
        ModelClass = getattr(smp, decoder_type)

        self.model = ModelClass(
            encoder_name=cfg.get("seg_encoder", "efficientnet-b5"),
            encoder_weights=cfg.get("seg_encoder_weights", "imagenet"),
            in_channels=cfg.get("seg_in_channels", 3),
            classes=cfg.get("seg_num_classes", 1),
            activation=None,   # raw logits — we apply sigmoid in loss
        )

        if cfg.get("gradient_checkpointing", False):
            # Enable gradient checkpointing on encoder for ~50% VRAM savings
            if hasattr(self.model.encoder, "set_grad_checkpointing"):
                self.model.encoder.set_grad_checkpointing(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)   # (B, 1, H, W) raw logits

    def get_encoder_params(self):
        return self.model.encoder.parameters()

    def get_decoder_params(self):
        params = list(self.model.decoder.parameters())
        params += list(self.model.segmentation_head.parameters())
        return params


def build_seg_model(cfg: dict) -> nn.Module:
    """Build and return the v2 segmentation model."""
    return SegModelV2(cfg)


# ─────────────────────────────────────────────────────────────
#  2. Image Classification Model
# ─────────────────────────────────────────────────────────────
class ImageCLSModelV2(nn.Module):
    """
    EfficientNet-B5 backbone + MLP classification head.

    Uses timm for EfficientNet backbone with pretrained ImageNet weights.
    Why EfficientNet-B5 over ResNet-50:
      - 85.1% ImageNet top-1 (vs 78.5% ResNet-50)
      - Better feature representations → faster convergence
      - Compound scaling: more depth + width + resolution simultaneously

    Input:  (B, 3, H, W) normalized float
    Output: (B, 2) raw logits
    """
    def __init__(self, cfg: dict):
        super().__init__()
        if not TIMM_AVAILABLE:
            # Fallback to torchvision EfficientNet
            import torchvision.models as tv
            backbone = tv.efficientnet_b5(weights=tv.EfficientNet_B5_Weights.IMAGENET1K_V1)
            feat_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            self.encoder = backbone
        else:
            self.encoder = timm.create_model(
                "efficientnet_b5",
                pretrained=True,
                num_classes=0,   # remove classification head → returns features
                global_pool="avg",
            )
            feat_dim = self.encoder.num_features

        dropout  = cfg.get("cls_dropout", 0.3)
        hidden   = cfg.get("cls_hidden_dim", 512)
        n_cls    = cfg.get("cls_num_classes", 2)

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden, n_cls),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W)
        feat = self.encoder(x)   # (B, feat_dim)
        return self.head(feat)   # (B, 2)

    def get_encoder_params(self):
        return self.encoder.parameters()

    def get_head_params(self):
        return self.head.parameters()


def build_cls_model(cfg: dict) -> nn.Module:
    return ImageCLSModelV2(cfg)


# ─────────────────────────────────────────────────────────────
#  3. CEUS Classification Model (Temporal)
# ─────────────────────────────────────────────────────────────
class CEUSCLSModelV2(nn.Module):
    """
    EfficientNet-B5 with temporal attention pooling for CEUS videos.

    Improvement over baseline (simple averaging):
      - Samples N=16 frames (was 8) for better temporal coverage
      - Temporal attention: learns WHICH frames are most discriminative
        (contrast enhancement peak, wash-out phase, etc.)
      - More robust to temporal misalignment across videos

    Input:  (B, T, 3, H, W) video tensor
    Output: (B, 2) raw logits

    Architecture:
      1. Sample n_frames evenly from T
      2. Resize each to 256×256 if needed
      3. Extract features per frame with shared EfficientNet
      4. Temporal attention pooling (learned softmax weights per frame)
      5. MLP head → logits
    """
    def __init__(self, cfg: dict):
        super().__init__()
        self.n_frames = cfg.get("ceus_n_frames", 16)

        if not TIMM_AVAILABLE:
            import torchvision.models as tv
            backbone = tv.efficientnet_b5(weights=tv.EfficientNet_B5_Weights.IMAGENET1K_V1)
            feat_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            self.encoder = backbone
        else:
            self.encoder = timm.create_model(
                "efficientnet_b5",
                pretrained=True,
                num_classes=0,
                global_pool="avg",
            )
            feat_dim = self.encoder.num_features

        # Temporal attention: scores each frame's importance
        self.temporal_attn = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        dropout = cfg.get("cls_dropout", 0.3)
        hidden  = cfg.get("cls_hidden_dim", 512)
        n_cls   = cfg.get("cls_num_classes", 2)

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden, n_cls),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # Sample n_frames evenly
        idx = torch.linspace(0, T - 1, self.n_frames).long()
        x_s = x[:, idx]                               # (B, n, C, H, W)

        # Resize to square if needed
        if H != 256 or W != 256:
            x_s = x_s.view(B * self.n_frames, C, H, W)
            x_s = F.interpolate(x_s, (256, 256), mode="bilinear", align_corners=False)
        else:
            x_s = x_s.view(B * self.n_frames, C, H, W)

        # Extract features per frame
        feats = self.encoder(x_s)                     # (B*n, feat_dim)
        feats = feats.view(B, self.n_frames, -1)       # (B, n, feat_dim)

        # Temporal attention pooling
        attn_scores = self.temporal_attn(feats)        # (B, n, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = (feats * attn_weights).sum(dim=1)    # (B, feat_dim)

        return self.head(pooled)

    def get_encoder_params(self):
        return list(self.encoder.parameters()) + list(self.temporal_attn.parameters())

    def get_head_params(self):
        return self.head.parameters()


def build_ceus_cls_model(cfg: dict) -> nn.Module:
    return CEUSCLSModelV2(cfg)


# ─────────────────────────────────────────────────────────────
#  4. Exponential Moving Average (EMA)
# ─────────────────────────────────────────────────────────────
class EMA:
    """
    Exponential Moving Average of model weights.

    Creates a shadow copy of the model that slowly tracks the training model.
    EMA model is used ONLY during validation and inference.
    Typically gives +0.5-2% improvement with no extra training cost.

    decay=0.999 means the EMA model updates by:
        ema_param = 0.999 * ema_param + 0.001 * model_param

    Usage:
        ema = EMA(model, decay=0.999)
        # After each optimizer.step():
        ema.update(model)
        # For validation:
        with ema.average_parameters():
            val_result = model(val_batch)
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay  = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: dict):
        self.shadow.load_state_dict(state_dict)


# ─────────────────────────────────────────────────────────────
#  5. Optimizer builder with differential learning rates
# ─────────────────────────────────────────────────────────────
def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """
    Builds AdamW with differential LR:
      - Pretrained encoder: cfg["encoder_lr"]  (10× smaller)
      - New decoder/head:   cfg["lr"]           (default)

    Rationale: Pretrained encoder should be fine-tuned gently to preserve
    ImageNet features, while the new decoder/head needs faster learning.
    """
    lr          = cfg.get("lr", 1e-4)
    encoder_lr  = cfg.get("encoder_lr", 1e-5)
    weight_decay= cfg.get("weight_decay", 1e-4)

    # Check if model exposes separate param groups
    if hasattr(model, "get_encoder_params") and hasattr(model, "get_decoder_params"):
        param_groups = [
            {"params": model.get_encoder_params(), "lr": encoder_lr},
            {"params": model.get_decoder_params(), "lr": lr},
        ]
    elif hasattr(model, "get_encoder_params") and hasattr(model, "get_head_params"):
        param_groups = [
            {"params": model.get_encoder_params(), "lr": encoder_lr},
            {"params": model.get_head_params(),    "lr": lr},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": lr}]

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Builds a learning rate scheduler based on cfg["scheduler"].
    """
    scheduler_type = cfg.get("scheduler", "cosine_warmrestart")
    epochs         = cfg.get("epochs", 60)
    T_0            = cfg.get("T_0", 10)
    T_mult         = cfg.get("T_mult", 2)
    eta_min        = cfg.get("eta_min", 1e-6)

    if scheduler_type == "cosine_warmrestart":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
    elif scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=eta_min
        )
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=epochs // 3, gamma=0.5
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")

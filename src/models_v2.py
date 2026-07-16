"""
src/models_v2.py
Competition-grade models using segmentation_models_pytorch (SMP).

Segmentation : UNet++ with EfficientNet-B5 encoder
Classification: EfficientNet-B5 with custom classification head
CEUS Cls      : Temporal-pooled EfficientNet-B5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_seg_model(cfg):
    """Build segmentation model using SMP."""
    import segmentation_models_pytorch as smp

    decoder_map = {
        "Unet": smp.Unet,
        "UnetPlusPlus": smp.UnetPlusPlus,
        "FPN": smp.FPN,
        "DeepLabV3Plus": smp.DeepLabV3Plus,
    }

    decoder_cls = decoder_map.get(cfg["decoder_type"], smp.UnetPlusPlus)

    model = decoder_cls(
        encoder_name=cfg["encoder_name"],
        encoder_weights=cfg["encoder_weights"],
        in_channels=cfg["in_channels"],
        classes=cfg["seg_classes"],
    )

    return model


class ImageCLSModelV2(nn.Module):
    """
    EfficientNet-B5 classifier for image_cls.
    Uses timm for backbone loading.
    Input : (B, 3, H, W)
    Output: (B, 2)
    """
    def __init__(self, cfg):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            cfg["cls_backbone"],
            pretrained=(cfg["encoder_weights"] == "imagenet"),
            num_classes=0,  # remove classification head — returns features
            drop_rate=cfg["dropout"],
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(cfg["dropout"]),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg["dropout"] * 0.5),
            nn.Linear(256, cfg["cls_classes"]),
        )

    def forward(self, x):
        features = self.backbone(x)       # (B, feat_dim)
        return self.head(features)         # (B, 2)


class CEUSCLSModelV2(nn.Module):
    """
    Temporal-pooled EfficientNet-B5 for CEUS video classification.
    Samples N frames, extracts features per frame, pools temporally.
    Input : (B, T, 3, H, W)  where T = ceus_n_frames
    Output: (B, 2)
    """
    def __init__(self, cfg):
        super().__init__()
        import timm
        self.n_frames = cfg["ceus_n_frames"]
        self.frame_size = cfg["ceus_frame_size"]

        self.backbone = timm.create_model(
            cfg["cls_backbone"],
            pretrained=(cfg["encoder_weights"] == "imagenet"),
            num_classes=0,
            drop_rate=cfg["dropout"],
        )
        feat_dim = self.backbone.num_features

        # Temporal attention pooling
        self.temporal_attn = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.head = nn.Sequential(
            nn.Dropout(cfg["dropout"]),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg["dropout"] * 0.5),
            nn.Linear(256, cfg["cls_classes"]),
        )

    def forward(self, x):
        """x: (B, T, 3, H, W) — full video frames (already sampled)"""
        B, T = x.shape[:2]

        # Resize frames if needed
        frames = x.view(B * T, *x.shape[2:])  # (B*T, 3, H, W)
        if frames.shape[-1] != self.frame_size or frames.shape[-2] != self.frame_size:
            frames = F.interpolate(frames, size=(self.frame_size, self.frame_size),
                                   mode="bilinear", align_corners=False)

        # Extract per-frame features
        feats = self.backbone(frames)          # (B*T, D)
        feats = feats.view(B, T, -1)           # (B, T, D)

        # Temporal attention pooling
        attn_weights = self.temporal_attn(feats)      # (B, T, 1)
        attn_weights = F.softmax(attn_weights, dim=1) # (B, T, 1)
        pooled = (feats * attn_weights).sum(dim=1)    # (B, D)

        return self.head(pooled)               # (B, 2)


def build_model(cfg):
    """Build appropriate model based on task config."""
    task = cfg["task"]
    if task == "seg":
        return build_seg_model(cfg)
    elif task == "cls":
        return ImageCLSModelV2(cfg)
    elif task == "ceus_cls":
        return CEUSCLSModelV2(cfg)
    else:
        raise ValueError(f"Unknown task: {task}")

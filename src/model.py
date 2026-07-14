"""
src/model.py
Baseline model definitions for all 5 UUSIVC tasks.

Models:
  ImageCLSModel  — ResNet-50 for image_cls
  CEUSCLSModel   — Temporal-pooled ResNet-50 for ceus_cls
  SegModel       — U-Net with ResNet-50 encoder for image_seg & ceus_seg
  VideoSegModel  — Frame-wise U-Net for video_seg (CardiacCH / CAMUS)

All models are binary (2-class for cls, 1-channel for seg).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ─────────────────────────────────────────────────────────────
#  Shared encoder helper
# ─────────────────────────────────────────────────────────────
def _resnet50_backbone(pretrained=True):
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    return models.resnet50(weights=weights)


# ─────────────────────────────────────────────────────────────
#  1. Image Classification  (image_cls)
# ─────────────────────────────────────────────────────────────
class ImageCLSModel(nn.Module):
    """
    ResNet-50 binary classifier for image_cls.
    Input : (B, 3, 256, 256) — normalized
    Output: (B, 2)           — raw logits
    """
    def __init__(self, num_classes=2, pretrained=True, dropout=0.3):
        super().__init__()
        bb = _resnet50_backbone(pretrained)
        self.encoder = nn.Sequential(*list(bb.children())[:-1])   # (B,2048,1,1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.encoder(x))


# ─────────────────────────────────────────────────────────────
#  2. CEUS Classification  (ceus_cls)
# ─────────────────────────────────────────────────────────────
class CEUSCLSModel(nn.Module):
    """
    ResNet-50 on temporally-sampled + averaged CEUS frames.
    Strategy: sample N evenly-spaced frames, run ResNet on each,
              average features → classify.

    Input : (B, 64, 3, 256, 512) — normalized
    Output: (B, 2)               — raw logits
    """
    def __init__(self, num_classes=2, pretrained=True, n_frames=8, dropout=0.3):
        super().__init__()
        self.n_frames = n_frames
        bb = _resnet50_backbone(pretrained)
        self.encoder = nn.Sequential(*list(bb.children())[:-1])   # (B,2048,1,1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):                                          # (B,T,3,H,W)
        B, T, C, H, W = x.shape
        # Sample n_frames evenly spaced along time axis
        idx = torch.linspace(0, T - 1, self.n_frames).long()
        x = x[:, idx]                                             # (B,n,3,H,W)
        x = x.reshape(B * self.n_frames, C, H, W)                # (B*n,3,H,W)
        # Resize to square for ResNet
        if H != 256 or W != 256:
            x = F.interpolate(x, (256, 256), mode="bilinear", align_corners=False)
        feat = self.encoder(x).flatten(1)                         # (B*n,2048)
        feat = feat.view(B, self.n_frames, 2048).mean(dim=1)      # (B,2048) avg
        return self.head(feat)


# ─────────────────────────────────────────────────────────────
#  Simple U-Net decoder blocks
# ─────────────────────────────────────────────────────────────
class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

class _UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = nn.Sequential(
            _ConvBnRelu(in_ch // 2 + skip_ch, out_ch),
            _ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ─────────────────────────────────────────────────────────────
#  3. Segmentation Model  (image_seg & ceus_seg)
# ─────────────────────────────────────────────────────────────
class SegModel(nn.Module):
    """
    U-Net with ResNet-50 encoder for binary segmentation.

    image_seg : Input  (B, 3, 256, 256)     Output (B, 1, 256, 256)
    ceus_seg  : Middle frame extracted before calling this model.
                Input  (B, 3, 256, 512)     Output (B, 1, 256, 512)
    """
    def __init__(self, pretrained=True):
        super().__init__()
        bb = _resnet50_backbone(pretrained)

        # Encoder stages — extract intermediate feature maps
        self.enc0 = nn.Sequential(bb.conv1, bb.bn1, bb.relu)      # /2  64ch
        self.pool = bb.maxpool                                     # /4
        self.enc1 = bb.layer1                                      # /4  256ch
        self.enc2 = bb.layer2                                      # /8  512ch
        self.enc3 = bb.layer3                                      # /16 1024ch
        self.enc4 = bb.layer4                                      # /32 2048ch

        # Decoder
        self.dec4 = _UpBlock(2048, 1024, 512)
        self.dec3 = _UpBlock(512,  512,  256)
        self.dec2 = _UpBlock(256,  256,  128)
        self.dec1 = _UpBlock(128,  64,   64)

        self.head = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        s0 = self.enc0(x)            # /2
        s1 = self.enc1(self.pool(s0))# /4
        s2 = self.enc2(s1)           # /8
        s3 = self.enc3(s2)           # /16
        s4 = self.enc4(s3)           # /32

        d = self.dec4(s4, s3)
        d = self.dec3(d,  s2)
        d = self.dec2(d,  s1)
        d = self.dec1(d,  s0)
        return self.head(d)          # raw logits (B,1,H,W) — apply sigmoid outside


# ─────────────────────────────────────────────────────────────
#  4. Video Segmentation  (video_seg — CardiacCH & CAMUS)
# ─────────────────────────────────────────────────────────────
class VideoSegModel(nn.Module):
    """
    Frame-wise U-Net for video_seg.

    Input : single 2D frame (B, 3, 256, 256) — extract frame before calling
    Output: (B, 1, 256, 256)                 — same as SegModel

    Strategy: iterate over annotated frames, run SegModel on each.
    This is a baseline — no temporal context (added in Phase 3).
    """
    def __init__(self, pretrained=True):
        super().__init__()
        self.seg = SegModel(pretrained=pretrained)

    def forward(self, x):           # x: (B, 3, H, W) — single frame
        return self.seg(x)          # (B, 1, H, W)


# ─────────────────────────────────────────────────────────────
#  Factory function
# ─────────────────────────────────────────────────────────────
def build_model(task: str, pretrained=True) -> nn.Module:
    """
    Returns the baseline model for a given task string.

    Usage:
        model = build_model('image_cls').cuda()
        model = build_model('image_seg').cuda()
        model = build_model('ceus_cls').cuda()
        model = build_model('ceus_seg').cuda()
        model = build_model('video_seg').cuda()
    """
    if task == "image_cls":
        return ImageCLSModel(pretrained=pretrained)
    elif task == "image_seg":
        return SegModel(pretrained=pretrained)
    elif task == "ceus_cls":
        return CEUSCLSModel(pretrained=pretrained)
    elif task == "ceus_seg":
        return SegModel(pretrained=pretrained)   # operates on middle frame
    elif task == "video_seg":
        return VideoSegModel(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown task: {task}. Must be one of: "
                         "image_cls, image_seg, ceus_cls, ceus_seg, video_seg")

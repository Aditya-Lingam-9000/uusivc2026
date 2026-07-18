import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchvision import models as tv_models

def build_seg_model_v2(cfg):
    model = smp.UnetPlusPlus(
        encoder_name=cfg.get("seg_encoder_name", "efficientnet-b5"),
        encoder_weights=cfg.get("seg_encoder_weights", "imagenet"),
        in_channels=cfg.get("in_channels", 3),
        classes=cfg.get("seg_num_classes", 1),
    )
    return model

class ClsModelV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        backbone_name = cfg.get("cls_backbone", "efficientnet_b5")
        pretrained = cfg.get("cls_pretrained", True)
        
        if backbone_name == "efficientnet_b5":
            if pretrained:
                weights = tv_models.EfficientNet_B5_Weights.IMAGENET1K_V1
                self.backbone = tv_models.efficientnet_b5(weights=weights)
            else:
                self.backbone = tv_models.efficientnet_b5()
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone_name == "convnext_base":
            if pretrained:
                weights = tv_models.ConvNeXt_Base_Weights.IMAGENET1K_V1
                self.backbone = tv_models.convnext_base(weights=weights)
            else:
                self.backbone = tv_models.convnext_base()
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 2)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

class CEUSClsModelV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.frame_model = ClsModelV2(cfg)
        
    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        frame_logits = self.frame_model(x) # (B*T, 2)
        
        # Temporal mean pooling
        frame_logits = frame_logits.view(B, T, 2)
        return frame_logits.mean(dim=1)

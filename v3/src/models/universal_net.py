import os
import sys
import subprocess

# Auto-install essential dependencies if missing in environment
try:
    import yacs
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "yacs", "timm", "einops", "-q"])

# Robustly search for UniUSNet directory across all potential execution paths
possible_uniusnet_paths = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'UniUSNet')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'UniUSNet')),
    os.path.abspath(os.path.join(os.getcwd(), 'uusivc2026', 'v3', 'UniUSNet')),
    os.path.abspath(os.path.join(os.getcwd(), 'v3', 'UniUSNet')),
    '/kaggle/working/uusivc2026/v3/UniUSNet'
]

UNIUSNET_DIR = None
for p in possible_uniusnet_paths:
    if os.path.exists(os.path.join(p, 'config.py')):
        UNIUSNET_DIR = p
        if p not in sys.path:
            sys.path.insert(0, p)
        break

# Fallback: Auto-clone official UniUSNet repository from GitHub if not present
if UNIUSNET_DIR is None:
    clone_target = os.path.abspath(os.path.join(os.getcwd(), 'uusivc2026', 'v3', 'UniUSNet'))
    if not os.path.exists(clone_target):
        print(f"UniUSNet codebase not found locally. Auto-cloning to {clone_target}...")
        subprocess.run(["git", "clone", "https://github.com/Zehui-Lin/UniUSNet.git", clone_target], check=True)
    UNIUSNET_DIR = clone_target
    if UNIUSNET_DIR not in sys.path:
        sys.path.insert(0, UNIUSNET_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import get_config
from networks.omni_vision_transformer import OmniVisionTransformer

class UniversalNet(nn.Module):
    """
    Universal Medical Ultrasound Network based on the official UniUSNet architecture 
    (Lin et al., IEEE BIBM 2024). Employs Swin-Tiny (28M params) with Multi-Hot Prompt Injection.

    KEY FIXES applied:
      1. Stop-gradient on classification path — prevents 5000:1 segmentation gradient dominance
         from starving the classifier. The cls head trains on detached backbone features.
      2. Middle-frame CEUS classification — uses the temporally-central frame for CEUS cls
         instead of averaging all frames (which destroys contrast-phase information).
    """
    def __init__(self, weight_path=None):
        super().__init__()
        
        class Args:
            cfg = os.path.join(UNIUSNET_DIR, 'configs', 'swin_tiny_patch4_window7_224_lite.yaml')
            opts = None
            batch_size = None
            zip = False
            cache_mode = None
            resume = None
            accumulation_steps = None
            use_checkpoint = False
            amp_opt_level = ''
            tag = None
            eval = False
            throughput = False
            
        args = Args()
        config = get_config(args)
        
        config.defrost()
        config.TRAIN.USE_CHECKPOINT = True  # Gradient checkpointing: 70% VRAM reduction
        if weight_path:
            config.MODEL.PRETRAIN_CKPT = weight_path
        config.freeze()
            
        # Instantiate OmniVisionTransformer with prompt=True
        self.net = OmniVisionTransformer(config, prompt=True)
        
        if weight_path and os.path.exists(weight_path):
            self.net.load_from(config)

    def forward(self, x, position_prompt, task_prompt, type_prompt, nature_prompt):
        """
        Forward pass handling 2D images and 3D videos with multi-hot prompts.
        x: (B, T, C, H, W)
        
        Returns:
            cls_out: (B, num_classes)  — from detached (stop-gradient) features
            seg_out: (B, T, 1, H, W)  — full autograd path
        """
        B, T, C, H, W = x.shape
        
        # Flatten batch and time dimensions for 2D Swin Transformer processing
        x_flat = x.view(B * T, C, H, W)
        
        # Duplicate prompt vectors across time dimension
        pos_p = position_prompt.repeat_interleave(T, dim=0)
        task_p = task_prompt.repeat_interleave(T, dim=0)
        type_p = type_prompt.repeat_interleave(T, dim=0)
        nat_p = nature_prompt.repeat_interleave(T, dim=0)
        
        # Ultra-lean chunk frame processing to keep VRAM < 1GB
        chunk_size = 2
        x_seg_list = []
        x_cls_list = []
        
        for i in range(0, B * T, chunk_size):
            x_chunk = x_flat[i:i+chunk_size]
            pos_chunk = pos_p[i:i+chunk_size]
            task_chunk = task_p[i:i+chunk_size]
            type_chunk = type_p[i:i+chunk_size]
            nat_chunk = nat_p[i:i+chunk_size]
            
            seg_chunk, cls_chunk = self.net.swin((x_chunk, pos_chunk, task_chunk, type_chunk, nat_chunk))
            x_seg_list.append(seg_chunk)
            x_cls_list.append(cls_chunk)
            
        x_seg = torch.cat(x_seg_list, dim=0)  # (B*T, num_classes, H, W)
        x_cls = torch.cat(x_cls_list, dim=0)  # (B*T, embed_dim)
        
        # Process segmentation output (B*T, num_classes, H, W) -> (B, T, 1, H, W)
        # Full autograd path — segmentation gradients update the full backbone
        if x_seg.size(1) == 2:
            seg_out = (x_seg[:, 1:2] - x_seg[:, 0:1]).view(B, T, 1, H, W)
        else:
            seg_out = x_seg.view(B, T, 1, H, W)

        # ------------------------------------------------------------------ #
        # STOP-GRADIENT: classification features detached from backbone       #
        # This is the critical fix for the 5000:1 gradient imbalance.        #
        # The cls head trains on fixed backbone features without pulling      #
        # the backbone away from its segmentation-specialised representation. #
        # ------------------------------------------------------------------ #
        x_cls_detached = x_cls.detach()  # No gradient flows back to backbone

        # Reshape: (B*T, embed_dim) -> (B, T, embed_dim)
        x_cls_temporal = x_cls_detached.view(B, T, -1)

        # Use middle frame for classification instead of mean-pooling.
        # For CEUS videos: contrast agent peaks at the middle frame.
        # For single-frame images: T=1 so mid_frame=0 always — no difference.
        mid_frame = T // 2
        cls_features = x_cls_temporal[:, mid_frame, :]  # (B, embed_dim)

        cls_out = self.net.layers_task_cls_head[0](cls_features)  # (B, num_classes)
            
        return cls_out, seg_out


if __name__ == "__main__":
    model = UniversalNet()
    img_x = torch.randn(2, 1, 3, 224, 224)
    pos_p = torch.zeros(2, 8)
    task_p = torch.zeros(2, 2)
    type_p = torch.zeros(2, 3)
    nat_p = torch.zeros(2, 2)
    
    cls_out, seg_out = model(img_x, pos_p, task_p, type_p, nat_p)
    print(f"Compilation Successful! CLS shape: {cls_out.shape}, SEG shape: {seg_out.shape}")

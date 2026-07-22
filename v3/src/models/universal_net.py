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
        
        if weight_path:
            config.defrost()
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
        """
        B, T, C, H, W = x.shape
        
        # Flatten batch and time dimensions for 2D Swin Transformer processing
        x_flat = x.view(B * T, C, H, W)
        
        # Duplicate prompt vectors across time dimension
        pos_p = position_prompt.repeat_interleave(T, dim=0)
        task_p = task_prompt.repeat_interleave(T, dim=0)
        type_p = type_prompt.repeat_interleave(T, dim=0)
        nat_p = nature_prompt.repeat_interleave(T, dim=0)
        
        # Pass through official Swin Omni Vision Transformer
        x_seg, x_cls = self.net.swin((x_flat, pos_p, task_p, type_p, nat_p))
        
        # Process segmentation output (B*T, num_classes, H, W) -> (B, T, 1, H, W)
        if x_seg.size(1) == 2:
            seg_out = (x_seg[:, 1:2] - x_seg[:, 0:1]).view(B, T, 1, H, W)
        else:
            seg_out = x_seg.view(B, T, 1, H, W)
            
        # Process classification output (B*T, num_classes) -> (B, num_classes)
        cls_out = x_cls.view(B, T, -1).mean(dim=1)
        
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

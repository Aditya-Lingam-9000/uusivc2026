import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
import os

class PromptController(nn.Module):
    """
    Generates FiLM (Feature-wise Linear Modulation) parameters based on 
    task, organ, and modality prompts to dynamically adapt the shared backbone.
    """
    def __init__(self, num_organs=10, num_modalities=3, embed_dim=64, feature_dim=1024):
        super().__init__()
        self.organ_embed = nn.Embedding(num_organs, embed_dim)
        self.modality_embed = nn.Embedding(num_modalities, embed_dim)
        
        # MLP to map concatenated embeddings to Gamma (scale) and Beta (shift)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, feature_dim * 2)
        )

    def forward(self, organ_idx, modality_idx):
        org_emb = self.organ_embed(organ_idx)
        mod_emb = self.modality_embed(modality_idx)
        
        # Combine prompts
        combined = torch.cat([org_emb, mod_emb], dim=-1)
        
        # Generate FiLM parameters
        out = self.mlp(combined)
        gamma, beta = out.chunk(2, dim=-1)
        
        # Reshape for broadcasting over spatial dimensions (B, C, 1, 1)
        return gamma.unsqueeze(-1).unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)


class ConvGRUCell(nn.Module):
    """Lightweight 2D ConvGRU cell for temporal processing of CEUS/Videos."""
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.reset_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)

    def forward(self, x, hidden):
        combined = torch.cat([x, hidden], dim=1)
        
        reset = torch.sigmoid(self.reset_gate(combined))
        update = torch.sigmoid(self.update_gate(combined))
        
        combined_reset = torch.cat([x, reset * hidden], dim=1)
        out = torch.tanh(self.out_gate(combined_reset))
        
        new_hidden = update * hidden + (1 - update) * out
        return new_hidden


class TemporalModule(nn.Module):
    """Processes video features across the time dimension with bottleneck to save VRAM."""
    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        self.compress = nn.Conv2d(feature_dim, hidden_dim, 1) # 1x1 conv bottleneck
        self.rnn = ConvGRUCell(hidden_dim, hidden_dim)
        self.expand = nn.Conv2d(hidden_dim, feature_dim, 1) # Expand back
        
    def forward(self, x):
        # x shape: (Batch, Time, Channels, Height, Width)
        B, T, C, H, W = x.shape
        
        # Compress channels first to save massive VRAM
        x = x.view(B * T, C, H, W)
        x_compressed = self.compress(x)
        x_compressed = x_compressed.view(B, T, -1, H, W)
        
        hidden = torch.zeros(B, x_compressed.size(2), H, W, device=x.device)
        
        temporal_features = []
        for t in range(T):
            hidden = self.rnn(x_compressed[:, t], hidden)
            temporal_features.append(hidden)
            
        temporal_stack = torch.stack(temporal_features, dim=1) # (B, T, hidden, H, W)
        
        # Expand back to original feature_dim for the unified heads
        temporal_stack = temporal_stack.view(B * T, -1, H, W)
        temporal_stack_expanded = self.expand(temporal_stack)
        temporal_stack_expanded = temporal_stack_expanded.view(B, T, C, H, W)
        
        final_hidden = temporal_stack_expanded[:, -1] # Last frame for classification
        
        return temporal_stack_expanded, final_hidden


class TimmEncoder(nn.Module):
    """
    Production-ready backbone wrapper for Kaggle using timm.
    Supports Swin Transformers and standard CNNs like ResNet.
    """
    def __init__(self, model_name='swin_base_patch4_window7_224', pretrained=True, weight_path=None):
        super().__init__()
        import timm
        
        # Determine if offline weights should be loaded
        if weight_path and os.path.exists(weight_path):
            print(f"Loading offline weights for {model_name} from {weight_path}")
            self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, img_size=256)
            self.backbone.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=False)
        else:
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=256)
            
        self.feature_dim = self.backbone.num_features
        
    def forward(self, x):
        # x is (B, 3, 256, 256)
        features = self.backbone.forward_features(x)
        
        # Handle Swin Transformer output dimensions (B, L, C) or (B, H, W, C)
        if features.dim() == 3:
            B, L, C = features.shape
            H = W = int(math.sqrt(L))
            features = features.transpose(1, 2).view(B, C, H, W)
        elif features.dim() == 4 and features.shape[1] != self.feature_dim:
            # Permute (B, H, W, C) to (B, C, H, W)
            features = features.permute(0, 3, 1, 2)
            
        return features

class UniversalNet(nn.Module):
    def __init__(self, backbone_name='swin_base_patch4_window7_224', num_classes=2, num_organs=15, num_modalities=3, weight_path=None):
        super().__init__()
        
        # 1. Shared Encoder (Timm Backbone)
        self.encoder = TimmEncoder(model_name=backbone_name, pretrained=True, weight_path=weight_path)
        self.feature_dim = self.encoder.feature_dim
        
        # 2. Prompting Module
        self.prompter = PromptController(num_organs, num_modalities, feature_dim=self.feature_dim)
        
        # 3. Temporal Module (Only used for videos)
        self.temporal_module = TemporalModule(self.feature_dim)
        
        # 4. Shared Heads
        # Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
        # Segmentation Head (Simple Upsampling decoder)
        self.segmenter = nn.Sequential(
            nn.Conv2d(self.feature_dim, 256, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False), # Absolute upsample
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False), # Absolute upsample to 256x256
            nn.Conv2d(64, 1, 1) # Single channel output for grayscale masks
        )

    def forward(self, x, organ_idx, modality_idx, is_video=None):
        """
        x: (B, T, C, H, W) for all inputs (images are padded to T=1 or max_T)
        """
        B, T, C, H, W = x.shape
        
        # Fold time into batch for shared 2D encoding
        x = x.view(B * T, C, H, W)
            
        # Extract features (Chunk & Checkpoint to save massive VRAM on long videos)
        chunk_size_enc = 8
        features_list = []
        for i in range(0, B * T, chunk_size_enc):
            x_chunk = x[i:i+chunk_size_enc]
            # Use gradient checkpointing to discard intermediate CNN activations
            f_chunk = checkpoint(self.encoder, x_chunk, use_reentrant=False)
            features_list.append(f_chunk)
            
        features = torch.cat(features_list, dim=0)
        
        # Generate and apply Prompts (FiLM)
        gamma, beta = self.prompter(organ_idx, modality_idx)
        
        # Duplicate prompts across the time dimension
        gamma = gamma.repeat_interleave(T, dim=0)
        beta = beta.repeat_interleave(T, dim=0)
            
        features = (features * (1 + gamma)) + beta
        
        # Unfold time and process temporally
        features = features.view(B, T, features.size(1), features.size(2), features.size(3))
        temporal_features, final_hidden = self.temporal_module(features)
        
        # Classification uses the final temporal state
        cls_out = self.classifier(self.global_pool(final_hidden).flatten(1))
        
        # Segmentation uses all temporal states
        seg_in = temporal_features.view(B * T, features.size(2), features.size(3), features.size(4))
        
        # CHUNK & CHECKPOINT TO SAVE MEMORY ON UPSAMPLING
        chunk_size = 8
        seg_out_list = []
        for i in range(0, B * T, chunk_size):
            seg_chunk_in = seg_in[i:i+chunk_size]
            # Use gradient checkpointing to discard intermediate upsampling activations
            seg_chunk_out = checkpoint(self.segmenter, seg_chunk_in, use_reentrant=False)
            seg_out_list.append(seg_chunk_out)
        
        seg_out = torch.cat(seg_out_list, dim=0)
        seg_out = seg_out.view(B, T, 1, seg_out.size(2), seg_out.size(3))
            
        return cls_out, seg_out

if __name__ == "__main__":
    # Local Dry-Run Test
    model = UniversalNet()
    
    # Mock Image Batch: 2 images, 3 channels, 256x256
    img_x = torch.randn(2, 3, 256, 256)
    organ_idx = torch.tensor([0, 1])
    mod_idx = torch.tensor([0, 0])
    
    img_cls, img_seg = model(img_x, organ_idx, mod_idx, is_video=False)
    print(f"Image CLS shape: {img_cls.shape}, SEG shape: {img_seg.shape}")
    
    # Mock Video Batch: 2 videos, 10 frames, 3 channels, 256x256
    vid_x = torch.randn(2, 10, 3, 256, 256)
    vid_cls, vid_seg = model(vid_x, organ_idx, mod_idx, is_video=True)
    print(f"Video CLS shape: {vid_cls.shape}, SEG shape: {vid_seg.shape}")
    print("Local architecture compilation successful.")

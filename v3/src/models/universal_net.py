import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """Processes video features across the time dimension."""
    def __init__(self, feature_dim):
        super().__init__()
        self.rnn = ConvGRUCell(feature_dim, feature_dim)
        
    def forward(self, x):
        # x shape: (Batch, Time, Channels, Height, Width)
        B, T, C, H, W = x.shape
        hidden = torch.zeros(B, C, H, W, device=x.device)
        
        temporal_features = []
        for t in range(T):
            hidden = self.rnn(x[:, t], hidden)
            temporal_features.append(hidden)
            
        # Return all temporal states for segmentation, or the final state for classification
        return torch.stack(temporal_features, dim=1), hidden


class UniversalNet(nn.Module):
    def __init__(self, backbone_name='resnet50', num_classes=2, num_organs=15, num_modalities=3):
        super().__init__()
        # 1. Shared Encoder (Using a standard CNN or Swin via timm. Here we mock a generic CNN structure)
        # Note: In production on Kaggle, we will replace this with `timm.create_model(backbone_name, pretrained=True)`
        self.feature_dim = 1024 # Assumed output channels of the backbone bottleneck
        
        # Simplified backbone mock for local testing
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, self.feature_dim, 3, padding=1),
            nn.ReLU()
        )
        
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
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 1, 1) # Single channel output for grayscale masks
        )

    def forward(self, x, organ_idx, modality_idx, is_video=False):
        """
        x: (B, C, H, W) for images OR (B, T, C, H, W) for videos
        """
        B = x.size(0)
        
        if is_video:
            T = x.size(1)
            # Fold time into batch for shared 2D encoding
            x = x.view(B * T, x.size(2), x.size(3), x.size(4))
            
        # Extract features
        features = self.encoder(x)
        
        # Generate and apply Prompts (FiLM)
        gamma, beta = self.prompter(organ_idx, modality_idx)
        if is_video:
            # Duplicate prompts across the time dimension
            gamma = gamma.repeat_interleave(T, dim=0)
            beta = beta.repeat_interleave(T, dim=0)
            
        features = (features * (1 + gamma)) + beta
        
        if is_video:
            # Unfold time and process temporally
            features = features.view(B, T, features.size(1), features.size(2), features.size(3))
            temporal_features, final_hidden = self.temporal_module(features)
            
            # Classification uses the final temporal state
            cls_out = self.classifier(self.global_pool(final_hidden).flatten(1))
            
            # Segmentation uses all temporal states
            seg_in = temporal_features.view(B * T, features.size(2), features.size(3), features.size(4))
            seg_out = self.segmenter(seg_in)
            seg_out = seg_out.view(B, T, 1, seg_out.size(2), seg_out.size(3))
            
        else:
            cls_out = self.classifier(self.global_pool(features).flatten(1))
            seg_out = self.segmenter(features)
            
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

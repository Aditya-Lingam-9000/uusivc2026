import os
import torch
import requests
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import our custom V3 modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.universal_net import UniversalNet
from src.dataset import UniversalDataset, get_balanced_sampler
from src.losses import UniversalLoss

# --- Hugging Face Downloader ---
def download_pretrained_weights(save_dir='./weights'):
    """
    Downloads the official UniUSNet baseline Swin-Unet weights from Hugging Face
    if they don't already exist locally (useful for Kaggle kernel init).
    """
    os.makedirs(save_dir, exist_ok=True)
    weight_path = os.path.join(save_dir, 'swin_base_patch4_window7_224_22k.pth')
    
    if not os.path.exists(weight_path):
        print(f"Downloading pre-trained weights to {weight_path}...")
        # Note: Replace with actual HF Hub URL for the specific weights needed
        url = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth"
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(weight_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        print("Download complete.")
    else:
        print("Pre-trained weights already exist locally.")
    
    return weight_path

# --- Training Loop ---
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Download Weights
    weight_path = download_pretrained_weights()
    
    # 2. Initialize Model and Loss
    # Note: On Kaggle, backbone_name would be 'swin_base_patch4_window7_224'
    # We will load the downloaded state_dict here in the real pipeline.
    model = UniversalNet(backbone_name='resnet50', num_classes=2).to(device)
    criterion = UniversalLoss(lambda_seg=1.0, lambda_bnd=0.5, lambda_cls=1.0, lambda_temp=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    # 3. Initialize Unified Dataset
    # On Kaggle, this will point to '/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN'
    # For local test, we assume a relative mock path or skip if not found
    data_dir = os.environ.get('UUSIVC_DATA_DIR', '/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN')
    
    if not os.path.exists(os.path.join(data_dir, 'dataset_json_fingerprints_v4')):
        print(f"Warning: Data directory {data_dir} not found. Skipping data loading for local test.")
        return
        
    train_dataset = UniversalDataset(data_dir=data_dir, split='Train')
    
    # Implement Weighted Random Sampler to fix class imbalance!
    sampler = get_balanced_sampler(train_dataset)
    
    # We must use batch_size=1 since videos and images have different dimensionality (B,C,H,W vs B,T,C,H,W)
    # To use larger batch sizes, we'd need a custom collate_fn that pads/groups by modality.
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=sampler, num_workers=4)
    
    # 4. Training Loop
    epochs = 10
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            x = batch['x'].to(device)
            organ_idx = batch['organ_idx'].to(device)
            modality_idx = batch['modality_idx'].to(device)
            is_video = batch['is_video'][0].item() # Extract boolean
            
            cls_target = batch['cls_target'].to(device)
            seg_target = batch['seg_target'].to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass (handles prompt gating dynamically)
            cls_preds, seg_preds = model(x, organ_idx, modality_idx, is_video=is_video)
            
            # Loss Calculation (handles missing targets automatically)
            loss = criterion(cls_preds, cls_target, seg_preds, seg_target, is_video=is_video)
            
            if loss > 0:
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            pbar.set_postfix({'loss': loss.item() if isinstance(loss, torch.Tensor) else 0.0})
            
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / len(train_loader):.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"./weights/v3_universal_model_ep{epoch+1}.pth")

if __name__ == "__main__":
    # Ensure determinism (Crucial for the "Worst of 3" evaluation rule)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    
    train()

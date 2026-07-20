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

# --- Validation Loop ---
def evaluate(model, val_loader, device):
    model.eval()
    task_metrics = {}
    
    print("\n" + "="*50)
    print("Running Detailed Validation...")
    
    with torch.no_grad():
        for batch in val_loader:
            x = batch['x'].to(device)
            organ_idx = batch['organ_idx'].to(device)
            modality_idx = batch['modality_idx'].to(device)
            
            cls_target = batch['cls_target'].to(device)
            seg_target = batch['seg_target'].to(device)
            tasks = batch['task']
            
            with torch.amp.autocast('cuda'):
                cls_preds, seg_preds = model(x, organ_idx, modality_idx)
                
            # Compute metrics per sample
            for i in range(len(tasks)):
                t = tasks[i]
                if t not in task_metrics:
                    task_metrics[t] = {'correct': 0, 'total_cls': 0, 'dice_sum': 0.0, 'total_frames': 0}
                    
                if t in ['image_cls', 'ceus_cls']:
                    c, tot = compute_accuracy(cls_preds[i:i+1], cls_target[i:i+1])
                    task_metrics[t]['correct'] += c
                    task_metrics[t]['total_cls'] += tot
                else:
                    d, tot = compute_dice(seg_preds[i:i+1], seg_target[i:i+1])
                    task_metrics[t]['dice_sum'] += d
                    task_metrics[t]['total_frames'] += tot
                    
    print("\n--- Validation Results ---")
    for t, m in task_metrics.items():
        if t in ['image_cls', 'ceus_cls']:
            acc = (m['correct'] / m['total_cls']) * 100 if m['total_cls'] > 0 else 0
            print(f"Task: {t:<15} | Accuracy: {acc:.2f}% ({m['correct']}/{m['total_cls']})")
        else:
            dice = (m['dice_sum'] / m['total_frames']) * 100 if m['total_frames'] > 0 else 0
            print(f"Task: {t:<15} | Dice Score: {dice:.2f}% ({m['total_frames']} frames evaluated)")
    print("="*50 + "\n")
    model.train()


# --- Training Loop ---
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Detected {num_gpus} GPUs! Enabling nn.DataParallel.")
    
    # 1. Download Weights
    weight_path = download_pretrained_weights()
    
    # 2. Initialize Model and Loss
    model = UniversalNet(backbone_name='resnet50', num_classes=2, num_organs=15).to(device)
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
        
    criterion = UniversalLoss(lambda_seg=1.0, lambda_bnd=0.5, lambda_cls=1.0, lambda_temp=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    # 3. Initialize Unified Dataset
    data_dir = os.environ.get('UUSIVC_DATA_DIR', '/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN')
    if not os.path.exists(os.path.join(data_dir, 'dataset_json_fingerprints_v4')):
        print(f"Warning: Data directory {data_dir} not found. Skipping data loading for local test.")
        return
        
    train_dataset = UniversalDataset(data_dir=data_dir, split='Train')
    val_dataset = UniversalDataset(data_dir=data_dir, split='Val')
    
    sampler = get_balanced_sampler(train_dataset)
    
    # Batch size 2 allows multi-GPU!
    batch_size = 2 if num_gpus > 1 else 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=pad_collate)
    
    scaler = torch.amp.GradScaler('cuda')
    
    # 4. Training Loop
    epochs = 10
    total_steps_per_epoch = len(train_loader)
    model.train()
    
    global_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        running_loss = 0.0
        epoch_start_time = time.time()
        
        print(f"\n[EPOCH {epoch+1}/{epochs}] Started...")
        
        for step, batch in enumerate(train_loader):
            step_start = time.time()
            
            x = batch['x'].to(device)
            organ_idx = batch['organ_idx'].to(device)
            modality_idx = batch['modality_idx'].to(device)
            
            cls_target = batch['cls_target'].to(device)
            seg_target = batch['seg_target'].to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                cls_preds, seg_preds = model(x, organ_idx, modality_idx)
                loss = criterion(cls_preds, cls_target, seg_preds, seg_target)
            
            if loss > 0:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                loss_val = loss.item()
                epoch_loss += loss_val
                running_loss += loss_val
                
            # Manual Logging every 100 steps
            if (step + 1) % 100 == 0 or (step + 1) == total_steps_per_epoch:
                avg_loss = running_loss / 100 if (step + 1) % 100 == 0 else running_loss / ((step + 1) % 100)
                running_loss = 0.0
                
                # ETA Calculations
                elapsed_epoch = time.time() - epoch_start_time
                steps_completed = step + 1
                time_per_step = elapsed_epoch / steps_completed
                eta_epoch = time_per_step * (total_steps_per_epoch - steps_completed)
                
                # VRAM
                vram_used = torch.cuda.memory_allocated(device) / (1024 ** 3)
                vram_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
                
                print(f"  Step [{step+1}/{total_steps_per_epoch}] "
                      f"| Loss: {avg_loss:.4f} "
                      f"| VRAM: {vram_used:.1f}GB (Res: {vram_reserved:.1f}GB) "
                      f"| ETA (Epoch): {eta_epoch/60:.1f}m")
                
        # End of Epoch
        epoch_duration = (time.time() - epoch_start_time) / 60
        total_eta = epoch_duration * (epochs - epoch - 1)
        
        print(f"\n[EPOCH {epoch+1} FINISHED] Average Loss: {epoch_loss / total_steps_per_epoch:.4f}")
        print(f"Epoch Duration: {epoch_duration:.1f}m | Total Training ETA: {total_eta/60:.1f} hrs")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"./weights/v3_universal_model_ep{epoch+1}.pth")
        
        # Run Validation
        evaluate(model, val_loader, device)

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    
    train()


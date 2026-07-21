import os
import torch
import requests
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import our custom V3 modules
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.universal_net import UniversalNet
from src.dataset import UniversalDataset, get_balanced_sampler, pad_collate
from src.losses import UniversalLoss
from src.metrics import compute_accuracy, compute_dice

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
            INV_TASK_MAPPING = {0: 'image_cls', 1: 'image_seg', 2: 'ceus_cls', 3: 'ceus_seg', 4: 'video_seg'}
            INV_ORGAN_MAPPING = {0: 'Appendix', 1: 'Breast', 2: 'Liver', 3: 'Prostate', 4: 'Thyroid', 
                                 5: 'Breast_luminal', 6: 'Cardiac', 7: 'Fetal_Head', 8: 'Kidney', 
                                 9: 'BreastCEUS', 10: 'LiverCEUS', 11: 'ProstateCEUS', 12: 'ThyroidCEUS', 13: 'CardiacCH',
                                 14: 'Unknown'}
                                 
            for i in range(len(tasks)):
                t_idx = tasks[i].item()
                o_idx = organ_idx[i].item()
                
                t = INV_TASK_MAPPING.get(t_idx, 'unknown')
                o = INV_ORGAN_MAPPING.get(o_idx, 'unknown')
                key = f"{t} ({o})"
                
                if key not in task_metrics:
                    task_metrics[key] = {'correct': 0, 'total_cls': 0, 'dice_sum': 0.0, 'total_frames': 0, 'task_type': t}
                    
                if t in ['image_cls', 'ceus_cls']:
                    c, tot = compute_accuracy(cls_preds[i:i+1], cls_target[i:i+1])
                    task_metrics[key]['correct'] += c
                    task_metrics[key]['total_cls'] += tot
                else:
                    d, tot = compute_dice(seg_preds[i:i+1], seg_target[i:i+1])
                    task_metrics[key]['dice_sum'] += d
                    task_metrics[key]['total_frames'] += tot
                    
    print("\n" + "-"*60)
    print(f"{'Task (Organ)':<35} | {'Metric':<10} | Score")
    print("-" * 60)
    
    # Sort keys for pretty printing
    for key in sorted(task_metrics.keys()):
        m = task_metrics[key]
        t = m['task_type']
        
        if t in ['image_cls', 'ceus_cls']:
            acc = (m['correct'] / m['total_cls']) * 100 if m['total_cls'] > 0 else 0
            print(f"{key:<35} | {'Accuracy':<10} | {acc:.2f}% ({m['correct']}/{m['total_cls']})")
        else:
            dice = (m['dice_sum'] / m['total_frames']) * 100 if m['total_frames'] > 0 else 0
            print(f"{key:<35} | {'Dice':<10} | {dice:.2f}% ({m['total_frames']} frames)")
            
    print("="*60 + "\n")
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
    # We now pass the downloaded local weight_path into the real Swin Transformer engine
    model = UniversalNet(backbone_name='swin_base_patch4_window7_224', num_classes=2, num_organs=15, weight_path=weight_path).to(device)
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
        
    criterion = UniversalLoss().to(device)
    
    # ADVANCED OPTIMIZATION: AdamW with tuned parameters for Swin Transformer
    # Add criterion parameters so the learnable task weights (log_vars) are optimized
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()), 
        lr=1e-3, 
        weight_decay=0.05
    )
    
    # 3. Initialize Unified Dataset
    train_dir = os.environ.get('UUSIVC_TRAIN_DIR', '/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN')
    # Use train_dir for both because we now dynamically split the TRAIN ground truth JSONs
    val_dir = train_dir
    
    if not os.path.exists(os.path.join(train_dir, 'dataset_json_fingerprints_v4')):
        print(f"Warning: Train data directory {train_dir} not found. Skipping data loading for local test.")
        return
        
    train_dataset = UniversalDataset(data_dir=train_dir, split='Train')
    val_dataset = UniversalDataset(data_dir=val_dir, split='Val')
    
    if len(val_dataset) == 0:
        print("\n" + "!"*50)
        print("WARNING: Validation dataset is EMPTY. Ensure 'private_val_for_participants.json' exists in your Kaggle input!")
        print("!"*50 + "\n")
    
    sampler = get_balanced_sampler(train_dataset)
    
    # Batch size 2 allows multi-GPU!
    batch_size = 2 if num_gpus > 1 else 1
    
    # Reduced num_workers to 2 to prevent CPU RAM OOM during long epochs
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2, collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=pad_collate)
    
    scaler = torch.amp.GradScaler('cuda')
    
    # Check for resuming
    start_epoch = int(os.environ.get('UUSIVC_START_EPOCH', '0'))
    if start_epoch > 0:
        resume_weight_path = os.environ.get('UUSIVC_RESUME_WEIGHTS', f'./weights/v3_universal_model_ep{start_epoch}.pth')
        print(f"Resuming training from Epoch {start_epoch + 1}. Loading weights: {resume_weight_path}")
        if os.path.exists(resume_weight_path):
            state_dict = torch.load(resume_weight_path, map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
            print("Weights loaded successfully!")
            
            print("\n==================================================")
            print(f"Running Validation on Loaded Epoch {start_epoch} Weights...")
            if len(val_loader) > 0:
                evaluate(model, val_loader, device)
            else:
                print("Validation loader is empty.")
        else:
            print(f"WARNING: Could not find resume weights at {resume_weight_path}")
            
    # 4. Training Loop
    epochs = 10
    total_steps_per_epoch = len(train_loader)
    
    # ADVANCED OPTIMIZATION: Gradient Accumulation (Simulate Batch Size 16 = 2 GPUs x 8 steps)
    grad_steps = 8 
    
    # ADVANCED OPTIMIZATION: OneCycleLR Scheduler
    total_optimization_steps = (total_steps_per_epoch * epochs) // grad_steps
    
    # Fast-forward scheduler if resuming
    last_epoch_step = (start_epoch * total_steps_per_epoch) // grad_steps - 1 if start_epoch > 0 else -1
    
    # PyTorch requires 'initial_lr' in the optimizer if last_epoch >= 0
    if last_epoch_step >= 0:
        for param_group in optimizer.param_groups:
            param_group.setdefault('initial_lr', 1e-3)
            
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-3, 
        total_steps=total_optimization_steps, 
        pct_start=0.1, 
        anneal_strategy='cos',
        last_epoch=last_epoch_step
    )
    
    model.train()
    
    global_start_time = time.time()
    
    for epoch in range(start_epoch, epochs):
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
                
                # Scale loss for gradient accumulation
                loss = loss / grad_steps
            
            if loss > 0:
                scaler.scale(loss).backward()
                
                # Step optimizer only after accumulating gradients
                if (step + 1) % grad_steps == 0 or (step + 1) == total_steps_per_epoch:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                
                # Unscale loss for accurate logging
                loss_val = loss.item() * grad_steps
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
                
                current_lr = scheduler.get_last_lr()[0]
                
                print(f"  Step [{step+1}/{total_steps_per_epoch}] "
                      f"| Loss: {avg_loss:.4f} "
                      f"| LR: {current_lr:.6f} "
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
        if len(val_loader) > 0:
            evaluate(model, val_loader, device)
        else:
            print("\n[SKIPPING VALIDATION] - Validation loader is empty.")

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    
    train()


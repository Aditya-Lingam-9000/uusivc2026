import os
import sys
import time
import requests
import gc
import subprocess
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Auto-install essential dependencies if missing in environment
try:
    import yacs
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "yacs", "timm", "einops", "-q"])

# Robustly search and append v3 module directories to sys.path
v3_candidate_paths = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
    os.path.abspath(os.path.join(os.getcwd(), 'uusivc2026', 'v3')),
    os.path.abspath(os.path.join(os.getcwd(), 'v3')),
    '/kaggle/working/uusivc2026/v3'
]

for p in v3_candidate_paths:
    if os.path.exists(os.path.join(p, 'src')):
        if p not in sys.path:
            sys.path.insert(0, p)
        break

from src.models.universal_net import UniversalNet
from src.dataset import UniversalDataset, get_balanced_sampler, pad_collate
from src.losses import UniversalLoss
from src.metrics import compute_accuracy, compute_dice

# --- Hugging Face / GitHub Downloader for Official Swin-Tiny Weights ---
def download_pretrained_weights(save_dir='./weights'):
    """
    Downloads official UniUSNet baseline Swin-Tiny weights if not present locally.
    """
    os.makedirs(save_dir, exist_ok=True)
    weight_path = os.path.join(save_dir, 'swin_tiny_patch4_window7_224.pth')
    
    if not os.path.exists(weight_path):
        print(f"Downloading official Swin-Tiny pre-trained weights to {weight_path}...")
        url = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(weight_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        print("Download complete.")
    else:
        print("Official Swin-Tiny pre-trained weights exist locally.")
    
    return weight_path

# --- Validation Loop ---
def evaluate(model, val_loader, device):
    model.eval()
    task_metrics = {}
    
    print("\n" + "="*50)
    print("Running Detailed Validation (UniUSNet Baseline)...")
    
    with torch.no_grad():
        for batch in val_loader:
            x = batch['x'].to(device)
            pos_p = batch['position_prompt'].to(device)
            task_p = batch['task_prompt'].to(device)
            type_p = batch['type_prompt'].to(device)
            nat_p = batch['nature_prompt'].to(device)
            
            cls_target = batch['cls_target'].to(device)
            seg_target = batch['seg_target'].to(device)
            tasks = batch['task']
            
            with torch.amp.autocast('cuda'):
                cls_preds, seg_preds = model(x, pos_p, task_p, type_p, nat_p)
                
            # Compute metrics per sample
            INV_TASK_MAPPING = {0: 'image_cls', 1: 'image_seg', 2: 'ceus_cls', 3: 'ceus_seg', 4: 'video_seg'}
            
            for i in range(len(tasks)):
                t_idx = tasks[i].item()
                t = INV_TASK_MAPPING.get(t_idx, 'unknown')
                
                # Extract organ name from prompt vector
                pos_vec = pos_p[i].tolist()
                ORGAN_POS_NAMES = ['Breast', 'Cardiac', 'Thyroid', 'Fetal_Head', 'Kidney', 'Appendix', 'Liver', 'Prostate']
                pos_idx = pos_vec.index(1.0) if 1.0 in pos_vec else 7
                o = ORGAN_POS_NAMES[pos_idx]
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
    
    # 1. Download Official Swin-Tiny Weights
    weight_path = download_pretrained_weights()
    
    # 2. Initialize OmniVisionTransformer Model and Loss
    model = UniversalNet(weight_path=weight_path).to(device)
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
        
    criterion = UniversalLoss().to(device)
    
    # Stable learning rate (1e-4) for end-to-end Swin-Tiny training
    base_lr = 1e-4
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()), 
        lr=base_lr, 
        weight_decay=0.05
    )
    
    # 3. Initialize Unified Dataset (224x224 input)
    train_dir = os.environ.get('UUSIVC_TRAIN_DIR', '/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN')
    val_dir = train_dir
    
    if not os.path.exists(os.path.join(train_dir, 'dataset_json_fingerprints_v4')):
        print(f"Warning: Train data directory {train_dir} not found. Skipping data loading for local test.")
        return
        
    train_dataset = UniversalDataset(data_dir=train_dir, split='Train')
    val_dataset = UniversalDataset(data_dir=val_dir, split='Val')
    
    if len(val_dataset) == 0:
        print("\n" + "!"*50)
        print("WARNING: Validation dataset is EMPTY. Ensure dataset is present!")
        print("!"*50 + "\n")
    
    sampler = get_balanced_sampler(train_dataset)
    batch_size = 2 if num_gpus > 1 else 1
    
    # BULLETPROOF RAM FIX: num_workers=0 to eliminate background multiprocessing leaks
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0, collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=pad_collate)
    
    scaler = torch.amp.GradScaler('cuda')
    
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
    
    epochs = 10
    total_steps_per_epoch = len(train_loader)
    grad_steps = 8
    total_optimization_steps = (total_steps_per_epoch * epochs) // grad_steps
    last_epoch_step = (start_epoch * total_steps_per_epoch) // grad_steps - 1 if start_epoch > 0 else -1
    
    if last_epoch_step >= 0:
        for param_group in optimizer.param_groups:
            param_group.setdefault('initial_lr', base_lr)
            
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=base_lr, 
        total_steps=total_optimization_steps, 
        pct_start=0.1, 
        anneal_strategy='cos',
        last_epoch=last_epoch_step
    )
    
    model.train()
    
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        running_loss = 0.0
        epoch_start_time = time.time()
        
        print(f"\n[EPOCH {epoch+1}/{epochs}] Started (Official UniUSNet Baseline)...")
        
        for step, batch in enumerate(train_loader):
            x = batch['x'].to(device)
            pos_p = batch['position_prompt'].to(device)
            task_p = batch['task_prompt'].to(device)
            type_p = batch['type_prompt'].to(device)
            nat_p = batch['nature_prompt'].to(device)
            
            cls_target = batch['cls_target'].to(device)
            seg_target = batch['seg_target'].to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                cls_preds, seg_preds = model(x, pos_p, task_p, type_p, nat_p)
                loss = criterion(cls_preds, cls_target, seg_preds, seg_target)
                loss = loss / grad_steps
            
            if loss > 0:
                scaler.scale(loss).backward()
                
                if (step + 1) % grad_steps == 0 or (step + 1) == total_steps_per_epoch:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                
                loss_val = loss.item() * grad_steps
                epoch_loss += loss_val
                running_loss += loss_val
                
            if (step + 1) % 500 == 0:
                gc.collect()
                
            if (step + 1) % 100 == 0 or (step + 1) == total_steps_per_epoch:
                avg_loss = running_loss / 100 if (step + 1) % 100 == 0 else running_loss / ((step + 1) % 100)
                running_loss = 0.0
                
                elapsed_epoch = time.time() - epoch_start_time
                steps_completed = step + 1
                time_per_step = elapsed_epoch / steps_completed
                eta_epoch = time_per_step * (total_steps_per_epoch - steps_completed)
                
                vram_used = torch.cuda.memory_allocated(device) / (1024 ** 3)
                vram_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
                current_lr = scheduler.get_last_lr()[0]
                
                print(f"  Step [{step+1}/{total_steps_per_epoch}] "
                      f"| Loss: {avg_loss:.4f} "
                      f"| LR: {current_lr:.6f} "
                      f"| VRAM: {vram_used:.1f}GB (Res: {vram_reserved:.1f}GB) "
                      f"| ETA (Epoch): {eta_epoch/60:.1f}m")
                
        epoch_duration = (time.time() - epoch_start_time) / 60
        total_eta = epoch_duration * (epochs - epoch - 1)
        
        print(f"\n[EPOCH {epoch+1} FINISHED] Average Loss: {epoch_loss / total_steps_per_epoch:.4f}")
        print(f"Epoch Duration: {epoch_duration:.1f}m | Total Training ETA: {total_eta/60:.1f} hrs")
        
        torch.save(model.state_dict(), f"./weights/v3_universal_model_ep{epoch+1}.pth")
        
        if len(val_loader) > 0:
            evaluate(model, val_loader, device)

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    
    train()

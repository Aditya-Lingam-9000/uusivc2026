import os
import json
import time
import torch
import torch.nn as nn
import requests
from torch.utils.data import DataLoader
from accelerate import Accelerator
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

# --- Evaluation Loop ---
def evaluate(model, val_loader, accelerator):
    model.eval()
    
    # Store all gathered predictions and targets across all GPUs
    all_cls_preds = []
    all_cls_targets = []
    all_seg_preds = []
    all_seg_targets = []
    all_tasks = []
    all_organs = []
    
    if accelerator.is_main_process:
        print("\n" + "="*50)
        print("Running Distributed Detailed Validation...")
    
    with torch.no_grad():
        for batch in val_loader:
            x = batch['x']
            organ_idx = batch['organ_idx']
            modality_idx = batch['modality_idx']
            
            cls_target = batch['cls_target']
            seg_target = batch['seg_target']
            tasks = batch['task']
            
            cls_preds, seg_preds = model(x, organ_idx, modality_idx)
                
            # Gather all tensors across all processes
            gathered_cls_preds, gathered_cls_targets, gathered_seg_preds, gathered_seg_targets, gathered_tasks, gathered_organs = accelerator.gather_for_metrics(
                (cls_preds, cls_target, seg_preds, seg_target, tasks, organ_idx)
            )
            
            all_cls_preds.append(gathered_cls_preds)
            all_cls_targets.append(gathered_cls_targets)
            all_seg_preds.append(gathered_seg_preds)
            all_seg_targets.append(gathered_seg_targets)
            all_tasks.append(gathered_tasks)
            all_organs.append(gathered_organs)
            
    # Compute metrics only on the main process to prevent double printing
    if accelerator.is_main_process:
        task_metrics = {}
        INV_TASK_MAPPING = {0: 'image_cls', 1: 'image_seg', 2: 'ceus_cls', 3: 'ceus_seg', 4: 'video_seg'}
        INV_ORGAN_MAPPING = {0: 'Appendix', 1: 'Breast', 2: 'Liver', 3: 'Prostate', 4: 'Thyroid', 
                             5: 'Breast_luminal', 6: 'Cardiac', 7: 'Fetal_Head', 8: 'Kidney', 
                             9: 'BreastCEUS', 10: 'LiverCEUS', 11: 'ProstateCEUS', 12: 'ThyroidCEUS', 13: 'CardiacCH',
                             14: 'Unknown'}
                             
        all_cls_preds = torch.cat(all_cls_preds, dim=0)
        all_cls_targets = torch.cat(all_cls_targets, dim=0)
        all_seg_preds = torch.cat(all_seg_preds, dim=0)
        all_seg_targets = torch.cat(all_seg_targets, dim=0)
        all_tasks = torch.cat(all_tasks, dim=0)
        all_organs = torch.cat(all_organs, dim=0)
        
        for i in range(len(all_tasks)):
            t_idx = all_tasks[i].item()
            o_idx = all_organs[i].item()
            
            t = INV_TASK_MAPPING.get(t_idx, 'unknown')
            o = INV_ORGAN_MAPPING.get(o_idx, 'unknown')
            key = f"{t} ({o})"
            
            if key not in task_metrics:
                task_metrics[key] = {'correct': 0, 'total_cls': 0, 'dice_sum': 0.0, 'total_frames': 0, 'task_type': t}
                
            if t in ['image_cls', 'ceus_cls']:
                c, tot = compute_accuracy(all_cls_preds[i:i+1], all_cls_targets[i:i+1])
                task_metrics[key]['correct'] += c
                task_metrics[key]['total_cls'] += tot
            else:
                d, tot = compute_dice(all_seg_preds[i:i+1], all_seg_targets[i:i+1])
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
    # 1. Initialize Accelerate DDP
    grad_steps = 8 
    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=grad_steps)
    
    if accelerator.is_main_process:
        print(f"Distributed Data Parallel Initiated. Detected {accelerator.num_processes} GPUs.")
    
    # 2. Download Weights
    if accelerator.is_main_process:
        download_pretrained_weights()
    accelerator.wait_for_everyone() # Wait for main process to download
    weight_path = "./weights/swin_base_patch4_window7_224_22k.pth"
    
    # 3. Initialize Model and Loss
    model = UniversalNet(backbone_name='swin_base_patch4_window7_224', num_classes=2, num_organs=15, weight_path=weight_path)
    criterion = UniversalLoss(lambda_seg=1.0, lambda_bnd=0.5, lambda_cls=1.0, lambda_temp=0.1)
    
    # ADVANCED OPTIMIZATION: AdamW with tuned parameters for Swin Transformer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    
    # 3. Initialize Unified Dataset
    train_dir = os.environ.get('UUSIVC_TRAIN_DIR', '/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN')
    val_dir = os.environ.get('UUSIVC_VAL_DIR', '/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL')
    
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
    
    # 4. Prepare with Accelerate
    # Batch size 2 per GPU = 4 total batch size with 2 GPUs
    batch_size = 2 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=pad_collate)
    
    epochs = 10
    total_steps_per_epoch = len(train_loader)
    total_optimization_steps = (total_steps_per_epoch * epochs) // grad_steps
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-3, 
        total_steps=total_optimization_steps, 
        pct_start=0.1, 
        anneal_strategy='cos'
    )
    
    # Bind everything to the DDP Multiprocessing Engine
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    
    model.train()
    
    global_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        running_loss = 0.0
        epoch_start_time = time.time()
        
        if accelerator.is_main_process:
            print(f"\n[EPOCH {epoch+1}/{epochs}] Started...")
        
        for step, batch in enumerate(train_loader):
            step_start = time.time()
            
            x = batch['x']
            organ_idx = batch['organ_idx']
            modality_idx = batch['modality_idx']
            
            cls_target = batch['cls_target']
            seg_target = batch['seg_target']
            
            # Use Accelerate accumulation context (handles gradient scaling and DDP sync implicitly)
            with accelerator.accumulate(model):
                cls_preds, seg_preds = model(x, organ_idx, modality_idx)
                loss = criterion(cls_preds, cls_target, seg_preds, seg_target)
                
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            loss_val = loss.item()
            epoch_loss += loss_val
            running_loss += loss_val
                
            # Manual Logging every 100 steps
            if accelerator.is_main_process and ((step + 1) % 100 == 0 or (step + 1) == total_steps_per_epoch):
                avg_loss = running_loss / 100 if (step + 1) % 100 == 0 else running_loss / ((step + 1) % 100)
                running_loss = 0.0
                
                # ETA Calculations
                elapsed_epoch = time.time() - epoch_start_time
                steps_completed = step + 1
                time_per_step = elapsed_epoch / steps_completed
                eta_epoch = time_per_step * (total_steps_per_epoch - steps_completed)
                
                # VRAM
                vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
                vram_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                
                current_lr = scheduler.get_last_lr()[0]
                
                print(f"  Step [{step+1}/{total_steps_per_epoch}] "
                      f"| Loss: {avg_loss:.4f} "
                      f"| LR: {current_lr:.6f} "
                      f"| VRAM: {vram_used:.1f}GB (Res: {vram_reserved:.1f}GB) "
                      f"| ETA (Epoch): {eta_epoch/60:.1f}m")
                
        # End of Epoch
        if accelerator.is_main_process:
            epoch_duration = (time.time() - epoch_start_time) / 60
            total_eta = epoch_duration * (epochs - epoch - 1)
            
            print(f"\n[EPOCH {epoch+1} FINISHED] Average Loss: {epoch_loss / total_steps_per_epoch:.4f}")
            print(f"Epoch Duration: {epoch_duration:.1f}m | Total Training ETA: {total_eta/60:.1f} hrs")
            
            # Save checkpoint
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), f"./weights/v3_universal_model_ep{epoch+1}.pth")
        
        # Run Validation
        if len(val_loader) > 0:
            evaluate(model, val_loader, accelerator)
        else:
            if accelerator.is_main_process:
                print("\n[SKIPPING VALIDATION] - Validation loader is empty.")

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    
    train()


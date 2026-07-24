"""
train_unified_v3.py  —  v4 (Phase-1+2+3 upgrades)
====================================================
Changes from previous version
------------------------------
  1. CosineAnnealingWarmRestarts replaces OneCycleLR
     - T_0 = 25 epochs, T_mult = 2  →  cycles at epochs 25, 75, 175 …
     - Periodic LR resets help escape local minima that OneCycleLR cannot
  2. base_lr raised to 3e-4  (matches effective batch scaling)
  3. grad_steps = 8  (effective batch = 16)
  4. epochs = 100  (Session 1)
  5. Robust checkpoint resume: saves and restores model + optimizer + scaler +
     scheduler state so every Kaggle session picks up exactly where the last
     one stopped. No scheduler warmup restarts; LR curve is continuous.
  6. organ_names now passed to UniversalLoss.forward() for per-task pos_weight
  7. Separate optimizer group for classification head (higher LR, lower wd)
  8. Full checkpoint dict (not just model state_dict) for seamless resuming
  9. UUSIVC_TOTAL_EPOCHS env-var lets you override total epochs without editing
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
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
from src.dataset import UniversalDataset, get_balanced_sampler, pad_collate, ORGAN_TO_POSITION
from src.losses import UniversalLoss
from src.metrics import compute_accuracy, compute_dice


# ---------------------------------------------------------------------------
# Weight download helper
# ---------------------------------------------------------------------------
def download_pretrained_weights(save_dir='./weights'):
    """Downloads official Swin-Tiny weights if not already present."""
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


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------
def evaluate(model, val_loader, device):
    """Runs detailed per-task validation and prints a formatted table."""
    model.eval()
    task_metrics = {}

    print("\n" + "="*60)
    print("Running Detailed Validation...")

    with torch.no_grad():
        for batch in val_loader:
            x         = batch['x'].to(device)
            pos_p     = batch['position_prompt'].to(device)
            task_p    = batch['task_prompt'].to(device)
            type_p    = batch['type_prompt'].to(device)
            nat_p     = batch['nature_prompt'].to(device)
            cls_target = batch['cls_target'].to(device)
            seg_target = batch['seg_target'].to(device)
            tasks     = batch['task']

            with torch.amp.autocast('cuda'):
                cls_preds, seg_preds = model(x, pos_p, task_p, type_p, nat_p)

            INV_TASK_MAPPING = {0: 'image_cls', 1: 'image_seg', 2: 'ceus_cls',
                                3: 'ceus_seg', 4: 'video_seg'}
            ORGAN_POS_NAMES = ['Breast', 'Cardiac', 'Thyroid', 'Fetal_Head',
                               'Kidney', 'Appendix', 'Liver', 'Prostate']

            for i in range(len(tasks)):
                t_idx = tasks[i].item()
                t = INV_TASK_MAPPING.get(t_idx, 'unknown')

                pos_vec = pos_p[i].tolist()
                pos_idx = pos_vec.index(1.0) if 1.0 in pos_vec else 7
                o = ORGAN_POS_NAMES[pos_idx]
                key = f"{t} ({o})"

                if key not in task_metrics:
                    task_metrics[key] = {
                        'correct': 0, 'total_cls': 0,
                        'dice_sum': 0.0, 'total_frames': 0,
                        'task_type': t
                    }

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

    cls_accs, seg_dices = [], []
    for key in sorted(task_metrics.keys()):
        m = task_metrics[key]
        t = m['task_type']
        if t in ['image_cls', 'ceus_cls']:
            acc = (m['correct'] / m['total_cls']) * 100 if m['total_cls'] > 0 else 0
            cls_accs.append(acc)
            print(f"{key:<35} | {'Accuracy':<10} | {acc:.2f}% ({m['correct']}/{m['total_cls']})")
        else:
            dice = (m['dice_sum'] / m['total_frames']) * 100 if m['total_frames'] > 0 else 0
            seg_dices.append(dice)
            print(f"{key:<35} | {'Dice':<10} | {dice:.2f}% ({m['total_frames']} frames)")

    if cls_accs:
        print(f"\n  ► Avg Classification Accuracy : {sum(cls_accs)/len(cls_accs):.2f}%")
    if seg_dices:
        print(f"  ► Avg Segmentation Dice        : {sum(seg_dices)/len(seg_dices):.2f}%")
    print("="*60 + "\n")

    model.train()
    return (sum(cls_accs)/len(cls_accs) if cls_accs else 0.0,
            sum(seg_dices)/len(seg_dices) if seg_dices else 0.0)


# ---------------------------------------------------------------------------
# Helper — extract organ names from a batch for per-task pos_weight in loss
# ---------------------------------------------------------------------------
ORGAN_POS_NAMES_LIST = ['Breast', 'Cardiac', 'Thyroid', 'Fetal_Head',
                        'Kidney', 'Appendix', 'Liver', 'Prostate']

def batch_organ_names(pos_p_batch):
    """
    Converts a position_prompt tensor (B, 8) to a list of organ name strings.
    Used to look up per-task pos_weights in UniversalLoss.
    """
    names = []
    for i in range(pos_p_batch.size(0)):
        vec = pos_p_batch[i].tolist()
        try:
            idx = vec.index(1.0)
        except ValueError:
            idx = 7  # fallback to 'indis'
        names.append(ORGAN_POS_NAMES_LIST[idx])
    return names


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train():
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus  = torch.cuda.device_count()
    print(f"Using device: {device}")
    if num_gpus > 1:
        print(f"Detected {num_gpus} GPUs! Enabling nn.DataParallel.")

    # ── 1. Download Official Swin-Tiny Weights ───────────────────────────────
    weight_path = download_pretrained_weights()

    # ── 2. Dataset ──────────────────────────────────────────────────────────
    train_dir = os.environ.get(
        'UUSIVC_TRAIN_DIR',
        '/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN'
    )

    if not os.path.exists(os.path.join(train_dir, 'dataset_json_fingerprints_v4')):
        print(f"Warning: Train data directory {train_dir} not found. Skipping.")
        return

    train_dataset = UniversalDataset(data_dir=train_dir, split='Train')
    val_dataset   = UniversalDataset(data_dir=train_dir, split='Val')

    if len(val_dataset) == 0:
        print("\n" + "!"*50)
        print("WARNING: Validation dataset is EMPTY.")
        print("!"*50 + "\n")

    sampler    = get_balanced_sampler(train_dataset)
    batch_size = 2 if num_gpus > 1 else 1

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler, num_workers=0, collate_fn=pad_collate)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False,  num_workers=0, collate_fn=pad_collate)

    # ── 3. Model ─────────────────────────────────────────────────────────────
    model = UniversalNet(weight_path=weight_path).to(device)
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)

    criterion = UniversalLoss().to(device)

    # ── 4. Dual optimizers ───────────────────────────────────────────────────
    # Optimizer A: full backbone + decoder + loss uncertainty params  (lr=3e-4)
    # Optimizer B: classification head only  (higher lr=5e-4, lower wd)
    base_lr = 3e-4

    # Identify cls-head parameters: only the linear head in layers_task_cls_head
    # OmniVisionTransformer wraps SwinTransformer as self.swin, so path is:
    # actual_model.net (OmniVisionTransformer) -> .swin (SwinTransformer) -> .layers_task_cls_head
    actual_model = model.module if hasattr(model, 'module') else model
    cls_head_params  = list(actual_model.net.swin.layers_task_cls_head.parameters())
    cls_head_ids     = {id(p) for p in cls_head_params}
    other_params     = [p for p in list(model.parameters()) + list(criterion.parameters())
                        if id(p) not in cls_head_ids]

    optimizer_seg = torch.optim.AdamW(other_params,    lr=base_lr, weight_decay=0.05)
    optimizer_cls = torch.optim.AdamW(cls_head_params, lr=5e-4,    weight_decay=0.01)

    # ── 5. Training hyperparameters ──────────────────────────────────────────
    total_epochs        = int(os.environ.get('UUSIVC_TOTAL_EPOCHS', '100'))
    grad_steps          = 8          # effective batch = batch_size × grad_steps

    # Cap steps per epoch so training doesn't take 500 min/epoch.
    # The weighted sampler loops infinitely, so we just stop after N steps.
    # Default 300: at ~4s/step → ~20 min/epoch, 100 epochs → ~33 hrs total.
    # Set UUSIVC_STEPS_PER_EPOCH=500 for more coverage at cost of longer epochs.
    full_loader_len     = len(train_loader)   # actual dataset size
    steps_per_epoch     = int(os.environ.get('UUSIVC_STEPS_PER_EPOCH',
                                              min(300, full_loader_len)))
    print(f"Dataset has {full_loader_len} batches. Capping epoch at {steps_per_epoch} steps.")

    # ── 6. Schedulers — CosineAnnealingWarmRestarts ──────────────────────────
    # Schedules operate on epoch-level, not step-level.
    # T_0=25, T_mult=2 → restarts at epoch 25, 75, 175, …
    scheduler_seg = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_seg, T_0=25, T_mult=2, eta_min=1e-6
    )
    scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_cls, T_0=25, T_mult=2, eta_min=1e-5
    )

    scaler = torch.amp.GradScaler('cuda')

    # ── 7. Checkpoint resume ─────────────────────────────────────────────────
    start_epoch = 0
    best_avg_dice = 0.0
    ckpt_save_dir = './weights'
    os.makedirs(ckpt_save_dir, exist_ok=True)

    # Priority 1: explicit resume path via env-var
    resume_ckpt_path = os.environ.get('UUSIVC_RESUME_CHECKPOINT', '')

    # Priority 2: auto-detect latest checkpoint in weights/ folder
    if not resume_ckpt_path:
        auto_ckpt = os.path.join(ckpt_save_dir, 'v3_latest_checkpoint.pth')
        if os.path.exists(auto_ckpt):
            resume_ckpt_path = auto_ckpt

    if resume_ckpt_path and os.path.exists(resume_ckpt_path):
        print(f"\n{'='*60}")
        print(f"Resuming training from checkpoint: {resume_ckpt_path}")
        ckpt = torch.load(resume_ckpt_path, map_location=device, weights_only=False)

        # Load model weights
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        elif 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            # Old format: raw state dict
            model.load_state_dict(ckpt)

        # Restore optimizer states if available
        if 'optimizer_seg_state_dict' in ckpt:
            optimizer_seg.load_state_dict(ckpt['optimizer_seg_state_dict'])
        if 'optimizer_cls_state_dict' in ckpt:
            optimizer_cls.load_state_dict(ckpt['optimizer_cls_state_dict'])

        # Restore scheduler states
        if 'scheduler_seg_state_dict' in ckpt:
            scheduler_seg.load_state_dict(ckpt['scheduler_seg_state_dict'])
        if 'scheduler_cls_state_dict' in ckpt:
            scheduler_cls.load_state_dict(ckpt['scheduler_cls_state_dict'])

        # Restore scaler
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])

        # Restore epoch counter
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch']   # epoch stored as completed epoch number

        if 'best_avg_dice' in ckpt:
            best_avg_dice = ckpt['best_avg_dice']

        print(f"Checkpoint loaded! Resuming from epoch {start_epoch + 1} / {total_epochs}")
        print(f"Best DSC so far: {best_avg_dice:.4f}")
        print('='*60 + '\n')

        # Run initial validation on loaded weights
        if len(val_loader) > 0:
            print(f"Validation on loaded weights (epoch {start_epoch}):")
            evaluate(model, val_loader, device)
    else:
        print(f"No checkpoint found — starting fresh from epoch 1 / {total_epochs}")

    # Advance schedulers to match start_epoch (so LR is correct on resume)
    if start_epoch > 0:
        for _ in range(start_epoch):
            scheduler_seg.step()
            scheduler_cls.step()

    # ── 8. Training loop ─────────────────────────────────────────────────────
    model.train()

    for epoch in range(start_epoch, total_epochs):
        epoch_loss   = 0.0
        running_loss = 0.0
        epoch_start  = time.time()

        print(f"\n[EPOCH {epoch+1}/{total_epochs}] LR_seg={scheduler_seg.get_last_lr()[0]:.6f}  "
              f"LR_cls={scheduler_cls.get_last_lr()[0]:.6f}")

        optimizer_seg.zero_grad()
        optimizer_cls.zero_grad()
        train_iter = iter(train_loader)   # fresh iterator each epoch (sampler re-shuffles)

        for step in range(steps_per_epoch):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            x          = batch['x'].to(device)
            pos_p      = batch['position_prompt'].to(device)
            task_p     = batch['task_prompt'].to(device)
            type_p     = batch['type_prompt'].to(device)
            nat_p      = batch['nature_prompt'].to(device)
            cls_target = batch['cls_target'].to(device)
            seg_target = batch['seg_target'].to(device)

            # Extract organ names for per-task pos_weight lookup in loss
            organ_names = batch_organ_names(pos_p)

            with torch.amp.autocast('cuda'):
                cls_preds, seg_preds = model(x, pos_p, task_p, type_p, nat_p)
                loss = criterion(cls_preds, cls_target, seg_preds, seg_target,
                                 organ_names=organ_names)
                loss = loss / grad_steps

            if loss > 0:
                scaler.scale(loss).backward()

                if (step + 1) % grad_steps == 0 or (step + 1) == steps_per_epoch:
                    # Gradient clipping for stability
                    scaler.unscale_(optimizer_seg)
                    scaler.unscale_(optimizer_cls)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    scaler.step(optimizer_seg)
                    scaler.step(optimizer_cls)
                    scaler.update()
                    optimizer_seg.zero_grad()
                    optimizer_cls.zero_grad()

                loss_val   = loss.item() * grad_steps
                epoch_loss  += loss_val
                running_loss += loss_val

            # Periodic memory cleanup
            if (step + 1) % 200 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            # Step-level logging every 100 steps
            if (step + 1) % 100 == 0 or (step + 1) == total_steps_per_epoch:
                window = 100 if (step + 1) % 100 == 0 else (step + 1) % 100
                avg_loss    = running_loss / max(window, 1)
                running_loss = 0.0

                elapsed     = time.time() - epoch_start
                time_per_step = elapsed / (step + 1)
                eta_epoch   = time_per_step * (total_steps_per_epoch - step - 1)

                vram_used  = torch.cuda.memory_allocated(device) / (1024 ** 3)
                vram_res   = torch.cuda.memory_reserved(device) / (1024 ** 3)

                print(f"  Step [{step+1}/{steps_per_epoch}] "
                      f"| Loss: {avg_loss:.4f} "
                      f"| LR_seg: {scheduler_seg.get_last_lr()[0]:.6f} "
                      f"| LR_cls: {scheduler_cls.get_last_lr()[0]:.6f} "
                      f"| VRAM: {vram_used:.1f}GB (Rsv: {vram_res:.1f}GB) "
                      f"| ETA: {eta_epoch/60:.1f}m")

        # ── End of epoch ────────────────────────────────────────────────────
        epoch_duration = (time.time() - epoch_start) / 60
        remaining_epochs = total_epochs - epoch - 1
        total_eta = epoch_duration * remaining_epochs

        print(f"\n[EPOCH {epoch+1} DONE] "
              f"Avg Loss: {epoch_loss / max(total_steps_per_epoch, 1):.4f} "
              f"| Duration: {epoch_duration:.1f}m "
              f"| ETA total: {total_eta/60:.1f} hrs")

        # Advance schedulers (epoch-based, called once per epoch)
        scheduler_seg.step()
        scheduler_cls.step()

        # ── Validation ────────────────────────────────────────────────────
        avg_cls_acc = 0.0
        avg_seg_dice = 0.0
        if len(val_loader) > 0:
            avg_cls_acc, avg_seg_dice = evaluate(model, val_loader, device)

        # ── Save per-epoch weight (for external reference) ────────────────
        ep_weight_path = os.path.join(ckpt_save_dir, f'v3_universal_model_ep{epoch+1}.pth')
        torch.save(model.state_dict(), ep_weight_path)

        # ── Save full checkpoint (the ONLY file needed for resume) ────────
        checkpoint = {
            'epoch':                    epoch + 1,   # completed epoch
            'total_epochs':             total_epochs,
            'model_state_dict':         model.state_dict(),
            'optimizer_seg_state_dict': optimizer_seg.state_dict(),
            'optimizer_cls_state_dict': optimizer_cls.state_dict(),
            'scheduler_seg_state_dict': scheduler_seg.state_dict(),
            'scheduler_cls_state_dict': scheduler_cls.state_dict(),
            'scaler_state_dict':        scaler.state_dict(),
            'best_avg_dice':            best_avg_dice,
            'avg_cls_acc':              avg_cls_acc,
            'avg_seg_dice':             avg_seg_dice,
        }
        latest_ckpt_path = os.path.join(ckpt_save_dir, 'v3_latest_checkpoint.pth')
        torch.save(checkpoint, latest_ckpt_path)
        print(f"[CHECKPOINT] Saved → {latest_ckpt_path}")

        # ── Save best model ───────────────────────────────────────────────
        if avg_seg_dice > best_avg_dice:
            best_avg_dice = avg_seg_dice
            best_path = os.path.join(ckpt_save_dir, 'v3_best_model.pth')
            torch.save(model.state_dict(), best_path)
            print(f"[BEST MODEL] New best DSC: {best_avg_dice:.4f}% → {best_path}")

    print("\n" + "="*60)
    print(f"Training complete! Epochs {start_epoch+1}–{total_epochs} finished.")
    print(f"Best validation DSC: {best_avg_dice:.4f}%")
    print("="*60)


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.manual_seed(42)

    train()

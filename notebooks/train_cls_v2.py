"""
notebooks/train_cls_v2.py
UUSIVC 2026 — Competition-Grade Classification Training (v2)

Trains classification models:
  - image_cls  (4 organs: Appendix, Breast, Liver, Prostate)
  - ceus_cls   (4 organs: BreastCEUS, LiverCEUS, ProstateCEUS, ThyroidCEUS)

Key improvements over v1:
  ✅ EfficientNet-B5 backbone (replaces ResNet-50)
  ✅ Temporal attention pooling for CEUS (replaces simple averaging)
  ✅ Focal Loss + Label Smoothing (handles class imbalance, LiverCEUS AUC fix)
  ✅ WeightedRandomSampler (balanced batches per organ)
  ✅ Mixup augmentation
  ✅ Albumentations augmentation pipeline
  ✅ Mixed Precision (AMP)
  ✅ EMA model for validation
  ✅ Resume from checkpoint
  ✅ Per-25-step logs with VRAM + ETA

HOW TO RUN on Kaggle:
    !pip install segmentation-models-pytorch albumentations timm --quiet
    !cd /kaggle/working && git clone https://github.com/Aditya-Lingam-9000/uusivc2026.git repo
    import sys; sys.path.insert(0, '/kaggle/working/repo')
    TRAIN_PATH = "/kaggle/input/.../TRAIN"
    VAL_PATH   = "/kaggle/input/.../VAL"

    TASK = "image_cls"   # or "ceus_cls"
    exec(open('/kaggle/working/repo/notebooks/train_cls_v2.py').read())

    # Resume:
    CFG["resume_from"] = f"/kaggle/working/checkpoints/{TASK}_v2_latest.pth"
    exec(open('/kaggle/working/repo/notebooks/train_cls_v2.py').read())
"""

import sys, os, json, random, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ── Force src reload ───────────────────────────────────────────
for mod in list(sys.modules.keys()):
    if mod.startswith("src"):
        del sys.modules[mod]

# ── Config / Paths ─────────────────────────────────────────────
TRAIN_PATH = globals().get("TRAIN_PATH", "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN")
VAL_PATH   = globals().get("VAL_PATH",   "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL")
TASK       = globals().get("TASK", "image_cls")   # "image_cls" | "ceus_cls"

from src.config import CFG
CFG["train_path"] = TRAIN_PATH
CFG["val_path"]   = VAL_PATH
CFG["ckpt_dir"]   = "/kaggle/working/checkpoints"

# ── Task-specific overrides ─────────────────────────────────────
if TASK == "image_cls":
    CFG["img_size"]    = 512
    CFG["batch_size"]  = 16
    CFG["grad_accum_steps"] = 2   # Effective batch = 32
    CFG["epochs"]      = 50
    CFG["ceus_n_frames"] = 8      # not used for image_cls
elif TASK == "ceus_cls":
    CFG["img_size"]    = 256
    CFG["batch_size"]  = 4        # CEUS videos are large
    CFG["grad_accum_steps"] = 8   # Effective batch = 32
    CFG["epochs"]      = 40
    CFG["ceus_n_frames"] = 16

CKPT_PREFIX = f"{TASK}_v2"

# ── Device ─────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")

torch.manual_seed(CFG["seed"]); random.seed(CFG["seed"]); np.random.seed(CFG["seed"])

# ── Imports ────────────────────────────────────────────────────
from src.dataset import get_partition_root
from src.models_v2 import build_cls_model, build_ceus_cls_model, EMA, build_optimizer, build_scheduler
from src.losses_v2 import FocalCELoss, build_cls_losses, get_organ_cls_weights
from src.augmentations_v2 import (
    get_cls_transforms, apply_cls_transform,
    mixup_batch, mixup_criterion, ALBUMENTATIONS_AVAILABLE
)
from src.trainer import Trainer, get_vram_info, fmt_vram, fmt_time, accuracy

print("✅ All imports OK")

# ─────────────────────────────────────────────────────────────
#  Load JSON
# ─────────────────────────────────────────────────────────────
PRIVATE_GT = f"{TRAIN_PATH}/dataset_json_fingerprints_v4/private_train_ground_truth.json"
PUBLIC_GT  = f"{TRAIN_PATH}/dataset_json_fingerprints_v4/public_all_ground_truth.json"

all_samples = []
for jp in [PRIVATE_GT, PUBLIC_GT]:
    if os.path.exists(jp):
        with open(jp) as f:
            all_samples.extend(json.load(f))

task_samples = [s for s in all_samples if s["task"] == TASK]
print(f"\nTotal {TASK} samples: {len(task_samples)}")

organ_counts = defaultdict(lambda: [0, 0])
for s in task_samples:
    organ_counts[s["organ"]][s.get("class_label_index", 0)] += 1
print("Per-organ class distribution:")
for organ, counts in sorted(organ_counts.items()):
    total = sum(counts)
    ratio = counts[0] / max(counts[1], 1)
    print(f"  {organ:20s}: class0={counts[0]:4d}  class1={counts[1]:4d}  "
          f"total={total:4d}  ratio={ratio:.2f}:1")

# ─────────────────────────────────────────────────────────────
#  Dataset classes
# ─────────────────────────────────────────────────────────────
from PIL import Image as PILImage

class ImageClsDataset(Dataset):
    def __init__(self, samples, train_root, val_root, transform):
        self.samples    = samples
        self.train_root = Path(train_root)
        self.val_root   = Path(val_root) if val_root else None
        self.transform  = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s  = self.samples[idx]
        pr = get_partition_root(self.train_root, self.val_root, s["data_partition_group"])
        img_path = pr / s["input_path_relative"]
        img_np   = np.array(PILImage.open(img_path).convert("RGB"))   # (H, W, 3) uint8
        img_t    = apply_cls_transform(self.transform, img_np)
        label    = s.get("class_label_index", 0)
        return {
            "input": img_t,
            "label": torch.tensor(label, dtype=torch.long),
            "organ": s["organ"],
        }


class CEUSClsDataset(Dataset):
    def __init__(self, samples, train_root, val_root, n_frames=16):
        self.samples    = samples
        self.train_root = Path(train_root)
        self.val_root   = Path(val_root) if val_root else None
        self.n_frames   = n_frames

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s  = self.samples[idx]
        pr = get_partition_root(self.train_root, self.val_root, s["data_partition_group"])
        npy_path = pr / s["input_path_relative"]

        video = np.load(npy_path)                               # (64, 256, 512, 3) uint8
        video_t = torch.tensor(video, dtype=torch.float32)
        video_t = video_t.permute(0, 3, 1, 2) / 255.0         # (64, 3, 256, 512)

        label = s.get("class_label_index", 0)
        return {
            "input": video_t,
            "label": torch.tensor(label, dtype=torch.long),
            "organ": s["organ"],
        }


# ─────────────────────────────────────────────────────────────
#  Train/Val split (stratified by organ AND class)
# ─────────────────────────────────────────────────────────────
random.seed(CFG["seed"])
strata = defaultdict(list)
for i, s in enumerate(task_samples):
    key = (s["organ"], s.get("class_label_index", 0))
    strata[key].append(i)

train_indices, val_indices = [], []
for key, indices in strata.items():
    random.shuffle(indices)
    n_val = max(1, int(len(indices) * CFG["val_split"]))
    val_indices.extend(indices[:n_val])
    train_indices.extend(indices[n_val:])

random.shuffle(train_indices)
train_s = [task_samples[i] for i in train_indices]
val_s   = [task_samples[i] for i in val_indices]
print(f"\nTrain: {len(train_s)}  |  Val: {len(val_s)}")

# ─────────────────────────────────────────────────────────────
#  Build datasets
# ─────────────────────────────────────────────────────────────
if TASK == "image_cls":
    train_tf = get_cls_transforms(CFG, mode="train")
    val_tf   = get_cls_transforms(CFG, mode="val")
    train_ds = ImageClsDataset(train_s, TRAIN_PATH, VAL_PATH, train_tf)
    val_ds   = ImageClsDataset(val_s,   TRAIN_PATH, VAL_PATH, val_tf)
else:  # ceus_cls
    train_ds = CEUSClsDataset(train_s, TRAIN_PATH, VAL_PATH, n_frames=CFG["ceus_n_frames"])
    val_ds   = CEUSClsDataset(val_s,   TRAIN_PATH, VAL_PATH, n_frames=CFG["ceus_n_frames"])

# ── WeightedRandomSampler for balanced batches ─────────────────
# Each sample gets weight proportional to 1 / class_count_in_organ
organ_class_counts = defaultdict(lambda: defaultdict(int))
for s in train_s:
    organ_class_counts[s["organ"]][s.get("class_label_index", 0)] += 1

sample_weights = []
for s in train_s:
    organ = s["organ"]
    label = s.get("class_label_index", 0)
    organ_total = sum(organ_class_counts[organ].values())
    w = organ_total / (2.0 * max(organ_class_counts[organ][label], 1))
    sample_weights.append(w)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

# ── Data Loaders ───────────────────────────────────────────────
nw = CFG["num_workers"] if TASK == "image_cls" else 2
train_loader = DataLoader(
    train_ds, batch_size=CFG["batch_size"],
    sampler=sampler,
    num_workers=nw, pin_memory=CFG["pin_memory"],
)
val_loader = DataLoader(
    val_ds, batch_size=CFG["batch_size"] * 2, shuffle=False,
    num_workers=nw, pin_memory=CFG["pin_memory"],
)
print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")
print(f"Effective batch size: {CFG['batch_size'] * CFG['grad_accum_steps']}")

# ─────────────────────────────────────────────────────────────
#  Build Model + Loss
# ─────────────────────────────────────────────────────────────
if TASK == "image_cls":
    model = build_cls_model(CFG)
else:
    model = build_ceus_cls_model(CFG)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(DEVICE)
print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")

# Combined class weights across all organs in this task
# Use macro-average of per-organ weights
all_organs = list(organ_class_counts.keys())
total_counts = [0, 0]
for s in train_s:
    total_counts[s.get("class_label_index", 0)] += 1
grand_total = sum(total_counts)
w0 = grand_total / (2.0 * max(total_counts[0], 1))
w1 = grand_total / (2.0 * max(total_counts[1], 1))
global_weights = torch.tensor([w0, w1], dtype=torch.float32, device=DEVICE)
print(f"Global class weights: [{w0:.3f}, {w1:.3f}]  (class counts: {total_counts})")

criterion = FocalCELoss(
    gamma=CFG["cls_focal_gamma"],
    class_weights=global_weights,
    label_smoothing=CFG["label_smoothing"],
).to(DEVICE)

# ─────────────────────────────────────────────────────────────
#  Optimizer / Scheduler / EMA
# ─────────────────────────────────────────────────────────────
raw_model = model.module if hasattr(model, "module") else model
optimizer  = build_optimizer(raw_model, CFG)
scheduler  = build_scheduler(optimizer, CFG)
ema        = EMA(raw_model, decay=CFG["ema_decay"])

# ─────────────────────────────────────────────────────────────
#  Custom training loop (with Mixup)
# ─────────────────────────────────────────────────────────────
from torch.cuda.amp import GradScaler, autocast

scaler     = GradScaler(enabled=CFG["use_amp"])
ckpt_dir   = CFG["ckpt_dir"]
epochs     = CFG["epochs"]
accum_steps= CFG["grad_accum_steps"]
log_steps  = CFG["log_steps"]
mixup_a    = CFG.get("mixup_alpha", 0.2)
best_acc   = 0.0
history    = []
start_epoch= 1

# Resume
resume_path = CFG.get("resume_from")
if resume_path and os.path.exists(resume_path):
    ckpt = torch.load(resume_path, map_location=DEVICE)
    raw_model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    best_acc    = ckpt.get("best_metric", 0.0)
    history     = ckpt.get("history", [])
    if "ema_state_dict" in ckpt:
        ema.load_state_dict(ckpt["ema_state_dict"])
    print(f"✅ Resumed from epoch {start_epoch-1} | best_acc={best_acc:.4f}")

print(f"\n{'='*65}")
print(f"  Training: {CKPT_PREFIX}  |  epochs {start_epoch}→{epochs}")
print(f"  AMP={CFG['use_amp']}  |  GradAccum={accum_steps}  |  "
      f"EffBatch={CFG['batch_size']*accum_steps}")
print(f"  Mixup alpha={mixup_a}  |  WarmupEpochs={CFG['warmup_epochs']}")
print(f"  VRAM at start: {fmt_vram()}")
print(f"{'='*65}\n")

for epoch in range(start_epoch, epochs + 1):
    t0 = time.time()

    # Warmup LR
    warmup = CFG.get("warmup_epochs", 5)
    if epoch <= warmup:
        factor = epoch / max(warmup, 1)
        base_lr = CFG["lr"]
        enc_lr  = CFG["encoder_lr"]
        for i, pg in enumerate(optimizer.param_groups):
            pg["lr"] = (enc_lr if i == 0 and len(optimizer.param_groups) > 1 else base_lr) * factor

    # ── Train epoch ────────────────────────────────────────────
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    n_total_steps = len(train_loader)

    for step, batch in enumerate(train_loader, 1):
        t_step = time.time()
        imgs   = batch["input"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        bs     = imgs.size(0)

        # Mixup (only for image_cls — not for CEUS videos which are large)
        use_mixup = TASK == "image_cls" and mixup_a > 0 and random.random() < 0.5
        if use_mixup:
            imgs, (la, lb, lam) = mixup_batch(imgs, labels, alpha=mixup_a)

        with autocast(enabled=CFG["use_amp"]):
            logits = model(imgs)
            if use_mixup:
                loss = mixup_criterion(criterion, logits, la, lb, lam) / accum_steps
            else:
                loss = criterion(logits, labels) / accum_steps

        scaler.scale(loss).backward()

        if step % accum_steps == 0 or step == n_total_steps:
            if CFG["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(raw_model)

        with torch.no_grad():
            loss_sum += loss.item() * accum_steps * bs
            preds     = logits.argmax(dim=1)
            correct  += (preds == labels).sum().item()
            total    += bs

        # Per-step logging
        if step % log_steps == 0 or step == n_total_steps:
            t_elapsed = time.time() - t0
            eta_epoch = (n_total_steps - step) * (t_elapsed / step)
            eta_total = (epochs - epoch) * (t_elapsed / step * n_total_steps)
            lr_str = "/".join(f"{pg['lr']:.2e}" for pg in optimizer.param_groups)
            print(
                f"    [E{epoch:02d} S{step:04d}/{n_total_steps}]  "
                f"loss={loss_sum/max(total,1):.4f}  acc={correct/max(total,1):.4f}  "
                f"|  LR: {lr_str}  "
                f"|  ETA_epoch: {fmt_time(eta_epoch)}  "
                f"|  ETA_total: {fmt_time(eta_total)}  "
                f"|  VRAM: {fmt_vram()}"
            )

    scheduler.step(epoch)
    train_loss = loss_sum / max(total, 1)
    train_acc  = correct / max(total, 1)

    # ── Val epoch (using EMA model) ─────────────────────────────
    ema.shadow.eval()
    val_loss_sum, val_correct, val_total = 0.0, 0, 0
    organ_correct = defaultdict(int)
    organ_total   = defaultdict(int)

    with torch.no_grad():
        for batch in val_loader:
            imgs   = batch["input"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            organs = batch["organ"]

            with autocast(enabled=CFG["use_amp"]):
                logits = ema.shadow(imgs)
                loss_v = criterion(logits, labels)

            val_loss_sum += loss_v.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct_mask = (preds == labels)
            val_correct += correct_mask.sum().item()
            val_total   += imgs.size(0)
            for i, org in enumerate(organs):
                organ_total[org]   += 1
                organ_correct[org] += correct_mask[i].item()

    val_acc  = val_correct / max(val_total, 1)
    val_loss = val_loss_sum / max(val_total, 1)
    t_epoch  = time.time() - t0
    eta_remain = (epochs - epoch) * t_epoch
    is_best  = val_acc > best_acc
    if is_best:
        best_acc = val_acc

    # Save checkpoints
    lr_str = "/".join(f"{pg['lr']:.2e}" for pg in optimizer.param_groups)
    ckpt_data = {
        "epoch": epoch, "best_metric": best_acc,
        "model_state_dict":     raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict":    scaler.state_dict(),
        "ema_state_dict":       ema.state_dict(),
        "history":              history,
    }
    torch.save(ckpt_data, f"{ckpt_dir}/{CKPT_PREFIX}_latest.pth")
    if is_best:
        torch.save(ckpt_data, f"{ckpt_dir}/{CKPT_PREFIX}_best.pth")

    # Epoch summary
    history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                    "val_loss": val_loss, "val_acc": val_acc})

    print(f"\n{'─'*65}")
    print(f"  [Epoch {epoch:02d}/{epochs}]  time={fmt_time(t_epoch)}  "
          f"ETA_total={fmt_time(eta_remain)}  LR={lr_str}")
    print(f"  Train  —  loss={train_loss:.4f}  acc={train_acc:.4f}")
    print(f"  Val    —  loss={val_loss:.4f}  acc={val_acc:.4f}  "
          f"{'← 💾 BEST' if is_best else f'(best={best_acc:.4f})'}")
    print(f"  VRAM: {fmt_vram()}")
    print("  Per-organ val accuracy:")
    for org in sorted(organ_total):
        a = organ_correct[org] / max(organ_total[org], 1)
        print(f"    {org:20s}: {a:.4f}  ({organ_correct[org]}/{organ_total[org]})")
    print(f"{'─'*65}")

# ─────────────────────────────────────────────────────────────
#  Save history JSON
# ─────────────────────────────────────────────────────────────
import json as _json
with open(f"{ckpt_dir}/{CKPT_PREFIX}_history.json", "w") as f:
    _json.dump(history, f, indent=2)

print(f"\n{'='*65}")
print(f"✅ {TASK} Training Complete!")
print(f"   Best val accuracy : {best_acc:.4f}")
print(f"   Checkpoints       : {ckpt_dir}/{CKPT_PREFIX}_best.pth")
print(f"{'='*65}")

# ─────────────────────────────────────────────────────────────
#  Cleanup
# ─────────────────────────────────────────────────────────────
del model, optimizer, scheduler, ema, train_loader, val_loader, train_ds, val_ds
import gc; gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

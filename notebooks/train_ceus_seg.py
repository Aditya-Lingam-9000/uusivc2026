"""
notebooks/train_ceus_seg.py
Phase A — CEUS Video Segmentation Training (ceus_seg task).

Extracts middle frame (index 7) from each 15-frame CEUS video,
runs standard U-Net segmentation on it.

HOW TO RUN:
    TRAIN_PATH = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN"
    VAL_PATH   = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL"
    exec(open('/kaggle/working/repo/notebooks/train_ceus_seg.py').read())

DATA:  ceus_seg videos — (15, 256, 512, 3) uint8 .npy + (256, 512) mask .npz
MODEL: SegModel (U-Net with ResNet-50 encoder)
"""

import sys, os, json, random, time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ── Force reload ──────────────────────────────────────────────
for mod in list(sys.modules.keys()):
    if mod.startswith("src"):
        del sys.modules[mod]

# ── Config ────────────────────────────────────────────────────
TRAIN   = globals().get("TRAIN_PATH", "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN")
VAL_DIR = globals().get("VAL_PATH",   "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL")

BATCH_SIZE   = 32
EPOCHS       = 20
LR           = 1e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT    = 0.15
SEED         = 42
CKPT_DIR     = "/kaggle/working/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")

torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

# ── Imports ───────────────────────────────────────────────────
from src.dataset import get_partition_root
from src.model import build_model
from src.losses import build_seg_loss

print("✅ All imports OK")

# ── Load samples ──────────────────────────────────────────────
PRIVATE_GT = f"{TRAIN}/dataset_json_fingerprints_v4/private_train_ground_truth.json"
PUBLIC_GT  = f"{TRAIN}/dataset_json_fingerprints_v4/public_all_ground_truth.json"

all_samples = []
for jp in [PRIVATE_GT, PUBLIC_GT]:
    if os.path.exists(jp):
        with open(jp) as f:
            all_samples.extend(json.load(f))

ceus_seg_samples = [s for s in all_samples if s["task"] == "ceus_seg"]
print(f"Total ceus_seg samples: {len(ceus_seg_samples)}")

organ_counts = defaultdict(int)
for s in ceus_seg_samples:
    organ_counts[s["organ"]] += 1
print("Per-organ counts:")
for organ, cnt in sorted(organ_counts.items()):
    print(f"  {organ:20s}: {cnt}")

# ── Custom Dataset ────────────────────────────────────────────
NORMALIZE = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

class CEUSSegDataset(Dataset):
    def __init__(self, samples, train_root, val_root, augment=False):
        self.samples = samples
        self.train_root = Path(train_root)
        self.val_root = Path(val_root) if val_root else None
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        part_root = get_partition_root(self.train_root, self.val_root,
                                       s["data_partition_group"])

        # Load video and extract middle frame
        npy_path = part_root / s["input_path_relative"]
        video = np.load(npy_path)              # (15, 256, 512, 3) uint8
        mid_frame = video[7]                   # (256, 512, 3)

        # Convert to tensor: (3, 256, 512) float [0, 1]
        frame_t = torch.tensor(mid_frame, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Load mask
        ann_path = part_root / s["annotation_path_relative"]
        npz = np.load(ann_path)
        mask = npz["mask"].astype(np.float32) / 255.0   # (256, 512) → {0.0, 1.0}
        mask_t = torch.tensor(mask).unsqueeze(0)         # (1, 256, 512)

        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                frame_t = torch.flip(frame_t, dims=[2])  # horizontal flip
                mask_t  = torch.flip(mask_t,  dims=[2])
            if random.random() > 0.7:
                frame_t = torch.flip(frame_t, dims=[1])  # vertical flip
                mask_t  = torch.flip(mask_t,  dims=[1])

        # Normalize with ImageNet stats
        frame_t = NORMALIZE(frame_t)

        return {
            "input": frame_t,     # (3, 256, 512)
            "mask":  mask_t,      # (1, 256, 512)
            "organ": s["organ"],
        }

# ── Train/Val split ───────────────────────────────────────────
random.seed(SEED)
organ_to_indices = defaultdict(list)
for i, s in enumerate(ceus_seg_samples):
    organ_to_indices[s["organ"]].append(i)

train_indices, val_indices = [], []
for organ, indices in organ_to_indices.items():
    random.shuffle(indices)
    n_val = max(1, int(len(indices) * VAL_SPLIT))
    val_indices.extend(indices[:n_val])
    train_indices.extend(indices[n_val:])

train_samples = [ceus_seg_samples[i] for i in train_indices]
val_samples   = [ceus_seg_samples[i] for i in val_indices]
print(f"Train: {len(train_samples)}  Val: {len(val_samples)}")

train_ds = CEUSSegDataset(train_samples, TRAIN, VAL_DIR, augment=True)
val_ds   = CEUSSegDataset(val_samples,   TRAIN, VAL_DIR, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

# ── Model ─────────────────────────────────────────────────────
model = build_model("ceus_seg", pretrained=True)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(DEVICE)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# ── Loss ──────────────────────────────────────────────────────
criterion = build_seg_loss(pos_weight=2.0).to(DEVICE)

# ── Optimizer ─────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Dice metric ───────────────────────────────────────────────
def dice_score(pred_logits, targets, threshold=0.5):
    probs = torch.sigmoid(pred_logits)
    preds_bin = (probs > threshold).float()
    intersection = (preds_bin * targets).sum()
    return (2.0 * intersection + 1e-6) / (preds_bin.sum() + targets.sum() + 1e-6)

# ── Training loop ─────────────────────────────────────────────
best_val_dice = 0.0
history = []

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # ── Train ──
    model.train()
    train_loss_sum, train_dice_sum, n_train = 0.0, 0.0, 0

    for batch in train_loader:
        imgs  = batch["input"].to(DEVICE)     # (B, 3, 256, 512)
        masks = batch["mask"].to(DEVICE)      # (B, 1, 256, 512)

        optimizer.zero_grad()
        logits = model(imgs)                  # (B, 1, H, W)

        # Resize logits to match mask if needed
        if logits.shape[2:] != masks.shape[2:]:
            logits = F.interpolate(logits, size=masks.shape[2:],
                                   mode="bilinear", align_corners=False)

        loss = criterion(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = imgs.size(0)
        train_loss_sum += loss.item() * bs
        train_dice_sum += dice_score(logits, masks).item() * bs
        n_train += bs

    scheduler.step()

    # ── Validate ──
    model.eval()
    val_loss_sum, val_dice_sum, n_val = 0.0, 0.0, 0

    with torch.no_grad():
        for batch in val_loader:
            imgs  = batch["input"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)

            logits = model(imgs)
            if logits.shape[2:] != masks.shape[2:]:
                logits = F.interpolate(logits, size=masks.shape[2:],
                                       mode="bilinear", align_corners=False)

            loss = criterion(logits, masks)
            bs = imgs.size(0)
            val_loss_sum += loss.item() * bs
            val_dice_sum += dice_score(logits, masks).item() * bs
            n_val += bs

    train_loss = train_loss_sum / max(n_train, 1)
    train_dice = train_dice_sum / max(n_train, 1)
    val_loss   = val_loss_sum   / max(n_val, 1)
    val_dice   = val_dice_sum   / max(n_val, 1)
    t_elapsed  = time.time() - t0

    print(f"\n[Epoch {epoch:02d}/{EPOCHS}]  time={t_elapsed:.0f}s  LR={scheduler.get_last_lr()[0]:.2e}")
    print(f"  Train — loss={train_loss:.4f}  dice={train_dice:.4f}")
    print(f"  Val   — loss={val_loss:.4f}  dice={val_dice:.4f}")

    history.append({
        "epoch": epoch, "train_loss": train_loss, "train_dice": train_dice,
        "val_loss": val_loss, "val_dice": val_dice,
    })

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        ckpt_path = f"{CKPT_DIR}/ceus_seg_best.pth"
        m = model.module if hasattr(model, "module") else model
        torch.save({
            "epoch": epoch, "val_dice": val_dice,
            "model_state_dict": m.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, ckpt_path)
        print(f"  💾 Saved best (val_dice={val_dice:.4f}) → {ckpt_path}")

with open(f"{CKPT_DIR}/ceus_seg_history.json", "w") as f:
    json.dump(history, f, indent=2)

print(f"\n{'='*50}")
print(f"✅ CEUS SEG TRAINING COMPLETE")
print(f"   Best val dice : {best_val_dice:.4f}")
print(f"   Checkpoint    : {CKPT_DIR}/ceus_seg_best.pth")
print(f"{'='*50}")

# ── Clean up variables to free memory for subsequent runs ─────
del model
del optimizer
del train_loader
del val_loader
del train_ds
del val_ds
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


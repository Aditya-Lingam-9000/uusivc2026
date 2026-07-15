"""
notebooks/train_image_seg.py
Phase 2 — Step 2: Image Segmentation Training (image_seg task).

Run on Kaggle T4x2 AFTER image_cls training.

HOW TO RUN:
    TRAIN_PATH = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN"
    VAL_PATH   = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL"
    exec(open('/kaggle/working/repo/notebooks/train_image_seg.py').read())

EXPECTED TIME: ~40-60 min for 15 epochs on T4x2
EXPECTED DICE: 0.65-0.75 after 15 epochs
"""

import sys, os, json, random, time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image

# ── Force reload modules ──────────────────────────────────────
for mod in list(sys.modules.keys()):
    if mod.startswith("src"):
        del sys.modules[mod]

# ── Config ────────────────────────────────────────────────────
TRAIN   = globals().get("TRAIN_PATH", "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN")
VAL_DIR = globals().get("VAL_PATH",   "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL")

IMAGE_SIZE   = 512
BATCH_SIZE   = 16        # smaller than cls because masks add memory
EPOCHS       = 15
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
from src.dataset import UUSIVCDataset, get_partition_root, load_mask_png
from src.model import build_model
from src.losses import build_seg_loss
import torchvision.transforms as T
import torchvision.transforms.functional as TF

print("✅ All imports OK")

# ── Joint image+mask transform ────────────────────────────────
class JointTransform:
    def __init__(self, size=IMAGE_SIZE, augment=True):
        self.size = size
        self.augment = augment
        self.normalize = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

    def __call__(self, img: Image.Image, mask: torch.Tensor):
        # Resize image
        img = TF.resize(img, [self.size, self.size])

        # Resize mask to same size (nearest neighbour to preserve binary values)
        mask = F.interpolate(mask.unsqueeze(0), size=(self.size, self.size),
                             mode="nearest").squeeze(0)   # (1, H, W)

        if self.augment:
            if random.random() > 0.5:
                img  = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.3:
                img  = TF.vflip(img)
                mask = TF.vflip(mask)
            img = T.ColorJitter(brightness=0.2, contrast=0.2)(img)

        img  = TF.to_tensor(img)
        img  = self.normalize(img)
        return img, mask                                   # (3,H,W), (1,H,W)

# ── Load dataset ──────────────────────────────────────────────
PRIVATE_GT = f"{TRAIN}/dataset_json_fingerprints_v4/private_train_ground_truth.json"
PUBLIC_GT  = f"{TRAIN}/dataset_json_fingerprints_v4/public_all_ground_truth.json"

full_ds = UUSIVCDataset(
    json_paths=[PRIVATE_GT, PUBLIC_GT],
    data_root=TRAIN, val_root=VAL_DIR,
    transform=None, task_filter=["image_seg"],
)
print(f"Total image_seg samples: {len(full_ds)}")

organ_counts = defaultdict(int)
for s in full_ds.samples:
    organ_counts[s["organ"]] += 1
print("Per-organ counts:")
for organ, cnt in sorted(organ_counts.items()):
    print(f"  {organ:20s}: {cnt}")

# ── Stratified split ──────────────────────────────────────────
organ_to_idx = defaultdict(list)
for i, s in enumerate(full_ds.samples):
    organ_to_idx[s["organ"]].append(i)

train_idx, val_idx = [], []
random.seed(SEED)
for organ, idxs in organ_to_idx.items():
    random.shuffle(idxs)
    n_val = max(1, int(len(idxs) * VAL_SPLIT))
    val_idx.extend(idxs[:n_val])
    train_idx.extend(idxs[n_val:])

print(f"Train: {len(train_idx)}  Val: {len(val_idx)}")

# ── IndexedDataset with joint transform ───────────────────────
class SegIndexedDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, indices, transform: JointTransform):
        self.base = base_ds
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        s    = self.base.samples[self.indices[i]]
        proot = get_partition_root(self.base.data_root, self.base.val_root,
                                    s["data_partition_group"])

        img_path  = proot / s["img_path_relative"]
        mask_path = proot / s["mask_path_relative"] if s.get("mask_path_relative") else None

        img  = Image.open(img_path).convert("RGB")
        mask = load_mask_png(mask_path) if mask_path else torch.zeros(1, 1, 1)

        img, mask = self.transform(img, mask)
        return {"input": img, "mask": mask, "organ": s["organ"]}

train_ds = SegIndexedDataset(full_ds, train_idx, JointTransform(augment=True))
val_ds   = SegIndexedDataset(full_ds, val_idx,   JointTransform(augment=False))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

# ── Model ─────────────────────────────────────────────────────
model = build_model("image_seg", pretrained=True)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(DEVICE)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# ── Loss ──────────────────────────────────────────────────────
criterion = build_seg_loss(pos_weight=2.0)   # upweight foreground

# ── Optimizer + Scheduler ─────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Dice metric ───────────────────────────────────────────────
def compute_dice(logits, targets, thresh=0.5):
    preds = (torch.sigmoid(logits) > thresh).float()
    inter = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice  = (2 * inter + 1e-6) / (union + 1e-6)
    return dice.mean().item()

# ── Training loop ─────────────────────────────────────────────
best_dice = 0.0
history   = []

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # ── Train ──
    model.train()
    t_loss, t_dice = 0.0, 0.0

    for batch in train_loader:
        imgs  = batch["input"].to(DEVICE)
        masks = batch["mask"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(imgs)                               # (B,1,H,W)
        # Resize logits to mask size if needed
        if logits.shape != masks.shape:
            logits = F.interpolate(logits, size=masks.shape[2:],
                                   mode="bilinear", align_corners=False)
        loss = criterion(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        t_loss += loss.item() * imgs.size(0)
        t_dice += compute_dice(logits.detach(), masks) * imgs.size(0)

    scheduler.step()
    n_train = len(train_ds)

    # ── Validate ──
    model.eval()
    v_loss, v_dice = 0.0, 0.0
    organ_dice = defaultdict(list)

    with torch.no_grad():
        for batch in val_loader:
            imgs   = batch["input"].to(DEVICE)
            masks  = batch["mask"].to(DEVICE)
            organs = batch["organ"]

            logits = model(imgs)
            if logits.shape != masks.shape:
                logits = F.interpolate(logits, size=masks.shape[2:],
                                       mode="bilinear", align_corners=False)
            loss  = criterion(logits, masks)
            dice  = compute_dice(logits, masks)

            v_loss += loss.item() * imgs.size(0)
            v_dice += dice * imgs.size(0)

            # Per-sample dice for organ breakdown
            preds = (torch.sigmoid(logits) > 0.5).float()
            for j, organ in enumerate(organs):
                inter = (preds[j] * masks[j]).sum()
                union = preds[j].sum() + masks[j].sum()
                d = (2*inter + 1e-6) / (union + 1e-6)
                organ_dice[organ].append(d.item())

    n_val = len(val_ds)
    t_elapsed = time.time() - t0

    print(f"\n[Epoch {epoch:02d}/{EPOCHS}]  time={t_elapsed:.0f}s  LR={scheduler.get_last_lr()[0]:.2e}")
    print(f"  Train — loss={t_loss/n_train:.4f}  dice={t_dice/n_train:.4f}")
    print(f"  Val   — loss={v_loss/n_val:.4f}  dice={v_dice/n_val:.4f}")
    print("  Per-organ val Dice:")
    for organ in sorted(organ_dice):
        d = np.mean(organ_dice[organ])
        print(f"    {organ:20s}: {d:.4f}  (n={len(organ_dice[organ])})")

    val_dice_epoch = v_dice / n_val
    history.append({"epoch": epoch, "train_loss": t_loss/n_train,
                    "train_dice": t_dice/n_train, "val_dice": val_dice_epoch})

    if val_dice_epoch > best_dice:
        best_dice = val_dice_epoch
        m = model.module if hasattr(model, "module") else model
        torch.save({
            "epoch": epoch, "val_dice": val_dice_epoch,
            "model_state_dict": m.state_dict(),
        }, f"{CKPT_DIR}/image_seg_best.pth")
        print(f"  💾 Saved best model (dice={val_dice_epoch:.4f})")

with open(f"{CKPT_DIR}/image_seg_history.json", "w") as f:
    json.dump(history, f, indent=2)

print(f"\n{'='*50}")
print(f"✅ TRAINING COMPLETE")
print(f"   Best val Dice  : {best_dice:.4f}")
print(f"   Checkpoint     : {CKPT_DIR}/image_seg_best.pth")

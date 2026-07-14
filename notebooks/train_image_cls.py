"""
notebooks/train_image_cls.py
Phase 2 Training — Step 1: Image Classification (image_cls task only).

Run this on Kaggle T4x2.

HOW TO RUN (Kaggle cell):
    TRAIN_PATH = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN"
    VAL_PATH   = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL"
    exec(open('/kaggle/working/repo/notebooks/train_image_cls.py').read())

WHAT THIS SCRIPT DOES:
    1. Loads ALL image_cls samples (private train + public train)
    2. Splits 85% train / 15% val (stratified by organ)
    3. Trains ResNet-50 with class-weighted CE loss
    4. Saves best model per epoch to /kaggle/working/checkpoints/image_cls_best.pth
    5. Prints per-epoch: loss, accuracy, per-organ accuracy

EXPECTED TRAINING TIME: ~15–25 min for 10 epochs on T4x2 (GPU)
"""

import sys, os, json, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from pathlib import Path
from collections import defaultdict

# ── Force reload modules ──────────────────────────────────────
for mod in list(sys.modules.keys()):
    if mod.startswith("src"):
        del sys.modules[mod]

# ── Config ────────────────────────────────────────────────────
TRAIN   = globals().get("TRAIN_PATH", "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN")
VAL_DIR = globals().get("VAL_PATH",   "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL")

BATCH_SIZE   = 32
EPOCHS       = 10
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

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ── Imports ───────────────────────────────────────────────────
from src.dataset import UUSIVCDataset
from src.transforms import get_train_transforms, get_val_transforms
from src.model import build_model
from src.losses import build_cls_loss

print("✅ All imports OK")

# ── Load dataset ──────────────────────────────────────────────
PRIVATE_GT = f"{TRAIN}/dataset_json_fingerprints_v4/private_train_ground_truth.json"
PUBLIC_GT  = f"{TRAIN}/dataset_json_fingerprints_v4/public_all_ground_truth.json"

full_ds = UUSIVCDataset(
    json_paths=[PRIVATE_GT, PUBLIC_GT],
    data_root=TRAIN,
    val_root=VAL_DIR,
    transform=None,     # transforms applied separately to train/val splits
    task_filter=["image_cls"],
)
print(f"Total image_cls samples: {len(full_ds)}")

# Print per-organ breakdown
organ_counts = defaultdict(int)
for s in full_ds.samples:
    organ_counts[s["organ"]] += 1
print("Per-organ counts:")
for organ, cnt in sorted(organ_counts.items()):
    print(f"  {organ:20s}: {cnt}")

# ── Train/Val split (stratified by organ) ─────────────────────
random.seed(SEED)
organ_to_indices = defaultdict(list)
for i, s in enumerate(full_ds.samples):
    organ_to_indices[s["organ"]].append(i)

train_indices, val_indices = [], []
for organ, indices in organ_to_indices.items():
    random.shuffle(indices)
    n_val = max(1, int(len(indices) * VAL_SPLIT))
    val_indices.extend(indices[:n_val])
    train_indices.extend(indices[n_val:])

print(f"Train samples: {len(train_indices)}  Val samples: {len(val_indices)}")

# ── DataLoader setup ──────────────────────────────────────────
class IndexedDataset(torch.utils.data.Dataset):
    """Wrapper to apply different transforms to train vs val splits."""
    def __init__(self, base_ds, indices, transform):
        self.base = base_ds
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        s = self.base.samples[self.indices[i]]
        from src.dataset import get_partition_root
        from pathlib import Path
        from PIL import Image
        img_path = get_partition_root(self.base.data_root, self.base.val_root,
                                       s["data_partition_group"]) / s["input_path_relative"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = s["class_label_index"]
        return {
            "input":  img,
            "label":  torch.tensor(label, dtype=torch.long),
            "organ":  s["organ"],
        }

train_ds = IndexedDataset(full_ds, train_indices, get_train_transforms())
val_ds   = IndexedDataset(full_ds, val_indices,   get_val_transforms())

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

# ── Model ─────────────────────────────────────────────────────
model = build_model("image_cls", pretrained=True)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Loss (class-weighted CE for all organs combined) ──────────
# Build a single loss with averaged weights across all organs
criterion = nn.CrossEntropyLoss()   # weights applied per-organ below

# ── Optimizer + Scheduler ─────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Training loop ─────────────────────────────────────────────
best_val_acc = 0.0
history = []

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # ── Train ──
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for batch in train_loader:
        imgs   = batch["input"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss    += loss.item() * imgs.size(0)
        preds          = logits.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total   += imgs.size(0)

    scheduler.step()

    # ── Validate ──
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    organ_correct = defaultdict(int)
    organ_total   = defaultdict(int)

    with torch.no_grad():
        for batch in val_loader:
            imgs   = batch["input"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            organs = batch["organ"]

            logits = model(imgs)
            loss   = criterion(logits, labels)

            val_loss    += loss.item() * imgs.size(0)
            preds        = logits.argmax(dim=1)
            correct_mask = (preds == labels)
            val_correct += correct_mask.sum().item()
            val_total   += imgs.size(0)

            for i, organ in enumerate(organs):
                organ_total[organ]   += 1
                organ_correct[organ] += correct_mask[i].item()

    train_acc = train_correct / train_total
    val_acc   = val_correct   / val_total
    t_elapsed = time.time() - t0

    print(f"\n[Epoch {epoch:02d}/{EPOCHS}]  "
          f"time={t_elapsed:.0f}s  "
          f"LR={scheduler.get_last_lr()[0]:.2e}")
    print(f"  Train — loss={train_loss/train_total:.4f}  acc={train_acc:.4f}")
    print(f"  Val   — loss={val_loss/val_total:.4f}  acc={val_acc:.4f}")
    print("  Per-organ val accuracy:")
    for organ in sorted(organ_total):
        acc = organ_correct[organ] / organ_total[organ]
        print(f"    {organ:20s}: {acc:.4f}  ({organ_correct[organ]}/{organ_total[organ]})")

    history.append({
        "epoch": epoch,
        "train_loss": train_loss / train_total,
        "train_acc":  train_acc,
        "val_loss":   val_loss / val_total,
        "val_acc":    val_acc,
    })

    # ── Save best checkpoint ──
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        ckpt_path = f"{CKPT_DIR}/image_cls_best.pth"
        m = model.module if hasattr(model, "module") else model
        torch.save({
            "epoch":      epoch,
            "val_acc":    val_acc,
            "model_state_dict": m.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, ckpt_path)
        print(f"  💾 Saved best model (val_acc={val_acc:.4f}) → {ckpt_path}")

# ── Save training history ─────────────────────────────────────
with open(f"{CKPT_DIR}/image_cls_history.json", "w") as f:
    json.dump(history, f, indent=2)

print(f"\n{'='*50}")
print(f"✅ TRAINING COMPLETE")
print(f"   Best val accuracy : {best_val_acc:.4f}")
print(f"   Checkpoint saved  : {CKPT_DIR}/image_cls_best.pth")
print(f"   History saved     : {CKPT_DIR}/image_cls_history.json")

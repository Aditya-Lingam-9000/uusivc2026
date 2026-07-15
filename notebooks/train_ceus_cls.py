"""
notebooks/train_ceus_cls.py
Phase A — CEUS Video Classification Training (ceus_cls task).

Run on Kaggle T4x2.

HOW TO RUN:
    TRAIN_PATH = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN"
    VAL_PATH   = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL"
    exec(open('/kaggle/working/repo/notebooks/train_ceus_cls.py').read())

DATA:  ceus_cls videos — (64, 256, 512, 3) uint8 .npy files
MODEL: CEUSCLSModel — ResNet-50 + temporal avg of 8 sampled frames
"""

import sys, os, json, random, time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ── Force reload ──────────────────────────────────────────────
for mod in list(sys.modules.keys()):
    if mod.startswith("src"):
        del sys.modules[mod]

# ── Config ────────────────────────────────────────────────────
TRAIN   = globals().get("TRAIN_PATH", "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN")
VAL_DIR = globals().get("VAL_PATH",   "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL")

BATCH_SIZE   = 4         # videos are large (~25 MB each)
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
from src.dataset import UUSIVCDataset, get_partition_root
from src.model import build_model

print("✅ All imports OK")

# ── Load dataset ──────────────────────────────────────────────
PRIVATE_GT = f"{TRAIN}/dataset_json_fingerprints_v4/private_train_ground_truth.json"
PUBLIC_GT  = f"{TRAIN}/dataset_json_fingerprints_v4/public_all_ground_truth.json"

# Load all samples and filter to ceus_cls
all_samples = []
for jp in [PRIVATE_GT, PUBLIC_GT]:
    if os.path.exists(jp):
        with open(jp) as f:
            all_samples.extend(json.load(f))

ceus_cls_samples = [s for s in all_samples if s["task"] == "ceus_cls"]
print(f"Total ceus_cls samples: {len(ceus_cls_samples)}")

# Per-organ breakdown
organ_counts = defaultdict(int)
for s in ceus_cls_samples:
    organ_counts[s["organ"]] += 1
print("Per-organ counts:")
for organ, cnt in sorted(organ_counts.items()):
    print(f"  {organ:20s}: {cnt}")

# ── Custom Dataset ────────────────────────────────────────────
class CEUSClsDataset(Dataset):
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
        npy_path = part_root / s["input_path_relative"]

        # Load video: (64, 256, 512, 3) uint8
        video = np.load(npy_path)
        video = torch.tensor(video, dtype=torch.float32)
        video = video.permute(0, 3, 1, 2) / 255.0   # (64, 3, 256, 512)

        # Simple augmentation: random temporal shift
        if self.augment and random.random() > 0.5:
            shift = random.randint(-3, 3)
            video = torch.roll(video, shifts=shift, dims=0)

        # Random horizontal flip (apply to all frames consistently)
        if self.augment and random.random() > 0.5:
            video = torch.flip(video, dims=[3])  # flip W dimension

        label = s.get("class_label_index", 0)
        return {
            "input": video,                                         # (64, 3, 256, 512)
            "label": torch.tensor(label, dtype=torch.long),
            "organ": s["organ"],
        }

# ── Train/Val split (stratified by organ) ─────────────────────
random.seed(SEED)
organ_to_indices = defaultdict(list)
for i, s in enumerate(ceus_cls_samples):
    organ_to_indices[s["organ"]].append(i)

train_indices, val_indices = [], []
for organ, indices in organ_to_indices.items():
    random.shuffle(indices)
    n_val = max(1, int(len(indices) * VAL_SPLIT))
    val_indices.extend(indices[:n_val])
    train_indices.extend(indices[n_val:])

train_samples = [ceus_cls_samples[i] for i in train_indices]
val_samples   = [ceus_cls_samples[i] for i in val_indices]
print(f"Train: {len(train_samples)}  Val: {len(val_samples)}")

train_ds = CEUSClsDataset(train_samples, TRAIN, VAL_DIR, augment=True)
val_ds   = CEUSClsDataset(val_samples,   TRAIN, VAL_DIR, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

# ── Model ─────────────────────────────────────────────────────
model = build_model("ceus_cls", pretrained=True)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(DEVICE)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# ── Class-weighted loss ───────────────────────────────────────
# Compute overall class weights across all organs
class_counts = [0, 0]
for s in train_samples:
    class_counts[s["class_label_index"]] += 1
total = sum(class_counts)
w0 = total / (2.0 * max(class_counts[0], 1))
w1 = total / (2.0 * max(class_counts[1], 1))
weights = torch.tensor([w0, w1], dtype=torch.float32, device=DEVICE)
print(f"Class weights: {weights.tolist()} (counts: {class_counts})")
criterion = nn.CrossEntropyLoss(weight=weights)

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
        videos = batch["input"].to(DEVICE)     # (B, 64, 3, 256, 512)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(videos)                 # (B, 2)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss    += loss.item() * videos.size(0)
        preds          = logits.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total   += videos.size(0)

    scheduler.step()

    # ── Validate ──
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    organ_correct = defaultdict(int)
    organ_total   = defaultdict(int)

    with torch.no_grad():
        for batch in val_loader:
            videos = batch["input"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            organs = batch["organ"]

            logits = model(videos)
            loss = criterion(logits, labels)

            val_loss    += loss.item() * videos.size(0)
            preds        = logits.argmax(dim=1)
            correct_mask = (preds == labels)
            val_correct += correct_mask.sum().item()
            val_total   += videos.size(0)

            for i, organ in enumerate(organs):
                organ_total[organ]   += 1
                organ_correct[organ] += correct_mask[i].item()

    train_acc = train_correct / max(train_total, 1)
    val_acc   = val_correct   / max(val_total, 1)
    t_elapsed = time.time() - t0

    print(f"\n[Epoch {epoch:02d}/{EPOCHS}]  "
          f"time={t_elapsed:.0f}s  "
          f"LR={scheduler.get_last_lr()[0]:.2e}")
    print(f"  Train — loss={train_loss/max(train_total,1):.4f}  acc={train_acc:.4f}")
    print(f"  Val   — loss={val_loss/max(val_total,1):.4f}  acc={val_acc:.4f}")
    print("  Per-organ val accuracy:")
    for organ in sorted(organ_total):
        acc = organ_correct[organ] / organ_total[organ]
        print(f"    {organ:20s}: {acc:.4f}  ({organ_correct[organ]}/{organ_total[organ]})")

    history.append({
        "epoch": epoch, "train_acc": train_acc, "val_acc": val_acc,
        "train_loss": train_loss / max(train_total, 1),
        "val_loss": val_loss / max(val_total, 1),
    })

    # ── Save best ──
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        ckpt_path = f"{CKPT_DIR}/ceus_cls_best.pth"
        m = model.module if hasattr(model, "module") else model
        torch.save({
            "epoch": epoch, "val_acc": val_acc,
            "model_state_dict": m.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, ckpt_path)
        print(f"  💾 Saved best (val_acc={val_acc:.4f}) → {ckpt_path}")

# ── Save history ──────────────────────────────────────────────
with open(f"{CKPT_DIR}/ceus_cls_history.json", "w") as f:
    json.dump(history, f, indent=2)

print(f"\n{'='*50}")
print(f"✅ CEUS CLS TRAINING COMPLETE")
print(f"   Best val accuracy : {best_val_acc:.4f}")
print(f"   Checkpoint saved  : {CKPT_DIR}/ceus_cls_best.pth")
print(f"{'='*50}")

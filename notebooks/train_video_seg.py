"""
notebooks/train_video_seg.py
Phase A — Cardiac Video Segmentation Training (video_seg task).

Flattens all annotated frames into individual 2D segmentation samples,
trains a frame-wise U-Net on them.

HOW TO RUN:
    TRAIN_PATH = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN"
    VAL_PATH   = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL"
    exec(open('/kaggle/working/repo/notebooks/train_video_seg.py').read())

DATA:  video_seg — (3, T, 256, 256) float32 [0,255] .npy + fnum_mask .npz
       CAMUS: 500 patients x 3 views, ~5 annotated frames each
       CardiacCH: ~69 videos, 3-5 annotated frames each
MODEL: SegModel (U-Net with ResNet-50 encoder) on single frames
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

# ── Force reload ──────────────────────────────────────────────
for mod in list(sys.modules.keys()):
    if mod.startswith("src"):
        del sys.modules[mod]

# ── Config ────────────────────────────────────────────────────
TRAIN   = globals().get("TRAIN_PATH", "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN")
VAL_DIR = globals().get("VAL_PATH",   "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL")

BATCH_SIZE   = 96
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

video_seg_samples = [s for s in all_samples if s["task"] == "video_seg"]
print(f"Total video_seg video samples: {len(video_seg_samples)}")

organ_counts = defaultdict(int)
for s in video_seg_samples:
    ds = s.get("dataset_name", s["organ"])
    organ_counts[ds] += 1
print("Per-dataset counts:")
for ds, cnt in sorted(organ_counts.items()):
    print(f"  {ds:20s}: {cnt}")

# ── Flatten into per-frame samples ────────────────────────────
# Each video has multiple annotated frames. We flatten them into
# individual (frame_image, frame_mask) pairs for training.
print("\nFlattening annotated frames...")
frame_samples = []   # list of (video_sample_dict, frame_index_str)

for s in video_seg_samples:
    part_root = get_partition_root(Path(TRAIN), Path(VAL_DIR) if VAL_DIR else None,
                                   s["data_partition_group"])
    ann_path = part_root / s["annotation_path_relative"]
    if ann_path.exists():
        try:
            npz = np.load(ann_path, allow_pickle=True)
            fnum_mask = npz["fnum_mask"].item()
            for frame_key in fnum_mask.keys():
                frame_samples.append((s, frame_key))
        except Exception as e:
            print(f"  Warning: Could not load {ann_path}: {e}")

print(f"Total annotated frames (flattened): {len(frame_samples)}")

# ── Custom Dataset ────────────────────────────────────────────
NORMALIZE = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

class VideoSegFrameDataset(Dataset):
    """Each item is a single annotated frame from a cardiac video."""

    def __init__(self, frame_list, train_root, val_root, augment=False):
        self.frame_list = frame_list
        self.train_root = Path(train_root)
        self.val_root = Path(val_root) if val_root else None
        self.augment = augment

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, idx):
        s, frame_key = self.frame_list[idx]
        frame_idx = int(frame_key)

        part_root = get_partition_root(self.train_root, self.val_root,
                                       s["data_partition_group"])

        # Load video: (3, T, 256, 256) float32 [0, 255]
        npy_path = part_root / s["input_path_relative"]
        video = np.load(npy_path)    # (3, T, 256, 256)

        # Extract frame from view 0 (first cardiac view)
        # video shape: (3_views, T_frames, 256, 256)
        frame = video[0, frame_idx]  # (256, 256) float32 [0, 255]
        frame = frame / 255.0        # normalize to [0, 1]

        # Convert grayscale to 3-channel for ImageNet pretrained encoder
        frame_3ch = np.stack([frame, frame, frame], axis=0)  # (3, 256, 256)
        frame_t = torch.tensor(frame_3ch, dtype=torch.float32)

        # Load mask for this frame
        ann_path = part_root / s["annotation_path_relative"]
        npz = np.load(ann_path, allow_pickle=True)
        fnum_mask = npz["fnum_mask"].item()
        mask_arr = fnum_mask[frame_key]        # (256, 256) float32 [0, 255]
        mask = (mask_arr / 255.0).clip(0, 1)   # {0.0, 1.0}
        mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # (1, 256, 256)

        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                frame_t = torch.flip(frame_t, dims=[2])  # horizontal flip
                mask_t  = torch.flip(mask_t,  dims=[2])
            if random.random() > 0.7:
                frame_t = torch.flip(frame_t, dims=[1])  # vertical flip
                mask_t  = torch.flip(mask_t,  dims=[1])

        # Apply ImageNet normalization
        frame_t = NORMALIZE(frame_t)

        return {
            "input": frame_t,   # (3, 256, 256)
            "mask":  mask_t,    # (1, 256, 256)
        }

# ── Train/Val split (by video, not by frame) ─────────────────
# Split at the video level so frames from the same video don't leak
random.seed(SEED)
video_ids = list(set(s["sample_id"] for s, _ in frame_samples))
random.shuffle(video_ids)
n_val_vids = max(1, int(len(video_ids) * VAL_SPLIT))
val_video_ids = set(video_ids[:n_val_vids])

train_frames = [(s, fk) for s, fk in frame_samples if s["sample_id"] not in val_video_ids]
val_frames   = [(s, fk) for s, fk in frame_samples if s["sample_id"] in val_video_ids]

print(f"Train frames: {len(train_frames)}  Val frames: {len(val_frames)}")
print(f"Train videos: {len(video_ids) - n_val_vids}  Val videos: {n_val_vids}")

train_ds = VideoSegFrameDataset(train_frames, TRAIN, VAL_DIR, augment=True)
val_ds   = VideoSegFrameDataset(val_frames,   TRAIN, VAL_DIR, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

# ── Model ─────────────────────────────────────────────────────
# Use SegModel directly (VideoSegModel is just a wrapper around SegModel)
model = build_model("image_seg", pretrained=True)
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
        imgs  = batch["input"].to(DEVICE)   # (B, 3, 256, 256)
        masks = batch["mask"].to(DEVICE)    # (B, 1, 256, 256)

        optimizer.zero_grad()
        logits = model(imgs)                # (B, 1, H, W)

        # Resize if encoder output size differs
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
        ckpt_path = f"{CKPT_DIR}/video_seg_best.pth"
        m = model.module if hasattr(model, "module") else model
        torch.save({
            "epoch": epoch, "val_dice": val_dice,
            "model_state_dict": m.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, ckpt_path)
        print(f"  💾 Saved best (val_dice={val_dice:.4f}) → {ckpt_path}")

with open(f"{CKPT_DIR}/video_seg_history.json", "w") as f:
    json.dump(history, f, indent=2)

print(f"\n{'='*50}")
print(f"✅ VIDEO SEG TRAINING COMPLETE")
print(f"   Best val dice : {best_val_dice:.4f}")
print(f"   Checkpoint    : {CKPT_DIR}/video_seg_best.pth")
print(f"{'='*50}")

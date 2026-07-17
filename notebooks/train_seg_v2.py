"""
notebooks/train_seg_v2.py
UUSIVC 2026 — Competition-Grade Segmentation Training (v2)

Trains a single UNet++ model PER seg task:
  - image_seg  (7 organs: Breast, Prostate, Breast_luminal, Fetal_Head, Heart, Kidney, Thyroid)
  - ceus_seg   (4 organs: BreastCEUS, LiverCEUS, ProstateCEUS, ThyroidCEUS)
  - video_seg  (cardiac frames: CardiacCH, CAMUS)

Key improvements over v1:
  ✅ EfficientNet-B5 + UNet++ (replaces ResNet-50 + basic UNet)
  ✅ Compound loss: Dice + Focal + Boundary (directly improves NSD)
  ✅ Albumentations heavy augmentation
  ✅ Mixed Precision (AMP) — 30-50% VRAM savings
  ✅ Gradient accumulation — effective batch size 32
  ✅ EMA model for validation
  ✅ Resume from checkpoint (Kaggle-safe)
  ✅ Detailed per-25-step logging with VRAM + ETA
  ✅ Per-organ metrics tracked

HOW TO RUN on Kaggle:
    !pip install segmentation-models-pytorch albumentations timm --quiet
    !cd /kaggle/working && git clone https://github.com/Aditya-Lingam-9000/uusivc2026.git repo
    import sys; sys.path.insert(0, '/kaggle/working/repo')
    TRAIN_PATH = "/kaggle/input/.../TRAIN"
    VAL_PATH   = "/kaggle/input/.../VAL"

    # Train one task at a time (set TASK below):
    TASK = "image_seg"   # or "ceus_seg" or "video_seg"
    exec(open('/kaggle/working/repo/notebooks/train_seg_v2.py').read())

    # Resume after timeout:
    CFG["resume_from"] = f"/kaggle/working/checkpoints/{TASK}_v2_latest.pth"
    exec(open('/kaggle/working/repo/notebooks/train_seg_v2.py').read())
"""

import sys, os, json, random, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Force src reload ───────────────────────────────────────────
for mod in list(sys.modules.keys()):
    if mod.startswith("src"):
        del sys.modules[mod]

# ── Config / Paths ─────────────────────────────────────────────
TRAIN_PATH = globals().get("TRAIN_PATH", "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN")
VAL_PATH   = globals().get("VAL_PATH",   "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL")
TASK       = globals().get("TASK", "image_seg")   # "image_seg" | "ceus_seg" | "video_seg"

from src.config import CFG
CFG["train_path"] = TRAIN_PATH
CFG["val_path"]   = VAL_PATH
CFG["ckpt_dir"]   = "/kaggle/working/checkpoints"
os.makedirs(CFG["ckpt_dir"], exist_ok=True)   # ← Create BEFORE any save

# ── Task-specific overrides ─────────────────────────────────────
if TASK == "image_seg":
    CFG["img_size"]         = 512
    CFG["batch_size"]       = 8    # increased from 4 (AMP+UNet++ fits on T4)
    CFG["grad_accum_steps"] = 4    # effective = 32
    CFG["epochs"]           = 60
elif TASK == "ceus_seg":
    CFG["img_size"]         = 256
    CFG["batch_size"]       = 12   # increased from 8
    CFG["grad_accum_steps"] = 3    # effective = 36
    CFG["epochs"]           = 50
elif TASK == "video_seg":
    CFG["img_size"]         = 256
    CFG["batch_size"]       = 28   # increased from 16 (pre-extracted 2D frames are tiny)
    CFG["grad_accum_steps"] = 2    # effective = 56
    CFG["epochs"]           = 40

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
from src.models_v2 import build_seg_model, EMA, build_optimizer, build_scheduler
from src.losses_v2 import CompoundSegLoss
from src.augmentations_v2 import (
    get_seg_transforms, get_seg_transforms_ceus, get_seg_transforms_video,
    apply_seg_transform, ALBUMENTATIONS_AVAILABLE
)
from src.trainer import Trainer

print("✅ All imports OK")
if not ALBUMENTATIONS_AVAILABLE:
    print("⚠️  Install albumentations for best augmentation: pip install albumentations")

# ─────────────────────────────────────────────────────────────
#  Load JSON ground-truth
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

organ_counts = defaultdict(int)
for s in task_samples:
    organ_counts[s.get("dataset_name", s["organ"])] += 1
print("Per-dataset counts:")
for ds, cnt in sorted(organ_counts.items()):
    print(f"  {ds:20s}: {cnt}")

# ─────────────────────────────────────────────────────────────
#  Dataset classes
# ─────────────────────────────────────────────────────────────
from PIL import Image as PILImage

class ImageSegDataset(Dataset):
    """Dataset for image_seg. Loads PNG image + PNG mask."""
    def __init__(self, samples, train_root, val_root, transform):
        self.samples    = samples
        self.train_root = Path(train_root)
        self.val_root   = Path(val_root) if val_root else None
        self.transform  = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        pr = get_partition_root(self.train_root, self.val_root, s["data_partition_group"])

        # Load image
        img_path = pr / s["img_path_relative"]
        img = np.array(PILImage.open(img_path).convert("RGB"))  # (H, W, 3) uint8

        # Load mask
        mask_path = pr / s["mask_path_relative"]
        mask_raw = np.array(PILImage.open(mask_path))
        if mask_raw.dtype == bool:
            mask_raw = mask_raw.astype(np.uint8) * 255
        if mask_raw.ndim == 3:
            mask_raw = mask_raw[:, :, 0]
        # mask_raw: (H, W) uint8

        img_t, mask_t = apply_seg_transform(self.transform, img, mask_raw)
        return {"input": img_t, "mask": mask_t, "organ": s["organ"]}


class CEUSSegDataset(Dataset):
    """Dataset for ceus_seg. Extracts MIDDLE FRAME + loads NPZ mask."""
    def __init__(self, samples, train_root, val_root, transform):
        self.samples    = samples
        self.train_root = Path(train_root)
        self.val_root   = Path(val_root) if val_root else None
        self.transform  = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        pr = get_partition_root(self.train_root, self.val_root, s["data_partition_group"])

        # Load video → extract middle frame
        video = np.load(pr / s["input_path_relative"])  # (15, 256, 512, 3) uint8
        mid_frame = video[len(video) // 2]              # (256, 512, 3) uint8

        # Load mask
        npz = np.load(pr / s["annotation_path_relative"])
        mask_raw = (npz["mask"] > 127).astype(np.uint8) * 255   # (256, 512) uint8

        img_t, mask_t = apply_seg_transform(self.transform, mid_frame, mask_raw)
        return {"input": img_t, "mask": mask_t, "organ": s["organ"]}


# Pre-extracted frame cache directory for video_seg
PREPROCESS_DIR = Path("/kaggle/working/preprocessed_video_seg_v2")

class VideoSegDataset(Dataset):
    """Dataset for video_seg. Pre-extracts frames to disk for fast loading."""
    def __init__(self, frame_list, transform):
        self.frame_list = frame_list
        self.transform  = transform

    def __len__(self): return len(self.frame_list)

    def __getitem__(self, idx):
        item = self.frame_list[idx]
        frame    = np.load(item["img_path"])    # (256, 256) uint8
        mask_raw = np.load(item["mask_path"])   # (256, 256) uint8 {0,1}
        # Convert to HWC uint8 for albumentations
        frame_rgb = np.stack([frame, frame, frame], axis=2)   # (H,W,3) uint8
        mask_u8   = (mask_raw > 0).astype(np.uint8) * 255
        img_t, mask_t = apply_seg_transform(self.transform, frame_rgb, mask_u8)
        return {"input": img_t, "mask": mask_t, "organ": "Cardiac"}


def preextract_video_frames(task_samples, train_root, val_root, preprocess_dir):
    """Pre-extract and save 2D cardiac frames to disk to avoid reloading 3D .npy."""
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    frame_list = []
    print(f"\nPre-extracting {TASK} frames to {preprocess_dir} ...")
    t0 = time.time()
    for idx, s in enumerate(task_samples):
        pr = get_partition_root(Path(train_root), Path(val_root) if val_root else None,
                                s["data_partition_group"])
        ann_path = pr / s["annotation_path_relative"]
        if not ann_path.exists():
            continue
        try:
            npz = np.load(ann_path, allow_pickle=True)
            fnum_mask = npz["fnum_mask"].item()
            npy_path = pr / s["input_path_relative"]
            video = None
            sid   = s["sample_id"]
            for frame_key, mask_arr in fnum_mask.items():
                fidx = int(frame_key)
                img_p  = preprocess_dir / f"{sid}_f{fidx}_img.npy"
                mask_p = preprocess_dir / f"{sid}_f{fidx}_mask.npy"
                if not img_p.exists():
                    if video is None:
                        video = np.load(npy_path)  # (3, T, 256, 256) float
                    frame = video[0, fidx]  # (256, 256) float
                    # Save as uint8 to halve disk usage (float32=262KB vs uint8=65KB)
                    np.save(img_p, np.clip(frame, 0, 255).astype(np.uint8))
                    np.save(mask_p, (mask_arr > 127).astype(np.uint8))
                frame_list.append({"img_path": str(img_p), "mask_path": str(mask_p),
                                   "sample_id": sid})
        except Exception as e:
            print(f"  Warning: {ann_path}: {e}")
        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(task_samples)} videos processed ...")
    print(f"Pre-extraction done in {time.time()-t0:.1f}s  |  {len(frame_list)} frames total")
    return frame_list


# ─────────────────────────────────────────────────────────────
#  Build datasets and loaders
# ─────────────────────────────────────────────────────────────
random.seed(CFG["seed"])

if TASK == "image_seg":
    train_tf = get_seg_transforms(CFG, mode="train")
    val_tf   = get_seg_transforms(CFG, mode="val")
    # Split by sample (stratified shuffle)
    random.shuffle(task_samples)
    n_val = max(1, int(len(task_samples) * CFG["val_split"]))
    val_s, train_s = task_samples[:n_val], task_samples[n_val:]
    train_ds = ImageSegDataset(train_s, TRAIN_PATH, VAL_PATH, train_tf)
    val_ds   = ImageSegDataset(val_s,   TRAIN_PATH, VAL_PATH, val_tf)

elif TASK == "ceus_seg":
    train_tf = get_seg_transforms_ceus(CFG, mode="train")
    val_tf   = get_seg_transforms_ceus(CFG, mode="val")
    random.shuffle(task_samples)
    n_val = max(1, int(len(task_samples) * CFG["val_split"]))
    val_s, train_s = task_samples[:n_val], task_samples[n_val:]
    train_ds = CEUSSegDataset(train_s, TRAIN_PATH, VAL_PATH, train_tf)
    val_ds   = CEUSSegDataset(val_s,   TRAIN_PATH, VAL_PATH, val_tf)

elif TASK == "video_seg":
    train_tf = get_seg_transforms_video(CFG, mode="train")
    val_tf   = get_seg_transforms_video(CFG, mode="val")
    frame_list = preextract_video_frames(task_samples, TRAIN_PATH, VAL_PATH, PREPROCESS_DIR)
    # Split by video ID to prevent data leakage
    video_ids = list({f["sample_id"] for f in frame_list})
    random.shuffle(video_ids)
    n_val_vids = max(1, int(len(video_ids) * CFG["val_split"]))
    val_vids   = set(video_ids[:n_val_vids])
    train_frames = [f for f in frame_list if f["sample_id"] not in val_vids]
    val_frames   = [f for f in frame_list if f["sample_id"] in val_vids]
    train_ds = VideoSegDataset(train_frames, train_tf)
    val_ds   = VideoSegDataset(val_frames,   val_tf)

print(f"\nTrain: {len(train_ds)}  |  Val: {len(val_ds)}")

# DataLoader settings: NW=2 (not 4) to reduce Python worker overhead
# Each worker process uses ~1-2GB RAM regardless of data size!
# persistent_workers=False: workers die between epochs, freeing RAM
# prefetch_factor=2: minimal pre-fetch queue to cap RAM
NW = 2  # KEY: 2 workers = ~2-4GB overhead; 4 workers = ~4-8GB overhead

train_loader = DataLoader(
    train_ds, batch_size=CFG["batch_size"], shuffle=True,
    num_workers=NW, pin_memory=True, drop_last=True,
    persistent_workers=False, prefetch_factor=2,
)
val_loader = DataLoader(
    val_ds, batch_size=CFG["batch_size"] * 2, shuffle=False,
    num_workers=NW, pin_memory=True,
    persistent_workers=False, prefetch_factor=2,
)
print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")
print(f"Effective batch size: {CFG['batch_size'] * CFG['grad_accum_steps']}")

# ─────────────────────────────────────────────────────────────
#  Build Model, Loss, Optimizer, Scheduler
# ─────────────────────────────────────────────────────────────
model = build_seg_model(CFG)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel params: {total_params:,}")

criterion = CompoundSegLoss(CFG).to(DEVICE)
print(f"Loss: Dice({CFG['seg_dice_weight']}) + Focal({CFG['seg_focal_weight']}) "
      f"+ Boundary({CFG['seg_boundary_weight']})")

raw_model = model.module if hasattr(model, "module") else model
optimizer  = build_optimizer(raw_model, CFG)
scheduler  = build_scheduler(optimizer, CFG)
ema        = EMA(raw_model, decay=CFG["ema_decay"])

# ─────────────────────────────────────────────────────────────
#  Train
# ─────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    cfg=CFG,
    device=DEVICE,
    task_type="seg",
    ema=ema,
    ckpt_prefix=CKPT_PREFIX,
)

best_dice, history = trainer.fit(train_loader, val_loader)

# ─────────────────────────────────────────────────────────────
#  Save history JSON
# ─────────────────────────────────────────────────────────────
import json as _json
with open(f"{CFG['ckpt_dir']}/{CKPT_PREFIX}_history.json", "w") as f:
    _json.dump(history, f, indent=2)

# ─────────────────────────────────────────────────────────────
#  Cleanup (free memory for next task)
# ─────────────────────────────────────────────────────────────
del model, optimizer, scheduler, ema, train_loader, val_loader, train_ds, val_ds
import gc; gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print(f"\n✅ {TASK} training complete. Best val dice: {best_dice:.4f}")

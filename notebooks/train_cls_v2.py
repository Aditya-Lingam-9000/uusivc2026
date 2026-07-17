"""
notebooks/train_cls_v2.py  —  UUSIVC 2026 Classification Training v2

RAM FIX: CEUS videos (25MB each) are pre-extracted to 16-frame .npy files
         (~6MB each) ONCE before training. GPU then reads 6MB instead of 25MB,
         reducing page-cache RAM from ~10GB to ~2.6GB.

KEY SETTINGS:
  num_workers       = 2  (not 4 — fewer Python worker processes)
  persistent_workers= False  (workers die between epochs, freeing RAM)
  prefetch_factor   = 2  (minimal pre-fetch queue)

HOW TO RUN:
    TRAIN_PATH = "/kaggle/input/.../TRAIN"
    VAL_PATH   = "/kaggle/input/.../VAL"
    TASK = "image_cls"   # or "ceus_cls"
    exec(open('/kaggle/working/repo/notebooks/train_cls_v2.py').read())

    # Resume after timeout:
    CFG["resume_from"] = f"/kaggle/working/checkpoints/{TASK}_v2_latest.pth"
    exec(open('/kaggle/working/repo/notebooks/train_cls_v2.py').read())
"""

import sys, os, json, random, time, gc, ctypes
# Disable OpenCV/MKL multithreading inside workers/main process to prevent RAM bloat
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast

# ── Force src reload ───────────────────────────────────────────
for _m in list(sys.modules.keys()):
    if _m.startswith("src"):
        del sys.modules[_m]

# ── Config / Paths ─────────────────────────────────────────────
TRAIN_PATH = globals().get("TRAIN_PATH", "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN")
VAL_PATH   = globals().get("VAL_PATH",   "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL")
TASK       = globals().get("TASK", "image_cls")

from src.config import CFG
CFG["train_path"] = TRAIN_PATH
CFG["val_path"]   = VAL_PATH
CFG["ckpt_dir"]   = "/kaggle/working/checkpoints"
os.makedirs(CFG["ckpt_dir"], exist_ok=True)   # ← CRITICAL: create before any save

# ── Task-specific settings ──────────────────────────────────────
if TASK == "image_cls":
    CFG["img_size"]         = 512
    CFG["batch_size"]       = 24    # increased from 16
    CFG["grad_accum_steps"] = 2     # effective = 48
    CFG["epochs"]           = 50
elif TASK == "ceus_cls":
    CFG["img_size"]         = 256
    CFG["batch_size"]       = 8     # safe with pre-extracted 6MB files
    CFG["grad_accum_steps"] = 4     # effective = 32
    CFG["epochs"]           = 40
    CFG["ceus_n_frames"]    = 16

CKPT_PREFIX = f"{TASK}_v2"
CKPT_DIR    = CFG["ckpt_dir"]

# ── Device ─────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    if torch.cuda.device_count() > 1:
        print(f"DataParallel: {torch.cuda.device_count()} GPUs")
torch.manual_seed(CFG["seed"]); random.seed(CFG["seed"]); np.random.seed(CFG["seed"])

# ── Imports ────────────────────────────────────────────────────
from src.dataset import get_partition_root
from src.models_v2 import build_cls_model, build_ceus_cls_model, EMA, build_optimizer, build_scheduler
from src.losses_v2 import FocalCELoss
from src.augmentations_v2 import get_cls_transforms, apply_cls_transform, mixup_batch, mixup_criterion, ALBUMENTATIONS_AVAILABLE
from src.trainer import fmt_vram, fmt_time, accuracy
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
#  CEUS PRE-EXTRACTION (THE RAM FIX)
#  Extracts 16 frames per video ONCE, saves as (16, 256, 512, 3) uint8
#  File size: 6.3MB vs original 25MB → 4× RAM reduction
#  Total disk: ~504 × 6.3MB = ~3.2GB (within 19.5GB Kaggle limit)
# ─────────────────────────────────────────────────────────────
CEUS_PREP_DIR = Path("/kaggle/working/preprocessed_ceus_cls")

def preextract_ceus(samples, train_root, val_root, prep_dir, n_frames=16):
    prep_dir = Path(prep_dir)
    prep_dir.mkdir(parents=True, exist_ok=True)
    T_total = 64
    frame_idx = np.linspace(0, T_total - 1, n_frames).astype(int)
    result = []
    n_extracted = 0

    for i, s in enumerate(samples):
        pr  = get_partition_root(Path(train_root), Path(val_root) if val_root else None,
                                 s["data_partition_group"])
        sid = s["sample_id"]
        save_path = prep_dir / f"{sid}.npy"

        if not save_path.exists():
            video = np.load(pr / s["input_path_relative"])  # (64,256,512,3) uint8
            frames = video[frame_idx]                        # (16,256,512,3) uint8
            np.save(save_path, frames)
            n_extracted += 1

        result.append({
            "frames_path": str(save_path),
            "label":  s.get("class_label_index", 0),
            "organ":  s["organ"],
        })
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(samples)} videos ready (new extractions: {n_extracted})")

    print(f"Pre-extraction done. {n_extracted} new files. {len(result)} total samples.")
    return result

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
        img_np = np.array(PILImage.open(pr / s["input_path_relative"]).convert("RGB"))
        img_t  = apply_cls_transform(self.transform, img_np)
        label  = s.get("class_label_index", 0)
        return {"input": img_t, "label": torch.tensor(label, dtype=torch.long), "organ": s["organ"]}


class CEUSClsDataset(Dataset):
    """
    Loads pre-extracted (16, 256, 512, 3) uint8 frames.
    File size: 6.3MB vs 25MB original → keeps RAM under 3GB page cache.
    """
    def __init__(self, sample_list):
        self.sample_list = sample_list

    def __len__(self): return len(self.sample_list)

    def __getitem__(self, idx):
        item   = self.sample_list[idx]
        frames = np.load(item["frames_path"], mmap_mode='r')           # (16,256,512,3) uint8
        # Copy to memory to avoid memory-map index overhead on CPU during permute/division
        frames_t = torch.tensor(frames.copy()).permute(0, 3, 1, 2).float() / 255.0  # (16,3,256,512)
        return {
            "input": frames_t,
            "label": torch.tensor(item["label"], dtype=torch.long),
            "organ": item["organ"],
        }

# ─────────────────────────────────────────────────────────────
#  Stratified train/val split
# ─────────────────────────────────────────────────────────────
random.seed(CFG["seed"])
strata = defaultdict(list)
for i, s in enumerate(task_samples):
    strata[(s["organ"], s.get("class_label_index", 0))].append(i)

train_idx, val_idx = [], []
for indices in strata.values():
    random.shuffle(indices)
    n_val = max(1, int(len(indices) * CFG["val_split"]))
    val_idx.extend(indices[:n_val])
    train_idx.extend(indices[n_val:])

random.shuffle(train_idx)
train_s = [task_samples[i] for i in train_idx]
val_s   = [task_samples[i] for i in val_idx]
print(f"\nTrain: {len(train_s)}  |  Val: {len(val_s)}")

# ─────────────────────────────────────────────────────────────
#  Build datasets and loaders
# ─────────────────────────────────────────────────────────────
NW = 2   # ← KEY: 2 workers max (each worker = ~1GB Python overhead)
         #         4 workers = 4GB overhead before loading any data!

if TASK == "image_cls":
    train_tf = get_cls_transforms(CFG, mode="train")
    val_tf   = get_cls_transforms(CFG, mode="val")
    train_ds = ImageClsDataset(train_s, TRAIN_PATH, VAL_PATH, train_tf)
    val_ds   = ImageClsDataset(val_s,   TRAIN_PATH, VAL_PATH, val_tf)
else:  # ceus_cls
    print("\n📦 Pre-extracting CEUS frames (runs once, reuses on resume)...")
    all_preext = preextract_ceus(task_samples, TRAIN_PATH, VAL_PATH, CEUS_PREP_DIR,
                                 n_frames=CFG["ceus_n_frames"])
    # Map back to train/val splits using sample_id
    sid_to_item = {s["sample_id"]: item
                   for s, item in zip(task_samples, all_preext)}
    train_pe = [sid_to_item[task_samples[i]["sample_id"]] for i in train_idx]
    val_pe   = [sid_to_item[task_samples[i]["sample_id"]] for i in val_idx]
    train_ds = CEUSClsDataset(train_pe)
    val_ds   = CEUSClsDataset(val_pe)

# WeightedRandomSampler (balanced batches per organ)
organ_class_cnt = defaultdict(lambda: defaultdict(int))
for s in train_s:
    organ_class_cnt[s["organ"]][s.get("class_label_index", 0)] += 1

sample_weights = []
for s in train_s:
    org, lbl  = s["organ"], s.get("class_label_index", 0)
    org_total = sum(organ_class_cnt[org].values())
    sample_weights.append(org_total / (2.0 * max(organ_class_cnt[org][lbl], 1)))

sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(
    train_ds,
    batch_size=CFG["batch_size"],
    sampler=sampler,
    num_workers=0,
    pin_memory=False,
)
val_loader = DataLoader(
    val_ds,
    batch_size=CFG["batch_size"] * 2,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)
print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")
print(f"Effective batch size: {CFG['batch_size'] * CFG['grad_accum_steps']}")

# ─────────────────────────────────────────────────────────────
#  Model + Loss
# ─────────────────────────────────────────────────────────────
model = build_cls_model(CFG) if TASK == "image_cls" else build_ceus_cls_model(CFG)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(DEVICE)
print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")

# Global class weights
total_counts = [0, 0]
for s in train_s:
    total_counts[s.get("class_label_index", 0)] += 1
gt = sum(total_counts)
global_weights = torch.tensor(
    [gt / (2.0 * max(total_counts[0], 1)), gt / (2.0 * max(total_counts[1], 1))],
    dtype=torch.float32, device=DEVICE
)
print(f"Class weights: {global_weights.tolist()}  counts: {total_counts}")

criterion = FocalCELoss(
    gamma=CFG["cls_focal_gamma"],
    class_weights=global_weights,
    label_smoothing=CFG["label_smoothing"],
).to(DEVICE)

raw_model = model.module if hasattr(model, "module") else model
optimizer = build_optimizer(raw_model, CFG)
scheduler = build_scheduler(optimizer, CFG)
ema       = EMA(raw_model, decay=CFG["ema_decay"])
scaler    = GradScaler('cuda', enabled=CFG["use_amp"])

# ─────────────────────────────────────────────────────────────
#  Resume
# ─────────────────────────────────────────────────────────────
best_acc, history, start_epoch = 0.0, [], 1
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
    print(f"✅ Resumed epoch {start_epoch-1} | best_acc={best_acc:.4f}")

epochs     = CFG["epochs"]
accum_steps= CFG["grad_accum_steps"]
log_steps  = CFG["log_steps"]
mixup_a    = CFG.get("mixup_alpha", 0.2)
warmup_e   = CFG.get("warmup_epochs", 5)

print(f"\n{'='*65}")
print(f"  {CKPT_PREFIX}  |  epochs {start_epoch}→{epochs}")
print(f"  AMP={CFG['use_amp']}  |  EffBatch={CFG['batch_size']*accum_steps}")
print(f"  Workers={NW}  |  persistent_workers=False  |  prefetch=2")
print(f"  VRAM at start: {fmt_vram()}")
print(f"{'='*65}\n")

# ─────────────────────────────────────────────────────────────
#  Training Loop
# ─────────────────────────────────────────────────────────────
for epoch in range(start_epoch, epochs + 1):
    t0 = time.time()

    # Warmup LR
    if epoch <= warmup_e:
        factor = epoch / max(warmup_e, 1)
        for i, pg in enumerate(optimizer.param_groups):
            base = CFG["encoder_lr"] if i == 0 and len(optimizer.param_groups) > 1 else CFG["lr"]
            pg["lr"] = base * factor

    # ── Train ───────────────────────────────────────────────
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    n_steps = len(train_loader)

    for step, batch in enumerate(train_loader, 1):
        imgs   = batch["input"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)
        bs     = imgs.size(0)

        # Mixup (image_cls only — CEUS videos are large, skip for memory)
        use_mix = (TASK == "image_cls" and mixup_a > 0 and random.random() < 0.5)
        if use_mix:
            imgs, (la, lb, lam) = mixup_batch(imgs, labels, alpha=mixup_a)

        with autocast('cuda', enabled=CFG["use_amp"]):
            logits = model(imgs)
            if use_mix:
                loss = mixup_criterion(criterion, logits, la, lb, lam) / accum_steps
            else:
                loss = criterion(logits, labels) / accum_steps

        scaler.scale(loss).backward()

        if step % accum_steps == 0 or step == n_steps:
            if CFG["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(raw_model)

        with torch.no_grad():
            loss_sum += loss.item() * accum_steps * bs
            correct  += (logits.argmax(1) == labels).sum().item()
            total    += bs

        if step % log_steps == 0 or step == n_steps:
            t_el  = time.time() - t0
            eta_e = (n_steps - step) * (t_el / step)
            eta_t = (epochs - epoch) * (t_el / step * n_steps)
            lr_s  = "/".join(f"{pg['lr']:.2e}" for pg in optimizer.param_groups)
            print(f"    [E{epoch:02d} S{step:04d}/{n_steps}]  "
                  f"loss={loss_sum/max(total,1):.4f}  acc={correct/max(total,1):.4f}  "
                  f"|  LR:{lr_s}  |  ETA_epoch:{fmt_time(eta_e)}  "
                  f"|  ETA_total:{fmt_time(eta_t)}  |  VRAM:{fmt_vram()}")

    scheduler.step(epoch)
    train_loss = loss_sum / max(total, 1)
    train_acc  = correct / max(total, 1)

    # ── Validate (EMA model) ────────────────────────────────
    ema.shadow.eval()
    vl_sum, vc, vt = 0.0, 0, 0
    org_c, org_t   = defaultdict(int), defaultdict(int)

    with torch.no_grad():
        for batch in val_loader:
            imgs   = batch["input"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)
            with autocast('cuda', enabled=CFG["use_amp"]):
                logits = ema.shadow(imgs)
                vl_sum += criterion(logits, labels).item() * imgs.size(0)
            preds = logits.argmax(1)
            mask  = (preds == labels)
            vc   += mask.sum().item()
            vt   += imgs.size(0)
            for i, org in enumerate(batch["organ"]):
                org_t[org] += 1
                org_c[org] += mask[i].item()

    val_acc  = vc / max(vt, 1)
    val_loss = vl_sum / max(vt, 1)
    t_epoch  = time.time() - t0
    is_best  = val_acc > best_acc
    if is_best:
        best_acc = val_acc

    # Save
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
    torch.save(ckpt_data, f"{CKPT_DIR}/{CKPT_PREFIX}_latest.pth")
    if is_best:
        torch.save(ckpt_data, f"{CKPT_DIR}/{CKPT_PREFIX}_best.pth")
    history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                    "val_loss": val_loss, "val_acc": val_acc})

    print(f"\n{'─'*65}")
    print(f"  [E{epoch:02d}/{epochs}]  time={fmt_time(t_epoch)}  "
          f"ETA={fmt_time((epochs-epoch)*t_epoch)}  LR={lr_str}")
    print(f"  Train: loss={train_loss:.4f}  acc={train_acc:.4f}")
    print(f"  Val:   loss={val_loss:.4f}  acc={val_acc:.4f}  "
          f"{'← 💾 BEST' if is_best else f'(best={best_acc:.4f})'}")
    print(f"  VRAM: {fmt_vram()}")
    print("  Per-organ val:")
    for org in sorted(org_t):
        a = org_c[org] / max(org_t[org], 1)
        print(f"    {org:20s}: {a:.4f}  ({org_c[org]}/{org_t[org]})")
    print(f"{'─'*65}")

    # ── Free page cache and heap between epochs ─────────────────────
    del ckpt_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"✅ {TASK} Done! best_acc={best_acc:.4f}")
print(f"   {CKPT_DIR}/{CKPT_PREFIX}_best.pth")
print(f"{'='*65}")

del model, optimizer, scheduler, ema, train_loader, val_loader, train_ds, val_ds
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

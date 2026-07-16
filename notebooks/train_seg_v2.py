"""
notebooks/train_seg_v2.py
GPU-maximized segmentation training — image_seg, ceus_seg, video_seg.

KEY OPTIMIZATIONS vs original:
  - ALL tasks pre-extracted to .npy (np.load is 5x faster than PIL)
  - num_workers=0 (no forking overhead / RAM leaks)
  - Step-level logging every LOG_EVERY steps
  - No boundary loss (scipy removed — was causing RAM leak)
  - DataParallel with >=2 GPUs
  - Larger batch → higher GPU utilization

HOW TO RUN:
    TRAIN_PATH = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN"
    VAL_PATH   = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL"
    exec(open('/kaggle/working/repo/notebooks/train_seg_v2.py').read())

OPTIONAL OVERRIDES (set before exec):
    CFG_OVERRIDES = {"epochs": 30, "batch_size": 8}
"""

import sys, os, json, random, time, gc
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from PIL import Image

# ── Force module reload ────────────────────────────────────────
for mod in list(sys.modules.keys()):
    if mod.startswith("src"):
        del sys.modules[mod]

# ── Config ────────────────────────────────────────────────────
from src.config import get_cfg, print_cfg
cfg = get_cfg("seg")

_gbl = globals()
if "TRAIN_PATH" in _gbl:
    cfg["train_root"] = _gbl["TRAIN_PATH"]
if "VAL_PATH" in _gbl:
    cfg["val_root"] = _gbl["VAL_PATH"]

cfg["seg_loss"] = "dice_focal"        # no boundary loss (no scipy)
cfg["num_workers"] = 0                # no fork/RAM leaks
cfg["pin_memory"] = False             # DataParallel handles its own pinning
cfg.update(_gbl.get("CFG_OVERRIDES", {}))

os.makedirs(cfg["ckpt_dir"], exist_ok=True)
os.makedirs(cfg["preprocess_dir"], exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPU = torch.cuda.device_count() if torch.cuda.is_available() else 0
LOG_EVERY = cfg.get("log_every_n_batches", 50)  # print every N batches

print(f"Device   : {DEVICE}")
if N_GPU > 0:
    for i in range(N_GPU):
        print(f"  GPU {i}  : {torch.cuda.get_device_name(i)}")
    print(f"  N_GPU  : {N_GPU}")

torch.manual_seed(cfg["seed"])
random.seed(cfg["seed"])
np.random.seed(cfg["seed"])

from src.dataset import get_partition_root
from src.models_v2 import build_seg_model
from src.losses_v2 import CompoundSegLoss
from src.augmentations_v2 import get_seg_train_transform, get_seg_val_transform
from src.trainer import (
    EMA, get_vram_usage, format_time,
    build_optimizer, build_scheduler,
    save_checkpoint, load_checkpoint, dice_score_fn,
)

print("✅ All imports OK")
print_cfg(cfg)

TRAIN = cfg["train_root"]
VAL_DIR = cfg["val_root"]
PREPROC = cfg["preprocess_dir"]

# ── Load training ground truth ─────────────────────────────────
PRIVATE_GT = f"{TRAIN}/dataset_json_fingerprints_v4/private_train_ground_truth.json"
PUBLIC_GT  = f"{TRAIN}/dataset_json_fingerprints_v4/public_all_ground_truth.json"

all_samples = []
for jp in [PRIVATE_GT, PUBLIC_GT]:
    if os.path.exists(jp):
        with open(jp) as f:
            all_samples.extend(json.load(f))
print(f"Total training samples loaded: {len(all_samples)}")


# ═══════════════════════════════════════════════════════════════
#  FAST NUMPY DATASET — loads pre-extracted .npy files only
# ═══════════════════════════════════════════════════════════════
class FastNpySegDataset(Dataset):
    """
    Loads pre-extracted (img.npy, mask.npy) pairs.
    np.load is ~5x faster than PIL for large images.
    No scipy, no PIL, no multiprocessing needed.
    """
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img_np  = np.load(r["img_path"])   # (H, W, 3) uint8
        mask_np = np.load(r["mask_path"])  # (H, W)    float32 {0,1}

        # Ensure uint8 for albumentations
        if img_np.dtype != np.uint8:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        if self.transform:
            out    = self.transform(image=img_np, mask=mask_np)
            img_t  = out["image"]    # (3, H, W) float32
            mask_t = out["mask"]     # (H, W)    float32
        else:
            img_t  = torch.tensor(img_np).permute(2, 0, 1).float() / 255.0
            mask_t = torch.tensor(mask_np).float()

        mask_t = mask_t.unsqueeze(0)  # (1, H, W)
        dist_t = torch.zeros_like(mask_t)  # unused (no boundary loss)

        return {
            "input":    img_t,
            "mask":     mask_t,
            "dist_map": dist_t,
            "organ":    r.get("organ", "Unknown"),
        }


# ═══════════════════════════════════════════════════════════════
#  PRE-EXTRACTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def preextract_image_seg(task_samples, train_root, val_root, preproc_dir):
    """
    Convert image_seg PNGs → (img.npy, mask.npy) pairs.
    Much faster than PIL loading in training loop.
    """
    out_dir = Path(preproc_dir) / "image_seg"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    skipped = 0
    t0 = time.time()
    total = len(task_samples)

    for idx, s in enumerate(task_samples):
        part_root = get_partition_root(
            Path(train_root),
            Path(val_root) if val_root else None,
            s["data_partition_group"],
        )
        sid = s["sample_id"]
        img_p  = out_dir / f"{sid}_img.npy"
        mask_p = out_dir / f"{sid}_msk.npy"

        if not img_p.exists():
            try:
                img_raw  = np.array(Image.open(part_root / s["img_path_relative"]).convert("RGB"))
                mask_raw = np.array(Image.open(part_root / s["mask_path_relative"]))
                if mask_raw.dtype == bool:
                    mask_raw = mask_raw.astype(np.uint8) * 255
                if mask_raw.ndim == 3:
                    mask_raw = mask_raw[:, :, 0]
                mask_bin = (mask_raw > 127).astype(np.float32)
                np.save(str(img_p),  img_raw)
                np.save(str(mask_p), mask_bin)
            except Exception as e:
                skipped += 1
                if skipped <= 10:
                    print(f"  [WARN] {sid}: {e}")
                continue

        records.append({
            "img_path":  str(img_p),
            "mask_path": str(mask_p),
            "sample_id": sid,
            "organ":     s.get("organ", s.get("dataset_name", "Unknown")),
        })

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (total - idx - 1)
            print(f"  [{idx+1}/{total}]  elapsed={format_time(elapsed)}  ETA={format_time(eta)}")

    elapsed = time.time() - t0
    print(f"  ✅ image_seg: {len(records)} records in {format_time(elapsed)} (skipped={skipped})")
    return records


def preextract_ceus_seg(task_samples, train_root, val_root, preproc_dir):
    """Extract middle frame from CEUS videos."""
    out_dir = Path(preproc_dir) / "ceus_seg"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for idx, s in enumerate(task_samples):
        part_root = get_partition_root(
            Path(train_root), Path(val_root) if val_root else None,
            s["data_partition_group"],
        )
        sid = s["sample_id"]
        img_p  = out_dir / f"{sid}_img.npy"
        mask_p = out_dir / f"{sid}_msk.npy"

        if not img_p.exists():
            try:
                video = np.load(part_root / s["input_path_relative"])  # (15,256,512,3)
                mid   = video[video.shape[0] // 2]                     # (256,512,3)
                np.save(str(img_p), mid.astype(np.uint8))

                npz   = np.load(part_root / s["annotation_path_relative"])
                mask  = npz["mask"].astype(np.float32) / 255.0
                np.save(str(mask_p), mask)
            except Exception as e:
                print(f"  [WARN] ceus_seg {sid}: {e}")
                continue

        records.append({
            "img_path":  str(img_p),
            "mask_path": str(mask_p),
            "sample_id": sid,
            "organ":     s.get("organ", "Unknown"),
        })

        if (idx + 1) % 200 == 0:
            print(f"  ceus_seg [{idx+1}/{len(task_samples)}]")

    print(f"  ✅ ceus_seg: {len(records)} records")
    return records


def preextract_video_seg(task_samples, train_root, val_root, preproc_dir):
    """Extract annotated frames from cardiac videos."""
    out_dir = Path(preproc_dir) / "video_seg"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for idx, s in enumerate(task_samples):
        part_root = get_partition_root(
            Path(train_root), Path(val_root) if val_root else None,
            s["data_partition_group"],
        )
        sid = s["sample_id"]

        try:
            npz       = np.load(part_root / s["annotation_path_relative"], allow_pickle=True)
            fnum_mask = npz["fnum_mask"].item()
            video     = None

            for frame_key, mask_arr in fnum_mask.items():
                fidx  = int(frame_key)
                img_p  = out_dir / f"{sid}_f{fidx}_img.npy"
                mask_p = out_dir / f"{sid}_f{fidx}_msk.npy"

                if not img_p.exists():
                    if video is None:
                        video = np.load(part_root / s["input_path_relative"])
                    frame = video[0, fidx]  # (256,256) float
                    frame_u8 = np.clip(frame, 0, 255).astype(np.uint8)
                    frame_3ch = np.stack([frame_u8, frame_u8, frame_u8], axis=-1)
                    mask_bin = (mask_arr / 255.0).clip(0, 1).astype(np.float32)
                    np.save(str(img_p),  frame_3ch)
                    np.save(str(mask_p), mask_bin)

                records.append({
                    "img_path":  str(img_p),
                    "mask_path": str(mask_p),
                    "sample_id": sid,
                    "organ":     s.get("organ", "Cardiac"),
                })

            del video
        except Exception as e:
            print(f"  [WARN] video_seg {sid}: {e}")

        if (idx + 1) % 200 == 0:
            print(f"  video_seg [{idx+1}/{len(task_samples)}]")

    print(f"  ✅ video_seg: {len(records)} records")
    return records


# ═══════════════════════════════════════════════════════════════
#  HELPER: build train/val split from records
# ═══════════════════════════════════════════════════════════════
def split_records(records, val_frac, seed):
    """Stratified split by organ."""
    random.seed(seed)
    organ_to_idx = defaultdict(list)
    for i, r in enumerate(records):
        organ_to_idx[r.get("organ", "Unknown")].append(i)
    train_idx, val_idx = [], []
    for idxs in organ_to_idx.values():
        random.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_frac))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    train_recs = [records[i] for i in train_idx]
    val_recs   = [records[i] for i in val_idx]
    return train_recs, val_recs


# ═══════════════════════════════════════════════════════════════
#  MAIN TASK LOOP
# ═══════════════════════════════════════════════════════════════
for current_task in cfg["seg_tasks"]:
    print(f"\n{'='*60}")
    print(f"  TRAINING: {current_task}")
    print(f"{'='*60}")

    task_samples = [s for s in all_samples if s["task"] == current_task]
    print(f"Total {current_task} samples: {len(task_samples)}")
    if not task_samples:
        print("  ⚠️  No samples, skipping")
        continue

    # ── Determine resolution and task prefix ──────────────────
    if current_task == "image_seg":
        img_size = cfg["img_size_seg"]
        cfg["ckpt_prefix"] = "image_seg"
    elif current_task == "ceus_seg":
        img_size = cfg["img_size_ceus_seg"]
        cfg["ckpt_prefix"] = "ceus_seg"
    elif current_task == "video_seg":
        img_size = cfg["img_size_video_seg"]
        cfg["ckpt_prefix"] = "video_seg"
    else:
        img_size = cfg["img_size_seg"]
        cfg["ckpt_prefix"] = current_task

    train_aug = get_seg_train_transform(img_size)
    val_aug   = get_seg_val_transform(img_size)

    # ── PRE-EXTRACTION ─────────────────────────────────────────
    print(f"\n  📦 Pre-extracting {current_task} to .npy ...")
    t_pre = time.time()
    if current_task == "image_seg":
        all_records = preextract_image_seg(task_samples, TRAIN, VAL_DIR, PREPROC)
    elif current_task == "ceus_seg":
        all_records = preextract_ceus_seg(task_samples, TRAIN, VAL_DIR, PREPROC)
    elif current_task == "video_seg":
        all_records = preextract_video_seg(task_samples, TRAIN, VAL_DIR, PREPROC)
    print(f"  Pre-extraction: {format_time(time.time()-t_pre)}")

    # For video_seg: split by sample_id to avoid leakage
    if current_task == "video_seg":
        all_ids = list(set(r["sample_id"] for r in all_records))
        random.seed(cfg["seed"])
        random.shuffle(all_ids)
        n_val = max(1, int(len(all_ids) * cfg["val_split"]))
        val_ids = set(all_ids[:n_val])
        train_records = [r for r in all_records if r["sample_id"] not in val_ids]
        val_records   = [r for r in all_records if r["sample_id"] in val_ids]
    else:
        train_records, val_records = split_records(all_records, cfg["val_split"], cfg["seed"])

    print(f"  Train: {len(train_records)}  Val: {len(val_records)}")

    # ── DataLoaders (num_workers=0 — no fork/RAM leaks) ────────
    train_ds = FastNpySegDataset(train_records, train_aug)
    val_ds   = FastNpySegDataset(val_records, val_aug)

    # Effective batch = batch_size * N_GPU (DataParallel doubles throughput)
    bs = cfg["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
    )
    print(f"  Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")
    print(f"  Effective batch/step: {bs} × {max(N_GPU,1)} GPUs = {bs*max(N_GPU,1)}")

    # ── Model ─────────────────────────────────────────────────
    model = build_seg_model(cfg)
    if N_GPU > 1:
        print(f"  Wrapping in DataParallel ({N_GPU} GPUs)")
        model = nn.DataParallel(model)
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    # ── Loss / Optimizer / Scheduler ──────────────────────────
    criterion = CompoundSegLoss(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler    = GradScaler("cuda", enabled=cfg["use_amp"])
    ema       = EMA(model, decay=cfg["ema_decay"]) if cfg["use_ema"] else None

    start_epoch, best_metric, history = load_checkpoint(
        model, optimizer, scheduler, scaler, cfg,
    )

    accum = cfg["grad_accum_steps"]
    patience_counter = 0

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t_epoch = time.time()

        # ═══════════ TRAIN ═══════════
        model.train()
        stats    = defaultdict(float)
        n_train  = 0

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            imgs     = batch["input"].to(DEVICE, non_blocking=True)
            masks    = batch["mask"].to(DEVICE, non_blocking=True)

            with autocast("cuda", enabled=cfg["use_amp"]):
                logits = model(imgs)
                if logits.shape[2:] != masks.shape[2:]:
                    logits = F.interpolate(
                        logits, size=masks.shape[2:],
                        mode="bilinear", align_corners=False,
                    )
                loss, loss_parts = criterion(logits, masks, None)
                loss_scaled = loss / accum

            scaler.scale(loss_scaled).backward()

            if (batch_idx + 1) % accum == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            bs_actual = imgs.size(0)
            stats["loss"]       += loss.item() * bs_actual
            stats["dice_loss"]  += loss_parts["dice"] * bs_actual
            stats["focal_loss"] += loss_parts["focal"] * bs_actual
            stats["dice"]       += dice_score_fn(logits, masks).item() * bs_actual
            n_train             += bs_actual

            # ── Step-level log ────────────────────────────────
            if (batch_idx + 1) % LOG_EVERY == 0:
                step_loss = stats["loss"] / max(n_train, 1)
                step_dice = stats["dice"] / max(n_train, 1)
                elapsed   = time.time() - t_epoch
                steps_done = batch_idx + 1
                steps_tot  = len(train_loader)
                eta_step   = elapsed / steps_done * (steps_tot - steps_done)
                lr_now     = optimizer.param_groups[0]["lr"]
                print(
                    f"  Ep[{epoch:02d}] "
                    f"Step[{steps_done:04d}/{steps_tot}] "
                    f"loss={step_loss:.4f} "
                    f"dice={step_dice:.4f} "
                    f"lr={lr_now:.2e} "
                    f"VRAM={get_vram_usage()} "
                    f"ETA={format_time(eta_step)}"
                )

        if scheduler:
            scheduler.step()

        # ═══════════ VALIDATE ════════
        model.eval()
        if ema:
            ema.apply(model)

        val_stats  = defaultdict(float)
        organ_dice = defaultdict(lambda: [0.0, 0])
        n_val      = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs   = batch["input"].to(DEVICE, non_blocking=True)
                masks  = batch["mask"].to(DEVICE, non_blocking=True)
                organs = batch["organ"]

                with autocast("cuda", enabled=cfg["use_amp"]):
                    logits = model(imgs)
                    if logits.shape[2:] != masks.shape[2:]:
                        logits = F.interpolate(
                            logits, size=masks.shape[2:],
                            mode="bilinear", align_corners=False,
                        )
                    loss, _ = criterion(logits, masks, None)

                bs_v = imgs.size(0)
                val_stats["loss"] += loss.item() * bs_v
                d = dice_score_fn(logits, masks).item()
                val_stats["dice"] += d * bs_v
                n_val += bs_v

                for i, organ in enumerate(organs):
                    od = dice_score_fn(logits[i:i+1], masks[i:i+1]).item()
                    organ_dice[organ][0] += od
                    organ_dice[organ][1] += 1

        if ema:
            ema.restore(model)

        # ═══════════ EPOCH SUMMARY ══
        train_loss = stats["loss"] / max(n_train, 1)
        train_dice = stats["dice"] / max(n_train, 1)
        val_loss   = val_stats["loss"] / max(n_val, 1)
        val_dice   = val_stats["dice"] / max(n_val, 1)
        elapsed    = time.time() - t_epoch
        eta_total  = elapsed * (cfg["epochs"] - epoch)
        lr_now     = optimizer.param_groups[0]["lr"]

        print(f"\n{'─'*60}")
        print(
            f"  ✅ EPOCH {epoch:02d}/{cfg['epochs']}  "
            f"time={format_time(elapsed)}  "
            f"ETA={format_time(eta_total)}  "
            f"LR={lr_now:.2e}"
        )
        print(
            f"  Train → loss={train_loss:.4f}  dice={train_dice:.4f}  "
            f"(dice_l={stats['dice_loss']/max(n_train,1):.4f}  "
            f"focal_l={stats['focal_loss']/max(n_train,1):.4f})"
        )
        print(f"  Val   → loss={val_loss:.4f}  dice={val_dice:.4f}  "
              f"(best={best_metric:.4f})  VRAM={get_vram_usage()}")

        if organ_dice:
            print("  Per-organ val dice:")
            for org in sorted(organ_dice):
                tot, cnt = organ_dice[org]
                print(f"    {org:22s}: {tot/cnt:.4f}  (n={cnt})")
        print(f"{'─'*60}")

        # ── Checkpoint ────────────────────────────────────────
        is_best = val_dice > best_metric
        if is_best:
            best_metric = val_dice
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                        best_metric, history, cfg, is_best=is_best)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_dice": train_dice,
            "val_loss": val_loss, "val_dice": val_dice,
        })

        if cfg["early_stop_patience"] > 0 and patience_counter >= cfg["early_stop_patience"]:
            print(f"\n  ⏹ Early stopping (patience={cfg['early_stop_patience']})")
            break

    # ── Save history ──────────────────────────────────────────
    pfx = cfg["ckpt_prefix"]
    with open(f"{cfg['ckpt_dir']}/{pfx}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  🎉 {current_task} DONE — Best val dice: {best_metric:.4f}")
    print(f"     Checkpoint: {cfg['ckpt_dir']}/{pfx}_best.pth")
    print(f"{'='*60}")

    # ── Cleanup GPU/RAM for next task ─────────────────────────
    del model, optimizer, scheduler, scaler, ema
    del train_ds, val_ds, train_loader, val_loader
    del all_records, train_records, val_records
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(2)  # give OS time to release memory

    best_metric = 0.0
    history = []
    cfg["resume_from"] = None

print(f"\n{'='*60}")
print(f"  🏁 ALL SEGMENTATION TASKS COMPLETE")
print(f"{'='*60}")

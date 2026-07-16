"""
notebooks/train_seg_v2.py  —  UUSIVC segmentation training (image_seg, ceus_seg, video_seg)

KEY DESIGN:
  - Pre-extract PNGs/videos → .npy files in /tmp/preprocessed/
      /tmp/ has 57.6 GB disk on Kaggle — will NEVER fill the 19.5 GB /kaggle/working/ limit
  - FastNpySegDataset reads .npy per batch — fast, zero RAM accumulation
  - num_workers=2 with persistent_workers — workers prefetch while GPU computes
  - GPU-native augmentations: only torch.flip + noise (NO TF.affine — it syncs to CPU!)
  - AMP + DataParallel for dual T4

HOW TO RUN:
    TRAIN_PATH = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN"
    VAL_PATH   = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL"
    CFG_OVERRIDES = {"batch_size": 16, "epochs": 40}
    exec(open('/kaggle/working/repo/notebooks/train_seg_v2.py').read())
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

# ── Force src module reload ────────────────────────────────────
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

# Use /tmp/ for pre-extracted .npy files — it has 57.6 GB disk space on Kaggle
# This avoids filling the 19.5 GB /kaggle/working/ limit
cfg["preprocess_dir"] = "/tmp/preprocessed"
cfg["seg_loss"] = "dice_focal"        # no boundary loss (no scipy)
cfg["num_workers"] = 2                # 2 workers prefetch while GPU computes
cfg["pin_memory"] = True              # faster CPU→GPU transfer
cfg.update(_gbl.get("CFG_OVERRIDES", {}))

os.makedirs(cfg["ckpt_dir"], exist_ok=True)
os.makedirs(cfg["preprocess_dir"], exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPU = torch.cuda.device_count() if torch.cuda.is_available() else 0
LOG_EVERY = cfg.get("log_every_n_batches", 25)

TRAIN   = cfg["train_root"]
VAL_DIR = cfg["val_root"]
PREPROC = cfg["preprocess_dir"]

# ── Src imports ────────────────────────────────────────────────
from src.models_v2    import build_seg_model
from src.losses_v2   import CompoundSegLoss
from src.trainer     import (EMA, build_optimizer, build_scheduler,
                              save_checkpoint, load_checkpoint,
                              dice_score_fn, format_time, get_vram_usage)
from src.dataset     import get_partition_root

print(f"  Device : {DEVICE}")
for i in range(N_GPU):
    print(f"  GPU {i}  : {torch.cuda.get_device_name(i)}")
print(f"  N_GPU  : {N_GPU}")
print("✅ All imports OK")
print_cfg(cfg)

# ── Load ground-truth JSON ─────────────────────────────────────
PRIVATE_GT = f"{TRAIN}/dataset_json_fingerprints_v4/private_train_ground_truth.json"
PUBLIC_GT  = f"{TRAIN}/dataset_json_fingerprints_v4/public_all_ground_truth.json"

all_samples = []
for jp in [PRIVATE_GT, PUBLIC_GT]:
    if os.path.exists(jp):
        with open(jp) as f:
            all_samples.extend(json.load(f))
print(f"Total training samples loaded: {len(all_samples)}")


# ═══════════════════════════════════════════════════════════════
#  DATASET — reads pre-extracted .npy files from /tmp/
# ═══════════════════════════════════════════════════════════════
class FastNpySegDataset(Dataset):
    """Reads (img.npy, mask.npy) pairs from disk. Fast np.load, zero RAM accumulation."""
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img_np  = np.load(r["img_path"])   # (H, W, 3) uint8
        mask_np = np.load(r["mask_path"])  # (H, W) float32 {0,1}

        # Simple flips — fast CPU ops
        if random.random() < 0.5:
            img_np  = img_np[:, ::-1, :]
            mask_np = mask_np[:, ::-1]
        if random.random() < 0.2:
            img_np  = img_np[::-1, :, :]
            mask_np = mask_np[::-1, :]

        # torch.tensor() always copies — safe for DataLoader workers
        img_t  = torch.tensor(np.ascontiguousarray(img_np), dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask_t = torch.tensor(np.ascontiguousarray(mask_np), dtype=torch.float32).unsqueeze(0)

        return {
            "input": img_t,
            "mask":  mask_t,
            "organ": r.get("organ", "Unknown"),
        }


class ValNpySegDataset(Dataset):
    """Reads (img.npy, mask.npy) for validation — no augmentation."""
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img_np  = np.load(r["img_path"])
        mask_np = np.load(r["mask_path"])
        # torch.tensor() always copies — safe for DataLoader workers
        img_t  = torch.tensor(np.ascontiguousarray(img_np), dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask_t = torch.tensor(np.ascontiguousarray(mask_np), dtype=torch.float32).unsqueeze(0)
        return {
            "input": img_t,
            "mask":  mask_t,
            "organ": r.get("organ", "Unknown"),
        }


# ── GPU-native normalization (fast, no CPU sync) ───────────────
_MEAN = None
_STD  = None

def gpu_normalize(imgs):
    global _MEAN, _STD
    if _MEAN is None or _MEAN.device != imgs.device:
        _MEAN = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1,3,1,1)
        _STD  = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1,3,1,1)
    return (imgs - _MEAN) / _STD


# ═══════════════════════════════════════════════════════════════
#  PRE-EXTRACTION → /tmp/preprocessed/  (57 GB disk, safe!)
# ═══════════════════════════════════════════════════════════════
def preextract_image_seg(task_samples, train_root, val_root, preproc_dir):
    out_dir = Path(preproc_dir) / "image_seg"
    out_dir.mkdir(parents=True, exist_ok=True)
    records, skipped = [], 0
    t0, total = time.time(), len(task_samples)

    for idx, s in enumerate(task_samples):
        part_root = get_partition_root(
            Path(train_root), Path(val_root) if val_root else None,
            s["data_partition_group"],
        )
        sid   = s["sample_id"]
        img_p = out_dir / f"{sid}_img.npy"
        msk_p = out_dir / f"{sid}_msk.npy"

        if not img_p.exists():
            try:
                img_raw  = np.array(Image.open(part_root / s["img_path_relative"]).convert("RGB"), dtype=np.uint8)
                mask_raw = np.array(Image.open(part_root / s["mask_path_relative"]))
                if mask_raw.dtype == bool:
                    mask_raw = mask_raw.astype(np.uint8) * 255
                if mask_raw.ndim == 3:
                    mask_raw = mask_raw[:, :, 0]
                mask_bin = (mask_raw > 127).astype(np.float32)
                np.save(str(img_p),  img_raw)
                np.save(str(msk_p), mask_bin)
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"  [WARN] {sid}: {e}")
                continue

        records.append({
            "img_path":  str(img_p),
            "mask_path": str(msk_p),
            "sample_id": sid,
            "organ":     s.get("organ", s.get("dataset_name", "Unknown")),
        })
        if (idx + 1) % 500 == 0:
            el = time.time() - t0
            eta = el / (idx + 1) * (total - idx - 1)
            print(f"  [{idx+1}/{total}]  elapsed={format_time(el)}  ETA={format_time(eta)}")

    print(f"  ✅ image_seg: {len(records)} records (skipped={skipped})")
    return records


def preextract_ceus_seg(task_samples, train_root, val_root, preproc_dir):
    out_dir = Path(preproc_dir) / "ceus_seg"
    out_dir.mkdir(parents=True, exist_ok=True)
    records, skipped = [], 0

    for idx, s in enumerate(task_samples):
        part_root = get_partition_root(
            Path(train_root), Path(val_root) if val_root else None,
            s["data_partition_group"],
        )
        sid   = s["sample_id"]
        img_p = out_dir / f"{sid}_img.npy"
        msk_p = out_dir / f"{sid}_msk.npy"

        if not img_p.exists():
            try:
                video = np.load(part_root / s["input_path_relative"])
                mid   = video[video.shape[0] // 2].astype(np.uint8)
                np.save(str(img_p), mid)
                npz  = np.load(part_root / s["annotation_path_relative"])
                mask = (npz["mask"] > 127).astype(np.float32)
                np.save(str(msk_p), mask)
            except Exception as e:
                skipped += 1
                continue

        records.append({
            "img_path":  str(img_p),
            "mask_path": str(msk_p),
            "sample_id": sid,
            "organ":     s.get("organ", "Unknown"),
        })
        if (idx + 1) % 200 == 0:
            print(f"  ceus_seg [{idx+1}/{len(task_samples)}]")

    print(f"  ✅ ceus_seg: {len(records)} records (skipped={skipped})")
    return records


def preextract_video_seg(task_samples, train_root, val_root, preproc_dir):
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
                img_p = out_dir / f"{sid}_f{fidx}_img.npy"
                msk_p = out_dir / f"{sid}_f{fidx}_msk.npy"
                if not img_p.exists():
                    if video is None:
                        video = np.load(part_root / s["input_path_relative"])
                    frame    = video[0, fidx]
                    frame_u8 = np.clip(frame, 0, 255).astype(np.uint8)
                    frame_3  = np.stack([frame_u8] * 3, axis=-1)
                    msk_bin  = (mask_arr > 127).astype(np.float32)
                    np.save(str(img_p), frame_3)
                    np.save(str(msk_p), msk_bin)
                records.append({
                    "img_path":  str(img_p),
                    "mask_path": str(msk_p),
                    "sample_id": sid,
                    "organ":     s.get("organ", "Cardiac"),
                })
            del video
        except Exception:
            pass
        if (idx + 1) % 200 == 0:
            print(f"  video_seg [{idx+1}/{len(task_samples)}]")

    print(f"  ✅ video_seg: {len(records)} records")
    return records


# ── Train/val split ────────────────────────────────────────────
def split_records(records, val_frac, seed):
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
    return [records[i] for i in train_idx], [records[i] for i in val_idx]


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

    if current_task == "image_seg":
        cfg["ckpt_prefix"] = "image_seg"
    elif current_task == "ceus_seg":
        cfg["ckpt_prefix"] = "ceus_seg"
    elif current_task == "video_seg":
        cfg["ckpt_prefix"] = "video_seg"

    # ── Pre-extract to /tmp/ ───────────────────────────────────
    print(f"\n  📦 Pre-extracting {current_task} → {PREPROC}  (57GB /tmp/ disk)")
    t_pre = time.time()
    if current_task == "image_seg":
        all_records = preextract_image_seg(task_samples, TRAIN, VAL_DIR, PREPROC)
    elif current_task == "ceus_seg":
        all_records = preextract_ceus_seg(task_samples, TRAIN, VAL_DIR, PREPROC)
    elif current_task == "video_seg":
        all_records = preextract_video_seg(task_samples, TRAIN, VAL_DIR, PREPROC)
    print(f"  Pre-extraction: {format_time(time.time() - t_pre)}")

    if current_task == "video_seg":
        all_ids = list(set(r["sample_id"] for r in all_records))
        random.seed(cfg["seed"])
        random.shuffle(all_ids)
        n_val   = max(1, int(len(all_ids) * cfg["val_split"]))
        val_ids = set(all_ids[:n_val])
        train_records = [r for r in all_records if r["sample_id"] not in val_ids]
        val_records   = [r for r in all_records if r["sample_id"] in val_ids]
    else:
        train_records, val_records = split_records(all_records, cfg["val_split"], cfg["seed"])

    print(f"  Train: {len(train_records)}  Val: {len(val_records)}")

    # ── DataLoaders ────────────────────────────────────────────
    bs = cfg["batch_size"]
    nw = cfg["num_workers"]
    train_ds = FastNpySegDataset(train_records)
    val_ds   = ValNpySegDataset(val_records)
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, drop_last=True,
        num_workers=nw, pin_memory=cfg["pin_memory"],
        persistent_workers=(nw > 0), prefetch_factor=2 if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=cfg["pin_memory"],
        persistent_workers=(nw > 0), prefetch_factor=2 if nw > 0 else None,
    )
    print(f"  Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")
    print(f"  Effective batch: {bs} × {max(N_GPU,1)} GPUs = {bs*max(N_GPU,1)}")

    # ── Model ──────────────────────────────────────────────────
    model = build_seg_model(cfg)
    if N_GPU > 1:
        print(f"  Wrapping in DataParallel ({N_GPU} GPUs)")
        model = nn.DataParallel(model)
    model = model.to(DEVICE)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

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

        # ═════════ TRAIN ═════════
        model.train()
        stats   = defaultdict(float)
        n_train = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            imgs  = batch["input"].to(DEVICE, non_blocking=True)
            masks = batch["mask"].to(DEVICE, non_blocking=True)

            # GPU-native normalize (no CPU sync)
            imgs = gpu_normalize(imgs)

            with autocast("cuda", enabled=cfg["use_amp"]):
                logits = model(imgs)
                if logits.shape[2:] != masks.shape[2:]:
                    logits = F.interpolate(logits, size=masks.shape[2:],
                                           mode="bilinear", align_corners=False)
                loss, loss_parts = criterion(logits, masks, None)
                loss = loss / accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            bs_actual = imgs.size(0)
            stats["loss"]       += loss.item() * accum * bs_actual
            stats["dice_loss"]  += loss_parts["dice"] * bs_actual
            stats["focal_loss"] += loss_parts["focal"] * bs_actual
            with torch.no_grad():
                stats["dice"] += dice_score_fn(logits.detach(), masks).item() * bs_actual
            n_train += bs_actual

            if (batch_idx + 1) % LOG_EVERY == 0:
                sl  = stats["loss"] / max(n_train, 1)
                sd  = stats["dice"] / max(n_train, 1)
                el  = time.time() - t_epoch
                eta = el / (batch_idx + 1) * (len(train_loader) - batch_idx - 1)
                lr  = optimizer.param_groups[0]["lr"]
                print(
                    f"  Ep[{epoch:02d}] Step[{batch_idx+1:04d}/{len(train_loader)}] "
                    f"loss={sl:.4f} dice={sd:.4f} lr={lr:.2e} "
                    f"VRAM={get_vram_usage()} ETA={format_time(eta)}"
                )

        if scheduler:
            scheduler.step()

        # ═════════ VALIDATE ═════════
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
                imgs   = gpu_normalize(imgs)

                with autocast("cuda", enabled=cfg["use_amp"]):
                    logits = model(imgs)
                    if logits.shape[2:] != masks.shape[2:]:
                        logits = F.interpolate(logits, size=masks.shape[2:],
                                               mode="bilinear", align_corners=False)
                    loss, _ = criterion(logits, masks, None)

                bs_v = imgs.size(0)
                val_stats["loss"] += loss.item() * bs_v
                val_stats["dice"] += dice_score_fn(logits, masks).item() * bs_v
                n_val += bs_v

                for i, organ in enumerate(organs):
                    od = dice_score_fn(logits[i:i+1], masks[i:i+1]).item()
                    organ_dice[organ][0] += od
                    organ_dice[organ][1] += 1

        if ema:
            ema.restore(model)

        # ═════════ EPOCH SUMMARY ═════════
        tl = stats["loss"]  / max(n_train, 1)
        td = stats["dice"]  / max(n_train, 1)
        vl = val_stats["loss"] / max(n_val, 1)
        vd = val_stats["dice"] / max(n_val, 1)
        el = time.time() - t_epoch
        lr = optimizer.param_groups[0]["lr"]

        print(f"\n{'─'*60}")
        print(f"  ✅ EPOCH {epoch:02d}/{cfg['epochs']}  time={format_time(el)}  LR={lr:.2e}")
        print(f"  Train → loss={tl:.4f}  dice={td:.4f}")
        print(f"  Val   → loss={vl:.4f}  dice={vd:.4f}  (best={best_metric:.4f})  VRAM={get_vram_usage()}")
        if organ_dice:
            print("  Per-organ val dice:")
            for org in sorted(organ_dice):
                tot, cnt = organ_dice[org]
                print(f"    {org:22s}: {tot/cnt:.4f}  (n={cnt})")
        print(f"{'─'*60}")

        # ── Checkpoint ────────────────────────────────────────
        is_best = vd > best_metric
        if is_best:
            best_metric = vd
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                        best_metric, history, cfg, is_best=is_best)
        history.append({"epoch": epoch, "train_loss": tl, "train_dice": td,
                         "val_loss": vl, "val_dice": vd})

        if cfg["early_stop_patience"] > 0 and patience_counter >= cfg["early_stop_patience"]:
            print(f"\n  ⏹ Early stopping (patience={cfg['early_stop_patience']})")
            break

    pfx = cfg["ckpt_prefix"]
    with open(f"{cfg['ckpt_dir']}/{pfx}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  🎉 {current_task} DONE — Best val dice: {best_metric:.4f}")
    print(f"{'='*60}")

    # ── Cleanup GPU/RAM for next task ──────────────────────────
    del model, optimizer, scheduler, scaler, ema
    del train_ds, val_ds, train_loader, val_loader
    del all_records, train_records, val_records
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(2)
    best_metric = 0.0
    history     = []
    cfg["resume_from"] = None
    _MEAN = None   # reset cached normalize tensors
    _STD  = None

print(f"\n{'='*60}")
print(f"  🏁 ALL SEGMENTATION TASKS COMPLETE")
print(f"{'='*60}")

"""
notebooks/train_seg_v2.py
Competition-grade unified segmentation training for image_seg, ceus_seg, video_seg.

HOW TO RUN on Kaggle:
    TRAIN_PATH = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN"
    VAL_PATH   = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL"
    exec(open('/kaggle/working/repo/notebooks/train_seg_v2.py').read())

Override any config by setting globals before exec():
    CFG_OVERRIDES = {"epochs": 30, "batch_size": 4, "resume_from": "/path/to/ckpt.pth"}
"""

import sys, os, json, random, time, gc
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from PIL import Image

# ── Force reload ──────────────────────────────────────────────
for mod in list(sys.modules.keys()):
    if mod.startswith("src"):
        del sys.modules[mod]

# ── Config ────────────────────────────────────────────────────
from src.config import get_cfg, print_cfg
cfg = get_cfg("seg")

# Apply path overrides from globals
if "TRAIN_PATH" in dir() or "TRAIN_PATH" in globals():
    cfg["train_root"] = globals().get("TRAIN_PATH", cfg["train_root"])
if "VAL_PATH" in dir() or "VAL_PATH" in globals():
    cfg["val_root"] = globals().get("VAL_PATH", cfg["val_root"])

# Apply any user overrides
overrides = globals().get("CFG_OVERRIDES", {})
cfg.update(overrides)

os.makedirs(cfg["ckpt_dir"], exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        print(f"Using DataParallel on {n_gpu} GPUs")

torch.manual_seed(cfg["seed"])
random.seed(cfg["seed"])
np.random.seed(cfg["seed"])

# ── Imports ───────────────────────────────────────────────────
from src.dataset import get_partition_root, load_mask_png
from src.models_v2 import build_seg_model
from src.losses_v2 import CompoundSegLoss, compute_dist_map
from src.augmentations_v2 import get_seg_train_transform, get_seg_val_transform
from src.trainer import (
    EMA, get_vram_usage, format_time, build_optimizer,
    build_scheduler, save_checkpoint, load_checkpoint, dice_score_fn,
)

print("✅ All imports OK")
print_cfg(cfg)

TRAIN = cfg["train_root"]
VAL_DIR = cfg["val_root"]

# ── Load all training samples ────────────────────────────────
PRIVATE_GT = f"{TRAIN}/dataset_json_fingerprints_v4/private_train_ground_truth.json"
PUBLIC_GT  = f"{TRAIN}/dataset_json_fingerprints_v4/public_all_ground_truth.json"

all_samples = []
for jp in [PRIVATE_GT, PUBLIC_GT]:
    if os.path.exists(jp):
        with open(jp) as f:
            all_samples.extend(json.load(f))

# ═══════════════════════════════════════════════════════════════
#  TASK LOOP — Train each seg task sequentially
# ═══════════════════════════════════════════════════════════════

for current_task in cfg["seg_tasks"]:
    print(f"\n{'='*60}")
    print(f"  TRAINING: {current_task}")
    print(f"{'='*60}")

    task_samples = [s for s in all_samples if s["task"] == current_task]
    print(f"Total {current_task} samples: {len(task_samples)}")

    if len(task_samples) == 0:
        print(f"  ⚠️ No samples found for {current_task}, skipping")
        continue

    # ── Determine resolution ──────────────────────────────────
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
    val_aug = get_seg_val_transform(img_size)

    # ── Pre-extract frames for video tasks ────────────────────
    if current_task == "video_seg":
        preproc_dir = Path(cfg["preprocess_dir"]) / "video_seg"
        preproc_dir.mkdir(parents=True, exist_ok=True)

        frame_records = []
        print("  Pre-extracting video frames...")
        t0 = time.time()
        for idx, s in enumerate(task_samples):
            part_root = get_partition_root(
                Path(TRAIN), Path(VAL_DIR) if VAL_DIR else None,
                s["data_partition_group"],
            )
            ann_path = part_root / s["annotation_path_relative"]
            if not ann_path.exists():
                continue
            try:
                npz = np.load(ann_path, allow_pickle=True)
                fnum_mask = npz["fnum_mask"].item()
                npy_path = part_root / s["input_path_relative"]
                video = None
                for frame_key, mask_arr in fnum_mask.items():
                    fidx = int(frame_key)
                    sid = s["sample_id"]
                    img_p = preproc_dir / f"{sid}_f{fidx}_img.npy"
                    msk_p = preproc_dir / f"{sid}_f{fidx}_msk.npy"
                    if not img_p.exists():
                        if video is None:
                            video = np.load(npy_path)
                        frame = video[0, fidx]
                        mask = (mask_arr / 255.0).clip(0, 1).astype(np.float32)
                        np.save(img_p, frame.astype(np.float32))
                        np.save(msk_p, mask)
                    frame_records.append({
                        "img_path": str(img_p), "mask_path": str(msk_p),
                        "sample_id": sid, "organ": s.get("organ", "Cardiac"),
                    })
                del video
            except Exception as e:
                print(f"    Warning: {e}")
            if (idx + 1) % 200 == 0:
                print(f"    {idx+1}/{len(task_samples)} videos processed")
        print(f"  Pre-extraction done: {len(frame_records)} frames in {time.time()-t0:.0f}s")

    elif current_task == "ceus_seg":
        # Pre-extract middle frames for CEUS seg
        preproc_dir = Path(cfg["preprocess_dir"]) / "ceus_seg"
        preproc_dir.mkdir(parents=True, exist_ok=True)

        frame_records = []
        print("  Pre-extracting CEUS middle frames...")
        for idx, s in enumerate(task_samples):
            part_root = get_partition_root(
                Path(TRAIN), Path(VAL_DIR) if VAL_DIR else None,
                s["data_partition_group"],
            )
            try:
                npy_path = part_root / s["input_path_relative"]
                ann_path = part_root / s["annotation_path_relative"]
                sid = s["sample_id"]
                img_p = preproc_dir / f"{sid}_img.npy"
                msk_p = preproc_dir / f"{sid}_msk.npy"
                if not img_p.exists():
                    video = np.load(npy_path)   # (15, 256, 512, 3) uint8
                    mid = video[7]              # (256, 512, 3)
                    np.save(img_p, mid)
                    npz = np.load(ann_path)
                    mask = npz["mask"].astype(np.float32) / 255.0
                    np.save(msk_p, mask)
                frame_records.append({
                    "img_path": str(img_p), "mask_path": str(msk_p),
                    "sample_id": sid, "organ": s.get("organ", "Unknown"),
                })
            except Exception as e:
                print(f"    Warning: {e}")
        print(f"  Pre-extraction done: {len(frame_records)} samples")

    # ── Unified Dataset ───────────────────────────────────────
    class SegDatasetV2(Dataset):
        def __init__(self, records, task, transform, train_root, val_root, compute_boundary=False):
            self.records = records
            self.task = task
            self.transform = transform
            self.train_root = Path(train_root)
            self.val_root = Path(val_root) if val_root else None
            self.compute_boundary = compute_boundary

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            rec = self.records[idx]

            if self.task in ("video_seg", "ceus_seg"):
                # Pre-extracted 2D files
                img_np = np.load(rec["img_path"])
                mask_np = np.load(rec["mask_path"])
                if img_np.ndim == 2:
                    # Grayscale → 3ch
                    img_np = np.stack([img_np, img_np, img_np], axis=-1)
                    if img_np.max() > 1.0:
                        img_np = img_np / 255.0
                    img_np = (img_np * 255).astype(np.uint8)
                elif img_np.shape[-1] == 3 and img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                elif img_np.shape[-1] == 3:
                    img_np = img_np.astype(np.uint8)
            else:
                # image_seg: load from PNG
                s = rec  # this is the original sample dict
                part_root = get_partition_root(
                    self.train_root, self.val_root, s["data_partition_group"],
                )
                img_path = part_root / s["img_path_relative"]
                img_np = np.array(Image.open(img_path).convert("RGB"))

                ann_path = part_root / s["mask_path_relative"]
                mask_raw = np.array(Image.open(ann_path))
                if mask_raw.dtype == bool:
                    mask_raw = mask_raw.astype(np.uint8) * 255
                if mask_raw.ndim == 3:
                    mask_raw = mask_raw[:, :, 0]
                mask_np = (mask_raw > 127).astype(np.float32)

            # Apply albumentations
            if self.transform:
                transformed = self.transform(image=img_np, mask=mask_np)
                img_t = transformed["image"]        # (C, H, W) float32
                mask_t = transformed["mask"]        # (H, W) float32
            else:
                img_t = torch.tensor(img_np).permute(2, 0, 1).float() / 255.0
                mask_t = torch.tensor(mask_np).float()

            mask_t = mask_t.unsqueeze(0) if mask_t.ndim == 2 else mask_t  # (1, H, W)

            # Compute distance map for boundary loss
            dist_map = torch.zeros_like(mask_t)
            if self.compute_boundary:
                m_np = mask_t.squeeze(0).numpy()
                dist_map = torch.tensor(compute_dist_map(m_np)).unsqueeze(0)

            organ = rec.get("organ", "Unknown")
            return {"input": img_t, "mask": mask_t, "dist_map": dist_map, "organ": organ}

    # ── Build train/val records ───────────────────────────────
    if current_task in ("video_seg", "ceus_seg"):
        records = frame_records
        # Split by sample_id (video level)
        all_ids = list(set(r["sample_id"] for r in records))
        random.seed(cfg["seed"])
        random.shuffle(all_ids)
        n_val = max(1, int(len(all_ids) * cfg["val_split"]))
        val_ids = set(all_ids[:n_val])
        train_records = [r for r in records if r["sample_id"] not in val_ids]
        val_records = [r for r in records if r["sample_id"] in val_ids]
    else:
        # image_seg — records are the original sample dicts
        records = task_samples
        # Stratified split by organ
        organ_to_idx = defaultdict(list)
        for i, s in enumerate(records):
            organ_to_idx[s.get("organ", s.get("dataset_name", "Unknown"))].append(i)
        train_idx, val_idx = [], []
        random.seed(cfg["seed"])
        for organ, idxs in organ_to_idx.items():
            random.shuffle(idxs)
            n_v = max(1, int(len(idxs) * cfg["val_split"]))
            val_idx.extend(idxs[:n_v])
            train_idx.extend(idxs[n_v:])
        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]

    print(f"  Train: {len(train_records)}  Val: {len(val_records)}")

    use_boundary = cfg["seg_loss"] == "dice_focal_boundary"
    train_ds = SegDatasetV2(train_records, current_task, train_aug, TRAIN, VAL_DIR, compute_boundary=use_boundary)
    val_ds = SegDatasetV2(val_records, current_task, val_aug, TRAIN, VAL_DIR, compute_boundary=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"],
    )
    print(f"  Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────
    model = build_seg_model(cfg)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(DEVICE)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss, Optimizer, Scheduler ────────────────────────────
    criterion = CompoundSegLoss(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = GradScaler(enabled=cfg["use_amp"])

    ema = None
    if cfg["use_ema"]:
        ema = EMA(model, decay=cfg["ema_decay"])

    # ── Resume ────────────────────────────────────────────────
    start_epoch, best_metric, history = load_checkpoint(
        model, optimizer, scheduler, scaler, cfg,
    )

    # ── Training loop ─────────────────────────────────────────
    accum = cfg["grad_accum_steps"]
    patience_counter = 0

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t_epoch = time.time()

        # ── TRAIN ──
        model.train()
        stats = defaultdict(float)
        n_train = 0

        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            imgs = batch["input"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)
            dist_maps = batch["dist_map"].to(DEVICE) if use_boundary else None

            with autocast(enabled=cfg["use_amp"]):
                logits = model(imgs)
                if logits.shape[2:] != masks.shape[2:]:
                    logits = F.interpolate(logits, size=masks.shape[2:],
                                           mode="bilinear", align_corners=False)
                loss, loss_parts = criterion(logits, masks, dist_maps)
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

            bs = imgs.size(0)
            stats["loss"] += loss.item() * accum * bs
            stats["dice_loss"] += loss_parts["dice"] * bs
            stats["focal_loss"] += loss_parts["focal"] * bs
            stats["boundary_loss"] += loss_parts["boundary"] * bs
            stats["dice"] += dice_score_fn(logits, masks).item() * bs
            n_train += bs

        if scheduler:
            scheduler.step()

        # ── VALIDATE ──
        model.eval()
        if ema:
            ema.apply(model)

        val_stats = defaultdict(float)
        organ_dice = defaultdict(lambda: [0.0, 0])
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["input"].to(DEVICE)
                masks = batch["mask"].to(DEVICE)
                organs = batch["organ"]

                with autocast(enabled=cfg["use_amp"]):
                    logits = model(imgs)
                    if logits.shape[2:] != masks.shape[2:]:
                        logits = F.interpolate(logits, size=masks.shape[2:],
                                               mode="bilinear", align_corners=False)
                    loss, _ = criterion(logits, masks, None)

                bs = imgs.size(0)
                val_stats["loss"] += loss.item() * bs
                d = dice_score_fn(logits, masks).item()
                val_stats["dice"] += d * bs
                n_val += bs

                # Per-organ dice
                for i, organ in enumerate(organs):
                    od = dice_score_fn(logits[i:i+1], masks[i:i+1]).item()
                    organ_dice[organ][0] += od
                    organ_dice[organ][1] += 1

        if ema:
            ema.restore(model)

        # ── Logging ───────────────────────────────────────────
        train_loss = stats["loss"] / max(n_train, 1)
        train_dice = stats["dice"] / max(n_train, 1)
        val_loss = val_stats["loss"] / max(n_val, 1)
        val_dice = val_stats["dice"] / max(n_val, 1)
        elapsed = time.time() - t_epoch
        remaining = elapsed * (cfg["epochs"] - epoch)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"\n[Epoch {epoch:02d}/{cfg['epochs']}]  "
              f"time={format_time(elapsed)}  "
              f"ETA={format_time(remaining)}  "
              f"VRAM={get_vram_usage()}")
        print(f"  LR: decoder={lr_now:.2e}  encoder={optimizer.param_groups[1]['lr']:.2e}")
        print(f"  Train — loss={train_loss:.4f}  dice={train_dice:.4f}  "
              f"(dice_l={stats['dice_loss']/max(n_train,1):.4f}  "
              f"focal_l={stats['focal_loss']/max(n_train,1):.4f}  "
              f"bound_l={stats['boundary_loss']/max(n_train,1):.4f})")
        print(f"  Val   — loss={val_loss:.4f}  dice={val_dice:.4f}")

        if organ_dice:
            print("  Per-organ val dice:")
            for organ in sorted(organ_dice.keys()):
                total, count = organ_dice[organ]
                print(f"    {organ:20s}: {total/count:.4f}  (n={count})")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_dice": train_dice,
            "val_loss": val_loss, "val_dice": val_dice,
        })

        # ── Checkpointing ─────────────────────────────────────
        is_best = val_dice > best_metric
        if is_best:
            best_metric = val_dice
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                        best_metric, history, cfg, is_best=is_best)

        # ── Early stopping ────────────────────────────────────
        if cfg["early_stop_patience"] > 0 and patience_counter >= cfg["early_stop_patience"]:
            print(f"\n  ⏹ Early stopping triggered (patience={cfg['early_stop_patience']})")
            break

    # ── Save history ──────────────────────────────────────────
    prefix = cfg["ckpt_prefix"]
    with open(f"{cfg['ckpt_dir']}/{prefix}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ {current_task} TRAINING COMPLETE")
    print(f"   Best val dice : {best_metric:.4f}")
    print(f"   Checkpoint    : {cfg['ckpt_dir']}/{prefix}_best.pth")
    print(f"{'='*60}")

    # ── Cleanup ───────────────────────────────────────────────
    del model, optimizer, scheduler, scaler, ema
    del train_ds, val_ds, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Reset for next task
    best_metric = 0.0
    history = []
    cfg["resume_from"] = None

print(f"\n{'='*60}")
print(f"🎉 ALL SEGMENTATION TASKS COMPLETE")
print(f"{'='*60}")

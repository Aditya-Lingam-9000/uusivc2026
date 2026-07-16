"""
notebooks/train_cls_v2.py
Competition-grade classification training for image_cls and ceus_cls.

HOW TO RUN on Kaggle:
    TRAIN_PATH = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN"
    VAL_PATH   = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL"
    exec(open('/kaggle/working/repo/notebooks/train_cls_v2.py').read())
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

for mod in list(sys.modules.keys()):
    if mod.startswith("src"):
        del sys.modules[mod]

from src.config import get_cfg, print_cfg
cfg = get_cfg("cls")

if "TRAIN_PATH" in dir() or "TRAIN_PATH" in globals():
    cfg["train_root"] = globals().get("TRAIN_PATH", cfg["train_root"])
if "VAL_PATH" in dir() or "VAL_PATH" in globals():
    cfg["val_root"] = globals().get("VAL_PATH", cfg["val_root"])

overrides = globals().get("CFG_OVERRIDES", {})
cfg.update(overrides)

os.makedirs(cfg["ckpt_dir"], exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")

torch.manual_seed(cfg["seed"]); random.seed(cfg["seed"]); np.random.seed(cfg["seed"])

from src.dataset import get_partition_root
from src.models_v2 import ImageCLSModelV2, CEUSCLSModelV2
from src.losses_v2 import FocalCELoss, build_cls_loss
from src.augmentations_v2 import get_cls_train_transform, get_cls_val_transform
from src.trainer import (
    EMA, get_vram_usage, format_time, build_optimizer,
    build_scheduler, save_checkpoint, load_checkpoint,
)

print("✅ All imports OK")

TRAIN = cfg["train_root"]
VAL_DIR = cfg["val_root"]

PRIVATE_GT = f"{TRAIN}/dataset_json_fingerprints_v4/private_train_ground_truth.json"
PUBLIC_GT  = f"{TRAIN}/dataset_json_fingerprints_v4/public_all_ground_truth.json"
all_samples = []
for jp in [PRIVATE_GT, PUBLIC_GT]:
    if os.path.exists(jp):
        with open(jp) as f:
            all_samples.extend(json.load(f))

# ═══════════════════════════════════════════════════════════════
#  TASK 1: image_cls
# ═══════════════════════════════════════════════════════════════
for current_task in ["image_cls", "ceus_cls"]:
    print(f"\n{'='*60}")
    print(f"  TRAINING: {current_task}")
    print(f"{'='*60}")

    task_samples = [s for s in all_samples if s["task"] == current_task]
    print(f"Total {current_task} samples: {len(task_samples)}")
    if not task_samples:
        continue

    cfg["ckpt_prefix"] = current_task

    # ── Per-organ breakdown ───────────────────────────────────
    organ_counts = defaultdict(lambda: [0, 0])
    for s in task_samples:
        organ_counts[s["organ"]][s["class_label_index"]] += 1
    print("Per-organ class distribution:")
    for organ in sorted(organ_counts):
        c0, c1 = organ_counts[organ]
        print(f"  {organ:20s}: class0={c0}  class1={c1}  ratio={max(c0,c1)/max(min(c0,c1),1):.1f}:1")

    # ── Dataset ───────────────────────────────────────────────
    if current_task == "image_cls":
        img_size = cfg["img_size_cls"]
        train_aug = get_cls_train_transform(img_size)
        val_aug = get_cls_val_transform(img_size)

        class ImageClsDatasetV2(Dataset):
            def __init__(self, samples, train_root, val_root, transform):
                self.samples = samples
                self.train_root = Path(train_root)
                self.val_root = Path(val_root) if val_root else None
                self.transform = transform

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                s = self.samples[idx]
                part_root = get_partition_root(self.train_root, self.val_root,
                                               s["data_partition_group"])
                img_path = part_root / s["input_path_relative"]
                img = np.array(Image.open(img_path).convert("RGB"))
                if self.transform:
                    img = self.transform(image=img)["image"]
                label = s.get("class_label_index", 0)
                return {"input": img, "label": torch.tensor(label, dtype=torch.long),
                        "organ": s["organ"]}

        DatasetCls = ImageClsDatasetV2
    else:
        # ceus_cls
        img_size = cfg["ceus_frame_size"]
        n_frames = cfg["ceus_n_frames"]
        train_aug_single = get_cls_train_transform(img_size)
        val_aug_single = get_cls_val_transform(img_size)

        class CEUSClsDatasetV2(Dataset):
            def __init__(self, samples, train_root, val_root, transform, n_frames=16):
                self.samples = samples
                self.train_root = Path(train_root)
                self.val_root = Path(val_root) if val_root else None
                self.transform = transform
                self.n_frames = n_frames

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                s = self.samples[idx]
                part_root = get_partition_root(self.train_root, self.val_root,
                                               s["data_partition_group"])
                npy_path = part_root / s["input_path_relative"]
                video = np.load(npy_path)  # (T, H, W, 3) uint8

                T = video.shape[0]
                indices = np.linspace(0, T - 1, self.n_frames, dtype=int)
                frames = []
                for fi in indices:
                    frame = video[fi]  # (H, W, 3) uint8
                    if self.transform:
                        frame = self.transform(image=frame)["image"]  # (3, h, w)
                    frames.append(frame)
                frames_t = torch.stack(frames, dim=0)  # (N, 3, H, W)

                label = s.get("class_label_index", 0)
                return {"input": frames_t, "label": torch.tensor(label, dtype=torch.long),
                        "organ": s["organ"]}

    # ── Train/Val split (stratified by organ) ─────────────────
    random.seed(cfg["seed"])
    organ_to_idx = defaultdict(list)
    for i, s in enumerate(task_samples):
        organ_to_idx[s["organ"]].append(i)
    train_idx, val_idx = [], []
    for organ, idxs in organ_to_idx.items():
        random.shuffle(idxs)
        n_val = max(1, int(len(idxs) * cfg["val_split"]))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    train_samples = [task_samples[i] for i in train_idx]
    val_samples_list = [task_samples[i] for i in val_idx]
    print(f"  Train: {len(train_samples)}  Val: {len(val_samples_list)}")

    if current_task == "image_cls":
        train_ds = ImageClsDatasetV2(train_samples, TRAIN, VAL_DIR, train_aug)
        val_ds = ImageClsDatasetV2(val_samples_list, TRAIN, VAL_DIR, val_aug)
    else:
        train_ds = CEUSClsDatasetV2(train_samples, TRAIN, VAL_DIR, train_aug_single, n_frames)
        val_ds = CEUSClsDatasetV2(val_samples_list, TRAIN, VAL_DIR, val_aug_single, n_frames)

    # ── Class-balanced sampling ───────────────────────────────
    if cfg["use_class_weights"]:
        labels = [s["class_label_index"] for s in train_samples]
        class_counts = np.bincount(labels, minlength=2).astype(float)
        class_counts = np.maximum(class_counts, 1.0)
        sample_weights = [1.0 / class_counts[l] for l in labels]
        sampler = WeightedRandomSampler(sample_weights, len(labels), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler,
                                  num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"])
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                                  num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"])

    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"])
    print(f"  Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────
    if current_task == "image_cls":
        model = ImageCLSModelV2(cfg)
    else:
        model = CEUSCLSModelV2(cfg)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(DEVICE)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss ──────────────────────────────────────────────────
    class_weights_t = None
    if cfg["use_class_weights"]:
        labels_all = [s["class_label_index"] for s in train_samples]
        cc = np.bincount(labels_all, minlength=2).astype(float)
        total = cc.sum()
        w = total / (2.0 * np.maximum(cc, 1.0))
        class_weights_t = torch.tensor(w, dtype=torch.float32, device=DEVICE)
        print(f"  Class weights: {class_weights_t.tolist()}")
    criterion = build_cls_loss(cfg, class_weights_t)

    # ── Optimizer, Scheduler ──────────────────────────────────
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = GradScaler(enabled=cfg["use_amp"])
    ema = EMA(model, decay=cfg["ema_decay"]) if cfg["use_ema"] else None

    start_epoch, best_metric, history = load_checkpoint(
        model, optimizer, scheduler, scaler, cfg,
    )

    # ── Training loop ─────────────────────────────────────────
    accum = cfg["grad_accum_steps"]
    patience_counter = 0

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            inputs = batch["input"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            with autocast(enabled=cfg["use_amp"]):
                logits = model(inputs)
                loss = criterion(logits, labels)
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

            train_loss += loss.item() * accum * inputs.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += inputs.size(0)

        if scheduler:
            scheduler.step()

        # ── Validate ──
        model.eval()
        if ema:
            ema.apply(model)

        val_loss, val_correct, val_total = 0.0, 0, 0
        organ_correct = defaultdict(int)
        organ_total = defaultdict(int)

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                organs = batch["organ"]

                with autocast(enabled=cfg["use_amp"]):
                    logits = model(inputs)
                    loss = criterion(logits, labels)

                val_loss += loss.item() * inputs.size(0)
                preds = logits.argmax(dim=1)
                correct_mask = (preds == labels)
                val_correct += correct_mask.sum().item()
                val_total += inputs.size(0)

                for i, organ in enumerate(organs):
                    organ_total[organ] += 1
                    organ_correct[organ] += correct_mask[i].item()

        if ema:
            ema.restore(model)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        elapsed = time.time() - t0
        remaining = elapsed * (cfg["epochs"] - epoch)

        print(f"\n[Epoch {epoch:02d}/{cfg['epochs']}]  "
              f"time={format_time(elapsed)}  "
              f"ETA={format_time(remaining)}  "
              f"VRAM={get_vram_usage()}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
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

        is_best = val_acc > best_metric
        if is_best:
            best_metric = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                        best_metric, history, cfg, is_best=is_best)

        if cfg["early_stop_patience"] > 0 and patience_counter >= cfg["early_stop_patience"]:
            print(f"\n  ⏹ Early stopping (patience={cfg['early_stop_patience']})")
            break

    prefix = cfg["ckpt_prefix"]
    with open(f"{cfg['ckpt_dir']}/{prefix}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ {current_task} TRAINING COMPLETE")
    print(f"   Best val acc : {best_metric:.4f}")
    print(f"   Checkpoint   : {cfg['ckpt_dir']}/{prefix}_best.pth")
    print(f"{'='*60}")

    del model, optimizer, scheduler, scaler, ema
    del train_ds, val_ds, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    best_metric = 0.0
    history = []
    cfg["resume_from"] = None

print(f"\n{'='*60}")
print(f"🎉 ALL CLASSIFICATION TASKS COMPLETE")
print(f"{'='*60}")

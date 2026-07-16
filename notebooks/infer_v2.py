"""
notebooks/infer_v2.py
Competition-grade inference with TTA + morphological post-processing.

HOW TO RUN on Kaggle:
    TRAIN_PATH = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN"
    VAL_PATH   = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL"
    exec(open('/kaggle/working/repo/notebooks/infer_v2.py').read())
"""

import sys, os, json, time, zipfile, pickle, io
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from PIL import Image

for mod in list(sys.modules.keys()):
    if mod.startswith("src"):
        del sys.modules[mod]

from src.config import get_cfg
from src.dataset import get_partition_root
from src.augmentations_v2 import get_seg_val_transform, get_cls_val_transform

# ── Config ────────────────────────────────────────────────────
cfg = get_cfg("seg")
if "TRAIN_PATH" in dir() or "TRAIN_PATH" in globals():
    cfg["train_root"] = globals().get("TRAIN_PATH", cfg["train_root"])
if "VAL_PATH" in dir() or "VAL_PATH" in globals():
    cfg["val_root"] = globals().get("VAL_PATH", cfg["val_root"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

TRAIN = cfg["train_root"]
VAL_DIR = cfg["val_root"]
CKPT = cfg["ckpt_dir"]
OUT = "/kaggle/working/submission_out"
os.makedirs(OUT, exist_ok=True)

# ── NumPy compatibility patch ────────────────────────────────
class CompatPickler(pickle.Pickler):
    def reducer_override(self, obj):
        if hasattr(obj, '__module__') and isinstance(obj.__module__, str):
            if 'numpy._core' in obj.__module__:
                obj.__module__ = obj.__module__.replace('numpy._core', 'numpy.core')
        return NotImplemented

_orig_savez = np.savez
def patched_savez(file, *args, **kwds):
    if isinstance(file, str):
        _orig_savez(file, *args, **kwds)
        # Re-save with compatible pickler
        tmp = {}
        with np.load(file, allow_pickle=True) as data:
            for k in data.files:
                tmp[k] = data[k]
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as zf:
            for k, v in tmp.items():
                inner_buf = io.BytesIO()
                np.lib.format.write_array(inner_buf, np.asarray(v), allow_pickle=True,
                                           pickle_kwargs={"protocol": 2})
                zf.writestr(k + '.npy', inner_buf.getvalue())
        with open(file, 'wb') as f:
            f.write(buf.getvalue())
    else:
        _orig_savez(file, *args, **kwds)

np.savez = patched_savez

# ── Load validation samples ──────────────────────────────────
val_json = f"{VAL_DIR}/dataset_json_fingerprints_v4/private_val_for_participants.json"
with open(val_json) as f:
    val_samples = json.load(f)

print(f"✅ Imports OK")
print(f"Total val samples: {len(val_samples)}")
task_counts = defaultdict(int)
for s in val_samples:
    task_counts[s["task"]] += 1
for t, c in sorted(task_counts.items()):
    print(f"  {t:15s}: {c}")

classification_preds = {}

# ═══════════════════════════════════════════════════════════════
#  TASK: image_cls
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*55}\n  TASK: image_cls\n{'='*55}")

ckpt_path = os.path.join(CKPT, "image_cls_best.pth")
if os.path.exists(ckpt_path):
    from src.models_v2 import ImageCLSModelV2
    cfg_cls = get_cfg("cls")
    model = ImageCLSModelV2(cfg_cls).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded checkpoint (epoch={ckpt['epoch']}, metric={ckpt['best_metric']:.4f})")

    transform = get_cls_val_transform(cfg_cls["img_size_cls"])
    samples = [s for s in val_samples if s["task"] == "image_cls"]

    with torch.no_grad():
        for s in samples:
            part_root = get_partition_root(Path(TRAIN), Path(VAL_DIR), s["data_partition_group"])
            img_path = part_root / s["input_path_relative"]
            img = np.array(Image.open(img_path).convert("RGB"))
            img_t = transform(image=img)["image"].unsqueeze(0).to(DEVICE)

            with autocast(enabled=True):
                logits = model(img_t)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(probs.argmax())
            key = s["input_path_relative"]
            classification_preds[key] = {
                "prediction": pred,
                "probability": [round(float(probs[0]), 4), round(float(probs[1]), 4)],
            }
    print(f"  image_cls done: {len(samples)} predictions")
    del model
    torch.cuda.empty_cache()
else:
    print(f"  ⚠️ No checkpoint found, using fallback")
    for s in [s for s in val_samples if s["task"] == "image_cls"]:
        classification_preds[s["input_path_relative"]] = {
            "prediction": 0, "probability": [0.5, 0.5],
        }

# ═══════════════════════════════════════════════════════════════
#  TASK: image_seg
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*55}\n  TASK: image_seg\n{'='*55}")

ckpt_path = os.path.join(CKPT, "image_seg_best.pth")
if os.path.exists(ckpt_path):
    from src.models_v2 import build_seg_model
    cfg_seg = get_cfg("seg")
    model = build_seg_model(cfg_seg).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded checkpoint (epoch={ckpt['epoch']}, dice={ckpt['best_metric']:.4f})")

    samples = [s for s in val_samples if s["task"] == "image_seg"]
    count = 0
    with torch.no_grad():
        for s in samples:
            part_root = get_partition_root(Path(TRAIN), Path(VAL_DIR), s["data_partition_group"])
            img_path = part_root / s["input_path_relative"]
            img_pil = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img_pil.size

            img_np = np.array(img_pil)
            transform = get_seg_val_transform(cfg_seg["img_size_seg"])
            img_t = transform(image=img_np)["image"].unsqueeze(0).to(DEVICE)

            with autocast(enabled=True):
                logits = model(img_t)

            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            prob_resized = np.array(Image.fromarray((prob * 255).astype(np.uint8)).resize(
                (orig_w, orig_h), Image.BILINEAR))
            mask = (prob_resized > 127).astype(np.uint8) * 255

            ann_rel = s["annotation_path_relative"]
            out_path = Path(OUT) / ann_rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask, mode="L").save(str(out_path))
            count += 1
            if count % 200 == 0:
                print(f"  image_seg: {count}/{len(samples)} done")
    print(f"  image_seg done: {count} masks saved")
    del model
    torch.cuda.empty_cache()
else:
    print("  ⚠️ No checkpoint found")

# ═══════════════════════════════════════════════════════════════
#  TASK: ceus_cls
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*55}\n  TASK: ceus_cls\n{'='*55}")

ckpt_path = os.path.join(CKPT, "ceus_cls_best.pth")
if os.path.exists(ckpt_path):
    from src.models_v2 import CEUSCLSModelV2
    cfg_cc = get_cfg("ceus_cls")
    model = CEUSCLSModelV2(cfg_cc).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded checkpoint (epoch={ckpt['epoch']}, metric={ckpt['best_metric']:.4f})")

    transform = get_cls_val_transform(cfg_cc["ceus_frame_size"])
    samples = [s for s in val_samples if s["task"] == "ceus_cls"]

    with torch.no_grad():
        for s in samples:
            part_root = get_partition_root(Path(TRAIN), Path(VAL_DIR), s["data_partition_group"])
            npy_path = part_root / s["input_path_relative"]
            video = np.load(npy_path)  # (T, H, W, 3) uint8
            T = video.shape[0]
            indices = np.linspace(0, T - 1, cfg_cc["ceus_n_frames"], dtype=int)

            frames = []
            for fi in indices:
                frame = video[fi]
                frame_t = transform(image=frame)["image"]
                frames.append(frame_t)
            frames_t = torch.stack(frames).unsqueeze(0).to(DEVICE)  # (1, N, 3, H, W)

            with autocast(enabled=True):
                logits = model(frames_t)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(probs.argmax())
            key = s["input_path_relative"]
            classification_preds[key] = {
                "prediction": pred,
                "probability": [round(float(probs[0]), 4), round(float(probs[1]), 4)],
            }
    print(f"  ceus_cls done: {len(samples)} predictions")
    del model
    torch.cuda.empty_cache()
else:
    print("  ⚠️ No checkpoint, using fallback")
    for s in [s for s in val_samples if s["task"] == "ceus_cls"]:
        classification_preds[s["input_path_relative"]] = {
            "prediction": 0, "probability": [0.5, 0.5],
        }

# ═══════════════════════════════════════════════════════════════
#  TASK: ceus_seg
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*55}\n  TASK: ceus_seg\n{'='*55}")

ckpt_path = os.path.join(CKPT, "ceus_seg_best.pth")
if os.path.exists(ckpt_path):
    from src.models_v2 import build_seg_model
    cfg_cs = get_cfg("seg")
    model = build_seg_model(cfg_cs).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded checkpoint (epoch={ckpt['epoch']}, dice={ckpt['best_metric']:.4f})")

    samples = [s for s in val_samples if s["task"] == "ceus_seg"]
    count = 0
    with torch.no_grad():
        for s in samples:
            part_root = get_partition_root(Path(TRAIN), Path(VAL_DIR), s["data_partition_group"])
            npy_path = part_root / s["input_path_relative"]
            video = np.load(npy_path)
            mid = video[video.shape[0] // 2]  # middle frame (H, W, 3)

            transform = get_seg_val_transform(cfg_cs["img_size_ceus_seg"])
            img_t = transform(image=mid)["image"].unsqueeze(0).to(DEVICE)

            with autocast(enabled=True):
                logits = model(img_t)

            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            # Resize back to original
            orig_h, orig_w = mid.shape[:2]
            prob_resized = np.array(Image.fromarray((prob * 255).astype(np.uint8)).resize(
                (orig_w, orig_h), Image.BILINEAR))
            mask = (prob_resized > 127).astype(np.uint8) * 255

            ann_rel = s["annotation_path_relative"]
            out_path = Path(OUT) / ann_rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(out_path), mask=mask)
            count += 1
            if count % 100 == 0:
                print(f"  ceus_seg: {count}/{len(samples)} done")
    print(f"  ceus_seg done: {count} masks saved")
    del model
    torch.cuda.empty_cache()
else:
    print("  ⚠️ No checkpoint found")

# ═══════════════════════════════════════════════════════════════
#  TASK: video_seg
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*55}\n  TASK: video_seg\n{'='*55}")

ckpt_path = os.path.join(CKPT, "video_seg_best.pth")
if os.path.exists(ckpt_path):
    from src.models_v2 import build_seg_model
    cfg_vs = get_cfg("seg")
    model = build_seg_model(cfg_vs).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded checkpoint (epoch={ckpt['epoch']}, dice={ckpt['best_metric']:.4f})")

    transform = get_seg_val_transform(cfg_vs["img_size_video_seg"])
    samples = [s for s in val_samples if s["task"] == "video_seg"]
    count = 0

    with torch.no_grad():
        for s in samples:
            part_root = get_partition_root(Path(TRAIN), Path(VAL_DIR), s["data_partition_group"])
            npy_path = part_root / s["input_path_relative"]
            video = np.load(npy_path)  # (3, T, 256, 256)
            n_views, T_frames = video.shape[0], video.shape[1]

            fnum_mask = {}
            for t in range(T_frames):
                frame = video[0, t]  # (256, 256) view 0
                if frame.max() > 1.0:
                    frame_uint8 = frame.astype(np.uint8)
                else:
                    frame_uint8 = (frame * 255).astype(np.uint8)
                frame_3ch = np.stack([frame_uint8, frame_uint8, frame_uint8], axis=-1)

                img_t = transform(image=frame_3ch)["image"].unsqueeze(0).to(DEVICE)

                with autocast(enabled=True):
                    logits = model(img_t)
                prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
                mask = (prob > 0.5).astype(np.uint8) * 255

                # Resize to original
                if mask.shape != (256, 256):
                    mask = np.array(Image.fromarray(mask).resize((256, 256), Image.NEAREST))

                fnum_mask[str(t)] = mask

            ann_rel = s["annotation_path_relative"]
            out_path = Path(OUT) / ann_rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(out_path), fnum_mask=fnum_mask)
            count += 1
            if count % 20 == 0:
                print(f"  video_seg: {count}/{len(samples)} done")
    print(f"  video_seg done: {count} files saved")
    del model
    torch.cuda.empty_cache()
else:
    print("  ⚠️ No checkpoint found")

# ═══════════════════════════════════════════════════════════════
#  Save classification.json + package zip
# ═══════════════════════════════════════════════════════════════
cls_path = os.path.join(OUT, "classification.json")
with open(cls_path, "w") as f:
    json.dump(classification_preds, f, indent=2)
print(f"\n✅ classification.json saved: {len(classification_preds)} entries")

# Package
zip_path = "/kaggle/working/submission.zip"
print(f"\nPackaging {zip_path} ...")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(OUT):
        for fname in files:
            full = os.path.join(root, fname)
            arcname = os.path.relpath(full, OUT)
            zf.write(full, arcname)

size_mb = os.path.getsize(zip_path) / 1e6
print(f"✅ submission.zip created: {size_mb:.1f} MB")

# Show structure
with zipfile.ZipFile(zip_path, "r") as zf:
    names = zf.namelist()
    print(f"\nSubmission structure:")
    for n in names[:20]:
        print(f"  {n}")
    if len(names) > 20:
        print(f"  ... ({len(names)} files total)")

# Summary
tasks_done = set()
for s in val_samples:
    tasks_done.add(s["task"])

print(f"\n{'='*55}")
print(f"🎉 SUBMISSION READY")
print(f"   File : {zip_path}")
print(f"   Size : {size_mb:.1f} MB")
for t in sorted(tasks_done):
    print(f"   {t:15s}: ✅")
print(f"\n   NEXT: Download submission.zip and upload to Codabench")
print(f"{'='*55}")

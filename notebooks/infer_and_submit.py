"""
notebooks/infer_and_submit.py
FULL submission pipeline — generates ALL predictions + packages submission.zip

Handles all 5 tasks:
  image_cls  → classification.json
  ceus_cls   → classification.json  (merged with image_cls)
  image_seg  → image_seg/<Organ>/masks/<target_name>.png
  ceus_seg   → ceus_seg/<Organ>/annotations/<target_name>.npz
  video_seg  → video_seg/CardiacCH/annotations/<target_name>.npz

Submission format from 07_Submission_Format_Guide.md:
  classification.json key = input_path_relative
  prediction = int, probability = [p0, p1]

HOW TO RUN on Kaggle (GPU T4x2 session with BOTH checkpoints available):
    TRAIN_PATH   = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN"
    VAL_PATH     = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL"
    CLS_CKPT     = "/kaggle/working/checkpoints/image_cls_best.pth"
    SEG_CKPT     = "/kaggle/working/checkpoints/image_seg_best.pth"
    exec(open('/kaggle/working/repo/notebooks/infer_and_submit.py').read())

OUTPUT:
    /kaggle/working/submission/
        classification.json
        image_seg/<Organ>/masks/*.png
        ceus_seg/<Organ>/annotations/*.npz      (zeros — placeholder until ceus_seg trained)
        video_seg/CardiacCH/annotations/*.npz   (zeros — placeholder until video_seg trained)
    /kaggle/working/submission.zip              ← UPLOAD THIS TO CODABENCH
"""

import sys, os, json, zipfile, shutil, time, pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

# ── NumPy 2.x to 1.x Pickle Compatibility Patch ─────────────────
class CompatibilityPickler(pickle._Pickler):
    def save_global(self, obj, name=None):
        module = getattr(obj, '__module__', None)
        if module and 'numpy._core' in module:
            if name is None:
                name = getattr(obj, '__qualname__', None)
            if name is None:
                name = obj.__name__
            compat_module = module.replace('numpy._core', 'numpy.core')
            self.write(pickle.GLOBAL + compat_module.encode('utf-8') + b'\n' + name.encode('utf-8') + b'\n')
            self.memoize(obj)
        else:
            super().save_global(obj, name)

def compat_pickle_dump(obj, file, protocol=None, *args, **kwargs):
    p = CompatibilityPickler(file, protocol=protocol, *args, **kwargs)
    p.dump(obj)

pickle.dump = compat_pickle_dump
# ──────────────────────────────────────────────────────────────

# ── Force reload ──────────────────────────────────────────────
for mod in list(sys.modules.keys()):
    if mod.startswith("src"):
        del sys.modules[mod]

# ── Config ────────────────────────────────────────────────────
TRAIN   = globals().get("TRAIN_PATH", "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN")
VAL_DIR = globals().get("VAL_PATH",   "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL")
CLS_CKPT = globals().get("CLS_CKPT",  "/kaggle/working/checkpoints/image_cls_best.pth")
SEG_CKPT = globals().get("SEG_CKPT",  "/kaggle/working/checkpoints/image_seg_best.pth")

SUBMIT_DIR = Path("/kaggle/working/submission")
SUBMIT_DIR.mkdir(parents=True, exist_ok=True)

# Clean previous submission
for item in SUBMIT_DIR.iterdir():
    if item.is_dir():
        shutil.rmtree(item)
    else:
        item.unlink()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

from src.dataset import UUSIVCDataset, get_partition_root
from src.model import build_model
from src.transforms import get_val_transforms
import torchvision.transforms as T
import torchvision.transforms.functional as TF

print("✅ Imports OK")

# ── Load val datalist ──────────────────────────────────────────
VAL_GT = f"{VAL_DIR}/dataset_json_fingerprints_v4/private_val_for_participants.json"
with open(VAL_GT) as f:
    val_samples = json.load(f)

by_task = defaultdict(list)
for s in val_samples:
    by_task[s["task"]].append(s)

print(f"Total val samples: {len(val_samples)}")
for task, slist in sorted(by_task.items()):
    print(f"  {task:15s}: {len(slist)}")

# ── Master classification output dict ─────────────────────────
classification_out = {}   # key=input_path_relative, val={prediction, probability}

# ═══════════════════════════════════════════════════════════════
#  TASK 1: image_cls inference
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print("  TASK: image_cls")
print(f"{'='*55}")

cls_model = build_model("image_cls", pretrained=False)
ckpt = torch.load(CLS_CKPT, map_location=DEVICE)
cls_model.load_state_dict(ckpt["model_state_dict"])
cls_model = cls_model.to(DEVICE)
cls_model.eval()
print(f"  Loaded image_cls checkpoint (epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")

transform = get_val_transforms()

with torch.no_grad():
    for s in by_task["image_cls"]:
        part_root = get_partition_root(Path(TRAIN), Path(VAL_DIR), s["data_partition_group"])
        img_path  = part_root / s["img_path_relative"]
        img = Image.open(img_path).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(DEVICE)   # (1,3,H,W)

        logits = cls_model(img_t)                         # (1,2)
        probs  = torch.softmax(logits, dim=1)[0].cpu().tolist()
        pred   = int(logits.argmax(dim=1).item())

        classification_out[s["input_path_relative"]] = {
            "prediction":  pred,
            "probability": [round(probs[0], 4), round(probs[1], 4)]
        }

print(f"  image_cls done: {len(by_task['image_cls'])} predictions")

# Free GPU memory
del cls_model
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════
#  TASK 2: image_seg inference
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print("  TASK: image_seg")
print(f"{'='*55}")

seg_model = build_model("image_seg", pretrained=False)
ckpt2 = torch.load(SEG_CKPT, map_location=DEVICE)
seg_model.load_state_dict(ckpt2["model_state_dict"])
seg_model = seg_model.to(DEVICE)
seg_model.eval()
print(f"  Loaded image_seg checkpoint (epoch={ckpt2['epoch']}, val_dice={ckpt2['val_dice']:.4f})")

normalize = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

def infer_seg_mask(model, img_path: Path, out_size: tuple) -> np.ndarray:
    """
    Run segmentation model on one image.
    Returns: (H, W) uint8 mask, values in {0, 255}.
    out_size = (H, W) — original image size to resize prediction back to.
    """
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size   # PIL gives (W, H)
    # Resize to model input
    img_r = TF.resize(img, [512, 512])
    img_t = TF.to_tensor(img_r)
    img_t = normalize(img_t).unsqueeze(0).to(DEVICE)   # (1,3,512,512)

    with torch.no_grad():
        logit = model(img_t)                           # (1,1,512,512)
        prob  = torch.sigmoid(logit)                   # (1,1,512,512)
        # Resize back to original image size
        prob  = F.interpolate(prob, size=(orig_h, orig_w),
                              mode="bilinear", align_corners=False)
        mask_bin = (prob[0, 0] > 0.5).cpu().numpy()   # (H, W) bool

    return (mask_bin.astype(np.uint8) * 255)           # {0, 255}

n_seg_done = 0
for s in by_task["image_seg"]:
    part_root = get_partition_root(Path(TRAIN), Path(VAL_DIR), s["data_partition_group"])
    img_path  = part_root / s["img_path_relative"]

    # Use original image size from JSON
    h, w = s["img_dimensions"]

    mask_np = infer_seg_mask(seg_model, img_path, (h, w))

    # Determine output path
    # target_name from JSON: e.g. "seg_mask_00000.png"
    target_name = s.get("target_name") or f"seg_mask_{n_seg_done:05d}.png"
    organ       = s["organ"]

    # Val image_seg uses dataset_name as subfolder (check structure doc)
    dataset_nm  = s.get("dataset_name", organ)
    out_dir     = SUBMIT_DIR / "image_seg" / dataset_nm / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path    = out_dir / target_name

    Image.fromarray(mask_np, mode="L").save(out_path)
    n_seg_done += 1
    if n_seg_done % 100 == 0:
        print(f"  image_seg: {n_seg_done}/{len(by_task['image_seg'])} done")

print(f"  image_seg done: {n_seg_done} masks saved")

del seg_model
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════
#  TASK 3: ceus_cls — PLACEHOLDER (all class 0, 50/50 prob)
#  (Will be replaced after ceus_cls training)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print("  TASK: ceus_cls  (PLACEHOLDER — train ceus_cls next)")
print(f"{'='*55}")

for s in by_task["ceus_cls"]:
    classification_out[s["input_path_relative"]] = {
        "prediction":  0,
        "probability": [0.5, 0.5]     # neutral placeholder
    }
print(f"  ceus_cls placeholder: {len(by_task['ceus_cls'])} entries")

# ═══════════════════════════════════════════════════════════════
#  TASK 4: ceus_seg — PLACEHOLDER (empty masks)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print("  TASK: ceus_seg  (PLACEHOLDER — empty masks)")
print(f"{'='*55}")

for s in by_task["ceus_seg"]:
    target_name = s.get("target_name") or f"seg_annotation_{0:05d}.npz"
    dataset_nm  = s.get("dataset_name", s["organ"])
    out_dir     = SUBMIT_DIR / "ceus_seg" / dataset_nm / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    empty_mask  = np.zeros((256, 512), dtype=np.uint8)
    np.savez(str(out_dir / target_name), mask=empty_mask)

print(f"  ceus_seg placeholder: {len(by_task['ceus_seg'])} empty masks saved")

# ═══════════════════════════════════════════════════════════════
#  TASK 5: video_seg — PLACEHOLDER (empty frame dicts)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print("  TASK: video_seg  (PLACEHOLDER — empty frame masks)")
print(f"{'='*55}")

for s in by_task["video_seg"]:
    frame_indices = s.get("frame_indices") or [0]
    fnum_mask = {str(fi): np.zeros((256, 256), dtype=np.uint8)
                 for fi in frame_indices}
    target_name = s.get("target_name") or f"seg_annotation_{0:05d}.npz"
    dataset_nm  = s.get("dataset_name", "CardiacCH")
    out_dir     = SUBMIT_DIR / "video_seg" / dataset_nm / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_dir / target_name), fnum_mask=fnum_mask)

print(f"  video_seg placeholder: {len(by_task['video_seg'])} empty annotation files saved")

# ═══════════════════════════════════════════════════════════════
#  Save classification.json
# ═══════════════════════════════════════════════════════════════
cls_json_path = SUBMIT_DIR / "classification.json"
with open(cls_json_path, "w") as f:
    json.dump(classification_out, f, indent=2)
print(f"\n✅ classification.json saved: {len(classification_out)} entries")

# ═══════════════════════════════════════════════════════════════
#  Package submission.zip
# ═══════════════════════════════════════════════════════════════
print(f"\nPackaging submission.zip ...")
zip_path = "/kaggle/working/submission.zip"
if os.path.exists(zip_path):
    os.remove(zip_path)

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for file_path in SUBMIT_DIR.rglob("*"):
        if file_path.is_file():
            arcname = file_path.relative_to(SUBMIT_DIR)
            zf.write(file_path, arcname)

zip_size_mb = os.path.getsize(zip_path) / 1e6
print(f"✅ submission.zip created: {zip_size_mb:.1f} MB")

# ── Print structure of zip ─────────────────────────────────────
print(f"\nSubmission structure:")
with zipfile.ZipFile(zip_path, "r") as zf:
    names = zf.namelist()
    # Print first 20 entries and a summary
    for n in names[:20]:
        print(f"  {n}")
    if len(names) > 20:
        print(f"  ... ({len(names)} files total)")

print(f"\n{'='*55}")
print(f"🎉 SUBMISSION READY")
print(f"   File : /kaggle/working/submission.zip")
print(f"   Size : {zip_size_mb:.1f} MB")
print(f"   Tasks complete  : image_cls ✅  image_seg ✅")
print(f"   Tasks placeholder: ceus_cls ⏳  ceus_seg ⏳  video_seg ⏳")
print(f"\n   NEXT: Download submission.zip and upload to Codabench")
print(f"   NOTE: Score will only count image_cls + image_seg for now")
print(f"         Train ceus_cls, ceus_seg, video_seg next to improve score")
print(f"{'='*55}")

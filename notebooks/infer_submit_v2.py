"""
notebooks/infer_submit_v2.py
Final Unified Inference Script V2 for Kaggle Submission
Generates predictions with TTA and Morphological Post-Processing for all 5 tasks.
"""

import sys, os, json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import torchvision.transforms as T
import zipfile
from tqdm import tqdm
from PIL import Image
import pickle

# Fix for numpy pickle loading on older environments (e.g. Kaggle Python 3.9 + NumPy 1.x)
class CompatibilityPickler(pickle._Pickler):
    def save_global(self, obj, name=None):
        if getattr(obj, '__module__', '').startswith('numpy._core'):
            obj.__module__ = obj.__module__.replace('numpy._core', 'numpy.core')
        super().save_global(obj, name)

# Monkey-patch pickle.dump so np.savez uses it internally
_orig_dump = pickle.dump
def patched_dump(obj, file, protocol=None, **kwargs):
    CompatibilityPickler(file, protocol, **kwargs).dump(obj)
pickle.dump = patched_dump

# -----------------------------------------------------------
# 1. PATHS AND KAGGLE SETUP (Modify these to point to your datasets)
# -----------------------------------------------------------
IMAGE_CLS_CKPT = "/kaggle/input/image-cls-v2-best/image_cls_v2_best.pth"
CEUS_CLS_CKPT  = "/kaggle/input/ceus-cls-v2-best/ceus_cls_v2_best.pth"
IMAGE_SEG_CKPT = "/kaggle/input/image-seg-v2-best/image_seg_v2_best.pth"
CEUS_SEG_CKPT  = "/kaggle/input/ceus-seg-v2-best/ceus_seg_v2_best.pth"
VIDEO_SEG_CKPT = "/kaggle/input/video-seg-v2-best/video_seg_v2_best.pth"

TRAIN = globals().get("TRAIN_PATH", "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN")
VAL_DIR = globals().get("VAL_PATH", "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUBMIT_DIR = Path("/kaggle/working")

# -----------------------------------------------------------
# 2. LOAD IMPORTS DYNAMICALLY
# -----------------------------------------------------------
for mod in list(sys.modules.keys()):
    if mod.startswith("src"): del sys.modules[mod]

from src.config import CFG
from src.models_v2 import build_seg_model_v2, ClsModelV2, CEUSClsModelV2
from src.postprocess import apply_tta_seg, morphological_cleanup
from src.dataset import get_partition_root

NORMALIZE = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# -----------------------------------------------------------
# 3. LOAD DATASET JSON
# -----------------------------------------------------------
PUBLIC_GT  = f"{VAL_DIR}/dataset_json_fingerprints_v4/public_all_ground_truth.json"
PRIVATE_GT = f"{VAL_DIR}/dataset_json_fingerprints_v4/private_val_for_participants.json"

all_samples = []
for p in [PUBLIC_GT, PRIVATE_GT]:
    if os.path.exists(p):
        with open(p) as f:
            all_samples.extend(json.load(f))

task_samples = {
    "image_cls": [s for s in all_samples if s["task"] == "image_cls"],
    "ceus_cls": [s for s in all_samples if s["task"] == "ceus_cls"],
    "image_seg": [s for s in all_samples if s["task"] == "image_seg"],
    "ceus_seg": [s for s in all_samples if s["task"] == "ceus_seg"],
    "video_seg": [s for s in all_samples if s["task"] == "video_seg"],
}

out_cls = {}
for task in ["image_seg", "ceus_seg", "video_seg"]:
    os.makedirs(SUBMIT_DIR / task, exist_ok=True)

# -----------------------------------------------------------
# TASK 1: image_cls
# -----------------------------------------------------------
print("\n--- image_cls ---")
if os.path.exists(IMAGE_CLS_CKPT) and task_samples["image_cls"]:
    model = ClsModelV2(CFG).to(DEVICE)
    model.load_state_dict(torch.load(IMAGE_CLS_CKPT, map_location=DEVICE)["model_state_dict"])
    model.eval()
    
    with torch.no_grad():
        for s in tqdm(task_samples["image_cls"]):
            p = get_partition_root(None, Path(VAL_DIR), s["data_partition_group"]) / s["input_path_relative"]
            if p.suffix.lower() in [".npy", ".npz"]:
                img = np.load(p, allow_pickle=True)
            else:
                img = np.array(Image.open(p).convert("RGB"))
            
            img_t = torch.tensor(img, dtype=torch.float32).permute(2,0,1)/255.0
            img_t = NORMALIZE(img_t).unsqueeze(0).to(DEVICE)
            
            logits = model(img_t)
            logits_hflip = model(torch.flip(img_t, dims=[3]))
            logits = (logits + logits_hflip) / 2.0
            
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            out_cls[s["input_path_relative"]] = {"prediction": int(probs.argmax()), "probability": probs.tolist()}

# -----------------------------------------------------------
# TASK 2: ceus_cls
# -----------------------------------------------------------
print("\n--- ceus_cls ---")
if os.path.exists(CEUS_CLS_CKPT) and task_samples["ceus_cls"]:
    model = CEUSClsModelV2(CFG).to(DEVICE)
    model.load_state_dict(torch.load(CEUS_CLS_CKPT, map_location=DEVICE)["model_state_dict"])
    model.eval()
    
    with torch.no_grad():
        for s in tqdm(task_samples["ceus_cls"]):
            p = get_partition_root(None, Path(VAL_DIR), s["data_partition_group"]) / s["input_path_relative"]
            if p.suffix.lower() in [".npy", ".npz"]:
                video = np.load(p, allow_pickle=True)
            else:
                video = np.array(Image.open(p).convert("RGB"))
            
            indices = np.linspace(0, video.shape[0]-1, 16, dtype=int)
            frames = video[indices]
            
            frames_t = torch.tensor(frames, dtype=torch.float32).permute(0,3,1,2)/255.0
            for i in range(16): frames_t[i] = NORMALIZE(frames_t[i])
            video_t = frames_t.unsqueeze(0).to(DEVICE)
            
            logits = model(video_t)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            out_cls[s["input_path_relative"]] = {"prediction": int(probs.argmax()), "probability": probs.tolist()}

# Write classification.json
with open(SUBMIT_DIR / "classification.json", "w") as f:
    json.dump(out_cls, f, indent=4)

# -----------------------------------------------------------
# TASK 3: image_seg
# -----------------------------------------------------------
print("\n--- image_seg ---")
if os.path.exists(IMAGE_SEG_CKPT) and task_samples["image_seg"]:
    model = build_seg_model_v2(CFG).to(DEVICE)
    model.load_state_dict(torch.load(IMAGE_SEG_CKPT, map_location=DEVICE)["model_state_dict"])
    model.eval()
    
    with torch.no_grad():
        for s in tqdm(task_samples["image_seg"]):
            p = get_partition_root(None, Path(VAL_DIR), s["data_partition_group"]) / s["input_path_relative"]
            if p.suffix.lower() in [".npy", ".npz"]:
                img = np.load(p, allow_pickle=True)
            else:
                img = np.array(Image.open(p).convert("RGB"))
            orig_h, orig_w = img.shape[:2]
            
            img_t = torch.tensor(img, dtype=torch.float32).permute(2,0,1)/255.0
            img_t = F.interpolate(img_t.unsqueeze(0), size=(CFG["img_size_seg"], CFG["img_size_seg"]), mode="bilinear", align_corners=False)
            img_t = NORMALIZE(img_t[0]).unsqueeze(0).to(DEVICE)
            
            logits = apply_tta_seg(model, img_t)
            logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            probs = torch.sigmoid(logits)[0,0].cpu().numpy()
            mask = morphological_cleanup(probs)
            
            ann_rel = s.get("annotation_path_relative")
            save_path = SUBMIT_DIR / (ann_rel if ann_rel else f"image_seg/{s['sample_id']}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            Image.fromarray((mask * 255).astype(np.uint8)).save(save_path)

# -----------------------------------------------------------
# TASK 4: ceus_seg
# -----------------------------------------------------------
print("\n--- ceus_seg ---")
if os.path.exists(CEUS_SEG_CKPT) and task_samples["ceus_seg"]:
    model = build_seg_model_v2(CFG).to(DEVICE)
    model.load_state_dict(torch.load(CEUS_SEG_CKPT, map_location=DEVICE)["model_state_dict"])
    model.eval()
    
    n_ceus_seg_done = 0
    with torch.no_grad():
        for s in tqdm(task_samples["ceus_seg"]):
            p = get_partition_root(None, Path(VAL_DIR), s["data_partition_group"]) / s["input_path_relative"]
            video = np.load(p, allow_pickle=True) # (15, 256, 512, 3)
            mid_frame = video[7]
            orig_h, orig_w = mid_frame.shape[:2]
            
            img_t = torch.tensor(mid_frame, dtype=torch.float32).permute(2,0,1)/255.0
            img_t = F.interpolate(img_t.unsqueeze(0), size=(CFG["img_size_seg"], CFG["img_size_seg"]), mode="bilinear", align_corners=False)
            img_t = NORMALIZE(img_t[0]).unsqueeze(0).to(DEVICE)
            
            logits = apply_tta_seg(model, img_t)
            logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            probs = torch.sigmoid(logits)[0,0].cpu().numpy()
            mask = morphological_cleanup(probs)
            
            mask_bin = (mask * 255).astype(np.uint8)
            
            # Save Path
            ann_rel = s.get("annotation_path_relative")
            if ann_rel:
                save_path = SUBMIT_DIR / ann_rel
            else:
                target_name = s.get("target_name") or f"seg_annotation_{n_ceus_seg_done:05d}.npz"
                dataset_nm  = s.get("dataset_name", s["organ"])
                save_path = SUBMIT_DIR / "ceus_seg" / dataset_nm / "annotations" / target_name
                
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(str(save_path), mask=mask_bin)
            n_ceus_seg_done += 1

# -----------------------------------------------------------
# TASK 5: video_seg
# -----------------------------------------------------------
print("\n--- video_seg ---")
if os.path.exists(VIDEO_SEG_CKPT) and task_samples["video_seg"]:
    model = build_seg_model_v2(CFG).to(DEVICE)
    model.load_state_dict(torch.load(VIDEO_SEG_CKPT, map_location=DEVICE)["model_state_dict"])
    model.eval()
    
    n_video_seg_done = 0
    with torch.no_grad():
        for s in tqdm(task_samples["video_seg"]):
            p = get_partition_root(None, Path(VAL_DIR), s["data_partition_group"]) / s["input_path_relative"]
            video = np.load(p, allow_pickle=True) # (3, T, 256, 256)
            
            frame_indices = s.get("frame_indices") or [0]
            fnum_mask = {}
            
            for fi in frame_indices:
                frame = video[0, fi] # View 0, frame fi
                orig_h, orig_w = frame.shape[:2]
                
                # Single channel float32 to RGB
                frame = frame / 255.0
                frame_3ch = np.stack([frame, frame, frame], axis=0)
                img_t = torch.tensor(frame_3ch, dtype=torch.float32).unsqueeze(0)
                img_t = F.interpolate(img_t, size=(CFG["img_size_seg"], CFG["img_size_seg"]), mode="bilinear", align_corners=False)
                img_t = NORMALIZE(img_t[0]).unsqueeze(0).to(DEVICE)
                
                logits = apply_tta_seg(model, img_t)
                logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
                probs = torch.sigmoid(logits)[0,0].cpu().numpy()
                mask = morphological_cleanup(probs)
                
                fnum_mask[str(fi)] = (mask * 255).astype(np.uint8)
                
            # Save Path
            ann_rel = s.get("annotation_path_relative")
            if ann_rel:
                save_path = SUBMIT_DIR / ann_rel
            else:
                target_name = s.get("target_name") or f"seg_annotation_{n_video_seg_done:05d}.npz"
                dataset_nm  = s.get("dataset_name", "CardiacCH")
                save_path = SUBMIT_DIR / "video_seg" / dataset_nm / "annotations" / target_name
                
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(str(save_path), fnum_mask=fnum_mask)
            n_video_seg_done += 1

print("\n--- Submission Packaging ---")
import subprocess
subprocess.run(["zip", "-r", "submission_v2.zip", "classification.json", "image_seg", "ceus_seg", "video_seg"], cwd=str(SUBMIT_DIR))
print("✅ submission_v2.zip ready in Kaggle working directory!")

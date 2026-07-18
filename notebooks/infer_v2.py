"""
notebooks/infer_v2.py
Unified Inference Script V2
Generates predictions with TTA and Morphological Post-Processing.
"""

import sys, os, json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import torchvision.transforms as T
import zipfile
from tqdm import tqdm

for mod in list(sys.modules.keys()):
    if mod.startswith("src"):
        del sys.modules[mod]

from src.config import CFG
from src.models_v2 import build_seg_model_v2, ClsModelV2, CEUSClsModelV2
from src.postprocess import apply_tta_seg, morphological_cleanup
from src.dataset import get_partition_root

VAL_DIR = globals().get("VAL_PATH", "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = CFG["ckpt_dir"]

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

NORMALIZE = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

out_cls = {}
os.makedirs("/kaggle/working/image_seg", exist_ok=True)
os.makedirs("/kaggle/working/ceus_seg", exist_ok=True)
os.makedirs("/kaggle/working/video_seg", exist_ok=True)

import pickle
class CompatibilityPickler(pickle.Pickler):
    def save_global(self, obj, name=None):
        if obj.__module__ == 'numpy._core.multiarray':
            obj.__module__ = 'numpy.core.multiarray'
        super().save_global(obj, name)

# --- image_cls ---
print("\n--- image_cls ---")
ckpt_path = f"{CKPT_DIR}/image_cls_v2_best.pth"
if os.path.exists(ckpt_path) and task_samples["image_cls"]:
    model = ClsModelV2(CFG).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE)["model_state_dict"])
    model.eval()
    
    with torch.no_grad():
        for s in tqdm(task_samples["image_cls"]):
            p = get_partition_root(None, Path(VAL_DIR), s["data_partition_group"]) / s["input_path_relative"]
            img = np.load(p, allow_pickle=True)
            img_t = torch.tensor(img, dtype=torch.float32).permute(2,0,1)/255.0
            img_t = NORMALIZE(img_t).unsqueeze(0).to(DEVICE)
            
            # Basic TTA (avg with H-Flip)
            logits = model(img_t)
            logits_hflip = model(torch.flip(img_t, dims=[3]))
            logits = (logits + logits_hflip) / 2.0
            
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            out_cls[s["input_path_relative"]] = {"prediction": int(probs.argmax()), "probability": probs.tolist()}

# --- ceus_cls ---
print("\n--- ceus_cls ---")
ckpt_path = f"{CKPT_DIR}/ceus_cls_v2_best.pth"
if os.path.exists(ckpt_path) and task_samples["ceus_cls"]:
    model = CEUSClsModelV2(CFG).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE)["model_state_dict"])
    model.eval()
    
    with torch.no_grad():
        for s in tqdm(task_samples["ceus_cls"]):
            p = get_partition_root(None, Path(VAL_DIR), s["data_partition_group"]) / s["input_path_relative"]
            video = np.load(p, allow_pickle=True)
            indices = np.linspace(0, video.shape[0]-1, 16, dtype=int)
            frames = video[indices]
            
            frames_t = torch.tensor(frames, dtype=torch.float32).permute(0,3,1,2)/255.0
            # NORMALIZE
            for i in range(16): frames_t[i] = NORMALIZE(frames_t[i])
            video_t = frames_t.unsqueeze(0).to(DEVICE)
            
            logits = model(video_t)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            out_cls[s["input_path_relative"]] = {"prediction": int(probs.argmax()), "probability": probs.tolist()}

# Save classification.json
with open("/kaggle/working/classification.json", "w") as f:
    json.dump(out_cls, f, indent=4)

# --- image_seg ---
print("\n--- image_seg ---")
ckpt_path = f"{CKPT_DIR}/image_seg_v2_best.pth"
from PIL import Image
if os.path.exists(ckpt_path) and task_samples["image_seg"]:
    model = build_seg_model_v2(CFG).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE)["model_state_dict"])
    model.eval()
    
    with torch.no_grad():
        for s in tqdm(task_samples["image_seg"]):
            p = get_partition_root(None, Path(VAL_DIR), s["data_partition_group"]) / s["input_path_relative"]
            img = np.load(p, allow_pickle=True)
            orig_h, orig_w = img.shape[:2]
            
            img_t = torch.tensor(img, dtype=torch.float32).permute(2,0,1)/255.0
            img_t = F.interpolate(img_t.unsqueeze(0), size=(CFG["img_size_seg"], CFG["img_size_seg"]), mode="bilinear", align_corners=False)
            img_t = NORMALIZE(img_t[0]).unsqueeze(0).to(DEVICE)
            
            # Apply Seg TTA
            logits = apply_tta_seg(model, img_t)
            
            logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            probs = torch.sigmoid(logits)[0,0].cpu().numpy()
            
            # Morphological Post-Processing
            mask = morphological_cleanup(probs)
            
            save_path = f"/kaggle/working/{s['annotation_path_relative']}"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            Image.fromarray((mask * 255).astype(np.uint8)).save(save_path)

print("\n--- Submission Packaging ---")
import subprocess
subprocess.run(["zip", "-r", "submission.zip", "classification.json", "image_seg", "ceus_seg", "video_seg"], cwd="/kaggle/working")
print("✅ submission.zip ready")

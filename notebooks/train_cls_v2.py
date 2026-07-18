"""
notebooks/train_cls_v2.py
Unified Classification Training V2
Trains image_cls and ceus_cls using competition-grade strategies.
"""
import sys, os, json, random, gc
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Force reload modules
for mod in list(sys.modules.keys()):
    if mod.startswith("src"):
        del sys.modules[mod]

from src.config import CFG
from src.augmentations_v2 import get_training_augmentation, get_validation_augmentation
from src.models_v2 import ClsModelV2, CEUSClsModelV2
from src.losses_v2 import FocalLoss
from src.trainer import UniversalTrainer
from src.dataset import get_partition_root

TRAIN = globals().get("TRAIN_PATH", "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN")
VAL_DIR = globals().get("VAL_PATH", "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(CFG["ckpt_dir"], exist_ok=True)

torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"]); random.seed(CFG["seed"])

PRIVATE_GT = f"{TRAIN}/dataset_json_fingerprints_v4/private_train_ground_truth.json"
PUBLIC_GT  = f"{TRAIN}/dataset_json_fingerprints_v4/public_all_ground_truth.json"
all_samples = []
for jp in [PRIVATE_GT, PUBLIC_GT]:
    if os.path.exists(jp):
        with open(jp) as f:
            all_samples.extend(json.load(f))

def accuracy_score(pred_logits, targets):
    preds = pred_logits.argmax(dim=1)
    return (preds == targets).float().mean()

# =======================================================================
# 1. Image Classification
# =======================================================================
print("\n--- Starting Image Classification V2 ---")
image_cls_samples = [s for s in all_samples if s["task"] == "image_cls"]

class ImageClsDatasetV2(Dataset):
    def __init__(self, samples, augment=None):
        self.samples = samples
        self.augment = augment
        
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        part_root = get_partition_root(Path(TRAIN), Path(VAL_DIR) if VAL_DIR else None, s["data_partition_group"])
        
        img = np.load(part_root / s["input_path_relative"], allow_pickle=True)
        if self.augment:
            res = self.augment(image=img)
            img = res["image"]
            
        label = s.get("class_label_index", 0)
        return {"input": img, "label": torch.tensor(label, dtype=torch.long)}

# Split
random.shuffle(image_cls_samples)
n_val = int(len(image_cls_samples) * 0.15)
train_ds = ImageClsDatasetV2(image_cls_samples[n_val:], augment=get_training_augmentation(CFG["img_size_cls"]))
val_ds = ImageClsDatasetV2(image_cls_samples[:n_val], augment=get_validation_augmentation(CFG["img_size_cls"]))

# Use a slightly larger batch size for classification
bs = CFG["batch_size"] * 2

train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=CFG["num_workers"], pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=CFG["num_workers"], pin_memory=True)

model = ClsModelV2(CFG).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG["T_0"], T_mult=CFG["T_mult"], eta_min=CFG["eta_min"])

# Calculate class weights for Focal Loss
labels = [s.get("class_label_index", 0) for s in image_cls_samples[n_val:]]
counts = [labels.count(0), labels.count(1)]
alpha = torch.tensor([1.0/max(c, 1) for c in counts]).to(DEVICE)
alpha = alpha / alpha.sum()

criterion = FocalLoss(alpha=alpha, gamma=CFG["focal_gamma"]).to(DEVICE)

trainer = UniversalTrainer(CFG, model, optimizer, scheduler, criterion, DEVICE, train_loader, val_loader, accuracy_score, task_name="image_cls_v2")
trainer.fit()

del model, optimizer, train_loader, val_loader, train_ds, val_ds, trainer
gc.collect(); torch.cuda.empty_cache()

# =======================================================================
# 2. CEUS Classification
# =======================================================================
print("\n--- Starting CEUS Classification V2 ---")
ceus_cls_samples = [s for s in all_samples if s["task"] == "ceus_cls"]

class CEUSClsDatasetV2(Dataset):
    def __init__(self, samples, augment=None):
        self.samples = samples
        self.augment = augment
        
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        part_root = get_partition_root(Path(TRAIN), Path(VAL_DIR) if VAL_DIR else None, s["data_partition_group"])
        video = np.load(part_root / s["input_path_relative"], allow_pickle=True) # (64, H, W, 3)
        
        # Subsample frames (e.g., 16 frames instead of 8 for better temporal resolution)
        indices = np.linspace(0, video.shape[0]-1, 16, dtype=int)
        frames = video[indices]
        
        processed_frames = []
        for f in frames:
            if self.augment:
                f = self.augment(image=f)["image"]
            processed_frames.append(f)
            
        video_t = torch.stack(processed_frames, dim=0) # (16, 3, H, W)
        label = s.get("class_label_index", 0)
        return {"input": video_t, "label": torch.tensor(label, dtype=torch.long)}

random.shuffle(ceus_cls_samples)
n_val = int(len(ceus_cls_samples) * 0.15)
train_ds = CEUSClsDatasetV2(ceus_cls_samples[n_val:], augment=get_training_augmentation(CFG["img_size_cls"]))
val_ds = CEUSClsDatasetV2(ceus_cls_samples[:n_val], augment=get_validation_augmentation(CFG["img_size_cls"]))

# CEUS takes more memory, use smaller batch size
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=CFG["num_workers"], pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=CFG["num_workers"], pin_memory=True)

model = CEUSClsModelV2(CFG).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG["T_0"], T_mult=CFG["T_mult"], eta_min=CFG["eta_min"])

labels = [s.get("class_label_index", 0) for s in ceus_cls_samples[n_val:]]
counts = [labels.count(0), labels.count(1)]
alpha = torch.tensor([1.0/max(c, 1) for c in counts]).to(DEVICE)
alpha = alpha / alpha.sum()

criterion = FocalLoss(alpha=alpha, gamma=CFG["focal_gamma"]).to(DEVICE)

trainer = UniversalTrainer(CFG, model, optimizer, scheduler, criterion, DEVICE, train_loader, val_loader, accuracy_score, task_name="ceus_cls_v2")
trainer.fit()

del model, optimizer, train_loader, val_loader, train_ds, val_ds, trainer
gc.collect(); torch.cuda.empty_cache()

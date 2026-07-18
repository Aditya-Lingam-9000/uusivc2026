"""
notebooks/train_seg_v2.py
Unified Segmentation Training V2
Trains image_seg, ceus_seg, and video_seg using competition-grade strategies.
"""
import sys, os, json, random, gc
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Force reload modules
for mod in list(sys.modules.keys()):
    if mod.startswith("src"):
        del sys.modules[mod]

from src.config import CFG
from src.augmentations_v2 import get_training_augmentation, get_validation_augmentation
from src.models_v2 import build_seg_model_v2
from src.losses_v2 import SegLossV2
from src.trainer import UniversalTrainer
from src.dataset import get_partition_root

# Config
TRAIN = globals().get("TRAIN_PATH", "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN")
VAL_DIR = globals().get("VAL_PATH", "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(CFG["ckpt_dir"], exist_ok=True)

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
random.seed(CFG["seed"])

# Load all samples
PRIVATE_GT = f"{TRAIN}/dataset_json_fingerprints_v4/private_train_ground_truth.json"
PUBLIC_GT  = f"{TRAIN}/dataset_json_fingerprints_v4/public_all_ground_truth.json"
all_samples = []
for jp in [PRIVATE_GT, PUBLIC_GT]:
    if os.path.exists(jp):
        with open(jp) as f:
            all_samples.extend(json.load(f))

# Helper metric
def dice_score(pred_logits, targets, threshold=0.5):
    probs = torch.sigmoid(pred_logits)
    preds_bin = (probs > threshold).float()
    intersection = (preds_bin * targets).sum()
    return (2.0 * intersection + 1e-6) / (preds_bin.sum() + targets.sum() + 1e-6)

# # =======================================================================
# # 1. Image Segmentation
# # =======================================================================
# print("\n--- Starting Image Segmentation V2 ---")
# image_seg_samples = [s for s in all_samples if s["task"] == "image_seg"]

# class ImageSegDatasetV2(Dataset):
#     def __init__(self, samples, augment=None):
#         self.samples = samples
#         self.augment = augment
        
#     def __len__(self): return len(self.samples)
    
#     def __getitem__(self, idx):
#         s = self.samples[idx]
#         part_root = get_partition_root(Path(TRAIN), Path(VAL_DIR) if VAL_DIR else None, s["data_partition_group"])
        
#         img_rel = s.get("input_path_relative") or s.get("img_path_relative")
#         ann_rel = s.get("annotation_path_relative") or s.get("mask_path_relative")
        
#         if img_rel is None:
#             raise KeyError(f"Sample has neither 'input_path_relative' nor 'img_path_relative': {s}")
            
#         img_path = part_root / img_rel
#         if img_path.suffix.lower() in [".npy", ".npz"]:
#             img = np.load(img_path, allow_pickle=True)
#             if isinstance(img, np.lib.npyio.NpzFile):
#                 img = img["arr_0"] # fallback
#         else:
#             img = np.array(Image.open(img_path).convert("RGB"))
            
#         if ann_rel is not None:
#             ann_path = part_root / ann_rel
#             if ann_path.suffix.lower() in [".npy", ".npz"]:
#                 npz = np.load(ann_path, allow_pickle=True)
#                 mask = npz["mask"].astype(np.float32) / 255.0  # (H, W)
#             else:
#                 mask = np.array(Image.open(ann_path)).astype(np.float32)
#                 if mask.ndim == 3: mask = mask[:,:,0]
#                 if mask.max() > 1.0: mask = mask / 255.0
#         else:
#             # Zero mask fallback if no annotation is found
#             mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        
#         if self.augment:
#             res = self.augment(image=img, mask=mask)
#             img, mask = res["image"], res["mask"]
            
#         mask = mask.unsqueeze(0)  # (1, H, W)
#         return {"input": img, "mask": mask}

# # Split
# random.shuffle(image_seg_samples)
# n_val = int(len(image_seg_samples) * 0.15)
# train_ds = ImageSegDatasetV2(image_seg_samples[n_val:], augment=get_training_augmentation(CFG["img_size_seg"]))
# val_ds = ImageSegDatasetV2(image_seg_samples[:n_val], augment=get_validation_augmentation(CFG["img_size_seg"]))

# train_loader = DataLoader(train_ds, batch_size=11, shuffle=True, num_workers=CFG["num_workers"], pin_memory=True)
# val_loader = DataLoader(val_ds, batch_size=11, shuffle=False, num_workers=CFG["num_workers"], pin_memory=True)

# model = build_seg_model_v2(CFG).to(DEVICE)
# optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG["T_0"], T_mult=CFG["T_mult"], eta_min=CFG["eta_min"])
# criterion = SegLossV2(w_dice=CFG["seg_loss_weights"]["dice"], w_focal=CFG["seg_loss_weights"]["focal"], w_bce=CFG["seg_loss_weights"]["bce"]).to(DEVICE)

# trainer = UniversalTrainer(CFG, model, optimizer, scheduler, criterion, DEVICE, train_loader, val_loader, dice_score, task_name="image_seg_v2")
# trainer.fit()

# del model, optimizer, train_loader, val_loader, train_ds, val_ds, trainer
# gc.collect(); torch.cuda.empty_cache()

# # =======================================================================
# # 2. CEUS Segmentation
# # =======================================================================
# print("\n--- Starting CEUS Segmentation V2 ---")
# ceus_seg_samples = [s for s in all_samples if s["task"] == "ceus_seg"]

# class CEUSSegDatasetV2(Dataset):
#     def __init__(self, samples, augment=None):
#         self.samples = samples
#         self.augment = augment
        
#     def __len__(self): return len(self.samples)
    
#     def __getitem__(self, idx):
#         s = self.samples[idx]
#         part_root = get_partition_root(Path(TRAIN), Path(VAL_DIR) if VAL_DIR else None, s["data_partition_group"])
        
#         npy_path = part_root / s["input_path_relative"]
#         video = np.load(npy_path, allow_pickle=True)
#         mid_frame = video[len(video)//2].astype(np.uint8) # (H, W, 3)
        
#         ann_path = part_root / s["annotation_path_relative"]
#         npz = np.load(ann_path, allow_pickle=True)
#         mask = npz["mask"].astype(np.float32) / 255.0 # (H, W)
        
#         if self.augment:
#             res = self.augment(image=mid_frame, mask=mask)
#             mid_frame, mask = res["image"], res["mask"]
            
#         mask_t = mask.clone().detach().float().unsqueeze(0) if torch.is_tensor(mask) else torch.tensor(mask).float().unsqueeze(0)
#         return {"input": mid_frame, "mask": mask_t}

# if ceus_seg_samples:
#     random.shuffle(ceus_seg_samples)
#     n_val = int(len(ceus_seg_samples) * 0.15)
#     train_ds = CEUSSegDatasetV2(ceus_seg_samples[n_val:], augment=get_training_augmentation(CFG["img_size_seg"]))
#     val_ds = CEUSSegDatasetV2(ceus_seg_samples[:n_val], augment=get_validation_augmentation(CFG["img_size_seg"]))

#     train_loader = DataLoader(train_ds, batch_size=11, shuffle=True, num_workers=CFG["num_workers"], pin_memory=True)
#     val_loader = DataLoader(val_ds, batch_size=11, shuffle=False, num_workers=CFG["num_workers"], pin_memory=True)

#     model = build_seg_model_v2(CFG).to(DEVICE)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG["T_0"], T_mult=CFG["T_mult"], eta_min=CFG["eta_min"])
#     criterion = SegLossV2(w_dice=CFG["seg_loss_weights"]["dice"], w_focal=CFG["seg_loss_weights"]["focal"], w_bce=CFG["seg_loss_weights"]["bce"]).to(DEVICE)

#     trainer = UniversalTrainer(CFG, model, optimizer, scheduler, criterion, DEVICE, train_loader, val_loader, dice_score, task_name="ceus_seg_v2")
#     trainer.fit()

#     del model, optimizer, train_loader, val_loader, train_ds, val_ds, trainer
#     gc.collect(); torch.cuda.empty_cache()

# =======================================================================
# 3. Video Segmentation
# =======================================================================
print("\n--- Starting Video Segmentation V2 ---")
video_seg_samples = [s for s in all_samples if s["task"] == "video_seg"]

import time
class VideoSegFrameDatasetV2(Dataset):
    def __init__(self, frame_list, augment=None):
        self.frame_list = frame_list
        self.augment = augment

    def __len__(self): return len(self.frame_list)

    def __getitem__(self, idx):
        item = self.frame_list[idx]
        frame = np.load(item["img_path"], allow_pickle=True)   # uint8 (256, 256)
        mask = np.load(item["mask_path"], allow_pickle=True)   # float32 (256, 256)
        
        frame_3ch = np.stack([frame, frame, frame], axis=-1)  # (256, 256, 3)
        
        if self.augment:
            res = self.augment(image=frame_3ch, mask=mask)
            frame_3ch, mask = res["image"], res["mask"]
            
        mask_t = mask.clone().detach().float().unsqueeze(0) if torch.is_tensor(mask) else torch.tensor(mask).float().unsqueeze(0)
        return {"input": frame_3ch, "mask": mask_t}

if video_seg_samples:
    CACHE_DIR = Path("/kaggle/working/video_seg_cache_v2")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame_list = []
    
    t_start = time.time()
    for idx, s in enumerate(video_seg_samples):
        part_root = get_partition_root(Path(TRAIN), Path(VAL_DIR) if VAL_DIR else None, s["data_partition_group"])
        ann_path = part_root / s["annotation_path_relative"]
        
        try:
            npz = np.load(ann_path, allow_pickle=True)
            fnum_mask = npz["fnum_mask"].item()
            video = None
            
            for f_str, mask_arr in fnum_mask.items():
                frame_idx = int(f_str)
                sample_id = s["sample_id"]
                img_save_path = CACHE_DIR / f"{sample_id}_f{frame_idx}_img.npy"
                mask_save_path = CACHE_DIR / f"{sample_id}_f{frame_idx}_mask.npy"
                
                if not img_save_path.exists() or not mask_save_path.exists():
                    if video is None:
                        video = np.load(part_root / s["input_path_relative"], allow_pickle=True)
                    frame = video[0, frame_idx].clip(0, 255).astype(np.uint8)
                    mask = (mask_arr / 255.0).clip(0, 1).astype(np.float32)
                    np.save(img_save_path, frame)
                    np.save(mask_save_path, mask)
                
                frame_list.append({"img_path": str(img_save_path), "mask_path": str(mask_save_path)})
        except Exception as e:
            pass

    random.shuffle(frame_list)
    n_val = int(len(frame_list) * 0.15)
    train_ds = VideoSegFrameDatasetV2(frame_list[n_val:], augment=get_training_augmentation(CFG["img_size_seg"]))
    val_ds = VideoSegFrameDatasetV2(frame_list[:n_val], augment=get_validation_augmentation(CFG["img_size_seg"]))

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=CFG["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=CFG["num_workers"], pin_memory=True)

    model = build_seg_model_v2(CFG).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG["T_0"], T_mult=CFG["T_mult"], eta_min=CFG["eta_min"])
    criterion = SegLossV2(w_dice=CFG["seg_loss_weights"]["dice"], w_focal=CFG["seg_loss_weights"]["focal"], w_bce=CFG["seg_loss_weights"]["bce"]).to(DEVICE)

    trainer = UniversalTrainer(CFG, model, optimizer, scheduler, criterion, DEVICE, train_loader, val_loader, dice_score, task_name="video_seg_v2")
    trainer.fit()

    del model, optimizer, train_loader, val_loader, train_ds, val_ds, trainer
    gc.collect(); torch.cuda.empty_cache()

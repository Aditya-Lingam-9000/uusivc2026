"""
src/dataset.py
UUSIVC 2026 — Unified DataLoader for all 5 tasks.

Confirmed shapes from EDA:
  image_cls  : PIL Image (variable H×W, RGB)
  image_seg  : PIL Image + PNG mask (variable H×W)
  ceus_cls   : .npy → (64, 256, 512, 3) uint8
  ceus_seg   : .npy video (15, 256, 512, 3) uint8 + .npz mask {'mask': (256,512)}
  video_seg  : .npy (3, T, 256, 256) float32 + .npz {'fnum_mask': dict}

Mask format quirks (from EDA):
  - BUS-BRA masks: bool → convert via .astype(np.uint8)*255
  - BUSIS, DDTI, KidneyUS masks: (H,W,3) → take channel 0
  - All others: (H,W) uint8 [0,255]
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────
#  Mask loading — handles all 3 format variants
# ─────────────────────────────────────────────
def load_mask_png(mask_path: Path) -> torch.Tensor:
    """
    Returns: FloatTensor shape (1, H, W), values in {0.0, 1.0}
    Handles: bool masks (BUS-BRA), 3-channel masks (BUSIS/DDTI/KidneyUS),
             single-channel masks (all private + Fetal_HC)
    """
    mask = np.array(Image.open(mask_path))
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    if mask.ndim == 3:
        mask = mask[:, :, 0]                       # take first channel
    binary = (mask > 127).astype(np.float32)       # 0.0 or 1.0
    return torch.tensor(binary).unsqueeze(0)        # (1, H, W)


# ─────────────────────────────────────────────
#  Main Dataset class
# ─────────────────────────────────────────────
class UUSIVCDataset(Dataset):
    """
    Unified dataset for UUSIVC 2026 — all 5 task types.

    Usage:
        ds = UUSIVCDataset(
            json_paths=[private_gt_path, public_gt_path],
            data_root="/kaggle/input/.../TRAIN",
            transform=get_train_transforms(),   # for image tasks
            task_filter=['image_cls', 'image_seg'],  # or None for all
        )
    """

    def __init__(self, json_paths, data_root, transform=None,
                 task_filter=None, max_samples=None):
        self.data_root = Path(data_root)
        self.transform = transform

        # Load all samples from one or more ground-truth JSON files
        self.samples = []
        for jp in json_paths:
            with open(jp) as f:
                data = json.load(f)
            self.samples.extend(data)

        # Optional task filter
        if task_filter:
            self.samples = [s for s in self.samples if s['task'] in task_filter]

        # Optional sample cap (useful for quick debugging)
        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"[UUSIVCDataset] Loaded {len(self.samples)} samples"
              f" | tasks={task_filter or 'ALL'}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        task = s['task']

        if task == 'image_cls':
            return self._image_cls(s)
        elif task == 'image_seg':
            return self._image_seg(s)
        elif task == 'ceus_cls':
            return self._ceus_cls(s)
        elif task == 'ceus_seg':
            return self._ceus_seg(s)
        elif task == 'video_seg':
            return self._video_seg(s)
        else:
            raise ValueError(f"Unknown task: {task}")

    # ── Task loaders ─────────────────────────────────────────

    def _image_cls(self, s):
        """
        Input : PIL Image → transform → (3, H, W) tensor
        Label : int (0 or 1) from class_label_index
        """
        img_path = self.data_root / s['input_path_relative']
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = s['class_label_index']  # int or None (val set)
        return {
            'input': img,
            'label': torch.tensor(label, dtype=torch.long) if label is not None else -1,
            'sample_id': s['sample_id'],
            'task': s['task'],
            'organ': s['organ'],
        }

    def _image_seg(self, s):
        """
        Input : PIL Image → transform → (3, H, W) tensor
        Target: binary mask → (1, H, W) float tensor
        """
        img_path = self.data_root / s['img_path_relative']
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        mask = None
        if s.get('mask_path_relative'):
            mask = load_mask_png(self.data_root / s['mask_path_relative'])

        return {
            'input': img,
            'mask': mask,
            'sample_id': s['sample_id'],
            'task': s['task'],
            'organ': s['organ'],
        }

    def _ceus_cls(self, s):
        """
        Input : .npy (64, 256, 512, 3) uint8
                → FloatTensor (64, 3, 256, 512), normalized 0-1
        Label : int (0 or 1)
        """
        npy_path = self.data_root / s['input_path_relative']
        video = np.load(npy_path)                                   # (64,256,512,3)
        video = torch.tensor(video, dtype=torch.float32)
        video = video.permute(0, 3, 1, 2) / 255.0                  # (64,3,256,512)
        label = s['class_label_index']
        return {
            'input': video,
            'label': torch.tensor(label, dtype=torch.long) if label is not None else -1,
            'sample_id': s['sample_id'],
            'task': s['task'],
            'organ': s['organ'],
        }

    def _ceus_seg(self, s):
        """
        Input : .npy (15, 256, 512, 3) uint8
                → FloatTensor (15, 3, 256, 512), normalized 0-1
        Target: .npz key='mask' → (1, 256, 512) binary float
        """
        npy_path = self.data_root / s['input_path_relative']
        video = np.load(npy_path)                                   # (15,256,512,3)
        video = torch.tensor(video, dtype=torch.float32)
        video = video.permute(0, 3, 1, 2) / 255.0                  # (15,3,256,512)

        mask = None
        if s.get('annotation_path_relative'):
            npz = np.load(self.data_root / s['annotation_path_relative'])
            m = npz['mask'].astype(np.float32) / 255.0             # (256,512)
            mask = torch.tensor(m).unsqueeze(0)                    # (1,256,512)

        return {
            'input': video,
            'mask': mask,
            'sample_id': s['sample_id'],
            'task': s['task'],
            'organ': s['organ'],
        }

    def _video_seg(self, s):
        """
        Input : .npy (3, T, 256, 256) float32  → normalized 0-1
                3 = cardiac views, T = variable number of frames
        Target: .npz key='fnum_mask' → dict {frame_int: (256,256) float tensor}
                Sparse — NOT all frames are annotated.
        Extra : CAMUS npz also has 'ef', 'edv', 'esv', 'spacing'
        """
        npy_path = self.data_root / s['input_path_relative']
        video = np.load(npy_path)                                   # (3,T,256,256)
        video = torch.tensor(video, dtype=torch.float32) / 255.0   # normalized

        mask_dict = None
        extra = {}
        if s.get('annotation_path_relative'):
            npz = np.load(self.data_root / s['annotation_path_relative'],
                          allow_pickle=True)
            fnum_mask = npz['fnum_mask'].item()                    # dict: {'9': arr, ...}
            mask_dict = {
                int(k): torch.tensor(v / 255.0, dtype=torch.float32)
                for k, v in fnum_mask.items()
            }
            # CAMUS has extra clinical info
            if 'ef' in npz:
                extra['ef'] = float(npz['ef'])
                extra['edv'] = float(npz['edv'])
                extra['esv'] = float(npz['esv'])

        return {
            'input': video,
            'mask_dict': mask_dict,
            'extra': extra,
            'sample_id': s['sample_id'],
            'task': s['task'],
            'organ': s['organ'],
        }

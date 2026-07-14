"""
src/dataset.py
UUSIVC 2026 — Unified DataLoader for all 5 tasks.

DIRECTORY STRUCTURE (confirmed from dataset_exploration_docs):
  TRAIN/
    Challenge_Data_Private_v2_fully_anonymized/Train/   ← private_train partition
      ceus_cls/{Organ}/{0,1}/*.npy
      ceus_seg/{Organ}CEUS/videos/*.npy
      ceus_seg/{Organ}CEUS/annotations/*.npz
      image_cls/{Organ}/{0,1}/*.jpg|png
      image_cls/Prostate/imgs/*.jpg        ← special: labels from JSON
      image_seg/{Organ}/imgs/*.png
      image_seg/{Organ}/masks/*.png
      video_seg/CardiacCH/videos/*.npy
      video_seg/CardiacCH/annotations/*.npz
    Challenge_Data_Public/                  ← public_all partition
      image_cls/{Dataset}/{0,1}/*.jpg|png
      image_seg/{Dataset}/imgs/*.png
      image_seg/{Dataset}/masks/*.png
      video_seg/CAMUS/videos/*.npy
      video_seg/CAMUS/annotations/*.npz
    dataset_json_fingerprints_v4/
      private_train_ground_truth.json
      public_all_ground_truth.json

  VAL/
    Challenge_Data_Private_v2_fully_anonymized/Val/    ← private_val partition
      ceus_cls/{Organ}/videos/*.npy        ← NO class folders in val
      ceus_seg/{Organ}CEUS/videos/*.npy    ← NO annotations in val
      image_cls/{Organ}/imgs/*.jpg|png     ← NO class folders in val
      image_seg/{Organ}/imgs/*.png         ← NO masks in val
      video_seg/CardiacCH/videos/*.npy     ← NO annotations in val
    dataset_json_fingerprints_v4/
      private_val_for_participants.json

KEY: input_path_relative in JSON is relative to the PARTITION ROOT, not TRAIN root.
  private_train → TRAIN/Challenge_Data_Private_v2_fully_anonymized/Train/
  public_all    → TRAIN/Challenge_Data_Public/
  private_val   → VAL/Challenge_Data_Private_v2_fully_anonymized/Val/

MASK FORMAT QUIRKS (from deep EDA):
  BUS-BRA masks : bool dtype → .astype(np.uint8) * 255
  BUSIS/DDTI/KidneyUS masks : (H,W,3) → take [:,:,0]
  All others : (H,W) uint8 [0,255]
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────
#  Partition root prefixes
# ─────────────────────────────────────────────
PRIVATE_TRAIN_PREFIX = Path("Challenge_Data_Private_v2_fully_anonymized") / "Train"
PUBLIC_PREFIX        = Path("Challenge_Data_Public")
PRIVATE_VAL_PREFIX   = Path("Challenge_Data_Private_v2_fully_anonymized") / "Val"


def get_partition_root(data_root: Path, val_root: Path | None, partition: str) -> Path:
    """
    Returns the absolute root directory for a given data_partition_group.

    Args:
        data_root : Path to TRAIN/ directory
        val_root  : Path to VAL/ directory (only needed for private_val)
        partition : value of sample['data_partition_group']
    """
    if partition == "private_train":
        return data_root / PRIVATE_TRAIN_PREFIX
    elif partition == "public_all":
        return data_root / PUBLIC_PREFIX
    elif partition == "private_val":
        base = val_root if val_root else data_root
        return base / PRIVATE_VAL_PREFIX
    else:
        # Fallback — should not happen with known JSON files
        return data_root


# ─────────────────────────────────────────────
#  Mask loading — handles all 3 format variants
# ─────────────────────────────────────────────
def load_mask_png(mask_path: Path) -> torch.Tensor:
    """
    Load a segmentation mask PNG, handling all 3 format variants found in EDA:
      - bool dtype (BUS-BRA)
      - 3-channel uint8 (BUSIS, DDTI, KidneyUS)
      - single-channel uint8 (all private + Fetal_HC)

    Returns: FloatTensor shape (1, H, W), values in {0.0, 1.0}
    """
    mask = np.array(Image.open(mask_path))

    # Variant 1: bool (BUS-BRA) → convert to uint8
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255

    # Variant 2: 3-channel (BUSIS, DDTI, KidneyUS) → take first channel
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # Binary threshold → float32
    binary = (mask > 127).astype(np.float32)
    return torch.tensor(binary).unsqueeze(0)   # (1, H, W)


# ─────────────────────────────────────────────
#  Main Dataset class
# ─────────────────────────────────────────────
class UUSIVCDataset(Dataset):
    """
    Unified dataset for UUSIVC 2026 — handles all 5 task types across
    private and public partitions for both TRAIN and VAL splits.

    Usage (train):
        ds = UUSIVCDataset(
            json_paths=[
                "TRAIN/dataset_json_fingerprints_v4/private_train_ground_truth.json",
                "TRAIN/dataset_json_fingerprints_v4/public_all_ground_truth.json",
            ],
            data_root="TRAIN",
            task_filter=['image_cls'],
        )

    Usage (val / inference):
        ds = UUSIVCDataset(
            json_paths=["VAL/dataset_json_fingerprints_v4/private_val_for_participants.json"],
            data_root="TRAIN",   # still needed for partition root resolution
            val_root="VAL",
            task_filter=None,
        )
    """

    def __init__(
        self,
        json_paths,
        data_root,
        val_root=None,
        transform=None,
        task_filter=None,
        max_samples=None,
    ):
        self.data_root = Path(data_root)
        self.val_root  = Path(val_root) if val_root else None
        self.transform = transform

        # Load all samples from one or more JSON ground-truth files
        self.samples = []
        for jp in json_paths:
            with open(jp) as f:
                data = json.load(f)
            self.samples.extend(data)

        # Optional task filter
        if task_filter:
            self.samples = [s for s in self.samples if s["task"] in task_filter]

        # Optional cap (useful for quick debugging)
        if max_samples:
            self.samples = self.samples[:max_samples]

        print(
            f"[UUSIVCDataset] Loaded {len(self.samples)} samples"
            f" | tasks={task_filter or 'ALL'}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        task = s["task"]

        if task == "image_cls":
            return self._image_cls(s)
        elif task == "image_seg":
            return self._image_seg(s)
        elif task == "ceus_cls":
            return self._ceus_cls(s)
        elif task == "ceus_seg":
            return self._ceus_seg(s)
        elif task == "video_seg":
            return self._video_seg(s)
        else:
            raise ValueError(f"Unknown task: {task}")

    # ── Internal: resolve file path from JSON relative path ─────

    def _resolve(self, s, rel_key: str) -> Path | None:
        """
        Resolve an absolute file path from a relative-path key in the sample dict.
        Returns None if the key is absent or null.
        """
        rel = s.get(rel_key)
        if not rel:
            return None
        partition = s.get("data_partition_group", "")
        part_root = get_partition_root(self.data_root, self.val_root, partition)
        return part_root / rel

    # ── Task loaders ─────────────────────────────────────────────

    def _image_cls(self, s):
        """
        Input : PIL Image (any size) → transform → (3, H, W) tensor
        Label : int 0/1 from class_label_index (None for val → returns -1)
        """
        img_path = self._resolve(s, "input_path_relative")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        label = s.get("class_label_index")
        return {
            "input":     img,
            "label":     torch.tensor(label, dtype=torch.long) if label is not None else torch.tensor(-1),
            "sample_id": s["sample_id"],
            "task":      s["task"],
            "organ":     s["organ"],
        }

    def _image_seg(self, s):
        """
        Input : PIL Image → transform → (3, H, W) tensor
        Target: binary PNG mask → (1, H, W) float tensor  [None for val]
        """
        img_path = self._resolve(s, "img_path_relative")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        mask = None
        mask_path = self._resolve(s, "mask_path_relative")
        if mask_path and mask_path.exists():
            mask = load_mask_png(mask_path)

        return {
            "input":     img,
            "mask":      mask,
            "sample_id": s["sample_id"],
            "task":      s["task"],
            "organ":     s["organ"],
        }

    def _ceus_cls(self, s):
        """
        Input : .npy (64, 256, 512, 3) uint8
                → FloatTensor (64, 3, 256, 512), normalized [0,1]
        Label : int 0/1  (None for val → -1)
        """
        npy_path = self._resolve(s, "input_path_relative")
        video = np.load(npy_path)                                    # (64,256,512,3)
        video = torch.tensor(video, dtype=torch.float32)
        video = video.permute(0, 3, 1, 2) / 255.0                   # (64,3,256,512)

        label = s.get("class_label_index")
        return {
            "input":     video,
            "label":     torch.tensor(label, dtype=torch.long) if label is not None else torch.tensor(-1),
            "sample_id": s["sample_id"],
            "task":      s["task"],
            "organ":     s["organ"],
        }

    def _ceus_seg(self, s):
        """
        Input : .npy (15, 256, 512, 3) uint8
                → FloatTensor (15, 3, 256, 512), normalized [0,1]
        Target: .npz key='mask' → (1, 256, 512) binary float  [None for val]
        """
        npy_path = self._resolve(s, "input_path_relative")
        video = np.load(npy_path)                                    # (15,256,512,3)
        video = torch.tensor(video, dtype=torch.float32)
        video = video.permute(0, 3, 1, 2) / 255.0                   # (15,3,256,512)

        mask = None
        ann_path = self._resolve(s, "annotation_path_relative")
        if ann_path and ann_path.exists():
            npz = np.load(ann_path)
            m   = npz["mask"].astype(np.float32) / 255.0            # (256,512)
            mask = torch.tensor(m).unsqueeze(0)                     # (1,256,512)

        return {
            "input":     video,
            "mask":      mask,
            "sample_id": s["sample_id"],
            "task":      s["task"],
            "organ":     s["organ"],
        }

    def _video_seg(self, s):
        """
        Input : .npy (3, T, 256, 256) float32  → normalized [0,1]
                Dim-0 = 3 cardiac views, Dim-1 = T variable frames

        Target (CAMUS)   : .npz keys=['fnum_mask','ef','edv','esv','spacing']
                           fnum_mask → dict {frame_int: (256,256) float tensor}
        Target (CardiacCH): .npz key='fnum_mask' same dict format, sparse frames
        """
        npy_path = self._resolve(s, "input_path_relative")
        video = np.load(npy_path)                                    # (3,T,256,256)
        video = torch.tensor(video, dtype=torch.float32) / 255.0

        mask_dict = None
        extra = {}
        ann_path = self._resolve(s, "annotation_path_relative")
        if ann_path and ann_path.exists():
            npz = np.load(ann_path, allow_pickle=True)
            fnum_mask = npz["fnum_mask"].item()                      # dict
            mask_dict = {
                int(k): torch.tensor(v / 255.0, dtype=torch.float32)
                for k, v in fnum_mask.items()
            }
            # CAMUS has additional clinical scalars
            for key in ("ef", "edv", "esv"):
                if key in npz:
                    extra[key] = float(npz[key])

        return {
            "input":     video,
            "mask_dict": mask_dict,
            "extra":     extra,
            "sample_id": s["sample_id"],
            "task":      s["task"],
            "organ":     s["organ"],
        }

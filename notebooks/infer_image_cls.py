"""
notebooks/infer_image_cls.py
Generate image_cls predictions for the VAL set.

Run this AFTER train_image_cls.py on the SAME Kaggle session
(so the checkpoint is still in /kaggle/working/checkpoints/).

HOW TO RUN:
    TRAIN_PATH = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN"
    VAL_PATH   = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL"
    exec(open('/kaggle/working/repo/notebooks/infer_image_cls.py').read())

OUTPUT:
    /kaggle/working/predictions/image_cls_predictions.json
    Format: [{"sample_id": "...", "class_label_index": 0}, ...]
"""

import sys, os, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

# ── Force reload ──────────────────────────────────────────────
for mod in list(sys.modules.keys()):
    if mod.startswith("src"):
        del sys.modules[mod]

TRAIN   = globals().get("TRAIN_PATH", "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN")
VAL_DIR = globals().get("VAL_PATH",   "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL")
CKPT    = "/kaggle/working/checkpoints/image_cls_best.pth"
PRED_DIR = "/kaggle/working/predictions"
os.makedirs(PRED_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from src.dataset import UUSIVCDataset
from src.transforms import get_val_transforms
from src.model import build_model

# ── Load val dataset ──────────────────────────────────────────
VAL_GT = f"{VAL_DIR}/dataset_json_fingerprints_v4/private_val_for_participants.json"

val_ds = UUSIVCDataset(
    json_paths=[VAL_GT],
    data_root=TRAIN,
    val_root=VAL_DIR,
    transform=get_val_transforms(),
    task_filter=["image_cls"],
)
print(f"Val image_cls samples: {len(val_ds)}")

val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                        num_workers=4, pin_memory=True)

# ── Load model from checkpoint ────────────────────────────────
model = build_model("image_cls", pretrained=False)
ckpt  = torch.load(CKPT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(DEVICE)
model.eval()
print(f"✅ Loaded checkpoint (epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")

# ── Inference ─────────────────────────────────────────────────
predictions = []

with torch.no_grad():
    for batch in val_loader:
        imgs       = batch["input"].to(DEVICE)
        sample_ids = batch["sample_id"]

        logits = model(imgs)                           # (B, 2)
        probs  = torch.softmax(logits, dim=1)          # (B, 2)
        preds  = logits.argmax(dim=1).cpu().tolist()   # [0 or 1]
        confs  = probs.max(dim=1).values.cpu().tolist()

        for sid, pred, conf in zip(sample_ids, preds, confs):
            predictions.append({
                "sample_id":         sid,
                "class_label_index": pred,
                "confidence":        round(conf, 4),
            })

# ── Save predictions ──────────────────────────────────────────
out_path = f"{PRED_DIR}/image_cls_predictions.json"
with open(out_path, "w") as f:
    json.dump(predictions, f, indent=2)

print(f"\n✅ Saved {len(predictions)} predictions → {out_path}")

# ── Print summary ─────────────────────────────────────────────
from collections import Counter, defaultdict
pred_labels = [p["class_label_index"] for p in predictions]
print(f"Prediction distribution: class_0={pred_labels.count(0)}  class_1={pred_labels.count(1)}")

# Per-organ breakdown
organ_preds = defaultdict(list)
for p in predictions:
    # Extract organ from sample_id: "private_val::image_cls::Appendix::xxxx"
    parts = p["sample_id"].split("::")
    organ = parts[2] if len(parts) >= 3 else "unknown"
    organ_preds[organ].append(p["class_label_index"])

print("\nPer-organ prediction distribution:")
for organ in sorted(organ_preds):
    preds_organ = organ_preds[organ]
    n = len(preds_organ)
    n1 = sum(preds_organ)
    print(f"  {organ:20s}: n={n:4d}  class_0={n-n1:4d} ({(n-n1)/n*100:.1f}%)  class_1={n1:4d} ({n1/n*100:.1f}%)")

print(f"\nPreview of first 3 predictions:")
for p in predictions[:3]:
    print(f"  {p}")

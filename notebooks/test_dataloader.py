"""
notebooks/test_dataloader.py
Run this on Kaggle (CPU session) to verify all 5 task loaders work.

HOW TO RUN:
  # In notebook cell:
  TRAIN_PATH = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN"
  VAL_PATH   = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL"
  exec(open('/kaggle/working/repo/notebooks/test_dataloader.py').read())
"""

import sys, os, traceback, importlib
import torch, numpy as np

# ── Force reload src modules so exec() always uses the latest code on disk ──
# This avoids the "old cached module" problem when you git pull without restarting kernel.
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("src"):
        del sys.modules[mod_name]
        print(f"  [cache cleared] {mod_name}")

# ── Path resolution ───────────────────────────────────────────
def _get(env_key, default):
    v = globals().get(env_key)
    return v if v else default

TRAIN = _get("TRAIN_PATH", "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN")
VAL   = _get("VAL_PATH",   "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL")
print(f"TRAIN : {TRAIN}")
print(f"VAL   : {VAL}")

# ── Import src ────────────────────────────────────────────────
try:
    from src.dataset import UUSIVCDataset
    from src.transforms import get_val_transforms
    print("✅ Imports OK")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    raise

# ── Verify JSON files exist ───────────────────────────────────
PRIVATE_GT = f"{TRAIN}/dataset_json_fingerprints_v4/private_train_ground_truth.json"
PUBLIC_GT  = f"{TRAIN}/dataset_json_fingerprints_v4/public_all_ground_truth.json"
VAL_GT     = f"{VAL}/dataset_json_fingerprints_v4/private_val_for_participants.json"

for p in [PRIVATE_GT, PUBLIC_GT, VAL_GT]:
    status = "✅" if os.path.exists(p) else "❌"
    print(f"{status} {os.path.basename(p)}")

# ── Test each task (TRAIN samples) ───────────────────────────
tasks = ["image_cls", "image_seg", "ceus_cls", "ceus_seg", "video_seg"]
results = {}

for task in tasks:
    print(f"\n{'='*55}")
    print(f"  TESTING TRAIN: {task}")
    print(f"{'='*55}")
    try:
        transform = get_val_transforms() if task in ["image_cls", "image_seg"] else None

        ds = UUSIVCDataset(
            json_paths=[PRIVATE_GT, PUBLIC_GT],
            data_root=TRAIN,
            val_root=VAL,
            transform=transform,
            task_filter=[task],
            max_samples=3,
        )
        print(f"  Dataset size: {len(ds)}")

        for i in range(min(2, len(ds))):
            sample = ds[i]
            print(f"\n  Sample [{i}]  organ={sample['organ']}")
            for k, v in sample.items():
                if k in ("sample_id", "task", "organ", "extra"):
                    continue
                if isinstance(v, torch.Tensor):
                    print(f"    {k:12s}: shape={list(v.shape)} dtype={v.dtype} "
                          f"min={v.min():.3f} max={v.max():.3f}")
                elif isinstance(v, dict):
                    print(f"    {k:12s}: dict keys={list(v.keys())[:4]}")
                elif v is None:
                    print(f"    {k:12s}: None (expected for val/unlabelled)")
                else:
                    print(f"    {k:12s}: {v}")

        results[task] = "PASSED ✅"
        print(f"\n  >>> {task} TRAIN: PASSED ✅")

    except Exception as e:
        results[task] = f"FAILED ❌ — {e}"
        print(f"\n  >>> {task} TRAIN: FAILED ❌  —  {e}")
        traceback.print_exc()

# ── Test VAL inference path ───────────────────────────────────
print(f"\n{'='*55}")
print("  TESTING VAL (inference, no labels)")
print(f"{'='*55}")
val_results = {}
for task in tasks:
    try:
        transform = get_val_transforms() if task in ["image_cls", "image_seg"] else None
        ds = UUSIVCDataset(
            json_paths=[VAL_GT],
            data_root=TRAIN,
            val_root=VAL,
            transform=transform,
            task_filter=[task],
            max_samples=2,
        )
        if len(ds) == 0:
            val_results[task] = "SKIPPED (0 samples in val)"
            continue
        sample = ds[0]
        val_results[task] = "PASSED ✅"
        # Confirm label is -1 (no ground truth)
        if "label" in sample:
            assert sample["label"].item() == -1, "Expected -1 label for val sample"
        if "mask" in sample:
            assert sample["mask"] is None, "Expected None mask for val sample"
        print(f"  {task:15s}: PASSED ✅  (label=-1 / mask=None confirmed)")
    except Exception as e:
        val_results[task] = f"FAILED ❌ — {e}"
        print(f"  {task:15s}: FAILED ❌ — {e}")
        traceback.print_exc()

# ── Final summary ──────────────────────────────────────────────
print(f"\n{'='*55}")
print("  FINAL RESULTS — TRAIN")
for task, r in results.items():
    print(f"  {task:15s}: {r}")
print(f"\n  FINAL RESULTS — VAL")
for task, r in val_results.items():
    print(f"  {task:15s}: {r}")

all_train = all("PASSED" in r for r in results.values())
all_val   = all("PASSED" in r or "SKIPPED" in r for r in val_results.values())

if all_train and all_val:
    print("\n🎉 ALL CHECKS PASSED — DataLoader is ready for model training!")
else:
    print("\n⚠️  Some checks failed — fix dataset.py before proceeding.")

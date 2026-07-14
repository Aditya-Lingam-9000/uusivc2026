"""
notebooks/test_dataloader.py
Run this on Kaggle (CPU session) to verify all 5 task loaders work.

HOW TO RUN ON KAGGLE:
  # Cell 1 — Setup
  TRAIN_PATH = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN"
  !git clone https://github.com/YOUR_USERNAME/uusivc2026 repo
  import sys; sys.path.insert(0, 'repo')
  !cp repo/notebooks/test_dataloader.py .

  # Cell 2 — Run
  exec(open('test_dataloader.py').read())

WHAT TO OBSERVE:
  - Each task prints: Dataset size + first 2 sample shapes
  - All 5 tasks should print "PASSED"
  - No KeyError, FileNotFoundError, or shape errors
"""

import sys
import os
import torch
import numpy as np
import argparse

# ── Path setup ────────────────────────────────────────────────
def get_train_path():
    train_path = globals().get('TRAIN_PATH', None)
    if train_path:
        return train_path
    has_args = any(a.startswith('--train') for a in sys.argv)
    if has_args:
        p = argparse.ArgumentParser()
        p.add_argument('--train', type=str)
        args, _ = p.parse_known_args()
        return args.train
    return "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN"

TRAIN = get_train_path()
print(f"Using TRAIN: {TRAIN}")

# ── Imports (assumes repo is on sys.path) ──────────────────────
try:
    from src.dataset import UUSIVCDataset
    from src.transforms import get_val_transforms, SegValTransform
    print("✅ Imports OK")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Make sure sys.path includes the repo root.")
    raise

# ── Ground truth JSON paths ────────────────────────────────────
PRIVATE_GT = f"{TRAIN}/dataset_json_fingerprints_v4/private_train_ground_truth.json"
PUBLIC_GT  = f"{TRAIN}/dataset_json_fingerprints_v4/public_all_ground_truth.json"

for p in [PRIVATE_GT, PUBLIC_GT]:
    if not os.path.exists(p):
        print(f"❌ JSON not found: {p}")
        raise FileNotFoundError(p)
    print(f"✅ Found: {os.path.basename(p)}")

# ── Test each task ─────────────────────────────────────────────
tasks = ['image_cls', 'image_seg', 'ceus_cls', 'ceus_seg', 'video_seg']
results = {}

for task in tasks:
    print(f"\n{'='*55}")
    print(f"  TESTING: {task}")
    print(f"{'='*55}")
    try:
        # Select transform based on task type
        if task in ['image_cls', 'image_seg']:
            transform = get_val_transforms()
        else:
            transform = None  # NPY tasks normalize inside dataset

        ds = UUSIVCDataset(
            json_paths=[PRIVATE_GT, PUBLIC_GT],
            data_root=TRAIN,
            transform=transform,
            task_filter=[task],
            max_samples=5,    # Only load 5 samples for testing
        )

        print(f"  Dataset size (capped at 5): {len(ds)}")

        # Load and inspect 2 samples
        for i in range(min(2, len(ds))):
            sample = ds[i]
            print(f"\n  Sample [{i}]:")
            print(f"    sample_id : {sample['sample_id'][:50]}...")
            print(f"    task      : {sample['task']}")
            print(f"    organ     : {sample['organ']}")
            for k, v in sample.items():
                if k in ('sample_id', 'task', 'organ'):
                    continue
                if isinstance(v, torch.Tensor):
                    print(f"    {k:12s}: Tensor shape={v.shape} dtype={v.dtype} "
                          f"min={v.min():.3f} max={v.max():.3f}")
                elif isinstance(v, dict):
                    print(f"    {k:12s}: dict with {len(v)} keys: {list(v.keys())[:3]}")
                elif v is None:
                    print(f"    {k:12s}: None")
                else:
                    print(f"    {k:12s}: {v}")

        results[task] = 'PASSED ✅'
        print(f"\n  >>> {task}: PASSED ✅")

    except Exception as e:
        import traceback
        results[task] = f'FAILED ❌ — {e}'
        print(f"\n  >>> {task}: FAILED ❌")
        traceback.print_exc()

# ── Final summary ──────────────────────────────────────────────
print(f"\n{'='*55}")
print("  FINAL RESULTS")
print(f"{'='*55}")
for task, result in results.items():
    print(f"  {task:15s}: {result}")

all_passed = all('PASSED' in r for r in results.values())
if all_passed:
    print("\n🎉 ALL TASKS PASSED — DataLoader is ready!")
else:
    print("\n⚠️  Some tasks failed — fix before proceeding to model training.")

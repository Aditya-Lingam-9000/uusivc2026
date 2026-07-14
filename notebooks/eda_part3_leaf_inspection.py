import os, json, glob, random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

import argparse
import sys

# ── Parse Paths from CLI or Notebook globals ─────────────────
def get_paths():
    # default guesses based on user's actual paths
    default_train = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN"
    default_val = "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL"
    
    # Check if globals exist (useful when running via exec() from a notebook)
    train_path = globals().get('TRAIN_PATH', None)
    val_path = globals().get('VAL_PATH', None)
    
    if train_path and val_path:
        print(f"Using TRAIN_PATH and VAL_PATH from notebook globals:")
        print(f"  TRAIN: {train_path}")
        print(f"  VAL:   {val_path}")
        return train_path, val_path
        
    # Check if run from command line with CLI arguments
    has_args = any(arg.startswith('--train') or arg.startswith('--val') for arg in sys.argv)
    
    if has_args:
        parser = argparse.ArgumentParser()
        parser.add_argument("--train", type=str, default=default_train)
        parser.add_argument("--val", type=str, default=default_val)
        args, _ = parser.parse_known_args()
        print(f"Parsed CLI arguments:")
        print(f"  TRAIN: {args.train}")
        print(f"  VAL:   {args.val}")
        return args.train, args.val
        
    # Fallback to defaults
    print(f"Using default paths:")
    print(f"  TRAIN: {default_train}")
    print(f"  VAL:   {default_val}")
    return default_train, default_val

TRAIN, VAL = get_paths()
OUT = "/kaggle/working/inspections"
os.makedirs(OUT, exist_ok=True)

print(f"TRAIN: {TRAIN}")
print(f"VAL:   {VAL}")

# Leaf subdirectory inspection
leaf_dirs = []
for root, dirs, files in os.walk(TRAIN):
    if not dirs and len(files) > 0:
        leaf_dirs.append(root)

print(f"\nFound {len(leaf_dirs)} leaf subdirectories in TRAIN. Let's inspect one sample from each:")

for i, ld in enumerate(sorted(leaf_dirs)):
    rel_path = os.path.relpath(ld, TRAIN)
    files = [f for f in os.listdir(ld) if os.path.isfile(os.path.join(ld, f))]
    if not files: continue
    
    sample_file = random.choice(files)
    sample_path = os.path.join(ld, sample_file)
    ext = Path(sample_file).suffix.lower()
    
    print(f"\n{'='*60}")
    print(f"Directory [{i+1}/{len(leaf_dirs)}]: {rel_path}")
    print(f"Sample File: {sample_file} ({ext})")
    
    if ext in ['.png', '.jpg']:
        try:
            img = Image.open(sample_path)
            arr = np.array(img)
            print(f"  Image Shape: {arr.shape} | Max Val: {arr.max()} | Min Val: {arr.min()}")
            
            # Save a plot of the image
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(arr, cmap='gray' if len(arr.shape) == 2 else None)
            ax.set_title(f"Sample:\n{rel_path}", fontsize=8)
            ax.axis('off')
            fig.savefig(f"{OUT}/sample_{i+1:02d}_{rel_path.replace(os.sep, '_')}.png", bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"  Error loading image: {e}")
            
    elif ext == '.npy':
        try:
            arr = np.load(sample_path)
            print(f"  Numpy Array Shape: {arr.shape} | Dtype: {arr.dtype} | Max Val: {arr.max()} | Min Val: {arr.min()}")
            if arr.ndim >= 2:
                # Plot midframe
                fig, ax = plt.subplots(figsize=(4, 4))
                frame = arr[arr.shape[0]//2] if arr.ndim == 3 else arr
                ax.imshow(frame, cmap='gray')
                ax.set_title(f"NPY midframe:\n{rel_path}", fontsize=8)
                ax.axis('off')
                fig.savefig(f"{OUT}/sample_{i+1:02d}_{rel_path.replace(os.sep, '_')}.png", bbox_inches='tight')
                plt.close(fig)
        except Exception as e:
            print(f"  Error loading .npy file: {e}")
            
    elif ext == '.npz':
        try:
            data = np.load(sample_path, allow_pickle=True)
            print(f"  NPZ keys: {list(data.keys())}")
            for k in data.keys():
                arr = data[k]
                if arr.dtype == object:
                    inner = arr.item()
                    print(f"    Key '{k}' contains object: {type(inner)}")
                    if isinstance(inner, dict):
                        print(f"      Dictionary has {len(inner)} frame keys (e.g., {list(inner.keys())[:3]}...)")
                else:
                    print(f"    Key '{k}' shape: {arr.shape} | Dtype: {arr.dtype} | Unique: {np.unique(arr)[:10]}")
        except Exception as e:
            print(f"  Error loading .npz file: {e}")
            
    elif ext == '.json':
        try:
            with open(sample_path) as f:
                js = json.load(f)
            print(f"  JSON Type: {type(js)}")
            if isinstance(js, dict):
                print(f"  JSON Keys: {list(js.keys())[:10]}")
            elif isinstance(js, list):
                print(f"  JSON List Length: {len(js)}")
                print(f"  First item keys: {list(js[0].keys()) if isinstance(js[0], dict) else 'Not a dict'}")
        except Exception as e:
            print(f"  Error loading JSON: {e}")

print(f"\nInspection visualizations saved in: {OUT}")

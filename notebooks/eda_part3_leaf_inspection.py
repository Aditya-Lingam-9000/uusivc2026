import os, json, glob, random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Detect paths
def find_dataset_dirs():
    train_dir, val_dir = None, None
    for root, dirs, files in os.walk("/kaggle/input"):
        for d in dirs:
            if d.upper() == "TRAIN": train_dir = os.path.join(root, d)
            elif d.upper() == "VAL": val_dir = os.path.join(root, d)
    return train_dir, val_dir

TRAIN, VAL = find_dataset_dirs()
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

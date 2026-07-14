# ============================================================
# UUSIVC 2026 — Deep EDA Part 2: NPY + NPZ Files (CEUS & Video)
# Run AFTER Part 1. Same Kaggle session.
# Attach both datasets:
#   /kaggle/input/uusivc-train-zip/
#   /kaggle/input/uusivc-val-zip/
# ============================================================

import os, glob, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

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
OUT   = "/kaggle/working/eda_outputs"
os.makedirs(OUT, exist_ok=True)

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", bbox_inches='tight', dpi=120)
    plt.close(fig)
    print(f"  ✓ saved {name}.png")

def find(base, ext):
    if not base or not os.path.exists(base):
        return []
    return sorted(glob.glob(f"{base}/**/*{ext}", recursive=True))

# ============================================================
# SECTION 5 — NPY FILES (CEUS videos & cardiac videos)
# ============================================================
print("\n" + "="*60)
print("SECTION 5: NPY FILES — Shape, dtype, value range")
print("="*60)

npy_files = find(TRAIN, ".npy")
print(f"Total .npy files in TRAIN: {len(npy_files)}")

# One sample per task+organ
npy_samples = {}
for fpath in npy_files:
    parts = list(Path(fpath).parts)
    for task in ['ceus_cls','ceus_seg','video_seg']:
        if task in parts:
            idx = parts.index(task)
            organ = parts[idx+1] if idx+1 < len(parts) else 'unknown'
            key = f"TRAIN/{task}/{organ}"
            if key not in npy_samples:
                npy_samples[key] = fpath
            break

# Also from VAL
for fpath in find(VAL, ".npy"):
    parts = list(Path(fpath).parts)
    for task in ['ceus_cls','ceus_seg','video_seg']:
        if task in parts:
            idx = parts.index(task)
            organ = parts[idx+1] if idx+1 < len(parts) else 'unknown'
            key = f"VAL/{task}/{organ}"
            if key not in npy_samples:
                npy_samples[key] = fpath
            break

print(f"\nSampling {len(npy_samples)} unique task/organ NPY files:")
npy_stats = []
for key, fpath in sorted(npy_samples.items()):
    try:
        arr = np.load(fpath)
        mn, mx, me = float(arr.min()), float(arr.max()), float(arr.mean())
        print(f"  {key:45s}  shape={arr.shape}  dtype={arr.dtype}  range=[{mn:.2f},{mx:.2f}]  mean={me:.2f}")
        npy_stats.append({'key': key, 'shape': arr.shape, 'dtype': str(arr.dtype),
                          'min': mn, 'max': mx, 'mean': me, 'path': fpath})
    except Exception as e:
        print(f"  ERROR {key}: {e}")

# ── Figure 4: visualise NPY frames ───────────────────────────
n = len(npy_stats)
if n > 0:
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = np.array(axes).flatten()
    for ax in axes: ax.axis('off')

    for i, s in enumerate(npy_stats):
        try:
            arr = np.load(s['path'])
            # Normalise for display
            a = arr.astype(np.float32)

            if a.ndim == 2:
                # single frame grayscale
                frame = a
            elif a.ndim == 3:
                # could be (T, H, W) or (H, W, C)
                if a.shape[0] < a.shape[1] and a.shape[0] < a.shape[2]:
                    # likely (T, H, W)
                    mid = a.shape[0] // 2
                    frame = a[mid]
                else:
                    # likely (H, W, C)
                    frame = a[:, :, 0] if a.shape[2] > 1 else a[:, :, 0]
            elif a.ndim == 4:
                # (T, H, W, C) or (B, T, H, W)
                frame = a[a.shape[0]//2, ..., 0] if a.shape[-1] in [1,3] else a[0, a.shape[1]//2]
            else:
                frame = a.reshape(a.shape[-2], a.shape[-1])

            # clip + normalize to 0-1
            lo, hi = np.percentile(frame, 1), np.percentile(frame, 99)
            if hi > lo:
                frame = np.clip((frame - lo) / (hi - lo), 0, 1)

            axes[i].imshow(frame, cmap='gray')
            title = s['key'].replace('TRAIN/','T/').replace('VAL/','V/')
            axes[i].set_title(f"{title}\n{s['shape']} {s['dtype']}", fontsize=6)
        except Exception as e:
            axes[i].set_title(f"ERR: {e}", fontsize=6)

    fig.suptitle("NPY File Visualisation — CEUS & Video (mid-frame)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    save(fig, "04_npy_frames")
else:
    print("Warning: No NPY stats collected, skipping 04_npy_frames.png")

# ============================================================
# SECTION 6 — NPZ FILES (CEUS + Video Annotations)
# ============================================================
print("\n" + "="*60)
print("SECTION 6: NPZ FILES — Keys, shapes, mask values")
print("="*60)

npz_files = find(TRAIN, ".npz")
print(f"Total .npz files in TRAIN: {len(npz_files)}")

npz_samples = {}
for fpath in npz_files:
    parts = list(Path(fpath).parts)
    for task in ['ceus_seg','video_seg']:
        if task in parts:
            idx = parts.index(task)
            organ = parts[idx+1] if idx+1 < len(parts) else 'unknown'
            key = f"{task}/{organ}"
            if key not in npz_samples:
                npz_samples[key] = fpath
            break

print(f"\nSampling {len(npz_samples)} unique NPZ files:")
npz_stats = []
for key, fpath in sorted(npz_samples.items()):
    try:
        npz = np.load(fpath, allow_pickle=True)
        keys = list(npz.keys())
        print(f"\n  {key}")
        print(f"    File: {Path(fpath).name}")
        print(f"    NPZ keys: {keys}")
        for k in keys:
            arr = npz[k]
            # allow_pickle means arr could be object array
            if arr.dtype == object:
                inner = arr.item()
                if isinstance(inner, dict):
                    sub_keys = list(inner.keys())[:5]
                    print(f"    key='{k}'  → dict with frame keys: {sub_keys}")
                    # show one frame's mask shape
                    for sk in sub_keys[:1]:
                        m = inner[sk]
                        m_arr = np.array(m)
                        print(f"      frame '{sk}': shape={m_arr.shape}  dtype={m_arr.dtype}  unique={np.unique(m_arr)}")
                else:
                    print(f"    key='{k}'  → object type: {type(inner)}")
            else:
                print(f"    key='{k}'  shape={arr.shape}  dtype={arr.dtype}  unique={np.unique(arr)[:8]}")
        npz_stats.append({'key': key, 'npz_keys': keys, 'path': fpath})
    except Exception as e:
        print(f"  ERROR {key}: {e}")

# ── Figure 5: NPZ mask visualisation ─────────────────────────
if len(npz_samples) > 0:
    fig, axes = plt.subplots(2, len(npz_samples), figsize=(len(npz_samples)*3, 6))
    if len(npz_samples) == 1:
        axes = axes.reshape(2, 1)
    axes = np.array(axes)

    for j, (key, fpath) in enumerate(sorted(npz_samples.items())):
        try:
            npz = np.load(fpath, allow_pickle=True)
            k = list(npz.keys())[0]
            arr = npz[k]

            if arr.dtype == object:
                inner = arr.item()
                if isinstance(inner, dict):
                    frame_keys = list(inner.keys())
                    # show first and middle frame mask
                    for plot_i, fk in enumerate(frame_keys[:2]):
                        mask = np.array(inner[fk])
                        axes[plot_i, j].imshow(mask, cmap='jet')
                        axes[plot_i, j].set_title(f"{key}\nframe={fk}", fontsize=6)
                        axes[plot_i, j].axis('off')
            else:
                # direct 2D mask
                axes[0, j].imshow(arr if arr.ndim == 2 else arr[0], cmap='jet')
                axes[0, j].set_title(f"{key}\nshape={arr.shape}", fontsize=6)
                axes[0, j].axis('off')
                axes[1, j].axis('off')
        except Exception as e:
            axes[0, j].set_title(f"ERR: {e}", fontsize=6)
            axes[1, j].axis('off')

    fig.suptitle("NPZ Annotation Masks — ceus_seg & video_seg", fontsize=12, fontweight='bold')
    plt.tight_layout()
    save(fig, "05_npz_masks")
else:
    print("Warning: No NPZ samples found, skipping 05_npz_masks.png")

# ============================================================
# SECTION 7 — CLASS DISTRIBUTION (TRAIN)
# ============================================================
print("\n" + "="*60)
print("SECTION 7: CLASS IMBALANCE — Train Label Distribution")
print("="*60)

cls_counts = {}

# From folder structure: 0/ and 1/ subfolders
for task in ['image_cls', 'ceus_cls']:
    for cls_dir in ['0', '1']:
        for fpath in find(TRAIN, ''):
            pass  # reuse structured approach below

# Count by walking the folder
for task in ['image_cls', 'ceus_cls']:
    task_dirs = glob.glob(f"{TRAIN}/**/{task}", recursive=True)
    for td in task_dirs:
        organs = [d for d in os.listdir(td) if os.path.isdir(os.path.join(td, d))]
        for organ in organs:
            organ_path = os.path.join(td, organ)
            # check for class folders
            c0 = glob.glob(f"{organ_path}/0/*")
            c1 = glob.glob(f"{organ_path}/1/*")
            imgs_dir = glob.glob(f"{organ_path}/imgs/*")
            if c0 or c1:
                key = f"{task}/{organ}"
                cls_counts[key] = {'class_0': len(c0), 'class_1': len(c1)}
                print(f"  {key:35s}  class0={len(c0):4d}  class1={len(c1):4d}  ratio={len(c1)/(len(c0)+1e-6):.2f}")
            elif imgs_dir:
                # labels from JSON — just count files
                key = f"{task}/{organ}"
                cls_counts[key] = {'from_json': len(imgs_dir)}
                print(f"  {key:35s}  imgs={len(imgs_dir)} (labels from JSON)")

# ── Figure 6: class balance bar chart ────────────────────────
keys_with_classes = {k: v for k, v in cls_counts.items() if 'class_0' in v}
if keys_with_classes:
    labels = list(keys_with_classes.keys())
    c0s = [keys_with_classes[k]['class_0'] for k in labels]
    c1s = [keys_with_classes[k]['class_1'] for k in labels]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(max(10, len(labels)*1.2), 6))
    ax.bar(x, c0s, label='Class 0 (Benign/Negative)', color='steelblue')
    ax.bar(x, c1s, bottom=c0s, label='Class 1 (Malignant/Positive)', color='tomato')
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Sample Count")
    ax.set_title("Class Distribution — image_cls & ceus_cls (Train)")
    ax.legend()
    plt.tight_layout()
    save(fig, "06_class_distribution")

# ============================================================
# SECTION 8 — IMAGE RESOLUTION HISTOGRAMS
# ============================================================
print("\n" + "="*60)
print("SECTION 8: RESOLUTION DISTRIBUTIONS (sample 200 per task)")
print("="*60)

resolution_data = {}  # task -> list of (H, W)

for task in ['image_cls', 'image_seg']:
    task_dirs = glob.glob(f"{TRAIN}/**/{task}", recursive=True)
    hs, ws = [], []
    for td in task_dirs:
        imgs = find(td, '.png')[:50] + find(td, '.jpg')[:50]
        for fp in imgs[:200]:
            if '/masks/' in fp:
                continue
            try:
                img = Image.open(fp)
                ws.append(img.width)
                hs.append(img.height)
            except: pass
    if hs:
        resolution_data[task] = (hs, ws)
        print(f"  {task}: H={min(hs)}–{max(hs)} (median {int(np.median(hs))})"
              f"  W={min(ws)}–{max(ws)} (median {int(np.median(ws))})")

# Also from VAL
for task in ['image_cls', 'image_seg']:
    task_dirs = glob.glob(f"{VAL}/**/{task}", recursive=True)
    hs, ws = [], []
    for td in task_dirs:
        imgs = find(td, '.png')[:30] + find(td, '.jpg')[:30]
        for fp in imgs[:100]:
            try:
                img = Image.open(fp)
                ws.append(img.width); hs.append(img.height)
            except: pass
    if hs:
        key = f"VAL/{task}"
        resolution_data[key] = (hs, ws)
        print(f"  VAL {task}: H={min(hs)}–{max(hs)} W={min(ws)}–{max(ws)}")

if resolution_data:
    n = len(resolution_data)
    fig, axes = plt.subplots(2, n, figsize=(n*5, 8))
    if n == 1: axes = axes.reshape(2, 1)
    for j, (task, (hs, ws)) in enumerate(resolution_data.items()):
        axes[0, j].hist(hs, bins=30, color='steelblue', edgecolor='white')
        axes[0, j].set_title(f"{task}\nHeight Distribution", fontsize=9)
        axes[0, j].set_xlabel("Height (px)")
        axes[1, j].hist(ws, bins=30, color='tomato', edgecolor='white')
        axes[1, j].set_title(f"Width Distribution", fontsize=9)
        axes[1, j].set_xlabel("Width (px)")
    fig.suptitle("Resolution Histograms — Image Tasks", fontsize=12, fontweight='bold')
    plt.tight_layout()
    save(fig, "07_resolution_histograms")

# ============================================================
# SECTION 9 — CEUS+VIDEO: Frame count distribution
# ============================================================
print("\n" + "="*60)
print("SECTION 9: NPY VIDEO — Frame Count Distribution")
print("="*60)

frame_counts = {}
for task in ['ceus_seg', 'ceus_cls', 'video_seg']:
    task_dirs = glob.glob(f"{TRAIN}/**/{task}", recursive=True)
    for td in task_dirs:
        npy_list = find(td, '.npy')[:50]  # sample 50
        fc = []
        for fp in npy_list:
            try:
                arr = np.load(fp)
                # T is first dim if T < H,W else last
                fc.append(arr.shape[0])
            except: pass
        if fc:
            organs = [d for d in os.listdir(td) if os.path.isdir(os.path.join(td, d))]
            key = f"{task}"
            if key not in frame_counts:
                frame_counts[key] = []
            frame_counts[key].extend(fc)
            print(f"  {task}: frames min={min(fc)} max={max(fc)} median={int(np.median(fc))}")

if frame_counts:
    n = len(frame_counts)
    fig, axes = plt.subplots(1, n, figsize=(n*5, 4))
    if n == 1: axes = [axes]
    for ax, (task, fc) in zip(axes, frame_counts.items()):
        ax.hist(fc, bins=20, color='mediumseagreen', edgecolor='white')
        ax.set_title(f"{task}\nFrame Counts", fontsize=9)
        ax.set_xlabel("Frames per sample")
    fig.suptitle("Video / CEUS Frame Count Distribution", fontsize=12, fontweight='bold')
    plt.tight_layout()
    save(fig, "08_frame_count_distribution")

# ============================================================
# SECTION 10 — FINAL SUMMARY TABLE
# ============================================================
print("\n" + "="*60)
print("SECTION 10: FINAL SUMMARY")
print("="*60)

summary = {
    "TRAIN_total_files": 24652,
    "VAL_total_files": 3229,
    "Tasks": ["image_cls", "image_seg", "ceus_cls", "ceus_seg", "video_seg"],
    "Private_organs_image_seg": ["Breast", "Breast_luminal", "Cardiac", "Fetal_Head", "Kidney", "Prostate", "Thyroid"],
    "Public_datasets": {
        "image_cls": ["Appendix", "BUS-BRA", "BUSI", "Fatty-Liver"],
        "image_seg": ["BUS-BRA", "BUSIS", "DDTI", "Fetal_HC", "KidneyUS"],
        "video_seg": ["CAMUS"]
    },
    "Gotchas": [
        "Prostate image_cls has imgs/ subfolder NOT 0/1 folders — labels from JSON",
        "CAUS uses .npz for video, CardiacCH uses .npy",
        "Val has NO masks or class labels — input only",
        "Breast_luminal is private-only, not in competition docs"
    ]
}

with open(f"{OUT}/dataset_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print(f"\n{'='*60}")
print(f"✅ EDA PART 2 COMPLETE!")
print(f"All outputs saved to: {OUT}")
print(f"Files created:")
for f in sorted(os.listdir(OUT)):
    size = os.path.getsize(f"{OUT}/{f}")
    print(f"  {f}  ({size//1024} KB)")

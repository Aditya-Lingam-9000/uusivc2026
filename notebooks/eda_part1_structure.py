# ============================================================
# UUSIVC 2026 — Deep EDA Part 1: Structure + JSON + Images
# Run this on Kaggle (CPU session fine, no GPU needed)
# Attach both datasets:
#   /kaggle/input/uusivc-train-zip/
#   /kaggle/input/uusivc-val-zip/
# ============================================================

import os, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# ── Dynamic Directory Detection ──────────────────────────────
def find_dataset_dirs():
    train_dir = None
    val_dir = None
    
    # Scan /kaggle/input recursively for directories named TRAIN and VAL (case insensitive)
    for root, dirs, files in os.walk("/kaggle/input"):
        for d in dirs:
            if d.upper() == "TRAIN":
                train_dir = os.path.join(root, d)
            elif d.upper() == "VAL":
                val_dir = os.path.join(root, d)
                
    # If not found, let's print what is in /kaggle/input and try to fallback to the first folder
    if not train_dir or not val_dir:
        print("WARNING: Could not automatically detect TRAIN or VAL folders in /kaggle/input.")
        print("Contents of /kaggle/input:")
        for p in glob.glob("/kaggle/input/**/*", recursive=True):
            print(f"  {p}")
            if "TRAIN" in p.upper() and os.path.isdir(p) and not train_dir:
                train_dir = p
            if "VAL" in p.upper() and os.path.isdir(p) and not val_dir:
                val_dir = p
                
    print(f"Detected TRAIN path: {train_dir}")
    print(f"Detected VAL path:   {val_dir}")
    return train_dir, val_dir

TRAIN, VAL = find_dataset_dirs()
OUT   = "/kaggle/working/eda_outputs"
os.makedirs(OUT, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────
def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", bbox_inches='tight', dpi=120)
    plt.close(fig)
    print(f"  ✓ saved {name}.png")

def find(base, ext):
    if not base or not os.path.exists(base):
        return []
    return sorted(glob.glob(f"{base}/**/*{ext}", recursive=True))

# ============================================================
# SECTION 1 — JSON FILES INSPECTION
# ============================================================
print("\n" + "="*60)
print("SECTION 1: JSON FILES")
print("="*60)

json_files = find(TRAIN, ".json")
print(f"Found {len(json_files)} JSON files:")
for jf in json_files:
    print(f"\n{'─'*50}")
    print(f"FILE: {Path(jf).name}")
    print(f"PATH: {jf}")
    with open(jf) as f:
        data = json.load(f)
    if isinstance(data, list):
        print(f"TYPE: list  |  LENGTH: {len(data)}")
        print(f"FIRST ITEM:\n{json.dumps(data[0], indent=2)}")
        if len(data) > 1:
            print(f"SECOND ITEM:\n{json.dumps(data[1], indent=2)}")
        # field analysis
        all_keys = set()
        for item in data:
            if isinstance(item, dict):
                all_keys.update(item.keys())
        print(f"ALL KEYS: {all_keys}")
        # count by task
        if 'task' in all_keys:
            tc = Counter(item.get('task','?') for item in data)
            print(f"BY TASK: {dict(tc)}")
        if 'dataset_name' in all_keys:
            dc = Counter(item.get('dataset_name','?') for item in data)
            print(f"BY DATASET: {dict(dc)}")
    elif isinstance(data, dict):
        print(f"TYPE: dict  |  KEYS: {list(data.keys())[:10]}")
        # show first 2 entries
        for i,(k,v) in enumerate(data.items()):
            if i >= 2: break
            print(f"  KEY={k}  VAL={json.dumps(v)[:200]}")

# Val JSON
val_json = find(VAL, ".json")
print(f"\nVAL JSON files: {val_json}")
for jf in val_json:
    print(f"\nFILE: {Path(jf).name}")
    with open(jf) as f:
        data = json.load(f)
    if isinstance(data, list):
        print(f"TYPE: list  |  LENGTH: {len(data)}")
        print(f"FIRST ITEM:\n{json.dumps(data[0], indent=2)}")
        all_keys = set()
        for item in data: all_keys.update(item.keys() if isinstance(item,dict) else [])
        print(f"ALL KEYS: {all_keys}")
        tc = Counter(item.get('task','?') for item in data if isinstance(item,dict))
        dc = Counter(item.get('dataset_name','?') for item in data if isinstance(item,dict))
        print(f"BY TASK: {dict(tc)}")
        print(f"BY DATASET: {dict(dc)}")
    elif isinstance(data, dict):
        print(f"TYPE: dict  |  KEYS: {list(data.keys())[:10]}")
        for i,(k,v) in enumerate(data.items()):
            if i >= 2: break
            print(f"  KEY={k}  VAL={json.dumps(v)[:200]}")

# ============================================================
# SECTION 2 — YAML / CONFIG FILES
# ============================================================
print("\n" + "="*60)
print("SECTION 2: YAML CONFIG FILES")
print("="*60)
yaml_files = find(TRAIN, ".yaml")
seen_dirs = set()
for yf in yaml_files:
    d = str(Path(yf).parent)
    if d in seen_dirs: continue
    seen_dirs.add(d)
    print(f"\nFILE: {yf}")
    with open(yf) as f:
        print(f.read()[:600])

# ============================================================
# SECTION 3 — IMAGE EXPLORATION (PNG + JPG)
# ============================================================
print("\n" + "="*60)
print("SECTION 3: IMAGE FILES — SHAPES, RANGES, VISUAL")
print("="*60)

# Collect one sample image per dataset/organ folder
img_samples = {}

def collect_image_samples(base, split_name):
    for ext in ['.png', '.jpg']:
        for fpath in find(base, ext):
            parts = Path(fpath).parts
            # find task + dataset_name from path
            for task in ['image_cls','image_seg','ceus_cls']:
                if task in parts:
                    idx = list(parts).index(task)
                    if idx+1 < len(parts):
                        organ = parts[idx+1]
                        key = f"{split_name}/{task}/{organ}"
                        if key not in img_samples:
                            img_samples[key] = fpath

collect_image_samples(TRAIN, "TRAIN")
collect_image_samples(VAL,   "VAL")

print(f"\nCollected {len(img_samples)} unique organ/task image samples")

# stats per sample
stats = []
for key, fpath in sorted(img_samples.items()):
    try:
        img = np.array(Image.open(fpath).convert('RGB'))
        h, w, c = img.shape
        stats.append({
            'key': key,
            'path': fpath,
            'H': h, 'W': w, 'C': c,
            'min': int(img.min()),
            'max': int(img.max()),
            'mean': round(float(img.mean()), 1)
        })
        print(f"  {key:50s}  shape={h}×{w}  range=[{img.min()},{img.max()}]  mean={img.mean():.1f}")
    except Exception as e:
        print(f"  ERROR {key}: {e}")

# ── Figure 1: grid of sample images ──────────────────────────
n = len(img_samples)
if n > 0:
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = np.array(axes).flatten()
    for ax in axes: ax.axis('off')

    for i, (key, fpath) in enumerate(sorted(img_samples.items())):
        try:
            img = np.array(Image.open(fpath).convert('RGB'))
            axes[i].imshow(img, cmap='gray' if img.ndim==2 else None)
            label = key.split('/')[-2] + '/' + key.split('/')[-1]
            axes[i].set_title(label, fontsize=7)
        except: pass

    fig.suptitle("Sample Images — All Organs/Tasks (Train + Val)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    save(fig, "01_image_samples_grid")
else:
    print("Warning: No image samples found, skipping 01_image_samples_grid.png")

# ── Figure 2: resolution scatter ─────────────────────────────
if stats:
    fig, ax = plt.subplots(figsize=(10, 6))
    Hs = [s['H'] for s in stats]
    Ws = [s['W'] for s in stats]
    labels_u = list(set(s['key'].split('/')[0] for s in stats))
    colors = {'TRAIN':'steelblue', 'VAL':'tomato'}
    for s in stats:
        split = s['key'].split('/')[0]
        ax.scatter(s['W'], s['H'], c=colors.get(split,'gray'), alpha=0.8, s=80)
        ax.annotate(s['key'].split('/')[-1], (s['W'], s['H']), fontsize=6, ha='center')
    handles = [mpatches.Patch(color=c, label=l) for l,c in colors.items()]
    ax.legend(handles=handles)
    ax.set_xlabel("Width (px)"); ax.set_ylabel("Height (px)")
    ax.set_title("Image Resolution Scatter — All Sampled Organs")
    plt.tight_layout()
    save(fig, "02_resolution_scatter")

# ============================================================
# SECTION 4 — MASK EXPLORATION (image_seg masks)
# ============================================================
print("\n" + "="*60)
print("SECTION 4: SEGMENTATION MASKS")
print("="*60)

mask_samples = {}
for fpath in find(TRAIN, ".png"):
    if '/masks/' in fpath:
        parts = Path(fpath).parts
        if 'image_seg' in parts:
            idx = list(parts).index('image_seg')
            organ = parts[idx+1] if idx+1 < len(parts) else 'unknown'
            key = f"image_seg/{organ}"
            if key not in mask_samples:
                mask_samples[key] = fpath

print(f"Found mask samples for {len(mask_samples)} organ categories:")
for key, fpath in sorted(mask_samples.items()):
    mask = np.array(Image.open(fpath))
    unique_vals = np.unique(mask)
    print(f"  {key:35s}  shape={mask.shape}  unique_vals={unique_vals}  dtype={mask.dtype}")

# ── Figure 3: image+mask pairs ───────────────────────────────
pairs = []
for key, mpath in sorted(mask_samples.items()):
    organ = key.split('/')[-1]
    base_dir = str(Path(mpath).parent.parent)
    img_candidates = find(os.path.join(base_dir, 'imgs'), '.png') + \
                     find(os.path.join(base_dir, 'imgs'), '.jpg')
    if img_candidates:
        pairs.append((organ, img_candidates[0], mpath))

n = len(pairs)
if n > 0:
    fig, axes = plt.subplots(n, 2, figsize=(8, n*3))
    if n == 1: axes = axes.reshape(1, 2)
    for i, (organ, ipath, mpath) in enumerate(pairs):
        try:
            img  = np.array(Image.open(ipath).convert('L'))
            mask = np.array(Image.open(mpath))
            axes[i,0].imshow(img, cmap='gray'); axes[i,0].set_title(f"{organ} — Image", fontsize=9)
            axes[i,1].imshow(mask, cmap='jet');  axes[i,1].set_title(f"{organ} — Mask (vals={np.unique(mask)})", fontsize=9)
            for ax in axes[i]: ax.axis('off')
        except Exception as e:
            axes[i,0].set_title(f"ERROR: {e}")
    fig.suptitle("Image + Mask Pairs — image_seg (Private Train)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    save(fig, "03_image_mask_pairs")
else:
    print("Warning: No image_seg mask pairs found, skipping 03_image_mask_pairs.png")

print("\n✅ PART 1 COMPLETE — check /kaggle/working/eda_outputs/")
print(f"   Files: 01_image_samples_grid.png, 02_resolution_scatter.png, 03_image_mask_pairs.png")

"""
src/config.py
Central configuration for UUSIVC 2026 competition-grade training.
Every tweak is configurable from here. Override any value by setting
globals before exec()ing a training script.
"""

import os

def get_cfg(task="seg"):
    """Return a config dict. task = 'seg' | 'cls' | 'ceus_cls'"""

    cfg = {
        # ── Paths ────────────────────────────────────────────
        "train_root": os.environ.get("TRAIN_PATH",
            "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN"),
        "val_root": os.environ.get("VAL_PATH",
            "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL"),
        "ckpt_dir": "/kaggle/working/checkpoints",
        "preprocess_dir": "/kaggle/working/preprocessed",

        # ── Task ─────────────────────────────────────────────
        "task": task,                     # "seg" | "cls" | "ceus_cls"
        "seg_tasks": ["image_seg", "ceus_seg", "video_seg"],
        "cls_tasks": ["image_cls"],
        "ceus_cls_tasks": ["ceus_cls"],

        # ── Model ────────────────────────────────────────────
        "encoder_name": "efficientnet-b5",
        "encoder_weights": "imagenet",
        "decoder_type": "UnetPlusPlus",   # "Unet" | "UnetPlusPlus" | "FPN" | "DeepLabV3Plus"
        "in_channels": 3,
        "seg_classes": 1,
        "cls_classes": 2,
        "dropout": 0.3,
        "cls_backbone": "efficientnet_b5",  # timm model name for classification

        # ── Resolution ───────────────────────────────────────
        "img_size_seg": 512,              # for image_seg
        "img_size_ceus_seg": (256, 512),  # (H, W) for ceus_seg — native resolution
        "img_size_video_seg": 256,        # for video_seg
        "img_size_cls": 384,              # for image_cls / ceus_cls

        # ── Training ─────────────────────────────────────────
        "epochs": 50,
        "batch_size": 8,
        "grad_accum_steps": 4,            # effective batch = batch_size * grad_accum
        "num_workers": 4,
        "pin_memory": True,
        "seed": 42,
        "val_split": 0.15,
        "early_stop_patience": 15,        # 0 = disabled

        # ── Optimizer ────────────────────────────────────────
        "optimizer": "adamw",
        "lr": 1e-4,
        "encoder_lr": 1e-5,              # differential LR for pretrained encoder
        "weight_decay": 1e-4,

        # ── Scheduler ────────────────────────────────────────
        "scheduler": "cosine_warm_restart",  # "cosine" | "cosine_warm_restart" | "onecycle"
        "T_0": 10,
        "T_mult": 2,
        "eta_min": 1e-7,
        "warmup_epochs": 3,

        # ── Loss (Segmentation) ──────────────────────────────
        "seg_loss": "dice_focal_boundary",  # "dice_focal" | "dice_focal_boundary"
        "dice_weight": 0.35,
        "focal_weight": 0.35,
        "boundary_weight": 0.30,
        "focal_gamma": 2.0,
        "focal_alpha": 0.25,

        # ── Loss (Classification) ────────────────────────────
        "cls_loss": "focal",               # "ce" | "focal"
        "label_smoothing": 0.1,
        "cls_focal_gamma": 2.0,
        "use_class_weights": True,

        # ── Augmentation ─────────────────────────────────────
        "use_heavy_aug": True,
        "mixup_alpha": 0.2,               # 0 = disabled
        "cutmix_alpha": 1.0,              # 0 = disabled
        "mixup_prob": 0.5,                # probability of applying mixup/cutmix per batch

        # ── Mixed Precision ──────────────────────────────────
        "use_amp": True,

        # ── Gradient Checkpointing ───────────────────────────
        "gradient_checkpointing": False,   # set True to save VRAM (slower)

        # ── EMA ──────────────────────────────────────────────
        "use_ema": True,
        "ema_decay": 0.999,

        # ── Checkpointing ────────────────────────────────────
        "save_best": True,
        "save_latest": True,
        "resume_from": None,              # path to checkpoint to resume from

        # ── CEUS Classification Specific ─────────────────────
        "ceus_n_frames": 16,              # number of frames to sample (was 8)
        "ceus_frame_size": 256,           # resize each frame to this

        # ── CEUS Segmentation Specific ───────────────────────
        "ceus_seg_frames": [3, 7, 11],    # frames to extract (multi-frame)

        # ── Post-processing ──────────────────────────────────
        "tta_enabled": False,             # enable during inference only
        "tta_transforms": ["hflip", "vflip"],
        "morphological_cleanup": True,
        "min_component_size": 50,

        # ── Logging ──────────────────────────────────────────
        "log_every_n_batches": 50,        # print batch progress every N batches
    }

    return cfg


def print_cfg(cfg):
    """Pretty-print config."""
    print("=" * 60)
    print("  CONFIGURATION")
    print("=" * 60)
    max_key_len = max(len(k) for k in cfg)
    for k, v in cfg.items():
        print(f"  {k:{max_key_len}s} : {v}")
    print("=" * 60)

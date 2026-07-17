"""
src/config.py
UUSIVC 2026 — Master Configuration File (v2)

HOW TO USE:
    from src.config import CFG
    # Modify any key before training starts
    CFG["epochs"] = 80
    CFG["resume_from"] = "/kaggle/working/checkpoints/seg_latest.pth"
"""

CFG = {
    # ══════════════════════════════════════════════════════
    #  PATHS  (set at runtime in your notebook)
    # ══════════════════════════════════════════════════════
    "train_path": "/kaggle/input/datasets/jyothiradithyalingam/uusivc-train-zip/TRAIN",
    "val_path":   "/kaggle/input/datasets/jyothiradithyalingam/uusivc-val-zip/VAL",
    "ckpt_dir":   "/kaggle/working/checkpoints",

    # ══════════════════════════════════════════════════════
    #  MODEL ARCHITECTURE
    # ══════════════════════════════════════════════════════
    # Segmentation
    "seg_encoder":        "efficientnet-b5",      # SMP encoder name
    "seg_decoder":        "UnetPlusPlus",          # SMP decoder type: "Unet", "UnetPlusPlus", "FPN", "DeepLabV3Plus"
    "seg_encoder_weights":"imagenet",
    "seg_in_channels":    3,
    "seg_num_classes":    1,                       # binary segmentation
    "seg_dropout":        0.2,

    # Classification
    "cls_encoder":        "efficientnet-b5",      # timm model name or smp encoder
    "cls_dropout":        0.3,
    "cls_num_classes":    2,                       # binary classification
    "cls_hidden_dim":     512,
    "ceus_n_frames":      16,                      # frames to sample from CEUS video (was 8)

    # ══════════════════════════════════════════════════════
    #  INPUT / RESOLUTION
    # ══════════════════════════════════════════════════════
    "img_size":      512,        # Resolution for image_seg (square crop/resize)
    "ceus_seg_h":    256,        # CEUS seg height (native)
    "ceus_seg_w":    512,        # CEUS seg width  (native)
    "video_seg_h":   256,        # Video seg height (native)
    "video_seg_w":   256,        # Video seg width  (native)

    # ══════════════════════════════════════════════════════
    #  TRAINING HYPER-PARAMETERS
    # ══════════════════════════════════════════════════════
    "epochs":            60,
    "batch_size":        8,
    "grad_accum_steps":  4,          # Effective batch = batch_size × grad_accum_steps
    "num_workers":       4,
    "pin_memory":        True,
    "seed":              42,
    "val_split":         0.15,

    # Learning Rate
    "lr":                1e-4,       # Decoder / head LR
    "encoder_lr":        1e-5,       # Pretrained encoder LR (10× smaller)
    "weight_decay":      1e-4,
    "warmup_epochs":     5,          # Linear warmup from lr/10 to lr

    # Scheduler
    "scheduler":         "cosine_warmrestart",  # "cosine" | "cosine_warmrestart" | "step"
    "T_0":               10,         # cosine warm restart period (epochs)
    "T_mult":            2,          # multiplier for next restart period
    "eta_min":           1e-6,

    # Regularisation
    "label_smoothing":   0.1,        # For classification CrossEntropy
    "mixup_alpha":       0.2,        # 0.0 to disable Mixup augmentation
    "cutmix_alpha":      1.0,        # 0.0 to disable CutMix augmentation
    "ema_decay":         0.999,      # Exponential Moving Average of weights
    "grad_clip":         1.0,        # Max gradient norm (0 to disable)

    # ══════════════════════════════════════════════════════
    #  LOSS FUNCTIONS
    # ══════════════════════════════════════════════════════
    # Segmentation loss: weighted sum of Dice + Focal + Boundary
    "seg_dice_weight":     0.4,
    "seg_focal_weight":    0.3,
    "seg_boundary_weight": 0.3,

    # Classification loss: Focal + Label Smoothing
    "cls_focal_gamma":   2.0,
    "cls_focal_alpha":   0.25,       # Overridden per-organ by class weights

    # ══════════════════════════════════════════════════════
    #  AMP / MEMORY OPTIMIZATION
    # ══════════════════════════════════════════════════════
    "use_amp":                  True,   # Mixed precision FP16 (saves 30-50% VRAM)
    "gradient_checkpointing":   False,  # Save activation memory (slows training 20%)

    # ══════════════════════════════════════════════════════
    #  AUGMENTATION
    # ══════════════════════════════════════════════════════
    "aug_hflip_p":              0.5,
    "aug_vflip_p":              0.3,
    "aug_rotate_limit":         30,
    "aug_shift_scale_rotate_p": 0.7,
    "aug_elastic_p":            0.3,
    "aug_grid_distort_p":       0.3,
    "aug_clahe_p":              0.5,
    "aug_brightness_p":         0.5,
    "aug_gauss_noise_p":        0.3,
    "aug_coarse_dropout_p":     0.3,

    # ══════════════════════════════════════════════════════
    #  CHECKPOINTING & RESUME
    # ══════════════════════════════════════════════════════
    "save_best":         True,   # Save best.pth when val metric improves
    "save_latest":       True,   # Save latest.pth every epoch (for resume)
    "resume_from":       None,   # Path to checkpoint to resume from (e.g. "latest.pth")

    # ══════════════════════════════════════════════════════
    #  LOGGING
    # ══════════════════════════════════════════════════════
    "log_steps":         25,     # Print step-level log every N steps
    "log_vram":          True,   # Print VRAM usage in logs

    # ══════════════════════════════════════════════════════
    #  INFERENCE / POST-PROCESSING
    # ══════════════════════════════════════════════════════
    "seg_threshold":     0.5,    # Default sigmoid threshold
    "tta_enabled":       True,   # Test-Time Augmentation
    "tta_hflip":         True,
    "tta_vflip":         True,
    "tta_scale_075":     False,  # Multi-scale TTA (slower, better)
    "tta_scale_125":     False,
    "morph_cleanup":     True,   # Morphological post-processing
    "morph_iterations":  2,
    "keep_largest_cc":   True,   # Keep only largest connected component
}

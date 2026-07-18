import os

CFG = {
    # ── General ──────────────────
    "seed": 42,
    "num_workers": 4,
    "ckpt_dir": "/kaggle/working/checkpoints",
    
    # ── Data & Augmentations ─────
    "img_size_seg": 512,  # Segmentation resolution
    "img_size_cls": 512,  # Classification resolution
    "in_channels": 3,
    "aug_prob_heavy": 0.5,
    "aug_prob_light": 0.3,
    
    # ── Model (Segmentation) ─────
    "seg_encoder_name": "efficientnet-b5",
    "seg_decoder_type": "UnetPlusPlus",
    "seg_encoder_weights": "imagenet",
    "seg_num_classes": 1,
    
    # ── Model (Classification) ───
    "cls_backbone": "efficientnet-b5",
    "cls_pretrained": True,
    
    # ── Training ─────────────────
    "epochs": 40,
    "batch_size": 16,
    "grad_accum_steps": 2,  # Effective batch = 32
    
    # ── Optimization ─────────────
    "lr": 2e-4,
    "encoder_lr_ratio": 0.1,  # Pretrained encoder gets 10x smaller LR
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
    
    # ── Scheduler ────────────────
    "scheduler": "CosineAnnealingWarmRestarts",
    "T_0": 10,
    "T_mult": 2,
    "eta_min": 1e-6,
    
    # ── Loss (Segmentation) ──────
    "seg_loss_weights": {
        "dice": 0.4,
        "focal": 0.4,
        "bce": 0.2
        # Note: Boundary loss could be added here, simplified for initial V2
    },
    
    # ── Loss (Classification) ────
    "focal_gamma": 2.0,
    "label_smoothing": 0.1,
    
    # ── Mixup / CutMix ───────────
    "mixup_prob": 0.5,
    "mixup_alpha": 0.2,
    
    # ── Memory & Performance ─────
    "use_amp": True,
    
    # ── Checkpointing ────────────
    "save_best": True,
    "save_latest": True,
    "resume_from": None,  # e.g. "/kaggle/working/checkpoints/latest.pth"
    
    # ── Logging ──────────────────
    "log_steps": 25,      # Print stats every 25 steps
}

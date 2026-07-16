"""
src/trainer.py
Universal trainer with AMP, gradient accumulation, resume, EMA, detailed logging.
"""

import os, time, json, copy
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict


class EMA:
    """Exponential Moving Average of model weights."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def get_vram_usage():
    """Return current VRAM usage string."""
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_mem / 1e9
        return f"{used:.1f}/{total:.1f} GB ({100*used/total:.1f}%)"
    return "N/A"


def format_time(seconds):
    """Format seconds to human readable."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m{seconds%60:.0f}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m"


def build_optimizer(model, cfg):
    """Build optimizer with differential learning rates."""
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    param_groups = [
        {"params": decoder_params, "lr": cfg["lr"]},
        {"params": encoder_params, "lr": cfg["encoder_lr"]},
    ]
    return torch.optim.AdamW(param_groups, weight_decay=cfg["weight_decay"])


def build_scheduler(optimizer, cfg):
    """Build LR scheduler."""
    if cfg["scheduler"] == "cosine_warm_restart":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg["T_0"], T_mult=cfg["T_mult"], eta_min=cfg["eta_min"],
        )
    elif cfg["scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"], eta_min=cfg["eta_min"],
        )
    elif cfg["scheduler"] == "onecycle":
        # Note: needs total_steps — caller must set this after creating dataloaders
        return None
    else:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"], eta_min=cfg["eta_min"],
        )


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_metric,
                    history, cfg, is_best=False):
    """Save best and/or latest checkpoint."""
    m = model.module if hasattr(model, "module") else model
    state = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": m.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "history": history,
        "cfg": cfg,
    }
    prefix = cfg.get("ckpt_prefix", cfg["task"])
    ckpt_dir = cfg["ckpt_dir"]

    if cfg["save_latest"]:
        path = os.path.join(ckpt_dir, f"{prefix}_latest.pth")
        torch.save(state, path)

    if is_best and cfg["save_best"]:
        path = os.path.join(ckpt_dir, f"{prefix}_best.pth")
        torch.save(state, path)
        print(f"  💾 Best model saved (metric={best_metric:.4f}) → {path}")


def load_checkpoint(model, optimizer, scheduler, scaler, cfg):
    """Resume from checkpoint. Returns start_epoch, best_metric, history."""
    resume_path = cfg.get("resume_from")
    if not resume_path or not os.path.exists(resume_path):
        return 1, 0.0, []

    print(f"  📂 Resuming from {resume_path}")
    ckpt = torch.load(resume_path, map_location="cpu")

    m = model.module if hasattr(model, "module") else model
    m.load_state_dict(ckpt["model_state_dict"])

    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    start_epoch = ckpt["epoch"] + 1
    best_metric = ckpt.get("best_metric", 0.0)
    history = ckpt.get("history", [])

    print(f"  ✅ Resumed from epoch {ckpt['epoch']}, best_metric={best_metric:.4f}")
    return start_epoch, best_metric, history


def dice_score_fn(logits, targets, threshold=0.5):
    """Compute batch-level Dice score."""
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        intersection = (preds * targets).sum()
        return (2 * intersection + 1e-6) / (preds.sum() + targets.sum() + 1e-6)

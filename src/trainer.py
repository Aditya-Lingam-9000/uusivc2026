"""
src/trainer.py
UUSIVC 2026 — Universal Trainer (v2)

Features:
  - Mixed Precision (AMP / FP16) — saves 30-50% VRAM
  - Gradient Accumulation          — larger effective batch sizes
  - Warmup + Cosine WarmRestart LR scheduler
  - EMA (Exponential Moving Average)
  - Detailed per-step logging every N steps
  - VRAM usage monitoring
  - Resume from checkpoint (epoch, optimizer, scheduler, scaler, EMA)
  - Saves only best.pth + latest.pth

Usage:
    from src.trainer import Trainer
    trainer = Trainer(model, optimizer, scheduler, criterion, cfg, device)
    trainer.fit(train_loader, val_loader, metric_fn)
"""

import os
import time
import math
import gc
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast


# ─────────────────────────────────────────────────────────────
#  VRAM helper
# ─────────────────────────────────────────────────────────────
def get_vram_info():
    """Returns (used_GB, total_GB) for the primary GPU."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    used  = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return used, total


def fmt_vram():
    used, total = get_vram_info()
    return f"{used:.1f}/{total:.1f}GB ({100*used/total:.1f}%)" if total > 0 else "N/A"


# ─────────────────────────────────────────────────────────────
#  ETA helpers
# ─────────────────────────────────────────────────────────────
def fmt_time(seconds: float) -> str:
    """Format seconds to HH:MM:SS or MM:SS."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{sec:02d}s"
    return f"{m}m{sec:02d}s"


# ─────────────────────────────────────────────────────────────
#  Metrics helpers
# ─────────────────────────────────────────────────────────────
def dice_coeff(preds_bin: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> float:
    """Compute Dice on already-binarized predictions."""
    flat_p = preds_bin.float().view(-1)
    flat_t = targets.float().view(-1)
    inter  = (flat_p * flat_t).sum()
    return ((2.0 * inter + smooth) / (flat_p.sum() + flat_t.sum() + smooth)).item()


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


# ─────────────────────────────────────────────────────────────
#  Main Trainer class
# ─────────────────────────────────────────────────────────────
class Trainer:
    """
    Universal trainer supporting both segmentation and classification.

    Args:
        model       : nn.Module (possibly DataParallel wrapped)
        optimizer   : torch.optim.Optimizer
        scheduler   : LR scheduler (CosineAnnealingWarmRestarts or similar)
        criterion   : Loss function
        cfg         : dict from src.config
        device      : torch.device
        task_type   : "seg" | "cls"
        ema         : optional EMA object
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        cfg: dict,
        device: torch.device,
        task_type: str = "seg",    # "seg" | "cls"
        ema=None,
        ckpt_prefix: str = "model",  # e.g. "seg", "cls", "ceus_cls"
    ):
        self.model       = model
        self.optimizer   = optimizer
        self.scheduler   = scheduler
        self.criterion   = criterion
        self.cfg         = cfg
        self.device      = device
        self.task_type   = task_type
        self.ema         = ema
        self.ckpt_prefix = ckpt_prefix

        self.ckpt_dir    = cfg.get("ckpt_dir", "/kaggle/working/checkpoints")
        self.epochs      = cfg.get("epochs", 60)
        self.accum_steps = cfg.get("grad_accum_steps", 4)
        self.use_amp     = cfg.get("use_amp", True) and torch.cuda.is_available()
        self.log_steps   = cfg.get("log_steps", 25)
        self.grad_clip   = cfg.get("grad_clip", 1.0)
        self.log_vram    = cfg.get("log_vram", True)

        self.scaler = GradScaler('cuda', enabled=self.use_amp)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # State for resume
        self.start_epoch  = 1
        self.best_metric  = 0.0
        self.history      = []

    # ── Checkpoint: save ────────────────────────────────────────
    def _save_checkpoint(self, epoch: int, metric: float, is_best: bool):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        ckpt = {
            "epoch":                epoch,
            "best_metric":          self.best_metric,
            "model_state_dict":     raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict":    self.scaler.state_dict(),
            "history":              self.history,
            "cfg":                  self.cfg,
        }
        if self.ema is not None:
            ckpt["ema_state_dict"] = self.ema.state_dict()

        # Always save latest (for resume)
        if self.cfg.get("save_latest", True):
            latest_path = os.path.join(self.ckpt_dir, f"{self.ckpt_prefix}_latest.pth")
            torch.save(ckpt, latest_path)

        # Save best when metric improved
        if is_best and self.cfg.get("save_best", True):
            best_path = os.path.join(self.ckpt_dir, f"{self.ckpt_prefix}_best.pth")
            torch.save(ckpt, best_path)
            return best_path
        return None

    # ── Checkpoint: load (resume) ─────────────────────────────
    def load_checkpoint(self, ckpt_path: str):
        """Load checkpoint and restore all training state."""
        if not os.path.exists(ckpt_path):
            print(f"  [Trainer] No checkpoint found at {ckpt_path}, starting fresh.")
            return
        ckpt = torch.load(ckpt_path, map_location=self.device)
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        raw_model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_metric = ckpt.get("best_metric", 0.0)
        self.history     = ckpt.get("history", [])
        if self.ema is not None and "ema_state_dict" in ckpt:
            self.ema.load_state_dict(ckpt["ema_state_dict"])
        print(f"  [Trainer] ✅ Resumed from epoch {ckpt['epoch']} | "
              f"best_metric={self.best_metric:.4f}")

    # ── Warmup LR ────────────────────────────────────────────
    def _warmup_lr(self, epoch: int):
        """Apply linear warmup for first warmup_epochs."""
        warmup = self.cfg.get("warmup_epochs", 5)
        if epoch <= warmup:
            base_lr = self.cfg.get("lr", 1e-4)
            enc_lr  = self.cfg.get("encoder_lr", 1e-5)
            factor  = epoch / max(warmup, 1)
            for i, pg in enumerate(self.optimizer.param_groups):
                target_lr = enc_lr if i == 0 and len(self.optimizer.param_groups) > 1 else base_lr
                pg["lr"] = target_lr * factor

    # ── Train one epoch ──────────────────────────────────────
    def _train_epoch(self, loader, epoch: int) -> dict:
        self.model.train()
        loss_sum = 0.0
        metric_sum = 0.0
        n_samples = 0
        step_count = 0
        t_epoch_start = time.time()

        self.optimizer.zero_grad(set_to_none=True)
        total_steps = len(loader)

        for step, batch in enumerate(loader, 1):
            t_step_start = time.time()

            if self.task_type == "seg":
                imgs  = batch["input"].to(self.device)
                masks = batch["mask"].to(self.device)
                bs    = imgs.size(0)

                with autocast('cuda', enabled=self.use_amp):
                    logits = self.model(imgs)
                    if logits.shape[2:] != masks.shape[2:]:
                        logits = F.interpolate(logits, size=masks.shape[2:],
                                               mode="bilinear", align_corners=False)
                    loss_out = self.criterion(logits, masks)
                    # Support both simple loss and (loss, loss_dict)
                    if isinstance(loss_out, tuple):
                        loss, loss_dict = loss_out
                    else:
                        loss, loss_dict = loss_out, {}
                    loss = loss / self.accum_steps

                self.scaler.scale(loss).backward()

                # Dice metric (no grad needed)
                with torch.no_grad():
                    preds_bin = (torch.sigmoid(logits.detach()) > 0.5).float()
                    metric_val = dice_coeff(preds_bin, masks)

            else:  # cls
                imgs   = batch["input"].to(self.device)
                labels = batch["label"].to(self.device)
                bs     = imgs.size(0)

                with autocast('cuda', enabled=self.use_amp):
                    logits = self.model(imgs)
                    loss_out = self.criterion(logits, labels)
                    if isinstance(loss_out, tuple):
                        loss, loss_dict = loss_out
                    else:
                        loss, loss_dict = loss_out, {}
                    loss = loss / self.accum_steps

                self.scaler.scale(loss).backward()

                with torch.no_grad():
                    metric_val = accuracy(logits.detach(), labels)

            loss_sum   += loss.item() * self.accum_steps * bs
            metric_sum += metric_val * bs
            n_samples  += bs
            step_count += 1

            # Gradient step every accum_steps
            if step_count % self.accum_steps == 0 or step == total_steps:
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema is not None:
                    self.ema.update(self.model.module if hasattr(self.model, "module") else self.model)

            # ── Per-step logging every log_steps ──────────────────
            if step % self.log_steps == 0 or step == total_steps:
                t_step = time.time() - t_step_start
                t_elapsed = time.time() - t_epoch_start
                steps_left = total_steps - step
                eta_epoch  = steps_left * (t_elapsed / step)
                eta_total  = (self.epochs - epoch) * (t_elapsed / step * total_steps)

                metric_name = "dice" if self.task_type == "seg" else "acc"
                cur_loss = loss_sum / max(n_samples, 1)
                cur_metric = metric_sum / max(n_samples, 1)

                lr_vals = [pg["lr"] for pg in self.optimizer.param_groups]
                lr_str  = "/".join(f"{lr:.2e}" for lr in lr_vals)

                loss_detail = ""
                if loss_dict:
                    loss_detail = "  |  " + "  ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())

                vram_str = f"  |  VRAM: {fmt_vram()}" if self.log_vram else ""

                print(
                    f"    [E{epoch:02d} S{step:04d}/{total_steps}]  "
                    f"loss={cur_loss:.4f}  {metric_name}={cur_metric:.4f}"
                    f"{loss_detail}"
                    f"  |  LR: {lr_str}"
                    f"  |  ETA_epoch: {fmt_time(eta_epoch)}"
                    f"  |  ETA_total: {fmt_time(eta_total)}"
                    f"{vram_str}"
                )

        return {
            "loss":   loss_sum / max(n_samples, 1),
            "metric": metric_sum / max(n_samples, 1),
        }

    # ── Validate one epoch ──────────────────────────────────
    def _val_epoch(self, loader) -> dict:
        # Use EMA model for validation if available
        eval_model = self.ema.shadow if self.ema is not None else self.model
        eval_model.eval()

        loss_sum = 0.0
        metric_sum = 0.0
        n_samples = 0

        with torch.no_grad():
            for batch in loader:
                if self.task_type == "seg":
                    imgs  = batch["input"].to(self.device)
                    masks = batch["mask"].to(self.device)
                    bs    = imgs.size(0)

                    with autocast('cuda', enabled=self.use_amp):
                        logits = eval_model(imgs)
                        if logits.shape[2:] != masks.shape[2:]:
                            logits = F.interpolate(logits, size=masks.shape[2:],
                                                   mode="bilinear", align_corners=False)
                        loss_out = self.criterion(logits, masks)
                        loss = loss_out[0] if isinstance(loss_out, tuple) else loss_out

                    preds_bin = (torch.sigmoid(logits) > 0.5).float()
                    metric_val = dice_coeff(preds_bin, masks)

                else:  # cls
                    imgs   = batch["input"].to(self.device)
                    labels = batch["label"].to(self.device)
                    bs     = imgs.size(0)

                    with autocast('cuda', enabled=self.use_amp):
                        logits = eval_model(imgs)
                        loss_out = self.criterion(logits, labels)
                        loss = loss_out[0] if isinstance(loss_out, tuple) else loss_out

                    metric_val = accuracy(logits, labels)

                loss_sum   += loss.item() * bs
                metric_sum += metric_val * bs
                n_samples  += bs

        return {
            "loss":   loss_sum / max(n_samples, 1),
            "metric": metric_sum / max(n_samples, 1),
        }

    # ── Main training loop ───────────────────────────────────
    def fit(self, train_loader, val_loader):
        """
        Full training loop.
        Prints detailed logs every log_steps steps per epoch.
        Saves best.pth + latest.pth.
        """
        # Load checkpoint if resuming
        resume_path = self.cfg.get("resume_from")
        if resume_path:
            self.load_checkpoint(resume_path)

        metric_name = "dice" if self.task_type == "seg" else "acc"
        print(f"\n{'='*65}")
        print(f"  Training: {self.ckpt_prefix} | {self.start_epoch}→{self.epochs} epochs")
        print(f"  AMP={self.use_amp} | GradAccum={self.accum_steps} | "
              f"EffBatch={self.cfg.get('batch_size',8)*self.accum_steps}")
        print(f"  VRAM at start: {fmt_vram()}")
        print(f"{'='*65}")

        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()

            # Warmup LR
            self._warmup_lr(epoch)

            # Train
            train_stats = self._train_epoch(train_loader, epoch)

            # Validate
            val_stats = self._val_epoch(val_loader)

            # Scheduler step
            if isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                self.scheduler.step(epoch)
            else:
                self.scheduler.step()

            t_epoch = time.time() - t0
            is_best = val_stats["metric"] > self.best_metric
            if is_best:
                self.best_metric = val_stats["metric"]

            # Record history
            self.history.append({
                "epoch":       epoch,
                f"train_loss": train_stats["loss"],
                f"train_{metric_name}": train_stats["metric"],
                f"val_loss":   val_stats["loss"],
                f"val_{metric_name}":   val_stats["metric"],
            })

            # Save checkpoints
            saved_path = self._save_checkpoint(epoch, val_stats["metric"], is_best)

            # ── Epoch summary ───────────────────────────────────
            lr_vals = [pg["lr"] for pg in self.optimizer.param_groups]
            lr_str  = "/".join(f"{lr:.2e}" for lr in lr_vals)
            t_remaining = (self.epochs - epoch) * t_epoch

            print(f"\n{'─'*65}")
            print(f"  [Epoch {epoch:02d}/{self.epochs}]  time={fmt_time(t_epoch)}  "
                  f"ETA_total={fmt_time(t_remaining)}  LR={lr_str}")
            print(f"  Train  —  loss={train_stats['loss']:.4f}  "
                  f"{metric_name}={train_stats['metric']:.4f}")
            print(f"  Val    —  loss={val_stats['loss']:.4f}  "
                  f"{metric_name}={val_stats['metric']:.4f}")
            print(f"  VRAM: {fmt_vram()}")
            if is_best:
                print(f"  💾 Best {metric_name}={self.best_metric:.4f} → "
                      f"{saved_path or self.ckpt_prefix+'_best.pth'}")
            else:
                print(f"  ─ Best remains: {metric_name}={self.best_metric:.4f}")
            print(f"{'─'*65}")

        print(f"\n{'='*65}")
        print(f"✅ Training complete!")
        print(f"   Task         : {self.ckpt_prefix}")
        print(f"   Best val {metric_name}: {self.best_metric:.4f}")
        print(f"   Best ckpt    : {self.ckpt_dir}/{self.ckpt_prefix}_best.pth")
        print(f"{'='*65}")

        return self.best_metric, self.history

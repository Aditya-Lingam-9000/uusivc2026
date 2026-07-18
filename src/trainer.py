import time
import json
import torch
import torch.nn.functional as F
from copy import deepcopy

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = deepcopy(model)
        self.model.eval()
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            for ema_p, model_p in zip(self.model.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m}m{s}s"
    elif m > 0:
        return f"{m}m{s}s"
    return f"{s}s"

class UniversalTrainer:
    def __init__(self, cfg, model, optimizer, scheduler, criterion, device, 
                 train_loader, val_loader, metric_fn, task_name="Task"):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.metric_fn = metric_fn
        self.task_name = task_name
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.get("use_amp", True))
        self.ema = EMA(model) if cfg.get("use_ema", False) else None
        
        self.best_metric = 0.0
        self.start_epoch = 1
        self.history = []
        
        self.ckpt_dir = cfg["ckpt_dir"]
        
        if cfg.get("resume_from"):
            self.load_checkpoint(cfg["resume_from"])

    def load_checkpoint(self, path):
        print(f"Resuming from {path}...")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_metric = ckpt.get("best_metric", 0.0)
        print(f"Resumed from epoch {self.start_epoch-1}, best metric={self.best_metric:.4f}")

    def save_checkpoint(self, path, epoch, is_best=False):
        m = self.model.module if hasattr(self.model, "module") else self.model
        state = {
            "epoch": epoch,
            "best_metric": self.best_metric,
            "model_state_dict": m.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }
        torch.save(state, path)
        if is_best:
            print(f"  💾 Saved best ({self.best_metric:.4f}) → {path}")
        else:
            print(f"  📦 Saved latest → {path}")

    def get_vram_usage(self):
        if not torch.cuda.is_available():
            return "N/A"
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return f"{allocated:.1f}GB / {reserved:.1f}GB"

    def train_epoch(self, epoch):
        self.model.train()
        train_loss_sum = 0.0
        train_metric_sum = 0.0
        n_total = 0
        
        grad_accum_steps = self.cfg.get("grad_accum_steps", 1)
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(self.train_loader):
            inputs = batch["input"].to(self.device)
            targets = batch["mask"].to(self.device) if "mask" in batch else batch["label"].to(self.device)
            bs = inputs.size(0)
            
            with torch.cuda.amp.autocast(enabled=self.cfg.get("use_amp", True)):
                outputs = self.model(inputs)
                
                # Resize if necessary (segmentation)
                if outputs.dim() == 4 and outputs.shape[2:] != targets.shape[2:]:
                    outputs = F.interpolate(outputs, size=targets.shape[2:], mode="bilinear", align_corners=False)
                    
                loss = self.criterion(outputs, targets)
                loss = loss / grad_accum_steps
                
            self.scaler.scale(loss).backward()
            
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                # Unscale for gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                if self.ema:
                    self.ema.update(self.model)
            
            train_loss_sum += loss.item() * grad_accum_steps * bs
            train_metric_sum += self.metric_fn(outputs, targets).item() * bs
            n_total += bs
            
            if (step + 1) % self.cfg.get("log_steps", 50) == 0:
                print(f"    Step {step+1}/{len(self.train_loader)} | Loss: {loss.item()*grad_accum_steps:.4f} | VRAM: {self.get_vram_usage()}")
                
        self.scheduler.step()
        return train_loss_sum / max(n_total, 1), train_metric_sum / max(n_total, 1)

    def validate_epoch(self):
        eval_model = self.ema.model if self.ema else self.model
        eval_model.eval()
        
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        n_total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch["input"].to(self.device)
                targets = batch["mask"].to(self.device) if "mask" in batch else batch["label"].to(self.device)
                bs = inputs.size(0)
                
                outputs = eval_model(inputs)
                if outputs.dim() == 4 and outputs.shape[2:] != targets.shape[2:]:
                    outputs = F.interpolate(outputs, size=targets.shape[2:], mode="bilinear", align_corners=False)
                    
                loss = self.criterion(outputs, targets)
                
                val_loss_sum += loss.item() * bs
                val_metric_sum += self.metric_fn(outputs, targets).item() * bs
                n_total += bs
                
        return val_loss_sum / max(n_total, 1), val_metric_sum / max(n_total, 1)

    def fit(self):
        print(f"========== STARTING {self.task_name} ==========")
        total_epochs = self.cfg["epochs"]
        
        for epoch in range(self.start_epoch, total_epochs + 1):
            t0 = time.time()
            
            train_loss, train_metric = self.train_epoch(epoch)
            val_loss, val_metric = self.validate_epoch()
            
            t_elapsed = time.time() - t0
            
            # Simple ETA calculation
            eta_seconds = t_elapsed * (total_epochs - epoch)
            
            print(f"\n[Epoch {epoch:02d}/{total_epochs}] time={format_time(t_elapsed)} ETA={format_time(eta_seconds)}")
            print(f"  VRAM: {self.get_vram_usage()} | LR: {get_lr(self.optimizer):.2e}")
            print(f"  Train — loss: {train_loss:.4f} metric: {train_metric:.4f}")
            print(f"  Val   — loss: {val_loss:.4f} metric: {val_metric:.4f}")
            
            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_metric": train_metric,
                "val_loss": val_loss,
                "val_metric": val_metric
            })
            
            if self.cfg.get("save_latest", True):
                self.save_checkpoint(f"{self.ckpt_dir}/{self.task_name}_latest.pth", epoch, is_best=False)
                
            if val_metric > self.best_metric:
                self.best_metric = val_metric
                if self.cfg.get("save_best", True):
                    self.save_checkpoint(f"{self.ckpt_dir}/{self.task_name}_best.pth", epoch, is_best=True)
                    
            with open(f"{self.ckpt_dir}/{self.task_name}_history.json", "w") as f:
                json.dump(self.history, f, indent=2)
                
        print(f"\n✅ {self.task_name} COMPLETE. Best Metric: {self.best_metric:.4f}")

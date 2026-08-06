"""
heritagenet/engine/trainer.py

The training loop, kept deliberately small and readable.

What it does
------------
Given a model, a train loader, and a val loader, it runs supervised training
with the pieces that matter on a small dataset and a 6 GB GPU:

  * mixed precision (AMP)      -> less VRAM, faster steps on the RTX 3050
  * class weights (optional)   -> counteract your lumpy per-class counts
  * label smoothing (optional) -> gentle regularizer, calms overconfidence
  * early stopping + restore   -> stop at the peak, rewind to the best weights
  * checkpointing              -> best model saved to disk automatically
  * a history dict             -> per-epoch curves for your report's plots

It intentionally does NOT compute macro-F1 / confusion matrix here — those
belong in eval (metrics.py), run once on the untouched test set at the end.
The trainer only needs a cheap signal (val accuracy) to decide when to stop.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# 1. Configuration — every knob in one place, with sane defaults.
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float =  5e-5                 # comfy stride for a fresh head (phase 1)
    weight_decay: float =  5e-4       # mild L2; helps on small data
    label_smoothing: float = 0.1     # try 0.05-0.1 later to curb overconfidence
    monitor: str = "val_acc"         # "val_acc" (maximize) or "val_loss" (minimize)
    patience: int = 5                # epochs of no improvement before stopping
    amp: bool = True                 # mixed precision; auto-disabled off CUDA
    ckpt_dir: str = "checkpoints"
    ckpt_name: str = "best.pt"
    verbose: bool = True


# ---------------------------------------------------------------------------
# 2. The trainer.
# ---------------------------------------------------------------------------
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: TrainConfig | None = None,
        class_weights: torch.Tensor | None = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.cfg = config or TrainConfig()

        # Only the parameters we left trainable (the head in phase 1) get updated.
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

        cw = class_weights.to(device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(
            weight=cw, label_smoothing=self.cfg.label_smoothing
        )

        # AMP is a no-op unless we're really on CUDA.
        self.use_amp = self.cfg.amp and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        self.history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
        }

    # -- one pass over a loader; train=True updates weights -----------------
    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, float]:
        self.model.train(train)
        total, correct, loss_sum = 0, 0, 0.0

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.set_grad_enabled(train):
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            loss_sum += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        return loss_sum / total, correct / total

    # -- the full training run ----------------------------------------------
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        cfg = self.cfg
        maximize = cfg.monitor == "val_acc"
        best_score = -float("inf") if maximize else float("inf")
        best_weights = copy.deepcopy(self.model.state_dict())
        best_epoch = 0
        epochs_since_improve = 0

        ckpt_path = Path(cfg.ckpt_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = self._run_epoch(train_loader, train=True)
            with torch.no_grad():
                va_loss, va_acc = self._run_epoch(val_loader, train=False)

            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_loss"].append(va_loss)
            self.history["val_acc"].append(va_acc)

            score = va_acc if maximize else va_loss
            improved = (score > best_score) if maximize else (score < best_score)

            if improved:
                best_score = score
                best_weights = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                epochs_since_improve = 0
                torch.save(
                    {"model_state": best_weights, "epoch": epoch, cfg.monitor: score},
                    ckpt_path / cfg.ckpt_name,
                )
                flag = "  <- best (saved)"
            else:
                epochs_since_improve += 1
                flag = ""

            if cfg.verbose:
                dt = time.time() - t0
                print(
                    f"epoch {epoch:2d}/{cfg.epochs} | "
                    f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
                    f"val loss {va_loss:.3f} acc {va_acc:.3f} | "
                    f"{dt:4.1f}s{flag}"
                )

            if epochs_since_improve >= cfg.patience:
                if cfg.verbose:
                    print(f"early stop: no {cfg.monitor} gain for {cfg.patience} epochs.")
                break

        # Restore the best weights — never leave the model in its tired,
        # over-trained final state.
        self.model.load_state_dict(best_weights)
        if cfg.verbose:
            print(f"restored best weights from epoch {best_epoch} "
                  f"({cfg.monitor}={best_score:.3f}).")

        self.history["best_epoch"] = best_epoch
        self.history["best_score"] = best_score
        return self.history


# ---------------------------------------------------------------------------
# 3. Helper: class weights from the training set, for imbalance.
# ---------------------------------------------------------------------------
def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """
    Inverse-frequency weights, normalized to mean 1.0. Feed the result to
    Trainer(class_weights=...). Rare classes get a larger loss weight so the
    model can't win by ignoring them — the right fix for your lumpy splits.
    """
    counts = torch.zeros(num_classes, dtype=torch.float)
    for y in labels:
        counts[y] += 1
    counts = counts.clamp(min=1.0)          # avoid divide-by-zero
    weights = counts.sum() / (num_classes * counts)
    return weights / weights.mean()
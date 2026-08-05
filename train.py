"""
train.py  —  driver script for HeritageNet phase-1 training.

Run from the repo root with the venv active:

    python train.py                       # defaults: mobilenet_v3_small, 20 epochs
    python train.py --epochs 30 --lr 5e-4 # override any knob

This file contains almost no logic of its own. Its whole job is to pick up the
modules we built (datamodule -> factory -> trainer) and run them in order.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from heritagenet.data.datamodule import HeritageDataModule
from heritagenet.models.factory import build_model, count_parameters
from heritagenet.engine.trainer import Trainer, TrainConfig

# Use our device helper if present; fall back to a plain check otherwise.
try:
    from heritagenet.utils.device import get_device
except Exception:  # pragma: no cover
    def get_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(dm, batch_size: int, num_workers: int):
    """
    Return (train_loader, val_loader), working with EITHER datamodule style:

      * if the datamodule exposes .train_dataloader()/.val_dataloader(), use them
      * otherwise wrap its .train_dataset/.val_dataset in DataLoaders here
    """
    if hasattr(dm, "train_dataloader") and callable(dm.train_dataloader):
        return dm.train_dataloader(), dm.val_dataloader()

    train_loader = DataLoader(
        dm.train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        dm.val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def main() -> None:
    p = argparse.ArgumentParser(description="Train HeritageNet (phase 1: frozen backbone).")
    p.add_argument("--data", default="data", help="root folder with train/val/test")
    p.add_argument("--model", default="mobilenet_v3_small")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()

    # 1. Device
    device = get_device()
    print(f"device: {device}")

    # 2. Data
    dm = HeritageDataModule(root=args.data, batch_size=args.batch_size)
    dm.setup()
    print(f"classes ({dm.num_classes}): {dm.classes}")

    train_loader, val_loader = make_loaders(dm, args.batch_size, args.num_workers)

    # 3. Model (phase 1: frozen backbone, fresh head)
    model = build_model(args.model, num_classes=dm.num_classes)
    trn, tot = count_parameters(model)
    print(f"model: {args.model} | trainable {trn:,} / total {tot:,} "
          f"({100 * (1 - trn / tot):.1f}% frozen)")

    # 4. Train
    cfg = TrainConfig(epochs=args.epochs, lr=args.lr)
    trainer = Trainer(model, device, cfg)
    history = trainer.fit(train_loader, val_loader)

    # 5. Save history so the dashboard can plot the real, current curves.
    import json
    from pathlib import Path
    Path("outputs").mkdir(parents=True, exist_ok=True)
    with open("outputs/history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("saved training history -> outputs/history.json")

    # 6. Report
    print(f"\nDone. best {cfg.monitor} = {history['best_score']:.3f} "
          f"at epoch {history['best_epoch']}.")
    print(f"best checkpoint: {cfg.ckpt_dir}/{cfg.ckpt_name}")


if __name__ == "__main__":
    main()
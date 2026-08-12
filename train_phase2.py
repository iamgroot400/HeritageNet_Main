"""
train_phase2.py  —  Phase-2 fine-tuning.

Continues from your phase-1 model, but UNFREEZES the backbone so the network's
"eyes" can adapt to your temples — at a much lower learning rate so the valuable
ImageNet features are nudged gently, not destroyed.

Starts from checkpoints/best.pt (phase 1) and saves to checkpoints/best_phase2.pt,
leaving the phase-1 baseline intact for comparison.

This model was created by tweaking the learning rate to(5e-5) and unfreezing the backbone in the original HeritageNet paper's
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from heritagenet.data.datamodule import HeritageDataModule
from heritagenet.models.factory import (build_model, set_backbone_trainable,
                                        count_parameters)
from heritagenet.engine.trainer import Trainer, TrainConfig

try:
    from heritagenet.utils.device import get_device
except Exception:  # pragma: no cover
    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(dm, batch_size, num_workers):
    if hasattr(dm, "train_dataloader") and callable(dm.train_dataloader):
        return dm.train_dataloader(), dm.val_dataloader()
    tl = DataLoader(dm.train_dataset, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True)
    vl = DataLoader(dm.val_dataset, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    return tl, vl


def main():
    ap = argparse.ArgumentParser(description="Phase-2 fine-tuning (unfreeze backbone).")
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="mobilenet_v3_small")
    ap.add_argument("--init-ckpt", default="checkpoints/best.pt",
                    help="phase-1 checkpoint to start from")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default= 1e-4,
                    help="LOW lr for gentle fine-tuning (phase 1 used 1e-3)")
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    device = get_device()
    print("device:", device)

    dm = HeritageDataModule(root=args.data, batch_size=args.batch_size).setup()
    print(f"classes ({dm.num_classes})")
    train_loader, val_loader = make_loaders(dm, args.batch_size, args.num_workers)

    # rebuild the architecture, then load the phase-1 weights (backbone + head)
    model = build_model(args.model, num_classes=dm.num_classes,
                        pretrained=False, freeze_backbone=True)
    ckpt = torch.load(args.init_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"loaded phase-1 weights from {args.init_ckpt} "
          f"(val_acc {ckpt.get('val_acc')})")

    # THE phase-2 step: unfreeze the whole network so features can adapt.
    set_backbone_trainable(model, True)
    trn, tot = count_parameters(model)
    print(f"after unfreeze: trainable {trn:,}/{tot:,} "
          f"({100 * trn / tot:.1f}% now training)")

    # gentle fine-tune; save to a SEPARATE checkpoint to preserve the baseline
    cfg = TrainConfig(epochs=args.epochs, lr=args.lr,
                      ckpt_dir="checkpoints", ckpt_name="best_phase2.pt")
    trainer = Trainer(model, device, cfg)
    history = trainer.fit(train_loader, val_loader)

    Path("outputs").mkdir(parents=True, exist_ok=True)
    with open("outputs/history_phase2.json", "w") as f:
        json.dump(history, f, indent=2)
    print("saved history -> outputs/history_phase2.json")

    print(f"\nphase-2 best {cfg.monitor} = {history['best_score']:.3f} "
          f"at epoch {history['best_epoch']}")
    print("phase-2 checkpoint: checkpoints/best_phase2.pt")
    print("(phase-1 baseline checkpoints/best.pt left untouched)")


if __name__ == "__main__":
    main()
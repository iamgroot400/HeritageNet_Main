"""
train_phase2.py  —  Phase-2 fine-tuning.  [EXPERIMENT: dropout 0.3]

Continues from the phase-1 model, unfreezes the backbone, fine-tunes gently.
This copy is configured for a SINGLE controlled experiment: dropout 0.3
(vs the champion's 0.2), same LR (1e-4), everything else at baseline.

Saves to a distinct checkpoint so it cannot touch best_phase2.pt or the champion.
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
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="LOW lr for gentle fine-tuning (phase 1 used 1e-3)")
    ap.add_argument("--dropout", type=float, default=0.2,
                    help="head dropout for this experiment (champion used 0.2)")
    ap.add_argument("--ckpt-name", default="best_phase2_lr5e5.pt",
                    help="distinct name so this run can't overwrite other models")
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    device = get_device()
    print("device:", device)

    dm = HeritageDataModule(root=args.data, batch_size=args.batch_size).setup()
    print(f"classes ({dm.num_classes})")
    train_loader, val_loader = make_loaders(dm, args.batch_size, args.num_workers)

    # rebuild architecture with the EXPERIMENT dropout, then load phase-1 weights
    model = build_model(args.model, num_classes=dm.num_classes,
                        pretrained=False, freeze_backbone=True,
                        dropout=args.dropout)
    ckpt = torch.load(args.init_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"loaded phase-1 weights from {args.init_ckpt} "
          f"(val_acc {ckpt.get('val_acc')})")
    print(f"EXPERIMENT: dropout={args.dropout}, lr={args.lr}")

    # unfreeze the whole network
    set_backbone_trainable(model, True)
    trn, tot = count_parameters(model)
    print(f"after unfreeze: trainable {trn:,}/{tot:,} "
          f"({100 * trn / tot:.1f}% now training)")

    cfg = TrainConfig(epochs=args.epochs, lr=args.lr,
                      ckpt_dir="checkpoints", ckpt_name=args.ckpt_name)
    trainer = Trainer(model, device, cfg)
    history = trainer.fit(train_loader, val_loader)

    Path("outputs").mkdir(parents=True, exist_ok=True)
    hist_name = f"outputs/history_{Path(args.ckpt_name).stem}.json"
    with open(hist_name, "w") as f:
        json.dump(history, f, indent=2)
    print(f"saved history -> {hist_name}")

    print(f"\nbest {cfg.monitor} = {history['best_score']:.3f} "
          f"at epoch {history['best_epoch']}")
    print(f"checkpoint: checkpoints/{args.ckpt_name}")
    print("(champion phase2_86acc_FINAL.pt and best_phase2.pt left untouched)")


if __name__ == "__main__":
    main()
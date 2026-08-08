"""
train_phase2_partial.py  —  PARTIAL fine-tuning (gradual unfreezing).

The middle ground between phase 1 (freeze all backbone) and phase 2 (unfreeze
all). Here we unfreeze ONLY the last few backbone blocks + the head, so:

  * early general features (edges/textures) stay frozen  -> protected
  * late temple-specific features adapt                  -> the useful part
  * far fewer trainable params than full phase 2         -> less overfitting

Fully isolated: reads the phase-1 model, writes to checkpoints/best_partial.pt
and outputs/history_partial.json. Touches NONE of your existing models/files.

Run from the repo root (venv active):

    python train_phase2_partial.py                     # unfreeze last 3 blocks
    python train_phase2_partial.py --unfreeze-last 2   # even fewer params
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from heritagenet.data.datamodule import HeritageDataModule
from heritagenet.models.factory import build_model, count_parameters
from heritagenet.engine.trainer import Trainer, TrainConfig

try:
    from heritagenet.utils.device import get_device
except Exception:  # pragma: no cover
    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_partial_trainable(model, unfreeze_last_n_blocks: int) -> None:
    """
    Freeze everything, then unfreeze ONLY:
      * the last `unfreeze_last_n_blocks` blocks of model.features (the backbone)
      * the entire classifier head (always trained)

    MobileNetV3-small has 13 feature blocks (0..12); unfreezing the last 3
    means blocks 10, 11, 12 become trainable, the rest stay frozen.
    """
    # 1. freeze the whole model
    for p in model.parameters():
        p.requires_grad = False

    # 2. unfreeze the last N backbone blocks
    n_blocks = len(model.features)
    start = max(0, n_blocks - unfreeze_last_n_blocks)
    for i in range(start, n_blocks):
        for p in model.features[i].parameters():
            p.requires_grad = True

    # 3. the head always trains
    for p in model.classifier.parameters():
        p.requires_grad = True

    print(f"unfroze backbone blocks {start}..{n_blocks - 1} (of 0..{n_blocks - 1}) + head")


def make_loaders(dm, batch_size, num_workers):
    if hasattr(dm, "train_dataloader") and callable(dm.train_dataloader):
        return dm.train_dataloader(), dm.val_dataloader()
    tl = DataLoader(dm.train_dataset, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True)
    vl = DataLoader(dm.val_dataset, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    return tl, vl


def main():
    ap = argparse.ArgumentParser(description="Partial fine-tuning (gradual unfreezing).")
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="mobilenet_v3_small")
    ap.add_argument("--init-ckpt", default="checkpoints/best.pt",
                    help="phase-1 checkpoint to start from")
    ap.add_argument("--unfreeze-last", type=int, default=2,
                    help="how many final backbone blocks to unfreeze (2=~63%, 1=~44%, 3=~82%)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    device = get_device()
    print("device:", device)

    dm = HeritageDataModule(root=args.data, batch_size=args.batch_size).setup()
    print(f"classes ({dm.num_classes})")
    train_loader, val_loader = make_loaders(dm, args.batch_size, args.num_workers)

    # rebuild, load phase-1 weights
    model = build_model(args.model, num_classes=dm.num_classes,
                        pretrained=False, freeze_backbone=True)
    ckpt = torch.load(args.init_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"loaded phase-1 weights from {args.init_ckpt}")

    # THE partial step: unfreeze only the last N backbone blocks + head
    set_partial_trainable(model, args.unfreeze_last)
    trn, tot = count_parameters(model)
    print(f"trainable {trn:,}/{tot:,} ({100 * trn / tot:.1f}% training) "
          f"-- vs full phase-2 which trains 100%")

    # save to a SEPARATE checkpoint so nothing existing is touched
    cfg = TrainConfig(epochs=args.epochs, lr=args.lr,
                      ckpt_dir="checkpoints", ckpt_name="best_partial.pt")
    trainer = Trainer(model, device, cfg)
    history = trainer.fit(train_loader, val_loader)

    Path("outputs").mkdir(parents=True, exist_ok=True)
    with open("outputs/history_partial.json", "w") as f:
        json.dump(history, f, indent=2)
    print("saved history -> outputs/history_partial.json")

    print(f"\npartial best {cfg.monitor} = {history['best_score']:.3f} "
          f"at epoch {history['best_epoch']}")
    print("partial checkpoint: checkpoints/best_partial.pt")
    print("(existing models untouched)")


if __name__ == "__main__":
    main()
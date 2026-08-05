"""
scripts/save_mistakes.py

Copy every misclassified TEST image into outputs/mistakes/, renamed:
    TRUE=<real site>__PRED=<what the model guessed>__<original filename>
so you can SEE why each confusion happens.

Run from the repo root (venv active):
    python scripts/save_mistakes.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from heritagenet.data.datamodule import HeritageDataModule
from heritagenet.models.factory import build_model
from heritagenet.utils.device import get_device


@torch.no_grad()
def main():
    device = get_device()
    out_dir = Path("outputs/mistakes")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    dm = HeritageDataModule(root="data", batch_size=32).setup()
    classes = dm.classes

    model = build_model("mobilenet_v3_small", num_classes=dm.num_classes,
                        pretrained=False, freeze_backbone=False)
    ckpt = torch.load("checkpoints/best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    # no shuffle, so prediction order matches dataset.samples order
    loader = DataLoader(dm.test_dataset, batch_size=32, shuffle=False, num_workers=0)

    preds = []
    for images, _ in loader:
        preds.extend(model(images.to(device)).argmax(1).cpu().tolist())

    samples = dm.test_dataset.samples
    assert len(samples) == len(preds), "prediction/sample count mismatch"

    n_saved = 0
    for (path, true_label), pred_label in zip(samples, preds):
        if true_label == pred_label:
            continue
        src = Path(path)
        dst = out_dir / f"TRUE={classes[true_label]}__PRED={classes[pred_label]}__{src.name}"
        shutil.copy2(src, dst)
        n_saved += 1
        print(f"  {classes[true_label]}  ->  {classes[pred_label]}   ({src.name})")

    print(f"\nsaved {n_saved} misclassified images to: {out_dir.resolve()}")
    if n_saved == 0:
        print("No mistakes on the test set!")


if __name__ == "__main__":
    main()
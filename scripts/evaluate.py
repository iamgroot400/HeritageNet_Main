"""
The trainer only ever looked at train/val. This script loads the best saved
checkpoint and runs it against test/ — data that influenced nothing — to get:

  * test accuracy          (overall correctness)
  * macro-F1               (average per-class F1; treats a rare site as
                            importantly as a common one — the fair metric
                            for your imbalanced data)
  * a per-class report     (precision/recall/F1 for every temple)
  * a confusion matrix PNG (which sites get mistaken for which)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import matplotlib
matplotlib.use("Agg")  # no display needed; we save to file
import matplotlib.pyplot as plt

from heritagenet.data.datamodule import HeritageDataModule
from heritagenet.models.factory import build_model
from heritagenet.utils.device import get_device


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run the model over a loader; return (y_true, y_pred) as numpy arrays."""
    model.eval()
    ys, ps = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        ps.append(logits.argmax(dim=1).cpu())
        ys.append(labels)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def plot_confusion(cm, classes, out_path: Path, normalize: bool = True):
    """Save a labeled confusion-matrix heatmap."""
    if normalize:
        with np.errstate(all="ignore"):
            cm_disp = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_disp = np.nan_to_num(cm_disp)
        fmt = ".2f"
        title = "Confusion Matrix (row-normalized)"
    else:
        cm_disp = cm
        fmt = "d"
        title = "Confusion Matrix (counts)"

    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(6, n * 0.7)))
    im = ax.imshow(cm_disp, cmap="Blues", vmin=0, vmax=cm_disp.max() if cm_disp.max() else 1)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    thresh = cm_disp.max() / 2 if cm_disp.max() else 0.5
    for i in range(n):
        for j in range(n):
            val = cm_disp[i, j]
            if val > 0:
                ax.text(j, i, format(val, fmt), ha="center", va="center",
                        color="white" if val > thresh else "black", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on the test set.")
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="mobilenet_v3_small")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    device = get_device()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dm = HeritageDataModule(root=args.data, batch_size=args.batch_size).setup()

    # pretrained=False: we load OUR trained weights, no need to fetch ImageNet.
    model = build_model(args.model, num_classes=dm.num_classes,
                        pretrained=False, freeze_backbone=False)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)

    y_true, y_pred = collect_predictions(model, dm.test_dataloader(), device)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\ntest images: {len(y_true)}")
    print(f"accuracy : {acc:.3f}")
    print(f"macro-F1 : {macro_f1:.3f}\n")
    print("per-class report:")
    print(classification_report(y_true, y_pred, target_names=dm.classes,
                                digits=3, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=range(dm.num_classes))
    plot_confusion(cm, dm.classes, out_dir / "confusion_matrix.png", normalize=True)
    plot_confusion(cm, dm.classes, out_dir / "confusion_matrix_counts.png", normalize=False)
    print(f"confusion matrices saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
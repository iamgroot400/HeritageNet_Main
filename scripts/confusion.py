"""
scripts/confusions.py — list every misclassification as (true -> predicted).
Reuses the trained model; prints the exact confused pairs so we can diagnose.
"""
from __future__ import annotations
from collections import Counter
import torch
from heritagenet.data.datamodule import HeritageDataModule
from heritagenet.models.factory import build_model
from heritagenet.utils.device import get_device


@torch.no_grad()
def main():
    device = get_device()
    dm = HeritageDataModule(root="data", batch_size=32).setup()
    model = build_model("mobilenet_v3_small", num_classes=dm.num_classes,
                        pretrained=False, freeze_backbone=False)
    ckpt = torch.load("checkpoints/best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    classes = dm.classes
    mistakes = Counter()
    per_class_total = Counter()

    for images, labels in dm.test_dataloader():
        preds = model(images.to(device)).argmax(1).cpu()
        for t, p in zip(labels.tolist(), preds.tolist()):
            per_class_total[t] += 1
            if t != p:
                mistakes[(t, p)] += 1

    if not mistakes:
        print("No misclassifications on the test set.")
        return

    print(f"{'TRUE site':<30} ->  {'PREDICTED as':<30}  count")
    print("-" * 72)
    for (t, p), c in mistakes.most_common():
        print(f"{classes[t]:<30} ->  {classes[p]:<30}  {c}")


if __name__ == "__main__":
    main()
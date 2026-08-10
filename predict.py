r"""
predict.py  —  Single-image prediction with an "I don't know" threshold.

Loads the trained model and classifies ONE image. If the model's top confidence
is below --threshold, it abstains ("Not recognized") instead of guessing.

This is pure post-processing on the existing model — NO retraining, and it does
not change any weights. Tune --threshold freely; it's just a number.

Run from the repo root (venv active):

    python predict.py --image path\to\photo.jpg
    python predict.py --image path\to\photo.jpg --threshold 0.7
    python predict.py --image path\to\photo.jpg --ckpt checkpoints\phase2_86acc_FINAL.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from heritagenet.data.datamodule import HeritageDataModule
from heritagenet.data.transforms import get_eval_transforms
from heritagenet.models.factory import build_model

try:
    from heritagenet.utils.device import get_device
except Exception:  # pragma: no cover
    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_class_names(data_root: str) -> list[str]:
    """Get the class list (in the model's index order) from the data folders."""
    dm = HeritageDataModule(root=data_root, batch_size=1)
    dm.setup()
    return dm.classes


@torch.no_grad()
def predict(image_path: str, model, classes, device, threshold: float, topk: int = 3):
    # deterministic eval transform — same preprocessing as test time
    tf = get_eval_transforms()
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)          # add batch dimension -> [1,3,224,224]

    probs = F.softmax(model(x), dim=1).squeeze(0)  # [num_classes]
    conf, idx = probs.max(dim=0)
    conf = conf.item()
    pred = classes[idx.item()]

    # top-k for context
    k = min(topk, len(classes))
    top_conf, top_idx = probs.topk(k)
    topk_list = [(classes[i], top_conf[j].item()) for j, i in enumerate(top_idx.tolist())]

    decision = pred if conf >= threshold else "Not recognized (below threshold)"
    return decision, conf, topk_list


def main():
    ap = argparse.ArgumentParser(description="Classify one image, with an unknown-reject threshold.")
    ap.add_argument("--image", required=True, help="path to the image to classify")
    ap.add_argument("--ckpt", default="checkpoints/phase2_86acc_FINAL.pt",
                    help="model checkpoint (default: the champion)")
    ap.add_argument("--model", default="mobilenet_v3_small")
    ap.add_argument("--data", default="data", help="used only to read class names")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="min top-confidence to accept a prediction (else 'Not recognized')")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    if not Path(args.image).is_file():
        raise SystemExit(f"Image not found: {args.image}")

    device = get_device()
    classes = load_class_names(args.data)

    model = build_model(args.model, num_classes=len(classes),
                        pretrained=False, freeze_backbone=False)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()

    decision, conf, topk_list = predict(args.image, model, classes, device,
                                        args.threshold, args.topk)

    print(f"\nimage: {args.image}")
    print(f"threshold: {args.threshold}")
    print(f"\n>>> {decision}   (top confidence {conf:.1%})\n")
    print(f"top-{len(topk_list)} guesses:")
    for name, c in topk_list:
        mark = "  <- chosen" if (name == decision) else ""
        print(f"  {c:6.1%}  {name}{mark}")
    if decision.startswith("Not recognized"):
        print("\n(the model was not confident enough — treated as an unknown site)")


if __name__ == "__main__":
    main()
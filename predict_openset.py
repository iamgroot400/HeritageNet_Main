r"""
predict_openset.py  —  Prediction with OPEN-SET rejection (embedding distance).

Two independent guards decide "known site" vs "not recognized":
  1. softmax confidence  >= --threshold          (the easy filter)
  2. cosine distance to nearest prototype <= cutoff (the real open-set filter)

A photo is accepted only if BOTH pass. The distance guard is what catches
out-of-distribution inputs (a person, a random object) that fool the softmax.

Needs export/prototypes.npz from build_prototypes.py first.

Run:  python predict_openset.py --image path\to\photo.jpg
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from heritagenet.data.datamodule import HeritageDataModule
from heritagenet.data.transforms import get_eval_transforms
from heritagenet.models.factory import build_model
try:
    from heritagenet.utils.device import get_device
except Exception:
    def get_device(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def embed(model, x):
    feat = model.features(x); pooled = model.avgpool(feat).flatten(1)
    z = model.classifier[0](pooled); z = model.classifier[1](z); return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--ckpt", default="checkpoints/phase2_86acc_FINAL.pt")
    ap.add_argument("--model", default="mobilenet_v3_small")
    ap.add_argument("--data", default="data")
    ap.add_argument("--protos", default="export/prototypes.npz")
    ap.add_argument("--threshold", type=float, default=0.55, help="min softmax confidence")
    ap.add_argument("--cutoff", type=float, default=None,
                    help="max distance to nearest prototype (default: from prototypes.npz)")
    args = ap.parse_args()

    if not Path(args.image).is_file(): raise SystemExit(f"image not found: {args.image}")
    device = get_device()

    npz = np.load(args.protos, allow_pickle=True)
    protos = npz["prototypes"]; classes = list(npz["classes"]); cutoff = float(npz["cutoff"])
    if args.cutoff is not None: cutoff = args.cutoff

    model = build_model(args.model, num_classes=len(classes), pretrained=False, freeze_backbone=False)
    ck = torch.load(args.ckpt, map_location=device); model.load_state_dict(ck.get("model_state", ck))
    model.to(device).eval()

    tf = get_eval_transforms()
    x = tf(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x); probs = F.softmax(logits, 1).squeeze(0).cpu().numpy()
        emb = embed(model, x).squeeze(0).cpu().numpy()

    # distances
    Pn = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
    en = emb / (np.linalg.norm(emb) + 1e-8)
    dists = 1.0 - Pn @ en
    nearest = int(dists.argmin()); near_dist = float(dists[nearest])

    top = int(probs.argmax()); conf = float(probs[top])
    conf_ok = conf >= args.threshold
    dist_ok = near_dist <= cutoff
    accepted = conf_ok and dist_ok

    print(f"\nimage: {args.image}")
    print(f"top class      : {classes[top]}  ({conf:.1%} confidence)")
    print(f"nearest proto  : {classes[nearest]}  (distance {near_dist:.3f}, cutoff {cutoff:.3f})")
    print(f"confidence gate: {'PASS' if conf_ok else 'FAIL'} (>= {args.threshold})")
    print(f"distance gate  : {'PASS' if dist_ok else 'FAIL'} (<= {cutoff:.3f})")
    print()
    if accepted:
        print(f">>> {classes[top]}  ({conf:.0%})")
    else:
        reason = "too far from any known site" if not dist_ok else "confidence too low"
        print(f">>> Not recognized  ({reason})")


if __name__ == "__main__":
    main()
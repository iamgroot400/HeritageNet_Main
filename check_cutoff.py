r"""
check_cutoff.py  —  Evaluate how well an open-set cutoff separates known vs unknown.

For every KNOWN test image, computes its cosine distance to the nearest prototype.
Reports, at a range of cutoffs, what fraction of known images would be (correctly)
accepted vs (wrongly) rejected. Optionally does the same for a folder of UNKNOWN
images you provide, so you can see the real separation.

Run:
  python check_cutoff.py
  python check_cutoff.py --unknown-dir path\to\some_non_heritage_photos
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from PIL import Image
from heritagenet.data.datamodule import HeritageDataModule
from heritagenet.data.transforms import get_eval_transforms
from heritagenet.models.factory import build_model
try:
    from heritagenet.utils.device import get_device
except Exception:
    def get_device(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def embed(model, x):
    f = model.features(x); p = model.avgpool(f).flatten(1)
    return model.classifier[1](model.classifier[0](p))


def nearest_dists(embs, protos):
    Pn = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
    En = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    return (1.0 - En @ Pn.T).min(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/phase2_86acc_FINAL.pt")
    ap.add_argument("--model", default="mobilenet_v3_small")
    ap.add_argument("--data", default="data")
    ap.add_argument("--protos", default="export/prototypes.npz")
    ap.add_argument("--unknown-dir", default=None, help="folder of non-heritage images (optional)")
    args = ap.parse_args()

    device = get_device()
    dm = HeritageDataModule(root=args.data, batch_size=32); dm.setup()
    model = build_model(args.model, num_classes=len(dm.classes), pretrained=False, freeze_backbone=False)
    ck = torch.load(args.ckpt, map_location=device); model.load_state_dict(ck.get("model_state", ck))
    model.to(device).eval()

    npz = np.load(args.protos, allow_pickle=True)
    protos = npz["prototypes"]; saved_cut = float(npz["cutoff"])

    # KNOWN: all test images
    loader = DataLoader(dm.test_dataset, batch_size=32, shuffle=False, num_workers=0)
    E = []
    with torch.no_grad():
        for x, _ in loader:
            E.append(embed(model, x.to(device)).cpu().numpy())
    d_known = nearest_dists(np.concatenate(E), protos)

    print(f"KNOWN test images: {len(d_known)}")
    print(f"  distance: min {d_known.min():.3f} | mean {d_known.mean():.3f} | "
          f"90th {np.percentile(d_known,90):.3f} | max {d_known.max():.3f}")

    # UNKNOWN (optional)
    d_unknown = None
    if args.unknown_dir:
        tf = get_eval_transforms()
        paths = [p for p in Path(args.unknown_dir).glob("*")
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}]
        if paths:
            U = []
            with torch.no_grad():
                for p in paths:
                    x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
                    U.append(embed(model, x).cpu().numpy())
            d_unknown = nearest_dists(np.concatenate(U), protos)
            print(f"\nUNKNOWN images ({len(d_unknown)}) from {args.unknown_dir}:")
            print(f"  distance: min {d_unknown.min():.3f} | mean {d_unknown.mean():.3f} | max {d_unknown.max():.3f}")

    # sweep cutoffs
    print(f"\n{'cutoff':>7} | {'known kept':>11} | {'unknown rejected':>16}")
    print("-" * 42)
    for c in [0.40,0.45,0.48,0.50,0.52,0.55,0.58,0.60,0.65,0.71]:
        kk = float((d_known <= c).mean())
        ur = f"{float((d_unknown > c).mean()):.0%}" if d_unknown is not None else "  (no unknowns given)"
        star = "  <- saved" if abs(c - saved_cut) < 0.005 else ""
        print(f"{c:7.2f} | {kk:>10.0%} | {ur:>16}{star}")

    print("\nGoal: a cutoff that keeps ~all known (high left column) AND")
    print("rejects unknowns (high right column). Pick the value in the gap.")
    if d_unknown is None:
        print("\nTip: pass --unknown-dir with real non-heritage photos to see the right column.")


if __name__ == "__main__":
    main()
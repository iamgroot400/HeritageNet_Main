r"""
build_prototypes.py  —  Open-set support via feature-space prototypes.

Computes, for each of the 26 sites, the average 1024-dim feature embedding of its
TRAINING images (its "prototype"). Saved to export/prototypes.npz.

At inference, a new photo's embedding is compared (cosine distance) to the nearest
prototype. If even the nearest is FAR, the photo is rejected as "not a known site"
-- this catches out-of-distribution inputs (people, random objects) that a plain
softmax threshold lets through.

Also calibrates a distance cutoff using the TEST set (known) vs random-noise inputs
(stand-in unknowns), and prints how well it separates them.

Run:  python build_prototypes.py
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from heritagenet.data.datamodule import HeritageDataModule
from heritagenet.models.factory import build_model
try:
    from heritagenet.utils.device import get_device
except Exception:
    def get_device(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def embed(model, x):
    """1024-dim embedding: features -> avgpool -> classifier up to (not incl.) final Linear."""
    feat = model.features(x)
    pooled = model.avgpool(feat).flatten(1)
    z = model.classifier[0](pooled)      # Linear 576->1024
    z = model.classifier[1](z)           # Hardswish
    return z                             # skip Dropout (eval) + final Linear


@torch.no_grad()
def all_embeddings(model, ds, device, bs=32):
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    E, Y = [], []
    for x, y in loader:
        E.append(embed(model, x.to(device)).cpu().numpy()); Y.extend(y.tolist())
    return np.concatenate(E), np.array(Y)


def cosine_dist_matrix(V, P):
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    return 1.0 - Vn @ Pn.T          # [N, C] cosine distance to each prototype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/phase2_86acc_FINAL.pt")
    ap.add_argument("--model", default="mobilenet_v3_small")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="export/prototypes.npz")
    args = ap.parse_args()

    device = get_device()
    dm = HeritageDataModule(root=args.data, batch_size=32); dm.setup()
    classes = dm.classes; C = len(classes)

    model = build_model(args.model, num_classes=C, pretrained=False, freeze_backbone=False)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck.get("model_state", ck)); model.to(device).eval()
    print(f"loaded {args.ckpt} | {C} classes")

    # ---- prototypes from TRAIN embeddings ----
    Etr, Ytr = all_embeddings(model, dm.train_dataset, device)
    protos = np.stack([Etr[Ytr == i].mean(0) for i in range(C)])   # [C,1024]
    print(f"built {C} prototypes from {len(Etr)} training embeddings")

    # ---- calibrate cutoff: TEST (known) vs random noise (unknown stand-in) ----
    Ete, Yte = all_embeddings(model, dm.test_dataset, device)
    d_known = cosine_dist_matrix(Ete, protos).min(1)               # nearest-proto dist per test img

    rng = np.random.default_rng(0)
    noise = torch.randn(len(Ete), 3, 224, 224)
    with torch.no_grad():
        En = np.concatenate([embed(model, noise[i:i+32].to(device)).cpu().numpy()
                             for i in range(0, len(noise), 32)])
    d_unknown = cosine_dist_matrix(En, protos).min(1)

    # cutoff between the known spread and unknown spread
    cut = float((np.percentile(d_known, 92) + np.percentile(d_unknown, 8)) / 2)
    keep_known = float((d_known <= cut).mean())
    reject_unknown = float((d_unknown > cut).mean())

    print(f"\ndistance summary (cosine to nearest prototype):")
    print(f"  known  (test)  : mean {d_known.mean():.3f}  90th pct {np.percentile(d_known,90):.3f}")
    print(f"  unknown(noise) : mean {d_unknown.mean():.3f}  10th pct {np.percentile(d_unknown,10):.3f}")
    print(f"  suggested cutoff: {cut:.3f}")
    print(f"    -> keeps {keep_known:.0%} of known test images")
    print(f"    -> rejects {reject_unknown:.0%} of unknown (noise) images")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, prototypes=protos, classes=np.array(classes), cutoff=cut)
    print(f"\nsaved -> {args.out}  (prototypes + classes + cutoff)")
    print("note: noise is a weak stand-in for real unknowns; refine the cutoff with")
    print("      real out-of-set photos (people, other buildings) for a true estimate.")


if __name__ == "__main__":
    main()
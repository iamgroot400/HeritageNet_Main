r"""
export_onnx_openset.py  —  Export model to ONNX with TWO outputs: logits + embedding.

The embedding (1024-dim, before the final layer) lets the browser compute the
open-set distance gate. Also writes labels.json and, if export/prototypes.npz
exists, converts it to export/prototypes.json for the web app.

Run:  python export_onnx_openset.py
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from heritagenet.data.datamodule import HeritageDataModule
from heritagenet.models.factory import build_model


class WithEmbedding(nn.Module):
    """Wrap the model so forward() returns (logits, embedding)."""
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x):
        feat = self.m.features(x)
        pooled = self.m.avgpool(feat).flatten(1)
        emb = self.m.classifier[1](self.m.classifier[0](pooled))     # 1024-dim
        logits = self.m.classifier[3](self.m.classifier[2](emb))     # 26 logits
        return logits, emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/phase2_86acc_FINAL.pt")
    ap.add_argument("--model", default="mobilenet_v3_small")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="export/heritagenet.onnx")
    ap.add_argument("--protos", default="export/prototypes.npz")
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    device = torch.device("cpu")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    dm = HeritageDataModule(root=args.data, batch_size=1); dm.setup()
    classes = dm.classes; C = len(classes)

    base = build_model(args.model, num_classes=C, pretrained=False, freeze_backbone=False)
    ck = torch.load(args.ckpt, map_location=device)
    base.load_state_dict(ck.get("model_state", ck))
    model = WithEmbedding(base); model.eval()
    print(f"loaded {args.ckpt} | {C} classes")

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["input"], output_names=["logits", "embedding"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True,
    )
    print(f"exported -> {out}  ({out.stat().st_size/1e6:.2f} MB)  [outputs: logits + embedding]")

    # verify parity
    import onnxruntime as ort
    with torch.no_grad(): t_logits, t_emb = model(dummy)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    o_logits, o_emb = sess.run(["logits", "embedding"], {"input": dummy.numpy()})
    d1 = float(np.max(np.abs(t_logits.numpy() - o_logits)))
    d2 = float(np.max(np.abs(t_emb.numpy() - o_emb)))
    print(f"verify: logits diff {d1:.2e} | embedding diff {d2:.2e} "
          f"{'PASS' if max(d1,d2) < 1e-4 else 'WARN'}")

    # labels.json
    with open(out.with_name("labels.json"), "w") as f:
        json.dump({i: c for i, c in enumerate(classes)}, f, indent=2)
    print(f"wrote {out.with_name('labels.json')}")

    # prototypes.npz -> prototypes.json (for the browser)
    pz = Path(args.protos)
    if pz.is_file():
        npz = np.load(pz, allow_pickle=True)
        protos = npz["prototypes"].astype(float)
        # pre-normalize each prototype (so JS only needs to normalize the query)
        norms = np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8
        protos_n = (protos / norms)
        data = {
            "classes": [str(c) for c in npz["classes"]],
            "cutoff": float(npz["cutoff"]),
            "prototypes_normalized": protos_n.round(6).tolist(),
        }
        with open(out.with_name("prototypes.json"), "w") as f:
            json.dump(data, f)
        print(f"wrote {out.with_name('prototypes.json')}  (cutoff {data['cutoff']:.3f})")
    else:
        print(f"note: {pz} not found — run build_prototypes.py first for the distance gate.")


if __name__ == "__main__":
    main()
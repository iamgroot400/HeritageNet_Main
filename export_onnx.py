r"""
export_onnx.py  —  Export the trained model to ONNX for on-device deployment.

Loads a checkpoint, exports to ONNX, then VERIFIES the ONNX model produces the
same outputs as the original PyTorch model (this is the step most people skip —
export can silently change predictions, so we check parity).

Also writes labels.json (index -> site name) so the deployed app knows what each
output means, and reports the file size.

Run from repo root (venv active):

    python export_onnx.py
    python export_onnx.py --ckpt checkpoints\phase2_86acc_FINAL.pt --out heritagenet.onnx
"""

from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch

from heritagenet.data.datamodule import HeritageDataModule
from heritagenet.models.factory import build_model

try:
    from heritagenet.utils.device import get_device
except Exception:  # pragma: no cover
    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser(description="Export HeritageNet to ONNX (with verification).")
    ap.add_argument("--ckpt", default="checkpoints/phase2_86acc_FINAL.pt",
                    help="checkpoint to export (default: the champion)")
    ap.add_argument("--model", default="mobilenet_v3_small")
    ap.add_argument("--data", default="data", help="used only to read class names")
    ap.add_argument("--out", default="export/heritagenet.onnx")
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    device = torch.device("cpu")   # export on CPU for portability
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- class names (so the app knows what each output index means) ----
    dm = HeritageDataModule(root=args.data, batch_size=1); dm.setup()
    classes = dm.classes
    num_classes = len(classes)

    # ---- load model ----
    model = build_model(args.model, num_classes=num_classes,
                        pretrained=False, freeze_backbone=False)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"loaded {args.ckpt} | {num_classes} classes")

    # ---- export ----
    dummy = torch.randn(1, 3, 224, 224, device=device)   # a fake batch of 1 image
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},  # allow any batch size
        opset_version=args.opset, do_constant_folding=True,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"exported -> {out_path}  ({size_mb:.2f} MB)")

    # ---- VERIFY parity: PyTorch vs ONNX Runtime on the same input ----
    import onnxruntime as ort
    with torch.no_grad():
        torch_out = model(dummy).cpu().numpy()

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["logits"], {"input": dummy.cpu().numpy()})[0]

    max_diff = float(np.max(np.abs(torch_out - onnx_out)))
    same_pred = int(torch_out.argmax()) == int(onnx_out.argmax())
    print(f"\nverification:")
    print(f"  max abs difference (PyTorch vs ONNX): {max_diff:.2e}")
    print(f"  same top prediction: {same_pred}")
    if max_diff < 1e-4 and same_pred:
        print("  PASS — ONNX matches PyTorch.")
    else:
        print("  WARNING — outputs differ more than expected; inspect before deploying.")

    # ---- labels sidecar ----
    labels_path = out_path.with_name("labels.json")
    with open(labels_path, "w") as f:
        json.dump({i: c for i, c in enumerate(classes)}, f, indent=2)
    print(f"\nwrote labels -> {labels_path}")
    print("done. deploy heritagenet.onnx + labels.json together.")


if __name__ == "__main__":
    main()
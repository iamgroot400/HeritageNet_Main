# HeritageNet

A lightweight computer-vision classifier that identifies **26 Nepali heritage sites**
(temples, stupas, and monasteries) from photographs, built with transfer learning on
**MobileNetV3-small** and designed for **on-device deployment**.

> Nepali heritage sites are barely represented in existing ML datasets. HeritageNet
> contributes both a trained model and an accompanying dataset (photographed by the team).

**Champion model:** 86.6% test accuracy · 85.8% macro-F1 (26 classes).

---

## Highlights

- **Transfer learning** from ImageNet-pretrained MobileNetV3-small (compact, mobile-friendly).
- **Two-phase training:** frozen-backbone baseline → full fine-tuning.
- **Rigorous evaluation:** macro-F1, per-class metrics, confusion analysis.
- **Open-set rejection:** distance-to-prototype gate declines out-of-distribution inputs
  ("I don't know") in addition to a confidence threshold.
- **On-device deployment:** exported to ONNX, runs in the browser via ONNX Runtime Web.

## Results

| Strategy | Trainable params | Test accuracy | Macro-F1 |
|---|---|---|---|
| Phase 1 (frozen backbone) | ~1.7% | 79.9% | 78.4% |
| Partial (last 2 blocks)   | ~63%  | 82.7% | 80.6% |
| **Phase 2 (full fine-tune)** | 100% | **86.6%** | **85.8%** |

Controlled dropout/learning-rate/partial-freezing experiments showed the model is
**data-limited** (not capacity- or regularization-limited): reducing overfitting did not
improve test accuracy, and the weakest classes are simply those with the fewest images.

## Project structure

```
heritagenet/            # installable package
  data/                 # datamodule + transforms
  models/               # model factory (build_model, freezing, head replacement)
  engine/               # trainer (AMP, early stopping, checkpointing)
  utils/                # device + seed helpers
scripts/                # split_dataset, evaluate, confusion, save_mistakes
train.py                # phase-1 training (frozen backbone)
train_phase2.py         # phase-2 fine-tuning (full unfreeze)
train_phase2_partial.py # partial fine-tuning (unfreeze last N blocks)
predict.py              # single-image prediction (+ confidence threshold)
predict_openset.py      # prediction with open-set distance rejection
build_prototypes.py     # compute per-class feature prototypes (open-set)
check_cutoff.py         # calibrate the open-set distance cutoff
export_onnx.py          # export model to ONNX (logits)
export_onnx_openset.py  # export model to ONNX (logits + embedding, for open-set)
notebooks/              # analysis dashboards
webapp/                 # browser demo (ONNX Runtime Web) — runs on-device
```

> **Not tracked in git:** `data/`, `datasets/`, `checkpoints/`, `outputs/`, `export/`,
> `.venv/`, and model weights (`*.pt`, `*.onnx`). See `.gitignore`. These are large and/or
> generated; the repo is code-only.

## Setup

Requires Python 3.12 and (for GPU) a CUDA-capable card. On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate.bat
pip install -e .
# Install PyTorch with CUDA (example for cu124):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Arrange images as `datasets/<site_name>/*.jpg`, then create the stratified split:

```bash
python scripts/split_dataset.py --clean
```

## Training

```bash
# Phase 1: frozen backbone (baseline)
python train.py

# Phase 2: full fine-tuning (champion recipe: lr 1e-4, dropout 0.2)
python train_phase2.py --lr 1e-4
```

Checkpoints are written to `checkpoints/`. Keep a protected copy of your best model
(this project used `checkpoints/phase2_86acc_FINAL.pt`).

> **Convention:** run one-off experiments via CLI flags / per-script arguments, and save
> each to a distinct checkpoint name. Do **not** edit shared defaults in
> `heritagenet/models/factory.py` or `heritagenet/engine/trainer.py` — that breaks
> baseline reproducibility.

## Evaluation

```bash
python scripts/evaluate.py --ckpt checkpoints/phase2_86acc_FINAL.pt
```

Reports accuracy, macro-F1, a per-class report, and confusion matrices.

## Open-set rejection

The classifier is closed-set: it always predicts one of the 26 sites. To decline
out-of-distribution inputs, HeritageNet adds a **feature-distance gate**:

```bash
python build_prototypes.py      # compute 26 class prototypes + a suggested cutoff
python check_cutoff.py           # inspect known-distance distribution; tune the cutoff
python predict_openset.py --image path/to/photo.jpg
```

An input is accepted only if (a) softmax confidence ≥ threshold **and**
(b) cosine distance to the nearest prototype ≤ cutoff. This reliably rejects clearly
dissimilar inputs (people, unrelated objects); it cannot reliably reject inputs that
share heritage-like visual features (e.g. unlisted temples or generated heritage scenes),
which is an inherent limitation of distance-based open-set methods.

## Deployment (on-device, browser)

```bash
python export_onnx_openset.py    # exports export/heritagenet.onnx (logits + embedding)
                                 # + labels.json + prototypes.json
```

Copy `heritagenet.onnx`, `labels.json`, `prototypes.json` into `webapp/` alongside
`index.html`, then serve locally:

```bash
cd webapp
python -m http.server 8000
# open http://localhost:8000
```

The web app runs the model entirely in the browser (ONNX Runtime Web) — images never
leave the device.

## Limitations

- **Data-limited:** several sites have few images; the weakest classes are the smallest.
- **Closed-set by nature:** the open-set gate mitigates but does not fully solve rejection
  of near-distribution unknowns (other temples, synthetic heritage images).
- **Regional scope:** limited to the 26 sites in the dataset.

## Dataset

The accompanying dataset (photographed by the team) is intended for open release under
**CC BY 4.0**. See the dataset repository for the data, class list, and dataset card.
*(Link to be added once published.)*

## License

Code: *(choose one — e.g. MIT or Apache-2.0 — add a LICENSE file).*
Dataset: **CC BY 4.0** (released separately).

## Team & Acknowledgements

Built as a research/thesis project. The dataset was photographed by the team.

**Mentors:** Sijan Shrestha · Hari Om Sah

**Team:** Ashutosh Chapagai· Rasim Mahato · Diva Awasthi · Kriti Puri · Suyog Adhikari · Pratik Sharma

With thanks to everyone who contributed to photographing and labelling the heritage sites.

# HeritageNet

Scalable heritage-site image classification. New classes are added simply by
creating a new folder under `data/train/` (and `val/`, `test/`) — no code
changes required. Final target model is MobileNet, trained with modern
techniques (transfer learning, augmentation, and later knowledge distillation).

## Project layout

```
HeritageNet/
├── heritagenet/          # installable package (all reusable logic)
│   ├── config.py         # typed YAML config system
│   ├── data/             # dataset detection, transforms, sampling
│   ├── models/           # model factory + distillation
│   ├── engine/           # training loop, losses, metrics
│   ├── utils/            # seed, device, checkpoint, logging
│   └── viz/              # plots, confusion matrix, Grad-CAM, embeddings
├── configs/              # one YAML per experiment
├── scripts/              # thin entry points (train, evaluate, predict, ...)
├── data/                 # images (gitignored) -> train/ val/ test/
├── checkpoints/          # saved weights (gitignored)
├── logs/                 # tensorboard + csv (gitignored)
├── outputs/              # plots & reports (gitignored)
├── notebooks/            # exploration
└── tests/                # unit tests
```

## Setup

```bash
# 1. create and activate a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# 2. install the CUDA build of torch matching your GPU driver:
#    https://pytorch.org/get-started/locally/
#    then install this package in editable mode:
pip install -e .

# 3. verify
python -c "from heritagenet.utils.device import get_device, describe_device; print(describe_device(get_device()))"
```

## Progress

- [x] Config system (`heritagenet/config.py`)
- [x] Reproducibility (`heritagenet/utils/seed.py`)
- [x] Device selection (`heritagenet/utils/device.py`)
- [ ] Data module (next)
- [ ] Model factory
- [ ] Training engine
- [ ] Evaluation
- [ ] Knowledge distillation
- [ ] Export (ONNX / TorchScript / TFLite)

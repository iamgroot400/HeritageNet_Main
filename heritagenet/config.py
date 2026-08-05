"""Typed configuration system for HeritageNet.

Every experiment setting lives in a YAML file and is loaded into typed
dataclasses. We deliberately avoid passing raw dictionaries around the codebase.

Why dataclasses instead of plain dicts:
  * Autocomplete + static checking: you write ``cfg.optim.lr`` and your editor
    knows it is a float. With a dict, ``cfg["optim"]["lr"]`` is unchecked.
  * Single source of defaults: a minimal YAML still yields a valid, complete
    config because missing keys fall back to the dataclass defaults.
  * Fail loud, fail early: a typo such as ``cfg.optim.learnign_rate`` raises an
    AttributeError immediately, instead of silently returning ``None`` at 3 a.m.
    two hours into training.
"""

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Union

import yaml


# --------------------------------------------------------------------------- #
# Config sections. Each dataclass groups related settings and defines defaults.
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """Where the images are and how they are loaded into batches."""

    root: str = "data"           # folder holding train/ val/ test/
    image_size: int = 224        # MobileNet's native input resolution
    batch_size: int = 32         # safe for a 4 GB RTX 3050 with AMP
    num_workers: int = 4         # parallel data-loading processes
    pin_memory: bool = True      # faster host->GPU transfer on CUDA


@dataclass
class ModelConfig:
    """Which architecture to build and how to configure its head."""

    name: str = "mobilenet_v3_large"
    pretrained: bool = True      # start from ImageNet weights (transfer learning)
    dropout: float = 0.2         # regularisation in the classifier head


@dataclass
class OptimConfig:
    """Optimiser and its hyper-parameters."""

    name: str = "adamw"          # adamw | sgd
    lr: float = 1e-3
    weight_decay: float = 1e-4   # L2 regularisation; crucial on small datasets
    momentum: float = 0.9        # used only by SGD


@dataclass
class TrainConfig:
    """The training loop's behaviour."""

    epochs: int = 50
    amp: bool = True                     # mixed precision (Ampere Tensor Cores)
    seed: int = 42                       # reproducibility
    early_stopping_patience: int = 10    # epochs w/o val improvement before stop
    label_smoothing: float = 0.1         # softens targets; helps generalisation


@dataclass
class Config:
    """Top-level config: one object that holds every section."""

    experiment_name: str = "default"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# --------------------------------------------------------------------------- #
# Loading / saving.
# --------------------------------------------------------------------------- #
def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a (possibly nested) dataclass from a plain dict.

    Only keys present in ``data`` override defaults; unknown keys are ignored so
    an old config never crashes a newer codebase, and vice-versa.
    """
    if not is_dataclass(cls):
        return data
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue  # keep the dataclass default
        value = data[f.name]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[f.name] = _from_dict(f.type, value)  # recurse into sub-config
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: Union[str, Path]) -> Config:
    """Read a YAML file and return a fully-populated :class:`Config`."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _from_dict(Config, raw)


def save_config(cfg: Config, path: Union[str, Path]) -> None:
    """Dump a :class:`Config` back to YAML.

    We call this at the start of every run and write it into the run's output
    folder, so each experiment carries an exact, human-readable record of how it
    was configured.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(asdict(cfg), fh, sort_keys=False)

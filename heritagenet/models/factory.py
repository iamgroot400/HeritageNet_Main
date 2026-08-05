"""
heritagenet/models/factory.py

One entry point for building every model: build_model(...).

No other file (trainer, eval, predict) should ever import torchvision models
directly. They ask the factory for "a model with N output classes" and get a
ready-to-train nn.Module back. All model-specific detail lives here alone.

Built around your constraint: few classes, few images each. That is why
transfer learning + a frozen backbone + a small trainable head is the DEFAULT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn
import torchvision.models as tvm


# ---------------------------------------------------------------------------
# 1. Registry: every supported backbone in one lookup table.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Backbone:
    ctor: Callable[..., nn.Module]
    weights: object


_REGISTRY: dict[str, _Backbone] = {
    "mobilenet_v3_small": _Backbone(
        tvm.mobilenet_v3_small,
        tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
    ),
    "mobilenet_v3_large": _Backbone(
        tvm.mobilenet_v3_large,
        tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    ),
    "efficientnet_v2_s": _Backbone(
        tvm.efficientnet_v2_s,
        tvm.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
    ),
}


def available_models() -> list[str]:
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# helper: locate the final nn.Linear inside model.classifier
# ---------------------------------------------------------------------------
def _final_linear(model: nn.Module) -> nn.Linear:
    last = None
    for layer in model.classifier:
        if isinstance(layer, nn.Linear):
            last = layer
    if last is None:
        raise RuntimeError("No nn.Linear found in classifier.")
    return last


# ---------------------------------------------------------------------------
# 2. Head replacement: swap ONLY the final Linear for OUR class count.
# ---------------------------------------------------------------------------
def _replace_classifier(model: nn.Module, num_classes: int, dropout: float) -> None:
    """
    torchvision classifiers are nn.Sequential. MobileNetV3-small's is:

        Linear(576->1024) -> Hardswish -> Dropout -> Linear(1024->1000)

    We must replace ONLY the final Linear (the 1000-way ImageNet layer),
    preserving the 576->1024 projection in front of it. Replacing the whole
    block would drop that projection and break the input dimension.

    We locate the final Linear by scanning the Sequential, read its in_features,
    and swap in a fresh Linear(in_features -> num_classes). We also set the
    dropout probability on any existing Dropout layer in the head.
    """
    classifier = model.classifier
    if not isinstance(classifier, nn.Sequential):
        raise TypeError(f"Expected nn.Sequential classifier, got {type(classifier)}.")

    # find the index of the last Linear
    last_idx = None
    for i, layer in enumerate(classifier):
        if isinstance(layer, nn.Linear):
            last_idx = i
    if last_idx is None:
        raise RuntimeError("No nn.Linear in classifier; cannot size the new head.")

    in_features = classifier[last_idx].in_features
    classifier[last_idx] = nn.Linear(in_features, num_classes)

    # tune existing dropout(s) to the requested strength
    for layer in classifier:
        if isinstance(layer, nn.Dropout):
            layer.p = dropout


# ---------------------------------------------------------------------------
# 3. Freezing: phase-1 (final layer only) vs phase-2 (whole network).
# ---------------------------------------------------------------------------
def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """
    Phase 1  set_backbone_trainable(model, False): freeze everything, then
             unfreeze ONLY the final classification Linear. This is a linear
             probe — the safest, least overfitting-prone start on small data.
    Phase 2  set_backbone_trainable(model, True): unfreeze the whole network
             for gentle fine-tuning at a much lower learning rate.
    """
    head = _final_linear(model)
    for p in model.parameters():
        p.requires_grad = trainable
    # the final head layer always trains
    for p in head.parameters():
        p.requires_grad = True


# ---------------------------------------------------------------------------
# 4. Public entry point.
# ---------------------------------------------------------------------------
def build_model(
    name: str = "mobilenet_v3_small",
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Available: {available_models()}")

    spec = _REGISTRY[name]
    model = spec.ctor(weights=spec.weights if pretrained else None)
    _replace_classifier(model, num_classes=num_classes, dropout=dropout)
    if freeze_backbone:
        set_backbone_trainable(model, trainable=False)
    return model


# ---------------------------------------------------------------------------
# 5. Diagnostics.
# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


if __name__ == "__main__":
    m = build_model("mobilenet_v3_small", num_classes=13)
    trn, tot = count_parameters(m)
    print("model: mobilenet_v3_small")
    print(f"trainable params: {trn:,}  /  total: {tot:,}")
    print(f"frozen: {100 * (1 - trn / tot):.1f}% of weights")
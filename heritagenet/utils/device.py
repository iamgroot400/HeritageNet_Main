"""
heritagenet/utils/device.py

Pick the best available compute device, once, in one place. Every other module
(trainer, eval, predict) calls get_device() instead of hard-coding "cuda" —
so the exact same code runs on your RTX 3050 laptop and on a CPU-only machine.
"""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """
    Return the best available device:
      * CUDA GPU if present  (your RTX 3050)
      * Apple MPS if on a Mac with Apple silicon
      * CPU otherwise (always works, just slower)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """
    Human-readable one-line summary of a device — handy for logging at the
    start of every run so your training logs record what they ran on.
    """
    if device.type == "cuda":
        idx = device.index or 0
        name = torch.cuda.get_device_name(idx)
        total_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        return f"CUDA:{idx} — {name} ({total_gb:.1f} GB)"
    if device.type == "mps":
        return "Apple MPS (Metal GPU)"
    return "CPU"
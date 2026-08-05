"""
heritagenet/utils/seed.py

Pin every source of randomness so runs are reproducible. Call set_seed() once
at the very start of any script (train, eval) before building data or models.

Why it matters for research: if two runs with identical settings give different
accuracy, you can't attribute a change to your idea rather than luck. Seeding
removes that ambiguity so your experiments are comparable and defensible.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Seed Python, NumPy, and PyTorch (CPU + CUDA).

    Parameters
    ----------
    seed : the number to seed everything with. Any fixed int works; 42 is a
        common convention. The value doesn't matter — keeping it *constant*
        across runs is what matters.
    deterministic : if True, ask cuDNN to use deterministic algorithms. This
        makes GPU results repeatable at a small speed cost. Keep it True while
        running experiments for the report; you can set False for a final
        speed-focused training run if you want.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make hash-based operations (e.g. set ordering) reproducible too.
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Faster, but introduces nondeterminism in some GPU ops.
        torch.backends.cudnn.benchmark = True
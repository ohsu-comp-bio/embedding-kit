"""
Reproducibility helper.

Embedding results depend on random initialization, shuffling and sampling.
Seeding every random number generator in one call makes experiments
repeatable across runs.

Ported from UniVII (``univii.utils.seed``).
"""
import random

import numpy as np
import torch


def set_seed(seed: int = 0, deterministic: bool = False) -> None:
    """
    Seed Python's ``random``, NumPy and PyTorch (CPU and all CUDA devices).

    Args:
        seed: Seed value.
        deterministic: Also force deterministic cuDNN kernels (slower, but
            bit-for-bit repeatable on GPU).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

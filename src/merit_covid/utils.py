"""Explicit reproducibility setup; importing the package never allocates a GPU."""

import random
import numpy as np


def seed_everything(seed: int, *, seed_torch: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)

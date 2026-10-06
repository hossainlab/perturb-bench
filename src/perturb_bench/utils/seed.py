"""Deterministic reproducibility utilities."""
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    """Set random seeds for python, numpy, and torch to guarantee reproducibility.

    Parameters
    ----------
    seed : int, default=0
        Random seed to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

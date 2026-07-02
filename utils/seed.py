"""Reproducibility helpers: global seeding and device selection."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed all RNGs used across the pipeline (Python, NumPy, PyTorch).

    Args:
        seed: Seed value applied to every RNG.
        deterministic: If True, force deterministic cuDNN/algorithm behavior.
            Slower but reproducible; disable for maximum training throughput.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.benchmark = True


def get_device(prefer: str | None = None) -> torch.device:
    """Resolve the best available compute device with CPU fallback.

    Args:
        prefer: Optional explicit device string (e.g. "cuda", "mps", "cpu").
            If given and available, it is used as-is.
    """
    if prefer is not None:
        if prefer == "cuda" and not torch.cuda.is_available():
            pass
        elif prefer == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            pass
        else:
            return torch.device(prefer)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

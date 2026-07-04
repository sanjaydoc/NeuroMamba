"""Small shared utilities: logging, seeding, device resolution."""
from __future__ import annotations

import logging
import os
import random

_LOGGER_CONFIGURED = False


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure a single logger for the package (idempotent)."""
    global _LOGGER_CONFIGURED
    logger = logging.getLogger("neuromamba")
    if not _LOGGER_CONFIGURED:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
        _LOGGER_CONFIGURED = True
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and (if available) PyTorch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(device: str = "auto") -> str:
    """Resolve "auto"/"cuda"/"cpu" to an available device string."""
    if device == "cpu":
        return "cpu"
    try:
        import torch

        if device in ("auto", "cuda") and torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"

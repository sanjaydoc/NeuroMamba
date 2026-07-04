"""NeuroMamba — a from-scratch selective state-space (Mamba/S6) protein language
model for de novo generation of novel PDZ-domain sequences.

Public API::

    from neuromamba import ProteinMamba, ProteinTokenizer, sample, train
    from neuromamba import oracle, metrics
"""
from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Dr. Sanjay Anbu"

from neuromamba.generate import sample, sample_from_checkpoint
from neuromamba.mamba import MambaBlock, ProteinMamba, build_mamba
from neuromamba.tokenizer import ProteinTokenizer

__all__ = [
    "ProteinMamba",
    "MambaBlock",
    "build_mamba",
    "ProteinTokenizer",
    "sample",
    "sample_from_checkpoint",
    "__version__",
    "__author__",
]

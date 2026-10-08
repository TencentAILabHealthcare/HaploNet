# Copyright (c) 2026, Tencent Inc. All rights reserved.
"""
HaploNet: Haplotype-aware methylation detection from ONT sequencing.

Usage:
    import haplonet
    model = haplonet.model.MethylationUSM(...)
"""

__version__ = "0.1.0"

from .model import MethylationUSM, USMBase, XNATokenizer
from .model.utils import masked_mean, collate_dense_tensors

__all__ = [
    "MethylationUSM",
    "USMBase", 
    "XNATokenizer",
    "masked_mean",
    "collate_dense_tensors",
]

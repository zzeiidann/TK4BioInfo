"""
Sequence Alignment Module
Provides tools for pairwise and multiple sequence alignment
"""

from .pairwise import (
    PairwiseAligner,
    AlignmentResult,
    pairwise
)

__all__ = [
    "PairwiseAligner",
    "AlignmentResult",
    "pairwise"
]
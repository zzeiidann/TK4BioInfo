"""
DNA distance calculations for phylogenetic analysis (R-compatible)
- Supports pairwise deletion like ape::dist.dna(..., pairwise.deletion=TRUE)
- Method aliases: 'p', 'jc69'/'jukes_cantor', 'k80'/'kimura'
"""
import numpy as np
import asyncio
from typing import Dict, List, Optional
import math
from functools import partial

_VALID = {"A", "C", "G", "T"}
_MISSING = {"-", "N", "?", "X"}  # treated as missing if pairwise_deletion=True


def _norm(seq: str) -> str:
    # Uppercase, convert U->T
    return seq.upper().replace("U", "T")


def _iter_valid_pairs(seq1: str, seq2: str):
    """Yield pairs where both bases are valid (A/C/G/T)."""
    for a, b in zip(_norm(seq1), _norm(seq2)):
        if a in _VALID and b in _VALID:
            yield a, b


def raw_distance(seq1: str, seq2: str) -> int:
    """Raw number of different positions (no missing handling)."""
    return sum(1 for a, b in zip(_norm(seq1), _norm(seq2)) if a != b)


def p_distance(seq1: str, seq2: str, pairwise_deletion: bool = False) -> float:
    """
    p-distance (proportion of different sites).
    If pairwise_deletion=True, only counts positions where both are A/C/G/T.
    """
    s1, s2 = _norm(seq1), _norm(seq2)
    if not pairwise_deletion and len(s1) != len(s2):
        raise ValueError("Sequences must have equal length when pairwise_deletion=False")

    if pairwise_deletion:
        valid = 0
        diff = 0
        for a, b in _iter_valid_pairs(s1, s2):
            valid += 1
            if a != b:
                diff += 1
        if valid == 0:
            return float("inf")
        return diff / valid
    else:
        differences = sum(1 for a, b in zip(s1, s2) if a != b)
        return differences / len(s1)


def jukes_cantor_distance(seq1: str, seq2: str, pairwise_deletion: bool = False) -> float:
    """
    Jukes-Cantor corrected distance:
      d = -3/4 * ln(1 - 4p/3)
    p computed with or without pairwise deletion.
    """
    p = p_distance(seq1, seq2, pairwise_deletion=pairwise_deletion)
    if p >= 0.75 or not np.isfinite(p):
        return float("inf")
    try:
        return -0.75 * math.log(1 - (4 * p / 3))
    except (ValueError, ZeroDivisionError):
        return float("inf")


def kimura_distance(seq1: str, seq2: str, pairwise_deletion: bool = False) -> float:
    """
    Kimura 2-parameter (K80):
      d = -0.5 * ln((1-2P-Q) * sqrt(1-2Q))
    P = transitions proportion, Q = transversions proportion.
    If pairwise_deletion=True, compute over valid A/C/G/T pairs only.
    """
    s1, s2 = _norm(seq1), _norm(seq2)
    if not pairwise_deletion and len(s1) != len(s2):
        raise ValueError("Sequences must have equal length when pairwise_deletion=False")

    purines = {"A", "G"}
    pyrimidines = {"C", "T"}

    transitions = 0
    transversions = 0
    valid = 0

    if pairwise_deletion:
        pairs = _iter_valid_pairs(s1, s2)
    else:
        pairs = zip(s1, s2)

    for a, b in pairs:
        # When pairwise_deletion=False, count all positions (assumes already A/C/G/T data)
        if pairwise_deletion is False:
            # No missing filtering here; sequences are expected to be clean MSAs.
            valid += 1
            if a != b:
                if (a in purines and b in purines) or (a in pyrimidines and b in pyrimidines):
                    transitions += 1
                else:
                    transversions += 1
        else:
            # pairwise_deletion=True path already filters to A/C/G/T via _iter_valid_pairs
            valid += 1
            if a != b:
                if (a in purines and b in purines) or (a in pyrimidines and b in pyrimidines):
                    transitions += 1
                else:
                    transversions += 1

    if valid == 0:
        return float("inf")

    P = transitions / valid
    Q = transversions / valid
    try:
        return -0.5 * math.log((1 - 2 * P - Q) * math.sqrt(1 - 2 * Q))
    except (ValueError, ZeroDivisionError):
        return float("inf")


def _select_dist_func(method: str, pairwise_deletion: bool):
    """Return a callable expecting (seq1, seq2) with pairwise_deletion bound."""
    m = (method or "p").lower()
    if m in ("raw",):
        return raw_distance
    if m in ("p", "pdist", "p-distance"):
        return partial(p_distance, pairwise_deletion=pairwise_deletion)
    if m in ("jc", "jc69", "jukes_cantor", "jukes-cantor"):
        return partial(jukes_cantor_distance, pairwise_deletion=pairwise_deletion)
    if m in ("k80", "kimura", "kimura2p", "kimura-2p"):
        return partial(kimura_distance, pairwise_deletion=pairwise_deletion)
    raise ValueError(f"Unknown distance method: {method}")


def compute_distance_matrix(
    sequences: Dict[str, str],
    method: str = "p",
    *,
    model: Optional[str] = None,
    pairwise_deletion: bool = False,
) -> np.ndarray:
    """
    Compute pairwise distance matrix.

    Args:
        sequences: dict {taxon -> sequence}
        method: 'p' | 'jc69' | 'k80' | 'raw' (case-insensitive)
        model: optional alias ('JC69', 'K80'); if given, overrides method
        pairwise_deletion: if True, ignore sites where either taxon has non-ACGT

    Returns:
        np.ndarray (n x n) symmetric distance matrix (diag=0)
    """
    if model is not None:
        method = model  # allow R-style naming

    taxa = list(sequences.keys())
    n = len(taxa)
    dist_matrix = np.zeros((n, n), dtype=float)

    dist_func = _select_dist_func(method, pairwise_deletion)

    for i in range(n):
        for j in range(i + 1, n):
            d = dist_func(sequences[taxa[i]], sequences[taxa[j]])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix


async def compute_distance_matrix_async(
    sequences: Dict[str, str],
    method: str = "p",
    *,
    model: Optional[str] = None,
    pairwise_deletion: bool = False,
) -> np.ndarray:
    """Async version of compute_distance_matrix."""
    if model is not None:
        method = model
    taxa = list(sequences.keys())
    n = len(taxa)
    dist_matrix = np.zeros((n, n), dtype=float)
    dist_func = _select_dist_func(method, pairwise_deletion)

    async def calc(i: int, j: int):
        loop = asyncio.get_event_loop()
        d = await loop.run_in_executor(None, dist_func, sequences[taxa[i]], sequences[taxa[j]])
        return i, j, d

    tasks = [calc(i, j) for i in range(n) for j in range(i + 1, n)]
    for i, j, d in await asyncio.gather(*tasks):
        dist_matrix[i, j] = d
        dist_matrix[j, i] = d
    return dist_matrix


def calculate_nucleotide_frequencies(sequences: Dict[str, str]) -> Dict[str, float]:
    """Global nucleotide frequencies over A/C/G/T only."""
    counts = {"A": 0, "C": 0, "G": 0, "T": 0}
    total = 0
    for seq in sequences.values():
        for base in _norm(seq):
            if base in counts:
                counts[base] += 1
                total += 1
    return {b: (counts[b] / total if total > 0 else 0.0) for b in counts}

"""
High-performance DNA distance calculations for phylogenetic analysis.

Fitur:
- Metode: 'raw', 'p', 'jc69' (Jukes–Cantor), 'k80' (Kimura 2-parameter)
- Pairwise deletion (abaikan gap/ambiguous per-pasangan)
- Backend:
    - 'auto'     : GPU (CuPy) jika ada dan memori cukup, kalau tidak CPU cepat
    - 'gpu'      : paksa CuPy (akan fallback ke CPU bila tak tersedia)
    - 'cpu'      : NumPy vectorized (cepat untuk n<=~200)
    - 'parallel' : multi-proses per-pasangan (aman memori; bagus saat n besar)
- Async helper untuk dipanggil dari event loop

Catatan GPU:
- Butuh CuPy: `pip install cupy-cuda12x` (sesuaikan versi CUDA)
- Untuk n dan L besar, mode GPU pakai broadcasting (n,n,L) -> gunakan
  `gpu_max_elems` agar auto fallback ke CPU ketika tidak muat.

© TK4BioInfo
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- opsional GPU (CuPy) ---
try:
    import cupy as cp  # type: ignore
    _HAS_CUPY = True
except Exception:
    cp = None  # type: ignore
    _HAS_CUPY = False

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat


# =========================
# Utilities
# =========================

_VALID = np.array([ord(c) for c in "ACGTacgt"])
_TRANSLATE = np.full(256, -1, dtype=np.int8)
# A=0, C=1, G=2, T=3
_TRANSLATE[ord('A')] = 0; _TRANSLATE[ord('a')] = 0
_TRANSLATE[ord('C')] = 1; _TRANSLATE[ord('c')] = 1
_TRANSLATE[ord('G')] = 2; _TRANSLATE[ord('g')] = 2
_TRANSLATE[ord('T')] = 3; _TRANSLATE[ord('t')] = 3

# karakter yang dianggap invalid (gap/ambiguous) -> kode -1
for bad in "-?NnXxRYMKSWHBVDrymkswhbvd":
    _TRANSLATE[ord(bad)] = -1


def _encode_numpy(seqs: Dict[str, str]) -> Tuple[np.ndarray, List[str]]:
    """
    Ubah dict {name: 'ACGT…'} menjadi array int8 shape (n, L) dengan
    A=0,C=1,G=2,T=3, lainnya -1. Jika panjang beda, dipotong ke min(L).
    """
    taxa = list(seqs.keys())
    # uppercase agar konsisten
    clean = [seqs[t].upper() for t in taxa]
    L = min(len(s) for s in clean)
    if L == 0:
        raise ValueError("Empty sequences detected.")

    arr = np.empty((len(taxa), L), dtype=np.int8)
    for i, s in enumerate(clean):
        # only first L chars (MSA seharusnya sama panjang)
        b = s.encode("ascii", errors="ignore")[:L]
        codes = _TRANSLATE[np.frombuffer(b, dtype=np.uint8)]
        arr[i] = codes
    return arr, taxa


# =========================
# Pairwise distance kernels (NumPy / CuPy)
# =========================

def _dist_matrix_numpy(enc: np.ndarray,
                       method: str,
                       pairwise_deletion: bool) -> np.ndarray:
    """
    NumPy vectorized distance matrix (broadcasting).
    Memori: O(n^2 L). Cepat untuk n <= ~200 dan L moderat.
    """
    X = enc  # (n, L)
    n, L = X.shape
    # valid mask per pair
    valid = (X[:, None, :] >= 0) & (X[None, :, :] >= 0)  # (n,n,L)
    if pairwise_deletion:
        denom = valid.sum(axis=2).astype(np.float64)      # (n,n)
    else:
        # jika tak pakai pairwise deletion, posisi invalid dibuang global
        glob_valid = (X >= 0).all(axis=0)
        valid = valid & glob_valid[None, None, :]
        denom = np.full((n, n), glob_valid.sum(), dtype=np.float64)
        if denom[0, 0] == 0:
            raise ValueError("No valid columns remain after filtering.")

    # mismatch
    mism = (X[:, None, :] != X[None, :, :]) & valid
    diff_counts = mism.sum(axis=2).astype(np.float64)  # (n,n)
    # hindari /0
    denom = np.where(denom == 0, np.nan, denom)

    if method == 'raw':
        D = diff_counts
    elif method == 'p':
        D = diff_counts / denom
    elif method in ('jc69', 'jukes_cantor'):
        p = diff_counts / denom
        # JC69
        with np.errstate(divide='ignore', invalid='ignore'):
            D = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
        D = np.where((p >= 0.75) | ~np.isfinite(D), np.inf, D)
    elif method in ('k80', 'kimura'):
        # transitions = abs diff == 2 (A<->G, C<->T) pada encoding 0,1,2,3
        absdiff = np.abs(X[:, None, :] - X[None, :, :])
        transi = (absdiff == 2) & mism           # (n,n,L)
        P = transi.sum(axis=2).astype(np.float64) / denom
        Q = (diff_counts - transi.sum(axis=2)) / denom
        with np.errstate(divide='ignore', invalid='ignore'):
            D = -0.5 * np.log((1 - 2*P - Q) * np.sqrt(1 - 2*Q))
        D = np.where(~np.isfinite(D), np.inf, D)
    else:
        raise ValueError(f"Unknown method: {method}")

    np.fill_diagonal(D, 0.0)
    return D


def _dist_matrix_cupy(enc: np.ndarray,
                      method: str,
                      pairwise_deletion: bool,
                      gpu_max_elems: int = 100_000_000) -> Optional[np.ndarray]:
    """
    Versi CuPy (GPU). Menghasilkan NumPy array (dipindah balik).
    Mengembalikan None jika tidak ada GPU atau problem memori.
    """
    if not _HAS_CUPY:
        return None

    n, L = enc.shape
    # cek estimasi elemen boolean (n*n*L). Jika > gpu_max_elems -> skip
    if n * n * L > gpu_max_elems:
        return None

    try:
        X = cp.asarray(enc)  # (n, L) int8 on GPU
        valid = (X[:, None, :] >= 0) & (X[None, :, :] >= 0)  # (n,n,L)
        if pairwise_deletion:
            denom = valid.sum(axis=2, dtype=cp.float64)
        else:
            glob_valid = (X >= 0).all(axis=0)
            valid = valid & glob_valid[None, None, :]
            denom = cp.full((n, n), glob_valid.sum(), dtype=cp.float64)
            if int(denom[0, 0].get()) == 0:
                return None

        mism = (X[:, None, :] != X[None, :, :]) & valid
        diff_counts = mism.sum(axis=2, dtype=cp.float64)
        denom = cp.where(denom == 0, cp.nan, denom)

        if method == 'raw':
            D = diff_counts
        elif method == 'p':
            D = diff_counts / denom
        elif method in ('jc69', 'jukes_cantor'):
            p = diff_counts / denom
            D = -0.75 * cp.log(1.0 - 4.0 * p / 3.0)
            D = cp.where((p >= 0.75) | ~cp.isfinite(D), cp.inf, D)
        elif method in ('k80', 'kimura'):
            absdiff = cp.abs(X[:, None, :] - X[None, :, :])
            transi = (absdiff == 2) & mism
            P = transi.sum(axis=2, dtype=cp.float64) / denom
            Q = (diff_counts - transi.sum(axis=2, dtype=cp.float64)) / denom
            D = -0.5 * cp.log((1 - 2*P - Q) * cp.sqrt(1 - 2*Q))
            D = cp.where(~cp.isfinite(D), cp.inf, D)
        else:
            return None

        cp.fill_diagonal(D, 0.0)
        return cp.asnumpy(D)
    except Exception:
        return None


# =========================
# Multi-process (memory-safe) backend
# =========================

def _pair_distance_worker(i: int, j: int, seq_list: List[str],
                          method: str, pairwise_deletion: bool) -> Tuple[int, int, float]:
    s1 = seq_list[i].upper()
    s2 = seq_list[j].upper()
    L = min(len(s1), len(s2))
    if L == 0:
        return i, j, np.nan

    # mask valid per posisi
    def _code(ch: str) -> int:
        o = ord(ch)
        if o < 256:
            return int(_TRANSLATE[o])
        return -1

    # cepat: buat view uint8, lalu translate (tanpa NumPy besar)
    a = np.frombuffer(s1.encode('ascii', 'ignore')[:L], dtype=np.uint8)
    b = np.frombuffer(s2.encode('ascii', 'ignore')[:L], dtype=np.uint8)
    A = _TRANSLATE[a]
    B = _TRANSLATE[b]
    valid = (A >= 0) & (B >= 0)
    denom = valid.sum()
    if denom == 0:
        return i, j, np.nan

    mism = (A != B) & valid
    if method == 'raw':
        return i, j, float(mism.sum())
    elif method == 'p':
        return i, j, float(mism.sum() / denom)
    elif method in ('jc69', 'jukes_cantor'):
        p = mism.sum() / denom
        if p >= 0.75:
            return i, j, float('inf')
        try:
            d = -0.75 * math.log(1 - 4*p/3)
        except Exception:
            d = float('inf')
        return i, j, d
    elif method in ('k80', 'kimura'):
        absdiff = np.abs(A - B)
        transi = (absdiff == 2) & mism
        P = transi.sum() / denom
        Q = (mism.sum() - transi.sum()) / denom
        try:
            d = -0.5 * math.log((1 - 2*P - Q) * math.sqrt(1 - 2*Q))
        except Exception:
            d = float('inf')
        return i, j, d
    else:
        raise ValueError(method)


def _dist_matrix_parallel(seqs: Dict[str, str],
                          method: str,
                          pairwise_deletion: bool,
                          n_jobs: Optional[int]) -> Tuple[np.ndarray, List[str]]:
    taxa = list(seqs.keys())
    n = len(taxa)
    D = np.zeros((n, n), dtype=float)
    seq_list = [seqs[t] for t in taxa]
    if n_jobs is None or n_jobs <= 0:
        n_jobs = os.cpu_count() or 1

    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        fut = ex.map(
            lambda args: _pair_distance_worker(args[0], args[1], seq_list, method, pairwise_deletion),
            pairs,
            chunksize=max(1, len(pairs)//(n_jobs*4))
        )
        for i, j, d in fut:
            D[i, j] = D[j, i] = d
    return D, taxa


# =========================
# Public API
# =========================

def compute_distance_matrix(sequences: Dict[str, str],
                            method: str = 'p',
                            pairwise_deletion: bool = True,
                            backend: str = 'auto',
                            n_jobs: Optional[int] = None,
                            gpu_max_elems: int = 100_000_000
                            ) -> Tuple[np.ndarray, List[str]]:
    """
    Hitung matriks jarak.

    Parameters
    ----------
    sequences : dict
        {taxon: sequence} (MSA). Gaps/ambiguous: - ? N (dll) -> diabaikan (kode -1).
    method : str
        'raw' | 'p' | 'jc69'/'jukes_cantor' | 'k80'/'kimura'.
    pairwise_deletion : bool
        True = abaikan posisi yang invalid spesifik per-pasangan (umum di phylo).
    backend : str
        'auto' | 'gpu' | 'cpu' | 'parallel'
    n_jobs : int or None
        Jumlah proses saat backend='parallel'. None/0 = semua CPU.
    gpu_max_elems : int
        Batas elemen boolean n*n*L untuk backend GPU (agar tak OOM).

    Returns
    -------
    (D, taxa) : (np.ndarray, list)
        D matriks jarak (n,n), taxa urutan label.
    """
    method = method.lower()
    if backend not in ('auto', 'gpu', 'cpu', 'parallel'):
        raise ValueError("backend must be 'auto' | 'gpu' | 'cpu' | 'parallel'")

    # BACKEND: parallel multi-proses (hemat memori, aman untuk n besar)
    if backend == 'parallel':
        return _dist_matrix_parallel(sequences, method, pairwise_deletion, n_jobs)

    # Encode ke NumPy dulu
    enc, taxa = _encode_numpy(sequences)

    # BACKEND: GPU
    if backend in ('auto', 'gpu'):
        D_gpu = _dist_matrix_cupy(enc, method, pairwise_deletion, gpu_max_elems=gpu_max_elems)
        if D_gpu is not None:
            return D_gpu, taxa
        if backend == 'gpu':
            # pengguna memaksa GPU tapi gagal -> fallback jelas
            # (tetap kembali ke CPU agar tidak crash)
            pass

    # BACKEND: CPU vectorized
    D = _dist_matrix_numpy(enc, method, pairwise_deletion)
    return D, taxa


# --------- convenience wrappers ---------

async def compute_distance_matrix_async(sequences: Dict[str, str],
                                        method: str = 'p',
                                        pairwise_deletion: bool = True,
                                        backend: str = 'auto',
                                        n_jobs: Optional[int] = None,
                                        gpu_max_elems: int = 100_000_000) -> Tuple[np.ndarray, List[str]]:
    """
    Versi async: menjalankan compute_distance_matrix di thread pool.
    (Tidak mempercepat komputasi intrinsik; hanya non-blocking untuk event loop.)
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: compute_distance_matrix(
            sequences, method=method,
            pairwise_deletion=pairwise_deletion,
            backend=backend, n_jobs=n_jobs,
            gpu_max_elems=gpu_max_elems
        )
    )

"""
Multiple Sequence Alignment (MSA) - Progressive (ClustalW-style)
- Async & threaded pairwise distance stage
- Optional GPU via CuPy (fallback to NumPy)
- Profile–profile alignment with affine gaps
"""

from __future__ import annotations
import math
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal

import numpy as _np

# Try GPU (CuPy) if available
try:
    import cupy as _cp
    _HAS_CUPY = True
except Exception:
    _cp = None
    _HAS_CUPY = False

# Local deps in project
from .distances import compute_distance_matrix
from .tree_builder import NeighborJoining, UPGMA
try:
    # optional: reuse your pairwise aligner if you want to score/seed
    from TK4BioInfo.seq_alignment import PairwiseAligner  # noqa
    _HAS_PAIRWISE = True
except Exception:
    _HAS_PAIRWISE = False


# -------------------------
# Utilities / alphabet
# -------------------------
_DNA = ("A", "C", "G", "T")
_DNA_GAP = ("A", "C", "G", "T", "-")

# very light DNA score (match=1, mismatch=-1)
def _dna_score(a: str, b: str) -> int:
    if a == "-" or b == "-":
        return 0  # gap scores handled by affine penalties, keep 0 here
    return 1 if a == b else -1

# minimal BLOSUM62 subset loader (reuse from your pairwise if desired)
# Here we import lazily from your module if present to avoid duplication.
_BLOSUM62 = None
if _HAS_PAIRWISE:
    try:
        from TK4BioInfo.seq_alignment import BLOSUM62 as _BLOSUM62  # noqa
    except Exception:
        _BLOSUM62 = None

def _protein_score(a: str, b: str) -> int:
    # fall back to simple if matrix not present
    if _BLOSUM62 is None:
        if a == "-" or b == "-":
            return 0
        return 1 if a == b else -1
    if a == "-" or b == "-":
        return 0
    return _BLOSUM62.get((a, b), _BLOSUM62.get((b, a), -4))


def _is_protein(seqs: Dict[str, str]) -> bool:
    letters = set("".join(seqs.values()).upper())
    # heuristic: if letters beyond DNA (BJZX, etc.), treat as protein
    return any(ch not in set("ACGTN-") for ch in letters)


# -------------------------
# Data structures
# -------------------------
@dataclass
class MSAResult:
    aligned: Dict[str, str]          # name -> aligned sequence
    order: List[str]                 # names order
    guide_newick: Optional[str]      # string tree (Newick) if wanted
    mode: Literal["global", "local"]


@dataclass
class _Profile:
    names: List[str]
    seqs: List[str]  # same length
    is_protein: bool

    def length(self) -> int:
        return len(self.seqs[0]) if self.seqs else 0

    def nseq(self) -> int:
        return len(self.seqs)

    def to_column_freqs(self):
        """
        Return (L x K) float32 matrix of column frequencies.
        K=5 (A,C,G,T,'-') for DNA, or 21 (20 aa + '-') for protein (simplified).
        """
        if self.is_protein:
            # Build a compact alphabet for protein (20 aa + gap)
            aa = "ARNDCQEGHILKMFPSTWYV"
            alph = list(aa) + ["-"]
        else:
            alph = list(_DNA) + ["-"]

        K = len(alph)
        alpha_index = {c: i for i, c in enumerate(alph)}
        L = self.length()
        freqs = _np.zeros((L, K), dtype=_np.float32)

        for col in range(L):
            # count letters in column
            cnt = _np.zeros(K, dtype=_np.float32)
            for s in self.seqs:
                ch = s[col] if col < len(s) else "-"
                ch = ch.upper()
                if ch not in alpha_index:
                    ch = "-"  # unknown -> gap
                cnt[alpha_index[ch]] += 1.0
            # normalize
            total = cnt.sum()
            if total > 0:
                freqs[col] = cnt / total
        return freqs, alph


# -------------------------
# Distance & guide tree
# -------------------------
def _pairwise_distance_matrix(
    seqs: Dict[str, str],
    method: str = "k80",
    pairwise_deletion: bool = True,
    n_jobs: int = 0
):
    # just call your distances module (already parallelized)
    return compute_distance_matrix(
        seqs, method=method
    ) if "pairwise_deletion" not in compute_distance_matrix.__code__.co_varnames else \
        compute_distance_matrix(seqs, method=method, pairwise_deletion=pairwise_deletion)


def _build_guide_tree(
    D, names: List[str], method: Literal["nj", "upgma"] = "nj"
):
    if method == "nj":
        tr = NeighborJoining(D, names).build_tree()
        newick = NeighborJoining.to_newick(tr)
        return tr, newick
    tr = UPGMA(D, names).build_tree()
    newick = UPGMA.to_newick(tr)
    return tr, newick


def _postorder_pairs(root) -> List[Tuple[List[str], List[str]]]:
    """
    Traverse guide tree; return list of join operations (clusterA_names, clusterB_names)
    in the order they should be merged (postorder).
    """
    ops = []

    def visit(node):
        if node.is_leaf():
            return [node.name]
        left = visit(node.children[0])
        right = visit(node.children[1])
        ops.append((left, right))
        return left + right

    visit(root)
    return ops


# -------------------------
# Profile–profile DP (affine)
# -------------------------
def _get_xp(backend: Literal["cpu", "gpu"]):
    if backend == "gpu" and _HAS_CUPY:
        return _cp
    return _np


def _column_score_matrix(
    freqsA: _np.ndarray, alphA: List[str],
    freqsB: _np.ndarray, alphB: List[str],
    is_protein: bool,
    xp
):
    """
    Precompute S[i,j] = score(column_i_of_A vs column_j_of_B)
    score = sum_pq fA[p]*fB[q]*score(p,q)
    Returns (L1 x L2) matrix on xp backend
    """
    L1, KA = freqsA.shape
    L2, KB = freqsB.shape
    S = xp.zeros((L1, L2), dtype=xp.float32)

    # build per-symbol score matrix
    if is_protein:
        # protein alphabet: alph has 21 (20 + gap)
        def s(a, b): return _protein_score(a, b)
    else:
        def s(a, b): return _dna_score(a, b)

    score_mat = _np.zeros((KA, KB), dtype=_np.float32)
    for i, a in enumerate(alphA):
        for j, b in enumerate(alphB):
            score_mat[i, j] = s(a, b)

    # move to xp
    score_mat = xp.asarray(score_mat)

    # compute S via tensor dot: S[i,j] = fA[i,:] * score_mat * fB[j,:]^T
    fA = xp.asarray(freqsA)   # (L1, KA)
    fB = xp.asarray(freqsB)   # (L2, KB)
    # (L1, KA) @ (KA, KB) -> (L1, KB), then @ (L2, KB)^T -> (L1, L2)
    S = fA @ score_mat @ fB.T
    return S


def _profile_profile_align(
    profA: _Profile,
    profB: _Profile,
    gap_open: float = 10.0,
    gap_ext: float = 0.5,
    mode: Literal["global", "local"] = "global",
    backend: Literal["cpu", "gpu"] = "cpu"
) -> _Profile:
    """
    DP with affine gaps on profile columns (Gotoh). Returns merged profile.
    """
    xp = _get_xp(backend)
    freqsA, alphA = profA.to_column_freqs()
    freqsB, alphB = profB.to_column_freqs()
    L1, L2 = freqsA.shape[0], freqsB.shape[0]
    is_protein = profA.is_protein or profB.is_protein

    # Precompute column-column scores
    S = _column_score_matrix(freqsA, alphA, freqsB, alphB, is_protein, xp)

    # Affine DP: M=match, X=gap in B (extend A), Y=gap in A (extend B)
    NEG = -1e9
    M = xp.full((L1 + 1, L2 + 1), NEG, dtype=xp.float32)
    X = xp.full((L1 + 1, L2 + 1), NEG, dtype=xp.float32)
    Y = xp.full((L1 + 1, L2 + 1), NEG, dtype=xp.float32)

    M[0, 0] = 0.0
    if mode == "global":
        for i in range(1, L1 + 1):
            X[i, 0] = -(gap_open + (i - 1) * gap_ext)
        for j in range(1, L2 + 1):
            Y[0, j] = -(gap_open + (j - 1) * gap_ext)
    else:
        M[0, :] = 0.0
        M[:, 0] = 0.0
        X[0, :] = 0.0
        X[:, 0] = 0.0
        Y[0, :] = 0.0
        Y[:, 0] = 0.0

    # Traceback pointers: 0=M,1=X,2=Y ; store ints to host at the end
    TB = _np.zeros((L1 + 1, L2 + 1), dtype=_np.uint8)
    SRC = _np.zeros((L1 + 1, L2 + 1), dtype=_np.uint8)  # 0=diag,1=up,2=left for chosen state

    # Fill
    best_score = 0.0
    best_pos = (L1, L2) if mode == "global" else (0, 0)

    for i in range(1, L1 + 1):
        # bring row score to host if on GPU for scalar access (small overhead)
        srow = S[i - 1]  # (L2,)
        for j in range(1, L2 + 1):
            # scores
            m_from = M[i - 1, j - 1]
            x_from = X[i - 1, j - 1]
            y_from = Y[i - 1, j - 1]
            score_match = (m_from if m_from > x_from else x_from)
            score_match = (score_match if score_match > y_from else y_from)
            score_match = score_match + (srow[j - 1])

            # X: gap in B (extend in A)
            x1 = M[i - 1, j] - (gap_open + gap_ext)
            x2 = X[i - 1, j] - gap_ext
            x_val = x1 if x1 > x2 else x2

            # Y: gap in A (extend in B)
            y1 = M[i, j - 1] - (gap_open + gap_ext)
            y2 = Y[i, j - 1] - gap_ext
            y_val = y1 if y1 > y2 else y2

            # choose state
            # M
            M[i, j] = score_match
            # X
            X[i, j] = x_val
            # Y
            Y[i, j] = y_val

            # best of states
            mm = float(M[i, j])
            xx = float(X[i, j])
            yy = float(Y[i, j])

            if mode == "local":
                # clamp at 0
                if mm < 0: mm = 0.0
                if xx < 0: xx = 0.0
                if yy < 0: yy = 0.0
                M[i, j] = mm; X[i, j] = xx; Y[i, j] = yy

            if mm >= xx and mm >= yy:
                TB[i, j] = 0  # M
                SRC[i, j] = 0  # diag
                cur = mm
            elif xx >= yy:
                TB[i, j] = 1  # X
                SRC[i, j] = 1  # up
                cur = xx
            else:
                TB[i, j] = 2  # Y
                SRC[i, j] = 2  # left
                cur = yy

            if mode == "local":
                if cur > best_score:
                    best_score = cur
                    best_pos = (i, j)

    # Traceback
    if mode == "global":
        i, j = L1, L2
    else:
        i, j = best_pos

    alnA_cols: List[int] = []
    alnB_cols: List[int] = []
    while i > 0 or j > 0:
        if mode == "local":
            # stop at 0
            v = max(float(M[i, j]), float(X[i, j]), float(Y[i, j]))
            if v <= 0: break

        state = TB[i, j]
        src = SRC[i, j]
        if state == 0:  # M, diag
            alnA_cols.append(i - 1)
            alnB_cols.append(j - 1)
            i -= 1; j -= 1
        elif state == 1:  # X, up
            alnA_cols.append(i - 1)
            alnB_cols.append(-1)  # gap
            i -= 1
        else:  # Y, left
            alnA_cols.append(-1)
            alnB_cols.append(j - 1)
            j -= 1

    alnA_cols.reverse()
    alnB_cols.reverse()

    # Build merged alignment strings
    outA = []
    outB = []
    for ca, cb in zip(alnA_cols, alnB_cols):
        if ca >= 0 and cb >= 0:
            outA.append("M")
            outB.append("M")
        elif ca >= 0:
            outA.append("A")  # take column from A, gap in B
            outB.append("G")
        else:
            outA.append("G")
            outB.append("B")  # take column from B, gap in A

    # realize sequences
    # Grab columns
    Acols = ["".join(s[col] for s in profA.seqs) if col >= 0 else None for col in alnA_cols]
    Bcols = ["".join(s[col] for s in profB.seqs) if col >= 0 else None for col in alnB_cols]

    merged_names = profA.names + profB.names
    merged = []

    for idx, nm in enumerate(profA.names):
        row = []
        for mark, acol in zip(outA, Acols):
            if mark == "M" or mark == "A":
                row.append(acol[idx])
            else:
                row.append("-")
        merged.append("".join(row))

    offset = 0
    for idx, nm in enumerate(profB.names):
        row = []
        for mark, bcol in zip(outB, Bcols):
            if mark == "M" or mark == "B":
                row.append(bcol[idx])
            else:
                row.append("-")
        merged.append("".join(row))

    return _Profile(
        names=profA.names + profB.names,
        seqs=merged,
        is_protein=is_protein
    )


# -------------------------
# Orchestrator (MSA)
# -------------------------
async def progressive_msa_async(
    sequences: Dict[str, str],
    mode: Literal["global", "local"] = "global",
    distance_method: Literal["k80", "jc69", "p", "raw"] = "k80",
    tree_method: Literal["nj", "upgma"] = "nj",
    gap_opening: float = 10.0,
    gap_extension: float = 0.5,
    backend: Literal["cpu", "gpu"] = "cpu",
    n_jobs: int = 0,
) -> MSAResult:
    """
    Build progressive MSA:
      1) distance matrix (parallel)
      2) guide tree (NJ/UPGMA)
      3) progressive profile–profile DP

    Returns aligned dict name->aligned_seq
    """
    names = list(sequences.keys())
    is_pro = _is_protein(sequences)

    loop = asyncio.get_running_loop()

    # 1) distance matrix (can be heavy; run in thread)
    def _dist_job():
        return _pairwise_distance_matrix(
            sequences, method=distance_method, pairwise_deletion=True, n_jobs=n_jobs
        )

    D = await loop.run_in_executor(None, _dist_job)

    # 2) guide tree
    def _tree_job():
        return _build_guide_tree(D, names, method=tree_method)

    tree, newick = await loop.run_in_executor(None, _tree_job)

    # 3) postorder merge operations
    ops = _postorder_pairs(tree)

    # init profiles (each sequence is its own profile)
    prof_map: Dict[str, _Profile] = {
        nm: _Profile([nm], [sequences[nm].upper()], is_pro) for nm in names
    }

    # perform progressive merges
    for left_names, right_names in ops:
        # build A profile
        A = _Profile(
            names=[],
            seqs=[],
            is_protein=is_pro
        )
        for nm in left_names:
            pr = prof_map[nm]
            if not A.seqs:
                A = pr
            else:
                # concatenate profiles with same columns? No, they should already be merged
                # If left_names has >1 entries not merged yet (rare here), we stack as independent rows if they share same length
                if pr.length() != A.length():
                    # pad to the max length
                    L = max(pr.length(), A.length())
                    def padrows(rows, L):
                        return [r + "-"*(L-len(r)) for r in rows]
                    A = _Profile(A.names, padrows(A.seqs, L), is_pro)
                    pr = _Profile(pr.names, padrows(pr.seqs, L), is_pro)
                A = _Profile(A.names + pr.names, A.seqs + pr.seqs, is_pro)

        B = _Profile(
            names=[],
            seqs=[],
            is_protein=is_pro
        )
        for nm in right_names:
            pr = prof_map[nm]
            if not B.seqs:
                B = pr
            else:
                if pr.length() != B.length():
                    L = max(pr.length(), B.length())
                    def padrows(rows, L):
                        return [r + "-"*(L-len(r)) for r in rows]
                    B = _Profile(B.names, padrows(B.seqs, L), is_pro)
                    pr = _Profile(pr.names, padrows(pr.seqs, L), is_pro)
                B = _Profile(B.names + pr.names, B.seqs + pr.seqs, is_pro)

        merged = _profile_profile_align(
            A, B, gap_open=gap_opening, gap_ext=gap_extension, mode=mode, backend=backend
        )
        # put merged back into map under all their names
        for nm in merged.names:
            prof_map[nm] = merged

    # all sequences now share the same merged profile object
    final_prof = next(iter(prof_map.values()))
    aligned = {nm: seq for nm, seq in zip(final_prof.names, final_prof.seqs)}

    return MSAResult(
        aligned=aligned,
        order=final_prof.names,
        guide_newick=newick,
        mode=mode
    )


def progressive_msa(sequences: Dict[str, str], **kwargs):
    """
    Smart sync wrapper:
    - Kalau tidak ada event loop: pakai asyncio.run()
    - Kalau sedang di Jupyter (loop sudah running): 
      coba nest_asyncio; jika gagal, jalankan di thread terpisah.
    """
    try:
        loop = asyncio.get_running_loop()
        # Sudah di dalam loop (mis. Jupyter)
        try:
            import nest_asyncio  # pip install nest_asyncio
            nest_asyncio.apply(loop)
            return loop.run_until_complete(progressive_msa_async(sequences, **kwargs))
        except Exception:
            # fallback: jalanin coroutine di thread terpisah
            from threading import Thread
            result_holder = {}
            def runner():
                result_holder["res"] = asyncio.run(progressive_msa_async(sequences, **kwargs))
            t = Thread(target=runner, daemon=True)
            t.start(); t.join()
            return result_holder["res"]
    except RuntimeError:
        # Tidak ada loop aktif (mode script biasa)
        return asyncio.run(progressive_msa_async(sequences, **kwargs))



# -------------------------
# Convenience I/O
# -------------------------
def write_fasta(aligned: Dict[str, str], path: str):
    with open(path, "w") as f:
        for k, v in aligned.items():
            f.write(f">{k}\n")
            # wrap 80 cols
            for i in range(0, len(v), 80):
                f.write(v[i:i+80] + "\n")

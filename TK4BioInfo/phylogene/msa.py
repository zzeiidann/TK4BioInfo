"""
Multiple Sequence Alignment (MSA) - Progressive (ClustalW-style)
WITH PROPER VERBOSE TRACKING!
"""

from __future__ import annotations
import asyncio
import gc
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal

import numpy as _np

try:
    import cupy as _cp
    _HAS_CUPY = True
except Exception:
    _cp = None
    _HAS_CUPY = False

from .distances import compute_distance_matrix
from .tree_builder import NeighborJoining, UPGMA

try:
    from TK4BioInfo.seq_alignment import PairwiseAligner
    from TK4BioInfo.seq_alignment import BLOSUM62 as _BLOSUM62
    _HAS_PAIRWISE = True
except Exception:
    _HAS_PAIRWISE = False
    _BLOSUM62 = None

_DNA = ("A", "C", "G", "T")

def _dna_score(a: str, b: str) -> int:
    if a == "-" or b == "-":
        return 0
    return 1 if a == b else -1

def _protein_score(a: str, b: str) -> int:
    if _BLOSUM62 is None:
        if a == "-" or b == "-":
            return 0
        return 1 if a == b else -1
    if a == "-" or b == "-":
        return 0
    return _BLOSUM62.get((a, b), _BLOSUM62.get((b, a), -4))

def _is_protein(seqs: Dict[str, str]) -> bool:
    letters = set("".join(seqs.values()).upper())
    return any(ch not in set("ACGTN-") for ch in letters)

@dataclass
class MSAResult:
    aligned: Dict[str, str]
    order: List[str]
    guide_newick: Optional[str]
    mode: Literal["global", "local"]

@dataclass
class _Profile:
    names: List[str]
    seqs: List[str]
    is_protein: bool

    def length(self) -> int:
        return len(self.seqs[0]) if self.seqs else 0

    def nseq(self) -> int:
        return len(self.seqs)

    def to_column_freqs(self):
        if self.is_protein:
            aa = "ARNDCQEGHILKMFPSTWYV"
            alph = list(aa) + ["-"]
        else:
            alph = list(_DNA) + ["-"]

        K = len(alph)
        alpha_index = {c: i for i, c in enumerate(alph)}
        L = self.length()
        freqs = _np.zeros((L, K), dtype=_np.float32)

        for col in range(L):
            cnt = _np.zeros(K, dtype=_np.float32)
            for s in self.seqs:
                ch = s[col].upper() if col < len(s) else "-"
                if ch not in alpha_index:
                    ch = "-"
                cnt[alpha_index[ch]] += 1.0
            total = cnt.sum()
            if total > 0:
                freqs[col] = cnt / total
        return freqs, alph

def _pairwise_distance_matrix(seqs, method, pairwise_deletion, n_jobs, verbose):
    if verbose:
        print(f"[MSA] Computing distances ({method})...", end=" ", flush=True)
    
    try:
        result = compute_distance_matrix(seqs, method=method)
    except TypeError:
        result = compute_distance_matrix(seqs, method=method, pairwise_deletion=pairwise_deletion)
    
    n = len(seqs)
    if isinstance(result, tuple):
        D = result[0]
    elif isinstance(result, dict):
        D = result.get('matrix', result.get('distance_matrix', result))
    else:
        D = result
    
    if hasattr(D, 'values'):
        D = D.values
    
    D = _np.asarray(D, dtype=_np.float64)
    
    if D.ndim != 2 or D.shape[0] != n or D.shape[1] != n:
        raise ValueError(f"Distance matrix shape {D.shape} invalid (expected {n}x{n})")
    
    if _np.any(_np.isnan(D)):
        if verbose:
            print(f"[WARN] NaN found", end=" ")
        D = _np.nan_to_num(D, nan=1.0, posinf=1.0, neginf=0.0)
    
    if _np.any(_np.isinf(D)):
        if verbose:
            print(f"[WARN] inf found", end=" ")
        D = _np.nan_to_num(D, nan=1.0, posinf=1.0, neginf=0.0)
    
    _np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    D = _np.maximum(D, 1e-10)
    
    if verbose:
        print(f"done ({n}x{n})")
    
    return D

def _build_guide_tree(D, names, method, verbose):
    if verbose:
        print(f"[MSA] Building tree ({method.upper()})...", end=" ", flush=True)
    
    D = _np.asarray(D, dtype=_np.float64)
    
    if method == "nj":
        tr = NeighborJoining(D, names).build_tree()
        newick = NeighborJoining.to_newick(tr)
    else:
        tr = UPGMA(D, names).build_tree()
        newick = UPGMA.to_newick(tr)
    
    if verbose:
        print("done")
    
    return tr, newick

def _postorder_pairs(root):
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

def _get_xp(backend):
    if backend == "gpu" and _HAS_CUPY:
        return _cp
    return _np

def _column_score_matrix(freqsA, alphA, freqsB, alphB, is_protein, xp):
    L1, KA = freqsA.shape
    L2, KB = freqsB.shape
    
    if is_protein:
        def s(a, b): return _protein_score(a, b)
    else:
        def s(a, b): return _dna_score(a, b)
    
    score_mat = _np.zeros((KA, KB), dtype=_np.float32)
    for i, a in enumerate(alphA):
        for j, b in enumerate(alphB):
            score_mat[i, j] = s(a, b)
    
    score_mat = xp.asarray(score_mat)
    fA = xp.asarray(freqsA)
    fB = xp.asarray(freqsB)
    S = fA @ score_mat @ fB.T
    return S

def _build_merged_profile(profA, profB, alnA_cols, alnB_cols, is_protein):
    merged = []
    
    for idx in range(profA.nseq()):
        row = []
        for ca, cb in zip(alnA_cols, alnB_cols):
            if ca >= 0:
                row.append(profA.seqs[idx][ca])
            else:
                row.append("-")
        merged.append("".join(row))
    
    for idx in range(profB.nseq()):
        row = []
        for ca, cb in zip(alnA_cols, alnB_cols):
            if cb >= 0:
                row.append(profB.seqs[idx][cb])
            else:
                row.append("-")
        merged.append("".join(row))
    
    return _Profile(
        names=profA.names + profB.names,
        seqs=merged,
        is_protein=is_protein
    )

def _profile_profile_align_banded(profA, profB, gap_open, gap_ext, mode, backend, band_width, verbose):
    """BANDED ALIGNMENT WITH VERBOSE"""
    xp = _get_xp(backend)
    freqsA, alphA = profA.to_column_freqs()
    freqsB, alphB = profB.to_column_freqs()
    L1, L2 = freqsA.shape[0], freqsB.shape[0]
    is_protein = profA.is_protein or profB.is_protein
    
    if verbose:
        print(f"      Column scores...", end=" ", flush=True)
    
    S = _column_score_matrix(freqsA, alphA, freqsB, alphB, is_protein, xp)
    
    if verbose:
        print(f"DP matrix ({L1}x{L2}, band={band_width})...", flush=True)
    
    NEG = -1e9
    band = band_width
    M, X, Y, TB = {}, {}, {}, {}
    
    M[(0, 0)] = 0.0
    X[(0, 0)] = NEG
    Y[(0, 0)] = NEG
    
    for i in range(1, min(L1 + 1, band + 1)):
        M[(i, 0)] = NEG
        X[(i, 0)] = -(gap_open + (i - 1) * gap_ext) if mode == "global" else 0.0
        Y[(i, 0)] = NEG
        TB[(i, 0)] = 1
    
    for j in range(1, min(L2 + 1, band + 1)):
        M[(0, j)] = NEG
        X[(0, j)] = NEG
        Y[(0, j)] = -(gap_open + (j - 1) * gap_ext) if mode == "global" else 0.0
        TB[(0, j)] = 2
    
    for i in range(1, L1 + 1):
        if verbose and L1 > 10000 and i % 5000 == 0:
            pct = (i / L1) * 100
            print(f"      [{pct:5.1f}%] row {i}/{L1}", flush=True)
        
        j_min = max(1, i - band)
        j_max = min(L2 + 1, i + band + 1)
        
        for j in range(j_min, j_max):
            m_prev = M.get((i-1, j-1), NEG)
            x_prev = X.get((i-1, j-1), NEG)
            y_prev = Y.get((i-1, j-1), NEG)
            score_match = max(m_prev, x_prev, y_prev) + float(S[i-1, j-1])
            
            x_val = max(M.get((i-1, j), NEG) - (gap_open + gap_ext),
                       X.get((i-1, j), NEG) - gap_ext)
            y_val = max(M.get((i, j-1), NEG) - (gap_open + gap_ext),
                       Y.get((i, j-1), NEG) - gap_ext)
            
            M[(i, j)] = score_match
            X[(i, j)] = x_val
            Y[(i, j)] = y_val
            
            mm = float(M[(i, j)])
            xx = float(X[(i, j)])
            yy = float(Y[(i, j)])
            
            if mode == "local":
                if mm < 0: mm = 0.0
                if xx < 0: xx = 0.0
                if yy < 0: yy = 0.0
                M[(i, j)] = mm
                X[(i, j)] = xx
                Y[(i, j)] = yy
            
            if mm >= xx and mm >= yy:
                TB[(i, j)] = 0
            elif xx >= yy:
                TB[(i, j)] = 1
            else:
                TB[(i, j)] = 2
    
    if verbose:
        print(f"      Traceback...", end=" ", flush=True)
    
    i, j = L1, L2
    alnA_cols, alnB_cols = [], []
    
    while i > 0 or j > 0:
        if (i, j) not in TB:
            if i > 0 and j > 0:
                alnA_cols.append(i - 1)
                alnB_cols.append(j - 1)
                i -= 1
                j -= 1
            elif i > 0:
                alnA_cols.append(i - 1)
                alnB_cols.append(-1)
                i -= 1
            else:
                alnA_cols.append(-1)
                alnB_cols.append(j - 1)
                j -= 1
            continue
        
        dir_code = TB[(i, j)]
        if dir_code == 0:
            alnA_cols.append(i - 1)
            alnB_cols.append(j - 1)
            i -= 1
            j -= 1
        elif dir_code == 1:
            alnA_cols.append(i - 1)
            alnB_cols.append(-1)
            i -= 1
        else:
            alnA_cols.append(-1)
            alnB_cols.append(j - 1)
            j -= 1
    
    alnA_cols.reverse()
    alnB_cols.reverse()
    
    if verbose:
        print("done", flush=True)
    
    return _build_merged_profile(profA, profB, alnA_cols, alnB_cols, is_protein)

def _profile_profile_align_full(profA, profB, gap_open, gap_ext, mode, backend, verbose):
    """FULL DP FOR SMALL ALIGNMENTS"""
    xp = _get_xp(backend)
    freqsA, alphA = profA.to_column_freqs()
    freqsB, alphB = profB.to_column_freqs()
    L1, L2 = freqsA.shape[0], freqsB.shape[0]
    is_protein = profA.is_protein or profB.is_protein
    
    S = _column_score_matrix(freqsA, alphA, freqsB, alphB, is_protein, xp)
    
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
    
    TB = _np.zeros((L1 + 1, L2 + 1), dtype=_np.int16)
    
    for i in range(1, L1 + 1):
        srow = S[i - 1]
        for j in range(1, L2 + 1):
            m_from = max(M[i - 1, j - 1], X[i - 1, j - 1], Y[i - 1, j - 1])
            score_match = m_from + float(srow[j - 1])
            
            x_val = max(M[i - 1, j] - (gap_open + gap_ext), X[i - 1, j] - gap_ext)
            y_val = max(M[i, j - 1] - (gap_open + gap_ext), Y[i, j - 1] - gap_ext)
            
            M[i, j] = score_match
            X[i, j] = x_val
            Y[i, j] = y_val
            
            mm = float(M[i, j])
            xx = float(X[i, j])
            yy = float(Y[i, j])
            
            if mode == "local":
                if mm < 0: mm = 0.0
                if xx < 0: xx = 0.0
                if yy < 0: yy = 0.0
                M[i, j] = mm
                X[i, j] = xx
                Y[i, j] = yy
            
            if mm >= xx and mm >= yy:
                TB[i, j] = 0
            elif xx >= yy:
                TB[i, j] = 1
            else:
                TB[i, j] = 2
    
    del M, X, Y, S
    gc.collect()
    
    i, j = L1, L2
    alnA_cols, alnB_cols = [], []
    
    while i > 0 or j > 0:
        dir_code = TB[i, j]
        if dir_code == 0:
            alnA_cols.append(i - 1)
            alnB_cols.append(j - 1)
            i -= 1
            j -= 1
        elif dir_code == 1:
            alnA_cols.append(i - 1)
            alnB_cols.append(-1)
            i -= 1
        else:
            alnA_cols.append(-1)
            alnB_cols.append(j - 1)
            j -= 1
    
    alnA_cols.reverse()
    alnB_cols.reverse()
    
    return _build_merged_profile(profA, profB, alnA_cols, alnB_cols, is_protein)

def _profile_profile_align(profA, profB, gap_open, gap_ext, mode, backend, band_width, verbose):
    """DISPATCHER"""
    L1, L2 = profA.length(), profB.length()
    matrix_size = (L1 + 1) * (L2 + 1) * 4
    matrix_size_gb = matrix_size / 1e9
    
    if matrix_size > 2e9:
        if verbose:
            print(f"\n[MSA] Large: {L1}x{L2} (~{matrix_size_gb:.1f}GB)")
        
        if band_width is None:
            band_width = max(1000, int(0.05 * max(L1, L2)))
        
        if verbose:
            print(f"[MSA] Banded (band={band_width})...")
        
        return _profile_profile_align_banded(
            profA, profB, gap_open, gap_ext, mode, backend, band_width, verbose
        )
    
    return _profile_profile_align_full(
        profA, profB, gap_open, gap_ext, mode, backend, verbose
    )

async def progressive_msa_async(
    sequences,
    mode="global",
    distance_method="k80",
    tree_method="nj",
    gap_opening=10.0,
    gap_extension=0.5,
    backend="cpu",
    n_jobs=0,
    verbose=False
):
    import time
    start_time = time.time()
    
    names = list(sequences.keys())
    is_pro = _is_protein(sequences)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[MSA] Progressive MSA")
        print(f"{'='*60}")
        print(f"Sequences: {len(names)}")
        print(f"Mode: {mode} | Distance: {distance_method} | Tree: {tree_method}")
        print(f"Gap: open={gap_opening}, ext={gap_extension}")
        print(f"{'='*60}\n")
    
    loop = asyncio.get_running_loop()
    
    def _dist_job():
        return _pairwise_distance_matrix(sequences, distance_method, True, n_jobs, verbose)
    
    D = await loop.run_in_executor(None, _dist_job)
    
    def _tree_job():
        return _build_guide_tree(D, names, tree_method, verbose)
    
    tree, newick = await loop.run_in_executor(None, _tree_job)
    
    ops = _postorder_pairs(tree)
    
    if verbose:
        print(f"[MSA] Progressive: {len(ops)} steps\n")
    
    prof_map = {nm: _Profile([nm], [sequences[nm].upper()], is_pro) for nm in names}
    
    for step_idx, (left_names, right_names) in enumerate(ops, 1):
        step_start = time.time()
        
        A = _Profile(names=[], seqs=[], is_protein=is_pro)
        for nm in left_names:
            pr = prof_map[nm]
            if not A.seqs:
                A = pr
            else:
                if pr.length() != A.length():
                    L = max(pr.length(), A.length())
                    A.seqs = [s + "-" * (L - len(s)) for s in A.seqs]
                    pr.seqs = [s + "-" * (L - len(s)) for s in pr.seqs]
                A = _Profile(A.names + pr.names, A.seqs + pr.seqs, is_pro)
        
        B = _Profile(names=[], seqs=[], is_protein=is_pro)
        for nm in right_names:
            pr = prof_map[nm]
            if not B.seqs:
                B = pr
            else:
                if pr.length() != B.length():
                    L = max(pr.length(), B.length())
                    B.seqs = [s + "-" * (L - len(s)) for s in B.seqs]
                    pr.seqs = [s + "-" * (L - len(s)) for s in pr.seqs]
                B = _Profile(B.names + pr.names, B.seqs + pr.seqs, is_pro)
        
        try:
            merged = _profile_profile_align(
                A, B, gap_opening, gap_extension, mode, backend, None, verbose
            )
        except MemoryError as e:
            print(f"\n[ERROR] {e}")
            raise
        
        step_elapsed = time.time() - step_start
        
        if verbose:
            progress_pct = (step_idx / len(ops)) * 100
            print(f"[{progress_pct:5.1f}%] Step {step_idx:3d}/{len(ops):3d}: "
                  f"{len(left_names):3d}+{len(right_names):3d} → "
                  f"len={merged.length():4d} ({step_elapsed:5.2f}s)")
        
        for nm in merged.names:
            prof_map[nm] = merged
        
        if step_idx % 10 == 0:
            gc.collect()
    
    final_prof = next(iter(prof_map.values()))
    aligned = {nm: seq for nm, seq in zip(final_prof.names, final_prof.seqs)}
    
    if verbose:
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"[MSA] COMPLETE!")
        print(f"Time: {total_time:.2f}s")
        print(f"Length: {final_prof.length()}")
        print(f"{'='*60}\n")
    
    return MSAResult(aligned, final_prof.names, newick, mode)

def progressive_msa(sequences, verbose=False, **kwargs):
    kwargs['verbose'] = verbose
    
    try:
        loop = asyncio.get_running_loop()
        try:
            import nest_asyncio
            nest_asyncio.apply(loop)
            return loop.run_until_complete(progressive_msa_async(sequences, **kwargs))
        except ImportError:
            if verbose:
                print("[DEBUG] nest_asyncio not installed, thread fallback...")
        except Exception as e:
            if verbose:
                print(f"[DEBUG] nest_asyncio failed ({e}), thread fallback...")
        
        from threading import Thread, Event
        result_holder = {"res": None, "error": None}
        done_event = Event()
        
        def runner():
            try:
                result_holder["res"] = asyncio.run(progressive_msa_async(sequences, **kwargs))
            except Exception as err:
                result_holder["error"] = err
            finally:
                done_event.set()
        
        t = Thread(target=runner, daemon=False)
        t.start()
        
        if not done_event.wait(timeout=7200):
            raise RuntimeError("Timeout after 2 hours")
        
        t.join()
        
        if result_holder["error"]:
            raise result_holder["error"]
        
        if result_holder["res"] is None:
            raise RuntimeError("Thread failed")
        
        return result_holder["res"]
        
    except RuntimeError:
        return asyncio.run(progressive_msa_async(sequences, **kwargs))

def write_fasta(aligned, path):
    with open(path, "w") as f:
        for k, v in aligned.items():
            f.write(f">{k}\n")
            for i in range(0, len(v), 80):
                f.write(v[i:i+80] + "\n")
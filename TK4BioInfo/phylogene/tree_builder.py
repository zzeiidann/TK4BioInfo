"""
Tree construction algorithms for phylogenetic analysis
"""
from __future__ import annotations
import numpy as np
import asyncio
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import math


# =========================
# Core tree data structure
# =========================
@dataclass
class TreeNode:
    """Represents a node in a phylogenetic tree"""
    name: Optional[str] = None
    branch_length: float = 0.0
    children: List['TreeNode'] = None
    parent: Optional['TreeNode'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_leaves(self) -> List['TreeNode']:
        if self.is_leaf():
            return [self]
        leaves: List['TreeNode'] = []
        for ch in self.children:
            leaves.extend(ch.get_leaves())
        return leaves


# =========================
# UPGMA (ultrametric)
# =========================
class UPGMA:
    """
    UPGMA (Unweighted Pair Group Method with Arithmetic Mean)
    Assumes molecular clock; produces ultrametric tree.
    """

    def __init__(self, distance_matrix: np.ndarray, taxa_names: List[str]):
        self.distance_matrix = np.asarray(distance_matrix, dtype=float).copy()
        self.taxa_names = list(taxa_names)
        self.n = len(self.taxa_names)

    def build_tree(self) -> TreeNode:
        # Inisialisasi: setiap takson jadi cluster sendiri
        clusters: Dict[int, TreeNode] = {i: TreeNode(name=nm) for i, nm in enumerate(self.taxa_names)}
        cluster_sizes: Dict[int, int] = {i: 1 for i in range(self.n)}
        # height: jarak dari root ke ujung cluster (ultrametric)
        height: Dict[int, float] = {i: 0.0 for i in range(self.n)}

        D = self.distance_matrix.copy()

        # Loop sampai sisa satu cluster (root)
        while len(clusters) > 1:
            ids = list(clusters.keys())

            # Cari pasangan dengan jarak minimum
            min_val = float('inf')
            min_i = min_j = -1
            for a_idx in range(len(ids)):
                for b_idx in range(a_idx + 1, len(ids)):
                    i, j = ids[a_idx], ids[b_idx]
                    if D[i, j] < min_val:
                        min_val = D[i, j]
                        min_i, min_j = i, j

            # Buat node gabungan; tinggi (height) cluster baru
            new_h = min_val / 2.0
            node = TreeNode()
            # panjang cabang anak = selisih height
            bl_i = max(0.0, new_h - height[min_i])
            bl_j = max(0.0, new_h - height[min_j])

            clusters[min_i].branch_length = bl_i
            clusters[min_j].branch_length = bl_j
            clusters[min_i].parent = node
            clusters[min_j].parent = node
            node.children = [clusters[min_i], clusters[min_j]]

            # Ukuran cluster baru
            new_size = cluster_sizes[min_i] + cluster_sizes[min_j]

            # Perbarui jarak ke cluster lain (REUSE indeks min_i sebagai cluster baru)
            for k in list(clusters.keys()):
                if k == min_i or k == min_j:
                    continue
                new_dist = (
                    cluster_sizes[min_i] * D[min_i, k] +
                    cluster_sizes[min_j] * D[min_j, k]
                ) / new_size
                D[min_i, k] = new_dist
                D[k,   min_i] = new_dist

            # Nonaktifkan baris/kolom min_j dan hapus dari aktif
            D[min_j, :] = np.inf
            D[:,   min_j] = np.inf

            # Update struktur cluster/size/height
            clusters[min_i] = node          # indeks min_i jadi cluster gabungan
            height[min_i] = new_h
            cluster_sizes[min_i] = new_size

            del clusters[min_j]
            del cluster_sizes[min_j]
            del height[min_j]

        # Tersisa satu node = root
        return next(iter(clusters.values()))

    # Opsional: utilitas lokal untuk debug cepat
    @staticmethod
    def to_newick(node: TreeNode) -> str:
        if node.is_leaf():
            return f"{node.name}:{node.branch_length:.6f}"
        kids = ",".join(UPGMA.to_newick(ch) for ch in node.children)
        return f"({kids}):{node.branch_length:.6f}"


# =========================
# Neighbor-Joining (unrooted)
# =========================
class NeighborJoining:
    """
    Neighbor-Joining (NJ)
    Tidak mengasumsikan molecular clock; menghasilkan tree tak berakar.
    """

    def __init__(self, distance_matrix: np.ndarray, taxa_names: List[str]):
        self.distance_matrix = np.asarray(distance_matrix, dtype=float).copy()
        self.taxa_names = list(taxa_names)
        self.n = len(self.taxa_names)

    def build_tree(self) -> TreeNode:
        clusters: Dict[int, TreeNode] = {i: TreeNode(name=nm) for i, nm in enumerate(self.taxa_names)}
        D = self.distance_matrix.copy()

        while len(clusters) > 2:
            ids = list(clusters.keys())
            m = len(ids)

            # Hitung jumlah baris aktif (sum jarak tiap cluster ke yg lain)
            row_sum = {idx: 0.0 for idx in ids}
            for a in ids:
                for b in ids:
                    if a != b:
                        row_sum[a] += D[a, b]

            # Q-matrix
            min_q = float('inf')
            pair_i = pair_j = -1
            for a_i in range(m):
                for b_i in range(a_i + 1, m):
                    i, j = ids[a_i], ids[b_i]
                    q = (m - 2) * D[i, j] - row_sum[i] - row_sum[j]
                    if q < min_q:
                        min_q = q
                        pair_i, pair_j = i, j

            i, j = pair_i, pair_j
            # Panjang cabang ke node gabungan
            dij = D[i, j]
            # rumus klasik NJ
            li = max(0.0, 0.5 * dij + (row_sum[i] - row_sum[j]) / (2 * (m - 2)))
            lj = max(0.0, dij - li)

            node = TreeNode()
            clusters[i].branch_length = li
            clusters[j].branch_length = lj
            clusters[i].parent = node
            clusters[j].parent = node
            node.children = [clusters[i], clusters[j]]

            # Update jarak ke cluster lain, REUSE indeks i sebagai cluster gabungan
            for k in list(clusters.keys()):
                if k == i or k == j:
                    continue
                Dik = D[i, k]
                Djk = D[j, k]
                new_dist = 0.5 * (Dik + Djk - dij)
                D[i, k] = new_dist
                D[k, i] = new_dist

            # Nonaktifkan j
            D[j, :] = np.inf
            D[:, j] = np.inf

            # Ganti cluster i dengan gabungan; hapus j
            clusters[i] = node
            del clusters[j]

        # Hubungkan dua cluster terakhir ke root
        ids = list(clusters.keys())
        if len(ids) == 2:
            a, b = ids
            root = TreeNode()
            la = max(0.0, D[a, b] / 2.0)
            lb = la
            clusters[a].branch_length = la
            clusters[b].branch_length = lb
            clusters[a].parent = root
            clusters[b].parent = root
            root.children = [clusters[a], clusters[b]]
            return root

        # fallback (harusnya tak terjadi)
        return next(iter(clusters.values()))

    # Opsional: utilitas lokal untuk debug cepat
    @staticmethod
    def to_newick(node: TreeNode) -> str:
        if node.is_leaf():
            return f"{node.name}:{node.branch_length:.6f}"
        kids = ",".join(NeighborJoining.to_newick(ch) for ch in node.children)
        return f"({kids})" if node.branch_length == 0 else f"({kids}):{node.branch_length:.6f}"


# =========================
# Maximum Parsimony (sederhana)
# =========================
class MaximumParsimony:
    """
    MP sederhana: skor Fitch + start tree dari NJ (jarak p-distance).
    """

    def __init__(self, sequences: Dict[str, str]):
        self.sequences = sequences
        self.taxa_names = list(sequences.keys())
        self.seq_length = len(next(iter(sequences.values())))

    def parsimony_score(self, tree: TreeNode, sequences: Dict[str, str]) -> int:
        total = 0
        for pos in range(self.seq_length):
            total += self._fitch_score(tree, sequences, pos)
        return total

    def _fitch_score(self, node: TreeNode, sequences: Dict[str, str], pos: int) -> int:
        if node.is_leaf():
            # set state di leaf
            setattr(node, f'_states_{pos}', {sequences[node.name][pos]})
            return 0

        score = 0
        child_sets = []
        for ch in node.children:
            score += self._fitch_score(ch, sequences, pos)
            child_sets.append(getattr(ch, f'_states_{pos}'))

        inter = set.intersection(*child_sets)
        if inter:
            setattr(node, f'_states_{pos}', inter)
        else:
            setattr(node, f'_states_{pos}', set.union(*child_sets))
            score += 1
        return score

    def build_tree(self) -> TreeNode:
        from .distances import compute_distance_matrix
        dist = compute_distance_matrix(self.sequences, method='p')
        nj = NeighborJoining(dist, self.taxa_names)
        return nj.build_tree()


# =========================
# Maximum Likelihood (JC69, sangat sederhana)
# =========================
class MaximumLikelihood:
    """
    ML sangat ringkas (JC69) untuk contoh.
    Memakai NJ (JC69) sebagai starting topology lalu optim cabang secara kasar.
    """

    def __init__(self, sequences: Dict[str, str], model: str = 'JC69'):
        self.sequences = sequences
        self.taxa_names = list(sequences.keys())
        self.seq_length = len(next(iter(sequences.values())))
        self.model = model

    def transition_probability(self, base1: str, base2: str, t: float) -> float:
        if self.model.upper() == 'JC69':
            if base1 == base2:
                return 0.25 + 0.75 * math.exp(-4.0 * t / 3.0)
            else:
                return 0.25 - 0.25 * math.exp(-4.0 * t / 3.0)
        # default uniform
        return 0.25

    def _site_likelihood(self, node: TreeNode, pos: int):
        bases = ('A', 'C', 'G', 'T')
        if node.is_leaf():
            b = self.sequences[node.name][pos]
            return {x: 1.0 if x == b else 0.0 for x in bases}

        # internal node: pruning
        like = {}
        for x in bases:
            prob = 0.25  # prior
            for ch in node.children:
                child_like = self._site_likelihood(ch, pos)
                s = 0.0
                for y in bases:
                    s += child_like[y] * self.transition_probability(x, y, ch.branch_length)
                prob *= s
            like[x] = prob

        # jika root, return total; jika bukan, return vector
        if node.parent is None:
            return sum(like.values())
        return like

    def log_likelihood(self, tree: TreeNode) -> float:
        ll = 0.0
        for pos in range(self.seq_length):
            site = self._site_likelihood(tree, pos)
            if site > 0:
                ll += math.log(site)
        return ll

    def build_tree(self) -> TreeNode:
        from .distances import compute_distance_matrix
        dist = compute_distance_matrix(self.sequences, method='jc69')
        tree = NeighborJoining(dist, self.taxa_names).build_tree()
        self._optimize_branch_lengths(tree)
        return tree

    def _optimize_branch_lengths(self, tree: TreeNode, iterations: int = 3):
        for _ in range(iterations):
            self._optimize_node(tree)

    def _optimize_node(self, node: TreeNode):
        if node.is_leaf():
            return
        for ch in node.children:
            self._optimize_node(ch)
            # grid search kecil
            best = ch.branch_length if ch.branch_length > 0 else 0.05
            best_ll = self.log_likelihood(self._root_of(node))
            for t in np.linspace(max(1e-3, best/4), best*2, 8):
                ch.branch_length = t
                ll = self.log_likelihood(self._root_of(node))
                if ll > best_ll:
                    best_ll, best = ll, t
            ch.branch_length = best

    def _root_of(self, node: TreeNode) -> TreeNode:
        while node.parent is not None:
            node = node.parent
        return node


# =========================
# Async builder helper
# =========================
async def build_tree_async(method: str, data: Union[np.ndarray, Dict],
                           taxa_names: Optional[List[str]] = None) -> TreeNode:
    loop = asyncio.get_event_loop()

    if method.lower() == 'upgma':
        builder = UPGMA(data, taxa_names)  # type: ignore[arg-type]
        return await loop.run_in_executor(None, builder.build_tree)
    elif method.lower() == 'nj':
        builder = NeighborJoining(data, taxa_names)  # type: ignore[arg-type]
        return await loop.run_in_executor(None, builder.build_tree)
    elif method.lower() == 'mp':
        builder = MaximumParsimony(data)  # type: ignore[arg-type]
        return await loop.run_in_executor(None, builder.build_tree)
    elif method.lower() == 'ml':
        builder = MaximumLikelihood(data)  # type: ignore[arg-type]
        return await loop.run_in_executor(None, builder.build_tree)
    else:
        raise ValueError(f"Unknown method: {method}")

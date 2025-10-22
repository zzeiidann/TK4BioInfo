"""
Utility functions for tree manipulation and analysis
"""
import numpy as np
import asyncio
from typing import List, Dict, Optional
from collections import defaultdict, Counter
from .tree_builder import TreeNode, NeighborJoining
from .distances import compute_distance_matrix


def bootstrap_tree(sequences: Dict[str, str], 
                   method: str = 'nj',
                   n_replicates: int = 100,
                   distance_method: str = 'jc69') -> List[TreeNode]:
    """
    Perform bootstrap resampling to assess tree confidence
    
    Args:
        sequences: Dictionary of aligned sequences
        method: Tree building method ('nj', 'upgma')
        n_replicates: Number of bootstrap replicates
        distance_method: Distance calculation method
    
    Returns:
        List of bootstrap trees
    
    Example:
        >>> from phylogene import bootstrap_tree
        >>> sequences = {'A': 'ACGT', 'B': 'ACTT', 'C': 'TCGT'}
        >>> trees = bootstrap_tree(sequences, n_replicates=100)
        >>> print(f"Generated {len(trees)} bootstrap trees")
    """
    from .tree_builder import UPGMA
    
    seq_length = len(list(sequences.values())[0])
    taxa_names = list(sequences.keys())
    bootstrap_trees = []
    
    for _ in range(n_replicates):
        # Resample positions with replacement
        positions = np.random.choice(seq_length, seq_length, replace=True)
        
        # Create bootstrap sequences
        boot_seqs = {}
        for taxon, seq in sequences.items():
            boot_seqs[taxon] = ''.join(seq[pos] for pos in positions)
        
        # Build tree from bootstrap sample
        dist_matrix = compute_distance_matrix(boot_seqs, method=distance_method)
        
        if method.lower() == 'nj':
            builder = NeighborJoining(dist_matrix, taxa_names)
        elif method.lower() == 'upgma':
            builder = UPGMA(dist_matrix, taxa_names)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        tree = builder.build_tree()
        bootstrap_trees.append(tree)
    
    return bootstrap_trees


async def bootstrap_tree_async(sequences: Dict[str, str],
                               method: str = 'nj',
                               n_replicates: int = 100,
                               distance_method: str = 'jc69') -> List[TreeNode]:
    """
    Asynchronously perform bootstrap resampling
    
    Faster than synchronous version for large datasets.
    
    Args:
        sequences: Dictionary of aligned sequences
        method: Tree building method
        n_replicates: Number of bootstrap replicates
        distance_method: Distance calculation method
    
    Returns:
        List of bootstrap trees
    
    Example:
        >>> import asyncio
        >>> trees = asyncio.run(bootstrap_tree_async(sequences, n_replicates=1000))
    """
    from .tree_builder import UPGMA, build_tree_async
    
    seq_length = len(list(sequences.values())[0])
    taxa_names = list(sequences.keys())
    
    async def build_bootstrap_tree(replicate_id: int):
        """Build single bootstrap tree"""
        # Resample positions
        positions = np.random.choice(seq_length, seq_length, replace=True)
        
        # Create bootstrap sequences
        boot_seqs = {}
        for taxon, seq in sequences.items():
            boot_seqs[taxon] = ''.join(seq[pos] for pos in positions)
        
        # Build tree
        dist_matrix = compute_distance_matrix(boot_seqs, method=distance_method)
        tree = await build_tree_async(method, dist_matrix, taxa_names)
        return tree
    
    # Execute all replicates concurrently
    tasks = [build_bootstrap_tree(i) for i in range(n_replicates)]
    bootstrap_trees = await asyncio.gather(*tasks)
    
    return bootstrap_trees


def consensus_tree(trees: List[TreeNode], 
                   threshold: float = 0.5) -> TreeNode:
    """
    Build consensus tree from multiple trees
    
    Uses majority-rule consensus: only includes splits present in
    threshold proportion of input trees.
    
    Args:
        trees: List of input trees
        threshold: Minimum frequency for split to be included (0 to 1)
    
    Returns:
        TreeNode: Consensus tree
    
    Example:
        >>> consensus = consensus_tree(bootstrap_trees, threshold=0.7)
    """
    if not trees:
        raise ValueError("Need at least one tree")
    
    # Get all taxa
    all_leaves = trees[0].get_leaves()
    taxa = [leaf.name for leaf in all_leaves]
    n_trees = len(trees)
    
    # Count bipartitions (splits)
    split_counts = defaultdict(int)
    
    for tree in trees:
        splits = _get_splits(tree)
        for split in splits:
            # Normalize split (smaller side first)
            s1, s2 = split
            if len(s1) > len(s2):
                s1, s2 = s2, s1
            split_counts[s1] += 1
    
    # Filter splits by threshold
    consensus_splits = []
    for split, count in split_counts.items():
        if count / n_trees >= threshold:
            consensus_splits.append((split, count / n_trees))
    
    # Build tree from splits (simplified version)
    # In practice, this would use a more sophisticated algorithm
    if consensus_splits:
        # Use most frequent split as base
        consensus_splits.sort(key=lambda x: x[1], reverse=True)
        return _build_from_splits(consensus_splits, taxa)
    
    # Fallback: return star tree
    root = TreeNode()
    for taxon in taxa:
        leaf = TreeNode(name=taxon, branch_length=1.0)
        root.children.append(leaf)
    return root


def _get_splits(node: TreeNode) -> List[tuple]:
    """Extract all bipartitions from tree"""
    if node.is_leaf():
        return []
    
    splits = []
    leaves = node.get_leaves()
    all_taxa = frozenset(leaf.name for leaf in leaves)
    
    for child in node.children:
        child_leaves = frozenset(leaf.name for leaf in child.get_leaves())
        other_leaves = all_taxa - child_leaves
        if child_leaves and other_leaves:
            splits.append((child_leaves, other_leaves))
        splits.extend(_get_splits(child))
    
    return splits


def _build_from_splits(splits: List[tuple], taxa: List[str]) -> TreeNode:
    """Build tree from bipartitions (simplified)"""
    root = TreeNode()
    
    if not splits:
        # Star tree
        for taxon in taxa:
            root.children.append(TreeNode(name=taxon, branch_length=1.0))
        return root
    
    # Use largest split
    main_split, support = splits[0]
    
    # Create two subtrees
    left = TreeNode()
    right = TreeNode()
    
    for taxon in taxa:
        if taxon in main_split:
            left.children.append(TreeNode(name=taxon, branch_length=1.0))
        else:
            right.children.append(TreeNode(name=taxon, branch_length=1.0))
    
    root.children = [left, right]
    return root


def root_tree(tree: TreeNode, outgroup: str) -> TreeNode:
    """
    Root tree using specified outgroup taxon
    
    Args:
        tree: Unrooted tree
        outgroup: Name of outgroup taxon
    
    Returns:
        TreeNode: Rooted tree
    
    Example:
        >>> rooted = root_tree(tree, outgroup='Outgroup_Species')
    """
    # Find outgroup leaf
    outgroup_leaf = None
    for leaf in tree.get_leaves():
        if leaf.name == outgroup:
            outgroup_leaf = leaf
            break
    
    if not outgroup_leaf:
        raise ValueError(f"Outgroup '{outgroup}' not found in tree")
    
    # Create new root between outgroup and rest of tree
    new_root = TreeNode()
    
    # Simple reroot: put outgroup on one side
    new_root.children = [outgroup_leaf]
    
    # Other taxa on other side
    other_node = TreeNode()
    for leaf in tree.get_leaves():
        if leaf.name != outgroup:
            other_node.children.append(leaf)
    
    new_root.children.append(other_node)
    
    return new_root


def ladderize(tree: TreeNode, reverse: bool = False) -> TreeNode:
    """
    Sort tree branches by number of descendants (ladderize)
    
    Makes tree easier to visualize.
    
    Args:
        tree: Input tree
        reverse: If True, sort in reverse order
    
    Returns:
        TreeNode: Ladderized tree
    
    Example:
        >>> ladderized = ladderize(tree)
    """
    if tree.is_leaf():
        return tree
    
    # Recursively ladderize children
    for i, child in enumerate(tree.children):
        tree.children[i] = ladderize(child, reverse)
    
    # Sort children by number of descendants
    tree.children.sort(
        key=lambda node: len(node.get_leaves()),
        reverse=reverse
    )
    
    return tree


def compute_branch_support(tree: TreeNode, 
                           bootstrap_trees: List[TreeNode]) -> TreeNode:
    """
    Add bootstrap support values to tree branches
    
    Args:
        tree: Original tree
        bootstrap_trees: List of bootstrap trees
    
    Returns:
        TreeNode: Tree with support values added
    
    Example:
        >>> bootstrap_trees = bootstrap_tree(sequences, n_replicates=100)
        >>> tree_with_support = compute_branch_support(tree, bootstrap_trees)
    """
    # Get splits from original tree
    original_splits = set(_get_splits(tree))
    
    # Count how many bootstrap trees contain each split
    split_support = defaultdict(int)
    n_boot = len(bootstrap_trees)
    
    for boot_tree in bootstrap_trees:
        boot_splits = _get_splits(boot_tree)
        for split in boot_splits:
            s1, s2 = split
            if len(s1) > len(s2):
                s1, s2 = s2, s1
            if (s1, s2) in original_splits or (s2, s1) in original_splits:
                split_support[s1] += 1
    
    # Add support values to nodes
    _add_support_values(tree, split_support, n_boot)
    
    return tree


def _add_support_values(node: TreeNode, 
                       split_support: Dict,
                       n_boot: int):
    """Recursively add support values to nodes"""
    if node.is_leaf():
        return
    
    # Get split for this node
    child_taxa = frozenset(leaf.name for leaf in node.get_leaves())
    
    # Add support if available
    if child_taxa in split_support:
        support = split_support[child_taxa] / n_boot * 100
        node.name = f"{support:.1f}"  # Store as node label
    
    # Recurse to children
    for child in node.children:
        _add_support_values(child, split_support, n_boot)


def calculate_tree_distance(tree1: TreeNode, tree2: TreeNode) -> float:
    """
    Calculate Robinson-Foulds distance between two trees
    
    Args:
        tree1: First tree
        tree2: Second tree
    
    Returns:
        float: RF distance (number of different splits)
    
    Example:
        >>> dist = calculate_tree_distance(tree1, tree2)
    """
    splits1 = set(_get_splits(tree1))
    splits2 = set(_get_splits(tree2))
    
    # Normalize splits
    norm_splits1 = set()
    for s1, s2 in splits1:
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        norm_splits1.add((s1, s2))
    
    norm_splits2 = set()
    for s1, s2 in splits2:
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        norm_splits2.add((s1, s2))
    
    # RF distance = splits in one tree but not the other
    diff = norm_splits1.symmetric_difference(norm_splits2)
    return len(diff)


def get_tree_stats(tree: TreeNode) -> Dict:
    """
    Calculate various statistics about the tree
    
    Args:
        tree: Input tree
    
    Returns:
        Dict with statistics (n_leaves, n_internal_nodes, total_length, etc.)
    
    Example:
        >>> stats = get_tree_stats(tree)
        >>> print(f"Tree has {stats['n_leaves']} leaves")
    """
    leaves = tree.get_leaves()
    
    def count_internal(node):
        if node.is_leaf():
            return 0
        return 1 + sum(count_internal(child) for child in node.children)
    
    def total_branch_length(node):
        length = node.branch_length
        for child in node.children:
            length += total_branch_length(child)
        return length
    
    def max_depth(node):
        if node.is_leaf():
            return node.branch_length
        return node.branch_length + max(max_depth(child) for child in node.children)
    
    return {
        'n_leaves': len(leaves),
        'n_internal_nodes': count_internal(tree),
        'total_branch_length': total_branch_length(tree),
        'max_depth': max_depth(tree),
        'is_ultrametric': _is_ultrametric(tree)
    }


def _is_ultrametric(tree: TreeNode, tolerance: float = 1e-6) -> bool:
    """Check if tree is ultrametric (all leaves equidistant from root)"""
    leaves = tree.get_leaves()
    
    def leaf_depth(node, depth=0):
        if node.is_leaf():
            return depth + node.branch_length
        # Return depth of first leaf found
        return leaf_depth(node.children[0], depth + node.branch_length)
    
    depths = []
    for leaf in leaves:
        # Calculate depth by traversing from root
        depths.append(_calculate_leaf_depth(tree, leaf))
    
    if not depths:
        return True
    
    return max(depths) - min(depths) < tolerance


def _calculate_leaf_depth(node: TreeNode, target_leaf: TreeNode, depth: float = 0) -> float:
    """Calculate depth of specific leaf from given node"""
    if node == target_leaf:
        return depth
    
    if node.is_leaf():
        return float('inf')  # Not found in this branch
    
    for child in node.children:
        result = _calculate_leaf_depth(child, target_leaf, depth + child.branch_length)
        if result != float('inf'):
            return result
    
    return float('inf')
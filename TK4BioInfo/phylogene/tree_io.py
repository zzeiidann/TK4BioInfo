"""
Tree reading and writing functions (Newick format)
"""
from typing import Optional
from .tree_builder import TreeNode


def parse_newick(newick_str: str) -> TreeNode:
    """
    Parse Newick format string into tree structure
    
    Args:
        newick_str: Newick format string (e.g., "(A:0.1,B:0.2):0.0;")
    
    Returns:
        TreeNode: Root of parsed tree
    
    Example:
        >>> from phylogene import parse_newick
        >>> tree = parse_newick("(A:0.1,B:0.2):0.0;")
        >>> print(tree.children[0].name)
        A
    """
    newick_str = newick_str.strip().rstrip(';')
    
    def parse_node(s: str, pos: int = 0) -> tuple:
        """Recursively parse newick string"""
        node = TreeNode()
        
        if s[pos] == '(':
            # Internal node
            pos += 1
            children = []
            
            while True:
                child, pos = parse_node(s, pos)
                children.append(child)
                
                if s[pos] == ',':
                    pos += 1
                elif s[pos] == ')':
                    pos += 1
                    break
            
            node.children = children
        else:
            # Leaf node - read name
            name_end = pos
            while name_end < len(s) and s[name_end] not in ':,()':
                name_end += 1
            node.name = s[pos:name_end]
            pos = name_end
        
        # Read branch length if present
        if pos < len(s) and s[pos] == ':':
            pos += 1
            length_end = pos
            while length_end < len(s) and s[length_end] not in ',()':
                length_end += 1
            try:
                node.branch_length = float(s[pos:length_end])
            except ValueError:
                node.branch_length = 0.0
            pos = length_end
        
        return node, pos
    
    tree, _ = parse_node(newick_str, 0)
    return tree


def to_newick(node: TreeNode, include_branch_lengths: bool = True) -> str:
    """
    Convert tree to Newick format string
    
    Args:
        node: Root node of tree
        include_branch_lengths: Whether to include branch lengths
    
    Returns:
        str: Newick format string
    
    Example:
        >>> from phylogene import to_newick
        >>> newick_str = to_newick(tree)
        >>> print(newick_str)
    """
    if node.is_leaf():
        if include_branch_lengths and node.branch_length > 0:
            return f"{node.name}:{node.branch_length:.6f}"
        return node.name or ""
    
    children_str = ','.join(to_newick(child, include_branch_lengths) 
                           for child in node.children)
    
    if include_branch_lengths and node.branch_length > 0:
        return f"({children_str}):{node.branch_length:.6f}"
    
    return f"({children_str})"


def read_tree(filename: str) -> TreeNode:
    """
    Read tree from Newick format file
    
    Args:
        filename: Path to Newick format file
    
    Returns:
        TreeNode: Root of tree
    
    Example:
        >>> from phylogene import read_tree
        >>> tree = read_tree('tree.nwk')
    """
    with open(filename, 'r') as f:
        newick_str = f.read().strip()
    return parse_newick(newick_str)


def write_tree(tree: TreeNode, filename: str, 
               include_branch_lengths: bool = True):
    """
    Write tree to Newick format file
    
    Args:
        tree: Root node of tree
        filename: Output file path
        include_branch_lengths: Whether to include branch lengths
    
    Example:
        >>> from phylogene import write_tree
        >>> write_tree(tree, 'output.nwk')
    """
    newick_str = to_newick(tree, include_branch_lengths)
    with open(filename, 'w') as f:
        f.write(newick_str + ';\\n')


def read_fasta(filename: str) -> dict:
    """
    Read sequences from FASTA format file
    
    Args:
        filename: Path to FASTA file
    
    Returns:
        dict: Dictionary mapping sequence names to sequences
    
    Example:
        >>> sequences = read_fasta('seqs.fasta')
    """
    sequences = {}
    current_name = None
    current_seq = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_name:
                    sequences[current_name] = ''.join(current_seq)
                current_name = line[1:].split()[0]  # Take first word after >
                current_seq = []
            else:
                current_seq.append(line)
        
        if current_name:
            sequences[current_name] = ''.join(current_seq)
    
    return sequences


def write_fasta(sequences: dict, filename: str, width: int = 60):
    """
    Write sequences to FASTA format file
    
    Args:
        sequences: Dictionary mapping names to sequences
        filename: Output file path
        width: Number of characters per line (default 60)
    
    Example:
        >>> sequences = {'Seq1': 'ACGT', 'Seq2': 'GGCC'}
        >>> write_fasta(sequences, 'output.fasta')
    """
    with open(filename, 'w') as f:
        for name, seq in sequences.items():
            f.write(f'>{name}\\n')
            # Write sequence in lines of specified width
            for i in range(0, len(seq), width):
                f.write(seq[i:i+width] + '\\n')
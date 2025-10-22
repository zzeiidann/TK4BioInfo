"""
Phylogenetic tree plotting and visualization
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from typing import Optional, Tuple, Dict
from .tree_builder import TreeNode


def plot_tree(tree: TreeNode, 
              figsize: Tuple[int, int] = (10, 8),
              show_branch_lengths: bool = True,
              show_node_labels: bool = False,
              font_size: int = 10,
              title: Optional[str] = None) -> plt.Figure:
    """
    Plot phylogenetic tree
    
    Args:
        tree: Root node of tree
        figsize: Figure size (width, height)
        show_branch_lengths: Display branch lengths on tree
        show_node_labels: Show labels for internal nodes
        font_size: Font size for labels
        title: Plot title
    
    Returns:
        matplotlib.figure.Figure: The figure object
    
    Example:
        >>> from phylogene import plot_tree, NeighborJoining
        >>> import numpy as np
        >>> 
        >>> dist = np.array([[0, 2, 4], [2, 0, 4], [4, 4, 0]])
        >>> nj = NeighborJoining(dist, ['A', 'B', 'C'])
        >>> tree = nj.build_tree()
        >>> fig = plot_tree(tree, title='My Phylogenetic Tree')
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate leaf positions
    leaves = tree.get_leaves()
    leaf_positions = {leaf: i for i, leaf in enumerate(leaves)}
    
    # Calculate node coordinates
    node_coords = {}
    _calculate_coordinates(tree, leaf_positions, node_coords, 0)
    
    # Draw tree
    _draw_tree(tree, node_coords, ax, show_branch_lengths, show_node_labels, font_size)
    
    # Set plot properties
    ax.set_xlim(-0.5, max(x for x, y in node_coords.values()) + 0.5)
    ax.set_ylim(-0.5, len(leaves) - 0.5)
    ax.set_yticks(range(len(leaves)))
    ax.set_yticklabels([leaf.name for leaf in leaves])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Distance', fontsize=font_size)
    
    if title:
        ax.set_title(title, fontsize=font_size + 2, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_phylo(tree: TreeNode, 
               layout: str = 'rectangular',
               figsize: Tuple[int, int] = (12, 8),
               **kwargs) -> plt.Figure:
    """
    Advanced phylogenetic tree plotting with multiple layout options
    
    Args:
        tree: Root node of tree
        layout: Layout type: 'rectangular', 'circular', 'radial'
        figsize: Figure size
        **kwargs: Additional arguments passed to plot_tree
    
    Returns:
        matplotlib.figure.Figure: The figure object
    
    Example:
        >>> fig = plot_phylo(tree, layout='circular', title='Circular Tree')
        >>> plt.show()
    """
    if layout == 'rectangular':
        return plot_tree(tree, figsize=figsize, **kwargs)
    elif layout == 'circular':
        return _plot_circular_tree(tree, figsize=figsize, **kwargs)
    elif layout == 'radial':
        return _plot_radial_tree(tree, figsize=figsize, **kwargs)
    else:
        raise ValueError(f"Unknown layout: {layout}")


def _calculate_coordinates(node: TreeNode, 
                           leaf_positions: Dict[TreeNode, int],
                           node_coords: Dict[TreeNode, Tuple[float, float]],
                           x_pos: float):
    """Calculate x,y coordinates for all nodes"""
    if node.is_leaf():
        y_pos = leaf_positions[node]
        node_coords[node] = (x_pos, y_pos)
        return y_pos
    
    # Calculate positions for children
    child_y_positions = []
    for child in node.children:
        child_x = x_pos + child.branch_length
        child_y = _calculate_coordinates(child, leaf_positions, node_coords, child_x)
        child_y_positions.append(child_y)
    
    # Internal node y-position is average of children
    y_pos = sum(child_y_positions) / len(child_y_positions)
    node_coords[node] = (x_pos, y_pos)
    
    return y_pos


def _draw_tree(node: TreeNode,
               node_coords: Dict[TreeNode, Tuple[float, float]],
               ax: plt.Axes,
               show_branch_lengths: bool,
               show_node_labels: bool,
               font_size: int):
    """Recursively draw tree branches"""
    if node not in node_coords:
        return
    
    x_node, y_node = node_coords[node]
    
    for child in node.children:
        if child not in node_coords:
            continue
        
        x_child, y_child = node_coords[child]
        
        # Draw horizontal line to child
        ax.plot([x_node, x_child], [y_child, y_child], 'k-', linewidth=1.5)
        
        # Draw vertical connecting line
        ax.plot([x_node, x_node], [y_node, y_child], 'k-', linewidth=1.5)
        
        # Show branch length
        if show_branch_lengths and child.branch_length > 0:
            mid_x = (x_node + x_child) / 2
            ax.text(mid_x, y_child + 0.1, f'{child.branch_length:.3f}',
                   fontsize=font_size - 2, ha='center', style='italic')
        
        # Recursively draw children
        _draw_tree(child, node_coords, ax, show_branch_lengths, 
                  show_node_labels, font_size)
    
    # Draw node circle
    if not node.is_leaf():
        ax.plot(x_node, y_node, 'ko', markersize=4)
        if show_node_labels and hasattr(node, 'name') and node.name:
            ax.text(x_node, y_node + 0.2, node.name, fontsize=font_size - 2,
                   ha='center', va='bottom')


def _plot_circular_tree(tree: TreeNode,
                        figsize: Tuple[int, int] = (10, 10),
                        show_branch_lengths: bool = False,
                        title: Optional[str] = None,
                        **kwargs) -> plt.Figure:
    """Plot tree in circular layout"""
    import numpy as np
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    leaves = tree.get_leaves()
    n_leaves = len(leaves)
    
    # Assign angles to leaves
    angles = {leaf: 2 * np.pi * i / n_leaves for i, leaf in enumerate(leaves)}
    
    # Calculate radial distances
    node_coords = {}
    _calculate_radial_coordinates(tree, angles, node_coords, 0)
    
    # Draw tree in polar coordinates
    _draw_circular_tree(tree, node_coords, ax, show_branch_lengths)
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    
    ax.set_ylim(0, max(r for r, theta in node_coords.values()) * 1.1)
    ax.grid(False)
    
    return fig


def _calculate_radial_coordinates(node: TreeNode,
                                  leaf_angles: Dict,
                                  node_coords: Dict,
                                  radius: float):
    """Calculate polar coordinates for circular layout"""
    if node.is_leaf():
        theta = leaf_angles[node]
        node_coords[node] = (radius, theta)
        return theta
    
    # Calculate for children
    child_angles = []
    for child in node.children:
        child_r = radius + child.branch_length
        child_theta = _calculate_radial_coordinates(child, leaf_angles, 
                                                    node_coords, child_r)
        child_angles.append(child_theta)
    
    # Internal node angle is average of children
    theta = sum(child_angles) / len(child_angles)
    node_coords[node] = (radius, theta)
    
    return theta


def _draw_circular_tree(node: TreeNode,
                       node_coords: Dict,
                       ax: plt.Axes,
                       show_branch_lengths: bool):
    """Draw tree in polar coordinates"""
    import numpy as np
    
    if node not in node_coords:
        return
    
    r_node, theta_node = node_coords[node]
    
    for child in node.children:
        if child not in node_coords:
            continue
        
        r_child, theta_child = node_coords[child]
        
        # Draw radial line
        ax.plot([theta_child, theta_child], [r_node, r_child], 'k-', linewidth=1.5)
        
        # Draw arc connecting to other children
        if len(node.children) > 1:
            child_angles = [node_coords[c][1] for c in node.children if c in node_coords]
            theta_min, theta_max = min(child_angles), max(child_angles)
            theta_arc = np.linspace(theta_min, theta_max, 50)
            r_arc = [r_node] * len(theta_arc)
            ax.plot(theta_arc, r_arc, 'k-', linewidth=1.5)
        
        # Add leaf labels
        if child.is_leaf():
            ax.text(theta_child, r_child + 0.1, child.name,
                   rotation=np.degrees(theta_child) - 90,
                   ha='left' if theta_child < np.pi else 'right',
                   va='center', fontsize=9)
        
        _draw_circular_tree(child, node_coords, ax, show_branch_lengths)


def _plot_radial_tree(tree: TreeNode,
                      figsize: Tuple[int, int] = (10, 10),
                      **kwargs) -> plt.Figure:
    """Plot tree in radial (unrooted) layout"""
    # Similar to circular but with different angle calculation
    return _plot_circular_tree(tree, figsize=figsize, **kwargs)
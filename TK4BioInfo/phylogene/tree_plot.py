"""
Phylogenetic tree plotting and visualization (phylogram-style)
"""
import math
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict
from .tree_builder import TreeNode

# ---------- helpers ----------
def _nice_scale(x: float) -> float:
    """nice round length for scalebar (1/2/5 * 10^k)"""
    if x <= 0 or math.isinf(x) or math.isnan(x):
        return 1.0
    k = math.floor(math.log10(x))
    f = x / (10 ** k)
    if f < 1.5:
        n = 1.0
    elif f < 3.5:
        n = 2.0
    elif f < 7.5:
        n = 5.0
    else:
        n = 10.0
    return n * (10 ** k)

# ---------- main API ----------
def plot_tree(
    tree: TreeNode,
    figsize: Tuple[int, int] = (10, 8),
    show_branch_lengths: bool = False,
    font_size: int = 10,
    title: Optional[str] = None,
    scalebar: bool = True,
    tip_padding: float = 0.02,
) -> plt.Figure:
    """
    Draw rectangular phylogram similar to ape::plot.phylo.
    - Names printed at tips (right side), not on y-axis.
    - Axis/box hidden; optional scale bar.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 1) kumpulkan daun (urutkan sesuai traversal)
    leaves = _collect_leaves(tree)

    # 2) hitung koordinat (x kumulatif = total panjang cabang)
    coords: Dict[int, Tuple[float, float]] = {}
    _calc_coords(tree, coords, x0=0.0, leaf_order={id(n): i for i, n in enumerate(leaves)})

    # 3) gambar cabang
    _draw_rect(tree, coords, ax, show_branch_lengths, font_size)

    # 4) label tip di ujung kanan
    max_x = max(x for (x, _) in coords.values())
    for leaf in leaves:
        x, y = coords[id(leaf)]
        ax.text(x + tip_padding * (max_x if max_x > 0 else 1.0), y, leaf.name,
                va="center", ha="left", fontsize=font_size)

    # 5) kosmetik seperti ape
    ax.set_xlim(-0.02 * max_x, max_x * (1.05 + tip_padding * 5))
    ymin = -0.5
    ymax = len(leaves) - 0.5
    ax.set_ylim(ymin, ymax)
    ax.axis("off")

    # scale bar
    if scalebar and max_x > 0:
        bar = _nice_scale(max_x / 20.0)  # kira2 seperti ape
        x0 = max_x * 0.02
        y0 = ymin + 0.05 * (ymax - ymin)
        ax.plot([x0, x0 + bar], [y0, y0], "k-", lw=2)
        ax.text(x0 + bar / 2, y0 - 0.03 * (ymax - ymin), f"{bar:g}", ha="center", va="top", fontsize=font_size)

    if title:
        ax.set_title(title, fontsize=font_size + 2, fontweight="bold")

    plt.tight_layout()
    return fig

def plot_phylo(
    tree: TreeNode,
    layout: str = "rectangular",
    figsize: Tuple[int, int] = (12, 8),
    **kwargs
) -> plt.Figure:
    if layout == "rectangular":
        return plot_tree(tree, figsize=figsize, **kwargs)
    elif layout == "circular":
        return _plot_circular(tree, figsize=figsize, **kwargs)
    elif layout == "radial":
        # untuk sekarang radial = circular sederhana
        return _plot_circular(tree, figsize=figsize, **kwargs)
    else:
        raise ValueError(f"Unknown layout: {layout}")

# ---------- rectangular layout internals ----------
def _collect_leaves(node: TreeNode):
    """Depth-first leaves in stable order."""
    if node.is_leaf():
        return [node]
    out = []
    for c in node.children:
        out.extend(_collect_leaves(c))
    return out

def _calc_coords(node: TreeNode,
                 coords: Dict[int, Tuple[float, float]],
                 x0: float,
                 leaf_order: Dict[int, int]) -> float:
    """Return y position; fill coords with (x,y) using id(node) keys."""
    if node.is_leaf():
        y = leaf_order[id(node)]
        coords[id(node)] = (x0, float(y))
        return float(y)

    ys = []
    coords[id(node)] = (x0, 0.0)  # placeholder; y diisi setelah anak
    for child in node.children:
        y_child = _calc_coords(child, coords, x0 + max(child.branch_length, 0.0), leaf_order)
        ys.append(y_child)
    y_here = sum(ys) / len(ys)
    x_here, _ = coords[id(node)]
    coords[id(node)] = (x_here, y_here)
    return y_here

def _draw_rect(node: TreeNode,
               coords: Dict[int, Tuple[float, float]],
               ax: plt.Axes,
               show_branch_lengths: bool,
               font_size: int):
    x0, y0 = coords[id(node)]
    for child in node.children:
        x1, y1 = coords[id(child)]
        # horizontal segment
        ax.plot([x0, x1], [y1, y1], "k-", lw=1.5)
        # vertical connector
        ax.plot([x0, x0], [y0, y1], "k-", lw=1.5)

        if show_branch_lengths and child.branch_length > 0:
            xm = (x0 + x1) / 2.0
            ax.text(xm, y1 + 0.2, f"{child.branch_length:.3f}", ha="center", va="bottom", fontsize=font_size-2)

        _draw_rect(child, coords, ax, show_branch_lengths, font_size)

# ---------- simple circular (optional) ----------
def _plot_circular(tree: TreeNode,
                   figsize: Tuple[int, int] = (10, 10),
                   title: Optional[str] = None,
                   **kwargs) -> plt.Figure:
    import numpy as np
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection="polar"))

    leaves = _collect_leaves(tree)
    angles = {id(leaf): 2*np.pi*i/len(leaves) for i, leaf in enumerate(leaves)}
    rcoords: Dict[int, Tuple[float, float]] = {}
    _calc_polar(tree, rcoords, angles, r0=0.0)

    # draw
    _draw_polar(tree, rcoords, ax)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=20)
    ax.grid(False)
    return fig

def _calc_polar(node: TreeNode,
                rcoords: Dict[int, Tuple[float, float]],
                angles: Dict[int, float],
                r0: float) -> float:
    if node.is_leaf():
        th = angles[id(node)]
        rcoords[id(node)] = (r0, th)
        return th
    ths = []
    rcoords[id(node)] = (r0, 0.0)
    for child in node.children:
        th = _calc_polar(child, rcoords, angles, r0 + max(child.branch_length, 0.0))
        ths.append(th)
    rcoords[id(node)] = (r0, sum(ths) / len(ths))
    return rcoords[id(node)][1]

def _draw_polar(node: TreeNode,
                rcoords: Dict[int, Tuple[float, float]],
                ax: plt.Axes):
    import numpy as np
    r0, th0 = rcoords[id(node)]
    for child in node.children:
        r1, th1 = rcoords[id(child)]
        ax.plot([th1, th1], [r0, r1], "k-", lw=1.5)
        if len(node.children) > 1:
            child_ths = [rcoords[id(c)][1] for c in node.children]
            thmin, thmax = min(child_ths), max(child_ths)
            th_arc = np.linspace(thmin, thmax, 50)
            ax.plot(th_arc, [r0]*len(th_arc), "k-", lw=1.5)
        if child.is_leaf():
            ax.text(th1, r1 + 0.05*r1 if r1 else 0.05, child.name,
                    rotation=np.degrees(th1) - 90, ha="left" if th1 < np.pi else "right",
                    va="center", fontsize=9)
        _draw_polar(child, rcoords, ax)

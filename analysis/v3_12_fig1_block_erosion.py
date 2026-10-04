"""Figure fig1_block_erosion: erosion of a held-out block by the buffer.

Draws two panels of a held-out block of side g = 10 with buffer b = 2 inside its 3 x 3
neighbourhood: neighbours that trim the block are shaded blue, the area removed by the
buffer orange (a strip of width b per trimming side, a quarter disc of radius b per
isolated trimming corner), and the retained core is dashed. Below each panel the retained
area fraction rho(S) = [(g - b L)(g - b V) - pi b^2 / 4 C] / g^2 is printed, with L and V the
number of trimming lateral and vertical sides and C the number of isolated trimming corners.
Panel (a): sides B_w, B_n and corner B_es; panel (b): all four sides and corner B_en.

No data is read: the geometry is synthetic. Geometry and colours follow
v3_fig1_block_erosion.py; fonts are embedded as TrueType (fonttype 42).
Output: fig1_block_erosion.pdf and fig1_block_erosion.png (300 dpi) in HERE.

Usage: python v3_12_fig1_block_erosion.py
"""
import json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge
HERE = "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/redacao_v3-12/figures_v3-12"
os.makedirs(HERE, exist_ok=True)
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e6e3"

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

ROTULO = {"L": r"$B_{\mathsf{w}}$", "R": r"$B_{\mathsf{e}}$", "B": r"$B_{\mathsf{s}}$", "T": r"$B_{\mathsf{n}}$", "BL": r"$B_{\mathsf{ws}}$", "BR": r"$B_{\mathsf{es}}$", "TL": r"$B_{\mathsf{wn}}$", "TR": r"$B_{\mathsf{en}}$"}  # displayed neighbour symbols; keys L, R, B, T (+ corners) = west, east, south, north


def draw_block(ax, S, title):
    """Draw one panel; S is the set of trimming neighbours (keys of ROTULO)."""
    g, b = 10.0, 2.0
    ax.set_aspect("equal"); ax.set_xlim(-g - 1, 2 * g + 1); ax.set_ylim(-g - 1, 2 * g + 1)
    ax.axis("off"); ax.set_title(title, fontsize=9, color=INK, loc="left")
    # Neighbours on the 3 x 3 lattice.
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            if i == 0 and j == 0:
                continue
            key = {(-1, 0): "L", (1, 0): "R", (0, -1): "B", (0, 1): "T",
                   (-1, -1): "BL", (1, -1): "BR", (-1, 1): "TL", (1, 1): "TR"}[(i, j)]
            trims = key in S
            ax.add_patch(Rectangle((i * g, j * g), g, g, facecolor=(BLUE if trims else "#f4f4f2"),
                                   alpha=0.25 if trims else 1.0, edgecolor=INK2, lw=0.6))
            ax.text(i * g + g / 2, j * g + g / 2, ROTULO[key], ha="center", va="center", fontsize=8,
                    color=INK if trims else INK2)
    # Held-out block, then the regions removed by the buffer.
    ax.add_patch(Rectangle((0, 0), g, g, facecolor="white", edgecolor=INK, lw=1.0))
    removed_kw = dict(facecolor=ORANGE, alpha=0.30, edgecolor="none")
    if "L" in S: ax.add_patch(Rectangle((0, 0), b, g, **removed_kw))
    if "R" in S: ax.add_patch(Rectangle((g - b, 0), b, g, **removed_kw))
    if "B" in S: ax.add_patch(Rectangle((0, 0), g, b, **removed_kw))
    if "T" in S: ax.add_patch(Rectangle((0, g - b), g, b, **removed_kw))
    corners = {"BL": ((0, 0), 0, "L", "B"), "BR": ((g, 0), 90, "R", "B"),
               "TL": ((0, g), 270, "L", "T"), "TR": ((g, g), 180, "R", "T")}
    for c, (ctr, ang, s1, s2) in corners.items():
        if c in S and s1 not in S and s2 not in S:  # a corner trims only when neither adjacent side does
            ax.add_patch(Wedge(ctr, b, ang, ang + 90, facecolor=ORANGE, alpha=0.30, edgecolor="none"))
    # Retained core outline.
    x0 = b if "L" in S else 0; x1 = g - b if "R" in S else g
    y0 = b if "B" in S else 0; y1 = g - b if "T" in S else g
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=INK, lw=1.2, ls="--"))
    lat = sum(k in S for k in ("L", "R")); ver = sum(k in S for k in ("B", "T"))
    ncorner = sum(1 for c, (_, _, s1, s2) in corners.items() if c in S and s1 not in S and s2 not in S)
    area = (g - b * lat) * (g - b * ver) - math.pi * b * b / 4 * ncorner
    ax.text(g / 2, -g - 0.2, r"$\rho(\mathcal{S})=%.3f$  (trimmed by: %s)" % (area / g / g, ", ".join(ROTULO[k] for k in sorted(S))),
            ha="center", va="top", fontsize=8, color=INK)

fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.6))
draw_block(axes[0], {"L", "T", "BR"}, "(a) two sides + one isolated corner")
draw_block(axes[1], {"L", "R", "B", "T", "TR"}, "(b) all four sides")
fig.text(0.5, 0.005, "blue: neighbours that trim the block   |   orange: area removed by the buffer   |   dashed: retained core",
         ha="center", fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(os.path.join(HERE, "fig1_block_erosion.pdf"))
fig.savefig(os.path.join(HERE, "fig1_block_erosion.png"), dpi=300)
plt.close(fig)

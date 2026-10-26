"""Estimand Drift under Spatial Block Hold-Out (MDPI Mathematics, revision 1): Figure 1 (block erosion).
Declared copy of analysis/v3_12_fig1_block_erosion.py (digest of the original: 6677f8fa7bb04337418339d3f8ddb97bc137b42097c02eebd0854ca03d3a5cd7;
the original is not altered). Synthetic geometry g = 10, b = 2; values of rho identical to the earlier figure. Changes: figure size of
394.36 pt (text width) x 3.8 in, 9 pt text and 8.5 pt footer; neighbour alpha 0.25 -> 0.40 and hatched removed area for grey-scale
contrast; sub-titles in the vocabulary of the caption; label B on the retained block; footer in two lines. PDF with fonttype 42, PNG
for checking, JSON with the grey-scale luminances. No data enter this figure.
Output directory: env R1_OUT (default results/figures_v4_rerun under the repository root)."""
import json, math, os, hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge
from matplotlib.colors import to_rgb

HERE_SCRIPT = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE_SCRIPT)))  # repository root
SCRIPT_REL = os.path.relpath(HERE_SCRIPT, ROOT).replace(os.sep, "/")
OUT = os.environ.get("R1_OUT", os.path.join(ROOT, "results", "figures_v4_rerun"))
os.makedirs(OUT, exist_ok=True)
TEXTWIDTH_PT = 394.36
W_IN = TEXTWIDTH_PT / 72.0
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e6e3"
ALPHA_BLUE, ALPHA_ORANGE, NEUTRAL = 0.40, 0.30, "#f4f4f2"
FS = 9.0
H_IN = 3.8    # panels (3.1 in) + two text lines under them + two-line footer

plt.rcParams.update({"font.family": "serif", "font.size": FS, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 0.5,
                     "mathtext.fontset": "dejavuserif"})

ROTULO = {"L": r"$B_{\mathsf{w}}$", "R": r"$B_{\mathsf{e}}$", "B": r"$B_{\mathsf{s}}$", "T": r"$B_{\mathsf{n}}$", "BL": r"$B_{\mathsf{ws}}$", "BR": r"$B_{\mathsf{es}}$", "TL": r"$B_{\mathsf{wn}}$", "TR": r"$B_{\mathsf{en}}$"}


def draw_block(ax, S, title):
    g, b = 10.0, 2.0
    ax.set_aspect("equal"); ax.set_xlim(-g - 1, 2 * g + 1); ax.set_ylim(-g - 1.2, 2 * g + 1)
    ax.axis("off"); ax.set_title(title, fontsize=FS, color=INK, loc="left")
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            if i == 0 and j == 0:
                continue
            key = {(-1, 0): "L", (1, 0): "R", (0, -1): "B", (0, 1): "T",
                   (-1, -1): "BL", (1, -1): "BR", (-1, 1): "TL", (1, 1): "TR"}[(i, j)]
            trims = key in S
            ax.add_patch(Rectangle((i * g, j * g), g, g, facecolor=(BLUE if trims else NEUTRAL),
                                   alpha=ALPHA_BLUE if trims else 1.0, edgecolor=INK2, lw=0.6))
            ax.text(i * g + g / 2, j * g + g / 2, ROTULO[key], ha="center", va="center", fontsize=FS,
                    color=INK if trims else INK2)
    ax.add_patch(Rectangle((0, 0), g, g, facecolor="white", edgecolor=INK, lw=1.0))
    rem = dict(facecolor=to_rgb(ORANGE) + (ALPHA_ORANGE,), edgecolor=ORANGE, hatch="////", lw=0.0)
    if "L" in S: ax.add_patch(Rectangle((0, 0), b, g, **rem))
    if "R" in S: ax.add_patch(Rectangle((g - b, 0), b, g, **rem))
    if "B" in S: ax.add_patch(Rectangle((0, 0), g, b, **rem))
    if "T" in S: ax.add_patch(Rectangle((0, g - b), g, b, **rem))
    corners = {"BL": ((0, 0), 0, "L", "B"), "BR": ((g, 0), 90, "R", "B"),
               "TL": ((0, g), 270, "L", "T"), "TR": ((g, g), 180, "R", "T")}
    for c, (ctr, ang, s1, s2) in corners.items():
        if c in S and s1 not in S and s2 not in S:
            ax.add_patch(Wedge(ctr, b, ang, ang + 90, **rem))
    x0 = b if "L" in S else 0; x1 = g - b if "R" in S else g
    y0 = b if "B" in S else 0; y1 = g - b if "T" in S else g
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=INK, lw=1.2, ls="--"))
    # label of the retained block (centre of the core)
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, r"$B$", ha="center", va="center", fontsize=FS, color=INK, fontweight="bold")
    lat = sum(k in S for k in ("L", "R")); ver = sum(k in S for k in ("B", "T"))
    ncorner = sum(1 for c, (_, _, s1, s2) in corners.items() if c in S and s1 not in S and s2 not in S)
    area = (g - b * lat) * (g - b * ver) - math.pi * b * b / 4 * ncorner
    rho = area / g / g
    ax.text(g / 2, -g - 0.2, r"$\rho(\mathcal{S})=%.3f$" % rho + "\n(trimmed by: %s)" % ", ".join(ROTULO[k] for k in sorted(S)),
            ha="center", va="top", fontsize=FS, color=INK, linespacing=1.15)
    return rho


fig, axes = plt.subplots(1, 2, figsize=(W_IN, H_IN))
rho_a = draw_block(axes[0], {"L", "T", "BR"}, "(a) two lateral + one\ndiagonal neighbour")
rho_b = draw_block(axes[1], {"L", "R", "B", "T", "TR"}, "(b) four lateral neighbours\n(+ one diagonal)")
fig.text(0.5, 0.012, "blue: neighbours that trim the block   |   hatched orange: area removed by the buffer\ndashed: area retained after trimming",
         ha="center", va="bottom", fontsize=8.5, color=INK2, linespacing=1.2)
fig.subplots_adjust(left=0.01, right=0.99, top=0.895, bottom=0.205, wspace=0.04)
fig.savefig(os.path.join(OUT, "fig1_block_erosion_v4.pdf"))
fig.savefig(os.path.join(OUT, "fig1_block_erosion_v4.png"), dpi=300)
plt.close(fig)


def lum(rgb):  # Rec. 709 on gamma-encoded sRGB, 0-255
    return 255 * (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])


def sobre_branco(cor, a):
    return tuple(a * c + (1 - a) for c in to_rgb(cor))


info = dict(
    script=SCRIPT_REL, script_sha256=hashlib.sha256(open(HERE_SCRIPT, "rb").read()).hexdigest(),
    original="analysis/v3_12_fig1_block_erosion.py", original_sha256="6677f8fa7bb04337418339d3f8ddb97bc137b42097c02eebd0854ca03d3a5cd7",
    largura_pt=TEXTWIDTH_PT, altura_pt=H_IN * 72, fonte_pt=FS, fonte_rodape_pt=8.5,
    rho_a=rho_a, rho_b=rho_b, rho_esperado={"a": 0.64 - math.pi * 0.04 / 4, "b": 0.36},
    luminancia_cinza_0a255={"vizinho_nao_aparador": lum(to_rgb(NEUTRAL)), "vizinho_aparador_alfa_0.40": lum(sobre_branco(BLUE, ALPHA_BLUE)),
                            "vizinho_aparador_alfa_0.25_v3": lum(sobre_branco(BLUE, 0.25)),
                            "area_removida_laranja_alfa_0.30_sobre_branco": lum(sobre_branco(ORANGE, ALPHA_ORANGE)),
                            "nota": "a area removida distingue-se tambem por hachura '////' (padrao, nao so cor)"},
)
with open(os.path.join(OUT, "fig1_block_erosion_v4.json"), "w") as f:
    json.dump(info, f, indent=1)
print(json.dumps(info, indent=1))

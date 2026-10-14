"""v3 figure style for the FMM paper: one look for every main figure.

    from fmmpaper import style3 as S
    S.setup(); fig = S.figure(h_mm=90); ...; S.label(ax, "a"); S.save(fig, "Fig4_domains")

Text 7 pt everywhere (Nature Comms maximum), panel letters 8 pt bold, 180 mm width. Colours are
semantic and shared across figures: the quasi-static channel is blue, CR1 is red, matching the
coolwarm maps; the beam-model route is orange and the reconstructed-map route teal (Fig. 3);
raw (uncorrected) values are grey open symbols, corrected values filled.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

MM = 1 / 25.4
W_DOUBLE, W_ONEHALF, W_SINGLE = 180 * MM, 120 * MM, 88 * MM
INK, GREY, LIGHT = "#222222", "#8c8c8c", "#c4c4c4"
QS, CR1 = "#3a6db3", "#b8352b"                 # channels (coolwarm ends, darkened for lines)
MODEL, MAP, POLY, GP = "#c47f00", "#009e8a", "#7b5ea7", "#8c8c8c"   # methods / routes
D1, D2 = "#d6604d", "#4393c3"                  # domains 1 and 2
PROBE = {"PPP-CONTAu": "#c47f00", "Multi75-G": "#009e8a", "SCM-PIT": "#7b5ea7", "Adama": "#1f3f7a"}   # first three validated: all pairs pass CVD; Adama (navy) separated by lightness
MAP_CMAP = "coolwarm"                          # spectral maps (dB) and signed images
ERR_CMAP = LinearSegmentedColormap.from_list("err", ["#ffffff", "#f6d3c4", "#e48a6f", "#b8352b", "#5c0d0d"])
OUT = Path(__file__).resolve().parents[1] / "figures" / "v3"


def setup():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "figure.titlesize": 7,
        "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.pad": 2, "ytick.major.pad": 2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "axes.titlepad": 3, "axes.labelpad": 2, "axes.titleweight": "normal",
        "lines.linewidth": 1.0, "lines.markersize": 3.5, "lines.markeredgewidth": 0.6,
        "legend.frameon": False, "legend.handlelength": 1.2, "legend.handletextpad": 0.4, "legend.borderaxespad": 0.2,
        "figure.facecolor": "white", "savefig.facecolor": "white", "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "mathtext.default": "regular", "pdf.fonttype": 42, "svg.fonttype": "none", "image.cmap": MAP_CMAP})


def figure(h_mm, w=W_DOUBLE):
    return plt.figure(figsize=(w, h_mm * MM))


def label(ax, s, dx=-0.16, dy=1.0):
    """Bold lowercase panel letter at the axes' top-left, outside."""
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def flabel(fig, x, y, s):
    fig.text(x, y, s, fontsize=8, fontweight="bold", va="top", ha="left")


def boxes(ax):
    """Closed box (used for maps and images)."""
    for sp in ax.spines.values():
        sp.set_visible(True)


def deff_label(kind=""):
    return f"$d_{{\\mathrm{{eff}}}}${(' ' + kind) if kind else ''} (pm V$^{{-1}}$)"


def sym_im(ax, img, vmax, extent=None, cmap=MAP_CMAP):
    return ax.imshow(img, cmap=cmap, vmin=-vmax, vmax=vmax, origin="lower", extent=extent, interpolation="nearest", rasterized=True)


def signed_projection(z, m1, m2):
    """Project a complex image onto its two-domain axis: s = Re(z exp(-i phi_P)), with P = (<z>_1 - <z>_2)/2,
    so domain 1 is positive and domain 2 negative; |s| on each domain equals the response along P."""
    P = (z[m1].mean() - z[m2].mean()) / 2
    return np.real(z * np.exp(-1j * np.angle(P)))


def save(fig, name, formats=("png", "pdf", "svg")):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        p = OUT / f"{name}.{ext}"; fig.savefig(p, dpi=600); print("wrote", p)
    plt.close(fig)

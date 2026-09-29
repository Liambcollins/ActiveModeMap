"""One figure style for every notebook: Nature Communications (Arial 7 pt, 88/180 mm).

Legends: frameless, placed in empty white space, no arrows (Liam's rule).

    from fmmpaper import plotting as fp
    fp.setup()                       # once per notebook
    fig, ax = fp.figure("single")   # APS/AIP column widths
    fp.map_db(ax, x, f, Z)          # position-frequency amplitude map in dB
    fp.save(fig, "fig2_sparse_recon")   # -> figures/fig2_sparse_recon.png + .pdf

Palette (fixed across the paper): measured = black, GP = blue, EB+GP = red,
EB alone = grey, low-rank = teal, oracle = black star, random = light grey.
Probes: SCM-PIT-A = purple, PPP-CONTAu = orange, SCM-PIT-B = green.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from . import config

INK = "#1a1a1a"
C = dict(meas="#000000", gp="#2166ac", ebgp="#b2182b", eb="#888888", lowrank="#1b9e77",
         oracle="#1a1a1a", random="#bbbbbb", qs="#555555", cr1="#b2182b", cr2="#2166ac",
         cr3="#1b9e77", cr4="#7570b3", cr5="#e6ab02")
PROBE_C = dict(scmpitA="#7570b3", pppcontau="#d95f02", scmpitB="#1b9e77", p0="#999999")
DOMAIN_C = {1: "#b2182b", 2: "#2166ac"}
# Nature Communications: 88 mm single, 120 mm 1.5-column, 180 mm double column
MM = 1 / 25.4
WIDTH_IN = dict(single=88 * MM, onehalf=120 * MM, double=180 * MM)
MAP_CMAP = "magma"
ERR_CMAP = "coolwarm"


def setup(dpi: int = 150):
    plt.rcParams.update({
        # Nature Comms: Arial (Liberation Sans is metric-identical where Arial is absent), 5-7 pt
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.linewidth": .7, "xtick.major.width": .7, "ytick.major.width": .7,
        "xtick.major.size": 2.6, "ytick.major.size": 2.6, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.grid": False, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "lines.linewidth": 1.2, "figure.dpi": dpi, "savefig.dpi": 600, "savefig.bbox": "tight",
        "mathtext.default": "regular", "pdf.fonttype": 42})


def figure(width="single", aspect=0.75, nrows=1, ncols=1, **kw):
    w = WIDTH_IN.get(width, width) if isinstance(width, str) else width
    return plt.subplots(nrows, ncols, figsize=(w, w * aspect), **kw)


def panel_label(ax, s, dx=-0.02, dy=1.02):
    """Nature style: bold lowercase letter, 8 pt, top-left outside the axes."""
    ax.text(dx, dy, s.lower(), transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="right")


def map_db(ax, x_um, freq_Hz, Z, vmin=-60, vmax=0, ref=None, fmax=None, fmin=None,
           cmap=MAP_CMAP, colorbar=True, label="|Z| (dB)"):
    """Position (x) vs frequency (y, kHz) amplitude map in dB re max (or ``ref``)."""
    A = np.abs(Z)
    ref = A.max() if ref is None else ref
    db = 20 * np.log10(np.maximum(A, 1e-30) / ref)
    f = freq_Hz / 1e3
    m = np.ones(f.size, bool)
    if fmin is not None:
        m &= f >= fmin
    if fmax is not None:
        m &= f <= fmax
    im = ax.pcolormesh(x_um, f[m], db[:, m].T, shading="nearest", cmap=cmap,
                       vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_xlabel("position (µm)"); ax.set_ylabel("frequency (kHz)")
    if colorbar:
        plt.colorbar(im, ax=ax, label=label, pad=0.02)
    return im


def mark_positions(ax, x, y=None, color="w", marker="v", size=14, **kw):
    """Triangles at measured positions along the top of a map."""
    y = ax.get_ylim()[1] if y is None else y
    ax.scatter(x, np.full(len(x), y), marker=marker, s=size, c=color, clip_on=False,
               edgecolors="k", linewidths=0.4, zorder=5, **kw)


def spectrum(ax, freq_Hz, z, label=None, color=C["meas"], db=False, **kw):
    a = np.abs(z)
    y = 20 * np.log10(np.maximum(a, 1e-30)) if db else a
    ax.plot(freq_Hz / 1e3, y, color=color, label=label, **kw)
    ax.set_xlabel("frequency (kHz)")


def save(fig, name: str, formats=("png", "pdf")):
    """Save to paper_analysis/figures/<name>.<fmt> and return the paths."""
    out = []
    for fmt in formats:
        p = config.FIG_DIR / f"{name}.{fmt}"
        fig.savefig(p)
        out.append(p)
    return out


def n_axis(ax, ticks=(3, 4, 6, 10, 20)):
    """Log x-axis for 'measured positions N' with plain integer ticks (no 3x10^0 labels)."""
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter())

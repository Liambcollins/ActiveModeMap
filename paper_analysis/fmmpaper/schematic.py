"""Measurement schematic for Fig. 1: laser spot stepped along a cantilever whose tip is in contact.

    from fmmpaper import schematic
    schematic.draw(ax)
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch


def draw(ax, n_sparse=6, color_laser="#b2182b", color_beam="#d9d9d9"):
    ax.set_xlim(-1.0, 19.5); ax.set_ylim(-2.35, 2.35); ax.set_aspect("equal"); ax.set_anchor("W"); ax.axis("off")
    L, y0, t = 9.6, 0.0, 0.22                         # lever length, top surface, thickness
    xtip = L - 0.45
    # clamp / chip
    ax.add_patch(Rectangle((-0.6, -0.9), 0.6, 2.2, facecolor="#bdbdbd", edgecolor="k", lw=0.6, hatch="////"))
    # lever
    ax.add_patch(Rectangle((0, y0 - t), L, t, facecolor=color_beam, edgecolor="k", lw=0.6))
    # tip
    ax.add_patch(Polygon([[xtip - 0.28, y0 - t], [xtip + 0.28, y0 - t], [xtip, -1.35]], closed=True,
                         facecolor="#737373", edgecolor="k", lw=0.5))
    # sample: two ferroelectric domains
    ax.add_patch(Rectangle((1.2, -2.3), 5.2, 0.95, facecolor="#f4a582", edgecolor="k", lw=0.5))
    ax.add_patch(Rectangle((6.4, -2.3), 4.9, 0.95, facecolor="#92c5de", edgecolor="k", lw=0.5))
    for xa, up in ((2.6, True), (4.6, True), (8.0, False), (10.0, False)):
        ax.annotate("", xy=(xa, -1.5 if up else -2.15), xytext=(xa, -2.15 if up else -1.5),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color="k", mutation_scale=7))
    ax.text(11.45, -1.83, "PPLN, two domains", fontsize=5.5, ha="left", va="center")
    # drive
    ax.text(xtip + 0.45, -0.95, "V$_{dc}$ + V$_{ac}$", fontsize=5.5, ha="left", va="center")
    # dense grid ticks and sparse laser spots
    xs_dense = np.linspace(0.6, L - 0.3, 60)
    ax.plot(xs_dense, np.full_like(xs_dense, y0 + 0.07), "|", color="0.55", ms=2.2, mew=0.5)
    xs = np.linspace(0.8, L - 0.35, n_sparse)
    for i, x in enumerate(xs):
        a = 1.0 if i == 3 else 0.35
        ax.add_patch(Polygon([[x - 0.1, 1.4], [x + 0.1, 1.4], [x + 0.03, y0 + 0.02], [x - 0.03, y0 + 0.02]],
                             closed=True, facecolor=color_laser, edgecolor="none", alpha=a))
        ax.plot(x, y0 + 0.02, "o", color=color_laser, ms=2.6, alpha=a, mec="none")
    ax.add_patch(Rectangle((xs[3] - 0.85, 1.4), 1.7, 0.45, facecolor="white", edgecolor="k", lw=0.6))
    ax.text(xs[3], 1.61, "laser", fontsize=5.5, ha="center", va="center")
    ax.annotate("", xy=(xs[4] + 0.2, 1.61), xytext=(xs[3] + 0.9, 1.61),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="k", mutation_scale=6))
    ax.text(xs[4] + 0.3, 1.61, "stepped along the lever", fontsize=5.5, ha="left", va="center")
    # distance axis
    ax.annotate("", xy=(7.6, -0.5), xytext=(0, -0.5), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.3", mutation_scale=6))
    ax.plot([0, 0], [-0.38, -0.62], color="0.3", lw=0.6)
    ax.text(0.15, -0.72, "x − x$_0$, distance from clamp", fontsize=5.5, color="0.3", ha="left", va="top")
    ax.plot([12.9, 13.5], [0.75, 0.75], "|", color="0.55", ms=3, mew=0.6)
    ax.text(13.9, 0.75, "dense map: every 1 µm", fontsize=5.5, color="0.25", va="center")
    ax.plot([13.2], [0.05], "o", color=color_laser, ms=3, mec="none")
    ax.text(13.9, 0.05, "sparse capture: N positions", fontsize=5.5, color="0.25", va="center")
    ax.text(-0.3, 1.45, "clamp", fontsize=5.5, ha="center", va="bottom")
    ax.text(xtip - 0.4, -1.0, "tip", fontsize=5.5, ha="right", va="center")
    return ax

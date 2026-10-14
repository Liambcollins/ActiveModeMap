"""Supplementary Fig. 9: Fig. 2 reconstructions scored at omitted positions, against the measurement floor,
and in derived quantities (resonance frequency, peak amplitude, node position), per band.

Needs results/err_concept_soft.npz from tools/err_concept.py (about 1.5 min).
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter
from matplotlib.patches import Patch
from fmmpaper import style3 as S, config

S.setup()
d = np.load(config.RESULTS_DIR / "err_concept_soft.npz")
NS = d["NS"]; names = list(d["names"]); fl = json.loads(str(d["floor"]))
ARMS = [("EB", "beam model", S.MODEL, "-"), ("EB+GP", "hybrid", S.MAP, "-"), ("low-rank", "polynomial", S.POLY, "--")]
G = lambda a, k: d[f"{a}|{k}"]
mk = dict(marker="o", ms=2.3, mew=0, lw=0.9)
fmtN = plt.FuncFormatter(lambda v, p: f"{v:.0f}")

fig = S.figure(h_mm=150)
gs = fig.add_gridspec(4, 5, hspace=0.55, wspace=0.16, left=0.085, right=0.99, top=0.93, bottom=0.07)
ROWS = [("nrmse", "complex error (%)", "noise", (3, 200), [3, 10, 30, 100], True),
        ("df", "frequency error\n(linewidths)", "df", (0.01, 2), [0.01, 0.1, 1], True),
        ("da", "peak amplitude\nerror (%)", "da", (0, 60), [0, 20, 40, 60], False),
        ("dx", "node position\nerror (µm)", None, (0, 30), [0, 10, 20, 30], False)]
for r, (key, ylab, fk, ylim, yt, ylog) in enumerate(ROWS):
    for i, b in enumerate(names):
        if key == "dx" and b in ("CR1", "CR2"):
            continue
        ax = fig.add_subplot(gs[r, i])
        if fk:
            ax.axhspan(ylim[0] if not ylog else 1e-4, fl[b][fk], color=S.LIGHT, alpha=0.55, lw=0); ax.axhline(fl[b][fk], color=S.GREY, lw=0.6)
        else:
            ax.axhline(1, color=S.GREY, lw=0.6, ls=":")
        for a, lab, c, ls in ARMS:
            v = G(a, key)[:, i]
            top = ylim[1] * (0.9 if ylog else 0.97); over = v > top
            ax.plot(NS, np.clip(v, ylim[0] if not ylog else 1e-4, top), color=c, ls=ls, **mk)
            if over.any(): ax.plot(NS[over], np.full(over.sum(), top), ls="none", marker="^", ms=3.4, mfc="white", mec=c, mew=0.7, clip_on=False, zorder=5)
        ax.set_xscale("log"); ax.set_xlim(2.7, 22)
        if ylog: ax.set_yscale("log")
        ax.set_ylim(*ylim)
        ax.xaxis.set_major_locator(FixedLocator([3, 6, 12, 20])); ax.xaxis.set_major_formatter(fmtN); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(FixedLocator(yt)); ax.yaxis.set_minor_formatter(NullFormatter())
        first = i == 0 or (key == "dx" and i == 2)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{v:g}") if first else NullFormatter())
        if first:
            ax.set_ylabel(ylab); S.label(ax, "abcd"[r], dx=-0.55 if i == 0 else -0.5, dy=1.02)
        if r == 0: ax.set_title(b)
        if key == "dx": ax.set_title(b, pad=2)
        if r == 3 and i == 3: ax.set_xlabel("number of sampled positions N")
fig.legend(handles=[plt.Line2D([], [], color=c, ls=ls, **mk, label=l) for _, l, c, ls in ARMS]
           + [Patch(color=S.LIGHT, label="measurement floor (adjacent positions)"),
              plt.Line2D([], [], color=S.GREY, lw=0.6, ls=":", label="map pitch (1 µm)"),
              plt.Line2D([], [], ls="none", marker="^", mfc="white", mec=S.INK, mew=0.7, ms=3.4, label="off scale")],
           loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.0), handlelength=1.8)
S.save(fig, "FigS9_error_floor")

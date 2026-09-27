"""FEM vs EB vs low-rank: the paradigm comparison, four panels."""
# Paths are resolved through config.py -- set AMM_* environment
# variables or edit that file to point at your data.
import sys as _sys, os as _os
_D = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_D, 'config.py')):
    _D = _os.path.dirname(_D)
_sys.path[:0] = [_D, _os.path.join(_D, 'src'),
                 _os.path.join(_D, 'figures')]
from config import DENSE_GRID_A, FEM_LADDER, EB_GEOMETRY, OUT, FIG, add_eb_to_path
add_eb_to_path()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

import ornl as O
SURF = O.WHITE
INK, INK2, MUT = O.INK, O.INK2, O.MUT
# hue = forward model; linestyle = with / without the GP discrepancy term
HUE = O.HUE
SERIES = [  # (key, model, label, style, marker)
    ('lowrank', 'lowrank', '3  low-rank', '-', 'o'),
    ('eb',      'eb',      '1  EB', '-', 's'),
    ('eb_gp',   'eb',      '2  EB + GP', '--', 's'),
    ('fem',     'fem',     '1b  FEM', '-', 'D'),
    ('fem_gp',  'fem',     '1b  FEM + GP', '--', 'D'),
]

D = pd.concat([pd.read_csv(str(OUT) + '/phys_bench.csv'),
               pd.read_csv(str(OUT) + '/fem_bench.csv')], ignore_index=True)
E = D[D.strategy == 'equispaced']

PANELS = [
    ('amp_map_pct', 'held-out complex-map NRMSE  (%)',
     'Held-out map error', True, None),
    ('dns_err', 'D-NS absolute error  (µm)',
     'D-NS accuracy', True, [(1.0, '1 µm'), (0.5, '0.5 µm')]),
    ('peak_neartip_pct', 'near-tip peak error  (%)',
     'Near-tip error, last 10 µm', True, None),
]

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.35), facecolor=SURF)
fig.subplots_adjust(left=.052, right=.838, top=.895, bottom=.135, wspace=.30)

for ax, (metric, ylab, sub, logy, rules) in zip(axes.ravel(), PANELS):
    ax.set_facecolor(SURF)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUT); ax.spines[s].set_linewidth(.8)
    ax.grid(True, which='major', color='#e6e5e0', lw=.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9.5, length=3, width=.8)

    if metric == 'k1':                     # Hertz band, drawn behind the data
        ax.axhspan(1000, 2154, color=O.GREEN, alpha=.10, lw=0, zorder=1)
        ax.axhline(1310, color=O.GREEN, lw=1.2, ls=(0, (1, 2)), zorder=2)
        ax.text(.985, 1310, 'Hertz 1310 N/m ', color='#004D21', fontsize=8.5,
                va='bottom', ha='right', zorder=6,
                transform=ax.get_yaxis_transform())
        ax.text(.985, 2154, 'independent estimates ', color='#004D21',
                fontsize=8.5, va='bottom', ha='right', alpha=.9, zorder=6,
                transform=ax.get_yaxis_transform())
    for y, lab in (rules or []):
        ax.axhline(y, color=MUT, lw=1.0, ls=(0, (4, 3)), zorder=2)


    for key, model, label, ls, mk in SERIES:
        if metric == 'k1' and key in ('lowrank',):
            continue
        if metric == 'k1' and key.endswith('_gp'):
            continue                        # identical to the bare arm by design
        g = E[E.rec == key].dropna(subset=[metric]).sort_values('n')
        if not len(g):
            continue
        c = HUE[model]
        ax.plot(g.n, g[metric], ls, color=c, lw=2.0, marker=mk, ms=5.2,
                mew=1.6, mec=SURF, zorder=4, clip_on=True)

    ax.set_xscale('log')
    ax.set_xticks([3, 5, 8, 12, 20, 50])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    if logy:
        ax.set_yscale('log')
        YT = {'amp_map_pct': [1.5, 2, 3, 5, 8, 12, 20, 30, 45],
              'dns_err': [.05, .1, .2, .5, 1, 2, 5, 10, 20, 40],
              'peak_neartip_pct': [8, 12, 20, 30, 50, 80, 130, 200, 300],
              'k1': [300, 400, 600, 900, 1300, 2000]}[metric]
        lo, hi = ax.get_ylim()
        ax.set_yticks([t for t in YT if lo <= t <= hi])
        ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(
                lambda v, _: ('%g' % v) if v >= 1 else ('%.2g' % v)))
        ax.get_yaxis().set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel('positions measured  (n)', color=INK2, fontsize=10)
    ax.set_ylabel(ylab, color=INK, fontsize=10.5)
    ax.set_title(sub, color=INK, fontsize=11.5, fontweight='600',
                 loc='left', pad=8)

# one legend, outside the plots, with the composite encoding spelled out
handles = [plt.Line2D([], [], color=HUE[m], ls=ls, lw=2.0, marker=mk, ms=5.2,
                      mew=1.6, mec=SURF, label=lab)
           for k, m, lab, ls, mk in SERIES]
leg = fig.legend(handles=handles, loc='center left', bbox_to_anchor=(.858, .70),
                 frameon=False, fontsize=10.5, labelcolor=INK2,
                 title='paradigm')
leg.get_title().set_color(INK)
leg.get_title().set_fontsize(10.5)
leg.get_title().set_fontweight('600')
fig.text(.858, .42,
         'hue = forward model\nsolid = physics alone\ndashed = + GP discrepancy',
         color=INK2, fontsize=9, va='top')
fig.text(.858, .235,
         'Equispaced designs.\nLower is better in\nevery panel.',
         color=INK2, fontsize=9, va='top')


fig.savefig(str(FIG) + '/d_al.png', dpi=170, facecolor=SURF)
print('wrote fig/d_al.png')

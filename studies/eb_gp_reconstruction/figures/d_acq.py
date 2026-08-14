"""Slide figure: choosing the forward model beats choosing the next point."""
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

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import ornl as O

O.style()
D = pd.concat([pd.read_csv(str(OUT) + '/phys_bench.csv'),
               pd.read_csv(str(OUT) + '/fem_bench.csv')], ignore_index=True)
ARMS = ['lowrank', 'eb', 'eb_gp', 'fem', 'fem_gp']
STR = [('equispaced', 'equispaced'), ('dopt_lowrank', 'D-optimal'),
       ('maxvar', 'max-variance'), ('random', 'random (median)')]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 4.3))
fig.subplots_adjust(left=.065, right=.985, top=.905, bottom=.135, wspace=.20)

# --- left: strategy spread WITHIN a paradigm, at n = 5
w = .19
for j, (sk, sl) in enumerate(STR):
    v = []
    for a_ in ARMS:
        g = D[(D.rec == a_) & (D.strategy == sk) & (D.n == 5)].amp_map_pct
        v.append(g.median() if len(g) else np.nan)
    a1.bar(np.arange(len(ARMS)) + (j - 1.5) * w, v, w * .9,
           color=O.GREEN, alpha=.30 + .23 * j, edgecolor=O.WHITE, lw=.8,
           label=sl, zorder=3)
a1.set_xticks(range(len(ARMS)))
a1.set_xticklabels([O.LBL[a_].replace('  ', '\n') for a_ in ARMS], fontsize=9.5)
a1.set_ylabel('held-out NRMSE at n = 5  (%)')
a1.legend(fontsize=9.3, labelcolor=O.INK2, ncol=2)
a1.set_ylim(0, 52)
O.clean(a1); a1.grid(axis='x', visible=False)
O.title(a1, 'Within a paradigm, the strategy barely matters')
a1.annotate('choosing the model:\n38.7 → 3.0 %',
            xy=(3.85, 6), xytext=(2.15, 27), fontsize=10, color='#004D21',
            fontweight='600', ha='center',
            arrowprops=dict(arrowstyle='->', color=O.GREEN, lw=1.4))
a1.annotate('choosing\nthe point:\n3.0 → 5.7 %', xy=(4.45, 8.5),
            xytext=(4.45, 21), fontsize=9.6, color=O.INK2, ha='center',
            arrowprops=dict(arrowstyle='->', color=O.INK2, lw=1.1))

# --- right: random-design spread, physics vs model-light
r = D[D.strategy == 'random']
NS = [3, 4, 5, 6, 8, 10, 12, 16, 20]
for a_ in ('lowrank', 'eb_gp', 'fem_gp'):
    s = r[r.rec == a_]
    med = [s[s.n == n].amp_map_pct.median() for n in NS]
    lo = [s[s.n == n].amp_map_pct.quantile(.25) for n in NS]
    hi = [s[s.n == n].amp_map_pct.quantile(.75) for n in NS]
    c = O.HUE[a_.split('_')[0]]
    a2.fill_between(NS, lo, hi, color=c, alpha=.18, lw=0, zorder=2)
    a2.plot(NS, med, ls='--' if a_.endswith('_gp') else '-', color=c, lw=2.0,
            marker=O.MK[a_], ms=5.2, mew=1.6, mec=O.WHITE, label=O.LBL[a_],
            zorder=4)
a2.set_xscale('log'); a2.set_yscale('log')
a2.set_xticks(NS); a2.set_yticks([2, 3, 5, 8, 12, 20, 30, 45])
import matplotlib.ticker as T
for A_ in (a2.get_xaxis(), a2.get_yaxis()):
    A_.set_major_formatter(T.FuncFormatter(lambda v, _: '%g' % v))
    A_.set_minor_locator(T.NullLocator())
a2.set_xlabel('positions measured  (n)')
a2.set_ylabel('held-out NRMSE  (%)')
a2.legend(fontsize=9.5, labelcolor=O.INK2, loc='lower left')
O.clean(a2)
O.title(a2, 'And is far less sensitive to a bad design')
a2.text(.985, .955, 'band = IQR over 24 random designs', transform=a2.transAxes,
        ha='right', va='top', fontsize=8.8, color=O.INK2, style='italic')

fig.savefig(str(FIG) + '/d_acq.png', dpi=170)
print('wrote fig/d_acq.png')

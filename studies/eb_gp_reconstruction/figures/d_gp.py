"""GP alone vs physics-informed: where the physics actually earns its keep."""
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

import numpy as np, pandas as pd, matplotlib, matplotlib.pyplot as plt
import matplotlib.ticker as T
import ornl as O

O.style()
D = pd.concat([pd.read_csv(str(OUT) + '/phys_bench.csv'),
               pd.read_csv(str(OUT) + '/fem_bench.csv'),
               pd.read_csv(str(OUT) + '/gp_bench.csv')], ignore_index=True)
E = D[D.strategy == 'equispaced']
PURPLE = O.MAGENTA
SER = [('lowrank', '3  low-rank', O.ORANGE, '-', 'o'),
       ('gp', 'GP only', PURPLE, '-', '^'),
       ('gp_oracle', 'GP only, oracle ℓ', PURPLE, ':', '^'),
       ('fem', '1b  FEM', O.GREEN, '-', 'D'),
       ('fem_gp', '1b  FEM + GP', O.GREEN, '--', 'D')]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 4.35))
fig.subplots_adjust(left=.055, right=.985, top=.90, bottom=.135, wspace=.20)

# ---------------- left: the error curves
for key, lab, c, ls, mk in SER:
    g = E[E.rec == key].dropna(subset=['amp_map_pct']).sort_values('n')
    a1.plot(g.n, g.amp_map_pct, ls=ls, color=c, lw=2.0, marker=mk, ms=5.2,
            mew=1.6, mec=O.WHITE, label=lab, zorder=4,
            alpha=.85 if key == 'gp_oracle' else 1.0)
a1.axvspan(2.8, 4.4, color=O.GREEN, alpha=.08, lw=0, zorder=1)
a1.text(3.05, 1.55, 'only here does\nbare physics win', color='#004D21',
        fontsize=9.4, fontweight='600', va='bottom')
a1.set_xscale('log'); a1.set_yscale('log')
a1.set_xticks([3, 4, 5, 6, 8, 12, 20, 50])
a1.set_yticks([1.5, 2, 3, 5, 8, 12, 20, 30, 45, 65])
for A_ in (a1.get_xaxis(), a1.get_yaxis()):
    A_.set_major_formatter(T.FuncFormatter(lambda v, _: '%g' % v))
    A_.set_minor_locator(T.NullLocator())
a1.set_xlabel('positions measured  (n)')
a1.set_ylabel('held-out complex-map NRMSE  (%)')
a1.legend(fontsize=9.4, labelcolor=O.INK2, loc='upper right')
O.clean(a1)
O.title(a1, 'GP alone is a much stronger baseline than low-rank')

# ---------------- right: the shrinking physics premium
piv = E.pivot_table(index='n', columns='rec', values='amp_map_pct')
ns = [n for n in piv.index if n >= 5]
ratio = (piv.loc[ns, 'gp'] / piv.loc[ns, 'fem_gp'])
b = a2.bar(range(len(ns)), ratio.values, .62, color=O.GREEN, alpha=.85,
           zorder=3)
a2.bar_label(b, fmt='%.2f×', fontsize=9.5, color=O.INK2, padding=3)
a2.axhline(1.0, color=O.INK2, lw=1.2, ls='--', zorder=2)
a2.set_xlim(-.62, len(ns) - .38)
a2.set_xticks(range(len(ns)))
a2.set_xticklabels([f'{n}' for n in ns])
a2.set_xlabel('positions measured  (n)')
a2.set_ylabel('GP-only error  ÷  FEM + GP error\n(1.0 = parity)')
a2.set_ylim(0, 2.9)
O.clean(a2); a2.grid(axis='x', visible=False)
O.title(a2, 'The physics premium shrinks as the GP gets data')
a2.annotate('by n = 20 the physics is worth\n1.19× — by n = 50, 1.14×',
            xy=(len(ns) - 1.0, .62), xytext=(2.5, 2.42), fontsize=9.5,
            color=O.INK2, ha='left',
            arrowprops=dict(arrowstyle='->', color=O.INK2, lw=1.1))
fig.savefig(str(FIG) + '/d_gp.png', dpi=170)
print('wrote fig/d_gp.png')

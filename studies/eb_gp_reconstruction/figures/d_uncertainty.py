"""Uncertainty from the data: where sigma lives, how it shrinks, is it honest.

sigma is the predictive sd of ONE REAL COMPONENT of Z (Re or Im), so a
calibrated arm has |Re(Zhat - Z)| <= sigma about 68 % of the time on withheld
positions.  Nothing entering sigma was fitted to the withheld data.
"""
import sys
import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_ROOT, 'config.py')):
    _ROOT = _os.path.dirname(_ROOT)
sys.path[:0] = [_ROOT, _os.path.join(_ROOT, 'src'),
                _os.path.join(_ROOT, 'figures')]
from config import (DENSE_GRID_A, FEM_LADDER, EB_REPO, EB_GEOMETRY, OUT, FIG,
                    add_eb_to_path)
add_eb_to_path()
import numpy as np, pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
from matplotlib.colors import LinearSegmentedColormap
import ornl as O

O.style()
R = pickle.load(open(str(OUT) + '/recs_cplx.pkl', 'rb'))
SW = pickle.load(open(str(OUT) + '/sweep_n.pkl', 'rb'))
x, f, Z, sel, TD = R['x'], R['f'], R['Z'], R['sel'], R['TD']
FK = f / 1e3
held = np.array([i for i in range(len(x)) if i not in set(sel)])
SEQ = LinearSegmentedColormap.from_list(
    'ornl_seq', ['#FFFFFF', '#B9D8C4', '#4E9E72', '#00662C', '#003320'])

fig = plt.figure(figsize=(13.2, 4.75))
a1 = fig.add_axes([.048, .155, .205, .645])
a2 = fig.add_axes([.385, .155, .245, .645])
a3 = fig.add_axes([.730, .155, .245, .645])

# ------------------------------------------- (a) fractional sigma at n = 8
# sigma relative to the arm's OWN prediction: leakage-free, and it answers the
# question the absolute map cannot -- "how many digits of this pixel do I
# believe?"  >0 dB means the error bar is as big as the value.
S = R['fem_gp']['sd']
frac = 20 * np.log10(np.maximum(S, 1e-30) / np.maximum(np.abs(R['fem_gp']['Zrec']), 1e-30))
im = a1.pcolormesh(x, FK, frac.T, cmap=SEQ, vmin=-40, vmax=10,
                   shading='nearest', rasterized=True)
a1.axvline(TD, color=O.MAGENTA, lw=1.1, ls='--')
for i in sel:
    a1.plot([x[i]], [FK[0]], marker='^', ms=5, color=O.MAGENTA, clip_on=False,
            zorder=9)
a1.set_xlabel('position from clamp  (µm)', fontsize=9.6)
a1.set_ylabel('frequency  (kHz)', fontsize=9.6)
a1.set_xticks([140, 180, 220])
O.clean(a1, grid=False)
cax = fig.add_axes([.260, .155, .009, .645])
cb = fig.colorbar(im, cax=cax, ticks=[-40, -30, -20, -10, 0, 10])
cb.set_label(r'$\sigma\,/\,|\hat{Z}|$   (dB)', fontsize=9.0, labelpad=2)
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8.5)
a1.set_title('Where the map is trustworthy', loc='left', fontweight='600',
             pad=19, color=O.INK)
a1.text(0, 1.012, 'FEM + GP, 8 revealed positions (▲)', transform=a1.transAxes,
        fontsize=9.0, color=O.INK2, va='bottom')
a1.annotate('on the antiresonance trough and at\nthe node, σ is the size of '
            'the signal', xy=(207, 335), xytext=(125, 408), fontsize=8.8,
            color=O.INK, ha='left', fontweight='600',
            arrowprops=dict(arrowstyle='->', color=O.INK, lw=1.1))

# ------------------------------------------- (b) sigma and error vs n
ns = sorted({r['n'] for r in SW})
def col(key, fld):
    return np.array([[r[fld] for r in SW if r['rec'] == key and r['n'] == n][0]
                     for n in ns])
ARMS = (('gp', 'GP only', O.MAGENTA), ('eb_gp', 'EB + GP', O.BLUE),
        ('fem_gp', 'FEM + GP', O.GREEN))
for key, lab, c in ARMS:
    a2.plot(ns, col(key, 'sd'), '-', color=c, lw=2.0, marker='o', ms=5,
            mec=O.WHITE, mew=1.3, label=f'{lab}  —  predicted σ')
    a2.plot(ns, col(key, 'err'), '--', color=c, lw=1.4, marker='s', ms=4.2,
            mec=O.WHITE, mew=1.1, alpha=.85, label=f'{lab}  —  actual error')
a2.set_xscale('log'); a2.set_yscale('log')
a2.set_xticks(ns); a2.set_xticklabels([str(n) for n in ns])
a2.get_xaxis().set_minor_locator(mt.NullLocator())
a2.set_xlabel('positions measured  (n)')
a2.set_ylabel(f'σ and |error| at {FK[int(np.argmax(np.abs(Z).max(0)))]:.1f} kHz')
a2.legend(fontsize=7.8, loc='upper right', labelcolor=O.INK2, ncol=1)
O.clean(a2)
a2.set_title('Both shrink — and the error bar tracks the error', loc='left',
             fontweight='600', pad=19, color=O.INK)
a2.text(0, 1.012, 'medians over the withheld positions', fontsize=9.0,
        color=O.INK2, transform=a2.transAxes, va='bottom')

# ------------------------------------------- (c) is sigma honest?
a3.axhspan(0.8, 1.25, color=O.GREEN, alpha=.11, lw=0, zorder=1)
a3.axhline(1.0, color=O.GREEN, lw=1.4, zorder=2)
a3.text(4.35, 1.04, 'calibrated', fontsize=9.2, color='#004D21',
        fontweight='600', va='bottom')
for key, lab, c in ARMS:
    v = col(key, 'z_rms')
    keep = v < 5
    a3.plot(np.array(ns)[keep], v[keep], '-', color=c, lw=2.0, marker='o',
            ms=5.5, mec=O.WHITE, mew=1.3, label=lab, zorder=5)
for c in (O.BLUE, O.GREEN):
    a3.plot([4], [3.42], marker='^', ms=8, color=c, mec=O.WHITE, mew=1.3,
            clip_on=False, zorder=6)
a3.annotate('n = 4: no discrepancy GP is fitted at all,\nso σ is the parameter '
            'term alone — wrong\nby 16× (EB) and 47× (FEM)',
            xy=(4.06, 3.30), xytext=(5.0, 2.35), fontsize=8.6, color=O.INK2,
            ha='left', arrowprops=dict(arrowstyle='->', color=O.INK2, lw=1.1))
a3.set_xscale('log')
a3.set_xticks(ns); a3.set_xticklabels([str(n) for n in ns])
a3.get_xaxis().set_minor_locator(mt.NullLocator())
a3.set_xlim(3.7, 33); a3.set_ylim(0, 3.6)
a3.set_xlabel('positions measured  (n)')
a3.set_ylabel('rms standardised residual  z')
a3.text(.03, .45, 'σ too small\n(over-confident)', transform=a3.transAxes,
        fontsize=9.0, color=O.INK2, va='bottom')
a3.text(.03, .015, 'σ too large\n(over-cautious)', transform=a3.transAxes,
        fontsize=9.0, color=O.INK2, va='bottom')
a3.legend(fontsize=9.4, loc='upper right', labelcolor=O.INK2)
O.clean(a3)
a3.set_title('Are the error bars honest?', loc='left', fontweight='600',
             pad=19, color=O.INK)
a3.text(0, 1.012, 'z = 1 means the stated σ is the actual error scale',
        transform=a3.transAxes, fontsize=9.0, color=O.INK2, va='bottom')
fig.savefig(f'{FIG}/d5_uncert.png', dpi=170)
plt.close(fig)

for k in ('gp', 'eb_gp', 'fem_gp'):
    m = R[k + '_m']
    print(f"n=8 {k:8s} z_rms {m['z_rms']:.2f}  1σ {m['cov1s']:.3f} (0.683)  "
          f"2σ {m['cov2s']:.3f} (0.954)  3σ {m['cov3s']:.3f} (0.997)")
print('wrote d5_uncert.png')

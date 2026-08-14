"""Deck figures: the reconstructed complex spectrum, amplitude and phase.

Colour jobs, assigned before any colour was chosen:
  log-amplitude   magnitude  -> SEQUENTIAL, one hue (ORNL green), light -> dark
  phase           CYCLIC     -> a cyclic map (twilight); a sequential or
                               diverging ramp would put a false discontinuity at
                               the ±180° wrap, which is exactly where the node
                               signature lives
  arms            identity   -> the study's existing convention, hue = forward
                               model (blue EB, green FEM, magenta GP-only) and
                               line style = ± GP, so identity is never carried
                               by colour alone
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
from matplotlib.colors import LinearSegmentedColormap
import ornl as O

O.style()
R = pickle.load(open(str(OUT) + '/recs_cplx.pkl', 'rb'))
x, f, Z, sel, TD, ir = R['x'], R['f'], R['Z'], R['sel'], R['TD'], R['ir']
FK = f / 1e3
AMAX = np.abs(Z).max()

SEQ = LinearSegmentedColormap.from_list(
    'ornl_seq', ['#FFFFFF', '#B9D8C4', '#4E9E72', '#00662C', '#003320'])
CYC = plt.get_cmap('twilight_shifted')

# map row: truth, then the model-light arm, then each forward model bare and +GP
MAPS = [(None, 'Ground truth', 'every position measured'),
        ('gp', 'GP only', None),
        ('eb', 'EB alone', None),
        ('eb_gp', 'EB + GP', None),
        ('fem', 'FEM alone', None),
        ('fem_gp', 'FEM + GP', None)]
# line panels: hue = forward model, dashed = + GP
LINES = [('eb', 'EB', O.BLUE, '-'), ('eb_gp', 'EB + GP', O.BLUE, '--'),
         ('fem', 'FEM', O.GREEN, '-'), ('fem_gp', 'FEM + GP', O.GREEN, '--'),
         ('gp', 'GP only', O.MAGENTA, '-')]
dB = lambda M: 20 * np.log10(np.maximum(np.abs(M), 1e-16) / AMAX)
ip = 63                                   # a withheld position, mid-span


def mapfmt(ax, first=False):
    ax.set_xlabel('position  (µm)', fontsize=9.0)
    if first:
        ax.set_ylabel('frequency  (kHz)', fontsize=9.4)
    else:
        ax.set_yticklabels([])
    O.clean(ax, grid=False)
    ax.set_xticks([140, 190, 220])
    ax.tick_params(labelsize=8.6)
    for i in sel:
        ax.plot([x[i]], [FK[0]], marker='^', ms=4.2, color=O.MAGENTA,
                clip_on=False, zorder=9)


def maprow(fig, kind):
    """Six panels across, shared colour bar. kind = 'amp' | 'phase'."""
    axs = [fig.add_axes([.044 + .1495 * k, .545, .118, .335]) for k in range(6)]
    for k, (ax, (key, lab, sub)) in enumerate(zip(axs, MAPS)):
        M = Z if key is None else R[key]['Zrec']
        if kind == 'amp':
            im = ax.pcolormesh(x, FK, dB(M).T, cmap=SEQ, vmin=-60, vmax=0,
                               shading='nearest', rasterized=True)
            ax.axvline(TD, color=O.MAGENTA, lw=1.0, ls='--')
            txt = sub or f"NRMSE {R[key + '_m']['nrmse_amp']:.2f} %"
        else:
            im = ax.pcolormesh(x, FK, np.degrees(np.angle(M)).T, cmap=CYC,
                               vmin=-180, vmax=180, shading='nearest',
                               rasterized=True)
            ax.axvline(TD, color=O.WHITE, lw=1.1, ls='--')
            txt = sub or f"{R[key + '_m']['phase_mae_deg']:.1f}°  ·  " \
                         f"{R[key + '_m']['nrmse_cplx']:.0f} %"
        mapfmt(ax, first=(k == 0))
        ax.set_title(lab, loc='left', fontsize=10.0, fontweight='600',
                     color=O.INK, pad=17)
        ax.text(0, 1.015, txt, transform=ax.transAxes, fontsize=8.6,
                color=O.INK2, va='bottom')
    cax = fig.add_axes([.951, .545, .0085, .335])
    cb = fig.colorbar(im, cax=cax,
                      ticks=None if kind == 'amp' else [-180, -90, 0, 90, 180])
    cb.set_label('|Z|  (dB re max)' if kind == 'amp' else 'phase  (deg)',
                 fontsize=9.0, labelpad=2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8.4)


# ===================================================== 1. amplitude =========
fig = plt.figure(figsize=(13.2, 6.55))
maprow(fig, 'amp')

a1 = fig.add_axes([.050, .085, .40, .335])
a1.plot(FK, dB(Z[ip]), '-', color=O.INK, lw=3.2, alpha=.28, label='measured',
        solid_capstyle='round')
for key, lab, c, ls in LINES:
    a1.plot(FK, dB(R[key]['Zrec'][ip]), ls=ls, color=c, lw=1.4, label=lab)
a1.set_xlim(FK[0], FK[-1]); a1.set_ylim(-72, 3)
a1.set_xlabel('frequency  (kHz)'); a1.set_ylabel('|Z|  (dB re max)')
a1.legend(fontsize=8.6, loc='upper right', labelcolor=O.INK2, ncol=3,
          columnspacing=1.0)
a1.annotate('both bare forward models miss the\nantiresonance; both + GP find it',
            xy=(400, -63), xytext=(408, -30), fontsize=9.2, color=O.INK2,
            ha='right', arrowprops=dict(arrowstyle='->', color=O.INK2, lw=1.1))
O.clean(a1)
O.title(a1, f'A withheld position:  x = {x[ip]:.1f} µm')

a2 = fig.add_axes([.555, .085, .40, .335])
t = np.abs(Z).max(1)
a2.plot(x, t / t.max(), '-', color=O.INK, lw=3.2, alpha=.28, label='measured',
        solid_capstyle='round')
for key, lab, c, ls in LINES:
    a2.plot(x, np.abs(R[key]['Zrec']).max(1) / t.max(), ls=ls, color=c, lw=1.4,
            label=lab)
a2.plot(x[sel], (t / t.max())[sel], 'o', ms=7, mfc=O.WHITE, mec=O.MAGENTA,
        mew=1.8, label='revealed (8)', zorder=6)
a2.axvline(TD, color=O.MAGENTA, lw=1.2, ls=':')
a2.set_yscale('log'); a2.set_ylim(1.2e-2, 2.8)
a2.set_xlabel('position from clamp  (µm)')
a2.set_ylabel('peak |Z| in band  (norm.)')
a2.legend(fontsize=8.6, loc='lower left', labelcolor=O.INK2, ncol=2,
          columnspacing=1.0)
O.clean(a2)
O.title(a2, 'Mode shape along the lever, and the detection node')
fig.savefig(f'{FIG}/d1_amp.png', dpi=170)
plt.close(fig)

# ===================================================== 2. phase =============
fig = plt.figure(figsize=(13.2, 6.55))
maprow(fig, 'phase')

a1 = fig.add_axes([.050, .085, .40, .335])
mw = FK <= 395           # past this |Z| is at the floor and the phase is random
a1.plot(FK[mw], np.unwrap(np.angle(Z[ip][mw])) * 180 / np.pi, '-', color=O.INK,
        lw=3.2, alpha=.28, label='measured', solid_capstyle='round')
for key, lab, c, ls in LINES:
    a1.plot(FK[mw], np.unwrap(np.angle(R[key]['Zrec'][ip][mw])) * 180 / np.pi,
            ls=ls, color=c, lw=1.4, label=lab)
a1.set_xlim(FK[0], 395); a1.set_ylim(-250, 430)
a1.set_xlabel('frequency  (kHz)')
a1.set_ylabel('unwrapped phase  (deg)')
a1.legend(fontsize=8.6, loc='upper left', labelcolor=O.INK2, ncol=3,
          columnspacing=1.0)
a1.annotate('EB and FEM alone both take the branch\nthe other way round — the '
            'same failure,\nnot a quirk of one geometry',
            xy=(330, -114), xytext=(272, -218), fontsize=9.2, color=O.INK2,
            ha='left', arrowprops=dict(arrowstyle='->', color=O.INK2, lw=1.1))
O.clean(a1)
O.title(a1, f'Phase through the resonance, withheld x = {x[ip]:.1f} µm')

a2 = fig.add_axes([.555, .085, .40, .335])
kres = int(np.argmax(np.abs(Z).max(0)))
a2.plot(x, np.degrees(np.angle(Z[:, kres])), '-', color=O.INK, lw=3.2,
        alpha=.28, label='measured', solid_capstyle='round')
for key, lab, c, ls in LINES:
    Zr = R[key]['Zrec'][:, kres]
    rot = np.exp(-1j * np.angle(np.sum(Zr[sel] * np.conj(Z[sel, kres]))))
    a2.plot(x, np.degrees(np.angle(Zr * rot)), ls=ls, color=c, lw=1.4,
            label=lab)
a2.axvline(TD, color=O.MAGENTA, lw=1.2, ls='--')
a2.annotate('the node is a 180° phase reversal —\nthe mode shape changing sign',
            xy=(TD + 0.6, -60), xytext=(TD - 8, -155), fontsize=9.2,
            color=O.INK2, ha='right',
            arrowprops=dict(arrowstyle='->', color=O.INK2, lw=1.1))
a2.set_xlabel('position from clamp  (µm)')
a2.set_ylabel('phase at resonance  (deg)')
a2.set_ylim(-190, 235); a2.set_yticks([-180, -90, 0, 90, 180])
a2.legend(fontsize=8.6, loc='upper left', labelcolor=O.INK2, ncol=3,
          columnspacing=1.0)
O.clean(a2)
O.title(a2, f'Phase across the beam at {FK[kres]:.1f} kHz')
fig.savefig(f'{FIG}/d2_phase.png', dpi=170)
plt.close(fig)

print('wrote d1_amp.png, d2_phase.png')
for k in ('gp', 'eb', 'eb_gp', 'fem', 'fem_gp'):
    m = R[k + '_m']
    print(f"{k:8s} amp {m['nrmse_amp']:6.2f} %  phase {m['phase_mae_deg']:6.1f}°"
          f"  cplx {m['nrmse_cplx']:7.2f} %  D-NS {m['dns']:7.2f} µm "
          f"(err {abs(m['dns']-TD):.2f})")

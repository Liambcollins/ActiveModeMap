"""Wide-band verdict figure: what each arm can even reach, and how well.

Panel A  the measured spectrum and the DOMAIN of each arm -- the physics arms
         exist only where their library covers u = f/f_res
Panel B  median |dB| error vs frequency at n = 8, per arm, masked to coverage
Panel C  the tail metric -- % of signal bins within 3 dB vs n
"""
import sys, gc, os
import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_ROOT, 'config.py')):
    _ROOT = _os.path.dirname(_ROOT)
sys.path[:0] = [_ROOT, _os.path.join(_ROOT, 'src'),
                _os.path.join(_ROOT, 'figures')]
from config import (DENSE_GRID_A, FEM_LADDER, EB_REPO, EB_GEOMETRY, OUT, FIG,
                    add_eb_to_path)
add_eb_to_path()
import numpy as np, pandas as pd, pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import cplxrec as CR, physrec as PR
import ornl as O

O.style()
M = pickle.load(open(str(OUT) + '/wideband_meta.pkl', 'rb'))
W = pd.read_csv(str(OUT) + '/wideband.csv')
x, f, Z, FLOOR, SELS = M['x'], M['f'], M['Z'], M['FLOOR'], M['SELS']
A = np.abs(Z)
FK = f / 1e3
SIG = np.median(A, axis=0) > 3.0 * FLOOR
N0 = 8
LIBS = [('eb_gp', 'EB + GP  (band-A library)', O.BLUE, '-'),
        ('fem_gp', 'FEM + GP  (band-A library)', O.GREEN, '-'),
        ('eb2_gp', 'EB + GP  (two-band library)', '#00456B', '--')]

# ---------------------------------------------------- per-frequency error at n=8
CUR = str(OUT) + '/wide_curves.pkl'
if os.path.exists(CUR):
    C = pickle.load(open(CUR, 'rb'))
else:
    sel = SELS[N0]
    held = np.array([i for i in range(len(x)) if i not in set(sel)])
    F = PR.noise_floor(A[sel])
    def dbe(o):
        return np.median(np.abs(20 * np.log10(
            (np.abs(o['Zrec'])[held] + F) / (A[held] + F))), axis=0)
    C = {}
    o = CR.rec_gp_cplx(x, sel, Z[sel], f)
    C['gp'] = dict(e=dbe(o), cov=np.ones(len(f), bool))
    for key, lib in (('eb_gp', 'eblib_cplx.npz'),
                     ('fem_gp', 'femlib_isotropic_cplx.npz'),
                     ('eb2_gp', 'eblib_x_cplx.npz')):
        CR.load(str(OUT) + '/' + lib)
        if key == 'eb2_gp':
            o = CR.rec_twoband_cplx(x, sel, Z[sel], f)
        else:
            o = CR.rec_phys_cplx(x, sel, Z[sel], f, loo=False,
                                 fit_band=(265e3, 500e3))
        C[key] = dict(e=dbe(o), cov=CR.coverage(f, o['theta']['f_res']),
                      f_res=o['theta']['f_res'], k1=o['theta']['k1'])
        CR._LIB.clear(); gc.collect()
    pickle.dump(C, open(CUR, 'wb'))

fig = plt.figure(figsize=(13.2, 5.05))
a1 = fig.add_axes([.052, .585, .60, .335])
a2 = fig.add_axes([.052, .145, .60, .355])
a3 = fig.add_axes([.735, .145, .238, .775])

# ------------------------------------------------------------------ panel A
pm = A.max(0)
a1.plot(FK, 20 * np.log10(pm / pm.max()), '-', color=O.INK, lw=1.1, alpha=.75)
a1.axhline(20 * np.log10(3 * FLOOR / pm.max()), color=O.MUT, lw=1.0, ls=':')
a1.text(60, 20 * np.log10(3 * FLOOR / pm.max()) - 5.0, '3× noise floor',
        fontsize=8.6, color=O.MUT, ha='left')
for nm, lo, hi in [('mode 1', 285, 302), ('mode 2', 880, 925)]:
    a1.axvspan(lo, hi, color=O.MAGENTA, alpha=.13, lw=0)
    a1.text((lo + hi) / 2, 3, nm, fontsize=9.0, color=O.MAGENTA, ha='center',
            va='bottom', fontweight='600')
a1.set_ylim(-46, 12); a1.set_xlim(FK[0], FK[-1])
a1.set_ylabel('max over x, |Z| (dB)', fontsize=9.6)
a1.set_xticklabels([])
O.clean(a1)
O.title(a1, 'The measured band — and what each arm can reach')
# coverage bars
yb = [-16, -25, -34]
for (key, lab, c, ls), y in zip(LIBS, yb):
    cov = C[key]['cov']
    lo, hi = FK[cov][0], FK[cov][-1]
    a1.plot([lo, hi], [y, y], '-', color=c, lw=5.0, solid_capstyle='butt',
            alpha=.9)
    a1.text(hi + 18, y, f'{lab}   {100*cov.mean():.0f} % of the band',
            fontsize=8.8, color=c, va='center')
a1.plot([FK[0], FK[-1]], [-7, -7], '-', color=O.MAGENTA, lw=5.0,
        solid_capstyle='butt', alpha=.9)
a1.text(FK[-1] - 20, -7, 'GP only   100 %', fontsize=8.8, color=O.MAGENTA,
        va='center', ha='right')

# ------------------------------------------------------------------ panel B
def binned(e, cov, nb=110):
    """Median error per frequency bin over ALL bins in coverage.

    Restricting this panel to signal bins would leave it almost empty -- only
    ~140 of 4000 bins clear 3x the floor, in two narrow clusters. So the curve
    runs everywhere and the sub-threshold span is shaded instead: read the
    shaded region as 'both arms are matching noise here'.
    """
    edges = np.linspace(FK[0], FK[-1], nb + 1)
    j = np.clip(np.digitize(FK, edges) - 1, 0, nb - 1)
    out = np.full(nb, np.nan)
    for b in range(nb):
        m = (j == b) & cov
        if m.sum() >= 2:
            out[b] = np.median(e[m])
    return .5 * (edges[:-1] + edges[1:]), out


# shade every frequency span whose median |Z| is below 3x the floor
d = np.diff(np.concatenate([[0], SIG.view(np.int8), [0]]))
for lo, hi in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
    a2.axvspan(FK[max(lo - 1, 0)], FK[min(hi, len(FK) - 1)],
               color=O.MAGENTA, alpha=.08, lw=0, zorder=0)

fc, eg = binned(C['gp']['e'], C['gp']['cov'])
a2.plot(fc, eg, '-', color=O.MAGENTA, lw=2.0, label='GP only')
for key, lab, c, ls in LIBS:
    fc2, e2 = binned(C[key]['e'], C[key]['cov'])
    a2.plot(fc2, e2, ls=ls, color=c, lw=1.6, label=lab.split('  (')[0] +
            (' (2-band)' if key == 'eb2_gp' else ''))
for key, lab, c, ls in LIBS:
    hi = FK[C[key]['cov']][-1]
    a2.axvline(hi, color=c, lw=1.0, ls=':', alpha=.8)
a2.set_yscale('log')
a2.set_xlim(FK[0], FK[-1])
a2.set_xlabel('frequency  (kHz)')
a2.set_ylabel('median |error|  (dB)')
a2.legend(fontsize=8.6, loc='lower right', labelcolor=O.INK2, ncol=2,
          columnspacing=1.0)
a2.text(.995, .965, 'dotted = coverage edge   ·   tinted = above 3× the noise '
        'floor', transform=a2.transAxes, fontsize=8.6, color=O.INK2,
        ha='right', va='top')
O.clean(a2)
O.title(a2, f'Where each arm is wrong, n = {N0}')

# ------------------------------------------------------------------ panel C
for key, lab, c, mk in (('gp', 'GP only', O.MAGENTA, 'o'),
                        ('gp@eb_gp', 'GP only, band-A mask', O.MAGENTA, 's'),
                        ('eb_gp', 'EB + GP', O.BLUE, 'o'),
                        ('fem_gp', 'FEM + GP', O.GREEN, 'o'),
                        ('eb2_gp', 'EB + GP (2-band)', '#00456B', 'D')):
    g = W[W.rec == key].sort_values('n')
    if not len(g):
        continue
    a3.plot(g.n, 100 * g.sig_frac3dB, '-' if '@' not in key else '--',
            color=c, lw=2.0, marker=mk, ms=5.5, mec=O.WHITE, mew=1.3,
            label=lab, zorder=4)
a3.set_xscale('log')
ns = sorted(W.n.unique())
a3.set_xticks(ns); a3.set_xticklabels([str(int(v)) for v in ns])
a3.get_xaxis().set_minor_locator(mt.NullLocator())
a3.set_xlabel('positions measured  (n)')
a3.set_ylabel('signal bins within 3 dB  (%)')
a3.set_ylim(95.8, 100.0)
a3.legend(fontsize=8.6, loc='lower right', labelcolor=O.INK2)
O.clean(a3)
O.title(a3, 'The tail, not the median')
fig.savefig(f'{FIG}/d6_wideband.png', dpi=170)
plt.close(fig)

print(f"{'arm':>22} {'cov%':>6} {'sig dB':>7} {'<=3dB%':>7} {'mode2 dB':>9} {'k1':>7}")
for key in ('gp', 'gp@eb_gp', 'eb_gp', 'fem_gp', 'gp@eb2_gp', 'eb2_gp'):
    g = W[(W.rec == key) & (W.n == N0)]
    if not len(g):
        continue
    r = g.iloc[0]
    m2 = r['mode 2 ridge']
    print(f"{key:>22} {100*r.cov_frac:6.1f} {r.sig:7.2f} "
          f"{100*r.sig_frac3dB:7.1f} {m2 if np.isfinite(m2) else float('nan'):9.2f} "
          f"{r.k1 if np.isfinite(r.k1) else float('nan'):7.1f}")
print('wrote d6_wideband.png')

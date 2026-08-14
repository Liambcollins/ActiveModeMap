"""Dual-peak (two-band) fit: identifiability, stability, and the trade-off.

Colour: hue stays the forward model (EB blue, FEM green) as everywhere else in
the deck.  The three EB FIT STRATEGIES are an ORDERED set -- band A, band A+B,
band A+B plus the GP, in increasing information -- so they get a lightness ramp
of the EB hue rather than three new hues, and every bar carries its value as a
direct label.  (ORNL accent1 #00454D, the obvious third categorical hue, fails
the dataviz validator's chroma floor: it reads grey.)
"""
import sys, glob, re, os
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
import ornl as O

O.style()
RES = str(OUT)
LAD = str(FEM_LADDER)
D = pd.read_csv(str(OUT) + '/dualpeak.csv')

F1_MEAS, F2_MEAS = 293.08e3, 902.52e3
R_MEAS = F2_MEAS / F1_MEAS
HERTZ = 1310.0
INDEP = (1000.0, 2154.0)

# EB blue, stepped by information content
EB3 = ['#9DCBE2', '#2E8BC0', '#00456B']
ARMS = [('A only', 'band A only', EB3[0]),
        ('A+B, per-band', 'A + B, per-band gain', EB3[1]),
        ('A+B, per-band +GP', 'A + B, per-band + GP', EB3[2])]

# ---------------------------------------------------------------- EB ratio curve
eb = np.load(RES + '/eb_f2f1.npy')
k_eb, r_eb = eb[:, 0], eb[:, 2] / eb[:, 1]
K_RATIO = float(np.interp(R_MEAS, r_eb, k_eb))

# ---------------------------------------------------------------- FEM ratio curve
CACHE = str(OUT) + '/fem_f2f1.npy'
if os.path.exists(CACHE):
    fem = np.load(CACHE)
else:
    import h5py
    rows = []
    for p in sorted(glob.glob(LAD + '/fem_Multi75G_isotropic_mech_k1_*_ld4.78.h5')):
        kk = float(re.search(r'k1_([0-9.]+)_ld', p).group(1))
        with h5py.File(p, 'r') as h:
            eig = h['eigen/frequencies_hz'][:]
            mtp = h['eigen/mode_types'][:] if 'eigen/mode_types' in h else None
        fl = np.sort((eig[mtp == 1] if mtp is not None else eig))
        fl = fl[fl > 50e3]
        if len(fl) >= 2:
            rows.append((kk, fl[0], fl[1]))
    fem = np.array(sorted(rows))
    np.save(CACHE, fem)
k_fem, r_fem = fem[:, 0], fem[:, 2] / fem[:, 1]

fig = plt.figure(figsize=(13.2, 4.55))
a1 = fig.add_axes([.050, .165, .262, .70])
a2 = fig.add_axes([.400, .165, .245, .70])
a3 = fig.add_axes([.732, .165, .243, .70])

# ------------------------------------------------------------------- panel A
a1.axhline(R_MEAS, color=O.MAGENTA, lw=1.6, ls='--', zorder=3)
a1.text(7600, R_MEAS + .035, f'measured  {R_MEAS:.3f}', color=O.MAGENTA,
        fontsize=9.4, fontweight='600', va='bottom', ha='right')
a1.axvspan(*INDEP, color=O.GREEN, alpha=.10, lw=0, zorder=1)
a1.plot(k_eb, r_eb, '-', color=O.BLUE, lw=2.2, label='EB  (2-segment, k₂ = 0)')
a1.plot(k_fem, r_fem, '-', color=O.GREEN, lw=2.2, label='FEM  (isotropic)')
a1.plot([K_RATIO], [R_MEAS], 'o', ms=10, mfc=O.WHITE, mec=O.BLUE, mew=2.2,
        zorder=6)
a1.annotate(f'EB reaches it at\nk₁ = {K_RATIO:.0f} N/m', xy=(K_RATIO, R_MEAS),
            xytext=(1500, 2.62), fontsize=9.4, color=O.BLUE, ha='left',
            fontweight='600',
            arrowprops=dict(arrowstyle='->', color=O.BLUE, lw=1.2))
a1.annotate(f'FEM saturates at {r_fem.max():.2f} —\nno k₁ gets it there',
            xy=(4600, r_fem.max() - .01), xytext=(1900, 2.44), fontsize=9.2,
            color=O.GREEN, ha='center', va='top',
            arrowprops=dict(arrowstyle='->', color=O.GREEN, lw=1.1))
a1.axvline(HERTZ, color=O.INK, lw=1.1, ls=':', zorder=2)
a1.text(HERTZ * 1.07, 2.24, 'Hertz 1310', color=O.INK, fontsize=9.0,
        va='center', rotation=90)
a1.text(1460, 2.03, 'independent estimates', color='#004D21',
        fontsize=8.8, ha='center')
a1.set_xscale('log'); a1.set_xlim(150, 8000); a1.set_ylim(2.0, 3.35)
a1.set_xticks([200, 500, 1000, 2000, 5000])
a1.get_xaxis().set_major_formatter(mt.FuncFormatter(lambda v, _: '%g' % v))
a1.get_xaxis().set_minor_locator(mt.NullLocator())
a1.set_xlabel('contact stiffness k₁  (N/m)')
a1.set_ylabel('f₂ / f₁')
a1.legend(fontsize=9.0, loc='upper left', labelcolor=O.INK2)
O.clean(a1)
O.title(a1, 'The mode ratio is what identifies k₁')

# ------------------------------------------------------------------- panel B
a2.axvspan(0, 100, color=O.WHITE, alpha=0, lw=0)
a2.axhspan(*INDEP, color=O.GREEN, alpha=.10, lw=0, zorder=1)
a2.axhline(K_RATIO, color=O.MAGENTA, lw=1.5, ls='--', zorder=3)
a2.text(30, K_RATIO * 1.05, f'ratio-implied {K_RATIO:.0f}', color=O.MAGENTA,
        fontsize=9.0, ha='right', va='bottom', fontweight='600')
a2.axhline(HERTZ, color=O.INK, lw=1.0, ls=':', zorder=2)
a2.text(3.1, HERTZ * 1.04, 'Hertz 1310', color=O.INK, fontsize=8.8, va='bottom')
for key, lab, c in ARMS[:2]:
    g = D[D.arm == key].sort_values('n')
    a2.plot(g.n, g.k1, '-', color=c, lw=2.2, marker='o', ms=5.5, mec=O.WHITE,
            mew=1.3, label=lab, zorder=5)
g = D[D.arm == 'A+B, 1 gain'].sort_values('n')
a2.plot(g.n, g.k1, ':', color=O.MUT, lw=1.6, marker='^', ms=5, mec=O.WHITE,
        mew=1.1, label='A + B, one gain', zorder=4)
a2.set_xscale('log'); a2.set_yscale('log')
ns = sorted(D.n.unique())
a2.set_xticks(ns); a2.set_xticklabels([str(int(v)) for v in ns])
a2.get_xaxis().set_minor_locator(mt.NullLocator())
a2.set_yticks([300, 500, 1000, 2000])
a2.get_yaxis().set_major_formatter(mt.FuncFormatter(lambda v, _: '%g' % v))
a2.get_yaxis().set_minor_locator(mt.NullLocator())
a2.set_xlim(2.7, 34); a2.set_ylim(260, 2400)
a2.set_xlabel('positions measured  (n)')
a2.set_ylabel('fitted k₁  (N/m)')
a2.legend(fontsize=8.8, loc='lower left', labelcolor=O.INK2)
pb = D[D.arm == 'A+B, per-band']
a2.annotate(f'{pb.k1.mean():.0f} ± {pb.k1.std(ddof=1):.0f} N/m at every n',
            xy=(10, pb.k1.mean()), xytext=(3.4, 1750), fontsize=9.2,
            color=EB3[1], ha='left', fontweight='600',
            arrowprops=dict(arrowstyle='->', color=EB3[1], lw=1.1))
O.clean(a2)
O.title(a2, 'Add band B and k₁ stops drifting')

# ------------------------------------------------------------------- panel C
regs = [('m1', 'mode-1\nridge'), ('m2', 'mode-2\nridge'),
        ('anti', 'anti-\nresonance')]
w = 0.26
for i, (key, lab, c) in enumerate(ARMS):
    g = D[(D.arm == key) & (D.n == 8)].iloc[0]
    v = [g[r] for r, _ in regs]
    b = a3.bar(np.arange(3) + (i - 1) * w, v, w * .92, color=c, label=lab,
               zorder=3, edgecolor=O.WHITE, linewidth=1.2)
    a3.bar_label(b, fmt='%.2f', fontsize=8.4, color=O.INK2, padding=2)
a3.set_yscale('log'); a3.set_ylim(0.09, 320)
a3.set_xticks(range(3)); a3.set_xticklabels([l for _, l in regs], fontsize=9.2)
a3.set_ylabel('median |error| at n = 8  (dB)')
a3.legend(fontsize=8.6, loc='upper left', labelcolor=O.INK2)
O.clean(a3); a3.grid(axis='x', visible=False)
O.title(a3, 'Both peaks at once, once the GP is on')
fig.savefig(f'{FIG}/d7_dualpeak.png', dpi=170)
plt.close(fig)

print(f'EB reaches f2/f1 = {R_MEAS:.4f} at k1 = {K_RATIO:.1f} N/m')
print(f'FEM isotropic saturates at {r_fem.max():.4f} (k1 = {k_fem[-1]:.0f})')
print(f'per-band fit: k1 = {pb.k1.mean():.1f} +/- {pb.k1.std(ddof=1):.1f} N/m, '
      f'model ratio {pb.f2_over_f1.mean():.4f}, '
      f'gain ratio {pb.gain_ratio_dB.mean():.2f} dB')
print('wrote d7_dualpeak.png')

"""Slide figure: the higher eigenmode is the real discrepancy, and it is where
the contact stiffness information lives."""
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

import numpy as np, pickle, h5py, glob, re, matplotlib.pyplot as plt
import matplotlib.ticker as T
import ornl as O
from run_phys import prep, truth_dns

O.style()
MEAS = 902.41 / 292.97                      # measured f2/f1
DD = str(FEM_LADDER)+'/'

# --- EB curve, computed with the repo solver over 200 kHz - 1.6 MHz
r = np.load(str(OUT) + '/eb_f2f1.npy')
ok = np.isfinite(r[:, 2])
kE, f1E, f2E = r[ok, 0], r[ok, 1], r[ok, 2]
ratE = f2E / f1E
k_imp = float(np.exp(np.interp(MEAS, ratE, np.log(kE))))

# --- FEM curves, straight from the eigen tables in the exports
def fem_ratio(pattern):
    out = []
    for p in glob.glob(DD + pattern):
        k1 = float(re.search(r'k1_([\d.]+)_', p).group(1))
        with h5py.File(p, 'r') as f:
            e = f['eigen/frequencies_hz'][:]; mt = f['eigen/mode_types'][:]
        fx = e[mt == 1]
        out.append((k1, fx[1] / fx[0]))
    return np.array(sorted(out))
FR = fem_ratio('fem_Multi75G_frictionless_mech_k1_*_ld4.78.h5')
IS = fem_ratio('fem_Multi75G_isotropic_mech_k1_*_ld4.78.h5')

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 4.35))
fig.subplots_adjust(left=.058, right=.985, top=.90, bottom=.135, wspace=.21)

# ================= left: f2/f1 vs k1 =======================================
a1.axhline(MEAS, color=O.MAGENTA, lw=1.8, ls='--', zorder=3)
a1.text(.015, MEAS + .02, f' measured  f₂/f₁ = {MEAS:.3f}', color=O.MAGENTA,
        fontsize=9.8, fontweight='600', va='bottom',
        transform=a1.get_yaxis_transform())
for y, lab in ((3.240, 'clamped–pinned 3.24'), (2.757, 'clamped–clamped 2.76')):
    a1.axhline(y, color=O.MUT, lw=.9, ls=':', zorder=2)
    a1.text(.985, y + .015, lab + ' ', color=O.MUT, fontsize=8.8, ha='right',
            va='bottom', transform=a1.get_yaxis_transform())
a1.plot(kE, ratE, '-', color=O.BLUE, lw=2.2, label='EB  (k₂ = 0)', zorder=5)
a1.plot(FR[:, 0], FR[:, 1], '-', color=O.GREEN, lw=2.0, marker='D', ms=5.2,
        mew=1.6, mec=O.WHITE, label='FEM frictionless', zorder=5)
a1.plot(IS[:, 0], IS[:, 1], 'o', color=O.GREEN, ms=9, mfc='none', mew=2.2,
        label='FEM isotropic', zorder=6)

a1.plot([k_imp], [MEAS], '*', ms=17, color=O.MAGENTA, mec=O.WHITE, mew=1.2,
        zorder=8)
a1.annotate(f'the measured ratio implies\nk₁ = {k_imp:.0f} N/m in EB',
            xy=(k_imp, MEAS), xytext=(1500, 2.30), fontsize=9.8,
            color=O.MAGENTA, ha='left',
            arrowprops=dict(arrowstyle='->', color=O.MAGENTA, lw=1.2))
a1.axvline(346, color=O.BLUE, lw=1.2, ls=(0, (4, 3)), zorder=3)
a1.annotate('band-A fit lands here\n(k₁ = 346 N/m, ratio 2.48)',
            xy=(346, 2.46), xytext=(180, 2.86), fontsize=9.6, color=O.BLUE,
            ha='left', arrowprops=dict(arrowstyle='->', color=O.BLUE, lw=1.1))
a1.axvspan(1000, 2154, color=O.GREEN, alpha=.10, lw=0, zorder=1)
a1.text(1460, 1.95, 'independent k₁\nestimates', color='#004D21', fontsize=9.2,
        ha='center', va='bottom')
a1.set_xscale('log')
a1.set_xticks([150, 300, 600, 1300, 3000, 8000, 20000])
a1.get_xaxis().set_major_formatter(T.FuncFormatter(lambda v, _: '%g' % v))
a1.get_xaxis().set_minor_locator(T.NullLocator())
a1.set_xlabel('contact stiffness k₁  (N/m)')
a1.set_ylabel('f₂ / f₁   (flexural)')
a1.set_ylim(1.85, 3.42)
a1.legend(fontsize=9.5, labelcolor=O.INK2, loc='lower right')
O.clean(a1)
O.title(a1, 'The mode ratio pins k₁ — and mode 1 alone does not')

# ================= right: the observable consequence ======================
x, f, A = prep(); TD, ir = truth_dns(x, f, A)
R = pickle.load(open(str(OUT) + '/recs.pkl', 'rb'))


def fanti(S):
    out = np.full(len(S), np.nan)
    for i, row in enumerate(S):
        j0 = int(np.argmax(row)); seg = row[j0 + 3:]
        if len(seg) < 5:
            continue
        k = int(np.argmin(seg))
        if 0 < k < len(seg) - 1:
            out[i] = f[j0 + 3 + k]
    return out


for S, c, ls, lab, mk in ((A, O.INK, '-', 'measured', None),
                          (R[('eb', 20)]['A'], O.BLUE, '--', 'EB', 's'),
                          (R[('fem', 20)]['A'], O.GREEN, '--', 'FEM', 'D')):
    fa = fanti(S); m = np.isfinite(fa) & (x > 196)
    a2.plot(x[m], fa[m] / 1e3, ls, color=c, lw=2.6 if mk is None else 1.8,
            alpha=.35 if mk is None else 1.0, marker=mk, ms=5.0, mew=1.4,
            mec=O.WHITE, label=lab, zorder=3 if mk is None else 4)
a2.axvline(TD, color=O.MAGENTA, lw=1.3, ls=':')
a2.text(TD - .4, 350, 'D-NS ', color=O.MAGENTA, fontsize=9.2, ha='right',
        rotation=90, va='top')
a2.annotate('both models sit BELOW the data:\nEB by 18.2 kHz (5.3 %),\n'
            'FEM by 11.2 kHz (4.1 %) —\nthe same order as their\nf₂/f₁ error',
            xy=(217.5, 306.5), xytext=(212.8, 399), fontsize=9.5,
            color=O.INK2, ha='left', va='top',
            arrowprops=dict(arrowstyle='->', color=O.INK2, lw=1.1))
a2.set_ylim(286, 404)
a2.set_xlabel('position from clamp  (µm)')
a2.set_ylabel('antiresonance frequency  (kHz)')
a2.legend(fontsize=9.5, labelcolor=O.INK2, loc='lower left')
O.clean(a2)
O.title(a2, 'A mode 2 that sits too low drags the antiresonance down')

fig.savefig(str(FIG) + '/d_modes.png', dpi=170)
print('wrote fig/d_modes.png   EB-implied k1 = %.0f N/m' % k_imp)

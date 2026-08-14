"""Transfer functions and resonance mode shapes, reconstruction vs raw data.

Scope 265-1000 kHz so BOTH contact modes are in frame, which means the only
physics arm that can appear is the two-band one -- the band-A libraries stop at
600 kHz (see the wide-band slides).  Raw data is drawn as a thick pale line and
the reconstructions over it, so agreement reads as the raw line disappearing.

Positions are chosen from the WITHHELD set: none of these four spectra was shown
to any reconstructor.
"""
import sys, os, gc
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
import cplxrec as CR, physrec as PR, activemodemap as amm
from activemodemap.lowrank import classify_null_from_map, resonance_index
import ornl as O

O.style()
D = str(DENSE_GRID_A)
CACHE = str(OUT) + '/tf_shapes.pkl'
N0 = 8
LO, HI = 265e3, 1000e3
M1 = (285e3, 302e3)
M2 = (880e3, 925e3)

if os.path.exists(CACHE):
    C = pickle.load(open(CACHE, 'rb'))
else:
    dn = np.load(D + '/DenseReference.npz', allow_pickle=True)
    x = dn['x_um'].astype(float)
    dl = pd.read_csv(D + '/DenseReference_log.csv', parse_dates=['timestamp'])
    dl['i'] = dl.tune_file.str.extract(r'_(\d{4})\.txt$')[0].astype(int)
    x = amm.fit_position_scale(dl[dl.i < 20], dl[dl.i >= 20],
                               verbose=False).apply(x)
    fa = dn['freq_Hz']
    mk = (fa >= LO) & (fa <= HI)
    f = fa[mk][::4]
    Z = dn['Z'][:, mk][:, ::4]
    sel = sorted(set(np.argmin(np.abs(x[:, None] -
                 np.linspace(x.min(), x.max(), N0)[None, :]),
                 axis=0).tolist()))
    C = dict(x=x, f=f, Z=Z, sel=sel)
    C['gp'] = CR.rec_gp_cplx(x, sel, Z[sel], f)['Zrec']
    CR.load(str(OUT) + '/eblib_x_cplx.npz')
    o = CR.rec_twoband_cplx(x, sel, Z[sel], f)
    C['eb2'] = o['Zrec']
    C['eb2_bare'] = CR.rec_twoband_cplx(x, sel, Z[sel], f, gp=False)['Zrec']
    C['theta'] = o['theta']
    CR._LIB.clear(); gc.collect()
    pickle.dump(C, open(CACHE, 'wb'))

x, f, Z, sel = C['x'], C['f'], C['Z'], C['sel']
FK = f / 1e3
A = np.abs(Z)
AMAX = A.max()
held = [i for i in range(len(x)) if i not in set(sel)]
ARMS = [('gp', 'GP only', O.MAGENTA, '-'),
        ('eb2_bare', 'EB two-band, no GP', '#9DCBE2', '-'),
        ('eb2', 'EB two-band + GP', '#00456B', '--')]
dB = lambda v: 20 * np.log10(np.maximum(np.abs(v), 1e-16) / AMAX)
FLOOR = float(np.median(A[:, (f > 600e3) & (f < 700e3)]))


def _spans(mask):
    """Contiguous True runs of a boolean mask, as (start, end) index pairs."""
    d = np.diff(np.concatenate([[0], mask.view(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0],
                    np.minimum(np.where(d == -1)[0], len(mask) - 1)))

# four withheld positions: clamp end, the mode-2 antinode, mid, next to the node
POS = [min(held, key=lambda i: abs(x[i] - t))
       for t in (130.0, 174.5, 204.5, 222.7)]

# ============================================================ 1. transfer fns
fig = plt.figure(figsize=(13.2, 6.05))
for k, ip in enumerate(POS):
    L, W = .050 + .239 * k, .196
    a = fig.add_axes([L, .560, W, .345])
    b = fig.add_axes([L, .110, W, .345])
    a.plot(FK, dB(Z[ip]), '-', color=O.INK, lw=3.4, alpha=.26,
           label='raw data', solid_capstyle='round')
    for key, lab, c, ls in ARMS:
        a.plot(FK, dB(C[key][ip]), ls=ls, color=c, lw=1.4, label=lab)
    a.set_ylim(-68, 4); a.set_xlim(FK[0], FK[-1])
    a.set_xticklabels([])
    O.clean(a)
    a.set_title(f'x = {x[ip]:.1f} µm' +
                ('   (next to the node)' if k == 3 else ''),
                loc='left', fontsize=10.4, fontweight='600', color=O.INK)
    if k == 0:
        a.set_ylabel('|Z|  (dB re max)', fontsize=9.8)
        hl, hlab = a.get_legend_handles_labels()
    else:
        a.set_yticklabels([])

    # phase only where the raw signal clears 3x the noise floor: elsewhere the
    # measured phase is uniform random and no arm can or should match it
    ok = A[ip] > 3.0 * FLOOR
    b.plot(FK, np.degrees(np.angle(Z[ip])), '-', color=O.INK, lw=3.4,
           alpha=.26, solid_capstyle='round')
    for key, lab, c, ls in ARMS:
        b.plot(FK, np.degrees(np.angle(C[key][ip])), ls=ls, color=c, lw=1.4)
    b.set_ylim(-195, 195); b.set_yticks([-180, -90, 0, 90, 180])
    b.set_xlim(FK[0], FK[-1])
    for lo_, hi_ in _spans(~ok):
        b.axvspan(FK[lo_], FK[hi_], color=O.GREY, alpha=.45, lw=0, zorder=0)
    b.set_xlabel('frequency  (kHz)', fontsize=9.8)
    O.clean(b)
    if k == 0:
        b.set_ylabel('phase  (deg)', fontsize=9.8)
    else:
        b.set_yticklabels([])
for aa in fig.axes:
    aa.tick_params(labelsize=9.0)
fig.legend(hl, hlab, loc='upper center', bbox_to_anchor=(.52, 1.005), ncol=4,
           fontsize=10.0, labelcolor=O.INK2, frameon=False, columnspacing=2.2)
fig.savefig(f'{FIG}/d8_tf.png', dpi=170)
plt.close(fig)

# ============================================================ 2. mode shapes
def band_peak(V, band):
    mb = (f >= band[0]) & (f <= band[1])
    j = np.argmax(np.abs(V[:, mb]), axis=1)
    idx = np.where(mb)[0][j]
    return np.abs(V[np.arange(len(V)), idx]), idx


fig = plt.figure(figsize=(13.2, 4.75))
a1 = fig.add_axes([.050, .155, .262, .70])
a2 = fig.add_axes([.398, .155, .262, .70])
a3 = fig.add_axes([.744, .155, .230, .70])

for ax, band, nm in ((a1, M1, 'Mode 1  (293 kHz)'), (a2, M2, 'Mode 2  (902 kHz)')):
    t, _ = band_peak(Z, band)
    ax.plot(x, t / t.max(), '-', color=O.INK, lw=3.4, alpha=.26,
            label='raw data (101 positions)', solid_capstyle='round')
    for key, lab, c, ls in ARMS:
        v, _ = band_peak(C[key], band)
        ax.plot(x, v / t.max(), ls=ls, color=c, lw=1.5, label=lab)
    ax.plot(x[sel], (t / t.max())[sel], 'o', ms=7.5, mfc=O.WHITE,
            mec=O.MAGENTA, mew=1.8, label='revealed (8)', zorder=6)
    ax.set_yscale('log')
    ax.set_xlabel('position from clamp  (µm)')
    O.clean(ax)
    O.title(ax, nm + ' shape along the lever')
a1.set_ylabel('peak |Z| in band  (norm.)')
a1.set_ylim(1.2e-2, 2.6)
a2.set_ylim(4e-2, 2.6)
a1.legend(fontsize=8.6, loc='lower left', labelcolor=O.INK2)
t1, _ = band_peak(Z, M1)
a1.annotate('the detection node —\n2 decades in 3 µm',
            xy=(223.0, 5.0e-2), xytext=(196, 4.2e-1), fontsize=9.2,
            color=O.INK2, ha='right',
            arrowprops=dict(arrowstyle='->', color=O.INK2, lw=1.1))
t2, _ = band_peak(Z, M2)
a2.annotate(f'mode 2 arches instead: maximum at\nx = {x[int(np.argmax(t2))]:.0f} µm, '
            f'falling {t2.max()/t2[0]:.1f}× toward the clamp',
            xy=(x[int(np.argmax(t2))], 1.15), xytext=(128, 1.45),
            fontsize=9.2, color=O.INK2, ha='left',
            arrowprops=dict(arrowstyle='->', color=O.INK2, lw=1.1))

# phase at each resonance, showing the sign changes
for band, nm, c, ls in ((M1, 'mode 1', O.GREEN, '-'), (M2, 'mode 2', O.BLUE, '-')):
    mb = (f >= band[0]) & (f <= band[1])
    kres = np.where(mb)[0][int(np.argmax(A[:, mb].max(0)))]
    ref = np.angle(Z[:, kres])
    a3.plot(x, np.degrees(ref), ls, color=c, lw=3.0, alpha=.30,
            solid_capstyle='round', label=f'{nm}, raw')  # noqa
    zr = C['eb2'][:, kres]
    rot = np.exp(-1j * np.angle(np.sum(zr[sel] * np.conj(Z[sel, kres]))))
    a3.plot(x, np.degrees(np.angle(zr * rot)), '--', color=c, lw=1.4,
            label=f'{nm}, EB 2-band + GP')
a3.set_ylim(-190, 260); a3.set_yticks([-180, -90, 0, 90, 180])
a3.set_xlabel('position from clamp  (µm)')
a3.set_ylabel('phase at resonance  (deg)')
a3.legend(fontsize=8.2, loc='center left', labelcolor=O.INK2, ncol=1)
O.clean(a3)
O.title(a3, 'Mode 1 reverses at its node')
fig.savefig(f'{FIG}/d9_shapes.png', dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- numbers
print(f'withheld positions plotted: ' +
      ', '.join(f'{x[i]:.1f}' for i in POS))
Fn = PR.noise_floor(A[sel])
for key, lab, *_ in ARMS:
    e = np.abs(20 * np.log10((np.abs(C[key][held]) + Fn) / (A[held] + Fn)))
    m1v, _ = band_peak(Z, M1); r1v, _ = band_peak(C[key], M1)
    m2v, _ = band_peak(Z, M2); r2v, _ = band_peak(C[key], M2)
    pk1 = 100 * np.mean(np.abs(r1v[held] / m1v[held] - 1))
    pk2 = 100 * np.mean(np.abs(r2v[held] / m2v[held] - 1))
    print(f'{lab:22s} median |dB| {np.median(e):5.2f}   mode-1 peak '
          f'{pk1:5.2f} %   mode-2 peak {pk2:5.2f} %')
t2v, _ = band_peak(Z, M2)
print(f'mode-2 shape: max at x = {x[int(np.argmax(t2v))]:.1f} um, '
      f'{t2v.max()/t2v[0]:.1f}x above the clamp end, '
      f'{t2v.max()/t2v[-1]:.1f}x above the tip end')
print('wrote d8_tf.png, d9_shapes.png')

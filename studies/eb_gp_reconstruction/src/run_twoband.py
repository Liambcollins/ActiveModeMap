"""Dual-peak (two-band) fit results, on the setup the published note used.

Replicates `twoband.prep_wide`'s scope -- 265-1000 kHz, decimated x4, 101
positions -- and `twoband.regions`' data-driven scoring windows, so every number
here is directly comparable to `claude/two-band-fit-results-2026-08-12.md`.

Three arms, all leakage-safe (noise floor, SNR windows, gains and GP length
scale from the revealed block only):

  A only          fit the mode-1 window alone, one gain
  A + B, 1 gain   fit both windows jointly, one gain for the whole band
  A + B, per-band fit both windows jointly, a separate gain per band

and the extras the published run could not report, because its library had no
complex channel: where each arm puts the mode-2 PEAK, the model's own f2/f1 at
the fitted k1, and the per-band gain ratio -- which is a direct read on the
drive-coupling asymmetry between the two modes.
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
import numpy as np, pandas as pd, pickle
import cplxrec as CR, physrec as PR, twoband as TB, activemodemap as amm
from activemodemap.lowrank import classify_null_from_map, resonance_index

LIB = str(OUT) + '/eblib_x_cplx.npz'
D = str(DENSE_GRID_A)
NS = [3, 4, 5, 6, 8, 10, 12, 16, 20, 30]
LO, HI, BLO = 265e3, 1000e3, 700e3
COARSE = (48, 15, 7)

print(CR.load(LIB), flush=True)
TB.BAND_B = (BLO, HI)

dn = np.load(D + '/DenseReference.npz', allow_pickle=True)
x = dn['x_um'].astype(float)
dl = pd.read_csv(D + '/DenseReference_log.csv', parse_dates=['timestamp'])
dl['i'] = dl.tune_file.str.extract(r'_(\d{4})\.txt$')[0].astype(int)
x = amm.fit_position_scale(dl[dl.i < 20], dl[dl.i >= 20], verbose=False).apply(x)
fa = dn['freq_Hz']
m = (fa >= LO) & (fa <= HI)
f = fa[m][::4]
Z = dn['Z'][:, m][:, ::4]
A = np.abs(Z)
r1, r2, anti, floor = TB.regions(A, f, x)
ir = resonance_index(f, Z)
cl = classify_null_from_map(x, Z, f, ir)
TD = cl['x_null_um'] if np.isfinite(cl['x_null_um']) else cl['x_bound_um']
F2_MEAS = float(f[(f > BLO)][int(np.argmax(A[:, f > BLO].max(0)))])
F1_MEAS = float(f[(f < BLO)][int(np.argmax(A[:, f < BLO].max(0)))])
print(f'{A.shape[0]} pos x {A.shape[1]} bins, {f[0]/1e3:.0f}-{f[-1]/1e3:.0f} kHz'
      f'   measured f1 {F1_MEAS/1e3:.2f}  f2 {F2_MEAS/1e3:.2f}  '
      f'f2/f1 {F2_MEAS/F1_MEAS:.4f}   truth D-NS {TD:.2f} um', flush=True)


def band_a_window(A_sel, freq):
    p = np.asarray(A_sel, float).max(0)
    a = freq < BLO
    return a & (p >= 0.10 * p[a].max())


def peak2(Amap):
    """Mode-2 peak frequency of a reconstructed map (max over positions)."""
    mb = f > BLO
    return float(f[mb][int(np.argmax(Amap[:, mb].max(0)))])


def dbscore(Arec, sel):
    held = np.array([i for i in range(len(x)) if i not in set(sel)])
    Fn = PR.noise_floor(A[sel])
    e = np.abs(20 * np.log10((Arec[held] + Fn) / (A[held] + Fn)))
    out = {}
    for nm, mask in (('m1', r1), ('m2', r2)):
        out[nm] = float(np.median(e[:, mask]))
    out['anti'] = float(np.median(e[anti[held]]))
    out['all'] = float(np.median(e))
    return out


rows = []
for n in NS:
    sel = sorted(set(np.argmin(np.abs(x[:, None] -
                 np.linspace(x.min(), x.max(), n)[None, :]), axis=0).tolist()))
    As = A[sel]
    Fn = PR.noise_floor(As)

    # ---- arm 1: band A only, one gain
    wa = band_a_window(As, f)
    th = PR.fit_eb(sel, As, f, F=Fn, win=wa)
    M = PR.eb_predict(th['log_k1'], th['f_res'], th['log_g'], f) + th['gain']
    ra = dict(arm='A only', k1=th['k1'], f_res=th['f_res'], g=th['g'],
              gain_a=th['gain'], gain_b=np.nan, f2_pred=peak2(np.exp(M)))
    ra.update(dbscore(np.exp(M), sel))

    # ---- arm 2: A + B, single gain
    o2 = TB.rec2(x, sel, As, f, gp=False)
    t2 = o2['theta']
    rb = dict(arm='A+B, 1 gain', k1=t2['k1'], f_res=t2['f_res'], g=t2['g'],
              gain_a=t2['gain'], gain_b=np.nan, f2_pred=peak2(o2['A']))
    rb.update(dbscore(o2['A'], sel))

    # ---- arm 3: A + B, per-band gain (the headline arm)
    t3 = TB.fit2g(sel, As, f, coarse=COARSE)
    M3 = PR.eb_predict(t3['log_k1'], t3['f_res'], t3['log_g'], f)
    M3 = M3 + np.where(f >= BLO, t3['gain_b'], t3['gain_a'])[None, :]
    A3 = np.exp(M3)
    rc = dict(arm='A+B, per-band', k1=t3['k1'], f_res=t3['f_res'],
              g=t3['g_damp'], gain_a=t3['gain_a'], gain_b=t3['gain_b'],
              f2_pred=peak2(A3))
    rc.update(dbscore(A3, sel))

    # ---- arm 4: A + B, per-band gain + discrepancy GP (the deployable arm)
    rd = dict(arm='A+B, per-band +GP', k1=t3['k1'], f_res=t3['f_res'],
              g=t3['g_damp'], gain_a=t3['gain_a'], gain_b=t3['gain_b'])
    if n >= 5:
        og = TB.rec2g(x, sel, As, f, gp=True)
        rd['f2_pred'] = peak2(og['A'])
        rd.update(dbscore(og['A'], sel))
        rd['ell'] = og['ell']
    else:
        rd['f2_pred'] = rc['f2_pred']
        rd.update({k: np.nan for k in ('m1', 'm2', 'anti', 'all')})
        rd['ell'] = None

    for r in (ra, rb, rc, rd):
        r['n'] = n
        r['f2_err_kHz'] = (r['f2_pred'] - F2_MEAS) / 1e3
        r['f2_over_f1'] = r['f2_pred'] / r['f_res']
        r['gain_ratio_dB'] = (20 * np.log10(np.exp(r['gain_b'] - r['gain_a']))
                              if np.isfinite(r['gain_b']) else np.nan)
        rows.append(r)

    print(f"n={n:3d}  " + "   ".join(
        f"{r['arm']}: k1={r['k1']:7.1f} f2={r['f2_pred']/1e3:7.1f}"
        f"({r['f2_err_kHz']:+6.1f}) m1={r['m1']:5.2f} m2={r['m2']:5.2f}"
        for r in (ra, rb, rc, rd)), flush=True)

df = pd.DataFrame(rows)
df.to_csv(str(OUT) + '/dualpeak.csv', index=False)

print('\n== k1 (N/m) by arm and budget ==')
print(df.pivot(index='n', columns='arm', values='k1').round(1).to_string())
print('\n== mode-2 peak placement error (kHz) ==')
print(df.pivot(index='n', columns='arm', values='f2_err_kHz').round(1).to_string())
print('\n== median |dB|, mode-1 ridge / mode-2 ridge / antiresonance ==')
for nm in ('m1', 'm2', 'anti'):
    print(f'-- {nm}')
    print(df.pivot(index='n', columns='arm', values=nm).round(2).to_string())
pb = df[df.arm == 'A+B, per-band']
print('\n== dB with the discrepancy GP added to the per-band arm ==')
print(df[df.arm == 'A+B, per-band +GP'][['n','m1','m2','anti','all','ell']].round(2).to_string(index=False))
print(f"\nper-band gain ratio (mode 2 relative to mode 1): "
      f"{pb.gain_ratio_dB.mean():.2f} +/- {pb.gain_ratio_dB.std(ddof=1):.2f} dB"
      f"  =  x{np.exp(-pb.gain_ratio_dB.mean()/20*np.log(10)):.2f} weaker "
      f"coupling to mode 2")
print(f"k1 per-band arm: {pb.k1.mean():.1f} +/- {pb.k1.std(ddof=1):.1f} N/m "
      f"over n = {NS[0]}-{NS[-1]}  (spread "
      f"{100*(pb.k1.max()-pb.k1.min())/pb.k1.mean():.1f} %)")
print(f"model f2/f1 at the fitted k1: {pb.f2_over_f1.mean():.4f} "
      f"(measured {F2_MEAS/F1_MEAS:.4f})")
print('\nwrote out/dualpeak.csv')

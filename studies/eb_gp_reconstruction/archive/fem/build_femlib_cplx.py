#!/usr/bin/env python3
"""FEM ladder -> resonance-aligned surrogate library, WITH the complex channel.

This is `build_fem_library.py` with one addition: alongside `logamp_u` (which is
reproduced bit-for-bit, so the published fit is unchanged) it also stores the
real and imaginary parts of the response on the same (k1, damping, u, position)
grid.  The published arm threw the phase away at `np.abs(...)`; keeping it is
what lets the physics arm predict a phase map, not only a log-amplitude map.

Interpolation of Re/Im is linear in exactly the same places the amplitude is
interpolated linearly, so the two channels are consistent by construction and
|Re + i Im| is not required to equal exp(logamp) (it does not, near the nulls,
because |.| does not commute with interpolation -- that is why both are kept).
"""
import json, os, sys
import numpy as np
import h5py

import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_ROOT, 'config.py')):
    _ROOT = _os.path.dirname(_ROOT)
sys.path[:0] = [_ROOT, _os.path.join(_ROOT, 'src'),
                _os.path.join(_ROOT, 'figures')]
from config import EB_REPO, EB_GEOMETRY, FEM_LADDER, OUT, FIG

DATA = str(FEM_LADDER)
NAME = 'Multi75G'
LATERAL = 'isotropic'
CHAN = 'z_displacement'
OUTF = str(OUT) + '/femlib_isotropic_cplx.npz'

MEAS_CX_UM = 214.1
U_LO, U_HI, U_N = 0.55, 2.05, 1700
MODE1_WINDOW_KHZ = (150.0, 600.0)


def refine_peak(fr, prof, lo_hz, hi_hz):
    m = (fr >= lo_hz) & (fr <= hi_hz)
    if m.sum() < 3:
        return float('nan')
    f, p = fr[m], prof[m]
    j = int(np.argmax(p))
    if 0 < j < len(p) - 1:
        y0, y1, y2 = p[j-1], p[j], p[j+1]
        den = y0 - 2*y1 + y2
        if abs(den) > 1e-30:
            return float(f[j] + 0.5*(y0-y2)/den * (f[j]-f[j-1]))
    return float(f[j])


def _q(fr, prof, fres):
    m = np.abs(fr - fres) < 40e3
    if m.sum() < 5:
        return None
    fb, Ab = fr[m], prof[m]
    j = int(Ab.argmax()); h = Ab[j]/np.sqrt(2)
    lo = j
    while lo > 0 and Ab[lo] > h: lo -= 1
    hi = j
    while hi < len(Ab)-1 and Ab[hi] > h: hi += 1
    return float(fb[j]/(fb[hi]-fb[lo])) if fb[hi] > fb[lo] else None


man = json.load(open(os.path.join(DATA, f'fem_ladder_manifest_{NAME}.json')))
runs = [r for r in man['runs'] if r['lateral'] == LATERAL]
grid = [round(float(d), 6) for d in man.get('log_damping_grid', [])]
if grid:
    runs = [r for r in runs if round(r['log_damping'], 6) in set(grid)]
    lds = sorted(grid)
else:
    lds = sorted({round(r['log_damping'], 6) for r in runs})
idx = {}
for r in runs:
    idx[(round(r['k1'], 6), round(r['log_damping'], 6))] = r
gone = [k for k, r in idx.items()
        if not os.path.exists(os.path.join(DATA, r['file']))]
for k in gone:
    del idx[k]
k1s = sorted({k for k, _ in idx})
k1s = [k for k in k1s if all((k, d) in idx for d in lds)]
print(f'{len(k1s)} complete rungs x {len(lds)} damping levels '
      f'(dropped {len(gone)} missing files)')

x_meas = np.load(str(OUT) + '/x_cal.npy')
eta = x_meas / MEAS_CX_UM
x_model_um = eta * 225.0

UG = np.linspace(U_LO, U_HI, U_N)
shape = (len(k1s), len(lds), U_N, len(x_model_um))
LOGU = np.zeros(shape, dtype=np.float32)
REU = np.zeros(shape, dtype=np.float32)
IMU = np.zeros(shape, dtype=np.float32)
RES = np.zeros((len(k1s), len(lds)))
rows, worst_nan = [], 0.0

for i, k1 in enumerate(k1s):
    for j, ld in enumerate(lds):
        r = idx[(k1, ld)]
        with h5py.File(os.path.join(DATA, r['file']), 'r') as f:
            fr = f['spectrogram/frequency_hz'][:]
            xf = f['spectrogram/position_x_um'][:]
            ZR = f[f'spectrogram/{CHAN}/det_real'][:]
            ZI = f[f'spectrogram/{CHAN}/det_imag'][:]
        A = np.abs(ZR + 1j*ZI)
        x_clamp = 225.0 + xf
        o = np.argsort(x_clamp)
        x_clamp, A, ZR, ZI = x_clamp[o], A[o], ZR[o], ZI[o]

        fres = refine_peak(fr, A.max(0), MODE1_WINDOW_KHZ[0]*1e3,
                           MODE1_WINDOW_KHZ[1]*1e3)
        RES[i, j] = fres
        target = UG * fres
        inside = (target >= fr.min()) & (target <= fr.max())
        worst_nan = max(worst_nan, float(np.mean(~inside)))

        bA = np.full((len(x_clamp), U_N), np.nan)
        bR = np.full((len(x_clamp), U_N), np.nan)
        bI = np.full((len(x_clamp), U_N), np.nan)
        for p in range(len(x_clamp)):
            bA[p, inside] = np.interp(target[inside], fr, np.maximum(A[p], 1e-300))
            bR[p, inside] = np.interp(target[inside], fr, ZR[p])
            bI[p, inside] = np.interp(target[inside], fr, ZI[p])
        for u in range(U_N):
            cA = bA[:, u]
            if np.all(np.isfinite(cA)):
                LOGU[i, j, u] = np.log(np.interp(x_model_um, x_clamp, cA))
                REU[i, j, u] = np.interp(x_model_um, x_clamp, bR[:, u])
                IMU[i, j, u] = np.interp(x_model_um, x_clamp, bI[:, u])
            else:
                LOGU[i, j, u] = REU[i, j, u] = IMU[i, j, u] = np.nan
        rows.append((k1, ld, fres, _q(fr, A.max(0), fres)))
    print(f'  k1={k1:8.1f}  f_res={RES[i].mean()/1e3:7.2f} kHz', flush=True)

mono = bool(np.all(np.diff(RES.mean(1)) > 0))
print(f'\nf_res monotonic in k1: {mono}  '
      f'({RES.mean(1)[0]/1e3:.1f} -> {RES.mean(1)[-1]/1e3:.1f} kHz)')
bad_q = []
for i, k1 in enumerate(k1s):
    qr = [q for (kk, ld, fr_, q) in rows if kk == k1]
    if len(qr) == len(lds) and all(qr) and any(b >= a for a, b in zip(qr, qr[1:])):
        bad_q.append(round(k1))
print(f'Q strictly decreasing with damping on every rung: {not bad_q}'
      + (f'  violations at {bad_q}' if bad_q else ''))
qs = [q for *_, q in rows if q]
print(f'Q range {min(qs):.0f} - {max(qs):.0f}   (measured 191 must be bracketed:'
      f' {min(qs) <= 191 <= max(qs)})')
print(f'worst u-grid overhang {100*worst_nan:.2f} %')
print(f'NaN fraction: logamp {100*np.mean(~np.isfinite(LOGU)):.2f} %')

GG = 10.0 ** np.array(lds)
free = man.get('free_eigen_khz') or []
np.savez_compressed(
    OUTF, K1=np.array(k1s), GG=GG, log10_damping=np.array(lds), UG=UG,
    RES=RES, RES_K=RES.mean(1), logamp_u=LOGU, re_u=REU, im_u=IMU,
    x_model=x_model_um, x_meas=x_meas, eta=eta,
    F_REF=float(free[0]*1e3) if free else float('nan'),
    lateral=LATERAL, chan=CHAN,
    note='FEM surrogate, resonance-aligned frame u=f/f_res, axes '
         '(k1, log_damping, u, position). logamp_u reproduces the published '
         'library exactly; re_u/im_u add the complex channel the published '
         'build discarded at np.abs().')
print(f'\nwrote {OUTF}: {LOGU.shape}')

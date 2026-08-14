#!/usr/bin/env python3
"""Extended EB library (mode 1 AND mode 2), complex, pre-aligned.

Same solver grid as the published `libraries/buildlib_x.py` -- FLIB 185-1350 kHz
in 3200 bins, 96 k1 x 4 damping -- so the two-band fit that produced
k1 = 999-1013 N/m is reproducible from this file.  Two changes:

1. Re/Im are kept, not just log|z|.
2. The aligned u grid is NON-UNIFORM.  Covering u = 0.55-3.45 at the uniform
   spacing the band-A library used (8.8e-4) would need 3300 samples and 1.5 GB
   of float32; but the response only has structure near u = 1 (mode 1 and its
   antiresonance) and near u = 3.08 (mode 2), and everything downstream
   interpolates with searchsorted, so a graded grid is free accuracy.  1870
   samples give <=5e-4 through mode 1 and ~1.1e-3 through mode 2 -- finer than
   the 364 Hz solver grid can support in either case -- for 870 MB.

Note eb_models.py and geometry.py are in EB-Solver-CResonance/fem_eb_comparison/py,
NOT in that repo's src/.
"""
import json, time, sys
import numpy as np

import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_ROOT, 'config.py')):
    _ROOT = _os.path.dirname(_ROOT)
sys.path[:0] = [_ROOT, _os.path.join(_ROOT, 'src'),
                _os.path.join(_ROOT, 'figures')]
from config import EB_REPO, EB_GEOMETRY, FEM_LADDER, OUT, FIG

EBPY = str(EB_REPO / 'fem_eb_comparison' / 'py')
EBSRC = str(EB_REPO / 'src')
GEO = str(EB_GEOMETRY)
sys.path[:0] = [EBSRC, EBPY]
from geometry import ProbeGeometry
import eb_models as EB, eb_cr_afm as E

OUTF = str(OUT) + '/eblib_x_cplx.npz'
K1 = np.geomspace(150., 20000., 96)
GG = np.geomspace(5000., 20000., 4)
FLIB = np.linspace(185e3, 1350e3, 3200)
WIN = (FLIB >= 250e3) & (FLIB <= 460e3)      # physrec's mode-1 search window
FW = FLIB[WIN]

# graded u grid: dense through mode 1, its antiresonance, and mode 2
UG = np.unique(np.concatenate([
    np.linspace(0.55, 0.90, 120, endpoint=False),
    np.linspace(0.90, 1.15, 500, endpoint=False),
    np.linspace(1.15, 1.70, 400, endpoint=False),
    np.linspace(1.70, 2.80, 250, endpoint=False),
    np.linspace(2.80, 3.45, 600)]))

G = json.load(open(GEO))
BASE = {k: G[k] for k in ['name', 'L', 'contact_x', 'tip_offset', 'H',
                          'thickness', 'w_top', 'w_bot', 'area', 'I', 'E',
                          'rho']}
P = ProbeGeometry(**BASE)
F_REF = P.f_free(1)
CX = P.contact_x * 1e6
MEAS_CX = 214.1
x_meas = np.load(str(OUT) + '/x_cal.npy')
x_model = (x_meas / MEAS_CX) * CX * 1e-6
NP = len(x_model)
print(f'f_free1 {F_REF:.1f} Hz   FLIB {FLIB[0]/1e3:.0f}-{FLIB[-1]/1e3:.0f} kHz '
      f'({len(FLIB)} bins)   UG {UG[0]:.2f}-{UG[-1]:.2f} ({len(UG)} samples, '
      f'graded)   {len(K1)}x{len(GG)} rungs', flush=True)


def peak(v):
    j = int(np.argmax(v))
    if 0 < j < len(v) - 1:
        a, b, c = v[j-1], v[j], v[j+1]
        den = a - 2*b + c
        if abs(den) > 1e-30:
            return FW[j] + 0.5*(a-c)/den*(FW[j]-FW[j-1])
    return FW[j]


shape = (len(K1), len(GG), len(UG), NP)
LOGU = np.zeros(shape, dtype=np.float32)
REU = np.zeros(shape, dtype=np.float32)
IMU = np.zeros(shape, dtype=np.float32)
RES = np.zeros((len(K1), len(GG)))
t0 = time.perf_counter()

for ik, k1 in enumerate(K1):
    for ig, g in enumerate(GG):
        cfg = EB.EBConfig(probe=P, model='2seg', phi=np.deg2rad(11.),
                          k1=float(k1), k2=0., g=float(g), n_points=600)
        c, con, exc = cfg.cantilever(), cfg.contact(), cfg.excitation('mech')
        LA = np.empty((len(FLIB), NP), dtype=np.float32)
        RE = np.empty((len(FLIB), NP), dtype=np.float32)
        IM = np.empty((len(FLIB), NP), dtype=np.float32)
        for jf, fq in enumerate(FLIB):
            r = E.solve_single_frequency(float(fq), c, con, exc,
                                         cfg.options(fq, fq, 1.),
                                         obs_x=c.contact_x)
            xm = np.asarray(r['x']); z = np.asarray(r['z'])
            LA[jf] = np.log(np.interp(x_model, xm, np.abs(z)) + 1e-300)
            RE[jf] = np.interp(x_model, xm, z.real)
            IM[jf] = np.interp(x_model, xm, z.imag)
        fres = peak(LA[WIN].max(1))
        RES[ik, ig] = fres
        ft = UG * fres
        idx = np.clip(np.searchsorted(FLIB, ft) - 1, 0, len(FLIB) - 2)
        w = ((ft - FLIB[idx]) / (FLIB[idx+1] - FLIB[idx]))[:, None]
        LOGU[ik, ig] = (1-w)*LA[idx] + w*LA[idx+1]
        REU[ik, ig] = (1-w)*RE[idx] + w*RE[idx+1]
        IMU[ik, ig] = (1-w)*IM[idx] + w*IM[idx+1]
    el = time.perf_counter() - t0
    done = (ik + 1) * len(GG); tot = len(K1) * len(GG)
    print(f'  k1={k1:9.1f}  f_res {RES[ik].mean()/1e3:7.2f} kHz  {done}/{tot}  '
          f'{el/60:5.1f} min  eta {el/done*(tot-done)/60:5.1f} min', flush=True)

RES_K = RES.mean(1)
print(f'\nf_res monotonic in k1: {bool(np.all(np.diff(RES_K) > 0))}  '
      f'({RES_K[0]/1e3:.1f} -> {RES_K[-1]/1e3:.1f} kHz)')
print(f'coverage at the fitted rung: u {UG[0]:.2f}-{UG[-1]:.2f} x f_res ~ '
      f'{UG[0]*293:.0f}-{UG[-1]*293:.0f} kHz for f_res = 293 kHz')
print(f'NaN fraction: {100*np.mean(~np.isfinite(LOGU)):.3f} %')
np.savez(OUTF, K1=K1, GG=GG, UG=UG, RES=RES, RES_K=RES_K,
         logamp_u=LOGU, re_u=REU, im_u=IMU,
         x_model=x_model*1e6, x_meas=x_meas, eta=x_meas/MEAS_CX,
         F_REF=float(F_REF), k_lever=float(P.k_static), k2=0.0,
         note='EB 2-segment (phi=11 deg, k2=0), modes 1 AND 2, '
              'resonance-aligned frame u=f/f_res on a GRADED u grid, axes '
              '(k1, g, u, position). Solver grid identical to buildlib_x.py.')
print(f'wrote {OUTF}  {LOGU.shape}')

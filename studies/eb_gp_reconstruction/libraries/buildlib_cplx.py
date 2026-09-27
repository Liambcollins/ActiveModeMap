#!/usr/bin/env python3
"""EB response library WITH the complex channel, pre-aligned.

`libraries/buildlib.py` writes a raw-frequency log-amplitude library and lets
physrec resonance-align it at import; this writes the aligned form directly and
keeps Re/Im alongside, so `cplxrec.load` accepts it exactly as it accepts the
FEM library.

Reproducing the published EB arm requires matching TWO interpolation choices
that differ between the EB and FEM paths, and getting them backwards is worth
several percent:

  position   buildlib.py interpolates |z| over x and THEN takes the log
  frequency  physrec's raw-format path interpolates the LOG amplitude linearly
             in f (build_fem_library interpolates the amplitude and logs after)

Both are reproduced verbatim below.  Re/Im are interpolated linearly in the same
two places, which is the honest complex analogue.

Note eb_models.py and geometry.py live in EB-Solver-CResonance/fem_eb_comparison/py,
NOT in its src/ -- src/ holds only the eb_cr_afm package.
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

OUTF = str(OUT) + '/eblib_cplx.npz'
K2 = 0.0                     # the published EB arm: no lateral contact spring
K1 = np.geomspace(150., 20000., 96)
GG = np.geomspace(5000., 20000., 4)
FLIB = np.linspace(185e3, 720e3, 1600)
UG = np.linspace(0.55, 2.05, 1700)
WIN = (FLIB >= 250e3) & (FLIB <= 460e3)      # physrec's mode-1 search window
FW = FLIB[WIN]

G = json.load(open(GEO))
BASE = {k: G[k] for k in ['name', 'L', 'contact_x', 'tip_offset', 'H',
                          'thickness', 'w_top', 'w_bot', 'area', 'I', 'E',
                          'rho']}
P = ProbeGeometry(**BASE)
F_REF = P.f_free(1)
CX = P.contact_x * 1e6                        # 225.0 um on the model beam
MEAS_CX = 214.1
x_meas = np.load(str(OUT) + '/x_cal.npy')
x_model = (x_meas / MEAS_CX) * CX * 1e-6      # metres
NP = len(x_model)
print(f'f_free1 {F_REF:.1f} Hz, k_lever {P.k_static:.3f} N/m, '
      f'{len(K1)}x{len(GG)} rungs x {len(FLIB)} frequencies')


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
                          k1=float(k1), k2=K2, g=float(g), n_points=600)
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
    done = (ik + 1) * len(GG)
    tot = len(K1) * len(GG)
    print(f'  k1={k1:9.1f}  f_res {RES[ik].mean()/1e3:7.2f} kHz  '
          f'{done}/{tot}  {el/60:5.1f} min  eta {el/done*(tot-done)/60:5.1f} min',
          flush=True)

RES_K = RES.mean(1)
print(f'\nf_res monotonic in k1: {bool(np.all(np.diff(RES_K) > 0))}  '
      f'({RES_K[0]/1e3:.1f} -> {RES_K[-1]/1e3:.1f} kHz)')
print(f'NaN fraction: {100*np.mean(~np.isfinite(LOGU)):.3f} %')
np.savez_compressed(OUTF, K1=K1, GG=GG, UG=UG, RES=RES, RES_K=RES_K,
                    logamp_u=LOGU, re_u=REU, im_u=IMU,
                    x_model=x_model*1e6, x_meas=x_meas,
                    eta=x_meas/MEAS_CX, F_REF=float(F_REF),
                    k_lever=float(P.k_static), k2=K2,
                    note='EB 2-segment (phi=11 deg, k2=0) surrogate in the '
                         'resonance-aligned frame u=f/f_res, axes '
                         '(k1, g, u, position). logamp_u reproduces the '
                         'published eblib path; re_u/im_u add the complex '
                         'channel buildlib.py never stored.')
print(f'wrote {OUTF}  {LOGU.shape}')

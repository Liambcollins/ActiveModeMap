"""Two-band fitting: include flexural mode 2 in both the data and the fit.

Band A (mode 1 + its antiresonance) carries the mode shape; band B (mode 2 at
902 kHz) carries the contact stiffness. This module fits BOTH resonance windows
jointly against the extended EB library and reconstructs 265-1000 kHz.

Leakage rule unchanged: windows, noise floor and GP length scale come from the
revealed block only.
"""
import numpy as np

import sys as _sys, os as _os
_D = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_D, 'config.py')):
    _D = _os.path.dirname(_D)
_sys.path[:0] = [_D, _os.path.join(_D, 'src'),
                 _os.path.join(_D, 'figures')]
from config import DENSE_GRID_A, OUT

import physrec as P
from run_phys import truth_dns

FDEC = 4
LO, HI = 265e3, 1000e3
BAND_B = (700e3, 1000e3)


def prep_wide():
    import pandas as pd, activemodemap as amm
    from setup_data import load as _  # noqa  (keeps import graph identical)
    D = str(DENSE_GRID_A)
    dn = np.load(D + '/DenseReference.npz', allow_pickle=True)
    x = dn['x_um'].astype(float)
    f = dn['freq_Hz']
    Z = dn['Z']
    dl = pd.read_csv(D + '/DenseReference_log.csv', parse_dates=['timestamp'])
    dl['i'] = dl.tune_file.str.extract(r'_(\d{4})\.txt$')[0].astype(int)
    ps = amm.fit_position_scale(dl[dl.i < 20], dl[dl.i >= 20], verbose=False)
    x = ps.apply(x)
    m = (f >= LO) & (f <= HI)
    return x, f[m][::FDEC], np.abs(Z[:, m][:, ::FDEC])


def win_two(A_sel, freq):
    """Per-band SNR windows from the REVEALED block, OR-ed together."""
    p = np.asarray(A_sel, float).max(axis=0)
    a = freq < BAND_B[0]
    b = ~a
    wa = a & (p >= 0.10 * p[a].max())
    wb = b & (p >= 0.10 * p[b].max())
    return wa | wb, wa, wb


def rec2(x_grid, sel_idx, A_sel, freq, gp=True,
         ells=(4., 8., 15., 30., 60., 120.)):
    """Two-band physics fit + optional GP discrepancy over position."""
    F = P.noise_floor(A_sel)
    win, wa, wb = win_two(A_sel, freq)
    th = P.fit_eb(sel_idx, A_sel, freq, F=F, win=win)
    M = P.eb_predict(th['log_k1'], th['f_res'], th['log_g'], freq) + th['gain']
    Tm = P.T(np.exp(M), F)
    ell = None
    sel = np.asarray(sel_idx)
    n = len(sel)
    if gp and n >= 5:
        R = P.T(A_sel, F) - Tm[sel]
        xs = x_grid[sel]
        noise = max(1e-4, float(np.var(R)) * 0.05)
        ell = min(ells, key=lambda e: P._press(xs, R, e, noise))
        v = float(np.var(R)) + 1e-18
        K = v * np.exp(-((xs[:, None] - xs[None, :]) ** 2) / (2 * ell ** 2)) \
            + noise * np.eye(n)
        ks = v * np.exp(-((x_grid[:, None] - xs[None, :]) ** 2) / (2 * ell ** 2))
        Tm = Tm + ks @ np.linalg.solve(K, R)
    return dict(A=P.Tinv(Tm, F), Tmap=Tm, F=F, theta=th, ell=ell,
                win=win, wa=wa, wb=wb)


def regions(A, freq, x):
    """Scoring regions on the TRUE field: mode-1 ridge, mode-2 ridge,
    antiresonance neighbourhoods, floor."""
    p = A.max(axis=0)
    a = freq < BAND_B[0]
    ridge1 = a & (p >= 0.10 * p[a].max())
    ridge2 = (~a) & (p >= 0.10 * p[~a].max())
    anti = np.zeros_like(A, bool)
    for i in range(len(x)):
        pa = np.where(a)[0]
        j0 = pa[int(np.argmax(A[i, a]))]
        seg = A[i, j0 + 3:len(pa)]
        if len(seg) > 5:
            k = j0 + 3 + int(np.argmin(seg))
            anti[i, max(0, k - 4):k + 5] = True
    floor = ~ridge1[None, :] & ~ridge2[None, :] & ~anti
    return ridge1, ridge2, anti, floor


def fit2g(sel_idx, A_sel, freq, F=None, coarse=(16, 11, 5), refine=3):
    """Two-band fit with a PER-BAND gain.

    Identical to physrec.fit_eb except the closed-form gain is solved
    separately for band A and band B. Physical reading: one scalar per band
    absorbs the drive-coupling / detection-observable error in the RELATIVE
    modal amplitude (EB is compared on displacement while the lever senses
    slope), the same way the single gain absorbs InvOLS. Two numbers, still
    closed-form, still leakage-safe.
    """
    sel = np.asarray(sel_idx)
    A_sel = np.asarray(A_sel, float)
    if F is None:
        F = P.noise_floor(A_sel)
    win, wa, wb = win_two(A_sel, freq)
    fw = freq[win]
    Ao = A_sel[:, win]
    Tobs = P.T(Ao, F)
    inb = (fw >= BAND_B[0])                    # band-B bins inside the window
    f_obs = float(fw[~inb][int(np.argmax(Ao[:, ~inb].max(0)))])

    def cost(p):
        M = P.eb_predict(p[0], p[1], p[2], fw, rows=sel)
        w = Ao > 3.0 * F
        g = np.zeros(2)
        for k, m in enumerate((~inb, inb)):
            wm = w & m[None, :]
            g[k] = (float(np.mean(np.log(Ao[wm]) - M[wm])) if wm.sum() > 4
                    else float(np.mean(Tobs[:, m] - M[:, m])))
        Mg = M + np.where(inb, g[1], g[0])[None, :]
        Tm = P.T(np.exp(Mg), F)
        return float(np.mean((Tm - Tobs) ** 2)), g

    best = (np.inf, None, None)
    for lk in np.linspace(P.LK1[0], P.LK1[-1], coarse[0]):
        for fr in f_obs * np.linspace(0.985, 1.015, coarse[1]):
            for lg in np.linspace(P.LGG[0], P.LGG[-1], coarse[2]):
                c, g = cost([lk, fr, lg])
                if c < best[0]:
                    best = (c, [lk, fr, lg], g)
    p = list(best[1])
    from scipy.optimize import minimize_scalar
    bnds = [(P.LK1[0], P.LK1[-1]), (f_obs * 0.98, f_obs * 1.02),
            (P.LGG[0], P.LGG[-1])]
    for _ in range(refine):
        for d, (lo, hi) in enumerate(bnds):
            r = minimize_scalar(lambda v, d=d: cost([*p[:d], v, *p[d + 1:]])[0],
                                bounds=(lo, hi), method='bounded',
                                options={'xatol': 1e-5})
            p[d] = float(r.x)
    c, g = cost(p)
    return dict(log_k1=p[0], f_res=p[1], log_g=p[2], gain_a=g[0], gain_b=g[1],
                rms=float(np.sqrt(c)), F=F, k1=float(np.exp(p[0])),
                g_damp=float(np.exp(p[2])), win=win, wa=wa, wb=wb)


def rec2g(x_grid, sel_idx, A_sel, freq, gp=True,
          ells=(4., 8., 15., 30., 60., 120.)):
    """Reconstruction from the per-band-gain fit (+ optional GP)."""
    th = fit2g(sel_idx, A_sel, freq)
    F = th['F']
    M = P.eb_predict(th['log_k1'], th['f_res'], th['log_g'], freq)
    M = M + np.where(freq >= BAND_B[0], th['gain_b'], th['gain_a'])[None, :]
    Tm = P.T(np.exp(M), F)
    sel = np.asarray(sel_idx)
    n = len(sel)
    ell = None
    if gp and n >= 5:
        R = P.T(A_sel, F) - Tm[sel]
        xs = x_grid[sel]
        noise = max(1e-4, float(np.var(R)) * 0.05)
        ell = min(ells, key=lambda e: P._press(xs, R, e, noise))
        v = float(np.var(R)) + 1e-18
        K = v * np.exp(-((xs[:, None] - xs[None, :]) ** 2) / (2 * ell ** 2)) \
            + noise * np.eye(n)
        ks = v * np.exp(-((x_grid[:, None] - xs[None, :]) ** 2) / (2 * ell ** 2))
        Tm = Tm + ks @ np.linalg.solve(K, R)
    return dict(A=P.Tinv(Tm, F), Tmap=Tm, F=F, theta=th, ell=ell)

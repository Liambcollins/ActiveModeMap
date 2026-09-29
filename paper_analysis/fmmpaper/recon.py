"""Sparse reconstruction arms, position selectors and held-out scoring.

Arms (all leakage-safe: they see only the positions they are given):

* ``rec_gp``      -- model-free complex GP over position, length scale by LOO/PRESS.
                     Transcribed from Manuscript_FMM/gparm.py (the draft's GP arm).
* ``rec_lowrank`` -- rank-r Chebyshev fit; thin wrapper on activemodemap.lowrank.
* EB+GP           -- the draft's physics arm needs the two-segment EB library
                     (EB-Solver-CResonance, ~3 GB binding). Not re-run here; published
                     values are read from the Manuscript_FMM pickles (see notebook 02).

Scoring follows the draft: complex NRMSE (%) on held-out positions only.
"""
from __future__ import annotations

import numpy as np

from . import config

# ------------------------------------------------------------------ selectors


def select_equispaced(x, n):
    """Indices of the n grid points closest to an even spacing over the span."""
    x = np.asarray(x)
    return sorted({int(np.argmin(np.abs(x - t))) for t in np.linspace(x.min(), x.max(), n)})


def select_random(x, n, rng):
    return sorted(rng.choice(len(x), n, replace=False).tolist())


# ------------------------------------------------------------------ GP arm
ELLS = (1.5, 2.5, 4.0, 8.0, 15.0, 30.0, 60.0, 120.0)


def _press(xs, R, ell, noise):
    d2 = (xs[:, None] - xs[None, :]) ** 2
    v = float(np.var(R)) + 1e-18
    K = v * np.exp(-d2 / (2 * ell ** 2)) + noise * np.eye(len(xs))
    Ki = np.linalg.inv(K)
    dg = np.clip(np.diag(Ki), 1e-12, None)
    return float(np.mean(((Ki @ R) / dg[:, None]) ** 2))


def _press_cplx(xs, R, ell, noise):
    return _press(xs, np.concatenate([R.real, R.imag], 1), ell, noise)


def _gp_posterior(x_grid, xs, R, ell, noise_frac=0.05):
    n = len(xs)
    Kc = np.exp(-((xs[:, None] - xs[None, :]) ** 2) / (2 * ell ** 2))
    kc = np.exp(-((x_grid[:, None] - xs[None, :]) ** 2) / (2 * ell ** 2))
    Ai = np.linalg.inv(Kc + noise_frac * np.eye(n))
    shape = np.clip(1.0 - np.einsum("ij,jk,ik->i", kc, Ai, kc), 0.0, None)
    v = 0.5 * (R.real.var(0) + R.imag.var(0) + 1e-30)
    return kc @ (Ai @ R), np.sqrt(np.outer(shape, v))


def rec_gp(x_grid, sel_idx, Z_sel, ells=ELLS):
    """Model-free complex GP. Returns dict(Zrec, sd, ell)."""
    x_grid = np.asarray(x_grid, float)
    sel = np.asarray(sel_idx)
    m = Z_sel.mean(0)
    R = Z_sel - m[None, :]
    xs = x_grid[sel]
    if len(sel) < 2:
        return dict(Zrec=np.repeat(m[None, :], len(x_grid), 0),
                    sd=np.full((len(x_grid), Z_sel.shape[1]), np.abs(m).mean()), ell=np.nan)
    nz = 0.05 * float(np.mean(np.abs(R) ** 2)) + 1e-18
    ell = min(ells, key=lambda e: _press_cplx(xs, R, e, nz))
    corr, sd = _gp_posterior(x_grid, xs, R, ell)
    return dict(Zrec=m[None, :] + corr, sd=sd, ell=ell)


# ------------------------------------------------------------------ low-rank arm
def rec_lowrank(x_grid, sel_idx, Z_sel, rank):
    """Rank-r Chebyshev reconstruction (activemodemap.lowrank.reconstruct_map)."""
    config.ensure_activemodemap()
    from activemodemap.lowrank import reconstruct_map
    r = reconstruct_map(np.asarray(x_grid, float), list(sel_idx), Z_sel, rank)
    return dict(Zrec=r["Zrec"], sd=r["std"], rank=min(rank, len(sel_idx)))


def cross_capture(x_design, Z_design, x_eval, arm="lowrank", **kw):
    """Predict spectra at positions from a *different* capture (live validation).

    Builds a joint grid (design + evaluation positions), marks only the design
    positions as measured, and returns the predictions at x_eval.
    """
    x_grid = np.concatenate([x_design, x_eval])
    sel = np.arange(len(x_design))
    fn = rec_lowrank if arm == "lowrank" else rec_gp
    r = fn(x_grid, sel, Z_design, **kw)
    return r["Zrec"][len(x_design):], r["sd"][len(x_design):]


# ------------------------------------------------------------------ metrics
def nrmse(Zrec, Ztrue, mask=None):
    """Complex NRMSE in percent, optionally over a frequency mask."""
    if mask is not None:
        Zrec, Ztrue = Zrec[..., mask], Ztrue[..., mask]
    return float(100 * np.linalg.norm(Zrec - Ztrue) / np.linalg.norm(Ztrue))


def score(Zrec, Ztrue, freq=None, bands=None) -> dict:
    """Standard score block: complex/amplitude NRMSE, dB agreement, phase MAE, per band.

    Zrec, Ztrue: [positions, freq] at held-out positions only.
    ``bands``: {name: (lo, hi)} evaluated in addition to the full band.
    """
    out = dict(nrmse=nrmse(Zrec, Ztrue),
               nrmse_amp=float(100 * np.linalg.norm(np.abs(Zrec) - np.abs(Ztrue))
                               / np.linalg.norm(np.abs(Ztrue))))
    # agreement in dB, restricted to where there is signal (> 1 % of the max per position)
    at, ar = np.abs(Ztrue), np.abs(Zrec)
    sig = at > 0.01 * at.max(1, keepdims=True)
    ddb = np.abs(20 * np.log10(np.maximum(ar, 1e-30) / np.maximum(at, 1e-30)))[sig]
    out["db_median"] = float(np.median(ddb))
    out["within_3dB"] = float(np.mean(ddb < 3.0))
    w = at
    out["phase_mae_deg"] = float(np.degrees((w * np.abs(np.angle(Zrec * np.conj(Ztrue)))).sum() / w.sum()))
    if bands and freq is not None:
        for name, (lo, hi) in bands.items():
            m = (freq >= lo) & (freq <= hi)
            out[f"nrmse_{name}"] = nrmse(Zrec, Ztrue, m)
    return out


def held_out(n_total, sel_idx):
    return np.setdiff1d(np.arange(n_total), np.asarray(sel_idx))


def budget_sweep(x, Z, ns, arm="gp", selector="equispaced", mask=None, rng=None,
                 n_random=0, **kw):
    """Error vs number of measured positions, on held-out positions of a dense map.

    Returns a list of dict(n, selector, nrmse, sel). With n_random > 0 also draws
    that many random subsets per n (selector='random').
    """
    fn = rec_gp if arm == "gp" else rec_lowrank
    rows = []
    for n in ns:
        designs = []
        if selector == "equispaced":
            designs.append(("equispaced", select_equispaced(x, n)))
        for _ in range(n_random):
            designs.append(("random", select_random(x, n, rng)))
        for name, sel in designs:
            r = fn(x, sel, Z[sel], **kw)
            h = held_out(len(x), sel)
            rows.append(dict(n=n, selector=name, arm=arm, sel=tuple(sel),
                             nrmse=nrmse(r["Zrec"][h], Z[h], mask)))
    return rows

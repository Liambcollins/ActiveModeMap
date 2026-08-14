"""Leakage-safe retrospective AL benchmark harness.

The reconstructor NEVER receives the full matrix: its signature is
(x_grid, sel_idx, Z_sel, **kw) -> dict(Zrec, std). Only score() sees Z_full.
"""
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

import numpy as np, time
import activemodemap as amm
from activemodemap.lowrank import chebyshev_basis, classify_null_from_map, resonance_index

# ---------------------------------------------------------------- reconstructors
def rec_lowrank(x_grid, sel_idx, Z_sel, rank=4):
    return amm.reconstruct_map(x_grid, sel_idx, Z_sel, rank)

def choose_rank_cv(x_grid, sel_idx, Z_sel, ranks=(2,3,4,5,6,7,8,10,12)):
    """Rank by closed-form leave-one-out PRESS, computed from ONE fit per rank.

    For a linear LS fit the LOO residual is e_j / (1 - h_jj) exactly, so no
    refitting is needed.  Uses ONLY the revealed points -- the held-out set is
    never touched, which is what keeps this leakage-free.
    """
    n = len(sel_idx)
    best, best_e = None, np.inf
    for r in ranks:
        if r > n - 2:            # need residual dof
            continue
        B = chebyshev_basis(x_grid, r)
        Bs = B[sel_idx]
        G = np.linalg.pinv(Bs.conj().T @ Bs)
        coef = G @ (Bs.conj().T @ Z_sel)
        e = Bs @ coef - Z_sel
        h = np.einsum('ij,jk,ik->i', Bs, G, Bs.conj()).real
        w = np.clip(1.0 - h, 1e-9, None)
        press = float((np.abs(e / w[:, None])**2).sum())
        if press < best_e:
            best_e, best = press, r
    return best if best is not None else 2

# ---------------------------------------------------------------- acquisition
def strat_dopt(x_grid, n, seeds, rank=4, **kw):
    return amm.d_optimal_order(x_grid, rank, seeds=seeds, n_select=n)[:n]

def strat_equispaced(x_grid, n, seeds=None, **kw):
    t = np.linspace(x_grid.min(), x_grid.max(), n)
    return sorted({int(np.argmin(np.abs(x_grid - v))) for v in t})

def strat_random(x_grid, n, seeds, rng=None, **kw):
    sel = list(seeds)
    pool = [i for i in range(len(x_grid)) if i not in sel]
    sel += list(rng.choice(pool, size=max(0, n - len(sel)), replace=False))
    return sorted(sel[:n])

def strat_maxvar(x_grid, n, seeds, Z_reveal=None, rank=4, **kw):
    """Adaptive: greedily add the position of maximum predictive std."""
    sel = list(seeds)
    while len(sel) < n:
        r = min(rank, max(2, len(sel) - 2)) or 2
        out = amm.reconstruct_map(x_grid, sel, Z_reveal(sel), min(rank, len(sel)))
        sc = out['std'].mean(1).copy()
        sc[sel] = -np.inf
        sel.append(int(np.argmax(sc)))
    return sorted(sel)

def strat_dns(x_grid, n, seeds, Z_reveal=None, rank=4, freq=None, ires=None, **kw):
    """Adaptive: add the position that most shrinks the bootstrap spread of the D-NS."""
    rng = np.random.default_rng(0)
    sel = list(seeds)
    while len(sel) < n:
        cand = [i for i in range(len(x_grid)) if i not in sel]
        # score candidates by leverage-weighted proximity to the current D-NS estimate
        out = amm.reconstruct_map(x_grid, sel, Z_reveal(sel), min(rank, len(sel)))
        cl = classify_null_from_map(x_grid, out['Zrec'], freq, ires)
        x0 = cl['x_null_um'] if np.isfinite(cl['x_null_um']) else (
             cl['x_bound_um'] if np.isfinite(cl['x_bound_um']) else x_grid[-1])
        lev = out['std'].mean(1)
        sc = [lev[i] / (1.0 + abs(x_grid[i] - x0)) for i in cand]
        sel.append(cand[int(np.argmax(sc))])
    return sorted(sel)

STRATS = dict(dopt_lowrank=strat_dopt, equispaced=strat_equispaced, random=strat_random,
              maxvar=strat_maxvar, dns_targeted=strat_dns)

# ---------------------------------------------------------------- scoring
def score(x_grid, sel_idx, out, Z_full, freq, ires, near_tip_um=10.0,
          truth_dns=np.nan):
    held = np.array([i for i in range(len(x_grid)) if i not in set(sel_idx)])
    Zr, Zt = out['Zrec'][held], Z_full[held]
    m = {}
    m['nrmse_cplx'] = 100*np.linalg.norm(Zr-Zt)/np.linalg.norm(Zt)
    at, ar = np.abs(Zt).max(1), np.abs(Zr).max(1)          # peak-in-band amplitude
    rel = np.abs(ar-at)/np.maximum(at, 1e-300)
    m['amp_mean_pct'], m['amp_max_pct'] = 100*rel.mean(), 100*rel.max()
    tip = held[x_grid[held] >= x_grid.max()-near_tip_um]
    if len(tip):
        att, arr = np.abs(Z_full[tip]).max(1), np.abs(out['Zrec'][tip]).max(1)
        m['amp_neartip_pct'] = 100*np.mean(np.abs(arr-att)/np.maximum(att,1e-300))
        m['nrmse_neartip'] = 100*np.linalg.norm(out['Zrec'][tip]-Z_full[tip])/np.linalg.norm(Z_full[tip])
    else:
        m['amp_neartip_pct'] = m['nrmse_neartip'] = np.nan
    cl = classify_null_from_map(x_grid, out['Zrec'], freq, ires)
    m['dns_status'] = cl['status']
    m['dns'] = cl['x_null_um'] if np.isfinite(cl['x_null_um']) else cl['x_bound_um']
    m['dns_err'] = abs(m['dns']-truth_dns) if np.isfinite(m['dns']) else np.nan
    m['n_crossings'] = len(cl['crossings_um'])
    # uncertainty calibration on held-out points, amplitude scale
    E = np.abs(np.abs(Zr)-np.abs(Zt)); S = out['std'][held]
    ok = S > 0
    m['calib_median'] = float(np.median(E[ok]/S[ok])) if ok.any() else np.nan
    m['calib_frac1s'] = float(np.mean(E[ok] <= S[ok])) if ok.any() else np.nan
    return m

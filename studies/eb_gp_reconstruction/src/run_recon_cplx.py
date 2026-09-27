"""Complex reconstructions at n = 8 for every arm, cached for the deck.

Leakage-safe: each arm sees only (x_grid, sel_idx, Z[sel], freq).  The scorer
alone sees the withheld positions.

Both libraries are loaded in turn and released, because each carries ~0.5-0.8 GB
of Re/Im blocks and physrec's globals are rebound wholesale by use_library --
a half-swapped physrec would silently mix the two forward models.
"""
import sys, gc
import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_ROOT, 'config.py')):
    _ROOT = _os.path.dirname(_ROOT)
sys.path[:0] = [_ROOT, _os.path.join(_ROOT, 'src'),
                _os.path.join(_ROOT, 'figures')]
from config import (DENSE_GRID_A, FEM_LADDER, EB_REPO, EB_GEOMETRY, OUT, FIG,
                    add_eb_to_path)
add_eb_to_path()
import numpy as np, pickle, time
import cplxrec as CR, physrec as PR
from setup_data import load as load_meas
from activemodemap.lowrank import classify_null_from_map, resonance_index

LIBS = {'eb': str(OUT) + '/eblib_cplx.npz',
        'fem': str(OUT) + '/femlib_isotropic_cplx.npz'}
FDEC, N = 4, 8

x, f, Z, _, ps = load_meas('A', 'calibrated')
f, Z = f[::FDEC], Z[:, ::FDEC]
A = np.abs(Z)
ir = resonance_index(f, Z)
cl = classify_null_from_map(x, Z, f, ir)
TD = cl['x_null_um'] if np.isfinite(cl['x_null_um']) else cl['x_bound_um']
sel = sorted(set(np.argmin(np.abs(x[:, None] -
                                 np.linspace(x.min(), x.max(), N)[None, :]),
                          axis=0).tolist()))
print(f'{len(x)} positions x {len(f)} frequencies   truth D-NS {TD:.2f} um   '
      f'f_res {f[ir]/1e3:.2f} kHz')
print('revealed:', np.round(x[sel], 1), flush=True)

R = dict(x=x, f=f, Z=Z, sel=sel, TD=TD, ir=ir, n=N)
hdr = (f"\n{'arm':10s} {'cplx%':>7} {'amp%':>7} {'phase':>7} {'D-NS':>8} "
       f"{'k1':>8} {'ell':>5} {'z_rms':>6} {'cov1s':>6}")
print(hdr)

# --- GP only: no library needed
o = CR.rec_gp_cplx(x, sel, Z[sel], f)
m, z = CR.score_cplx(x, sel, o, Z, f, TD)
R['gp'], R['gp_m'], R['gp_z'] = o, m, z
print(f"{'gp':10s} {m['nrmse_cplx']:7.2f} {m['nrmse_amp']:7.2f} "
      f"{m['phase_mae_deg']:6.1f}d {m['dns']:8.2f} {'':>8} "
      f"{str(o['ell']):>5} {m['z_rms']:6.2f} {m['cov1s']:6.3f}", flush=True)

for tag, lib in LIBS.items():
    print(CR.load(lib), flush=True)
    for nm, kw in ((tag, dict(gp=False)), (tag + '_gp', {})):
        t0 = time.perf_counter()
        o = CR.rec_phys_cplx(x, sel, Z[sel], f, **kw)
        m, z = CR.score_cplx(x, sel, o, Z, f, TD)
        m['t_fit'] = time.perf_counter() - t0
        R[nm], R[nm + '_m'], R[nm + '_z'] = o, m, z
        th = o['theta'] or {}
        print(f"{nm:10s} {m['nrmse_cplx']:7.2f} {m['nrmse_amp']:7.2f} "
              f"{m['phase_mae_deg']:6.1f}d {m['dns']:8.2f} "
              f"{th.get('k1', float('nan')):8.1f} {str(o['ell']):>5} "
              f"{m['z_rms']:6.2f} {m['cov1s']:6.3f}", flush=True)
    # published log-amplitude arms from the SAME library, as the cross-check
    for nm, fn, kw in ((tag + '_logamp', PR.rec_eb, {}),
                       (tag + '_gp_logamp', PR.rec_eb_gp, {})):
        oo = fn(x, sel, A[sel], f, **kw)
        held = np.array([i for i in range(len(x)) if i not in set(sel)])
        e = 100 * np.linalg.norm(oo['A'][held] - A[held]) / np.linalg.norm(A[held])
        R[nm] = dict(amp_nrmse=e, k1=(oo.get('theta') or {}).get('k1'))
        print(f'    published log-amp {nm:18s} amp NRMSE {e:6.2f} %  '
              f"k1 {R[nm]['k1'] or float('nan'):7.1f}")
    CR._LIB.clear(); gc.collect()

oo = PR.rec_gp_only(x, sel, A[sel], f,
                    ells=(1.5, 2.5, 4., 8., 15., 30., 60., 120.))
held = np.array([i for i in range(len(x)) if i not in set(sel)])
R['gp_logamp'] = dict(amp_nrmse=100 * np.linalg.norm(oo['A'][held] - A[held])
                      / np.linalg.norm(A[held]), k1=None)
print(f"    published log-amp gp                 amp NRMSE "
      f"{R['gp_logamp']['amp_nrmse']:6.2f} %")

pickle.dump(R, open(str(OUT) + '/recs_cplx.pkl', 'wb'))
print('\nwrote out/recs_cplx.pkl')

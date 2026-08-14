"""How the predictive sigma and the true error shrink as positions are added."""
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
import numpy as np, pickle
import cplxrec as CR
from setup_data import load as load_meas
from activemodemap.lowrank import classify_null_from_map, resonance_index

NS = [4, 5, 6, 8, 10, 12, 16, 20, 30]
LIBS = {'eb_gp': str(OUT) + '/eblib_cplx.npz',
        'fem_gp': str(OUT) + '/femlib_isotropic_cplx.npz'}
x, f, Z, _, _ = load_meas('A', 'calibrated'); f, Z = f[::4], Z[:, ::4]
ir = resonance_index(f, Z); cl = classify_null_from_map(x, Z, f, ir)
TD = cl['x_null_um'] if np.isfinite(cl['x_null_um']) else cl['x_bound_um']
kres = int(np.argmax(np.abs(Z).max(0)))
SELS = {n: sorted(set(np.argmin(np.abs(x[:, None] -
        np.linspace(x.min(), x.max(), n)[None, :]), axis=0).tolist()))
        for n in NS}
rows = []


def run(key, fn, kw):
    for n in NS:
        sel = SELS[n]
        held = np.array([i for i in range(len(x)) if i not in set(sel)])
        o = fn(x, sel, Z[sel], f, **kw)
        m, _ = CR.score_cplx(x, sel, o, Z, f, TD)
        e = np.abs(o['Zrec'][held] - Z[held]) / np.sqrt(2)
        rows.append(dict(n=len(sel), rec=key,
                         sd=float(np.median(o['sd'][held][:, kres])),
                         sd_par=float(np.median(o['sd_par'][held][:, kres])),
                         sd_gp=float(np.median(o['sd_gp'][held][:, kres])),
                         err=float(np.sqrt(np.mean(e[:, kres] ** 2))),
                         z_rms=m['z_rms'], cov1=m['cov1s'],
                         nrmse=m['nrmse_cplx'], amp=m['nrmse_amp'],
                         phase=m['phase_mae_deg'], dns=m['dns'],
                         k1=(o['theta'] or {}).get('k1')))
        print(f"  {key:8s} n={n:3d}  cplx {m['nrmse_cplx']:6.2f}%  "
              f"z {m['z_rms']:5.2f}  dns {m['dns']:7.2f}", flush=True)


run('gp', CR.rec_gp_cplx, {})
for key, lib in LIBS.items():
    print(CR.load(lib), flush=True)
    run(key, CR.rec_phys_cplx, {})
    CR._LIB.clear(); gc.collect()
pickle.dump(rows, open(str(OUT) + '/sweep_n.pkl', 'wb'))
print('wrote out/sweep_n.pkl', len(rows), 'rows')

"""GP-only arm: same kernel machinery as the discrepancy GP, no forward model.

Two variants:
  gp        -- length scale chosen by leakage-safe PRESS on the revealed block.
               This is the deployable number.
  gp_oracle -- length scale chosen by minimising the HELD-OUT error. This is an
               ORACLE DIAGNOSTIC, not a usable method: it is here only to
               separate "the GP cannot represent this field" from "PRESS cannot
               pick a length scale from 3 points".
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

import numpy as np, pandas as pd, time, sys
import physrec as PR
from harness import STRATS
from run_phys import prep, truth_dns, score

ELLS = (1.5, 2.5, 4., 8., 15., 30., 60., 120.)

if __name__ == '__main__':
    NRAND = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    x, f, A = prep(); TD, ir = truth_dns(x, f, A)
    print(f'{len(x)} pos x {len(f)} freq   truth D-NS {TD:.2f} um', flush=True)
    seeds = sorted({0, len(x) - 1, len(x) // 2})
    NS = [3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 30, 50]
    rows = []
    for sn in ['equispaced', 'dopt_lowrank', 'random', 'maxvar']:
        reps = NRAND if sn == 'random' else 1
        for rep in range(reps):
            rng = np.random.default_rng(2000 + rep)
            for n in NS:
                sel = sorted(set(STRATS[sn](
                    x, n, seeds, rank=4, rng=rng,
                    Z_reveal=lambda q: A[list(q)].astype(complex),
                    freq=f, ires=ir)))
                if len(sel) < 3:
                    continue
                t0 = time.perf_counter()
                o = PR.rec_gp_only(x, sel, A[sel], f, ells=ELLS)
                dt = time.perf_counter() - t0
                m = score(x, sel, o['A'], A, f, TD)
                m.update(rec='gp', strategy=sn, rep=rep, n=len(sel), t_fit=dt,
                         ell=o['ell'], k1=None, f_free=None, g=None, rank=None)
                rows.append(m)
                # --- oracle diagnostic: best possible length scale
                best = None
                for e in ELLS:
                    oo = PR.rec_gp_only(x, sel, A[sel], f, ells=(e,))
                    mm = score(x, sel, oo['A'], A, f, TD)
                    if best is None or mm['amp_map_pct'] < best[0]:
                        best = (mm['amp_map_pct'], mm, e)
                m2 = dict(best[1])
                m2.update(rec='gp_oracle', strategy=sn, rep=rep, n=len(sel),
                          t_fit=dt, ell=best[2], k1=None, f_free=None, g=None,
                          rank=None)
                rows.append(m2)
        print(f'  {sn} done ({reps} rep)', flush=True)
    pd.DataFrame(rows).to_csv(str(OUT) + '/gp_bench.csv', index=False)
    print('wrote out/gp_bench.csv rows', len(rows))

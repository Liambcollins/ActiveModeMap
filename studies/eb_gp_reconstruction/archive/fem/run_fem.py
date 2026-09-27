"""FEM arm of the AL replay: paradigm 1b (FEM) and 1b+GP.

Same reveal loop, same scorer, same acquisition strategies as run_phys.py --
only the forward-model library differs.  physrec.use_library() is called ONCE,
before any fit, because it rebinds module globals: a half-swapped physrec would
silently mix EB and FEM forward models.

Leakage rule is unchanged: the reconstructors see only (sel_idx, A_sel, freq).
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

if __name__ == '__main__':
    from config import LIBRARIES
    lib = sys.argv[1] if len(sys.argv) > 1 else str(LIBRARIES['fem'])
    NRAND = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    tag = sys.argv[3] if len(sys.argv) > 3 else 'fem'
    out = sys.argv[4] if len(sys.argv) > 4 else str(OUT) + '/fem_bench.csv'

    print(PR.use_library(lib), flush=True)
    x, f, A = prep()
    TD, ir = truth_dns(x, f, A)
    print(f'{len(x)} pos x {len(f)} freq   ground-truth D-NS {TD:.2f} um',
          flush=True)
    seeds = sorted({0, len(x) - 1, len(x) // 2})
    RECS = {tag: PR.rec_eb, tag + '_gp': PR.rec_eb_gp}
    NS = [3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 30, 50]
    rows = []
    for rn, rf in RECS.items():
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
                    o = rf(x, sel, A[sel], f)
                    dt = time.perf_counter() - t0
                    m = score(x, sel, o['A'], A, f, TD)
                    th = o.get('theta') or {}
                    m.update(rec=rn, strategy=sn, rep=rep, n=len(sel), t_fit=dt,
                             k1=th.get('k1'), f_free=th.get('f_free'),
                             g=th.get('g'), rank=th.get('rank'),
                             ell=o.get('ell'))
                    rows.append(m)
            print(f'  {rn}/{sn} done ({reps} rep)', flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'wrote {out} rows {len(rows)}')

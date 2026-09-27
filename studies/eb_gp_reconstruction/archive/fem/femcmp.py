"""Merge the FEM arm into the paradigm comparison and report the crossovers."""
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

import numpy as np, pandas as pd

EB = pd.read_csv(str(OUT) + '/phys_bench.csv')
FE = pd.read_csv(str(OUT) + '/fem_bench.csv')
D = pd.concat([EB, FE], ignore_index=True)
ORDER = ['lowrank', 'eb', 'eb_gp', 'fem', 'fem_gp']
LBL = {'lowrank': '3 low-rank', 'eb': '1 EB', 'eb_gp': '2 EB+GP',
       'fem': '1b FEM', 'fem_gp': '1b FEM+GP'}


def table(metric, strategy='equispaced'):
    t = D[D.strategy == strategy].pivot_table(index='n', columns='rec',
                                              values=metric)
    return t.reindex(columns=[c for c in ORDER if c in t.columns])


def n_to_reach(metric, thresh, strategy='equispaced'):
    t = table(metric, strategy)
    out = {}
    for c in t.columns:
        ok = t.index[t[c] <= thresh]
        out[LBL[c]] = int(ok[0]) if len(ok) else None
    return out


if __name__ == '__main__':
    for metric, name in [('amp_map_pct', 'held-out complex-map NRMSE (%)'),
                         ('dns_err', 'D-NS absolute error (um)'),
                         ('peak_neartip_pct', 'near-tip peak error (%)')]:
        print(f'\n=== {name} -- equispaced ===')
        print(table(metric).round(2).to_string())

    print('\n=== positions needed to reach a target (equispaced) ===')
    for th in (10, 5, 3, 2):
        print(f'  NRMSE <= {th:>2} % : {n_to_reach("amp_map_pct", th)}')
    for th in (1.0, 0.5):
        print(f'  D-NS  <= {th:>3} um: {n_to_reach("dns_err", th)}')

    print('\n=== AL vs equispaced, per paradigm (NRMSE %, n=5) ===')
    piv = D[D.n == 5].pivot_table(index='rec', columns='strategy',
                                  values='amp_map_pct')
    print(piv.reindex([c for c in ORDER if c in piv.index]).round(2).to_string())

    print('\n=== random-design spread (NRMSE %, median [IQR]) ===')
    r = D[D.strategy == 'random']
    for rec in [c for c in ORDER if c in set(r.rec)]:
        s = r[r.rec == rec]
        row = []
        for n in (3, 5, 8, 12, 20):
            v = s[s.n == n].amp_map_pct
            if len(v):
                row.append(f'n={n}: {v.median():.1f} [{v.quantile(.25):.1f}-'
                           f'{v.quantile(.75):.1f}]')
        print(f'  {LBL[rec]:11s} ' + '  '.join(row))

    print('\n=== fitted contact stiffness (equispaced) ===')
    k = D[D.strategy == 'equispaced'].pivot_table(index='n', columns='rec',
                                                  values='k1')
    print(k.reindex(columns=[c for c in ORDER if c in k.columns]
                    ).round(0).to_string())
    print('  Hertz prediction 1310 N/m; mode-A shape fit 1000-2154 N/m')

    print('\n=== compute cost (median seconds per reconstruction) ===')
    print(D.groupby('rec').t_fit.median().reindex(
        [c for c in ORDER if c in set(D.rec)]).round(3).to_string())

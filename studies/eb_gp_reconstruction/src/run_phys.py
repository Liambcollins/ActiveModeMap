"""Physics-informed AL replay: EB / EB+GP / model-light low-rank, same target."""
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
from activemodemap.lowrank import classify_null_from_map, resonance_index
from setup_data import load as load_meas

FDEC=4
def prep(mode='A',axis='calibrated'):
    x,f,Z,ires,ps=load_meas(mode,axis)
    return x,f[::FDEC],np.abs(Z[:,::FDEC])          # AMPLITUDE, not log

def truth_dns(x,f,A):
    Zc=A.astype(complex); ir=resonance_index(f,Zc)
    cl=classify_null_from_map(x,Zc,f,ir)
    d=cl['x_null_um'] if np.isfinite(cl['x_null_um']) else cl['x_bound_um']
    return d,ir

def score(x,sel,Ap,At,f,tdns,near=10.0):
    held=np.array([i for i in range(len(x)) if i not in set(sel)])
    m={}
    m['amp_map_pct']=100*np.linalg.norm(Ap[held]-At[held])/np.linalg.norm(At[held])
    pt,pp=At[held].max(1),Ap[held].max(1)
    rel=np.abs(pp-pt)/pt
    m['peak_mean_pct'],m['peak_max_pct']=100*rel.mean(),100*rel.max()
    tip=held[x[held]>=x.max()-near]
    m['peak_neartip_pct']=100*float(np.mean(np.abs(Ap[tip].max(1)-At[tip].max(1))/At[tip].max(1))) if len(tip) else np.nan
    Zc=Ap.astype(complex); ir=resonance_index(f,Zc[sel])
    cl=classify_null_from_map(x,Zc,f,ir)
    d=cl['x_null_um'] if np.isfinite(cl['x_null_um']) else cl['x_bound_um']
    m['dns']=d; m['dns_err']=abs(d-tdns) if np.isfinite(d) else np.nan
    m['dns_status']=cl['status']; m['n_cross']=len(cl['crossings_um'])
    return m

if __name__=='__main__':
    import physrec as PR
    from harness import STRATS
    NRAND=int(sys.argv[1]) if len(sys.argv)>1 else 24
    x,f,A=prep(); TD,ir=truth_dns(x,f,A)
    print(f'{len(x)} pos x {len(f)} freq   ground-truth D-NS {TD:.2f} um',flush=True)
    seeds=sorted({0,len(x)-1,len(x)//2})
    RECS={'eb':PR.rec_eb,'eb_gp':PR.rec_eb_gp,'lowrank':PR.rec_lowrank_la}
    NS=[3,4,5,6,7,8,10,12,16,20,30,50]
    rows=[]
    for rn,rf in RECS.items():
        for sn in ['equispaced','dopt_lowrank','random','maxvar']:
            reps=NRAND if sn=='random' else 1
            for rep in range(reps):
                rng=np.random.default_rng(2000+rep)
                for n in NS:
                    sel=sorted(set(STRATS[sn](x,n,seeds,rank=4,rng=rng,
                                   Z_reveal=lambda q:A[list(q)].astype(complex),freq=f,ires=ir)))
                    if len(sel)<3: continue
                    t0=time.perf_counter(); out=rf(x,sel,A[sel],f); dt=time.perf_counter()-t0
                    m=score(x,sel,out['A'],A,f,TD)
                    th=out.get('theta') or {}
                    m.update(rec=rn,strategy=sn,rep=rep,n=len(sel),t_fit=dt,
                             k1=th.get('k1'),f_free=th.get('f_free'),g=th.get('g'),
                             rank=th.get('rank'),ell=out.get('ell'))
                    rows.append(m)
            print(f'  {rn}/{sn} done ({reps} rep)',flush=True)
    pd.DataFrame(rows).to_csv(str(OUT) + '/phys_bench.csv',index=False)
    print('wrote out/phys_bench.csv rows',len(rows))

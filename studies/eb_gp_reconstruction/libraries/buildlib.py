"""Precompute an EB response library over (k1, g, Omega) x position.

Parametrisation: at fixed k_lever the EB solution depends only on alpha = k1/k
and the dimensionless frequency Omega = f/f_free1.  Scaling the beam mass at
fixed (k, k1) scales every frequency by the same factor (verified to 0.1%), so
one library at a reference f_free1 serves every trial f_free1 by rescaling the
frequency axis.  That turns each fit into pure interpolation.
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

import json, time, sys, numpy as np
from geometry import ProbeGeometry
import eb_models as EB, eb_cr_afm as E
G=json.load(open(str(EB_GEOMETRY)))
BASE={k:G[k] for k in ['name','L','contact_x','tip_offset','H','thickness','w_top','w_bot','area','I','E','rho']}
P=ProbeGeometry(**BASE)
F_REF=P.f_free(1)                     # 74795 Hz reference free resonance
CX=P.contact_x*1e6                    # 225.0 um
MEAS_CX=214.1                         # our lever's clamp->tip distance
# target positions: our 101 calibrated positions, expressed on the MODEL beam via eta
import pandas as pd, activemodemap as amm
dl=pd.read_csv(str(DENSE_GRID_A)+'/DenseReference_log.csv',parse_dates=['timestamp'])
dl['i']=dl.tune_file.str.extract(r'_(\d{4})\.txt$')[0].astype(int)
ps=amm.fit_position_scale(dl[dl.i<20],dl[dl.i>=20],verbose=False)
x_meas=ps.apply(np.load(str(DENSE_GRID_A)+'/DenseReference.npz')['x_um'].astype(float))
eta=x_meas/MEAS_CX
x_model=eta*CX*1e-6                   # metres on the model beam
K1=np.geomspace(150.,20000.,96)
GG=np.geomspace(5000.,20000.,4)
FLIB=np.linspace(185e3,720e3,1600)    # library frequency axis (= Omega*F_REF)
LA=np.zeros((len(K1),len(GG),len(FLIB),len(x_model)),dtype=np.float32)
t0=time.perf_counter(); done=0; tot=len(K1)*len(GG)*len(FLIB)
for ik,k1 in enumerate(K1):
    for ig,g in enumerate(GG):
        cfg=EB.EBConfig(probe=P,model='2seg',phi=np.deg2rad(11.),k1=float(k1),k2=0.,
                        g=float(g),n_points=600)
        c=cfg.cantilever(); con=cfg.contact(); exc=cfg.excitation('mech')
        for jf,fq in enumerate(FLIB):
            r=E.solve_single_frequency(float(fq),c,con,exc,cfg.options(fq,fq,1.),obs_x=c.contact_x)
            xm=np.asarray(r['x']); zz=np.abs(np.asarray(r['z']))
            LA[ik,ig,jf]=np.log(np.interp(x_model,xm,zz)+1e-300)
            done+=1
        el=time.perf_counter()-t0
        print(f'  k1={k1:8.1f} g={g:7.0f}  {done}/{tot}  {el:6.1f}s  eta {el/done*(tot-done):6.1f}s',flush=True)
np.savez_compressed(str(OUT) + '/eblib.npz',K1=K1,GG=GG,FLIB=FLIB,F_REF=F_REF,
                    x_model=x_model,x_meas=x_meas,eta=eta,logamp=LA,k_lever=P.k_static)
print('saved out/eblib.npz  logamp',LA.shape,f'{LA.nbytes/1e6:.0f} MB raw')

"""Physics-informed reconstructors for the retrospective AL replay.

Leakage rule: every reconstructor sees ONLY (sel_idx, LA_sel, freq).  The scorer
alone sees the full matrix.  k1, f_res, g and the GP length scale are refitted
from the revealed points at every reveal step.

Library trick: the EB solution depends only on alpha = k1/k_lever and the
dimensionless frequency, so one library serves every trial f_free1 by rescaling
frequency.  Crucially the library is resampled into a RESONANCE-ALIGNED frame
u = f / f_res(k1) before any interpolation -- otherwise interpolating across k1
mixes peaks sitting at different frequencies and the error blows up near
resonance (measured: 3.8 % rms, 42 % worst, vs 0.2 % after alignment).
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

import numpy as np
from scipy.optimize import minimize_scalar

def _peak(v, _FW=None):
    fw = _FW if _FW is not None else globals()['_FW']
    j=int(np.argmax(v))
    if 0<j<len(v)-1:
        a,b,c=v[j-1],v[j],v[j+1]; den=a-2*b+c
        if abs(den)>1e-30: return fw[j]+0.5*(a-c)/den*(fw[j]-fw[j-1])
    return fw[j]

# The EB library is the historical default, but it is optional: `use_library`
# rebinds every global this module touches, so an arm that supplies its own
# library (FEM, or an EB variant) does not need eblib.npz to exist.  Without
# this guard importing physrec on a machine that has only the FEM ladder fails
# at import time, before use_library can be called.
K1=GG=FLIB=UG=LOGU=RES=RES_K=LK1=LGG=None
NK=NG=NF=NP=0
F_REF=float('nan')
_W=_FW=None
import os as _os2
if _os2.path.exists(str(OUT) + '/eblib.npz'):
    _L=np.load(str(OUT) + '/eblib.npz')
    K1,GG,FLIB,F_REF=_L['K1'],_L['GG'],_L['FLIB'],float(_L['F_REF'])
    _RAW=_L['logamp']                                # (nk, ng, nf_lib, npos)
    NK,NG,NF,NP=_RAW.shape
    LK1,LGG=np.log(K1),np.log(GG)
    # ---- mode-1 resonance per rung, in a WINDOW (mode 2 outruns mode 1 at
    # ---- soft contact under the F = k1*d1 drive), parabolically refined
    _W=(FLIB>=250e3)&(FLIB<=460e3)
    _FW=FLIB[_W]
    RES=np.zeros((NK,NG))
    for i in range(NK):
        for j in range(NG): RES[i,j]=_peak(_RAW[i,j][_W].max(1))
    RES_K=RES.mean(1)
    # ---- resample into the resonance-aligned frame u = f / f_res ----
    UG=np.linspace(0.55,2.05,1700)
    LOGU=np.zeros((NK,NG,len(UG),NP),dtype=np.float32)
    for i in range(NK):
        for j in range(NG):
            ft=UG*RES[i,j]
            idx=np.clip(np.searchsorted(FLIB,ft)-1,0,NF-2)
            w=((ft-FLIB[idx])/(FLIB[idx+1]-FLIB[idx]))[:,None]
            LOGU[i,j]=(1-w)*_RAW[i,j][idx]+w*_RAW[i,j][idx+1]
    del _RAW

def noise_floor(A_sel):
    """Local detector noise scale, estimated from the REVEALED block only.

    The measured band contains exact zeros, so log|Z| runs to -690 and a handful
    of bins dominate any least-squares log-amplitude objective (it dragged k1 and
    g to the library edges).  Working in T(A) = log(A + F) with F the noise scale
    treats model and data identically, compresses the nulls to a finite depth,
    and is the right transform for an amplitude with additive complex noise."""
    A=np.asarray(A_sel,float)
    return max(float(np.median(A.min(axis=1))),1e-9)

def T(A,F):  return np.log(np.asarray(A,float)+F)
def Tinv(t,F): return np.maximum(np.exp(np.asarray(t,float))-F,0.0)

def _ax(logv,q):
    q=float(np.clip(q,logv[0],logv[-1]))
    j=int(np.clip(np.searchsorted(logv,q)-1,0,len(logv)-2))
    return j,(q-logv[j])/(logv[j+1]-logv[j])

def _cubic_w(t):
    """Catmull-Rom weights for the 4 samples bracketing a fractional index t in [0,1)."""
    t2,t3=t*t,t*t*t
    return (-0.5*t3+t2-0.5*t, 1.5*t3-2.5*t2+1.0, -1.5*t3+2.0*t2+0.5*t, 0.5*t3-0.5*t2)

def eb_predict(log_k1,f_res,log_g,freq,rows=None):
    """log-amplitude map (npos, nfreq). f_res pins mode 1 to the observed peak,
    which fixes f_free1, so (k1, f_res) are near-orthogonal not degenerate.
    Cubic in log k1 (the mode shape turns over fastest there), linear in log g
    and in the resonance-aligned frequency u."""
    ik,wk=_ax(LK1,log_k1); ig,wg=_ax(LGG,log_g)
    u=np.clip(np.asarray(freq,float)/f_res,UG[0],UG[-1])
    ju=np.clip(np.searchsorted(UG,u)-1,0,len(UG)-2)
    wu=((u-UG[ju])/(UG[ju+1]-UG[ju]))[:,None]
    kw=_cubic_w(wk); kidx=[np.clip(ik-1+d,0,NK-1) for d in range(4)]
    out=0.0
    for a,wa in zip(kidx,kw):
        if wa==0.0: continue
        for b,wb in ((ig,1-wg),(ig+1,wg)):
            if wb==0.0: continue
            blk=LOGU[a,b] if rows is None else LOGU[a,b][:,rows]
            out=out+wa*wb*((1-wu)*blk[ju]+wu*blk[ju+1])
    return np.asarray(out).T

def f_free_of(log_k1,f_res):
    ik,wk=_ax(LK1,log_k1)
    return f_res/((1-wk)*RES_K[ik]+wk*RES_K[ik+1])*F_REF

def use_library(path):
    """Swap the forward model for a different response library, in place.

    Everything downstream of the resonance-aligned frame -- eb_predict, fit_eb,
    rec_eb, rec_eb_gp -- touches the library only through the module globals
    rebound here, so pointing this at build_fem_library.py's output runs the
    whole reveal loop against FEM instead of EB with no other change. That is
    the point of aligning both libraries onto the same u = f / f_res grid.

    Accepts either format:
      raw      K1, GG, FLIB, logamp     -- aligned here, as at import
      aligned  K1, GG, UG,   logamp_u   -- already aligned (the FEM builder)

    Returns a one-line description. Call it BEFORE any fit; the reveal loop
    caches nothing across calls, but a half-swapped module would silently mix
    two forward models.
    """
    global K1,GG,FLIB,F_REF,UG,LOGU,RES,RES_K,LK1,LGG,NK,NG,NF,NP,_W,_FW
    L=np.load(path)
    K1,GG=L['K1'],L['GG']
    LK1,LGG=np.log(K1),np.log(GG)
    F_REF=float(L['F_REF']) if 'F_REF' in L.files else float('nan')
    if 'logamp_u' in L.files:                     # pre-aligned (FEM)
        UG=L['UG']; LOGU=np.ascontiguousarray(L['logamp_u'])
        RES=L['RES']; RES_K=L['RES_K']
        NK,NG,_,NP=LOGU.shape; NF=len(UG); FLIB=None
        kind='pre-aligned'
    else:                                          # raw grid (EB)
        FLIB=L['FLIB']; raw=L['logamp']
        NK,NG,NF,NP=raw.shape
        _W=(FLIB>=250e3)&(FLIB<=460e3); _FW=FLIB[_W]
        RES=np.zeros((NK,NG))
        for i in range(NK):
            for j in range(NG): RES[i,j]=_peak(raw[i,j][_W].max(1))
        RES_K=RES.mean(1)
        UG=np.linspace(0.55,2.05,1700)
        LOGU=np.zeros((NK,NG,len(UG),NP),dtype=np.float32)
        for i in range(NK):
            for j in range(NG):
                ft=UG*RES[i,j]
                idx=np.clip(np.searchsorted(FLIB,ft)-1,0,NF-2)
                w=((ft-FLIB[idx])/(FLIB[idx+1]-FLIB[idx]))[:,None]
                LOGU[i,j]=(1-w)*raw[i,j][idx]+w*raw[i,j][idx+1]
        kind='raw -> aligned here'
    if LOGU.shape[3]!=101:
        raise ValueError(f"{path}: library has {LOGU.shape[3]} positions, the "
                         "measured grid has 101 -- rebuild with --x-meas")
    nan=float(np.mean(~np.isfinite(LOGU)))
    return (f"{path}: {kind}, k1 {K1[0]:.0f}-{K1[-1]:.0f} N/m ({NK} rungs) x "
            f"{NG} damping, f_res {RES_K[0]/1e3:.1f}-{RES_K[-1]/1e3:.1f} kHz, "
            f"{nan*100:.2f} % NaN")

def snr_window(A_sel,freq,frac=0.10):
    """High-SNR frequency window around the contact resonance, from the revealed
    data only.  EB reproduces the resonant mode shape (2.8-8.8 % norm-relative)
    but NOT the full off-resonance response (51-63 % at every k1), so fitting the
    whole band drives k1 to the library edge.  Restricting to the resonance window
    is what scripts/eb_fit_real.py does by cropping to the mode band."""
    p=np.asarray(A_sel,float).max(axis=0)
    return p>=frac*p.max()

def fit_eb(sel_idx,A_sel,freq,F=None,coarse=(16,11,5),refine=3,win=None):
    sel=np.asarray(sel_idx); A_sel=np.asarray(A_sel,float)
    if F is None: F=noise_floor(A_sel)
    if win is None: win=snr_window(A_sel,freq)
    fw=freq[win]; Ao=A_sel[:,win]; Tobs=T(Ao,F)
    f_obs=float(fw[int(np.argmax(Ao.max(0)))])             # from revealed data only
    def cost(p):
        M=eb_predict(p[0],p[1],p[2],fw,rows=sel)            # log model amplitude
        w=Ao>3.0*F
        gn=float(np.mean(np.log(Ao[w])-M[w])) if w.sum()>8 else float(np.mean(Tobs-M))
        Tm=T(np.exp(M+gn),F)
        return float(np.mean((Tm-Tobs)**2)),gn
    best=(np.inf,None,0.0)
    for lk in np.linspace(LK1[0],LK1[-1],coarse[0]):
        for fr in f_obs*np.linspace(0.985,1.015,coarse[1]):
            for lg in np.linspace(LGG[0],LGG[-1],coarse[2]):
                c,gn=cost([lk,fr,lg])
                if c<best[0]: best=(c,[lk,fr,lg],gn)
    p=list(best[1])
    bnds=[(LK1[0],LK1[-1]),(f_obs*0.98,f_obs*1.02),(LGG[0],LGG[-1])]
    for _ in range(refine):
        for d,(lo,hi) in enumerate(bnds):
            r=minimize_scalar(lambda v,d=d:cost([*p[:d],v,*p[d+1:]])[0],
                              bounds=(lo,hi),method='bounded',options={'xatol':1e-5})
            p[d]=float(r.x)
    c,gn=cost(p)
    return dict(log_k1=p[0],f_res=p[1],log_g=p[2],gain=gn,rms=float(np.sqrt(c)),F=F,win=win,
                k1=float(np.exp(p[0])),g=float(np.exp(p[2])),f_free=f_free_of(p[0],p[1]))

def rec_eb(x_grid,sel_idx,A_sel,freq,loo_sd=True,**kw):
    F=noise_floor(A_sel); win=snr_window(A_sel,freq)
    th=fit_eb(sel_idx,A_sel,freq,F=F,win=win)
    M=eb_predict(th['log_k1'],th['f_res'],th['log_g'],freq)+th['gain']
    n=len(sel_idx)
    if loo_sd and n>=4:
        folds=list(range(n)) if n<=12 else list(np.linspace(0,n-1,12).astype(int))
        Ps=[]
        for j in folds:
            k=[i for i in range(n) if i!=j]
            t2=fit_eb([sel_idx[i] for i in k],A_sel[k],freq,F=F,win=win,coarse=(8,7,3),refine=1)
            Ps.append(eb_predict(t2['log_k1'],t2['f_res'],t2['log_g'],freq)+t2['gain'])
        sd=np.std(np.asarray(Ps),axis=0)+th['rms']*0.25
    else:
        sd=np.full_like(M,max(th['rms'],1e-6))
    return dict(A=Tinv(T(np.exp(M),F),F),Tmap=T(np.exp(M),F),F=F,
                sd=np.maximum(sd,1e-6),theta=th,ell=None)

# ------------------------------------------------------------- GP discrepancy
def _press(xs,R,ell,noise):
    d2=(xs[:,None]-xs[None,:])**2; v=float(np.var(R))+1e-18
    K=v*np.exp(-d2/(2*ell**2))+noise*np.eye(len(xs))
    Ki=np.linalg.inv(K); dg=np.clip(np.diag(Ki),1e-12,None)
    A=Ki@R
    return float(np.mean((A/dg[:,None])**2))

def rec_eb_gp(x_grid,sel_idx,A_sel,freq,ells=(4.,8.,15.,30.,60.,120.),**kw):
    base=rec_eb(x_grid,sel_idx,A_sel,freq)
    F=base['F']; sel=np.asarray(sel_idx)
    R=T(A_sel,F)-base['Tmap'][sel]; n=len(sel)
    if n<5: return dict(A=base['A'],Tmap=base['Tmap'],F=F,sd=base['sd'],theta=base['theta'],ell=None)
    xs=x_grid[sel]; noise=max(1e-4,float(np.var(R))*0.05)
    ell=min(ells,key=lambda e:_press(xs,R,e,noise))
    v=float(np.var(R))+1e-18
    K=v*np.exp(-((xs[:,None]-xs[None,:])**2)/(2*ell**2))+noise*np.eye(n)
    ks=v*np.exp(-((x_grid[:,None]-xs[None,:])**2)/(2*ell**2))
    corr=ks@np.linalg.solve(K,R)
    Tm=base['Tmap']+corr
    return dict(A=Tinv(Tm,F),Tmap=Tm,F=F,sd=base['sd'],theta=base['theta'],ell=ell)

# ------------------------------------------------- model-light low-rank baseline
from activemodemap.lowrank import chebyshev_basis
def rec_lowrank_la(x_grid,sel_idx,A_sel,freq,rank=None,ranks=(2,3,4,5,6,8,10),**kw):
    """Chebyshev in position, independent solve per frequency, in the SAME
    noise-floored frame as the physics models. Rank by LOO-PRESS."""
    F=noise_floor(A_sel); LA_sel=T(A_sel,F)
    n=len(sel_idx)
    if rank is None:
        best=(np.inf,2)
        for r in ranks:
            if r>n-2: continue
            B=chebyshev_basis(x_grid,r); Bs=B[sel_idx]
            Gi=np.linalg.pinv(Bs.T@Bs); coef=Gi@(Bs.T@LA_sel)
            e=Bs@coef-LA_sel
            h=np.einsum('ij,jk,ik->i',Bs,Gi,Bs)
            p=float(np.mean((e/np.clip(1-h,1e-9,None)[:,None])**2))
            if p<best[0]: best=(p,r)
        rank=best[1]
    B=chebyshev_basis(x_grid,rank); Bs=B[sel_idx]
    coef=np.linalg.lstsq(Bs,LA_sel,rcond=None)[0]
    M=B@coef
    res=Bs@coef-LA_sel; dof=max(n-rank,1)
    sig=np.sqrt((res**2).sum(0)/dof)
    Gi=np.linalg.pinv(Bs.T@Bs); lev=np.einsum('ij,jk,ik->i',B,Gi,B)
    return dict(A=Tinv(M,F),Tmap=M,F=F,sd=np.sqrt(np.outer(np.maximum(lev,0),sig**2))+1e-9,
                theta=dict(rank=rank),ell=None)

def rec_gp_only(x_grid,sel_idx,A_sel,freq,
                ells=(1.5,2.5,4.,8.,15.,30.,60.,120.),**kw):
    """Pure GP: identical kernel machinery to rec_eb_gp, no forward model.

    The ONLY difference from rec_eb_gp is the mean function -- a per-frequency
    constant fitted to the revealed positions, instead of the physics
    prediction.  Same T(A)=log(A+F) transform, same isotropic squared-exponential
    over position shared across frequencies, same PRESS length-scale selection,
    same noise rule.  So a gap between the two arms is attributable to the mean
    function, not to kernel bookkeeping.

    The length-scale grid is WIDER than rec_eb_gp's on purpose: the discrepancy
    GP only has to model a residual, while this one carries the whole field, so
    it is given shorter scales to work with rather than being handicapped.
    """
    F=noise_floor(A_sel); sel=np.asarray(sel_idx); n=len(sel)
    Ts=T(A_sel,F)
    m=Ts.mean(axis=0)                     # per-frequency constant mean
    R=Ts-m[None,:]
    Tm=np.repeat(m[None,:],len(x_grid),axis=0)
    ell=None
    if n>=3:
        xs=x_grid[sel]; noise=max(1e-4,float(np.var(R))*0.05)
        ell=min(ells,key=lambda e:_press(xs,R,e,noise))
        v=float(np.var(R))+1e-18
        K=v*np.exp(-((xs[:,None]-xs[None,:])**2)/(2*ell**2))+noise*np.eye(n)
        ks=v*np.exp(-((x_grid[:,None]-xs[None,:])**2)/(2*ell**2))
        Tm=Tm+ks@np.linalg.solve(K,R)
    sd=np.full_like(Tm,float(np.std(R))+1e-6)
    return dict(A=Tinv(Tm,F),Tmap=Tm,F=F,sd=sd,theta=dict(ell=ell),ell=ell)

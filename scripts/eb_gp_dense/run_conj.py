import numpy as np, json
from eb_phase import *
from eb_gp_test import subset_indices, dns_um_from_end
from activemodemap.lowrank import reconstruct_map
from prep import load_dense, to_model_grid, estimate_sigma, OM1
def fwhm_Q(f,a):
    i=int(np.argmax(a)); h=a[i]/np.sqrt(2); j=i
    while j<len(a)-1 and a[j]>h: j+=1
    k=i
    while k>0 and a[k]>h: k-=1
    return f[i], f[i]/(f[j]-f[k])
dense=load_dense(); L=dense['L_um']; x_um=dense['x_um']
m=build_phase_model(L_um=L); xi,Zg0,_=to_model_grid(dense,m); 
Zg=np.conj(Zg0)                                  # <-- phase convention fixed
sigma=estimate_sigma(Zg)
f=m.omega/OM1*m.geom.f0_hz; cols=[int(np.argmin(np.abs(m.xi-v))) for v in xi]
truth=dns_um_from_end(x_um,Zg,f,L); norm=np.abs(Zg).max()
print('truth D-NS on conj data: %.2f um (was %.2f)'%(truth, dns_um_from_end(x_um,Zg0,f,L)))
cr=lambda M: float(np.sqrt(np.mean(np.abs(M-Zg)**2))/norm)
ip=int(np.argmin(np.abs(x_um-260)))
rows=[]
for n in (4,5,6,8,10,14,20):
    idx=subset_indices(x_um.size,n); data=[{'x':xi[i],'plus':Zg[i]} for i in idx]
    post=PhasePosterior(m,sigma=sigma,rng=np.random.default_rng(0)); post.fit(data)
    r=m.response(post.theta_map)
    eb=(r['A0']*r['phase']*(post._blur(r['piezo'])+r['eps']*post._blur(r['elec'])))[:,cols].T
    hy=PhaseHybrid(post).update(data).corrected_maps()[:,cols].T
    rank=min(4,len(idx)-2) if len(idx)>=6 else max(2,len(idx)-1)
    lr=reconstruct_map(x_um,idx,Zg[idx],rank=rank)['Zrec']
    fe,qe=fwhm_Q(f,np.abs(eb[ip])); fm_,qm=fwhm_Q(f,np.abs(Zg[ip]))
    t=post.theta_map
    row=dict(n=len(idx),theta=t.tolist(),crmse_eb=cr(eb),crmse_hy=cr(hy),crmse_lr=cr(lr),
             dns_eb=dns_um_from_end(x_um,eb,f,L,guess_um=L-truth),dns_hy=dns_um_from_end(x_um,hy,f,L,guess_um=L-truth),
             dns_lr=dns_um_from_end(x_um,lr,f,L,guess_um=L-truth),f_eb=fe/1e3,Q_eb=qe,peak=float(np.abs(eb[ip]).max()/np.abs(Zg[ip]).max()))
    rows.append(row)
    print('n=%2d | cplxRMSE EB %.4f EB+GP %.4f LR %.4f | D-NS EB %6.2f EB+GP %6.2f LR %6.2f | f_cr %.3f/%.3f Q %.0f/%.0f peak x%.2f | a=%.2f kc=%.2f Qc=%.2f eps=%.2f zeta=%.4f ph=%+.0f'%(
        row['n'],row['crmse_eb'],row['crmse_hy'],row['crmse_lr'],row['dns_eb'],row['dns_hy'],row['dns_lr'],fe/1e3,fm_/1e3,qe,qm,row['peak'],t[0],t[1],t[2],t[3],10**t[5],t[6]))
json.dump(dict(rows=rows,truth=truth),open('conj_results.json','w'))

import numpy as np, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from eb_phase import *
from eb_gp_test import build_model, SingleDomainPosterior, subset_indices, dns_um_from_end
from prep import load_dense, to_model_grid, estimate_sigma, OM1
plt.rcParams.update({'font.family':['Liberation Sans','DejaVu Sans'],'font.size':9,'axes.linewidth':0.8})
ORNL=(27/255,94/255,32/255); C_EB,C_HY,C_LR,C_TR,C_OLD='#b06000','#1f4e79','#2f9e5f','0.25','#999999'
old=json.load(open('zeta_results.json')) if False else None
base=json.load(open('conj_results.json')); rows=base['rows']; truth=base['truth']; N=[r['n'] for r in rows]
# "before" numbers (fixed zeta, raw phase) from the earlier run
before={4:(0.146,0.100,-7.60,0.38),6:(0.162,0.090,-1.73,0.99),8:(0.141,0.067,-8.23,0.17),10:(0.141,0.058,-8.31,0.80),14:(0.162,0.055,-1.73,0.95),20:(0.161,0.043,-1.73,0.53)}
dense=load_dense(); L=dense['L_um']; x_um=dense['x_um']; tip=L-x_um
m=build_phase_model(L_um=L); xi,Zg0,_=to_model_grid(dense,m); Zg=np.conj(Zg0); sigma=estimate_sigma(Zg)
f=m.omega/OM1*m.geom.f0_hz; cols=[int(np.argmin(np.abs(m.xi-v))) for v in xi]
n=10; idx=subset_indices(x_um.size,n); data=[{'x':xi[i],'plus':Zg[i]} for i in idx]
post=PhasePosterior(m,sigma=sigma,rng=np.random.default_rng(0)); post.fit(data)
r=m.response(post.theta_map); eb=(r['A0']*r['phase']*(post._blur(r['piezo'])+r['eps']*post._blur(r['elec'])))[:,cols].T
hy=PhaseHybrid(post).update(data).corrected_maps()[:,cols].T
# old fit for the spectrum overlay
m0=build_model(L_um=L); _,Zg_raw,_=to_model_grid(dense,m0); p0=SingleDomainPosterior(m0,sigma=estimate_sigma(Zg_raw),rng=np.random.default_rng(0))
p0.fit([{'x':xi[i],'plus':Zg_raw[i]} for i in idx]); r0=m0.response(p0.theta_map)
eb0=(r0['A0']*(p0._blur(r0['piezo'])+r0['eps']*p0._blur(r0['elec'])))[:,cols].T
ip=int(np.argmin(np.abs(x_um-260))); ires=int(np.argmax(np.abs(Zg[ip])))

fig=plt.figure(figsize=(12,7.4)); gs=fig.add_gridspec(2,2,hspace=0.45,wspace=0.27,left=0.07,right=0.98,top=0.85,bottom=0.08)
ax=fig.add_subplot(gs[0,0])
ax.semilogy(list(before),[before[k][0] for k in before],'o-',color=C_OLD,ms=4,label='EB as shipped (ζ fixed, raw phase)')
ax.semilogy(N,[r['crmse_eb'] for r in rows],'o-',color=C_EB,ms=4,label='EB: ζ fitted, phase fitted, conj. data')
ax.semilogy(N,[r['crmse_hy'] for r in rows],'o-',color=C_HY,ms=4,label='EB + GP (same)')
ax.semilogy(N,[r['crmse_lr'] for r in rows],'o-',color=C_LR,ms=4,label='low-rank')
ax.set_xlabel('measured positions'); ax.set_ylabel('complex map RMSE (norm.)')
ax.set_title('Map error: shipped vs corrected physics path',color=ORNL,fontweight='bold',fontsize=10.5,loc='left'); ax.legend(frameon=False,fontsize=8)
ax2=fig.add_subplot(gs[0,1]); ax2.axhline(0,color='0.6',lw=0.7); ax2.axhspan(-1,1,color='0.88',zorder=0)
ax2.plot(list(before),[before[k][2] for k in before],'o-',color=C_OLD,ms=4,label='EB as shipped')
ax2.plot(N,[r['dns_eb']-truth for r in rows],'o-',color=C_EB,ms=4,label='EB corrected')
ax2.plot(N,[r['dns_hy']-truth for r in rows],'o-',color=C_HY,ms=4,label='EB + GP')
ax2.plot(N,[r['dns_lr']-truth for r in rows],'o-',color=C_LR,ms=4,label='low-rank')
ax2.set_xlabel('measured positions'); ax2.set_ylabel('D-NS error (µm)')
ax2.set_title('Null vs dense truth (%.2f µm from tip)'%truth,color=ORNL,fontweight='bold',fontsize=10.5,loc='left'); ax2.legend(frameon=False,fontsize=8,ncol=2)
ax3=fig.add_subplot(gs[1,0])
ax3.plot(f/1e3,np.abs(Zg[ip]),color=C_TR,lw=2,label='measured')
ax3.plot(f/1e3,np.abs(eb0[ip]),color=C_OLD,lw=1.1,label='EB as shipped')
ax3.plot(f/1e3,np.abs(eb[ip]),color=C_EB,lw=1.2,ls='--',label='EB corrected')
ax3.set_xlim(60,67); ax3.set_xlabel('frequency (kHz)'); ax3.set_ylabel('|response| (norm.)')
ax3.set_title('Line shape at x = 260 µm, n = %d'%n,color=ORNL,fontweight='bold',fontsize=10.5,loc='left'); ax3.legend(frameon=False,fontsize=8)
ax4=fig.add_subplot(gs[1,1])
w=30; i0=ires
for Z_,c,lab,lw in ((Zg0,C_OLD,'measured, raw Igor phase',1.6),(Zg,C_TR,'measured, conjugated',2.0),(eb,C_EB,'EB corrected',1.2)):
    ph=np.degrees(np.unwrap(np.angle(Z_[ip][i0-w:i0+w]))); ph-=ph[0]
    ax4.plot(f[i0-w:i0+w]/1e3,ph,color=c,lw=lw,ls='--' if lab.startswith('EB') else '-',label=lab)
ax4.set_xlabel('frequency (kHz)'); ax4.set_ylabel('phase change through resonance (deg)')
ax4.set_title('The sign convention: +158° measured vs −159° modelled',color=ORNL,fontweight='bold',fontsize=10.5,loc='left'); ax4.legend(frameon=False,fontsize=8)
fig.suptitle('Why bare EB looked wrong on the PPP-CONTAu — and what fixes it',color=ORNL,fontweight='bold',fontsize=13,x=0.07,ha='left',y=0.965)
fig.text(0.07,0.905,'Three changes: intrinsic damping ζ fitted (→ 0.0041, Q_int ≈ 120, not the hard-coded 0.002); a global instrument phase fitted; measured spectra conjugated to match the model\'s exp(±iωt) convention.\nComplex map RMSE of bare EB falls 7× (0.141 → 0.020). Remaining null offset (−1.7 µm) tracks the unpublished tip setback (10 µm assumed).',fontsize=8.5,style='italic',color='0.35')
fig.savefig('EB_corrected_summary.png',dpi=190); print('saved')

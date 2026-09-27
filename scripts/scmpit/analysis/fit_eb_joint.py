"""Joint EB fit: frequencies (CR1-3) + quasi-static shape (1/InvOLS) + CR1 shape (|P|_CR1)
+ CR1 Q. Then: can the model alone turn a CR1 amplitude into d33?"""
import numpy as np
from scipy.optimize import least_squares
from forward_model import EBForwardModel, ProbeGeometry

d = np.load('stage1_compact.npz'); e = np.load('eb_fit.npz')
X, inv = d['X'], d['invols']; f0 = float(d['f0_free']); VAC = float(d['vac'])
CR = np.array([float(d['cr1']), float(d['cr2']), float(d['cr3'])])
P_qs, P_cr1, Q_x, E_meas = e['P_qs'], e['P_cr1'], e['Q_x'], e['E_meas']
static_meas = 1.0 / inv
x0 = float(e['x0_pos']); L = float(e['L_true']); xt = X - x0
Q_meas = float(np.median(Q_x[1:]))
iL = int(np.argmax(X)); notip = np.arange(X.size) != iL

def model(p, f_list, L_um=L):
    la, lk, sb, lz, th = p
    g = ProbeGeometry(name='SCM-PIT', f0_hz=f0, k_lever=1.697, L_um=L_um, tip_setback_um=sb,
                      tip_height_um=th, tilt_deg=11.0)
    m = EBForwardModel(geom=g, n_modes=14, nx=453, nf=2)
    m.omega = np.asarray(f_list, float) / m.freq_scale_hz; m.f_hz = m.omega * m.freq_scale_hz
    theta = np.array([la, lk, 2.3, 0.0, 0.0, lz])
    return m, m.response(theta)

def peaks(p):
    m, r = model(p, np.linspace(150e3, 2.0e6, 2400))
    i = np.argmin(np.abs(m.xi - 0.55)); a = np.abs(r['piezo'][:, i])
    pk = sorted([k for k in range(1, a.size-1) if a[k] >= a[k-1] and a[k] >= a[k+1]], key=lambda k: -a[k])
    out = []
    for k in pk:
        if all(abs(m.f_hz[k]-m.f_hz[o]) > 60e3 for o in out): out.append(k)
        if len(out) == 3: break
    return np.sort(m.f_hz[out]) if len(out) == 3 else None

def shapes(p, f1):
    m, r = model(p, [30e3, f1])
    xi_um = m.xi * L
    zq = np.interp(xt, xi_um, np.abs(r['piezo'][0])); zc = np.interp(xt, xi_um, np.abs(r['piezo'][1]))
    eq = np.interp(xt, xi_um, np.abs(r['elec'][0])); ec = np.interp(xt, xi_um, np.abs(r['elec'][1]))
    return zq, zc, eq, ec, m, r

def q_model(p, f1):
    m, r = model(p, np.linspace(f1-10e3, f1+10e3, 1201))
    i = np.argmin(np.abs(m.xi - 0.55)); a = np.abs(r['piezo'][:, i]); j = int(np.argmax(a)); h = a[j]/np.sqrt(2)
    lo = j
    while lo > 0 and a[lo] > h: lo -= 1
    hi = j
    while hi < a.size-1 and a[hi] > h: hi += 1
    return m.f_hz[j] / (m.f_hz[hi]-m.f_hz[lo])

def resid(p):
    fk = peaks(p)
    if fk is None: return np.full(3 + 7 + 7 + 1, 5.0)
    zq, zc, *_ = shapes(p, fk[0])
    g = np.sum(static_meas*zq)/np.sum(zq*zq)
    r_static = np.log(static_meas/(g*zq))[notip]               # shape only
    E_mod = zc / zq
    r_E = np.log(E_meas/E_mod)[notip]                          # ABSOLUTE enhancement, no free scale
    r_Q = np.log(q_model(p, fk[0]) / Q_meas)
    return np.concatenate([3*np.log(fk/CR), r_static, 0.5*r_E, [r_Q]])

best = None
for la in (2.4, 2.8, 3.2):
    for lk in (1.0, 2.0, 3.0):
        for th in (10.0, 15.0):
            p0 = [la, lk, 5.0, np.log10(2e-3), th]
            try:
                r = least_squares(resid, p0, bounds=([1.5, 0.0, 0.3, -3.5, 5.0], [5.0, 4.0, 40.0, -1.0, 25.0]),
                                  diff_step=2e-3, max_nfev=80)
            except Exception as ex:
                continue
            if best is None or r.cost < best.cost:
                best = r; print(f'  improved: cost {r.cost:.4f} from p0 {p0}')
p = best.x
fk = peaks(p); zq, zc, eq, ec, m, r = shapes(p, fk[0]); Qm = q_model(p, fk[0])
print(f'\nJOINT FIT: k*/k_lever={10**p[0]:.0f}, kcone/k_lever={10**p[1]:.1f}, setback={p[2]:.1f} um, '
      f'zeta={10**p[3]:.2e}, tip height={p[4]:.1f} um, L={L:.1f} um')
print(f'  CR1/2/3 model {fk/1e3} vs meas {CR/1e3}  ({100*(fk/CR-1)} %)')
print(f'  Q model {Qm:.0f} vs meas {Q_meas:.0f}')
g = np.sum(static_meas*zq)/np.sum(zq*zq)
print(f'  static shape rms {100*np.sqrt(np.mean(np.log(static_meas/(g*zq))[notip]**2)):.1f} %')
E_mod = zc/zq
print('\n  x_um   x_true   E_meas  E_model  ratio   1/InvOLS/model')
for i in np.argsort(X)[::-1]:
    print(f'{X[i]:7.1f} {xt[i]:7.1f} {E_meas[i]:8.1f} {E_mod[i]:8.1f} {E_meas[i]/E_mod[i]:6.3f}   {static_meas[i]/(g*zq[i]):6.3f}')

DIV = 32.0; amp_free = inv[iL]/DIV
Hq = zq/zq[iL]; Hc = zc/zq[iL]
d_qs_invols = P_qs*inv/DIV/VAC*1e12
d_cr1_model = P_cr1*amp_free/VAC/Hc*1e12          # ONLY free-end InvOLS + EB model
d_cr1_hybrid = P_cr1*inv/DIV/VAC/E_mod*1e12         # per-position InvOLS + EB enhancement
d_qs_model = P_qs*amp_free/VAC/Hq*1e12
print('\n  x_um   d_qs(InvOLS)   d_qs(EB)   d_CR1(EB only)   d_CR1(InvOLS+EB)')
for i in np.argsort(X)[::-1]:
    print(f'{X[i]:7.1f} {d_qs_invols[i]:11.2f} {d_qs_model[i]:10.2f} {d_cr1_model[i]:14.2f} {d_cr1_hybrid[i]:16.2f}')
for nm, v in [('qs InvOLS', d_qs_invols), ('qs EB', d_qs_model), ('CR1 EB', d_cr1_model), ('CR1 InvOLS+EB', d_cr1_hybrid)]:
    vv = v[notip]
    print(f'  {nm:14s} median {np.median(vv):6.2f}  spread {vv.max()/vv.min():5.2f}x  sd/med {100*np.std(vv)/np.median(vv):4.1f} %  (free end {v[iL]:.2f})')
np.savez('eb_joint.npz', p=p, fk=fk, Qm=Qm, xt=xt, zq=zq, zc=zc, eq=eq, ec=ec, E_mod=E_mod, g_static=g,
         d_qs_invols=d_qs_invols, d_qs_model=d_qs_model, d_cr1_model=d_cr1_model, d_cr1_hybrid=d_cr1_hybrid,
         xi_um=m.xi*L, zp_qs_full=np.abs(r['piezo'][0]), zp_cr1_full=np.abs(r['piezo'][1]),
         ze_qs_full=np.abs(r['elec'][0]), ze_cr1_full=np.abs(r['elec'][1]))

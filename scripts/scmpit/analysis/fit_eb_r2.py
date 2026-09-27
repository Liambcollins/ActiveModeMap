r"""Blind Euler-Bernoulli fit, Run 2.

The residual sees ONLY: the three contact-resonance frequencies, the shape of
1/InvOLS(x) (overall gain projected out) and the CR1 quality factor.  Every
amplitude -- and the measured enhancement E(x) -- is withheld, then used to score
the prediction.  This is the same estimator as the companion campaign
(fit_eb_noE.py); only the input file changes.
"""
import sys

import numpy as np
from scipy.optimize import least_squares

from forward_model import EBForwardModel, ProbeGeometry

d = np.load('/home/claude/scmpit_analysis/r2_eb_inputs.npz')
X, inv = d['X'], d['invols']
f0 = float(d['f0_free']); VAC = float(d['vac']); KLEV = float(d['k_lever'])
CR = d['CR']
P_qs, P_cr1, Q_x, E_meas = d['P_qs'], d['P_cr1'], d['Q_x'], d['E_meas']
static_meas = d['static_meas']

CASE = sys.argv[1] if len(sys.argv) > 1 else 'primary'
if CASE == 'altL':
    x0, L = float(d['x0_alt']), float(d['L_alt'])
else:
    x0, L = float(d['x0_pos']), float(d['L_true'])
xt = X - x0
iL = int(np.argmax(X)); notip = np.arange(X.size) != iL
Q_meas = float(np.median(Q_x[notip]))
print(f'[{CASE}] x0 {x0:+.2f} um, L {L:.1f} um, Q_meas {Q_meas:.0f}, '
      f'CR {CR/1e3} kHz, f0_free {f0/1e3:.2f} kHz')


def model(p, f_list, L_um=L):
    la, lk, sb, lz, th = p
    g = ProbeGeometry(name='SCM-PIT', f0_hz=f0, k_lever=KLEV, L_um=L_um, tip_setback_um=sb,
                      tip_height_um=th, tilt_deg=11.0)
    m = EBForwardModel(geom=g, n_modes=14, nx=453, nf=2)
    m.omega = np.asarray(f_list, float) / m.freq_scale_hz
    m.f_hz = m.omega * m.freq_scale_hz
    theta = np.array([la, lk, 2.3, 0.0, 0.0, lz])
    return m, m.response(theta)


def peaks(p):
    m, r = model(p, np.linspace(150e3, 2.1e6, 2500))
    i = np.argmin(np.abs(m.xi - 0.55)); a = np.abs(r['piezo'][:, i])
    pk = sorted([k for k in range(1, a.size - 1) if a[k] >= a[k - 1] and a[k] >= a[k + 1]],
                key=lambda k: -a[k])
    out = []
    for k in pk:
        if all(abs(m.f_hz[k] - m.f_hz[o]) > 60e3 for o in out):
            out.append(k)
        if len(out) == 3:
            break
    return np.sort(m.f_hz[out]) if len(out) == 3 else None


def shapes(p, f1):
    m, r = model(p, [30e3, f1])
    xi_um = m.xi * L
    zq = np.interp(xt, xi_um, np.abs(r['piezo'][0]))
    zc = np.interp(xt, xi_um, np.abs(r['piezo'][1]))
    eq = np.interp(xt, xi_um, np.abs(r['elec'][0]))
    ec = np.interp(xt, xi_um, np.abs(r['elec'][1]))
    return zq, zc, eq, ec, m, r


def q_model(p, f1):
    m, r = model(p, np.linspace(f1 - 10e3, f1 + 10e3, 1201))
    i = np.argmin(np.abs(m.xi - 0.55)); a = np.abs(r['piezo'][:, i])
    j = int(np.argmax(a)); h = a[j] / np.sqrt(2)
    lo = j
    while lo > 0 and a[lo] > h:
        lo -= 1
    hi = j
    while hi < a.size - 1 and a[hi] > h:
        hi += 1
    return m.f_hz[j] / (m.f_hz[hi] - m.f_hz[lo])


def resid(p):
    fk = peaks(p)
    if fk is None:
        return np.full(3 + notip.sum() * 2 + 1, 5.0)
    zq, zc, *_ = shapes(p, fk[0])
    g = np.sum(static_meas * zq) / np.sum(zq * zq)
    r_static = np.log(static_meas / (g * zq))[notip]
    r_E = np.log(E_meas / (zc / zq))[notip]
    r_Q = np.log(q_model(p, fk[0]) / Q_meas)
    return np.concatenate([3 * np.log(fk / CR), r_static, 0.0 * r_E, [r_Q]])   # E withheld


best = None
for la in (2.4, 2.8, 3.2):
    for lk in (1.0, 2.0, 3.0):
        for th in (10.0, 15.0):
            p0 = [la, lk, 5.0, np.log10(2e-3), th]
            try:
                r = least_squares(resid, p0,
                                  bounds=([1.5, 0.0, 0.3, -3.5, 5.0], [5.0, 4.0, 40.0, -1.0, 25.0]),
                                  diff_step=2e-3, max_nfev=80)
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
                print(f'  improved: cost {r.cost:.4f} from p0 {p0}')

p = best.x
fk = peaks(p); zq, zc, eq, ec, m, r = shapes(p, fk[0]); Qm = q_model(p, fk[0])
print(f'\nBLIND FIT [{CASE}]: k*/k_lever={10**p[0]:.0f}, kcone/k_lever={10**p[1]:.1f}, '
      f'setback={p[2]:.2f} um, zeta={10**p[3]:.2e}, tip height={p[4]:.1f} um, L={L:.1f} um')
print(f'  CR1/2/3 model {np.round(fk/1e3,2)} vs meas {np.round(CR/1e3,2)} kHz  '
      f'({np.round(100*(fk/CR-1),2)} %)')
print(f'  Q model {Qm:.0f} vs meas {Q_meas:.0f}  ({100*(Qm/Q_meas-1):+.1f} %)')
g = np.sum(static_meas * zq) / np.sum(zq * zq)
print(f'  static shape rms {100*np.sqrt(np.mean(np.log(static_meas/(g*zq))[notip]**2)):.2f} %')

E_mod = zc / zq
print('\n  x_um  x_beam   E_meas  E_model   E_meas/E_model   static/model')
for i in range(X.size - 1, -1, -1):
    tag = '  <- free end (excluded)' if i == iL else ''
    print(f'{X[i]:6.1f} {xt[i]:7.1f} {E_meas[i]:8.1f} {E_mod[i]:8.1f} {E_meas[i]/E_mod[i]:12.3f}'
          f' {static_meas[i]/(g*zq[i]):14.3f}{tag}')
rr = (E_meas / E_mod)[notip]
print(f'  blind enhancement prediction: {rr.min():.3f}-{rr.max():.3f}x '
      f'(median {np.median(rr):.3f}, sd {100*rr.std()/rr.mean():.1f} %)')

DIV = 32.0
amp_free = inv[iL] / DIV
Hq = zq / zq[iL]; Hc = zc / zq[iL]
d_qs_invols = P_qs * inv / DIV / VAC * 1e12          # per-position InvOLS, quasi-static
d_qs_model = P_qs * amp_free / VAC / Hq * 1e12       # free-end InvOLS + EB shape
d_cr1_model = P_cr1 * amp_free / VAC / Hc * 1e12     # free-end InvOLS + EB, resonance channel
d_cr1_hybrid = P_cr1 * inv / DIV / VAC / E_mod * 1e12  # per-position InvOLS + EB enhancement
d_cr1_Q = P_cr1 * inv / DIV / VAC / Q_x * 1e12       # the naive E = Q shortcut

print('\n  x_um   d_qs(InvOLS)  d_qs(EB)  d_CR1(EB only)  d_CR1(InvOLS+EB)  d_CR1(E=Q)')
for i in range(X.size - 1, -1, -1):
    print(f'{X[i]:6.1f} {d_qs_invols[i]:12.2f} {d_qs_model[i]:9.2f} {d_cr1_model[i]:14.2f} '
          f'{d_cr1_hybrid[i]:17.2f} {d_cr1_Q[i]:11.2f}')
for nm, v in [('qs InvOLS', d_qs_invols), ('qs EB', d_qs_model), ('CR1 EB only', d_cr1_model),
              ('CR1 InvOLS+EB', d_cr1_hybrid), ('CR1 E=Q', d_cr1_Q)]:
    vv = v[notip]
    print(f'  {nm:14s} median {np.median(vv):7.2f}  spread x{vv.max()/vv.min():5.2f}  '
          f'sd/med {100*np.std(vv)/np.median(vv):5.1f} %   (free end {v[iL]:.2f})')

out = f'/home/claude/scmpit_analysis/r2_eb_fit{"_altL" if CASE == "altL" else ""}.npz'
np.savez(out, p=p, case=CASE, fk=fk, Qm=Qm, Q_meas=Q_meas, X=X, xt=xt, x0=x0, L=L,
         zq=zq, zc=zc, eq=eq, ec=ec, E_mod=E_mod, E_meas=E_meas, g_static=g,
         d_qs_invols=d_qs_invols, d_qs_model=d_qs_model, d_cr1_model=d_cr1_model,
         d_cr1_hybrid=d_cr1_hybrid, d_cr1_Q=d_cr1_Q, Q_x=Q_x, invols=inv,
         xi_um=m.xi * L, zp_qs_full=np.abs(r['piezo'][0]), zp_cr1_full=np.abs(r['piezo'][1]),
         ze_qs_full=np.abs(r['elec'][0]), ze_cr1_full=np.abs(r['elec'][1]))
print('wrote', out)

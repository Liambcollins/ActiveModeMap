"""Does the EB transfer-function recovery of d33 work at CR2 and CR3?
Uses the BLIND fit (frequencies + Q_CR1 + static shape only; eb_joint_noE.npz)."""
import numpy as np
from forward_model import EBForwardModel, ProbeGeometry

d = np.load('stage1_compact.npz'); e = np.load('eb_fit.npz'); nb = np.load('eb_joint_noE.npz')
F, X, Z = d['F'], d['X'], d['Z'].astype(np.complex128)
bias, spot, inv = d['bias'], d['spot'], d['invols']; f0 = float(d['f0_free']); VAC = 1.0
CRm = {'CR1': float(d['cr1']), 'CR2': float(d['cr2']), 'CR3': float(d['cr3'])}
xt = e['xt']; L = float(e['L_true']); p = nb['p']; fk = nb['fk']
a1, b1, a2, b2 = e['a1'], e['b1'], e['a2'], e['b2']
P = (a1 - a2) / 2; B = (b1 + b2) / 2
iL = int(np.argmax(X)); order = np.argsort(X)[::-1]
qs = (F >= 15e3) & (F < 45e3); P_qs = np.median(np.abs(P[:, qs]), 1)
DIV = 32.0; amp_free = inv[iL] / DIV
d_qs = P_qs * inv / DIV / VAC * 1e12

def model(f_list):
    la, lk, sb, lz, th = p
    g = ProbeGeometry(name='SCM-PIT', f0_hz=f0, k_lever=1.697, L_um=L, tip_setback_um=sb, tip_height_um=th, tilt_deg=11.0)
    m = EBForwardModel(geom=g, n_modes=14, nx=453, nf=2)
    m.omega = np.asarray(f_list, float) / m.freq_scale_hz; m.f_hz = m.omega * m.freq_scale_hz
    return m, m.response(np.array([la, lk, 2.3, 0.0, 0.0, lz]))

def q_from(fr, amp):
    j = int(np.argmax(amp)); h = amp[j] / np.sqrt(2); lo = j
    while lo > 0 and amp[lo] > h: lo -= 1
    hi = j
    while hi < amp.size - 1 and amp[hi] > h: hi += 1
    return fr[j] / max(fr[hi] - fr[lo], 1), j

WIN = {'CR1': 12e3, 'CR2': 25e3, 'CR3': 60e3}
mq, rq = model([30e3]); xi_um = mq.xi * L
zq_full = np.abs(rq['piezo'][0]); zq = np.interp(xt, xi_um, zq_full)
out = {}
print(f'blind EB params: k*/k={10**p[0]:.0f}, kcone/k={10**p[1]:.1f}, setback={p[2]:.1f}, zeta={10**p[3]:.2e}, h={p[4]:.1f}')
for k, name in enumerate(['CR1', 'CR2', 'CR3']):
    fm = CRm[name]; w = np.abs(F - fm) <= WIN[name]
    # measured: peak of |P| in the window per position (mode may shift with x slightly), Q from FWHM
    Pk = np.zeros(X.size); Qx = np.zeros(X.size); fpk = np.zeros(X.size)
    for i in range(X.size):
        A = np.abs(P[i, w]); Qx[i], j = q_from(F[w], A); Pk[i] = A[j]; fpk[i] = F[w][j]
    Q_meas = float(np.median(Qx[np.argsort(Pk)[-5:]]))            # Q from the 5 strongest positions
    # model: mode shape at the model's own resonance, and the model Q there
    mm, rm = model(np.linspace(fk[k] - 3 * fk[k] / Q_meas, fk[k] + 3 * fk[k] / Q_meas, 1201))
    im = np.argmin(np.abs(mm.xi - (0.55 if k == 0 else 0.35)))
    Q_mod, jm = q_from(mm.f_hz, np.abs(rm['piezo'][:, im]))
    # per-position model amplitude AT the model peak frequency (peak of |z| over the sweep, per x)
    zc_full = np.abs(rm['piezo']).max(axis=0)
    zc = np.interp(xt, xi_um, zc_full)
    E_meas = Pk / P_qs; E_mod = zc / zq
    E_mod_Q = E_mod * (Q_meas / Q_mod)                            # per-mode Q correction
    Hc = zc / zq[iL]; Hc_Q = Hc * (Q_meas / Q_mod)
    d_blind = Pk * amp_free / VAC / Hc * 1e12
    d_blindQ = Pk * amp_free / VAC / Hc_Q * 1e12
    d_Qonly = Pk * inv / DIV / VAC / Qx * 1e12
    # node proximity: model |z| relative to its own max along the beam
    rel = zc / zc_full.max()
    good = rel > 0.25
    print(f'\n=== {name}: measured {fm/1e3:.1f} kHz (model {fk[k]/1e3:.1f}), Q_meas {Q_meas:.0f}, Q_model {Q_mod:.0f}')
    print('   x_true  |P|_pk(mV)  E_meas  E_model  E_mod*Qm/Qmod  rel|z|   d_qs   d(EB blind)  d(EB,Q-corr)  d(÷Q)')
    for i in order:
        flag = '' if good[i] else '  <- near node'
        print(f'  {xt[i]:6.1f}  {Pk[i]*1e3:8.3f}  {E_meas[i]:7.1f} {E_mod[i]:8.1f} {E_mod_Q[i]:12.1f}  {rel[i]:6.2f}  {d_qs[i]:6.2f}  {d_blind[i]:10.2f}  {d_blindQ[i]:11.2f}  {d_Qonly[i]:7.2f}{flag}')
    g = good.copy(); g[iL] = g[iL] and k > 0        # CR1 tip is the node
    for lab, v in [('EB blind', d_blind), ('EB Q-corr', d_blindQ), ('÷Q', d_Qonly)]:
        vv = v[g]
        print(f'   {lab:10s} away from nodes (n={g.sum()}): median {np.median(vv):6.2f}  spread {vv.max()/vv.min():5.2f}x  sd/med {100*np.std(vv)/np.median(vv):5.1f} %')
    out[name] = dict(f_meas=fm, f_model=float(fk[k]), Q_meas=Q_meas, Q_model=float(Q_mod), Pk=Pk, Qx=Qx, fpk=fpk,
                     E_meas=E_meas, E_mod=E_mod, E_mod_Q=E_mod_Q, rel=rel, good=g, d_blind=d_blind, d_blindQ=d_blindQ,
                     d_Qonly=d_Qonly, zc_full=zc_full, xi_um=xi_um)
np.savez('higher_modes.npz', **{f'{n}_{k}': v for n, dd in out.items() for k, v in dd.items()}, d_qs=d_qs, xt=xt, zq_full=zq_full)

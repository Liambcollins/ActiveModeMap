"""Fit the EB contact-resonance model to the SCM-PIT stage-1 data and derive the
CR1 -> quasi-static transfer, so d33 can be recovered from CR1 alone.

Outputs eb_fit.npz used by plot_stage1.py.
"""
import numpy as np
from scipy.optimize import least_squares
from forward_model import EBForwardModel, ProbeGeometry, _LAMBDAS

d = np.load('stage1_compact.npz')
F, X, Z = d['F'], d['X'], d['Z'].astype(np.complex128)
bias, spot, inv = d['bias'], d['spot'], d['invols']
CR = dict(cr1=float(d['cr1']), cr2=float(d['cr2']), cr3=float(d['cr3']))
f0, L, VAC = float(d['f0_free']), float(d['L_um']), float(d['vac'])
scale_hz = f0 / _LAMBDAS[0] ** 2

# ---- two-domain complex line fits -------------------------------------------------
def linefit(s):
    i = np.where(spot == s)[0]; V = bias[i]
    A = np.stack([np.ones_like(V), V], 1)
    Zs = Z[i]; nV, npos, nf = Zs.shape
    c, *_ = np.linalg.lstsq(A, Zs.reshape(nV, -1), rcond=None)
    return c[0].reshape(npos, nf), c[1].reshape(npos, nf)
a1, b1 = linefit(1); a2, b2 = linefit(2)
P = (a1 - a2) / 2; B = (b1 + b2) / 2

qs = (F >= 15e3) & (F < 45e3)
P_qs = np.median(np.abs(P[:, qs]), axis=1)
# CR1 peak per position, refined
w1 = (F >= 275e3) & (F <= 296e3)
j1 = np.array([np.argmax(np.abs(P[i, w1])) for i in range(X.size)])
f_cr1_x = F[w1][j1]
P_cr1 = np.abs(P[np.arange(X.size), np.where(w1)[0][j1]])
def q_of(i):
    A = np.abs(P[i, w1]); j = j1[i]; half = A[j] / np.sqrt(2)
    lo = j
    while lo > 0 and A[lo] > half: lo -= 1
    hi = j
    while hi < A.size - 1 and A[hi] > half: hi += 1
    return F[w1][j] / max(F[w1][hi] - F[w1][lo], 1)
Q_x = np.array([q_of(i) for i in range(X.size)])
E_meas = P_cr1 / P_qs
static_meas = 1.0 / inv                       # displacement sensitivity shape (arbitrary units)

# ---- EB model -----------------------------------------------------------------------
# SCM-PIT: Nanosensors, tip height 10-15 um; tilt 11 deg (Cypher holder)
def make_model(setback, L_um=L, tip_h=12.5, nf=None, f_list=None):
    g = ProbeGeometry(name='SCM-PIT', f0_hz=f0, k_lever=1.697, L_um=L_um,
                      tip_setback_um=setback, tip_height_um=tip_h, tilt_deg=11.0)
    m = EBForwardModel(geom=g, n_modes=14, nx=465, nf=2, omega_lo=1.0, omega_hi=2.0)
    if f_list is not None:                         # evaluate at exactly these Hz
        m.omega = np.asarray(f_list, float) / m.freq_scale_hz
        m.f_hz = m.omega * m.freq_scale_hz
    return m

F_QS = 30e3
def peaks(theta, setback):
    """model CR1..CR3 (Hz) from a fine sweep."""
    m = make_model(setback, f_list=np.linspace(150e3, 2.0e6, 3000))
    r = m.response(theta)
    i_mid = np.argmin(np.abs(m.xi - 0.55))
    amp = np.abs(r['piezo'][:, i_mid])
    pk = [k for k in range(1, amp.size - 1) if amp[k] >= amp[k - 1] and amp[k] >= amp[k + 1]]
    pk = sorted(pk, key=lambda k: -amp[k])
    out = []
    for k in pk:
        if all(abs(m.f_hz[k] - m.f_hz[o]) > 60e3 for o in out):
            out.append(k)
        if len(out) == 3: break
    return np.sort(m.f_hz[out])

def theta_of(p):
    log_alpha, log_kcone = p[0], p[1]
    return np.array([log_alpha, log_kcone, np.log10(200.0), 0.0, 0.0, np.log10(1e-3)])

def resid_freq(p):
    fk = peaks(theta_of(p), p[2])
    if fk.size < 3: return np.array([10, 10, 10.0])
    tgt = np.array([CR['cr1'], CR['cr2'], CR['cr3']])
    return np.log(fk / tgt)

best = None
for a0 in (2.0, 2.5, 3.0, 3.5):
    for k0 in (1.5, 2.0, 2.5, 3.0):
        for sb in (5.0, 10.0, 15.0):
            try:
                r = least_squares(resid_freq, [a0, k0, sb], bounds=([1.0, 1.0, 0.5], [5.0, 4.0, 40.0]),
                                  diff_step=1e-3, max_nfev=60)
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
p = best.x
fk = peaks(theta_of(p), p[2])
print(f'EB fit: log_alpha={p[0]:.3f} (k*/k_lever={10**p[0]:.0f}), log_kcone={p[1]:.3f}, setback={p[2]:.1f} um')
print(f'  model CR1/2/3 = {fk/1e3} kHz  vs measured {CR["cr1"]/1e3:.1f} {CR["cr2"]/1e3:.1f} {CR["cr3"]/1e3:.1f}')
print(f'  ratios model {fk[1]/fk[0]:.3f} {fk[2]/fk[0]:.3f}  measured {CR["cr2"]/CR["cr1"]:.3f} {CR["cr3"]/CR["cr1"]:.3f}')

# ---- shapes: model transfer at qs and CR1 along the beam ------------------------------
# model Q at CR1 (fine sweep, mid-lever) -- the E ratio scales with Q
mq = make_model(p[2], f_list=np.linspace(fk[0]-8e3, fk[0]+8e3, 1601))
rq = mq.response(theta_of(p)); iq = np.argmin(np.abs(mq.xi-0.55))
Aq = np.abs(rq['piezo'][:, iq]); jq = int(np.argmax(Aq)); half = Aq[jq]/np.sqrt(2)
lo = jq
while lo > 0 and Aq[lo] > half: lo -= 1
hi = jq
while hi < Aq.size-1 and Aq[hi] > half: hi += 1
Q_model = mq.f_hz[jq]/(mq.f_hz[hi]-mq.f_hz[lo]); Q_meas = float(np.median(Q_x[1:]))
print(f'model Q at CR1 = {Q_model:.0f}; measured Q (median, excl. free end) = {Q_meas:.0f}')
m = make_model(p[2], f_list=[F_QS, fk[0], CR['cr1']])
r = m.response(theta_of(p))
xi_um = m.xi * L
zp_qs = np.abs(r['piezo'][0]); zp_cr1 = np.abs(r['piezo'][1])
ze_qs = np.abs(r['elec'][0]); ze_cr1 = np.abs(r['elec'][1])

# position calibration, ANCHORED: the spot at x=232 is the free end (CR1 node collapse
# confirms it), so fit only the clamp offset x0 on the static end-loaded shape:
#   1/InvOLS(x) ~ (x-x0)^2 (3L - (x-x0)),  L = 232 - x0
def _shape(x, Lb, x0):
    xx = np.clip(x - x0, 1e-6, None); return xx ** 2 * (3 * Lb - xx)
def resid_x0(q):
    x0 = q[0]; Lb = L - x0
    sh = _shape(X, Lb, x0); g = np.sum(static_meas * sh) / np.sum(sh * sh)
    return np.log(static_meas / (g * sh))
rp = least_squares(resid_x0, [0.0], bounds=([-60.0], [45.0]))
x0_pos = float(rp.x[0]); s_pos = 1.0
L_true = L - x0_pos
xt = X - x0_pos
print(f'position calibration (anchored at the free end): clamp offset {x0_pos:+.2f} um -> lever {L_true:.1f} um, '
      f'rms {100*np.sqrt(np.mean(rp.fun**2)):.2f} %')
# rebuild the EB model on the calibrated length so xi*L_true is the true axis
m = make_model(p[2], L_um=L_true, f_list=[F_QS, fk[0], CR['cr1']])
r = m.response(theta_of(p))
xi_um = m.xi * L_true
zp_qs = np.abs(r['piezo'][0]); zp_cr1 = np.abs(r['piezo'][1])
ze_qs = np.abs(r['elec'][0]); ze_cr1 = np.abs(r['elec'][1])
def interp(shape, xt): return np.interp(xt, xi_um, shape)
rq_static = np.log(static_meas / (np.sum(static_meas*interp(zp_qs,xt))/np.sum(interp(zp_qs,xt)**2) * interp(zp_qs,xt)))
print(f'  EB quasi-static shape vs 1/InvOLS on the calibrated axis: rms {100*np.sqrt(np.mean(rq_static**2)):.2f} %')

E_model = interp(zp_cr1, xt) / interp(zp_qs, xt) * (Q_meas / Q_model)   # Q-rescaled
zp_cr1 = zp_cr1 * (Q_meas / Q_model)
# electrostatic/piezo pathway ratio shapes (arbitrary overall eps): compare to |b|/|P|(x)
eb_ratio_qs = interp(ze_qs, xt) / interp(zp_qs, xt)
eb_ratio_cr1 = interp(ze_cr1, xt) / interp(zp_cr1, xt)
print('\n  x_um   x_true   E_meas   E_model   ratio')
for i in np.argsort(X)[::-1]:
    print(f'{X[i]:7.1f} {xt[i]:7.1f} {E_meas[i]:8.1f} {E_model[i]:8.1f} {E_meas[i]/E_model[i]:7.3f}')

# ---- d33 recoveries --------------------------------------------------------------------
DIV = 32.0
iL = int(np.argmax(X))
amp_free = inv[iL] / DIV                                       # detector constant at the free end, m/V
d_qs_invols = P_qs * inv / DIV / VAC * 1e12                    # per-position InvOLS chain
d_cr1_Q = P_cr1 * inv / DIV / VAC / Q_x * 1e12                 # naive: divide by Q
# model chain: P(x) = d33 * VAC * H(x,f) * (1/AmpInvOLS_free) with H normalised to free-end qs
Hqs = interp(zp_qs, xt) / interp(zp_qs, xt[iL:iL+1])[0]
Hcr1 = interp(zp_cr1, xt) / interp(zp_qs, xt[iL:iL+1])[0]
d_qs_model = P_qs * amp_free / VAC / Hqs * 1e12
d_cr1_model = P_cr1 * amp_free / VAC / Hcr1 * 1e12
# hybrid: CR1 with per-position InvOLS and model E only (no static shape needed)
d_cr1_hybrid = P_cr1 * inv / DIV / VAC / E_model * 1e12

print('\n  x_um   d_qs(InvOLS)  d_qs(EB)  d_CR1/Q   d_CR1(EB)  d_CR1(InvOLS/E_EB)')
for i in np.argsort(X)[::-1]:
    print(f'{X[i]:7.1f} {d_qs_invols[i]:11.2f} {d_qs_model[i]:9.2f} {d_cr1_Q[i]:9.2f} {d_cr1_model[i]:10.2f} {d_cr1_hybrid[i]:12.2f}')
for nm, v in [('qs InvOLS', d_qs_invols), ('qs EB', d_qs_model), ('CR1/Q', d_cr1_Q), ('CR1 EB', d_cr1_model), ('CR1 hybrid', d_cr1_hybrid)]:
    print(f'  {nm:11s} median {np.median(v):6.2f}  spread {v.max()/v.min():5.2f}x  sd/med {100*np.std(v)/np.median(v):4.1f} %')

# ---- V_cpd, |b|/|P| per band ---------------------------------------------------------
def band_stats(mask, weight_by_b=True):
    a = (a1 + a2) / 2; b = B
    V = -(a[:, mask] / b[:, mask]).real
    w = np.abs(b[:, mask])
    v = (V * w).sum(1) / w.sum(1)
    # spread across the band as a rough error
    sd = np.sqrt(((V - v[:, None]) ** 2 * w).sum(1) / w.sum(1))
    ratio = (np.abs(b[:, mask]) * w).sum(1) / (np.abs(P[:, mask]) * w).sum(1)
    return v, sd, ratio
cr1m = np.zeros(F.size, bool)
for i in range(X.size):
    pass
cr1m = (np.abs(F - CR['cr1']) <= 4e3)
v_qs, sd_qs, r_qs = band_stats(qs)
v_c1, sd_c1, r_c1 = band_stats(cr1m)
gq = np.median(r_qs / eb_ratio_qs); gc = np.median(r_c1 / eb_ratio_cr1)
print('\n  x_um   Vcpd_qs   +/-    Vcpd_CR1  +/-   |b|/|P|_qs  EBpred   |b|/|P|_CR1  EBpred')
for i in np.argsort(X)[::-1]:
    print(f'{X[i]:7.1f} {v_qs[i]:+8.3f} {sd_qs[i]:5.2f}  {v_c1[i]:+8.3f} {sd_c1[i]:5.2f}   {r_qs[i]:8.4f} {gq*eb_ratio_qs[i]:8.4f}   {r_c1[i]:8.4f} {gc*eb_ratio_cr1[i]:8.4f}')
print(f'  (EB electrostatic/piezo shape scaled by one number per band: qs x{gq:.3g}, CR1 x{gc:.3g})')

np.savez('eb_fit.npz', L_true=L_true, eb_ratio_qs=eb_ratio_qs, eb_ratio_cr1=eb_ratio_cr1, Q_model=Q_model, Q_meas=Q_meas, X=X, xt=xt, s_pos=s_pos, x0_pos=x0_pos, theta=theta_of(p), setback=p[2],
         fk_model=fk, xi_um=xi_um, zp_qs=zp_qs, zp_cr1=zp_cr1, ze_qs=ze_qs, ze_cr1=ze_cr1,
         E_meas=E_meas, E_model=E_model, Q_x=Q_x, f_cr1_x=f_cr1_x, P_qs=P_qs, P_cr1=P_cr1,
         static_meas=static_meas, inv=inv,
         d_qs_invols=d_qs_invols, d_qs_model=d_qs_model, d_cr1_Q=d_cr1_Q, d_cr1_model=d_cr1_model,
         d_cr1_hybrid=d_cr1_hybrid, v_qs=v_qs, sd_qs=sd_qs, r_qs=r_qs, v_c1=v_c1, sd_c1=sd_c1, r_c1=r_c1,
         a1=a1, b1=b1, a2=a2, b2=b2, F=F)
print('\nsaved eb_fit.npz')

r"""Stage 8 close-out: end-of-campaign anchor vs. this morning's stage 1.

Compares the close-out reference repeat at position A (154.9 stage-um, 500 nN,
V_ac 1.000 V, bias +-9 V interleaved, both domains) against the IDENTICAL
conditions measured this morning in stage 1, and the close-out mode-shape walk
against stage 1's own 8-position grid.

Two channels are compared, deliberately:
  * quasi-static band (15-45 kHz plateau) -- needs only InvOLS/32, no resonance
    model, so it is the drift measurement with the fewest assumptions;
  * CR1 peak -- needs the enhancement, so it is reported as V_cpd / |b|/|P| /
    frequency rather than as an absolute pm/V.
"""
import glob
import json
import os

import numpy as np

UP = '/mnt/user-data/uploads/ActiveModeMap/DomainsB_SCMPIT'
REF = os.path.join(UP, 'closeout_ref_A_checkpoint.npz')
WALK = sorted(glob.glob(os.path.join(UP, 'closeout_wideband_*_checkpoint.npz')))
CAL = os.path.join(UP, 'closeout_cal.json')

X_A = 154.9
IX_A = 4                       # position index of 154.9 um in stage 1
VAC = 1.0
QS_BAND = (15e3, 45e3)         # the flat |P| plateau stage 1 used for quasi-static d33
CR1_WIN = (255e3, 325e3)       # wide enough for both 285.5 (am) and ~295.5 kHz (pm)
INVOLS_AM = 9.8193e-07         # stage 1, position A (from stage1_compact)
INVOLS_PM = 1.171e-6           # in force from stage 5 onward at this position


def amp_invols(invols):
    return invols / 32.0


def band_mean(F, Z, band):
    m = (F >= band[0]) & (F <= band[1])
    return Z[..., m].mean(axis=-1)


def cr1_peak(F, Z):
    m = (F >= CR1_WIN[0]) & (F <= CR1_WIN[1])
    fr, zz = F[m], Z[..., m]
    k = int(np.argmax(np.abs(zz)))
    return float(fr[k]), zz[k]


def decompose(bias, z1, z2):
    """Z_s(V) = a_s + b_s V for the two domains -> P, b, V_cpd."""
    A1 = np.polyfit(bias, z1, 1)          # [b1, a1]
    A2 = np.polyfit(bias, z2, 1)
    b1, a1 = A1[0], A1[1]
    b2, a2 = A2[0], A2[1]
    P = (a1 - a2) / 2.0
    b = (b1 + b2) / 2.0
    vcpd = -(a1 + a2) / (2.0 * b) if abs(b) > 0 else np.nan
    return dict(a1=a1, a2=a2, b1=b1, b2=b2, P=P, b=b,
                v_cpd=float(np.real(vcpd)),
                ratio=float(abs(b) / abs(P)) if abs(P) > 0 else np.nan,
                flip_deg=float(np.degrees(abs(np.angle(a1) - np.angle(a2)))),
                b_bal=float(abs(b2) / abs(b1)) if abs(b1) > 0 else np.nan)


def channels(F, Zc, bias, spot, label, invols):
    """Zc: (n_cond, n_freq) at ONE position. Returns qs and CR1 decompositions."""
    out = {'label': label, 'invols_m_per_V': invols}
    s1 = spot == 1
    s2 = spot == 2

    qs = band_mean(F, Zc, QS_BAND)
    d_qs = decompose(bias[s1], qs[s1], qs[s2])
    d_qs['d33_pm_per_V'] = float(abs(d_qs['P']) * amp_invols(invols) / VAC * 1e12)
    out['qs'] = d_qs

    f_pk, _ = cr1_peak(F, Zc[np.argmax(bias == 0)])
    m = (F >= CR1_WIN[0]) & (F <= CR1_WIN[1])
    k = int(np.argmin(np.abs(F[m] - f_pk)))
    zc = Zc[:, m][:, k]
    d_cr = decompose(bias[s1], zc[s1], zc[s2])
    d_cr['f_cr1_Hz'] = f_pk
    d_cr['amp_domain1_pm'] = float(abs(zc[s1][bias[s1] == 0][0]) * amp_invols(invols) * 1e12)
    d_cr['amp_domain2_pm'] = float(abs(zc[s2][bias[s2] == 0][0]) * amp_invols(invols) * 1e12)
    out['cr1'] = d_cr
    return out


def show(tag, d):
    q, c = d['qs'], d['cr1']
    print(f'\n  {tag}   (InvOLS {d["invols_m_per_V"]:.4g} m/V)')
    print(f'    quasi-static 15-45 kHz : d33 {q["d33_pm_per_V"]:6.2f} pm/V   '
          f'V_cpd {q["v_cpd"]:+6.3f} V   |b|/|P| {q["ratio"]:.3f}   '
          f'flip {q["flip_deg"]:.0f} deg   |b2/b1| {q["b_bal"]:.2f}')
    print(f'    CR1 peak {c["f_cr1_Hz"] / 1e3:7.2f} kHz : V_cpd {c["v_cpd"]:+6.3f} V   '
          f'|b|/|P| {c["ratio"]:.3f}   flip {c["flip_deg"]:.0f} deg   |b2/b1| {c["b_bal"]:.2f}')
    print(f'    CR1 |Z| at 0 V         : domain 1 {c["amp_domain1_pm"]:8.1f} pm   '
          f'domain 2 {c["amp_domain2_pm"]:8.1f} pm   ratio {c["amp_domain1_pm"] / c["amp_domain2_pm"]:.3f}')


# ---------------------------------------------------------------- morning ---
s1 = np.load('stage1_compact.npz', allow_pickle=True)
F_am = s1['F']
Z_am = s1['Z'][:, IX_A, :]
bias_am, spot_am = s1['bias'], s1['spot']
print(f'stage 1 (morning): {Z_am.shape[0]} conditions at x = {s1["X"][IX_A]} um, '
      f'F {F_am[0]:.0f}-{F_am[-1] / 1e3:.0f} kHz, {F_am.size} pts')

# ---------------------------------------------------------------- evening ---
d = np.load(REF, allow_pickle=True)
F_pm = d['freq_Hz']
Z_pm = d['Z'][:, 0, :]
cond_txt = str(d['conditions'])
print(f'stage 8 (close-out): {Z_pm.shape[0]} conditions at x = {d["x_um"][0]} um, '
      f'F {F_pm[0]:.0f}-{F_pm[-1] / 1e3:.0f} kHz, {F_pm.size} pts')

# take bias/spot from the checkpoint itself, then assert it matches stage 1's
# order -- the comparison is only meaningful if the two runs are condition-aligned.
_c = json.loads(cond_txt)
bias_pm = np.array([c['bias_V'] for c in _c], float)
spot_pm = np.array([c['spot'] for c in _c], int)
assert np.array_equal(bias_pm, bias_am) and np.array_equal(spot_pm, spot_am), \
    f'condition order differs:\n  morning {list(zip(bias_am, spot_am))}\n  closeout {list(zip(bias_pm, spot_pm))}'
print('condition order verified: close-out matches stage 1 exactly')

am = channels(F_am, Z_am, bias_am, spot_am, 'stage 1 (12:52, morning)', INVOLS_AM)
pm = channels(F_pm, Z_pm, bias_pm, spot_pm, 'stage 8 (22:19, close-out)', INVOLS_PM)

print('\n' + '=' * 78)
print('REFERENCE CONDITION, IDENTICAL SETTINGS, ~9.5 h APART  (position A, 500 nN, 1 V)')
print('=' * 78)
show('MORNING ', am)
show('CLOSE-OUT', pm)

print('\n  DRIFT')
print(f'    V_cpd  (quasi-static) : {am["qs"]["v_cpd"]:+.3f} -> {pm["qs"]["v_cpd"]:+.3f} V   '
      f'({pm["qs"]["v_cpd"] - am["qs"]["v_cpd"]:+.3f} V)')
print(f'    V_cpd  (CR1)          : {am["cr1"]["v_cpd"]:+.3f} -> {pm["cr1"]["v_cpd"]:+.3f} V   '
      f'({pm["cr1"]["v_cpd"] - am["cr1"]["v_cpd"]:+.3f} V)')
print(f'    d33    (quasi-static) : {am["qs"]["d33_pm_per_V"]:.2f} -> '
      f'{pm["qs"]["d33_pm_per_V"]:.2f} pm/V   '
      f'({100 * (pm["qs"]["d33_pm_per_V"] / am["qs"]["d33_pm_per_V"] - 1):+.1f} %)')
print(f'    CR1 frequency         : {am["cr1"]["f_cr1_Hz"] / 1e3:.2f} -> '
      f'{pm["cr1"]["f_cr1_Hz"] / 1e3:.2f} kHz   '
      f'({100 * (pm["cr1"]["f_cr1_Hz"] / am["cr1"]["f_cr1_Hz"] - 1):+.2f} %)')
print(f'    |b|/|P| (CR1)         : {am["cr1"]["ratio"]:.3f} -> {pm["cr1"]["ratio"]:.3f}')
print(f'    domain amp ratio (CR1): {am["cr1"]["amp_domain1_pm"] / am["cr1"]["amp_domain2_pm"]:.3f} -> '
      f'{pm["cr1"]["amp_domain1_pm"] / pm["cr1"]["amp_domain2_pm"]:.3f}')

# ------------------------------------------------------------------ walk ----
walk = None
if WALK:
    w = np.load(WALK[0], allow_pickle=True)
    Fw, Zw, xw = w['freq_Hz'], w['Z'][0], w['x_um']
    order = np.argsort(xw)
    xw = xw[order]; Zw = Zw[order]
    fw = np.zeros(xw.size); aw = np.zeros(xw.size)
    for j in range(xw.size):
        f, z = cr1_peak(Fw, Zw[j])
        fw[j], aw[j] = f, abs(z)
    # morning comparison: stage 1's own 8 positions at 0 V spot 1
    i0 = int(np.where((bias_am == 0) & (spot_am == 1))[0][0])
    xa = s1['X']; fa = np.zeros(xa.size); aa = np.zeros(xa.size)
    for j in range(xa.size):
        f, z = cr1_peak(F_am, s1['Z'][i0, j, :])
        fa[j], aa[j] = f, abs(z)
    print('\n' + '=' * 78)
    print('MODE-SHAPE WALK, 0 V / 500 nN / spot 1  (morning stage 1 vs close-out)')
    print('=' * 78)
    print(f'  {"x (um)":>8}  {"CR1 am (kHz)":>13}  {"CR1 pm (kHz)":>13}  {"d f (kHz)":>10}  '
          f'{"amp am":>10}  {"amp pm":>10}  {"pm/am":>7}')
    for j in range(xw.size):
        k = int(np.argmin(np.abs(xa - xw[j])))
        print(f'  {xw[j]:8.1f}  {fa[k] / 1e3:13.2f}  {fw[j] / 1e3:13.2f}  '
              f'{(fw[j] - fa[k]) / 1e3:+10.2f}  {aa[k]:10.5f}  {aw[j]:10.5f}  '
              f'{aw[j] / aa[k]:7.2f}')
    print(f'  mean CR1 shift {np.mean([(fw[j] - fa[int(np.argmin(np.abs(xa - xw[j])))]) for j in range(xw.size)]) / 1e3:+.2f} kHz;'
          f'  CR1 spread across the beam: morning {(fa.max() - fa.min()) / 1e3:.2f} kHz, '
          f'close-out {(fw.max() - fw.min()) / 1e3:.2f} kHz')
    walk = dict(x_um=xw.tolist(), f_closeout_Hz=fw.tolist(), amp_closeout=aw.tolist(),
                x_morning_um=xa.tolist(), f_morning_Hz=fa.tolist(), amp_morning=aa.tolist())

# --------------------------------------------------------------- detection --
cal = None
if os.path.exists(CAL):
    cal = json.load(open(CAL))
    p3 = cal.get('part3_detection') or {}
    print('\n' + '=' * 78)
    print('DETECTION CALIBRATION')
    print('=' * 78)
    print(f'  InvOLS at position A: pre-flight {p3.get("invols_preflight_m_per_V")}  ->  '
          f'close-out {p3.get("invols_final_m_per_V")}   '
          f'(x{p3.get("invols_ratio"):.3f})' if p3.get('invols_ratio') else '')
    print(f'  panel spring constant: {p3.get("spring_panel_N_per_m")} N/m '
          f'(campaign {p3.get("spring_campaign_N_per_m")} N/m)')
    fa_ = cal.get('freeair') or {}
    if fa_:
        print(f'  free-air tune: {fa_.get("note") or fa_.get("failed") or ""}')

np.save('stage8_summary.npy', dict(morning=am, closeout=pm, walk=walk, cal=cal),
        allow_pickle=True)
print('\nsaved stage8_summary.npy')

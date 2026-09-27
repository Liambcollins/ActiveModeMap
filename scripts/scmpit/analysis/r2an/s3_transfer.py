"""R2 cantilever transfer function: P_qs(x), P_CR1(x), Q(x), E(x), and the
position calibration. Everything the transfer-function section needs, from R2's
own bias survey plus the per-position InvOLS in the force-curve notes."""
import glob, json, os, re
import numpy as np
from igor2.binarywave import load as ibw_load
import common as C

R2 = C.R2
BIAS = os.path.join(R2, '10_bias_survey')
QS = (15e3, 45e3)
CR1W = (255e3, 330e3)


def note_val(note, key):
    for line in note.decode('latin-1', 'ignore').replace('\r', '\n').split('\n'):
        if ':' in line:
            k, v = line.split(':', 1)
            if k.strip() == key:
                try:
                    return float(v.strip())
                except ValueError:
                    return None
    return None


# --- per-position InvOLS from the force curve taken at that position ---------
# The pre-flight measured InvOLS at 17 positions on a 10 um grid; interpolate that
# (log-linear, it spans a decade and is smooth) onto the 8 survey positions. The
# force curves sitting in this folder are used only as a cross-check: file order is
# not a safe index, because the pre-flight's own final -- and rejected -- curve was
# written after its log closed and lands here by timestamp.
pf = json.load(open(os.path.join(R2, '01_preflight', 'preflight_result.json')))
pf_x = np.array(sorted(float(k) for k in pf['invols_by_x']))
pf_v = np.array([float(pf['invols_by_x'][f'{k:.1f}']) for k in pf_x])
def invols_at(xq):
    return float(np.exp(np.interp(xq, pf_x, np.log(pf_v))))

fcs = sorted(glob.glob(os.path.join(BIAS, 'FSCMPIT*.ibw')))
fc_v = [note_val(ibw_load(f)['wave']['note'], 'InvOLS') for f in fcs]
print(f'{len(fcs)} force curves in this folder; InvOLS read from their notes:')
for f, v in zip(fcs, fc_v):
    print(f'   {os.path.basename(f)}  {v:.4e}')

F, Z, x, cond = C.load_ckpt(os.path.join(BIAS, 'domains_bias_checkpoint_500nN_scmpit.npz'))
bias = np.array([c['bias_V'] for c in cond], float)
spot = np.array([c['spot'] for c in cond], int)
s1, s2 = spot == 1, spot == 2
# run_series stores positions ascending; the survey MEASURED free end first
inv_by_x = {float(xx): invols_at(float(xx)) for xx in x}
# cross-check: the survey's own force curves, in measurement order (free end first),
# ignoring any that fall outside the pre-flight range (the stray pre-flight curve)
meas = sorted(x)[::-1]
usable = [v for v in fc_v if v is not None and v < 1.05 * pf_v.max()]
print('\ncross-check, interpolated vs the force curve measured at that position:')
for xx, v in zip(meas, usable):
    print(f'   x {xx:6.1f}  interp {invols_at(xx):.4e}  force curve {v:.4e}  '
          f'ratio {v / invols_at(xx):.3f}')
if len(usable) < len(meas):
    print(f'   ({len(meas) - len(usable)} position(s) have no usable curve in this folder; '
          f'a boundary file landed in the next stage)')

qm = (F >= QS[0]) & (F <= QS[1])
cm = (F >= CR1W[0]) & (F <= CR1W[1])
i0 = int(np.where((bias == 0) & s1)[0][0])

rows = []
for j, xx in enumerate(x):
    zq = Z[:, j, qm].mean(axis=-1)
    dq = C.decompose(bias[s1], zq[s1], zq[s2])
    a = np.abs(Z[i0, j, cm]); k = int(np.argmax(a))
    f_cr1 = float(F[cm][k])
    zc = Z[:, j, cm][:, k]
    dc = C.decompose(bias[s1], zc[s1], zc[s2])
    # Q from the half-power width of |Z| at 0 V, spot 1
    half = a[k] / np.sqrt(2.0)
    lo = k
    while lo > 0 and a[lo] > half:
        lo -= 1
    hi = k
    while hi < a.size - 1 and a[hi] > half:
        hi += 1
    fwhm = float(F[cm][hi] - F[cm][lo])
    (bb1, aa1) = np.polyfit(bias[s1], zq[s1], 1)
    (bb2, aa2) = np.polyfit(bias[s2], zq[s2], 1)
    (cb1, ca1) = np.polyfit(bias[s1], zc[s1], 1)
    (cb2, ca2) = np.polyfit(bias[s2], zc[s2], 1)
    cx = dict(qs_a1=[aa1.real, aa1.imag], qs_b1=[bb1.real, bb1.imag],
              qs_a2=[aa2.real, aa2.imag], qs_b2=[bb2.real, bb2.imag],
              cr_a1=[ca1.real, ca1.imag], cr_b1=[cb1.real, cb1.imag],
              cr_a2=[ca2.real, ca2.imag], cr_b2=[cb2.real, cb2.imag],
              bias_pts=bias[s1].tolist(),
              qs_meas1=np.abs(zq[s1]).tolist(), qs_meas2=np.abs(zq[s2]).tolist(),
              cr_meas1=np.abs(zc[s1]).tolist(), cr_meas2=np.abs(zc[s2]).tolist())
    rows.append(dict(coef=cx, x_um=float(xx), x_clamp=float(xx - C.CLAMP),
                     invols=inv_by_x[float(xx)],
                     f_cr1_Hz=f_cr1, fwhm_Hz=fwhm, Q=f_cr1 / fwhm if fwhm else None,
                     P_qs=dq['P_abs'], P_cr1=dc['P_abs'],
                     E=dc['P_abs'] / dq['P_abs'],
                     vcpd_qs=dq['v_cpd'], vcpd_cr1=dc['v_cpd'],
                     ratio_qs=dq['ratio'], ratio_cr1=dc['ratio'],
                     flip_qs=dq['flip_deg']))

print(f'\n{"x_clamp":>8} {"InvOLS":>11} {"CR1 kHz":>9} {"Q":>6} {"|P|qs":>10} {"|P|CR1":>10} '
      f'{"E":>8} {"|b/P|qs":>8} {"|b/P|CR1":>9}')
for r in rows:
    print(f'{r["x_clamp"]:8.1f} {r["invols"]:11.4e} {r["f_cr1_Hz"]/1e3:9.2f} {r["Q"]:6.0f} '
          f'{r["P_qs"]:10.3e} {r["P_cr1"]:10.3e} {r["E"]:8.1f} {r["ratio_qs"]:8.3f} '
          f'{r["ratio_cr1"]:9.3f}')

# --- position calibration: 1/InvOLS ~ (x-x0)^2 (3L-(x-x0)) -------------------
xs = np.array([r['x_um'] for r in rows])
iv = np.array([r['invols'] for r in rows])
y = 1.0 / iv
# constrained: the laser cannot sit beyond the free end, so L >= max(x) - x0
best = None
for x0 in np.arange(-40, 40, 0.1):
    s = xs - x0
    if s.min() <= 0:
        continue
    for L in np.arange(max(190.0, s.max()), 300, 0.5):
        m = s ** 2 * (3 * L - s)
        g = np.sum(y * m) / np.sum(m * m)
        rms = float(np.sqrt(np.mean((y / (g * m) - 1) ** 2)))
        if best is None or rms < best[0]:
            best = (rms, float(x0), float(L), float(g))
rms, x0, L, g = best
print(f'\nposition calibration: x0 = {x0:+.1f} um, L = {L:.1f} um, {100*rms:.1f} % rms')
print(f'  free-end position from clamp: {max(xs) - x0:.1f} um   (SCM-PIT nominal 225 um)')
print(f'  R1 for reference: x0 = +6.1 um, L = 225.9 um, 3 % rms')

# d33 from the quasi-static channel, per position
VAC = 1.0
for r in rows:
    r['d33_qs_pm_per_V'] = r['P_qs'] * r['invols'] / 32.0 / VAC * 1e12
d = np.array([r['d33_qs_pm_per_V'] for r in rows])
print(f'\nd33 (quasi-static, |P|*InvOLS/32/Vac): {d.min():.2f}-{d.max():.2f} pm/V, '
      f'median {np.median(d):.2f}, spread {d.max()/d.min():.2f}x')

C.jdump(dict(rows=rows, x0_um=x0, L_um=L, calib_rms=rms,
             note='R2 bias survey, 500 nN, Vac 1 V; InvOLS log-linearly interpolated from the 17-point pre-flight curve, cross-checked against the per-position force curves (1-3 %)'),
        's3_transfer.json')

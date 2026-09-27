"""R2 load extension: stage 40 (load ladder, 5 loads x 8 positions, 0 V),
stage 41 (bias x load at the working position), stage 42 (post-load walk).

Reduction conventions are taken unchanged from s3_transfer.py so that the load
series is directly comparable with the 500 nN bias survey:
  - quasi-static observable = COMPLEX MEAN of Z over 15-45 kHz
  - CR1 observable          = Z at the peak bin of the 0 V / spot-1 reference
  - two-domain decomposition Z_s(V) = a_s + b_s V  (common.decompose)
The 0 V ladder has no bias series, so it yields the transfer function (f, Q, E)
but NOT d33; every d33-vs-load number comes from stage 41.
"""
import glob, json, os
import numpy as np
from igor2.binarywave import load as ibw_load
import common as C

R2 = C.R2
LAD, BVL, PC = (os.path.join(R2, s) for s in ('40_load_ladder', '41_bias_vs_load', '42_postcheck'))
QS = (15e3, 45e3)
CR1W = (250e3, 340e3)          # observed 287-296 kHz across the whole ladder
SNR_MIN = 8.0                  # peak / band median; below this there is no resonance
DIV, VAC = 32.0, 1.0

pf = json.load(open(os.path.join(R2, '01_preflight', 'preflight_result.json')))
pf_x = np.array(sorted(float(k) for k in pf['invols_by_x']))
pf_v = np.array([float(pf['invols_by_x'][f'{k:.1f}']) for k in pf_x])
invols_at = lambda xq: float(np.exp(np.interp(xq, pf_x, np.log(pf_v))))


def cr1_peak(F, z):
    """Peak, half-power width and Q in the CR1 window -- or None if there is no
    resonance there (noise floors are monotone, so they peak at a band edge)."""
    m = (F >= CR1W[0]) & (F <= CR1W[1])
    f, a = F[m], np.abs(z[m])
    k = int(np.argmax(a))
    snr = float(a[k] / max(np.median(a), 1e-18))
    if snr < SNR_MIN or k <= 1 or k >= a.size - 2:
        return None
    h = a[k] / np.sqrt(2.0)
    lo = k
    while lo > 0 and a[lo] > h:
        lo -= 1
    hi = k
    while hi < a.size - 1 and a[hi] > h:
        hi += 1
    w = float(f[hi] - f[lo])
    return dict(f_Hz=float(f[k]), amp=float(a[k]), snr=snr, fwhm_Hz=w,
                Q=float(f[k] / w) if w > 0 else None, k=k)


out = {}

# ---------------- stage 40: the ladder -------------------------------------
lad, drops = {}, []
for p in sorted(glob.glob(os.path.join(LAD, 'load_ladder_r2_*nN_checkpoint.npz'))):
    load = float(os.path.basename(p).split('_')[3].replace('nN', ''))
    F, Z, x, cond = C.load_ckpt(p)
    qm = (F >= QS[0]) & (F <= QS[1])
    rows = []
    for i, xx in enumerate(np.asarray(x, float)):
        z = Z[0, i]
        pk = cr1_peak(F, z)
        aq = float(np.abs(z[qm].mean()))
        r = dict(x_um=float(xx), x_clamp=float(xx) - C.CLAMP, invols=invols_at(float(xx)),
                 A_qs_0V=aq, cr1=pk)
        if pk:
            r.update(f_cr1_Hz=pk['f_Hz'], fwhm_Hz=pk['fwhm_Hz'], Q=pk['Q'],
                     A_cr1_0V=pk['amp'], E_0V=pk['amp'] / aq if aq else None)
        else:
            drops.append((load, float(xx)))
        rows.append(r)
    lad[f'{load:.0f}'] = rows
    ok = [r for r in rows if r.get('Q')]
    f = np.array([r['f_cr1_Hz'] for r in ok]); q = np.array([r['Q'] for r in ok])
    e = np.array([r['E_0V'] for r in ok])
    print(f'load {load:5.0f} nN: {len(ok)}/8 positions with CR1   f {f.mean()/1e3:7.2f} kHz '
          f'(sd {f.std():5.1f} Hz)   Q {np.median(q):5.0f}   E {e.min():6.1f}-{e.max():6.1f}')
out['ladder'] = lad
out['ladder_no_resonance'] = [dict(load_nN=l, x_um=xx) for l, xx in drops]
out['probe_checks'] = json.load(open(os.path.join(LAD, 'load_ladder_r2_probe_checks.json')))
if drops:
    print('  no detectable CR1 at: ' + ', '.join(f'{l:.0f} nN / x={xx:.1f} um' for l, xx in drops))

# the InvOLS the 500 nN step flagged
print('\nforce curves in 40_load_ladder, last six (the 500 nN step logged a jump):')
for fp in sorted(glob.glob(os.path.join(LAD, 'FSCMPIT*.ibw')))[-6:]:
    note = ibw_load(fp)['wave']['note'].decode('latin-1', 'ignore').replace('\r', '\n')
    v = [l.split(':', 1)[1].strip() for l in note.split('\n')
         if l.split(':', 1)[0].strip() == 'InvOLS']
    print(f'   {os.path.basename(fp)}  InvOLS {v[0] if v else "n/a"}')

# ---------------- stage 41: bias x load ------------------------------------
bvl = {}
for p in sorted(glob.glob(os.path.join(BVL, 'bias_vs_load_r2_*nN_checkpoint.npz'))):
    load = float(os.path.basename(p).split('_')[4].replace('nN', ''))
    F, Z, x, cond = C.load_ckpt(p)
    bias = np.array([c['bias_V'] for c in cond], float)
    spot = np.array([c['spot'] for c in cond], int)
    s1, s2 = spot == 1, spot == 2
    xx = float(np.asarray(x, float).ravel()[0]); inv = invols_at(xx)
    qm = (F >= QS[0]) & (F <= QS[1])
    i0 = int(np.where((bias == 0) & s1)[0][0])
    zq = Z[:, 0, qm].mean(axis=-1)                       # complex band mean, as in s3
    pk = cr1_peak(F, Z[i0, 0])
    cm = (F >= CR1W[0]) & (F <= CR1W[1])
    zc = Z[:, 0, cm][:, pk['k']]
    o1, o2 = np.argsort(bias[s1]), np.argsort(bias[s2])
    assert np.allclose(bias[s1][o1], bias[s2][o2])
    dq = C.decompose(bias[s1][o1], zq[s1][o1], zq[s2][o2])
    dc = C.decompose(bias[s1][o1], zc[s1][o1], zc[s2][o2])
    rec = dict(x_um=xx, x_clamp=xx - C.CLAMP, invols=inv, n_cond=len(cond),
               f_cr1_Hz=pk['f_Hz'], Q=pk['Q'], qs=dq, cr1=dc,
               E=dc['P_abs'] / dq['P_abs'],
               d33_qs_pm_per_V=dq['P_abs'] * inv / DIV / VAC * 1e12,
               bias_pts=bias[s1][o1].tolist(),
               qs_meas1=np.abs(zq[s1][o1]).tolist(), qs_meas2=np.abs(zq[s2][o2]).tolist(),
               cr_meas1=np.abs(zc[s1][o1]).tolist(), cr_meas2=np.abs(zc[s2][o2]).tolist())
    print(f'load {load:5.0f} nN: V_cpd {dq["v_cpd"]:+.3f} V (CR1 {dc["v_cpd"]:+.3f})   '
          f'|P| {dq["P_abs"]:.3e}   |b|/|P| {dq["ratio"]:.3f}   flip {dq["flip_deg"]:.1f} deg   '
          f'E {rec["E"]:.0f}   d33 {rec["d33_qs_pm_per_V"]:.2f} pm/V')
    bvl[f'{load:.0f}'] = rec
out['bias_vs_load'] = bvl

# ---------------- stage 42: the post-load walk -----------------------------
F, Z, x, cond = C.load_ckpt(os.path.join(PC, 'postcheck_walk_8pos_checkpoint.npz'))
qm = (F >= QS[0]) & (F <= QS[1])
walk = []
for i, xx in enumerate(np.asarray(x, float)):
    pk = cr1_peak(F, Z[0, i])
    aq = float(np.abs(Z[0, i][qm].mean()))
    walk.append(dict(x_um=float(xx), x_clamp=float(xx) - C.CLAMP, A_qs_0V=aq,
                     f_cr1_Hz=pk['f_Hz'], Q=pk['Q'], A_cr1_0V=pk['amp'],
                     E_0V=pk['amp'] / aq if aq else None))
out['postcheck'] = walk
fw = np.array([w['f_cr1_Hz'] for w in walk])
print(f'\npostcheck walk: CR1 {fw.mean()/1e3:.2f} kHz, spread {np.ptp(fw):.0f} Hz')

C.jdump(out, 's4_load.json')

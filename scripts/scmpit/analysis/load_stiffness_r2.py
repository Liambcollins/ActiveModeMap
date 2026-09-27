r"""What the load ladder's CR1 frequency can and cannot say about contact stiffness.

The blind EB fit of §3.4 fixes the lever geometry.  Holding it, sweep the only
remaining parameter k*/k_lever and ask where the measured CR1 sits on that curve.
The answer is the point of the section: above ~200 nN the measured frequency is at
or beyond the model's stiff-contact (pinned) asymptote, so CR1 carries no stiffness
information there at all -- while Q, and therefore the enhancement, keeps moving.
"""
import json

import numpy as np

from forward_model import EBForwardModel, ProbeGeometry

S4 = json.load(open('/mnt/user-data/uploads/ActiveModeMap/DomainsB_SCMPIT_R2/analysis/s4_load.json'))
FIT = np.load('/home/claude/scmpit_analysis/r2_eb_fit.npz')
p = FIT['p']; L = float(FIT['L'])
_, lk, sb, lz, th = p
f0, KLEV = 63801.0, 1.697


def cr1_of(la):
    g = ProbeGeometry(name='SCM-PIT', f0_hz=f0, k_lever=KLEV, L_um=L, tip_setback_um=sb,
                      tip_height_um=th, tilt_deg=11.0)
    m = EBForwardModel(geom=g, n_modes=14, nx=453, nf=2)
    m.omega = np.linspace(230e3, 360e3, 1600) / m.freq_scale_hz
    m.f_hz = m.omega * m.freq_scale_hz
    r = m.response(np.array([la, lk, 2.3, 0.0, 0.0, lz]))
    i = np.argmin(np.abs(m.xi - 0.55))
    return float(m.f_hz[int(np.argmax(np.abs(r['piezo'][:, i])))])


las = np.arange(1.6, 5.01, 0.1)
curve = np.array([cr1_of(v) for v in las])
asym = float(curve.max())
print(f'model CR1 vs contact stiffness: {curve.min()/1e3:.1f} kHz at k*/k = {10**las[0]:.0f} '
      f'-> pinned asymptote {asym/1e3:.2f} kHz')

loads = sorted(float(k) for k in S4['ladder'])
rows = []
for ld in loads:
    rs = [r for r in S4['ladder'][f'{ld:.0f}'] if r.get('Q')]
    f = float(np.mean([r['f_cr1_Hz'] for r in rs]))
    fsd = float(np.std([r['f_cr1_Hz'] for r in rs]))
    Q = float(np.median([r['Q'] for r in rs]))
    E = [r['E_0V'] for r in rs]
    if f < asym:
        la = float(np.interp(f, curve, las))
        # local sensitivity dln f / dln k*
        j = int(np.argmin(np.abs(las - la)))
        j = min(max(j, 1), las.size - 2)
        sens = float((np.log(curve[j + 1]) - np.log(curve[j - 1]))
                     / (np.log(10 ** las[j + 1]) - np.log(10 ** las[j - 1])))
        det = True
    else:
        la, sens, det = float('nan'), 0.0, False
    rows.append(dict(load_nN=ld, f_cr1_Hz=f, f_sd_Hz=fsd, Q=Q, n_pos=len(rs),
                     E_min=float(min(E)), E_max=float(max(E)), E_med=float(np.median(E)),
                     k_ratio=(10 ** la if det else None),
                     k_star_N_per_m=(KLEV * 10 ** la if det else None),
                     dlnf_dlnk=sens, determined=det))
    tag = (f'k*/k_lever {10**la:8.0f}   k* {KLEV*10**la:8.1f} N/m   dlnf/dlnk {sens:.4f}'
           if det else 'AT/ABOVE the model pinned limit -- k* not determined')
    print(f'{ld:5.0f} nN  ({len(rs)}/8 pos)  CR1 {f/1e3:7.2f} ± {fsd:5.0f} Hz   Q {Q:5.0f}   '
          f'E {np.median(E):6.0f}   {tag}')

f = np.array([r['f_cr1_Hz'] for r in rows]); Q = np.array([r['Q'] for r in rows])
Em = np.array([r['E_med'] for r in rows])
print(f'\nacross 50 -> 500 nN (10x load):  CR1 {100*(f[-1]/f[0]-1):+.2f} %   '
       f'Q x{Q[-1]/Q[0]:.2f}   median E x{Em[-1]/Em[0]:.2f}')
print(f'  measured 500 nN CR1 is {100*(f[-1]/asym-1):+.2f} % relative to the pinned asymptote')

bvl = S4['bias_vs_load']
print('\nstage 41, working position x = 154.9 um:')
for k in sorted(bvl, key=float):
    r = bvl[k]
    print(f'{float(k):5.0f} nN   V_cpd {r["qs"]["v_cpd"]:+.3f} V   d33 {r["d33_qs_pm_per_V"]:.2f} pm/V'
          f'   |b|/|P| {r["qs"]["ratio"]:.3f}   flip {r["qs"]["flip_deg"]:.1f} deg   E {r["E"]:.0f}')
v = np.array([bvl[k]['qs']['v_cpd'] for k in sorted(bvl, key=float)])
d = np.array([bvl[k]['d33_qs_pm_per_V'] for k in sorted(bvl, key=float)])
print(f'  V_cpd over 100-500 nN: {v.mean():+.3f} ± {v.std():.3f} V (range {np.ptp(v)*1e3:.0f} mV)')
print(f'  d33   over 100-500 nN: {d.min():.2f}-{d.max():.2f} pm/V (x{d.max()/d.min():.2f})')

pcw = S4['postcheck']
fp = np.array([w['f_cr1_Hz'] for w in pcw])
print(f'\npost-load walk: CR1 {fp.mean()/1e3:.2f} kHz, spread {np.ptp(fp):.0f} Hz')

json.dump(dict(rows=rows, model_asymptote_Hz=asym,
               model_curve=dict(log_alpha=las.tolist(), f_Hz=curve.tolist()),
               note='k* inverted from CR1 with the blind-fit geometry held fixed; '
                    'above 200 nN the measurement exceeds the model pinned limit'),
          open('/home/claude/scmpit_analysis/r2_load_stiffness.json', 'w'), indent=1)
print('wrote r2_load_stiffness.json')

"""R1 vs R2: AC drive series, and the close-out reference condition + mode-shape walk."""
import glob, json, os, re
import numpy as np
import common as C

res = {}

# ------------------------------------------------------------------ AC series
print('=== AC DRIVE SERIES ===')
ac = {}
INVOLS = {'R1': 1.171e-6, 'R2': 1.2807e-6}     # position A, in force during each series
for tag, P in (('R1', C.P1), ('R2', C.P2)):
    files = sorted(glob.glob(os.path.join(P['acdir'], 'ac_series_A_vac*mV_checkpoint.npz')))
    rows = []
    for fp in files:
        vac = float(re.search(r'vac(\d+\.\d+)mV', os.path.basename(fp)).group(1)) / 1e3
        F, Z, x, cond = C.load_ckpt(fp)
        for i, c in enumerate(cond):
            z = Z[i, 0, :]
            pk = C.peak_in(F, z, C.BANDS['cr1'])
            rows.append(dict(vac_V=vac, bias_V=c['bias_V'], spot=c['spot'],
                             f_Hz=pk['f_Hz'], amp_V=pk['amp'], phase_deg=pk['phase_deg'],
                             snr=pk['snr']))
    ac[tag] = dict(rows=rows, invols=INVOLS[tag], n_levels=len(files))
    print(f'  {tag}: {len(files)} drive levels, {len(rows)} conditions')
    # d33 per domain at 0 V, using the raw amplitude / (Vac * E) -- E filled in later;
    # here report the raw amplitude-per-volt which is E*d33, plus the SNR trend
    print(f'   {"Vac mV":>8} {"spot1 amp/Vac":>14} {"spot2 amp/Vac":>14} {"s1 snr":>7} {"s2 snr":>7} '
          f'{"ph1":>7} {"ph2":>7}')
    for vac in sorted({r['vac_V'] for r in rows}):
        r1 = [r for r in rows if r['vac_V'] == vac and r['spot'] == 1 and r['bias_V'] == 0]
        r2 = [r for r in rows if r['vac_V'] == vac and r['spot'] == 2 and r['bias_V'] == 0]
        if not (r1 and r2):
            continue
        a1 = r1[0]['amp_V'] * INVOLS[tag] / 32 / vac * 1e12
        a2 = r2[0]['amp_V'] * INVOLS[tag] / 32 / vac * 1e12
        print(f'   {vac*1e3:8.1f} {a1:14.1f} {a2:14.1f} {r1[0]["snr"]:7.1f} {r2[0]["snr"]:7.1f} '
              f'{r1[0]["phase_deg"]:7.1f} {r2[0]["phase_deg"]:7.1f}')
    # noise-floor test: at low Vac the measured AMPLITUDE (not amp/Vac) should approach a constant
    lows = sorted({r['vac_V'] for r in rows if r['vac_V'] <= 0.03})
    amps = [np.mean([r['amp_V'] for r in rows if r['vac_V'] == v and r['bias_V'] == 0])
            for v in lows]
    print(f'   low-Vac raw amplitude (V): ' +
          ', '.join(f'{v*1e3:.0f}mV:{a*1e6:.2f}uV' for v, a in zip(lows, amps)))
res['ac'] = ac

# ------------------------------------------------------------------ close-out
print('\n=== CLOSE-OUT: reference condition at position A ===')
co = {}
for tag, P in (('R1', C.P1), ('R2', C.P2)):
    F, Z, x, cond = C.load_ckpt(P['co_ref'])
    bias = np.array([c['bias_V'] for c in cond], float)
    spot = np.array([c['spot'] for c in cond], int)
    s1, s2 = spot == 1, spot == 2
    qm = (F >= C.QS_BAND[0]) & (F <= C.QS_BAND[1])
    zq = Z[:, 0, qm].mean(axis=-1)
    dq = C.decompose(bias[s1], zq[s1], zq[s2])
    i0 = int(np.where((bias == 0) & s1)[0][0])
    pk = C.peak_in(F, Z[i0, 0, :], C.BANDS['cr1'])
    cm = (F >= C.BANDS['cr1'][0]) & (F <= C.BANDS['cr1'][1])
    k = int(np.argmin(np.abs(F[cm] - pk['f_Hz'])))
    zc = Z[:, 0, cm][:, k]
    dc = C.decompose(bias[s1], zc[s1], zc[s2])
    j = json.load(open(P['co_json']))
    iv = (j.get('part3_detection') or {}).get('invols_final_m_per_V')
    fa = (j.get('freeair') or {})
    a1 = abs(zc[s1][bias[s1] == 0][0]); a2 = abs(zc[s2][bias[s2] == 0][0])
    co[tag] = dict(cr1_Hz=pk['f_Hz'], qs=dq, cr1=dc, invols_final=iv,
                   freeair_Hz=fa.get('peak_50_80kHz_Hz'), freeair_snr=fa.get('peak_snr_dB'),
                   amp1=float(a1), amp2=float(a2), ratio=float(a1 / a2),
                   ph1=float(np.degrees(np.angle(zc[s1][bias[s1] == 0][0]))),
                   ph2=float(np.degrees(np.angle(zc[s2][bias[s2] == 0][0]))))
    c = co[tag]
    print(f'  {tag}: CR1 {pk["f_Hz"]/1e3:7.2f} kHz   qs Vcpd {dq["v_cpd"]:+6.3f}   '
          f'cr1 Vcpd {dc["v_cpd"]:+6.3f}   |b|/|P| {dc["ratio"]:.3f}   flip {dq["flip_deg"]:.1f}')
    print(f'      domain amps at 0V: {a1*1e3:.4f} / {a2*1e3:.4f} mV  ratio {a1/a2:.3f}   '
          f'phases {c["ph1"]:+.1f} / {c["ph2"]:+.1f} deg')
    print(f'      InvOLS final {iv}   free-air peak {c["freeair_Hz"]} Hz @ {c["freeair_snr"]} dB')

print('\n=== CLOSE-OUT: mode-shape walk ===')
wk = {}
for tag, P in (('R1', C.P1), ('R2', C.P2)):
    F, Z, x, cond = C.load_ckpt(P['co_walk'])
    Z = Z[0] if Z.ndim == 3 else Z
    o = np.argsort(x); x = x[o]; Z = Z[o]
    f = []; a = []
    for j in range(x.size):
        pk = C.peak_in(F, Z[j], C.BANDS['cr1'])
        f.append(pk['f_Hz']); a.append(pk['amp'])
    wk[tag] = dict(x_um=x.tolist(), f=f, a=a)
    f = np.array(f)
    print(f'  {tag}: CR1 {f.mean()/1e3:7.2f} kHz, spread {(f.max()-f.min())/1e3:.2f} kHz over '
          f'{x.min():.0f}-{x.max():.0f} um')
res['closeout'] = co; res['walk'] = wk

C.jdump(res, 's2_ac_closeout.json')

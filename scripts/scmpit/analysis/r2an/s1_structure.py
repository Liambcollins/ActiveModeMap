"""R1 vs R2: pre-flight resonances, bias-survey channels, dense-map mode shape and nodes."""
import json, os
import numpy as np
import common as C

res = {}

# ---------------------------------------------------------------- pre-flight
print('=== PRE-FLIGHT ===')
pf = {}
for tag, P in (('R1', C.P1), ('R2', C.P2)):
    p = json.load(open(P['preflight']))
    pf[tag] = dict(cr1=p['cr1_Hz'], cr2=p['cr2_Hz'], cr3=p['cr3_Hz'],
                   fwhm=p['cr1_fwhm_Hz'], x_lo=p['x_lo_um'], x_hi=p['x_hi_um'],
                   n=p['n_good_positions'], stop=p['stop_reason'],
                   invols_by_x={float(k): (float(v) if v not in (None,'None') else None)
                                for k, v in (p.get('invols_by_x') or {}).items()},
                   cr1_by_x={float(k): (float(v) if v not in (None,'None') else None)
                             for k, v in (p.get('cr1_by_x') or {}).items()})
    print(f'  {tag}: CR1 {p["cr1_Hz"]/1e3:8.2f}  CR2 {p["cr2_Hz"]/1e3:8.2f}  CR3 {p["cr3_Hz"]/1e3:9.2f} kHz   '
          f'Q1~{p["cr1_Hz"]/p["cr1_fwhm_Hz"]:.0f}   span {p["x_lo_um"]:.0f}-{p["x_hi_um"]:.0f}')
for m in ('cr1', 'cr2', 'cr3'):
    a, b = pf['R1'][m], pf['R2'][m]
    print(f'  {m.upper()} shift: {(b-a)/1e3:+8.2f} kHz  ({100*(b/a-1):+6.2f} %)')
print(f'  CR2/CR1: R1 {pf["R1"]["cr2"]/pf["R1"]["cr1"]:.3f}   R2 {pf["R2"]["cr2"]/pf["R2"]["cr1"]:.3f}')
print(f'  CR3/CR1: R1 {pf["R1"]["cr3"]/pf["R1"]["cr1"]:.3f}   R2 {pf["R2"]["cr3"]/pf["R2"]["cr1"]:.3f}')
# InvOLS at shared positions
sh = sorted(set(pf['R1']['invols_by_x']) & set(pf['R2']['invols_by_x']))
print(f'  InvOLS at {len(sh)} shared x:')
for x in sh:
    v1, v2 = pf['R1']['invols_by_x'][x], pf['R2']['invols_by_x'][x]
    if v1 is None or v2 is None:
        print(f'     x={x:6.1f}  missing'); continue
    print(f'     x={x:6.1f}  R1 {v1:.4e}  R2 {v2:.4e}  ratio {v2/v1:5.3f}')
res['preflight'] = pf

# ------------------------------------------------------------- bias survey
print('\n=== BIAS SURVEY: channels vs position ===')
bs = {}
for tag, P in (('R1', C.P1), ('R2', C.P2)):
    F, Z, x, cond = C.load_ckpt(P['bias'])
    bias = np.array([c['bias_V'] for c in cond], float)
    spot = np.array([c['spot'] for c in cond], int)
    s1, s2 = spot == 1, spot == 2
    i0 = int(np.where((bias == 0) & s1)[0][0])
    qm = (F >= C.QS_BAND[0]) & (F <= C.QS_BAND[1])
    rows = []
    for j, xx in enumerate(x):
        zq = Z[:, j, qm].mean(axis=-1)
        dq = C.decompose(bias[s1], zq[s1], zq[s2])
        pk = C.peak_in(F, Z[i0, j, :], C.BANDS['cr1'])
        cm = (F >= C.BANDS['cr1'][0]) & (F <= C.BANDS['cr1'][1])
        k = int(np.argmin(np.abs(F[cm] - pk['f_Hz'])))
        zc = Z[:, j, cm][:, k]
        dc = C.decompose(bias[s1], zc[s1], zc[s2])
        rows.append(dict(x_um=float(xx), x_clamp=float(xx - C.CLAMP),
                         cr1_Hz=pk['f_Hz'], cr1_amp=pk['amp'],
                         qs_vcpd=dq['v_cpd'], qs_ratio=dq['ratio'], qs_P=dq['P_abs'],
                         qs_flip=dq['flip_deg'], cr1_vcpd=dc['v_cpd'], cr1_ratio=dc['ratio'],
                         cr1_P=dc['P_abs'], cr1_flip=dc['flip_deg']))
    bs[tag] = dict(rows=rows, x=x.tolist())
    print(f'  {tag}: {len(rows)} positions, {Z.shape[0]} conditions')
    print(f'   {"x(clamp)":>9} {"CR1 kHz":>9} {"qs Vcpd":>8} {"qs |b|/|P|":>10} {"cr1 Vcpd":>9} {"cr1 |b|/|P|":>11} {"flip":>6}')
    for r in rows:
        print(f'   {r["x_clamp"]:9.1f} {r["cr1_Hz"]/1e3:9.2f} {r["qs_vcpd"]:8.3f} {r["qs_ratio"]:10.3f} '
              f'{r["cr1_vcpd"]:9.3f} {r["cr1_ratio"]:11.3f} {r["qs_flip"]:6.1f}')
    v = np.array([r['qs_vcpd'] for r in rows])
    print(f'   qs Vcpd over positions: mean {v.mean():+.3f} +- {v.std():.3f} V')
res['bias'] = bs

# ---------------------------------------------------------------- dense map
print('\n=== DENSE MAP: mode shape and nodes ===')
dm = {}
for tag, P in (('R1', C.P1), ('R2', C.P2)):
    F, Z, x, cond = C.load_ckpt(P['dense'])
    Z = Z[0] if Z.ndim == 3 else Z
    o = np.argsort(x); x = x[o]; Z = Z[o]
    ent = dict(x_um=x.tolist(), x_clamp=(x - C.CLAMP).tolist())
    for nm, band in C.BANDS.items():
        m = (F >= band[0]) & (F <= band[1])
        sub = np.abs(Z[:, m])
        k = np.argmax(sub, axis=1)
        ent[f'{nm}_f'] = F[m][k].tolist()
        ent[f'{nm}_a'] = sub[np.arange(sub.shape[0]), k].tolist()
    dm[tag] = ent
    print(f'  {tag}: {x.size} positions {x.min():.0f}-{x.max():.0f} stage-um')
    for nm in ('cr1', 'cr2', 'cr3'):
        f = np.array(ent[f'{nm}_f']); a = np.array(ent[f'{nm}_a'])
        xc = np.array(ent['x_clamp'])
        # interior minima of amplitude = nodes
        nodes = [float(xc[i]) for i in range(2, a.size - 2)
                 if a[i] == a[max(0, i-3):i+4].min() and a[i] < 0.5 * np.median(a)]
        # merge neighbours
        merged = []
        for nd in nodes:
            if not merged or nd - merged[-1][-1] > 8:
                merged.append([nd])
            else:
                merged[-1].append(nd)
        nodes = [float(np.mean(g)) for g in merged]
        print(f'    {nm.upper()}: f {f.mean()/1e3:8.2f} +- {f.std()/1e3:.2f} kHz   '
              f'amax at x_clamp {xc[int(np.argmax(a))]:6.1f} um   nodes {[round(n,1) for n in nodes]}')
        ent[f'{nm}_nodes'] = nodes
res['dense'] = dm

C.jdump(res, 's1_structure.json')

import numpy as np, json, pickle

VAC_MV = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000]
BASE = '/mnt/user-data/uploads/ActiveModeMap/DomainsB_SCMPIT/ac_series_A_vac{:07.2f}mV_checkpoint.npz'
CR1_WIN = (230e3, 340e3)
INVOLS = 1.171e-6            # m/V, position A, stable across this session (1.1710e-6 stage5, 1.1692e-6 stage6)
AMP_INVOLS = INVOLS / 32.0   # m/V
E1, E2 = 198.2, 225.8        # in-situ enhancement, domain 1 / domain 2 (from stage 5)

def peak(freq, Z):
    m = (freq >= CR1_WIN[0]) & (freq <= CR1_WIN[1])
    fr = freq[m]; a = np.abs(Z[m])
    k = int(np.argmax(a))
    peak_ok = a[k] > 3 * np.median(a)
    return fr[k], a[k], np.degrees(np.angle(Z[m][k])), peak_ok

rows = []
for mv in VAC_MV:
    vac = mv / 1000.0
    d = np.load(BASE.format(mv), allow_pickle=True)
    freq = d['freq_Hz']; Z = d['Z']
    conds = json.loads(str(d['conditions']))
    rec = dict(vac_mV=mv, vac_V=vac)
    for i, c in enumerate(conds):
        f, a_raw, ph, ok = peak(freq, Z[i, 0])
        amp_m = a_raw * AMP_INVOLS
        E = E1 if c['spot'] == 1 else E2
        d33 = amp_m / (vac * E) * 1e12   # pm/V
        tag = f"s{c['spot']}_{c['bias_V']:.2f}V"
        rec[f'{tag}_f'] = f; rec[f'{tag}_amp_pm'] = amp_m * 1e12; rec[f'{tag}_d33'] = d33
        rec[f'{tag}_ph'] = ph; rec[f'{tag}_ok'] = ok
    rows.append(rec)

print(f"{'Vac(mV)':>8s}  {'s1@0V d33':>10s} {'s2@0V d33':>10s} {'s1@1.33 d33':>12s} {'s2@1.33 d33':>12s}  {'ratio@0V':>9s} {'ratio@1.33':>10s}")
for r in rows:
    d1_0, d2_0 = r['s1_0.00V_d33'], r['s2_0.00V_d33']
    d1_v, d2_v = r['s1_1.33V_d33'], r['s2_1.33V_d33']
    print(f"{r['vac_mV']:8.1f}  {d1_0:10.2f} {d2_0:10.2f} {d1_v:12.2f} {d2_v:12.2f}  {d1_0/d2_0:9.3f} {d1_v/d2_v:10.3f}")

with open('stage7_rows.pkl', 'wb') as f:
    pickle.dump(rows, f)

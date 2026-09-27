import numpy as np, json, glob, re

VAC_MV = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000]
BASE = '/mnt/user-data/uploads/ActiveModeMap/DomainsB_SCMPIT/ac_series_A_vac{:07.2f}mV_checkpoint.npz'
CR1_WIN = (230e3, 340e3)

def peak(freq, Z):
    m = (freq >= CR1_WIN[0]) & (freq <= CR1_WIN[1])
    fr = freq[m]; a = np.abs(Z[m])
    k = int(np.argmax(a))
    # sanity: is this actually a resonance (peaked), or just noise/DC leakage?
    # true CR1 peaks are narrow; check the peak stands above the window median
    peak_ok = a[k] > 3 * np.median(a)
    return fr[k], a[k], np.degrees(np.angle(Z[m][k])), peak_ok

rows = []
for mv in VAC_MV:
    d = np.load(BASE.format(mv), allow_pickle=True)
    freq = d['freq_Hz']; Z = d['Z']  # (4,1,nfreq)
    conds = json.loads(str(d['conditions']))
    rec = dict(vac_mV=mv)
    for i, c in enumerate(conds):
        f, a, ph, ok = peak(freq, Z[i, 0])
        tag = f"s{c['spot']}_{c['bias_V']:.2f}V"
        rec[f'{tag}_f'] = f; rec[f'{tag}_amp_pm'] = a * 1e12; rec[f'{tag}_ph'] = ph; rec[f'{tag}_ok'] = ok
    rows.append(rec)

for r in rows:
    print(f"Vac={r['vac_mV']:7.1f} mV  "
          f"s1@0V: {r['s1_0.00V_amp_pm']:8.2f} pm ({'ok' if r['s1_0.00V_ok'] else 'FAIL'})  "
          f"s2@0V: {r['s2_0.00V_amp_pm']:8.2f} pm ({'ok' if r['s2_0.00V_ok'] else 'FAIL'})  "
          f"s1@1.33V: {r['s1_1.33V_amp_pm']:8.2f} ({'ok' if r['s1_1.33V_ok'] else 'FAIL'})  "
          f"s2@1.33V: {r['s2_1.33V_amp_pm']:8.2f} ({'ok' if r['s2_1.33V_ok'] else 'FAIL'})")

import pickle
with open('stage7_rows.pkl', 'wb') as f:
    pickle.dump(rows, f)

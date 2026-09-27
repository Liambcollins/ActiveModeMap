import numpy as np
from igor2.binarywave import load as ibw_load

UP = '/mnt/user-data/uploads/ActiveModeMap/DomainsB_SCMPIT/'
FILES = {
    'A_qs20k_0V_30mV':     'LoAC0575.ibw',
    'A_cr1_0V_30mV':       'LoAC0576.ibw',
    'A_cr1_Vcpdtrue_30mV': 'LoAC0577.ibw',
}
Msk = np.load('stage5_mask.npz'); M1, M2 = Msk['M1'], Msk['M2']
Msk2 = np.load('stage5_noise.npz'); M1e, M2e = Msk2['M1e'], Msk2['M2e']

data = {}
for name, fn in FILES.items():
    w = ibw_load(UP + fn)['wave']
    amp = w['wData'][:, :, 1].astype(float)
    phase = w['wData'][:, :, 3].astype(float)
    data[name] = dict(amp=amp, phase=phase)

def circ_mean_deg(deg):
    r = np.radians(deg)
    return np.degrees(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r))))

def circ_sep_deg(a, b):
    diff = (a - b) % 360
    return diff if diff <= 180 else 360 - diff

VAC = 0.03
print(f"{'frame':22s} {'amp1(pm)':>9s} {'amp2(pm)':>9s} {'ratio':>7s} {'ph1':>8s} {'ph2':>8s} {'sep':>7s}")
res = {}
for name, d in data.items():
    amp = d['amp'] * 1e12; ph = d['phase']
    a1 = np.median(amp[M1]); a2 = np.median(amp[M2])
    p1 = circ_mean_deg(ph[M1]); p2 = circ_mean_deg(ph[M2])
    sep = circ_sep_deg(p1, p2)
    res[name] = dict(a1=a1, a2=a2, ratio=a1/a2, p1=p1, p2=p2, sep=sep)
    print(f"{name:22s} {a1:9.3f} {a2:9.3f} {a1/a2:7.3f} {p1:8.1f} {p2:8.1f} {sep:7.1f}")

# in-situ enhancement at 30 mV
e1 = res['A_cr1_0V_30mV']['a1'] / res['A_qs20k_0V_30mV']['a1']
e2 = res['A_cr1_0V_30mV']['a2'] / res['A_qs20k_0V_30mV']['a2']
print(f"\nin-situ E at 30 mV: domain1={e1:.1f}  domain2={e2:.1f}  (1V in-situ was 198.2 / 225.8)")

# d33 using per-domain in-situ E (at 30 mV, matching this drive)
for name in ('A_cr1_0V_30mV', 'A_cr1_Vcpdtrue_30mV'):
    r = res[name]
    d1 = r['a1'] / (VAC * e1); d2 = r['a2'] / (VAC * e2)
    print(f"{name}: d33 domain1={d1:.2f}  domain2={d2:.2f}  pm/V")
d1q = res['A_qs20k_0V_30mV']['a1'] / VAC; d2q = res['A_qs20k_0V_30mV']['a2'] / VAC
print(f"A_qs20k_0V_30mV: d33 domain1={d1q:.2f}  domain2={d2q:.2f}  pm/V")

# noise (eroded masks), referred to Vac
print(f"\n{'frame':22s} {'noise1(pm)':>10s} {'noise2(pm)':>10s} {'contrast(pm)':>12s} {'SNR':>7s}  {'noise_eq(pm/V)':>15s}")
for name, d in data.items():
    amp = d['amp'] * 1e12
    is_qs = 'qs' in name
    E = 1.0 if is_qs else (e1+e2)/2
    n1 = np.std(amp[M1e]); n2 = np.std(amp[M2e])
    c = abs(np.median(amp[M1e]) - np.median(amp[M2e]))
    noise_pool = np.sqrt((n1**2+n2**2)/2)
    snr = c/noise_pool
    noise_eq = noise_pool/(VAC*E)
    print(f"{name:22s} {n1:10.3f} {n2:10.3f} {c:12.2f} {snr:7.2f}  {noise_eq:15.4f}")

np.savez('stage6_raw.npz', **{f'{k}_amp': v['amp'] for k, v in data.items()},
         **{f'{k}_phase': v['phase'] for k, v in data.items()})

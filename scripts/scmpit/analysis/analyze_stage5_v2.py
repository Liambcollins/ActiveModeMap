import numpy as np
from sklearn.cluster import KMeans

d = np.load('stage5_raw.npz')

# --- domain mask from the highest-SNR frame (position A, CR1, 0V): same sample XY
#     raster is used at every position/bias in this stage, so one mask applies to all.
ph_ref = d['A_cr1_0V_phase']
amp_ref = d['A_cr1_0V_amp']
X = np.stack([np.cos(np.radians(ph_ref)).ravel(), np.sin(np.radians(ph_ref)).ravel()], axis=1)
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
lab = km.labels_.reshape(ph_ref.shape)
# Label 0 = "domain 1", 1 = "domain 2" -- pin which is which by the higher median amplitude side
if np.median(amp_ref[lab == 0]) < np.median(amp_ref[lab == 1]):
    pass
M1, M2 = (lab == 0), (lab == 1)
print(f'domain masks: {M1.sum()} / {M2.sum()} px  (of {lab.size})')
np.savez('stage5_mask.npz', M1=M1, M2=M2)

def circ_mean_deg(deg):
    r = np.radians(deg)
    return np.degrees(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r))))

def circ_sep_deg(a, b):
    """Smallest separation between two mean phases, 0-180 deg."""
    diff = (a - b) % 360
    return diff if diff <= 180 else 360 - diff

frames = ['A_qs20k_0V', 'A_cr1_0V', 'A_cr1_Vcpd', 'A_cr1_0V_repeat',
          'B_qs20k_0V', 'B_cr1_0V', 'B_cr1_Vcpd', 'B_qs20k_0V_2Vac']
E_BY_POS = {'A': 200.2, 'B': 2.8}
VAC_BY_FRAME = {'B_qs20k_0V_2Vac': 2.0}

print(f"\n{'frame':20s} {'amp1(pm)':>9s} {'amp2(pm)':>9s} {'ratio':>7s} {'ph1':>8s} {'ph2':>8s} {'sep':>7s}  {'d33_1':>7s} {'d33_2':>7s}")
results = {}
for name in frames:
    amp = d[f'{name}_amp']; ph = d[f'{name}_phase']
    pos = name[0]; is_qs = 'qs' in name
    vac = VAC_BY_FRAME.get(name, 1.0)
    E = 1.0 if is_qs else E_BY_POS[pos]
    a1 = np.median(amp[M1]); a2 = np.median(amp[M2])
    p1 = circ_mean_deg(ph[M1]); p2 = circ_mean_deg(ph[M2])
    sep = circ_sep_deg(p1, p2)
    ratio = a1 / a2
    d1 = a1 / (vac * E) * 1e12
    d2 = a2 / (vac * E) * 1e12
    results[name] = dict(a1=a1, a2=a2, ratio=ratio, p1=p1, p2=p2, sep=sep, d1=d1, d2=d2, vac=vac, E=E)
    print(f"{name:20s} {a1*1e12:9.2f} {a2*1e12:9.2f} {ratio:7.3f} {p1:8.1f} {p2:8.1f} {sep:7.1f}  {d1:7.2f} {d2:7.2f}")

np.save('stage5_results.npy', results, allow_pickle=True)

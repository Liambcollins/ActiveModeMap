import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

d = np.load('stage5_raw.npz')
n = np.load('stage5_noise.npz'); M1e, M2e = n['M1e'], n['M2e']

INK, SEC, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
C1, C2 = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9.5, 'axes.edgecolor': '#c3c2b7',
                     'axes.labelcolor': SEC, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.titlecolor': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'legend.frameon': False})

E_insitu = {'A': (198.2, 225.8), 'B': (25.3, 18.1)}
pairs = [('A', 'A_qs20k_0V', 'A_cr1_0V', 200.2), ('B', 'B_qs20k_0V', 'B_cr1_0V', 2.8)]

rows = []
for pos, qsname, crname, E_stage1 in pairs:
    e1, e2 = E_insitu[pos]; E_is = (e1 + e2) / 2
    for label, name, E in [('quasi-static', qsname, 1.0), ('CR1 (stage-1 E)', crname, E_stage1), ('CR1 (in-situ E)', crname, E_is)]:
        amp = d[f'{name}_amp'] * 1e12
        n1 = np.std(amp[M1e]); n2 = np.std(amp[M2e])
        c = abs(np.median(amp[M1e]) - np.median(amp[M2e]))
        noise_pool = np.sqrt((n1**2 + n2**2) / 2)
        rows.append(dict(pos=pos, label=label, snr=c / noise_pool, noise_eq=noise_pool / E))

fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6))

ax = axs[0]
labels = ['quasi-static', 'CR1\n(stage-1 E)', 'CR1\n(in-situ E)']
xA = [r['snr'] for r in rows if r['pos'] == 'A']
xB = [r['snr'] for r in rows if r['pos'] == 'B']
xi = np.arange(3)
ax.bar(xi - 0.18, xA, width=0.34, color=C1, label='position A')
ax.bar(xi + 0.18, xB, width=0.34, color=C2, label='position B')
ax.set_xticks(xi); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel('SNR  (domain contrast / pixel noise)')
ax.set_title('A.  Raw image SNR: CR1 is not better than quasi-static here', loc='left')
ax.legend(fontsize=8.5)

ax = axs[1]
xA = [r['noise_eq'] for r in rows if r['pos'] == 'A']
xB = [r['noise_eq'] for r in rows if r['pos'] == 'B']
ax.bar(xi - 0.18, xA, width=0.34, color=C1, label='position A')
ax.bar(xi + 0.18, xB, width=0.34, color=C2, label='position B')
ax.set_xticks(xi); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_yscale('log')
ax.set_ylabel('input-referred noise (pm/V), log')
ax.set_title('B.  Referred to the driving voltage: CR1 noise ≈ or > quasi-static', loc='left')
ax.legend(fontsize=8.5)

fig.suptitle('Noise: CR1 (resonance) vs quasi-static, same pixels, same 0 V frame, eroded domain interiors only\n'
             '(pixel-to-pixel spread within one nominally-uniform domain, not shot noise)',
             fontsize=10.5, color=INK, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.savefig('figJ_noise_comparison.png', dpi=150)
print('saved figJ')
for r in rows:
    print(r)

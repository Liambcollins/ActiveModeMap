import numpy as np, matplotlib, pickle
matplotlib.use('Agg'); import matplotlib.pyplot as plt

rows = pickle.load(open('stage7_rows.pkl', 'rb'))
vac = np.array([r['vac_mV'] for r in rows]) / 1000.0
s1_0 = np.array([r['s1_0.00V_d33'] for r in rows])
s2_0 = np.array([r['s2_0.00V_d33'] for r in rows])
s1_v = np.array([r['s1_1.33V_d33'] for r in rows])
s2_v = np.array([r['s2_1.33V_d33'] for r in rows])
ph1_0 = np.array([r['s1_0.00V_ph'] for r in rows])
ph2_0 = np.array([r['s2_0.00V_ph'] for r in rows])

def sep(p1, p2):
    d = (p1 - p2) % 360
    return np.where(d <= 180, d, 360 - d)

INK, SEC, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
C1, C2 = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9.5, 'axes.edgecolor': '#c3c2b7',
                     'axes.labelcolor': SEC, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.titlecolor': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'legend.frameon': False, 'lines.linewidth': 1.8})

fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.8))

ax = axs[0]
ax.plot(vac, s1_0, 'o-', color=C1, ms=6, mec=SURF, mew=0.8, label='domain "spot 1", 0 V')
ax.plot(vac, s2_0, 's-', color=C2, ms=6, mec=SURF, mew=0.8, label='domain "spot 2", 0 V')
ax.plot(vac, s1_v, 'o--', color=C1, ms=6, mec=SURF, mew=0.8, alpha=0.55, label='spot 1, +1.33 V')
ax.plot(vac, s2_v, 's--', color=C2, ms=6, mec=SURF, mew=0.8, alpha=0.55, label='spot 2, +1.33 V')
ax.axvspan(0.003, 0.04, color=MUTED, alpha=0.12, lw=0)
ax.text(0.008, 36, 'noise-floor bias\n(amplitude pulled up\nat low SNR)', fontsize=7.5, color=SEC)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('drive amplitude Vac (V)'); ax.set_ylabel('d33 (pm/V)')
ax.set_title('A.  Clean, flat linearity 50 mV → 2 V (40× range)', loc='left')
ax.legend(fontsize=7.5, loc='upper right')

ax = axs[1]
s = sep(ph1_0, ph2_0)
sv = sep(np.array([r['s1_1.33V_ph'] for r in rows]), np.array([r['s2_1.33V_ph'] for r in rows]))
ax.plot(vac, s, 'o-', color=C1, ms=6, mec=SURF, mew=0.8, label='0 V')
ax.plot(vac, sv, 's-', color=C2, ms=6, mec=SURF, mew=0.8, label='+1.33 V')
ax.axhline(180, color=MUTED, ls='--', lw=1)
ax.set_xscale('log'); ax.set_ylim(60, 200)
ax.set_xlabel('drive amplitude Vac (V)'); ax.set_ylabel('domain phase separation (deg)')
ax.set_title('B.  Phase separation holds near 180° once above the noise floor', loc='left')
ax.legend(fontsize=8)

ax = axs[2]
V = [0.77, 1.33, 1.22]
labels = ['this morning\n(stage 1)', 'stage 5\n(post-ladder, 1 V)', 'stage 7\n(evening, 1-2 V)']
ax.plot(range(3), V, 'o-', color=C1, ms=9, mec=SURF, mew=1)
for i, (v, l) in enumerate(zip(V, labels)):
    ax.annotate(f'{v:.2f} V', (i, v), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=9, color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('extrapolated Vcpd (V)')
ax.set_title('C.  Vcpd keeps drifting through the session -- but decelerating', loc='left')
ax.set_ylim(0.5, 1.6)

fig.suptitle('Stage 7: 15-point AC series, position A, both domains, 0 V and +1.33 V — confirms clean linearity above\n'
             '~50 mV, but domain amplitudes have shifted enough since stage 5 that the null has moved to ~1.2 V',
             fontsize=11, color=INK, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.savefig('figM_stage7_ac_series.png', dpi=150)
print('saved figM')

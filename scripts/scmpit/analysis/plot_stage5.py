import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

rows = np.load('stage5_extrap.npy', allow_pickle=True)[()]
INK, SEC, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
C1, C2 = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9.5, 'axes.edgecolor': '#c3c2b7',
                     'axes.labelcolor': SEC, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.titlecolor': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'legend.frameon': False, 'lines.linewidth': 1.8})

fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6))

ax = axs[0]
for (pos, col, lab) in [('A', C1, 'position A (148.8 µm, E≈200)'), ('B', C2, 'position B (225.9 µm, free end, E≈2.8)')]:
    r = rows[pos]; V0, V1 = 0.0, 0.77
    Vline = np.linspace(-0.3, 1.7, 50)
    a1 = r['a1'][0] + r['m1'] * Vline
    a2 = r['a2'][0] + r['m2'] * Vline
    ax.plot(Vline, a1 / a2 if False else a1, '--', color=col, lw=1.1, alpha=0.55)
    ax.plot(Vline, a2, '--', color=col, lw=1.1, alpha=0.55)
    ax.plot([V0, V1], r['a1'], 'o', color=col, ms=7, mec=SURF, mew=1)
    ax.plot([V0, V1], r['a2'], 's', color=col, ms=7, mec=SURF, mew=1)
    ax.axvline(r['Vstar'], color=col, lw=1, ls=':')
ax.set_yscale('log')
ax.set_xlabel('DC bias (V)'); ax.set_ylabel('CR1 domain amplitude (pm)')
ax.set_title("A.  Both domains' amplitude vs bias, extrapolated to their crossing", loc='left')
ax.text(0.02, 0.03, 'both positions independently extrapolate to\nVcpd (true) ≈ +1.33 V (this morning: +0.77 V)',
        transform=ax.transAxes, fontsize=8.5, color=SEC, va='bottom')
h = [plt.Line2D([], [], color=MUTED, marker='o', ls='', label='domain 1'),
     plt.Line2D([], [], color=MUTED, marker='s', ls='', label='domain 2'),
     plt.Line2D([], [], color=C1, lw=2, label='position A'),
     plt.Line2D([], [], color=C2, lw=2, label='position B')]
ax.legend(handles=h, fontsize=7.5, loc='upper right')

ax = axs[1]
labels = ['0 V\n(as measured)', '+0.77 V\n(this morning\'s Vcpd)', '+1.33 V\n(extrapolated null)']
for pos, col, lab in [('A', C1, 'position A'), ('B', C2, 'position B')]:
    r = rows[pos]
    ratios = [r['a1'][0]/r['a2'][0], r['a1'][1]/r['a2'][1], 1.0]
    xj = np.array([0,1,2]) + (0.08 if pos=='B' else -0.08)
    ax.plot(xj, ratios, 'o-', color=col, ms=7, mec=SURF, mew=1, label=lab)
ax.axhline(1.0, color=MUTED, ls='--', lw=1)
ax.set_xticks([0,1,2]); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel('domain amplitude ratio')
ax.set_title('B.  Amplitude equalizes only once the drifted Vcpd is corrected', loc='left')
ax.legend(fontsize=8.5, loc='upper right')

fig.suptitle('Stage 5 quantitative CR1 imaging: phase separation is 178–180° everywhere (confirmed), but amplitude\n'
             'nulling needs a same-session Vcpd — this morning\'s +0.77 V under-corrects; both positions agree on +1.33 V',
             fontsize=10.5, color=INK, x=0.01, ha='left')
fig.tight_layout(rect=(0,0,1,0.86))
fig.savefig('figH_stage5_vcpd_correction.png', dpi=150)
print('saved')

import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

d6 = np.load('stage6_raw.npz')
Msk = np.load('stage5_mask.npz'); M1, M2 = Msk['M1'], Msk['M2']

INK, SEC, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
C1, C2 = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9.5, 'axes.edgecolor': '#c3c2b7',
                     'axes.labelcolor': SEC, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.titlecolor': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'legend.frameon': False, 'lines.linewidth': 1.8})
ext = [0, 6, 0, 6]

fig = plt.figure(figsize=(13.5, 8.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])

# top row: images at 30 mV -- qs (noise-limited), CR1 0V, CR1 +1.33V
titles_imgs = [('A_qs20k_0V_30mV', 'quasi-static, 0 V\n(SNR 0.2 -- below noise floor)'),
               ('A_cr1_0V_30mV', 'CR1, 0 V\n(SNR 2.8)'),
               ('A_cr1_Vcpdtrue_30mV', 'CR1, +1.33 V (extrapolated null)\n(domains equalized)')]
for c, (name, title) in enumerate(titles_imgs):
    ax = fig.add_subplot(gs[0, c])
    amp = d6[f'{name}_amp'] * 1e12
    vmax = np.percentile(amp, 99)
    im = ax.imshow(amp, extent=ext, cmap='magma', vmin=0, vmax=vmax)
    ax.set_title(title, loc='left', fontsize=8.8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label('pm', fontsize=7); cb.ax.tick_params(labelsize=7)
    ax.set_xlabel('µm')
    if c == 0: ax.set_ylabel('µm')

# bottom-left: ratio validation across drive amplitude / bias
ax = fig.add_subplot(gs[1, 0])
# 1 V data (from stage 5) + 30 mV data (stage 6), position A only
V_1V = [0.0, 0.77]; ratio_1V = [1.656, 1.238]
V_30mV = [0.0, 1.33]; ratio_30mV = [1.623, 0.980]
ax.plot(V_1V, ratio_1V, 'o-', color=C1, ms=8, mec=SURF, mew=1, label='1 V drive (stage 5)')
ax.plot(V_30mV, ratio_30mV, 's-', color=C2, ms=8, mec=SURF, mew=1, label='30 mV drive (stage 6)')
ax.axhline(1.0, color=MUTED, ls='--', lw=1)
ax.axvline(1.33, color=SEC, ls=':', lw=1)
ax.text(1.33, 1.58, 'predicted null\n(from 1 V data)', fontsize=7.5, color=SEC, ha='center')
ax.set_xlabel('DC bias (V)'); ax.set_ylabel('domain amplitude ratio')
ax.set_title('A.  30 mV independently confirms the +1.33 V null', loc='left', fontsize=10)
ax.legend(fontsize=8)

# bottom-middle: d33 cross-check across drive amplitude
ax = fig.add_subplot(gs[1, 1])
cats = ['0 V', '1.33 V\n(null)']
d1_1V = [11.97, 10.36]; d2_1V = [6.35, 7.34]     # from figK / stage5 (1V, in-situ E)
d1_30 = [10.29, 8.18]; d2_30 = [5.56, 7.33]       # 30 mV, using TRUSTED 1V E
x = np.arange(2)
ax.plot(x - 0.05, d1_1V, 'o-', color=C1, ms=8, mec=SURF, mew=1, label='domain 1, 1 V')
ax.plot(x - 0.05, d2_1V, 'o--', color=C1, ms=8, mec=SURF, mew=1, alpha=0.55, label='domain 2, 1 V')
ax.plot(x + 0.05, d1_30, 's-', color=C2, ms=8, mec=SURF, mew=1, label='domain 1, 30 mV')
ax.plot(x + 0.05, d2_30, 's--', color=C2, ms=8, mec=SURF, mew=1, alpha=0.55, label='domain 2, 30 mV')
ax.axhspan(7.5, 8.5, color=GRID, alpha=0.6, lw=0)
ax.text(0.02, 7.9, 'spectroscopy 7.9-8.0 pm/V', fontsize=7.5, color=SEC, transform=ax.get_yaxis_transform())
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_ylabel('d33 (pm/V)')
ax.set_title('B.  d33 at 30 mV tracks 1 V (both using the 1V-calibrated E)', loc='left', fontsize=10)
ax.legend(fontsize=7.5, ncol=2, loc='upper right')

# bottom-right: noise vs drive amplitude
ax = fig.add_subplot(gs[1, 2])
vac_pts = [0.03, 1.0]
qs_noise = [None, 0.536]   # 30 mV qs is below noise floor -- not a real number, shown as open marker at estimate
cr1_noise = [1.324, 0.517]
ax.plot(vac_pts, cr1_noise, 'o-', color=C1, ms=8, mec=SURF, mew=1, label='CR1, referred to Vac')
ax.plot([1.0], [0.536], 's', color=C2, ms=8, mec=SURF, mew=1, label='quasi-static, referred to Vac')
ax.plot([0.03], [8.14], 's', color=C2, ms=8, mec=SURF, mew=1, markerfacecolor=SURF, label='quasi-static (below noise floor -- not real signal)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('drive amplitude (V)'); ax.set_ylabel('input-referred noise (pm/V)')
ax.set_title('C. Below ~100 mV, qs disappears into\nnoise; CR1 does not', loc='left', fontsize=9.5)
ax.legend(fontsize=7.2, loc='upper left')

fig.suptitle('Stage 6 (30 mV drive) confirms stage 5\'s extrapolated Vcpd = +1.33 V independently, and shows why\n'
             'contact resonance actually earns its keep: not lower fractional noise at strong drive, but a usable\n'
             'signal where quasi-static drops below the detector\'s noise floor',
             fontsize=11, color=INK, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.savefig('figL_stage6_lowac.png', dpi=150)
print('saved figL')

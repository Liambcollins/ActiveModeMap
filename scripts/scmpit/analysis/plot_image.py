import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
d = np.load('image_qs_corrected.npz'); Z = d['Z']; E = d['E'][()]; MA = d['MA']; MB = d['MB']; VAC = float(d['vac'])
Zc = Z - E
INK, SEC, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
C1, C2 = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9, 'axes.labelcolor': SEC, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.titlecolor': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'legend.frameon': False, 'axes.edgecolor': '#c3c2b7'})
ext = [0, 6, 0, 6]
fig, axs = plt.subplots(1, 4, figsize=(15, 3.9))
for ax in axs: ax.set_xlabel('µm')
im0 = axs[0].imshow(np.abs(Z) * 1e12 / VAC, extent=ext, cmap='magma', vmin=0, vmax=14)
axs[0].set_title('raw  |Z| / V_ac', loc='left'); axs[0].set_ylabel('µm')
im1 = axs[1].imshow(np.abs(Zc) * 1e12 / VAC, extent=ext, cmap='magma', vmin=0, vmax=14)
axs[1].set_title('corrected  |Z − E| / V_ac', loc='left')
for im, ax in ((im0, axs[0]), (im1, axs[1])):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label('d₃₃ (pm/V)', fontsize=8); cb.ax.tick_params(labelsize=8)
diff = (np.abs(Z) - np.abs(Zc)) * 1e12 / VAC
im2 = axs[2].imshow(diff, extent=ext, cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0, vmin=-0.45, vmax=0.45))
axs[2].set_title('electrostatic term removed', loc='left')
cb = fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.03); cb.set_label('Δd₃₃ (pm/V)', fontsize=8); cb.ax.tick_params(labelsize=8)
ax = axs[3]
bins = np.linspace(0, 14, 120)
ax.hist(np.abs(Z[MA]) * 1e12 / VAC, bins=bins, color=C1, alpha=0.45, label='domain A, raw')
ax.hist(np.abs(Z[MB]) * 1e12 / VAC, bins=bins, color=C2, alpha=0.45, label='domain B, raw')
ax.hist(np.abs(Zc[MA]) * 1e12 / VAC, bins=bins, histtype='step', color=C1, lw=1.6, label='domain A, corrected')
ax.hist(np.abs(Zc[MB]) * 1e12 / VAC, bins=bins, histtype='step', color=C2, lw=1.6, label='domain B, corrected')
ax.axvline(7.90, color=MUTED, ls='--', lw=1)
ax.text(7.75, ax.get_ylim()[1] * 0.92, 'spectroscopy 7.90', rotation=90, ha='right', va='top', fontsize=8, color=SEC)
ax.set_xlabel('d₃₃ (pm/V)'); ax.set_ylabel('pixels'); ax.set_title('per-domain distributions', loc='left')
ax.legend(fontsize=7.5, loc='upper left'); ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Quasi-static PFM image (20 kHz, free end, 472 nN, V_ac = 2 V) converted to absolute d₃₃ — '
             'the two-domain correction removes a 0.60 pm common electrostatic vector (3.1 % of |P|), '
             'closing the A:B asymmetry from 1.063 to 1.000',
             fontsize=9.5, color=INK, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig('figF_image_correction.png', dpi=150)
print('ok')

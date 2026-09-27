import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

d = np.load('stage5_raw.npz')
INK, SEC, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9, 'axes.edgecolor': '#c3c2b7',
                     'axes.labelcolor': SEC, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.titlecolor': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'legend.frameon': False})

ext = [0, 6, 0, 6]
fig, axs = plt.subplots(2, 4, figsize=(15.5, 7.6))

rows = [
    ('A', 'position A (148.8 µm, E≈200)', 'A_qs20k_0V', 'A_cr1_0V', 200.2),
    ('B', 'position B (free end, 225.9 µm)', 'B_qs20k_0V', 'B_cr1_0V', 2.8),
]
for r, (pos, lab, qsname, crname, E) in enumerate(rows):
    amp_qs = d[f'{qsname}_amp'] * 1e12
    amp_cr = d[f'{crname}_amp'] * 1e12
    ph_qs = d[f'{qsname}_phase']
    ph_cr = d[f'{crname}_phase']

    ax = axs[r, 0]
    im = ax.imshow(amp_qs, extent=ext, cmap='magma', vmin=0, vmax=np.percentile(amp_qs, 99))
    ax.set_title(f'{lab}\nquasi-static |Z| (pm)', loc='left', fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=7)

    ax = axs[r, 1]
    im = ax.imshow(ph_qs, extent=ext, cmap='twilight', vmin=-180, vmax=180)
    ax.set_title('quasi-static phase (deg)', loc='left', fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=7)

    ax = axs[r, 2]
    im = ax.imshow(amp_cr, extent=ext, cmap='magma', vmin=0, vmax=np.percentile(amp_cr, 99))
    ax.set_title(f'CR1 |Z| (pm)  [E={E:.0f}]', loc='left', fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=7)

    ax = axs[r, 3]
    im = ax.imshow(ph_cr, extent=ext, cmap='twilight', vmin=-180, vmax=180)
    ax.set_title('CR1 phase (deg)', loc='left', fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=7)

for ax in axs[-1, :]:
    ax.set_xlabel('µm')
for ax in axs[:, 0]:
    ax.set_ylabel('µm')

fig.suptitle('Resonance (CR1) vs quasi-static PFM images, both laser positions, 0 V, 500 nN, 1 V drive — '
             'same 6 µm frame, same two ferroelectric domains',
             fontsize=11, color=INK, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig('figI_resonance_images.png', dpi=150)
print('saved figI')

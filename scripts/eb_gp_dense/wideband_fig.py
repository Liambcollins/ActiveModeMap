"""Figure for the wideband recovery test: maps + per-mode x-profiles."""
import sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
import wideband as W

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
DSN = sys.argv[sys.argv.index('--data') + 1] if '--data' in sys.argv else 'ppp'
W.configure(DSN)
TAG = 'wideband' if DSN == 'ppp' else 'wideband_' + DSN
d = np.load(f'{TAG}_maps_N{N}.npz')
x, f, Z, idx = d['x'], d['f'], d['Z'], d['idx']
arms = [('measured', Z), ('EB-wide', d['eb']), ('EB+GP-wide', d['hy']), ('EB-band1 (extrapolated)', d['ebb']), (f'low-rank r={min(8, N-1)}', d['lrh'])]
fb = W.DATASETS[DSN]['floor_band']; floor = float(np.median(np.abs(Z[:, (f > fb[0]) & (f < fb[1])])))
vmin, vmax = 20 * np.log10(floor), 0.0

fig = plt.figure(figsize=(16, 13))
NM = len(W.MODES_KHZ + W.EXTRA_KHZ) if DSN != 'ppp' else 5
gs = fig.add_gridspec(3, max(5, NM), height_ratios=[1.1, 1.1, 1.3], hspace=0.38, wspace=0.28)
# row 0: |Z| maps in dB; row 1: dB error maps
for k, (name, M) in enumerate(arms):
    ax = fig.add_subplot(gs[0, k])
    im = ax.pcolormesh(f / 1e3, x, 20 * np.log10(np.abs(M) + 1e-15), vmin=vmin, vmax=vmax, cmap='magma', shading='auto', rasterized=True)
    ax.set_title(name, fontsize=10); ax.set_xlabel('f (kHz)')
    if k == 0: ax.set_ylabel('x (µm)')
    if k == 0:
        for i in idx: ax.axhline(x[i], color='c', lw=0.6, alpha=0.7)
    if k == 4: fig.colorbar(im, ax=ax, label='|Z| (dB re max)', fraction=0.05)
    ax2 = fig.add_subplot(gs[1, k])
    if k == 0:
        ax2.pcolormesh(f / 1e3, x, np.degrees(np.angle(Z)), cmap='twilight', shading='auto', rasterized=True)
        ax2.set_title('measured phase (deg)', fontsize=10); ax2.set_ylabel('x (µm)')
    else:
        err = 20 * np.log10((np.abs(M) + 1e-15) / (np.abs(Z) + 1e-15))
        err = np.where(np.abs(Z) > 3 * floor, err, np.nan)
        im2 = ax2.pcolormesh(f / 1e3, x, err, vmin=-12, vmax=12, cmap='RdBu_r', shading='auto', rasterized=True)
        ax2.set_title('amplitude error (dB), signal bins', fontsize=10)
        if k == 4: fig.colorbar(im2, ax=ax2, label='dB', fraction=0.05)
    ax2.set_xlabel('f (kHz)')
# row 2: x-profiles at each mode peak
for k, fk in enumerate((W.MODES_KHZ + W.EXTRA_KHZ) if DSN != 'ppp' else W.MODES_KHZ):
    ax = fig.add_subplot(gs[2, k])
    w = np.abs(f - fk * 1e3) <= 4e3 * max(1.0, fk / 400.0)
    jt = int(np.argmax(np.abs(Z[:, w]).mean(0)))
    for name, M in arms:
        prof = np.abs(M[:, w][:, jt])
        ax.semilogy(x, prof / np.abs(Z[:, w][:, jt]).max(), lw=2.2 if name == 'measured' else 1.2,
                    color='k' if name == 'measured' else None, label=name)
    for i in idx: ax.axvline(x[i], color='c', lw=0.5, alpha=0.6)
    ax.axhline(floor / np.abs(Z[:, w][:, jt]).max(), color='grey', ls=':', lw=0.8)
    ax.set_title(f'mode profile at {f[w][jt]/1e3:.1f} kHz', fontsize=10); ax.set_xlabel('x (µm)')
    ax.set_ylim(3e-4, 2)
    if k == 0: ax.set_ylabel('|Z| / max at this f'); ax.legend(fontsize=7, loc='lower left')
fig.suptitle(f'{"PPP-CONTAu" if DSN == "ppp" else "SCM-PIT (Multi75G, Dense_Grid_B)"} wideband recovery from N = {N} of {x.size} positions (cyan = revealed), {f[0]/1e3:.0f}–{f[-1]/1e3:.0f} kHz, phase-corrected data', fontsize=12)
fig.savefig(f'{TAG}_N{N}.png', dpi=110, bbox_inches='tight')
print('saved', f'{TAG}_N{N}.png')

import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
h = np.load('higher_modes.npz'); xt = h['xt']; d_qs = h['d_qs']
C1, C2, C3, C4, C5 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
INK, SEC, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': SEC,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.titlecolor': INK, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'legend.frameon': False, 'lines.linewidth': 1.5})
MS = dict(markersize=6, markeredgewidth=1.2, markeredgecolor=SURF)
fig, axs = plt.subplots(2, 3, figsize=(13, 7.2))
gains = {}
for k, (name, col) in enumerate([('CR1', C2), ('CR2', C3), ('CR4' if False else 'CR3', C4)]):
    E_meas, E_mod, good = h[f'{name}_E_meas'], h[f'{name}_E_mod'], h[f'{name}_good'].astype(bool)
    fm, fmod, Qm, Qmod = float(h[f'{name}_f_meas']), float(h[f'{name}_f_model']), float(h[f'{name}_Q_meas']), float(h[f'{name}_Q_model'])
    g = float(np.median((E_meas / E_mod)[good])); gains[name] = g
    xi, zc, zq = h[f'{name}_xi_um'], h[f'{name}_zc_full'], h['zq_full']
    ax = axs[0, k]
    ax.plot(xi, g * zc / zq, '-', color=col, lw=1.2, alpha=0.8, label=f'EB shape × {g:.2f}')
    ax.plot(xi, zc / zq, '--', color=col, lw=1.0, alpha=0.6, label='EB, as computed')
    ax.plot(xt[good], E_meas[good], 'o', color=col, label='measured  |P|_mode / |P|_qs', **MS)
    ax.plot(xt[~good], E_meas[~good], 'o', color=SURF, markeredgecolor=col, markeredgewidth=1.5, markersize=6, label='near a node (excluded)')
    ax.set_yscale('log'); ax.set_ylim(1, 1500); ax.set_xlim(0, 235)
    ax.set_title(f'{name}  {fm/1e3:.0f} kHz   Q {Qm:.0f} (model {Qmod:.0f})', loc='left')
    ax.set_ylabel('enhancement over quasi-static' if k == 0 else '')
    ax.legend(loc='lower left', fontsize=7.5)
    ax = axs[1, k]
    ax.axhspan(7.5, 8.5, color=GRID, alpha=0.5, lw=0)
    ax.plot(xt, d_qs, '-', color=C1, lw=1.2, alpha=0.8, label='quasi-static (per-position InvOLS)')
    db = h[f'{name}_d_blind']
    ax.plot(xt[good], db[good], 's', color=col, label=f'{name} ÷ EB transfer, as computed', **MS)
    ax.plot(xt[good], (db / g)[good], 'D', color=col, markerfacecolor=SURF, markeredgewidth=1.5, markersize=6,
            label=f'{name} ÷ (EB × {g:.2f})')
    ax.plot(xt[~good], np.clip(db[~good], 0, 90), 's', color=SURF, markeredgecolor=col, markeredgewidth=1.5, markersize=6)
    ax.set_yscale('log'); ax.set_ylim(3, 100); ax.set_xlim(0, 235)
    ax.set_xlabel('distance from clamp (µm)'); ax.set_ylabel('d_eff  (pm/V), log' if k == 0 else '')
    ax.legend(loc='upper left', fontsize=7.5)
    vv = (db / g)[good]
    ax.text(230, 3.4, f'shape-corrected: {np.median(vv):.1f} pm/V, spread {vv.max()/vv.min():.2f}×', ha='right', fontsize=8, color=SEC)
fig.suptitle('Higher modes: the EB model predicts each mode shape (top) but under-predicts the modal gain by a mode-dependent factor (bottom)',
             fontsize=10, color=INK, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig('figE_higher_modes.png', dpi=150)
print(gains)

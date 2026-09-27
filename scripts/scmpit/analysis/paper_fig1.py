r"""Paper Figure 1 — recovery of the cantilever transfer function (Run 2)."""
import json

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

UP = '/mnt/user-data/uploads/ActiveModeMap/DomainsB_SCMPIT_R2'
S1 = json.load(open(f'{UP}/s1_structure.json'))
TR = json.load(open(f'{UP}/analysis/s3_transfer.json'))
WB = json.load(open('/home/claude/scmpit_analysis/r2_wideband.json'))

INK, SEC, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8.5, 'axes.edgecolor': '#c3c2b7',
                     'axes.labelcolor': SEC, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.titlecolor': INK, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'legend.frameon': False,
                     'lines.linewidth': 1.5})

rows = TR['rows']
xc = np.array([r['x_clamp'] for r in rows])
E = np.array([r['E'] for r in rows])
Q = np.array([r['Q'] for r in rows])
Pqs = np.array([r['P_qs'] for r in rows])
Pcr = np.array([r['P_cr1'] for r in rows])
inv = np.array([r['invols'] for r in rows])
d33 = np.array([r['d33_qs_pm_per_V'] for r in rows])
dense = S1['dense']['R2']
xd = np.array(dense['x_clamp'])

fig, axs = plt.subplots(2, 3, figsize=(14.6, 7.9))

# (a) measured mode shapes + nodes ------------------------------------------
ax = axs[0, 0]
for nm, c, ls in (('cr1', BLUE, '-'), ('cr2', ORANGE, '-'), ('cr3', AQUA, '-')):
    a = np.array(dense[f'{nm}_a']); a = a / a.max()
    ax.plot(xd, a, ls, color=c, lw=1.5, label=nm.upper())
    for nd in dense.get(f'{nm}_nodes', []):
        ax.axvline(nd, color=c, lw=0.9, alpha=0.5, ls=':')
        ax.annotate(f'{nd:.0f}', xy=(nd, 1.03), fontsize=6.8, color=c, ha='center')
ax.set_xlabel('distance from clamp (µm)'); ax.set_ylabel('|Z| / max')
ax.set_ylim(0, 1.13)
ax.set_title('a   Mode shapes, 161 positions at 1 µm', loc='left')
ax.legend(fontsize=8, loc='lower center', ncol=3)

# (b) frequency flatness = the reference is drift-free ----------------------
ax = axs[0, 1]
for nm, c in (('cr1', BLUE), ('cr2', ORANGE), ('cr3', AQUA)):
    f = np.array(dense[f'{nm}_f']) / 1e3
    a = np.array(dense[f'{nm}_a']); a = a / a.max()
    ok = a > 0.25          # at a node the peak position is meaningless, not drifting
    ax.plot(xd[ok], f[ok] - f[ok].mean(), '-', color=c, lw=1.2,
            label=f'{nm.upper()}  {f[ok].mean():.1f} kHz, sd {f[ok].std()*1e3:.0f} Hz')
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xlabel('distance from clamp (µm)'); ax.set_ylabel('f − mean (kHz)')
ax.set_ylim(-1.2, 1.2)
ax.set_title('b   Contact state is stationary over the 3.2 h map', loc='left')
ax.legend(fontsize=7.4, loc='upper left')
ax.annotate('points within a node excluded — there the peak\nposition is undefined, not drifting',
            xy=(0.03, 0.05), xycoords='axes fraction', fontsize=7, color=SEC)

# (c) enhancement vs Q -- the core transfer-function point -------------------
ax = axs[0, 2]
ax.semilogy(xc, E, 'o-', color=BLUE, ms=5, mec=SURF, mew=0.8, label='measured $E(x)$')
ax.semilogy(xc, Q, 's--', color=ORANGE, ms=5, mec=SURF, mew=0.8, label='$Q$ (≈233, flat)')
i = int(np.argmin(np.abs(E - Q)))
ax.axvline(xc[i], color=MUTED, lw=0.9, ls=':')
ax.annotate(f'$E=Q$ only near\n{xc[i]:.0f} µm', xy=(xc[i], Q[i]), xytext=(-8, 34),
            textcoords='offset points', fontsize=7.4, color=SEC, ha='right',
            arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
ax.annotate(f'×{E.max()/E.min():.0f} across the beam', xy=(0.04, 0.08),
            xycoords='axes fraction', fontsize=8, color=INK)
ax.set_xlabel('distance from clamp (µm)'); ax.set_ylabel('CR1 enhancement')
ax.set_title('c   Enhancement is not $Q$ — it spans 35–520', loc='left')
ax.legend(fontsize=8, loc='upper right')

# (d) static shape and the clamped-beam fit ---------------------------------
ax = axs[1, 0]
x0, L = TR['x0_um'], TR['L_um']
s = xc + 6.1 - x0                      # back to stage coords, then to beam coord
s = np.array([r['x_um'] for r in rows]) - x0
m = s ** 2 * (3 * L - s)
g = np.sum((1 / inv) * m) / np.sum(m * m)
xx = np.linspace(s.min(), s.max(), 200)
ax.plot(np.array([r['x_um'] for r in rows]), 1 / inv * 1e-6, 'o', color=BLUE, ms=6,
        mec=SURF, mew=0.9, label='1/InvOLS, measured')
ax.plot(xx + x0, g * xx ** 2 * (3 * L - xx) * 1e-6, '-', color=ORANGE, lw=1.4,
        label=f'clamped beam, $x_0$={x0:+.1f} µm, $L$={L:.0f} µm')
ax.annotate(f'{100*TR["calib_rms"]:.1f} % rms', xy=(0.97, 0.06), xycoords='axes fraction',
            fontsize=8, color=SEC, ha='right')
ax.set_xlabel('laser position, stage coordinate (µm)')
ax.set_ylabel('1/InvOLS  (10$^6$ V/m)')
ax.set_title('d   Static shape fixes the position axis', loc='left')
ax.legend(fontsize=7.6, loc='upper left')

# (e) the two channels along the beam ---------------------------------------
ax = axs[1, 1]
ax.semilogy(xc, Pqs, 'o-', color=BLUE, ms=5, mec=SURF, mew=0.8, label='$|P|$ quasi-static')
ax.semilogy(xc, Pcr, 's-', color=ORANGE, ms=5, mec=SURF, mew=0.8, label='$|P|$ at CR1')
ax.set_xlabel('distance from clamp (µm)'); ax.set_ylabel('$|P|$  (V, detector units)')
ax.set_title('e   The same $d_{33}$ seen through two very different gains', loc='left')
ax.legend(fontsize=8, loc='center left')
ax2 = ax.twinx()
ax2.plot(xc, d33, '^:', color=AQUA, ms=5, mec=SURF, mew=0.8)
ax2.set_ylabel('$d_{33}$ from quasi-static (pm/V)', color=AQUA)
ax2.tick_params(axis='y', colors=AQUA); ax2.grid(False); ax2.set_ylim(0, 14)
ax2.annotate(f'median {np.median(d33):.2f} pm/V,\nspread ×{d33.max()/d33.min():.2f}',
             xy=(0.42, 0.10), xycoords='axes fraction', fontsize=7.6, color=AQUA)

# (f) wideband reconstruction ------------------------------------------------
ax = axs[1, 2]
ranks = sorted(int(k) for k in WB['R2']['scores'])
for tag, c, mk in (('R2', BLUE, 'o'), ('R1', MUTED, 's')):
    n = [WB[tag]['scores'][str(r)]['nrmse'] for r in ranks]
    ax.plot(ranks, n, mk + '-', color=c, ms=6, mec=SURF, mew=0.8,
            label=f'{tag}, all modes')
for tag, c in (('R2', ORANGE), ('R1', '#c9c7c0')):
    n = [WB[tag]['scores'][str(r)]['per_mode']['cr3'] for r in ranks]
    ax.plot(ranks, n, '^--', color=c, ms=6, mec=SURF, mew=0.8, label=f'{tag}, CR3 band')
ax.set_xticks(ranks)
ax.set_xlabel('low-rank truncation'); ax.set_ylabel('NRMSE on 7 held-out positions')
ax.set_title('f   8 positions reconstruct the wideband field', loc='left')
ax.legend(fontsize=7.4, loc='upper right', ncol=2)
ax.annotate('18 min of measurement\nstands in for 192 min', xy=(0.04, 0.08),
            xycoords='axes fraction', fontsize=7.6, color=SEC)

fig.suptitle('Figure 1.  Recovery of the cantilever transfer function from the Run 2 dataset',
             fontsize=10.5, color=INK, x=0.008, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig('/home/claude/scmpit_analysis/paper_fig1_transfer.png', dpi=200)
print('saved paper_fig1_transfer.png')
print(f'  E span {E.min():.1f}-{E.max():.1f}, Q {Q.mean():.0f}+-{Q.std():.0f}')
print(f'  d33 qs {d33.min():.2f}-{d33.max():.2f}, median {np.median(d33):.2f}')
print(f'  calib x0 {x0:+.1f} L {L:.1f} rms {100*TR["calib_rms"]:.1f}%')

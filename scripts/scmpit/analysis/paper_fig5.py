r"""Paper Figure 5 — drive linearity, the detection floor, and when resonance helps."""
import json

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

NF = json.load(open('/home/claude/scmpit_analysis/r2_noisefloor.json'))
IM = json.load(open('/home/claude/scmpit_analysis/r2_images.json'))

INK, SEC, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8.5, 'axes.edgecolor': '#c3c2b7',
                     'axes.labelcolor': SEC, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.titlecolor': INK, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'legend.frameon': False,
                     'lines.linewidth': 1.5})

INVOLS, E = 1.2807e-6, {1: 197.4, 2: 169.7}       # position A, R2
fig, axs = plt.subplots(1, 3, figsize=(14.6, 4.5))

# (a) apparent d33 vs drive -------------------------------------------------
ax = axs[0]
# Normalised to each domain's own high-drive plateau. The absolute scale is left
# out deliberately: converting the spectroscopy CR1 amplitude to pm/V needs an
# enhancement, and the enhancement measured in the imaging channel is not the one
# that belongs to the spectroscopy channel (see text). Linearity is a ratio and is
# unaffected by that offset.
for spot, c, mk in ((1, BLUE, 'o'), (2, ORANGE, 's')):
    e = NF[f'R2_s{spot}']
    v = np.array(e['v']); a = np.array(e['a'])
    resp = a / v
    plateau = resp[v >= 0.1].mean()
    ax.plot(v * 1e3, resp / plateau, mk, color=c, ms=5.5, mec=SURF, mew=0.7,
            label=f'domain {spot}')
    vv = np.logspace(np.log10(v.min()), np.log10(v.max()), 300)
    mdl = np.sqrt((e['k'] * vv) ** 2 + e['n0'] ** 2) / vv
    ax.plot(vv * 1e3, mdl / plateau, '-', color=c, lw=1.1, alpha=0.6)
ax.axhline(1.0, color=AQUA, lw=1.2)
ax.axvspan(1.5, 40, color=MUTED, alpha=0.10, lw=0)
ax.annotate('below the detection floor:\napparent response is biased\nhigh, not nonlinear',
            xy=(2.1, 2.4), fontsize=7.4, color=SEC)
ax.annotate('flat to ±4 % over 50 mV – 2 V', xy=(0.97, 0.10), xycoords='axes fraction',
            fontsize=7.6, color=AQUA, ha='right')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('$V_{ac}$ (mV)')
ax.set_ylabel('response / high-drive plateau')
ax.set_title('a   Flat from 50 mV to 2 V; the rise below is the floor', loc='left')
ax.legend(fontsize=8, loc='lower left')

# (b) raw amplitude: the floor is a constant --------------------------------
ax = axs[1]
for spot, c, mk in ((1, BLUE, 'o'), (2, ORANGE, 's')):
    e = NF[f'R2_s{spot}']
    v = np.array(e['v']); a = np.array(e['a'])
    ax.plot(v * 1e3, a * 1e6, mk, color=c, ms=5.5, mec=SURF, mew=0.7, label=f'domain {spot}')
    vv = np.logspace(np.log10(v.min()), np.log10(v.max()), 300)
    ax.plot(vv * 1e3, np.sqrt((e['k'] * vv) ** 2 + e['n0'] ** 2) * 1e6, '-', color=c,
            lw=1.1, alpha=0.6)
    ax.plot(vv * 1e3, e['k'] * vv * 1e6, '--', color=c, lw=0.9, alpha=0.4)
n0 = NF['R2_s1']['n0']
ax.axhline(n0 * 1e6, color=AQUA, ls='-', lw=1.2)
ax.annotate(f'$n_0$ = {n0*1e6:.0f} µV', xy=(2.2, n0 * 1e6 * 1.25), fontsize=8, color=AQUA)
ax.annotate('dashed: pure $kV_{ac}$\nsolid: $\\sqrt{(kV_{ac})^2+n_0^2}$',
            xy=(0.60, 0.13), xycoords='axes fraction', fontsize=7.4, color=SEC)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('$V_{ac}$ (mV)'); ax.set_ylabel('measured $|Z|$ (µV, detector)')
ax.set_title('b   A two-parameter floor fits three decades to 4–6 %', loc='left')
ax.legend(fontsize=8, loc='upper left')

# (c) when does resonance help? ---------------------------------------------
ax = axs[2]
fr, lo = IM['R2']['frames'], IM['R2']['lowac']
groups = ['quasi-static\n1 V', 'CR1\n1 V', 'quasi-static\n30 mV', 'CR1\n30 mV']
noise = [fr['A_qs20k_0V']['noise_ref'], fr['A_cr1_0V']['noise_ref'],
         lo['A_qs20k_0V_30mV']['noise_ref'], lo['A_cr1_0V_30mV']['noise_ref']]
snr = [fr['A_qs20k_0V']['snr'], fr['A_cr1_0V']['snr'],
       lo['A_qs20k_0V_30mV']['snr'], lo['A_cr1_0V_30mV']['snr']]
xi = np.arange(4)
cols = [BLUE, ORANGE, BLUE, ORANGE]
ax.bar(xi, noise, 0.6, color=cols)
for i, (nv, sv) in enumerate(zip(noise, snr)):
    ax.text(i, nv * 1.12, f'{nv:.2f}', ha='center', fontsize=8, color=INK)
    ax.text(i, 0.055, f'SNR\n{sv:.2f}', ha='center', fontsize=7.6, color='white'
            if nv > 0.4 else SEC)
ax.set_yscale('log')
ax.set_xticks(xi); ax.set_xticklabels(groups, fontsize=7.8)
ax.set_ylabel('input-referred noise (pm/V)')
ax.set_ylim(0.04, 30)
ax.axvline(1.5, color=MUTED, lw=0.9, ls=':')
ax.annotate('at strong drive the resonance\nchannel is the noisier one', xy=(0.5, 3.0),
            fontsize=7.4, color=SEC, ha='center')
ax.annotate('at weak drive quasi-static\ncollapses; CR1 barely moves', xy=(2.5, 17),
            fontsize=7.4, color=SEC, ha='center')
ax.set_title('c   Resonance earns its keep only at low drive', loc='left')

fig.suptitle('Figure 5.  Drive-amplitude linearity, the additive detection floor, '
             'and the actual case for contact resonance',
             fontsize=10.5, color=INK, x=0.006, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.915))
fig.savefig('/home/claude/scmpit_analysis/paper_fig5_linearity.png', dpi=200)
print('saved paper_fig5_linearity.png')
for spot in (1, 2):
    e = NF[f'R2_s{spot}']
    v = np.array(e['v']); a = np.array(e['a'])
    r = a / v; hi = v >= 0.05
    pl = r[v >= 0.1].mean()
    print(f'  domain {spot}: plateau flat to +-{100*(r[hi]/pl).std():.1f} % over >=50 mV, '
          f'k {e["k"]*1e3:.1f} mV/V, n0 {e["n0"]*1e6:.0f} uV, resid {100*e["resid"]:.1f} %')
print('  noise/SNR:', [f'{g.strip()}: {n:.2f} pm/V, SNR {s:.2f}'
                       for g, n, s in zip(groups, noise, snr)])

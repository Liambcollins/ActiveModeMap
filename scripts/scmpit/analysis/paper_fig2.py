r"""Paper Figure 2 — exploring load: what moves, what saturates, what does not move at all."""
import json

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

UP = '/mnt/user-data/uploads/ActiveModeMap/DomainsB_SCMPIT_R2'
S4 = json.load(open(f'{UP}/analysis/s4_load.json'))
LS = json.load(open('/home/claude/scmpit_analysis/r2_load_stiffness.json'))
TR = json.load(open(f'{UP}/analysis/s3_transfer.json'))

INK, SEC, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8.5, 'axes.edgecolor': '#c3c2b7',
                     'axes.labelcolor': SEC, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.titlecolor': INK, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'legend.frameon': False,
                     'lines.linewidth': 1.5})

rows = LS['rows']
F = np.array([r['load_nN'] for r in rows])
f1 = np.array([r['f_cr1_Hz'] for r in rows]) / 1e3
fsd = np.array([r['f_sd_Hz'] for r in rows]) / 1e3
Q = np.array([r['Q'] for r in rows])
asym = LS['model_asymptote_Hz'] / 1e3

fig, axs = plt.subplots(2, 2, figsize=(11.4, 8.2))

# (a) frequency barely moves; Q doubles --------------------------------------
ax = axs[0, 0]
ax.errorbar(F, f1, yerr=fsd, fmt='o-', color=BLUE, ms=6, mec=SURF, mew=0.8,
            ecolor=BLUE, elinewidth=1, capsize=3, label='CR1 frequency')
ax.axhline(asym, color=MUTED, ls='--', lw=1.1)
ax.annotate(f'model stiff-contact (pinned) limit, {asym:.1f} kHz',
            xy=(60, asym + 0.08), fontsize=7.4, color=SEC)
ax.annotate(f'10× load moves CR1 by {100*(f1[-1]/f1[0]-1):+.2f} %',
            xy=(0.97, 0.12), xycoords='axes fraction', fontsize=8, color=BLUE, ha='right')
ax.set_xscale('log')
ax.set_xlabel('applied load (nN)'); ax.set_ylabel('CR1 frequency (kHz)', color=BLUE)
ax.tick_params(axis='y', colors=BLUE)
ax2 = ax.twinx()
ax2.plot(F, Q, 's-', color=ORANGE, ms=6, mec=SURF, mew=0.8)
ax2.set_ylabel('CR1 $Q$', color=ORANGE); ax2.tick_params(axis='y', colors=ORANGE)
ax2.grid(False); ax2.set_ylim(90, 270)
ax2.annotate(f'$Q$ ×{Q[-1]/Q[0]:.2f}', xy=(300, Q[-2] + 8), fontsize=8.5, color=ORANGE)
ax.set_title('a   Load changes the damping, not the frequency', loc='left')

# (b) why: the resonance is already pinned -----------------------------------
ax = axs[0, 1]
la = np.array(LS['model_curve']['log_alpha'])
fc = np.array(LS['model_curve']['f_Hz']) / 1e3
ax.semilogx(10 ** la, fc, '-', color=SEC, lw=1.6, label='EB model, blind-fit geometry')
ax.axhline(asym, color=MUTED, ls='--', lw=1.1)
undet = [r['f_cr1_Hz'] / 1e3 for r in rows if not r['determined']]
ax.axhspan(min(undet), max(undet), color=MAGENTA, alpha=0.18, lw=0)
ax.annotate('200, 350 and 500 nN measured\n— all above the asymptote',
            xy=(70, max(undet) + 0.5), fontsize=7.4, color=MAGENTA)
for r in rows:
    if r['determined']:
        ax.plot(r['k_ratio'], r['f_cr1_Hz'] / 1e3, 'o', color=AQUA, ms=8, mec=SURF, mew=1)
        ax.annotate(f'{r["load_nN"]:.0f} nN', xy=(r['k_ratio'], r['f_cr1_Hz'] / 1e3),
                    xytext=(2, -13), textcoords='offset points', fontsize=7.6, color=AQUA)
ax.annotate('above 200 nN the measurement sits at or past\nthe asymptote: CR1 carries no stiffness\n'
            'information there, at any precision',
            xy=(0.03, 0.14), xycoords='axes fraction', fontsize=7.6, color=INK)
ax.set_xlabel('contact stiffness  $k^*/k_{\\mathrm{lever}}$'); ax.set_ylabel('CR1 frequency (kHz)')
ax.set_ylim(268, 299)
ax.set_title('b   The frequency saturates before 200 nN', loc='left')
ax.legend(fontsize=7.6, loc='lower right')

# (c) the enhancement does move ----------------------------------------------
ax = axs[1, 0]
cols = [MAGENTA, ORANGE, YELLOW, AQUA, BLUE]
for (ld, c) in zip(sorted(S4['ladder'], key=float), cols):
    rs = [r for r in S4['ladder'][ld] if r.get('E_0V')]
    xc = [r['x_clamp'] for r in rs]; e = [r['E_0V'] for r in rs]
    ax.semilogy(xc, e, 'o-', color=c, ms=5, mec=SURF, mew=0.7, label=f'{float(ld):.0f} nN')
n50 = S4['ladder_no_resonance']
common = sorted(set.intersection(*[{r['x_um'] for r in S4['ladder'][l] if r.get('E_0V')}
                                   for l in S4['ladder']]))
med = {l: np.median([{r['x_um']: r for r in S4['ladder'][l]}[x]['E_0V'] for x in common])
       for l in S4['ladder']}
lo, hi = sorted(S4['ladder'], key=float)[0], sorted(S4['ladder'], key=float)[-1]
per = [{r['x_um']: r for r in S4['ladder'][hi]}[x]['E_0V']
       / {r['x_um']: r for r in S4['ladder'][lo]}[x]['E_0V'] for x in common]
ax.annotate(f'over the six positions present at every load, median $E$\n'
            f'rises ×{med[hi]/med[lo]:.2f} (×{min(per):.1f}–{max(per):.1f} position by position)',
            xy=(0.97, 0.97), xycoords='axes fraction', fontsize=7.4, color=INK, ha='right',
            va='top')
ax.annotate('at 50 nN the two clamp-proximal positions gave no detectable\n'
            'CR1, and their quasi-static amplitude is also anomalous —\n'
            'those three acquisitions are excluded throughout',
            xy=(0.97, 0.84), xycoords='axes fraction', fontsize=7.2, color=SEC, ha='right',
            va='top')
ax.set_xlabel('distance from clamp (µm)'); ax.set_ylabel('enhancement $E$ at 0 V')
ax.set_title(f'c   Enhancement rises ×{med[hi]/med[lo]:.1f} with load, at every position',
             loc='left')
ax.legend(fontsize=7.6, loc='lower left', ncol=2, title='load', title_fontsize=7.4)

# (d) what load does NOT change: V_cpd and d33 -------------------------------
ax = axs[1, 1]
bvl = S4['bias_vs_load']
ks = sorted(bvl, key=float)
Fb = np.array([float(k) for k in ks])
vc = np.array([bvl[k]['qs']['v_cpd'] for k in ks])
d33 = np.array([bvl[k]['d33_qs_pm_per_V'] for k in ks])
ax.plot(Fb, vc, 'o-', color=BLUE, ms=7, mec=SURF, mew=0.9, label='$V_{cpd}$, quasi-static')
ax.axhline(1.057, color=BLUE, ls=':', lw=1.2)
ax.annotate('bias survey, 2 h earlier: +1.057 V', xy=(520, 1.057), xytext=(0, 6),
            textcoords='offset points', fontsize=7.4, color=BLUE, ha='right')
ax.set_xlabel('applied load (nN)'); ax.set_ylabel('$V_{cpd}$ (V)', color=BLUE)
ax.tick_params(axis='y', colors=BLUE); ax.set_ylim(0.6, 1.25)
d33c = np.array([bvl[k]['d33_qs_drift_corrected'] for k in ks])
ax3 = ax.twinx()
ax3.plot(Fb, d33, '^--', color=MUTED, ms=6, mec=SURF, mew=0.9)
ax3.plot(Fb, d33c, '^-', color=AQUA, ms=7, mec=SURF, mew=0.9)
ax3.axhspan(8.62 * 0.95, 8.62 * 1.05, color=AQUA, alpha=0.13, lw=0)
ax3.annotate('bias survey interpolated to this position, ±5 %', xy=(112, 8.62 * 0.925),
             fontsize=7.3, color=AQUA)
ax3.annotate('grey: pre-flight InvOLS\nteal: InvOLS as the probe\nchecks measured it',
             xy=(408, 6.45), fontsize=7.2, color=SEC)
ax3.set_ylabel('$d_{33}$, quasi-static (pm/V)', color=AQUA)
ax3.tick_params(axis='y', colors=AQUA); ax3.grid(False); ax3.set_ylim(5.5, 10.5)
ax.annotate(f'$V_{{cpd}}$ within {np.ptp(vc)*1e3:.0f} mV over a 5× load range;\n'
            f'$d_{{33}}$ within ×{d33.max()/d33.min():.2f}, and within 4 % of the survey once\n'
            'the measured InvOLS drift is applied',
            xy=(0.04, 0.06), xycoords='axes fraction', fontsize=7.4, color=SEC)
ax.set_title('d   Contact potential and $d_{33}$ are load-insensitive', loc='left')

fig.suptitle('Figure 2.  The load dimension costs an hour — and the answer is that load moves '
             'the damping, not the material number',
             fontsize=10.5, color=INK, x=0.008, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.952))
fig.savefig('/home/claude/scmpit_analysis/paper_fig2_load.png', dpi=200)
print('saved paper_fig2_load.png')
print(f'  CR1 {f1[0]:.2f}->{f1[-1]:.2f} kHz ({100*(f1[-1]/f1[0]-1):+.2f} %), Q {Q[0]:.0f}->{Q[-1]:.0f}')
print(f'  V_cpd {vc.min():+.3f}..{vc.max():+.3f}, d33 {d33.min():.2f}..{d33.max():.2f}, '
      f'drift-corrected {d33c.min():.2f}..{d33c.max():.2f}')
print(f'  common-6 median E ratio {med[hi]/med[lo]:.2f}, per-position {min(per):.2f}-{max(per):.2f}')
print(f'  no-resonance points: {n50}')

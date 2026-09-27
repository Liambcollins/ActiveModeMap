r"""Paper Figure 6 -- the blind Euler-Bernoulli fit and d33 versus position.

The fit sees only the three contact-resonance frequencies, the shape of
1/InvOLS(x) and the CR1 quality factor.  All amplitudes -- and the measured
enhancement E(x) -- are withheld from the residual and used only to score it.
"""
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

A = np.load('/home/claude/scmpit_analysis/r2_eb_fit.npz')          # L at its lower bound
B = np.load('/home/claude/scmpit_analysis/r2_eb_fit_altL.npz')     # L = 225.9 um (lever length)

INK, SEC, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8.5, 'axes.edgecolor': '#c3c2b7',
                     'axes.labelcolor': SEC, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.titlecolor': INK, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'legend.frameon': False,
                     'lines.linewidth': 1.5})

CLAMP = 6.1
xc = B['X'] - CLAMP
iL = int(np.argmax(B['X']))
interior = np.arange(xc.size) != iL

fig, axs = plt.subplots(1, 3, figsize=(14.6, 4.6))

# (a) the blind prediction of the enhancement -------------------------------
ax = axs[0]
ax.semilogy(xc, B['E_meas'], 'o-', color=BLUE, ms=6, mec=SURF, mew=0.8,
            label='measured $E(x)$ (withheld)')
ax.semilogy(xc, B['E_mod'], 's--', color=ORANGE, ms=6, mec=SURF, mew=0.8,
            label='EB prediction, $L$ = 226 µm')
ax.semilogy(xc, A['E_mod'], '^:', color=MUTED, ms=5, mec=SURF, mew=0.8,
            label='EB prediction, $L$ = 208 µm')
ax.semilogy(xc, B['Q_x'], '-', color=MAGENTA, lw=1.2, label='$Q$ (the usual shortcut)')
r = (B['E_meas'] / B['E_mod'])[interior]
ax.annotate(f'interior positions: predicted to\n{r.min():.2f}–{r.max():.2f}× '
            f'(sd {100*r.std()/r.mean():.0f} %)', xy=(0.03, 0.09), xycoords='axes fraction',
            fontsize=7.8, color=INK)
ax.set_xlabel('distance from clamp (µm)'); ax.set_ylabel('CR1 enhancement')
ax.set_title('a   Enhancement predicted without seeing an amplitude', loc='left')
ax.legend(fontsize=7.4, loc='upper right')

# (b) d33 versus position, three routes --------------------------------------
ax = axs[1]
qs = B['d_qs_invols']
cr_eb = B['d_cr1_hybrid']
cr_q = B['d_cr1_Q']
ax.plot(xc, qs, 'o-', color=BLUE, ms=6, mec=SURF, mew=0.8, label='quasi-static (InvOLS)')
ax.plot(xc, cr_eb, 's-', color=ORANGE, ms=6, mec=SURF, mew=0.8,
        label='CR1 ÷ EB-predicted $E$')
ax.plot(xc, cr_q, '^--', color=MAGENTA, ms=6, mec=SURF, mew=0.8, label='CR1 ÷ $Q$')
ax.plot(xc[iL], cr_eb[iL], 's', color=SURF, ms=6, mec=ORANGE, mew=1.2)
ax.axhspan(np.median(qs[interior]) * 0.95, np.median(qs[interior]) * 1.05,
           color=AQUA, alpha=0.16, lw=0)
ax.axhline(np.median(qs[interior]), color=AQUA, lw=1.2)
ax.annotate(f'quasi-static median {np.median(qs[interior]):.2f} pm/V ±5 %',
            xy=(0.97, 0.40), xycoords='axes fraction', fontsize=7.6, color=AQUA, ha='right')
ax.annotate('CR1 ÷ $Q$ tracks the mode shape,\nnot the piezoresponse: ×'
            f'{cr_q[interior].max()/cr_q[interior].min():.1f} across the beam',
            xy=(0.33, 0.20), xycoords='axes fraction', fontsize=7.4, color=MAGENTA)
ax.annotate('open marker: free end, where $E$\nis smallest and the model weakest',
            xy=(xc[iL], cr_eb[iL]), xytext=(-8, 34), textcoords='offset points',
            fontsize=7, color=SEC, ha='right',
            arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
ax.set_yscale('log')
ax.set_yticks([2, 5, 10, 20]); ax.set_yticklabels(['2', '5', '10', '20'])
ax.set_ylim(0.9, 30)
ax.set_xlabel('distance from clamp (µm)'); ax.set_ylabel('$d_{33}$ (pm/V)')
ax.set_title('b   $d_{33}$ along the beam: the two channels agree', loc='left')
ax.legend(fontsize=7.8, loc='lower left')

# (c) how well does each route reproduce the quasi-static answer? ------------
ax = axs[2]
routes = [('quasi-static\nper-position\nInvOLS', qs, BLUE),
          ('quasi-static\nfree-end InvOLS\n+ EB shape', B['d_qs_model'], AQUA),
          ('CR1\nfree-end InvOLS\n+ EB', B['d_cr1_model'], ORANGE),
          ('CR1\nper-position\nInvOLS + EB', cr_eb, YELLOW),
          ('CR1 ÷ $Q$\n(the shortcut)', cr_q, MAGENTA)]
xi = np.arange(len(routes))
med = [np.median(v[interior]) for _, v, _ in routes]
lo = [np.percentile(v[interior], 10) for _, v, _ in routes]
hi = [np.percentile(v[interior], 90) for _, v, _ in routes]
cols = [c for _, _, c in routes]
ax.bar(xi, med, 0.58, color=cols)
ax.errorbar(xi, med, yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)],
            fmt='none', ecolor=INK, elinewidth=1.1, capsize=4)
for i, (_, v, _) in enumerate(routes):
    vv = v[interior]
    ax.text(i, hi[i] + 0.7, f'{med[i]:.2f}', ha='center', fontsize=8.5, color=INK)
    ax.text(i, 0.6, f'±{100*vv.std()/np.median(vv):.0f} %', ha='center', fontsize=7.8,
            color='white')
ax.axhline(med[0], color=BLUE, ls=':', lw=1.1)
ax.set_xticks(xi); ax.set_xticklabels([n for n, _, _ in routes], fontsize=6.3)
ax.set_ylabel('$d_{33}$ over the 7 interior positions (pm/V)')
ax.set_ylim(0, 24)
ax.annotate('bars: median over the 7 interior positions; whiskers 10–90th percentile.\n'
            'The resonance channel reaches the quasi-static answer to 1.5 % using no\n'
            'amplitude calibration beyond the free-end InvOLS.',
            xy=(0.02, 0.86), xycoords='axes fraction', fontsize=7.2, color=SEC, ha='left')
ax.set_title('c   Only the model-based route recovers the scale', loc='left')

fig.suptitle('Figure 6.  A blind Euler–Bernoulli fit turns an on-resonance amplitude into '
             '$d_{33}$; dividing by $Q$ does not',
             fontsize=10.5, color=INK, x=0.006, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.915))
fig.savefig('/home/claude/scmpit_analysis/paper_fig6_eb_d33.png', dpi=200)
print('saved paper_fig6_eb_d33.png')
for n, v, _ in routes:
    vv = v[interior]
    print(f'  {n.replace(chr(10), " "):44s} median {np.median(vv):6.2f}  sd {100*vv.std()/np.median(vv):5.1f} %'
          f'  spread x{vv.max()/vv.min():.2f}')
print(f'  CR1(EB)/quasi-static median ratio: {np.median(cr_eb[interior])/np.median(qs[interior]):.4f}')

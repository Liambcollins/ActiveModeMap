r"""Paper Figure 3 — contact potential: determination, agreement, and equalisation."""
import json

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

UP = '/mnt/user-data/uploads/ActiveModeMap/DomainsB_SCMPIT_R2'
TR = json.load(open(f'{UP}/analysis/s3_transfer.json'))
IM = json.load(open('/home/claude/scmpit_analysis/r2_images.json'))

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
# the working position: closest to 154.9 stage-um
iA = int(np.argmin(np.abs(np.array([r['x_um'] for r in rows]) - 154.9)))
cA = rows[iA]['coef']

fig, axs = plt.subplots(2, 2, figsize=(11.4, 8.0))

# (a) bias dependence at the working position -------------------------------
ax = axs[0, 0]
bp = np.array(cA['bias_pts'], float)
o = np.argsort(bp)
vv = np.linspace(-9.5, 9.5, 200)


def line(a, b):
    A = complex(*a); B = complex(*b)
    return np.abs(A + B * vv)


for key, c, lab in ((('qs_a1', 'qs_b1'), BLUE, 'domain 1'),
                    (('qs_a2', 'qs_b2'), ORANGE, 'domain 2')):
    ax.plot(vv, line(cA[key[0]], cA[key[1]]) * 1e3, '-', color=c, lw=1.5, label=f'{lab}, fit')
ax.plot(bp[o], np.array(cA['qs_meas1'])[o] * 1e3, 'o', color=BLUE, ms=5, mec=SURF, mew=0.8)
ax.plot(bp[o], np.array(cA['qs_meas2'])[o] * 1e3, 's', color=ORANGE, ms=5, mec=SURF, mew=0.8)
vq = rows[iA]['vcpd_qs']
ax.axvline(vq, color=AQUA, lw=1.2, ls='--')
ax.annotate(f'$V_{{cpd}}$ = {vq:+.3f} V\n(curves cross)', xy=(vq, 0.26), xytext=(14, 26),
            textcoords='offset points', fontsize=8, color=AQUA,
            arrowprops=dict(arrowstyle='-', color=AQUA, lw=0.8))
ax.set_xlabel('DC bias (V)'); ax.set_ylabel('quasi-static $|Z|$ (mV)')
ax.set_title(f'a   Bias dependence at {xc[iA]:.0f} µm from clamp, both domains', loc='left')
ax.legend(fontsize=8, loc='upper left')

# (b) V_cpd along the beam, two channels ------------------------------------
ax = axs[0, 1]
vqs = np.array([r['vcpd_qs'] for r in rows])
vcr = np.array([r['vcpd_cr1'] for r in rows])
rq = np.array([r['ratio_qs'] for r in rows])
ax.plot(xc, vqs, 'o-', color=BLUE, ms=5, mec=SURF, mew=0.8, label='quasi-static channel')
ax.plot(xc, vcr, 's-', color=ORANGE, ms=5, mec=SURF, mew=0.8, label='CR1 channel')
inner = rq > 0.08
ax.axhline(vqs[inner].mean(), color=AQUA, ls=':', lw=1.2)
ax.annotate(f'mean over positions where $b$ is\nmeasurable: {vqs[inner].mean():+.3f} '
            f'± {vqs[inner].std():.3f} V', xy=(0.03, 0.06), xycoords='axes fraction',
            fontsize=7.6, color=SEC)
ax.annotate('at the free end the electrostatic term\nvanishes, so $V_{cpd}$ is ill-determined there',
            xy=(xc[-1], vqs[-1]), xytext=(-30, -34), textcoords='offset points',
            fontsize=7.2, color=SEC, ha='right',
            arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
ax.set_xlabel('distance from clamp (µm)'); ax.set_ylabel('$V_{cpd}$ (V)')
ax.set_title('b   Quasi-static channel is consistent along the beam;\n     a single-bin CR1 estimate sits ~0.3 V low', loc='left', fontsize=9)
ax.legend(fontsize=8, loc='upper left')

# (c) the independent determinations ----------------------------------------
ax = axs[1, 0]
names = ['bias survey\n(quasi-static)', 'imaging, pos A\n(3-point)',
         'imaging, free end\n(2-point)', 'close-out\n(4 h later)']
vals = [1.057, 1.024, 1.031, 0.836]
cols = [BLUE, ORANGE, ORANGE, MUTED]
xi = np.arange(len(vals))
ax.bar(xi, vals, 0.55, color=cols)
for i, v in enumerate(vals):
    ax.text(i, v + 0.025, f'{v:.3f}', ha='center', fontsize=8.5, color=INK)
band = [min(vals[:3]), max(vals[:3])]
ax.axhspan(band[0], band[1], color=AQUA, alpha=0.20, lw=0)
ax.annotate(f'three same-session determinations span {1e3*(band[1]-band[0]):.0f} mV',
            xy=(1.0, band[1] + 0.13), fontsize=8, color=AQUA, ha='center')
ax.axhline(0.77, color=MAGENTA, ls='--', lw=1.3)
ax.annotate('companion run imported 0.77 V from 4 h earlier:\n560 mV low, and the domains never equalised',
            xy=(0.50, 0.62), xycoords='axes fraction', fontsize=7.4, color=MAGENTA, ha='center')
ax.set_xticks(xi); ax.set_xticklabels(names, fontsize=7.8)
ax.set_ylabel('$V_{cpd}$ (V)'); ax.set_ylim(0, 1.32)
ax.set_title('c   Independent determinations agree to 30 mV', loc='left')

# (d) equalisation: ratio vs bias -------------------------------------------
ax = axs[1, 1]
fr = IM['R2']['frames']
pts_A = [(fr['A_cr1_0V']['bias'], fr['A_cr1_0V']['ratio']),
         (fr['A_cr1_Vhalf']['bias'], fr['A_cr1_Vhalf']['ratio']),
         (fr['A_cr1_Vcpd']['bias'], fr['A_cr1_Vcpd']['ratio'])]
pts_B = [(fr['B_cr1_0V']['bias'], fr['B_cr1_0V']['ratio']),
         (fr['B_cr1_Vcpd']['bias'], fr['B_cr1_Vcpd']['ratio'])]
a = np.array(pts_A); b = np.array(pts_B)
ax.plot(a[:, 0], a[:, 1], 'o-', color=BLUE, ms=7, mec=SURF, mew=0.9,
        label='position A  ($E\\approx$ 180)')
ax.plot(b[:, 0], b[:, 1], 's--', color=ORANGE, ms=7, mec=SURF, mew=0.9,
        label='free end  ($E\\approx$ 31)')
r1 = IM['R1']['frames']
ax.plot([r1['A_cr1_0V']['bias'], r1['A_cr1_Vcpd']['bias']],
        [r1['A_cr1_0V']['ratio'], r1['A_cr1_Vcpd']['ratio']], '^:', color=MAGENTA, ms=7,
        mec=SURF, mew=0.9, label='companion run, imported $V_{cpd}$')
ax.axhline(1.0, color=AQUA, lw=1.3, ls='-')
ax.annotate('equal amplitude', xy=(0.06, 1.012), fontsize=7.8, color=AQUA)
for xv, yv in (a[-1], b[-1]):
    pass
ax.annotate('1.011', xy=(a[-1, 0], a[-1, 1]), xytext=(6, -12), textcoords='offset points',
            fontsize=8, color=BLUE)
ax.annotate('1.012', xy=(b[-1, 0], b[-1, 1]), xytext=(6, 6), textcoords='offset points',
            fontsize=8, color=ORANGE)
ax.annotate('0.808', xy=(r1['A_cr1_Vcpd']['bias'], r1['A_cr1_Vcpd']['ratio']),
            xytext=(-4, -14), textcoords='offset points', fontsize=8, color=MAGENTA, ha='right')
ax.set_xlabel('DC bias (V)'); ax.set_ylabel('domain amplitude ratio  $A_1/A_2$')
ax.set_ylim(0.55, 1.12)
ax.set_title('d   At the measured null the domains equalise', loc='left')
ax.legend(fontsize=7.8, loc='lower right')

fig.suptitle('Figure 3.  Contact potential must be measured in the same session — '
             'and when it is, on-resonance PFM equalises',
             fontsize=10.5, color=INK, x=0.008, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.952))
fig.savefig('/home/claude/scmpit_analysis/paper_fig3_cpd.png', dpi=200)
print('saved paper_fig3_cpd.png')
print(f'  working position index {iA}, x_clamp {xc[iA]:.1f}, Vcpd_qs {vqs[iA]:+.3f}')
print(f'  Vcpd along beam (b measurable): {vqs[inner].mean():+.3f} +- {vqs[inner].std():.3f} '
      f'over {inner.sum()} positions')

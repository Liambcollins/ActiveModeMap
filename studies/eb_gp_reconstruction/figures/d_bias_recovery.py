"""Bias-dependence recovery, two slides.

Slide A -- the effect is real, and it is electrostatic:
  the V-shaped cancellation, the straight line in the complex plane, V_cpd.
Slide B -- what survives the cancellation, and why it is not piezoresponse:
  the electrostatic sensitivity along the beam IS the mode shape the sparse
  reconstruction recovers; the residual is not, so it cannot be tip-sample.
"""
import sys
import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_ROOT, 'config.py')):
    _ROOT = _os.path.dirname(_ROOT)
sys.path[:0] = [_ROOT, _os.path.join(_ROOT, 'src'),
                _os.path.join(_ROOT, 'figures')]
from config import (DENSE_GRID_A, FEM_LADDER, EB_REPO, EB_GEOMETRY, OUT, FIG,
                    add_eb_to_path)
add_eb_to_path()
import numpy as np, pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ornl as O

O.style()
B = np.load(str(OUT) + '/bias_analysis.npz')
R = pickle.load(open(str(OUT) + '/recs_cplx.pkl', 'rb'))
xb, V, Zp = B['x'], B['V'], B['Zp']          # Zp (nbias, npos)
a, b, P, se, vcpd, Q = B['a'], B['b'], B['P'], B['se'], B['vcpd'], B['Q']
absb = np.abs(b)
ok = absb > 0.1 * absb.max()                 # positions with usable mode-A gain
VC, VCsd = vcpd[ok].mean(), vcpd[ok].std(ddof=1)

# a fixed, never-cycled hue order; every series is also directly labelled
POSC = [O.GREEN, O.BLUE, O.ORANGE, O.MAGENTA, O.TEAL, O.MUT, O.INK2]

# ======================================================= slide A ============
fig = plt.figure(figsize=(13.2, 4.65))
a1 = fig.add_axes([.048, .155, .245, .70])
a2 = fig.add_axes([.385, .155, .245, .70])
a3 = fig.add_axes([.730, .155, .240, .70])

Vf = np.linspace(-6.6, 6.6, 400)
for i, xx in enumerate(xb):
    a1.plot(V, np.abs(Zp[:, i]) * 1e3, 'o', ms=5.5, color=POSC[i],
            mec=O.WHITE, mew=1.0, zorder=4)
    a1.plot(Vf, np.abs(a[i] + b[i] * Vf) * 1e3, '-', color=POSC[i], lw=1.4,
            alpha=.9, label=f'x = {xx:.0f} µm')
a1.axvline(VC, color=O.INK, lw=1.2, ls='--')
a1.text(VC + .35, .30, f'$V_{{cpd}}$ = {VC:.2f} V', rotation=90, fontsize=9.4,
        color=O.INK, ha='left', va='bottom', fontweight='600')
a1.set_yscale('log'); a1.set_ylim(.25, 900)
a1.set_xlabel('DC bias  (V)')
a1.set_ylabel(r'|Z| at the mode-A peak  ($\times10^{-3}$)')
a1.legend(fontsize=8.2, ncol=4, loc='upper center', labelcolor=O.INK2,
          columnspacing=.8, handlelength=1.2, handletextpad=.4)
O.clean(a1)
O.title(a1, 'Amplitude collapses ~100× at one bias')

SHOW = [0, 4, 5]
for i in SHOW:
    j = list(xb).index(xb[i])
    a2.plot(np.real(Zp[:, i]) * 1e3, np.imag(Zp[:, i]) * 1e3, 'o', ms=6,
            color=POSC[i], mec=O.WHITE, mew=1.0, zorder=4,
            label=f'x = {xb[i]:.0f} µm')
    a2.plot(np.real(a[i] + b[i] * Vf) * 1e3, np.imag(a[i] + b[i] * Vf) * 1e3,
            '-', color=POSC[i], lw=1.3, alpha=.85)
a2.plot([0], [0], '+', ms=13, mew=2.0, color=O.INK, zorder=6)
a2.text(4.0, 5.0, 'origin', fontsize=9.4, color=O.INK)
a2.set_xlabel(r'Re Z  ($\times10^{-3}$)')
a2.set_ylabel(r'Im Z  ($\times10^{-3}$)')
a2.set_aspect('equal', adjustable='datalim')
a2.legend(fontsize=9.0, loc='upper right', labelcolor=O.INK2)
a2.annotate('a straight line through the origin =\none force source scaling '
            'with $(V-V_{cpd})$', xy=(2, -3), xytext=(-44, -68),
            fontsize=9.2, color=O.INK2, ha='left',
            arrowprops=dict(arrowstyle='->', color=O.INK2, lw=1.1))
O.clean(a2)
O.title(a2, 'Z(V) in the complex plane, 3.5 % from a line')

a3.axhspan(VC - VCsd, VC + VCsd, color=O.GREEN, alpha=.13, lw=0, zorder=1)
a3.axhline(VC, color=O.GREEN, lw=1.4, zorder=2)
a3.plot(xb[ok], vcpd[ok], 'o', ms=8, color=O.GREEN, mec=O.WHITE, mew=1.5,
        zorder=5, label='mode-A sensitive')
a3.plot(xb[~ok], vcpd[~ok], 'o', ms=8, mfc=O.WHITE, mec=O.MUT, mew=1.5,
        zorder=5, label='at the node — excluded')
a3.set_xlabel('position from clamp  (µm)')
a3.set_ylabel('$V_{cpd}$ from the complex null  (V)')
a3.set_ylim(-2.9, -1.05); a3.set_xlim(100, 233)
a3.text(.03, .06, f'{VC:.2f} ± {VCsd:.2f} V over {ok.sum()} positions',
        transform=a3.transAxes, fontsize=10, color='#004D21', fontweight='600')
a3.legend(fontsize=9.0, loc='upper left', labelcolor=O.INK2)
O.clean(a3)
O.title(a3, 'One contact potential along the lever')
fig.savefig(f'{FIG}/d3_bias1.png', dpi=170)
plt.close(fig)

# ======================================================= slide B ============
x, Z, sel = R['x'], R['Z'], R['sel']
shape_true = np.abs(Z).max(1)
shape_rec = np.abs(R['fem_gp']['Zrec']).max(1)
inside = (xb >= x.min()) & (xb <= x.max())
sr_at = np.interp(xb, x, shape_rec)
scale = np.exp(np.mean(np.log(absb[inside & ok] / sr_at[inside & ok])))

fig = plt.figure(figsize=(13.2, 4.65))
a1 = fig.add_axes([.052, .155, .265, .68])
a2 = fig.add_axes([.415, .155, .240, .68])
a3 = fig.add_axes([.742, .155, .232, .68])

a1.plot(x, scale * shape_true, '-', color=O.INK, lw=3.0, alpha=.26,
        label='dense mode map (101 pos)', solid_capstyle='round')
a1.plot(x, scale * shape_rec, '-', color=O.GREEN, lw=1.8,
        label='FEM + GP from 8 positions')
sr_eb = np.abs(R['eb_gp']['Zrec']).max(1)
a1.plot(x, scale * sr_eb * np.exp(np.mean(np.log(shape_rec / sr_eb))), '--',
        color=O.BLUE, lw=1.4, label='EB + GP from 8 positions')
a1.plot(xb[ok], absb[ok], 'o', ms=9, color=O.ORANGE, mec=O.WHITE, mew=1.6,
        zorder=6, label='measured $|b|$ = electrostatic slope')
a1.plot(xb[~ok], absb[~ok], 'o', ms=9, mfc=O.WHITE, mec=O.ORANGE, mew=1.8,
        zorder=6)
a1.set_yscale('log'); a1.set_ylim(1.4e-4, 7e-2)
a1.set_xlabel('position from clamp  (µm)')
a1.set_ylabel(r'$|\partial Z/\partial V|$  (per volt)')
a1.legend(fontsize=8.8, loc='upper right', labelcolor=O.INK2)
a1.annotate('predicted to ≤ 4 %\nover 132–196 µm', xy=(180, 6.6e-3),
            xytext=(126, 1.1e-3), fontsize=9.4, color=O.INK2, ha='left',
            arrowprops=dict(arrowstyle='->', color=O.INK2, lw=1.1))
a1.annotate('open = on the node\n(different load — see notes)', xy=(223.4, 4.0e-4),
            xytext=(178, 1.9e-4), fontsize=8.8, color=O.MUT, ha='left',
            arrowprops=dict(arrowstyle='->', color=O.MUT, lw=1.0))
O.clean(a1)
O.title(a1, 'Bias sensitivity IS the mode shape')

w = .36
ix = np.arange(len(xb))
a2.bar(ix - w / 2, absb, w * .95, color=O.ORANGE, label='electrostatic $|b|$',
       zorder=3)
a2.bar(ix + w / 2, P, w * .95, color=O.TEAL, label=r'residual $|P_\perp|$',
       zorder=3)
a2.set_yscale('log'); a2.set_ylim(8e-5, 9e-2)
a2.set_xticks(ix); a2.set_xticklabels([f'{v:.0f}' for v in xb])
a2.set_xlabel('position from clamp  (µm)')
a2.set_ylabel('per volt   /   absolute')
a2.legend(fontsize=9.2, loc='upper right', labelcolor=O.INK2)
a2.text(0, 1.012, f'$|b|$ spans {absb.max()/absb.min():.0f}×,  $|P_\\perp|$ only {P.max()/P.min():.0f}×   ·   corr = {np.corrcoef(np.log(P), np.log(absb))[0,1]:+.2f}',
        transform=a2.transAxes, fontsize=9.0, color=O.INK2, va='bottom')
O.clean(a2); a2.grid(axis='x', visible=False)
a2.set_title('Piezoresponse would track $|b|$. It does not.',
             loc='left', fontweight='600', pad=19, color=O.INK)

dv = P / absb
sev = se / absb
a3.errorbar(np.arange(ok.sum()), dv[ok], yerr=3 * sev[ok], fmt='o', ms=8,
            color=O.GREEN, mec=O.WHITE, mew=1.4, ecolor=O.GREEN, elinewidth=1.4,
            capsize=4, zorder=5)
a3.axhline(0, color=O.MUT, lw=1.0)
mu, sd = dv[ok].mean(), dv[ok].std(ddof=1)
a3.axhspan(mu - sd, mu + sd, color=O.GREEN, alpha=.13, lw=0, zorder=1)
a3.axhline(mu, color=O.GREEN, lw=1.3, ls='--', zorder=2)
a3.set_xticks(range(ok.sum()))
a3.set_xticklabels([f'{v:.0f}' for v in xb[ok]])
a3.set_xlabel('position from clamp  (µm)')
a3.set_ylabel('uncancellable part, as DC volts')
a3.set_ylim(-.45, .65)
a3.text(.03, .95, f'{mu:.3f} ± {sd:.3f} V\n= {100*mu/6:.1f} % of a 6 V drive',
        transform=a3.transAxes, fontsize=10, color='#004D21',
        fontweight='600', va='top')
a3.text(.03, .10, 'error bars 3σ', transform=a3.transAxes, fontsize=9,
        color=O.INK2)
O.clean(a3)
O.title(a3, 'The residual, calibration-free')
fig.savefig(f'{FIG}/d4_bias2.png', dpi=170)
plt.close(fig)

# a couple of numbers the slide text quotes
pred_err = 100 * np.abs(scale * sr_at[inside & ok] / absb[inside & ok] - 1)
print('shape-vs-|b| agreement over the overlapping sensitive positions: '
      f'{pred_err.mean():.1f} % mean, {pred_err.max():.1f} % max '
      f'({(inside & ok).sum()} points)')
print(f'V_cpd {VC:.2f} +/- {VCsd:.2f} V ; residual {mu:.3f} +/- {sd:.3f} V ; '
      f'Q(V=0) {Q[ok].mean():.0f} +/- {Q[ok].std(ddof=1):.0f}')
print('wrote d3_bias1.png, d4_bias2.png')

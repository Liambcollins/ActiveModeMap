"""Remaining deck figures, ORNL palette."""
# Paths are resolved through config.py -- set AMM_* environment
# variables or edit that file to point at your data.
import sys as _sys, os as _os
_D = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_D, 'config.py')):
    _D = _os.path.dirname(_D)
_sys.path[:0] = [_D, _os.path.join(_D, 'src'),
                 _os.path.join(_D, 'figures')]
from config import DENSE_GRID_A, FEM_LADDER, EB_GEOMETRY, OUT, FIG, add_eb_to_path
add_eb_to_path()

import numpy as np, pandas as pd, pickle, matplotlib, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import ornl as O

O.style()
R = pickle.load(open(str(OUT) + '/recs.pkl', 'rb'))
x, f, A, TD, ir = R['x'], R['f'], R['A'], R['TD'], R['ir']
D = pd.concat([pd.read_csv(str(OUT) + '/phys_bench.csv'),
               pd.read_csv(str(OUT) + '/fem_bench.csv')], ignore_index=True)
E = D[D.strategy == 'equispaced']
SEQ = LinearSegmentedColormap.from_list('ornl_seq',
                                        ['#FFFFFF', '#B9D8C4', '#4E9E72',
                                         '#00662C', '#003320'])
DIV = LinearSegmentedColormap.from_list('ornl_div',
                                        ['#006BA6', '#8FC3DC', '#F4F4F2',
                                         '#E8B871', '#C87A00'])
dB = lambda M: 20 * np.log10(np.maximum(M, 1e-16) / A.max())
FK = f / 1e3


def fmt(ax):
    ax.set_xlabel('position from clamp  (µm)')
    ax.set_ylabel('frequency  (kHz)')


# ================= 1. ground truth 2-D spectrum ============================
fig = plt.figure(figsize=(13.2, 4.5))
a1 = fig.add_axes([.055, .155, .40, .70])
a2 = fig.add_axes([.545, .155, .40, .70])
im = a1.pcolormesh(x, FK, dB(A).T, cmap=SEQ, vmin=-60, vmax=0,
                   shading='nearest', rasterized=True)
a1.axvline(TD, color=O.MAGENTA, lw=1.5, ls='--')
a1.text(TD - 1.5, FK[-1] - 4, 'D-NS 224.1 µm ', color=O.MAGENTA, fontsize=9.5,
        ha='right', va='top', fontweight='600')
a1.axhline(FK[ir], color=O.INK, lw=.9, ls=':', alpha=.7)
a1.text(x[0] + 1.5, FK[ir] + 2.0, f'resonance {FK[ir]:.1f} kHz', color=O.WHITE,
        fontsize=9.4, va='bottom', fontweight='600')
fmt(a1); O.clean(a1, grid=False)
O.title(a1, 'Ground truth: every position measured (101 × 412)')
cb = fig.colorbar(im, ax=a1, pad=.02, fraction=.045)
cb.set_label('|Z|  (dB re max)', fontsize=9.5)
cb.outline.set_visible(False)

pk = A.max(1)
a2.plot(x, pk / pk.max(), '-', color=O.GREEN, lw=2.0)
a2.axvline(TD, color=O.MAGENTA, lw=1.5, ls='--')
a2.set_yscale('log')
a2.set_ylim(2e-3, 1.6)
a2.annotate('amplitude collapses ~2 decades\nover the last few µm — this is the\n'
            'feature that makes the map hard',
            xy=(TD - .6, 4e-3), xytext=(x[0] + 3, 6e-3), fontsize=9.6,
            color=O.INK2, ha='left',
            arrowprops=dict(arrowstyle='->', color=O.INK2, lw=1.1))
a2.set_xlabel('position from clamp  (µm)')
a2.set_ylabel('peak |Z| at resonance  (norm.)')
O.clean(a2)
O.title(a2, 'The mode shape along the lever, at resonance')
fig.savefig(str(FIG) + '/d_truth2d.png', dpi=170); plt.close(fig)

# ================= 2. transfer functions vs data ===========================
POS = [0, 55, 88, 96]
fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.9), sharey=True)
fig.subplots_adjust(left=.055, right=.985, top=.90, bottom=.145, wspace=.10)
for ax, ip in zip(axes, POS):
    ax.plot(FK, dB(A[ip]), '-', color=O.INK, lw=2.4, alpha=.30,
            label='measured', solid_capstyle='round')
    ax.plot(FK, dB(R[('eb', 20)]['A'][ip]), '-', color=O.BLUE, lw=1.5,
            label='EB')
    ax.plot(FK, dB(R[('fem', 20)]['A'][ip]), '-', color=O.GREEN, lw=1.5,
            label='FEM')
    O.clean(ax)
    ax.set_xlabel('frequency  (kHz)')
    ax.set_title(f'x = {x[ip]:.1f} µm' + ('   (near D-NS)' if ip == 96 else ''),
                 loc='left', fontsize=10.5, fontweight='600', color=O.INK)
    ax.set_xlim(FK[0], FK[-1])
axes[0].set_ylabel('|Z|  (dB re max)')
axes[0].set_ylim(-86, 3)
axes[0].legend(fontsize=9.5, loc='lower right', labelcolor=O.INK2)
fig.savefig(str(FIG) + '/d_tf.png', dpi=170); plt.close(fig)

# ================= 3. recovered mode shapes ================================
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), sharey=True)
fig.subplots_adjust(left=.055, right=.985, top=.90, bottom=.145, wspace=.09)
for ax, n in zip(axes, (4, 8, 20)):
    t = A.max(1)
    ax.plot(x, t / t.max(), '-', color=O.INK, lw=3.0, alpha=.25,
            label='ground truth')
    for key in ('lowrank', 'eb', 'fem_gp'):
        M = R[(key, n)]['A'].max(1)
        ax.plot(x, M / t.max(), '-', color=O.HUE[key.split('_')[0]], lw=1.6,
                ls='--' if key.endswith('_gp') else '-', label=O.LBL[key])
    sel = R[('fem', n)]['sel']
    ax.plot(x[sel], (t / t.max())[sel], 'o', ms=7, mfc=O.WHITE, mec=O.MAGENTA,
            mew=1.8, label='revealed', zorder=6)
    ax.axvline(TD, color=O.MAGENTA, lw=1.2, ls=':')
    ax.set_yscale('log'); ax.set_ylim(1.5e-3, 2.2)
    ax.set_xlabel('position from clamp  (µm)')
    O.clean(ax)
    ax.set_title(f'n = {n} positions', loc='left', fontsize=11,
                 fontweight='600', color=O.INK)
axes[0].set_ylabel('peak |Z| at resonance  (norm.)')
axes[0].legend(fontsize=9, loc='lower left', labelcolor=O.INK2, ncol=1)
fig.savefig(str(FIG) + '/d_modeshape.png', dpi=170); plt.close(fig)

# ================= 4. recovered 2-D spectra + residuals ====================
fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.85))
fig.subplots_adjust(left=.048, right=.965, top=.88, bottom=.145, wspace=.40)
panels = [(dB(A).T, 'Ground truth', SEQ, dict(vmin=-60, vmax=0), '|Z| (dB)'),
          (dB(R[('fem_gp', 8)]['A']).T, 'FEM + GP, n = 8', SEQ,
           dict(vmin=-60, vmax=0), '|Z| (dB)'),
          ((dB(R[('fem_gp', 8)]['A']) - dB(A)).T, 'residual: FEM + GP, n = 8',
           DIV, dict(vmin=-18, vmax=18), 'Δ|Z| (dB)'),
          ((dB(R[('lowrank', 8)]['A']) - dB(A)).T, 'residual: low-rank, n = 8',
           DIV, dict(vmin=-18, vmax=18), 'Δ|Z| (dB)')]
for k, (ax, (M, t, cm, kw, cl)) in enumerate(zip(axes, panels)):
    im = ax.pcolormesh(x, FK, M, cmap=cm, shading='nearest', rasterized=True,
                       **kw)
    ax.axvline(TD, color=O.MAGENTA, lw=1.1, ls='--')
    ax.set_xlabel('position  (µm)', fontsize=9.8)
    O.clean(ax, grid=False)
    ax.set_title(t, loc='left', fontsize=10.5, fontweight='600', color=O.INK)
    cb = fig.colorbar(im, ax=ax, pad=.04, fraction=.05)
    if k in (0, 2):
        cb.set_label(cl, fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8.5)
    if k:
        ax.set_yticklabels([])
axes[0].set_ylabel('frequency  (kHz)', fontsize=9.8)
fig.savefig(str(FIG) + '/d_2drecon.png', dpi=170); plt.close(fig)

# ================= 5. Q6: FEM beats EB, and the k1 evidence ================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 4.3))
fig.subplots_adjust(left=.06, right=.985, top=.905, bottom=.135, wspace=.22)
w = .34
ns = [3, 4, 5, 6, 8]
for i, (key, lab) in enumerate((('eb', '1  EB'), ('fem', '1b  FEM'))):
    v = [E[(E.rec == key) & (E.n == n)].amp_map_pct.iloc[0] for n in ns]
    b = a1.bar(np.arange(len(ns)) + (i - .5) * w, v, w * .92,
               color=O.HUE[key], label=lab, zorder=3)
    a1.bar_label(b, fmt='%.1f', fontsize=9, color=O.INK2, padding=2)
a1.set_xticks(range(len(ns))); a1.set_xticklabels([f'n = {n}' for n in ns])
a1.set_ylabel('held-out complex-map NRMSE  (%)')
a1.set_ylim(0, 17.5)
a1.legend(fontsize=10, labelcolor=O.INK2)
O.clean(a1); a1.grid(axis='x', visible=False)
O.title(a1, 'FEM wins at every small sample count')

for key, lab in (('eb', '1  EB'), ('fem', '1b  FEM')):
    g = E[E.rec == key].sort_values('n')
    a2.plot(g.n, g.k1, '-', color=O.HUE[key], lw=2.0, marker=O.MK[key],
            ms=5.5, mew=1.6, mec=O.WHITE, label=lab, zorder=4)
a2.axhspan(1000, 2154, color=O.GREEN, alpha=.10, lw=0, zorder=1)
a2.axhline(1310, color=O.GREEN, lw=1.3, ls=(0, (1, 2)), zorder=2)
a2.text(.98, 1330, 'Hertz 1310 N/m ', color='#004D21', fontsize=9.5,
        ha='right', va='bottom', transform=a2.get_yaxis_transform())
a2.text(.98, 2180, 'independent estimates 1000–2154 ', color='#004D21',
        fontsize=9.2, ha='right', va='bottom',
        transform=a2.get_yaxis_transform())
a2.annotate('EB is 3.4× too soft — it buys its fit\nby distorting the contact '
            'parameter', xy=(9, 370), xytext=(4.3, 560), fontsize=9.8,
            color=O.BLUE, ha='left',
            arrowprops=dict(arrowstyle='->', color=O.BLUE, lw=1.1))
a2.set_xscale('log'); a2.set_yscale('log')
a2.set_xticks([3, 4, 5, 6, 8, 10, 12, 16, 20, 30, 50])
a2.set_yticks([300, 500, 800, 1300, 2000])
for A_ in (a2.get_xaxis(), a2.get_yaxis()):
    A_.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: '%g' % v))
    A_.set_minor_locator(matplotlib.ticker.NullLocator())
a2.set_xlabel('positions measured  (n)')
a2.set_ylabel('fitted contact stiffness  (N/m)')
for key, lab, dy in (('eb', '1  EB', 1.0), ('fem', '1b  FEM', 1.0)):
    g = E[E.rec == key].sort_values('n')
    a2.text(53, g.k1.iloc[-1] * dy, lab, color=O.HUE[key], fontsize=10.5,
            fontweight='600', va='center', ha='left')
a2.set_xlim(2.75, 95)
O.clean(a2)
O.title(a2, 'Only FEM recovers a physical contact stiffness')
fig.savefig(str(FIG) + '/d_q6.png', dpi=170); plt.close(fig)

# ================= 6. Q7: FEM still needs the GP ===========================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 4.3))
fig.subplots_adjust(left=.06, right=.985, top=.905, bottom=.135, wspace=.22)
for key in ('lowrank', 'eb', 'eb_gp', 'fem', 'fem_gp'):
    g = E[E.rec == key].sort_values('n')
    a1.plot(g.n, g.amp_map_pct, ls='--' if key.endswith('_gp') else '-',
            color=O.HUE[key.split('_')[0]], lw=2.0, marker=O.MK[key], ms=5.2,
            mew=1.6, mec=O.WHITE, label=O.LBL[key], zorder=4)
a1.axhspan(5.4, 7.1, color=O.GREEN, alpha=.09, lw=0, zorder=1)
a1.annotate('bare FEM never reaches 5 %:\nits discrepancy is larger and more\n'
            'structured than EB\'s at high n',
            xy=(24, 6.4), xytext=(15.5, 17.5), fontsize=9.6, color='#004D21',
            ha='left', arrowprops=dict(arrowstyle='->', color=O.GREEN, lw=1.1))
a1.set_xscale('log'); a1.set_yscale('log')
a1.set_xticks([3, 4, 5, 6, 8, 10, 12, 16, 20, 30, 50])
a1.set_yticks([1.5, 2, 3, 5, 8, 12, 20, 30, 45])
for A_ in (a1.get_xaxis(), a1.get_yaxis()):
    A_.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: '%g' % v))
    A_.set_minor_locator(matplotlib.ticker.NullLocator())
a1.set_xlabel('positions measured  (n)')
a1.set_ylabel('held-out complex-map NRMSE  (%)')
a1.legend(fontsize=9.2, labelcolor=O.INK2, loc='upper right')
O.clean(a1)
O.title(a1, 'The GP is what removes the physics floor')

th = [10, 5, 3, 2]
arms = ['lowrank', 'eb', 'eb_gp', 'fem', 'fem_gp']
a2.set_axis_off()
cw, ch = 1.0, 1.0
for j, q in enumerate(th):
    a2.text(1.5 + j, len(arms) + .34, f'≤ {q} %', ha='center', va='bottom',
            fontsize=10.5, fontweight='600', color=O.INK)
a2.text(2.9, len(arms) + .95, 'positions needed to reach an NRMSE target',
        ha='center', va='bottom', fontsize=11.5, fontweight='600', color=O.INK)
for i, a_ in enumerate(arms):
    yy = len(arms) - 1 - i
    a2.add_patch(plt.Rectangle((.62, yy + .18), .24, .64,
                              color=O.HUE[a_.split('_')[0]],
                              alpha=1.0 if not a_.endswith('_gp') else .5,
                              clip_on=False))
    a2.text(.50, yy + .5, O.LBL[a_], ha='right', va='center', fontsize=10.5,
            color=O.INK)
    t = E[E.rec == a_].sort_values('n')
    for j, q in enumerate(th):
        ok = t.n[t.amp_map_pct <= q]
        v = int(ok.iloc[0]) if len(ok) else None
        # tint by how few positions it takes; "never" stays plainly empty
        sh = (0.0 if v is None else
              max(.10, min(.80, 1.0 - (np.log(v) - np.log(4)) /
                                      (np.log(30) - np.log(4)) * .9)))
        a2.add_patch(plt.Rectangle((1.0 + j, yy + .06), cw - .06, ch - .12,
                                   facecolor=O.GREEN, alpha=sh * .55,
                                   edgecolor=O.WHITE, lw=1.4))
        a2.text(1.5 + j - .03, yy + .5, 'never' if v is None else str(v),
                ha='center', va='center',
                fontsize=13 if v else 9.5, fontweight='600' if v else '400',
                color=O.INK if v else O.INK2,
                style='normal' if v else 'italic')
a2.set_xlim(.0, 5.05); a2.set_ylim(-.55, len(arms) + 1.25)
fig.savefig(str(FIG) + '/d_q7.png', dpi=170); plt.close(fig)

# ================= 7. the caveat: model-locked D-NS ========================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 4.2))
fig.subplots_adjust(left=.06, right=.985, top=.905, bottom=.135, wspace=.20)
for key in ('fem', 'fem_gp', 'eb'):
    g = E[E.rec == key].dropna(subset=['dns']).sort_values('n')
    a1.plot(g.n, g.dns, ls='--' if key.endswith('_gp') else '-',
            color=O.HUE[key.split('_')[0]], lw=2.0, marker=O.MK[key], ms=5.2,
            mew=1.6, mec=O.WHITE, label=O.LBL[key], zorder=4)
a1.axhline(TD, color=O.MAGENTA, lw=1.4, ls='--', zorder=2)
a1.text(.02, TD + .07, ' ground truth 224.09 µm', color=O.MAGENTA,
        fontsize=9.5, ha='left', va='bottom',
        transform=a1.get_yaxis_transform())
a1.annotate('bare FEM returns the SAME value\nfor every n from 4 to 16 — a '
            'property\nof the model, not of the data',
            xy=(14, 223.58), xytext=(52, 221.75), fontsize=9.6,
            color='#004D21', ha='right', va='bottom',
            arrowprops=dict(arrowstyle='->', color=O.GREEN, lw=1.1))
a1.set_xscale('log')
a1.set_xticks([3, 4, 5, 6, 8, 10, 12, 16, 20, 30, 50])
a1.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda v, _: '%g' % v))
a1.get_xaxis().set_minor_locator(matplotlib.ticker.NullLocator())
a1.set_xlabel('positions measured  (n)')
a1.set_ylabel('estimated D-NS  (µm)')
a1.set_ylim(221.15, 225.6)
a1.legend(fontsize=9.5, loc='upper left', labelcolor=O.INK2)
O.clean(a1)
O.title(a1, 'D-NS estimate vs sample count')

r = D[(D.strategy == 'random') & (D.n == 5)]
lab, vals, cols = [], [], []
for key in ('eb', 'eb_gp', 'fem', 'fem_gp'):
    v = r[r.rec == key].dns.dropna().values
    if len(v):
        lab.append(O.LBL[key]); vals.append(v)
        cols.append(O.HUE[key.split('_')[0]])
bp = a2.boxplot(vals, vert=False, widths=.55, patch_artist=True,
                medianprops=dict(color=O.WHITE, lw=1.8),
                whiskerprops=dict(color=O.INK2, lw=1.1),
                capprops=dict(color=O.INK2, lw=1.1),
                flierprops=dict(marker='o', ms=4, mfc=O.INK2, mec='none',
                                alpha=.6))
for p, c in zip(bp['boxes'], cols):
    p.set(facecolor=c, edgecolor='none', alpha=.9)
a2.axvline(TD, color=O.MAGENTA, lw=1.4, ls='--')
a2.set_yticklabels(lab)
a2.set_xlabel('estimated D-NS across 24 random 5-point designs  (µm)')
a2.annotate('spread 0.04 µm — the answer does not\ncare where you measured',
            xy=(223.60, 2.94), xytext=(224.80, 1.30), fontsize=9.6,
            color='#004D21', ha='right', va='center',
            arrowprops=dict(arrowstyle='->', color=O.GREEN, lw=1.1))
a2.annotate('spread 3.1 µm — this arm\nactually responds to the data',
            xy=(222.10, 4.0), xytext=(221.35, 4.78), fontsize=9.6,
            color='#004D21', ha='left', va='top',
            arrowprops=dict(arrowstyle='->', color=O.GREEN, lw=1.1))
a2.set_xlim(221.25, 224.85); a2.set_ylim(.45, 5.05)
O.clean(a2); a2.grid(axis='y', visible=False)
O.title(a2, 'Does the estimate move when the design moves?')
fig.savefig(str(FIG) + '/d_caveat.png', dpi=170); plt.close(fig)
print('wrote 6 figures')

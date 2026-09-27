r"""R1 vs R2 summary: noise-floor fit, label cross-check, and the comparison figure."""
import json

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

UP = '/mnt/user-data/uploads/ActiveModeMap/DomainsB_SCMPIT_R2'
S1 = json.load(open(f'{UP}/s1_structure.json'))
S2 = json.load(open(f'{UP}/s2_ac_closeout.json'))
IM = json.load(open('/home/claude/scmpit_analysis/r2_images.json'))
WB = json.load(open('/home/claude/scmpit_analysis/r2_wideband.json'))

INK, SEC, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9, 'axes.edgecolor': '#c3c2b7',
                     'axes.labelcolor': SEC, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.titlecolor': INK, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'legend.frameon': False,
                     'lines.linewidth': 1.6})
CL = {'R1': BLUE, 'R2': ORANGE}

# ---------------------------------------------------------------- label check
print('=== DOMAIN LABEL CROSS-CHECK (images vs spectroscopy) ===')
for tag in ('R1', 'R2'):
    img = IM[tag]['frames']['A_cr1_0V']['ratio']          # geometry-anchored domain1/domain2
    rows = S2['ac'][tag]['rows']
    hi = [r for r in rows if abs(r['vac_V'] - 1.0) < 1e-9 and r['bias_V'] == 0]
    s1 = next(r['amp_V'] for r in hi if r['spot'] == 1)
    s2 = next(r['amp_V'] for r in hi if r['spot'] == 2)
    print(f'  {tag}: image domain1/domain2 = {img:.3f}   spectroscopy spot1/spot2 = {s1/s2:.3f}   '
          f'-> {"MATCH (domain1 = spot1)" if abs(img - s1/s2) < 0.12 else "labels differ"}')

# -------------------------------------------------------------- noise floor
print('\n=== AC SERIES: additive-noise-floor fit  |Z| = sqrt((k*Vac)^2 + n0^2) ===')
nf = {}
for tag in ('R1', 'R2'):
    rows = [r for r in S2['ac'][tag]['rows'] if r['bias_V'] == 0]
    for spot in (1, 2):
        v = np.array(sorted({r['vac_V'] for r in rows}))
        a = np.array([np.mean([r['amp_V'] for r in rows if r['vac_V'] == vv and r['spot'] == spot])
                      for vv in v])
        hi = v >= 0.1
        k = float(np.mean(a[hi] / v[hi]))
        n2 = np.maximum(a ** 2 - (k * v) ** 2, 0)
        n0 = float(np.sqrt(np.mean(n2[v <= 0.01]))) if (v <= 0.01).any() else float('nan')
        pred = np.sqrt((k * v) ** 2 + n0 ** 2)
        resid = float(np.median(np.abs(pred / a - 1)))
        nf[f'{tag}_s{spot}'] = dict(v=v.tolist(), a=a.tolist(), k=k, n0=n0, resid=resid)
        print(f'  {tag} spot {spot}: k = {k*1e3:.2f} mV/V,  n0 = {n0*1e6:.1f} uV,  '
              f'median |pred/meas - 1| = {100*resid:.1f} %   '
              f'(signal = floor at Vac = {n0/k*1e3:.1f} mV)')

# ------------------------------------------------------------------- figure
fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.8))

# A. CR1 frequency along the beam, both campaigns + both close-outs
ax = axs[0, 0]
for tag in ('R1', 'R2'):
    d = S1['dense'][tag]
    ax.plot(d['x_clamp'], np.array(d['cr1_f']) / 1e3, '-', color=CL[tag], lw=1.4,
            label=f'{tag} dense map ({len(d["x_clamp"])} pos)')
    w = S2['walk'][tag]
    ax.plot(np.array(w['x_um']) - 6.1, np.array(w['f']) / 1e3, 'o', color=CL[tag], ms=4,
            mec=SURF, mew=0.7, label=f'{tag} close-out walk')
ax.axhline(285.50, color=MUTED, ls='--', lw=1)
ax.text(70, 285.9, 'R1 pre-flight 285.50 kHz', fontsize=7.5, color=SEC)
ax.set_xlabel('distance from clamp (µm)'); ax.set_ylabel('CR1 frequency (kHz)')
ax.set_title('A.  The new contact state held overnight', loc='left')
ax.legend(fontsize=7.5, loc='center left')

# B. mode-shape amplitude, normalised
ax = axs[0, 1]
for tag in ('R1', 'R2'):
    d = S1['dense'][tag]
    a = np.array(d['cr1_a']); a = a / a.max()
    ax.plot(d['x_clamp'], a, '-', color=CL[tag], lw=1.5, label=f'{tag} CR1')
    for nm, ls in (('cr2', '--'), ('cr3', ':')):
        b = np.array(d[f'{nm}_a']); b = b / b.max()
        ax.plot(d['x_clamp'], b, ls, color=CL[tag], lw=1.1, alpha=0.75,
                label=f'{tag} {nm.upper()}')
for tag in ('R1', 'R2'):
    for nm in ('cr2', 'cr3'):
        for nd in S1['dense'][tag].get(f'{nm}_nodes', []):
            ax.axvline(nd, color=CL[tag], lw=0.8, alpha=0.45)
ax.set_xlabel('distance from clamp (µm)'); ax.set_ylabel('|Z| / max')
ax.set_title('B.  Every node moved ~11–14 µm toward the free end', loc='left')
ax.legend(fontsize=6.8, ncol=2, loc='upper left')

# C. Vcpd trajectory across both campaigns
ax = axs[0, 2]
t = [12.87, 20.10, 21.27, 22.32, 24.05, 25.05, 29.45]
v = [0.77, 1.33, 1.22, 1.191, 1.057, 1.027, 0.836]
lab = ['R1 s1', 'R1 s5', 'R1 s7', 'R1 close', 'R2 s1', 'R2 s5', 'R2 close']
col = [BLUE] * 4 + [ORANGE] * 3
ax.plot(t[:4], v[:4], '-', color=BLUE, lw=1.4)
ax.plot(t[4:], v[4:], '-', color=ORANGE, lw=1.4)
ax.plot([t[3], t[4]], [v[3], v[4]], ':', color=MUTED, lw=1.2)
offs = [(0, 11), (-2, 13), (26, -4), (10, -24), (-4, -26), (16, 9), (0, 11)]
for ti, vi, li, ci, of in zip(t, v, lab, col, offs):
    ax.plot(ti, vi, 'o', color=ci, ms=6, mec=SURF, mew=0.9)
    ax.annotate(f'{li}\n{vi:.2f}', xy=(ti, vi), xytext=of, textcoords='offset points',
                ha='center', fontsize=7, color=INK)
ax.axvspan(18.6, 19.77, color=MAGENTA, alpha=0.13, lw=0)
ax.text(19.2, 0.62, 'load\nladder', ha='center', fontsize=7.5, color=MAGENTA)
ax.set_xlabel('hours from R1 start (12:52 = 12.87)'); ax.set_ylabel('$V_{cpd}$ (V)')
ax.set_ylim(0.55, 1.62)
ax.set_title('C.  $V_{cpd}$ jumped once, then relaxed monotonically', loc='left')

# D. AC series with the noise-floor model
ax = axs[1, 0]
for tag in ('R1', 'R2'):
    for spot, mk in ((1, 'o'), (2, 's')):
        e = nf[f'{tag}_s{spot}']
        v = np.array(e['v']); a = np.array(e['a'])
        ax.plot(v * 1e3, a / v * 1e3, mk, color=CL[tag], ms=4, mec=SURF, mew=0.6,
                alpha=0.9, label=f'{tag} spot {spot}')
        vv = np.logspace(np.log10(v.min()), np.log10(v.max()), 200)
        ax.plot(vv * 1e3, np.sqrt((e['k'] * vv) ** 2 + e['n0'] ** 2) / vv * 1e3, '-',
                color=CL[tag], lw=1.0, alpha=0.55)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('$V_{ac}$ (mV)'); ax.set_ylabel('apparent |Z| / $V_{ac}$  (mV/V)')
ax.set_title('D.  The low-drive rise is a fixed additive floor, not nonlinearity', loc='left')
ax.legend(fontsize=7, ncol=2, loc='upper right')

# E. in-situ enhancement and quasi-static d33 at both positions
ax = axs[1, 1]
w = 0.18
labels, r1v, r2v = [], [], []
for pos, fr in (('A', 'A_cr1_0V'), ('B', 'B_cr1_0V')):
    for dom in (1, 2):
        labels.append(f'{pos}\nD{dom}')
        r1v.append(IM['R1']['frames'][fr][f'E{dom}'])
        r2v.append(IM['R2']['frames'][fr][f'E{dom}'])
xi = np.arange(len(labels))
ax.bar(xi - w, r1v, 2 * w, color=BLUE, label='R1')
ax.bar(xi + w, r2v, 2 * w, color=ORANGE, label='R2')
for i, (a, b) in enumerate(zip(r1v, r2v)):
    ax.text(i, max(a, b) * 1.06, f'×{b/a:.2f}', ha='center', fontsize=7.5, color=SEC)
ax.set_xticks(xi); ax.set_xticklabels(labels)
ax.set_yscale('log'); ax.set_ylabel('in-situ CR1 enhancement E')
ax.set_title('E.  Enhancement fell at A, grew at the free end', loc='left')
ax.legend(fontsize=8, loc='upper right')

# F. quasi-static d33 -- the model-free number
ax = axs[1, 2]
labels, r1v, r2v = [], [], []
for pos, fr in (('A', 'A_qs20k_0V'), ('B', 'B_qs20k_0V')):
    for dom in (1, 2):
        labels.append(f'{pos}\nD{dom}')
        r1v.append(IM['R1']['frames'][fr][f'd33_{dom}'])
        r2v.append(IM['R2']['frames'][fr][f'd33_{dom}'])
xi = np.arange(len(labels))
ax.bar(xi - w, r1v, 2 * w, color=BLUE, label='R1')
ax.bar(xi + w, r2v, 2 * w, color=ORANGE, label='R2')
for i, (a, b) in enumerate(zip(r1v, r2v)):
    ax.text(i, max(a, b) + 0.35, f'{100*(b/a-1):+.0f}%', ha='center', fontsize=7.5, color=SEC)
ax.axhspan(7.9, 8.02, color=AQUA, alpha=0.22, lw=0)
ax.text(3.42, 4.4, 'band = R1 stage-1\nspectroscopy\n7.9-8.0 pm/V', fontsize=7, color=AQUA, ha='right')
ax.set_xticks(xi); ax.set_xticklabels(labels)
ax.set_ylabel('quasi-static $d_{33}$ (pm/V)')
ax.set_ylim(0, 13.6)
ax.set_title('F.  The sample reproduces within 6 %', loc='left')
ax.legend(fontsize=8, loc='upper left')

fig.suptitle('SCM-PIT on PPLN DomainsB — R1 (2026-09-20, with load ladder) vs R2 (overnight repeat, same probe, no load ladder)\n'
             'R2 ran 22:56–05:28 into DomainsB_SCMPIT_R2; 12 stages, zero failures',
             fontsize=10.5, color=INK, x=0.008, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.925))
fig.savefig('/home/claude/scmpit_analysis/figO_r1_vs_r2.png', dpi=150)
print('\nsaved figO_r1_vs_r2.png')
json.dump(nf, open('/home/claude/scmpit_analysis/r2_noisefloor.json', 'w'), indent=1)

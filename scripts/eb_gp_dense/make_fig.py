import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from eb_gp_test import (SingleDomainPosterior, SingleDomainHybrid, build_model,
                        subset_indices, dns_um_from_end, run)
from activemodemap.lowrank import reconstruct_map
from prep import load_dense, to_model_grid, estimate_sigma, OM1

plt.rcParams.update({'font.family': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
                     'font.size': 9, 'axes.linewidth': 0.8})
ORNL = (27 / 255, 94 / 255, 32 / 255)
C_EB, C_HY, C_LR, C_TR = '#b06000', '#1f4e79', '#2f9e5f', '0.25'


def fwhm_Q(f, a):
    i = int(np.argmax(a)); h = a[i] / np.sqrt(2)
    j = i
    while j < len(a) - 1 and a[j] > h: j += 1
    k = i
    while k > 0 and a[k] > h: k -= 1
    bw = f[j] - f[k]
    return (f[i], f[i] / bw if bw > 0 else np.nan)


res = run(n_list=(4, 5, 6, 8, 10, 14, 20), verbose=False)
rows, truth = res['rows'], res['truth_dns']
dense = load_dense()
model = build_model(L_um=dense['L_um'])
xi, Zg, scale = to_model_grid(dense, model)
sigma = estimate_sigma(Zg)
f_model = res['f_model']; x_um = res['x_um']; L = res['L']

N = [r['n'] for r in rows]

# --- one detailed fit at n = 10, for the spectrum and profile panels ---------
n_show = 10
idx = subset_indices(x_um.size, n_show)
data = [{'x': xi[i], 'plus': Zg[i]} for i in idx]
post = SingleDomainPosterior(model, sigma=sigma, rng=np.random.default_rng(0))
post.fit(data)
r0 = model.response(post.theta_map)
cols = [int(np.argmin(np.abs(model.xi - v))) for v in xi]
eb = (r0['A0'] * (post._blur(r0['piezo']) + r0['eps'] * post._blur(r0['elec'])))[:, cols].T
hy = SingleDomainHybrid(post).update(data).corrected_maps()[:, cols].T
lr = reconstruct_map(x_um, idx, Zg[idx], rank=4)['Zrec']

f_meas, q_meas = fwhm_Q(f_model, np.abs(Zg[np.argmin(np.abs(x_um - 260))]))
f_eb, q_eb = fwhm_Q(f_model, np.abs(eb[np.argmin(np.abs(x_um - 260))]))

fig = plt.figure(figsize=(11.5, 7.2))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.26,
                      left=0.075, right=0.975, top=0.86, bottom=0.085)

# A: complex RMSE vs n
ax = fig.add_subplot(gs[0, 0])
for key, c, lab in (('crmse_eb', C_EB, 'EB only'), ('crmse_hy', C_HY, 'EB + GP'),
                    ('crmse_lr', C_LR, 'low-rank (model-light)')):
    ax.semilogy(N, [r[key] for r in rows], 'o-', ms=4, color=c, label=lab)
ax.set_xlabel('measured positions'); ax.set_ylabel('complex map RMSE (norm.)')
ax.set_title('Map error vs number of positions', color=ORNL, fontweight='bold',
             fontsize=10.5, loc='left')
ax.legend(frameon=False, fontsize=8.5)

# B: D-NS error vs n
ax2 = fig.add_subplot(gs[0, 1])
ax2.axhline(0, color='0.6', lw=0.7)
for key, c, lab in (('dns_eb', C_EB, 'EB only'), ('dns_hy', C_HY, 'EB + GP'),
                    ('dns_lr', C_LR, 'low-rank')):
    ax2.plot(N, [r[key] - truth for r in rows], 'o-', ms=4, color=c, label=lab)
ax2.axhspan(-1, 1, color='0.85', zorder=0)
ax2.set_xlabel('measured positions'); ax2.set_ylabel('D-NS error (µm)')
ax2.set_title(f'Null position vs dense truth ({truth:.2f} µm from tip)',
              color=ORNL, fontweight='bold', fontsize=10.5, loc='left')
ax2.legend(frameon=False, fontsize=8.5)

# C: spectrum at one position
ip = int(np.argmin(np.abs(x_um - 260)))
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(f_model / 1e3, np.abs(Zg[ip]), color=C_TR, lw=2.0, label='measured')
ax3.plot(f_model / 1e3, np.abs(eb[ip]), color=C_EB, lw=1.1, label='EB only')
ax3.plot(f_model / 1e3, np.abs(hy[ip]), color=C_HY, lw=1.1, ls='--', label='EB + GP')
ax3.set_xlim(58, 70)
ax3.set_xlabel('frequency (kHz)'); ax3.set_ylabel('|response| (norm.)')
ax3.set_title(f'Spectrum at x = {x_um[ip]:.0f} µm  ·  measured Q {q_meas:.0f}, '
              f'EB Q {q_eb:.0f}', color=ORNL, fontweight='bold', fontsize=10.5, loc='left')
ax3.legend(frameon=False, fontsize=8.5)

# D: on-resonance profile vs position
ires = int(np.argmax(np.abs(Zg[ip])))
ax4 = fig.add_subplot(gs[1, 1])
tip = L - x_um
for M, c, lab, ls in ((Zg, C_TR, 'measured (75 pos)', '-'), (eb, C_EB, 'EB only', '-'),
                      (hy, C_HY, 'EB + GP', '--'), (lr, C_LR, 'low-rank', ':')):
    ax4.semilogy(tip, np.abs(M[:, ires]), ls, color=c, lw=1.6 if lab.startswith('meas') else 1.1,
                 label=lab)
ax4.axvline(truth, color='#b00020', lw=0.9)
ax4.set_xlabel('distance from free end (µm)'); ax4.set_ylabel('|response| at f$_{res}$')
ax4.set_title('On-resonance profile and the null', color=ORNL, fontweight='bold',
              fontsize=10.5, loc='left')
ax4.legend(frameon=False, fontsize=8.5)

fig.suptitle('EB + GP-discrepancy recovery tested on the PPP-CONT-AU dense sweep',
             color=ORNL, fontweight='bold', fontsize=13, x=0.075, ha='left', y=0.965)
fig.text(0.075, 0.915,
         f'75 measured positions (x = 75–445 µm), 50–80 kHz band around contact mode 1 '
         f'at 63.47 kHz · single domain, 0 V, 100 nN · sparse subsets equispaced; '
         f'n = {n_show} shown in the lower panels',
         fontsize=8.5, style='italic', color='0.35')
fig.savefig('EBGP_dense_test.png', dpi=190)

print('theta at n=%d: %s' % (n_show, np.round(post.theta_map, 3)))
print('  names', model.PARAM_NAMES, ' bounds lo', model.PRIOR_LO, 'hi', model.PRIOR_HI)
print('measured f_res %.3f kHz Q %.0f | EB f_res %.3f kHz Q %.0f'
      % (f_meas / 1e3, q_meas, f_eb / 1e3, q_eb))
print('\n n  | cplx RMSE  EB / EB+GP / LR | D-NS err  EB / EB+GP / LR')
for r in rows:
    print('%3d | %.4f %.4f %.4f | %+7.2f %+7.2f %+7.2f'
          % (r['n'], r['crmse_eb'], r['crmse_hy'], r['crmse_lr'],
             r['dns_eb'] - truth, r['dns_hy'] - truth, r['dns_lr'] - truth))

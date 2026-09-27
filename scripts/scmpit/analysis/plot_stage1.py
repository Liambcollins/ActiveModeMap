import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = np.load('stage1_compact.npz'); e = np.load('eb_fit.npz'); j = np.load('eb_joint.npz'); nb = np.load('eb_joint_noE.npz')
F, X, Z = d['F'], d['X'], d['Z']; bias, spot, inv = d['bias'], d['spot'], d['invols']
CR1 = float(d['cr1']); VAC = 1.0
a1, b1, a2, b2 = e['a1'], e['b1'], e['a2'], e['b2']
xt = e['xt']; L = float(e['L_true'])
o = np.argsort(X)[::-1]                # free end first
iL = int(np.argmax(X))

# palette (dataviz reference instance, light)
C1, C2, C3, C4, C5 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
INK, SEC, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9, 'axes.edgecolor': '#c3c2b7',
                     'axes.labelcolor': SEC, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.titlecolor': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'legend.frameon': False, 'lines.linewidth': 1.5,
                     'lines.markersize': 6})
MS = dict(markersize=6, markeredgewidth=1.2, markeredgecolor=SURF)

qs = (F >= 15e3) & (F < 45e3)
w1 = (F >= 275e3) & (F <= 296e3)
jpk = np.array([np.where(w1)[0][np.argmax(np.abs((a1 - a2)[i, w1]))] for i in range(X.size)])

# ---------------------------------------------------------------- Fig A: bias dependence
fig, axs = plt.subplots(2, 8, figsize=(17, 5.2), sharex=True)
Vb = np.linspace(-9.5, 9.5, 100)
for col, i in enumerate(o):
    for row, (lab, sel) in enumerate([('quasi-static 15–45 kHz', None), ('CR1 peak', None)]):
        ax = axs[row, col]
        for s, c, nm in [(1, C1, 'domain 1 (spot 1)'), (2, C2, 'domain 2 (spot 2)')]:
            idx = np.where(spot == s)[0]
            if row == 0:
                y = np.array([np.median(np.abs(Z[k, i, qs])) for k in idx]) * 1e3
                aa, bb = (a1, b1) if s == 1 else (a2, b2)
                fit = np.median(np.abs(aa[i, qs][None, :] + Vb[:, None] * bb[i, qs][None, :]), axis=1) * 1e3
            else:
                y = np.abs(Z[idx, i, jpk[i]]) * 1e3
                aa, bb = (a1, b1) if s == 1 else (a2, b2)
                fit = np.abs(aa[i, jpk[i]] + Vb * bb[i, jpk[i]]) * 1e3
            ax.plot(Vb, fit, color=c, lw=1.2, alpha=0.8)
            ax.plot(bias[idx], y, 'o', color=c, label=nm if (row == 0 and col == 0) else None, **MS)
        if row == 0:
            ax.set_title(f'x = {xt[i]:.0f} µm' + ('  (free end)' if i == iL else ''), fontsize=9)
        if col == 0:
            ax.set_ylabel(f'|Z|  (mV)\n{lab}')
        ax.set_ylim(bottom=0)
        ax.axvline(0, color=GRID, lw=0.8)
    axs[1, col].set_xlabel('tip bias (V)')
axs[0, 0].legend(loc='lower right', fontsize=7.5)
fig.suptitle('Two-domain bias dependence, SCM-PIT at 500 nN — points: data, lines: |a + bV| complex fit  '
             '(the two |Z| minima straddle V_cpd; the domains flip sign of P, not of b)', fontsize=10, color=INK, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig('figA_bias_dependence.png', dpi=150); plt.close(fig)

# ---------------------------------------------------------------- Fig B: CPD and channel ratio
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ax1.errorbar(xt, e['v_qs'], yerr=e['sd_qs'], fmt='o', color=C1, ecolor=C1, elinewidth=1, capsize=2, label='quasi-static (15–45 kHz)', **MS)
ax1.errorbar(xt + 1.5, e['v_c1'], yerr=e['sd_c1'], fmt='s', color=C2, ecolor=C2, elinewidth=1, capsize=2, label='CR1 (285.5 kHz)', **MS)
vmed = np.median(e['v_c1'])
ax1.axhline(vmed, color=C2, lw=0.8, ls='--')
ax1.text(xt.min() - 8, vmed - 0.35, f'CR1 median {vmed:+.2f} V', color=SEC, fontsize=8)
ax1.set_ylim(-1.5, 3.5); ax1.set_xlabel('distance from clamp (µm)'); ax1.set_ylabel('V_cpd  (V)')
ax1.set_title('Contact potential recovered per position', loc='left')
ax1.legend(loc='upper left', fontsize=8)
ax1.annotate('free end: b → 0 in the\nquasi-static band, CPD\nundetermined there', xy=(xt[iL], 1.0), xytext=(150, 2.6),
             fontsize=8, color=SEC, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
ax2.plot(xt, e['r_qs'], 'o', color=C1, label='quasi-static, measured', **MS)
ax2.plot(xt, e['r_c1'], 's', color=C2, label='CR1, measured', **MS)
gq = np.median(e['r_qs'] / e['eb_ratio_qs']); gc = np.median(e['r_c1'] / e['eb_ratio_cr1'])
xs = np.linspace(xt.min(), xt.max(), 200)
ax2.plot(xs, np.interp(xs, xt[np.argsort(xt)], (gq * e['eb_ratio_qs'])[np.argsort(xt)]), color=C1, lw=1.2, alpha=0.7, label='EB model (distributed electrostatic load)')
ax2.plot(xs, np.interp(xs, xt[np.argsort(xt)], (gc * e['eb_ratio_cr1'])[np.argsort(xt)]), color=C2, lw=1.2, alpha=0.7, label='EB model, CR1')
ax2.set_xlabel('distance from clamp (µm)'); ax2.set_ylabel('|b| / |P|   (per volt)')
ax2.set_title('Electrostatic-to-piezo channel ratio', loc='left'); ax2.set_ylim(bottom=0)
ax2.legend(loc='upper right', fontsize=8)
fig.tight_layout(); fig.savefig('figB_cpd_channel_ratio.png', dpi=150); plt.close(fig)

# ---------------------------------------------------------------- Fig C: d33 vs position
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ax1.axhspan(7.5, 8.5, color=GRID, alpha=0.5, lw=0)
ax1.plot(xt, e['d_qs_invols'], 'o-', color=C1, label='quasi-static  |P|·InvOLS(x)/32 / V_ac', **MS)
ax1.plot(xt, nb['d_cr1_model'], 's-', color=C2, label='CR1  |P|_CR1·InvOLS(tip)/32 / (V_ac·H_EB)  — EB fitted blind to CR1 shape', **MS)
ax1.plot(xt, j['d_cr1_model'], '^-', color=C3, label='CR1, EB fitted with CR1 shape', **MS)
ax1.plot(xt, nb['d_qs_model'], 'D-', color=C4, label='quasi-static via EB static shape + InvOLS(tip) only', **MS)
ax1.set_ylim(4, 13); ax1.set_xlabel('distance from clamp (µm)'); ax1.set_ylabel('d_eff  (pm/V)')
ax1.set_title('d33 recovered along the lever', loc='left'); ax1.legend(loc='lower left', fontsize=7.5)
ax1.annotate(f'free end (CR1 node):\nblind EB gives {nb["d_cr1_model"][iL]:.1f} — off scale', xy=(xt[iL], 12.9), xytext=(160, 11.6),
             fontsize=7.5, color=SEC, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
ax1.set_xlim(35, 235)
ax2.plot(xt, e['d_cr1_Q'], 'o-', color=C5, label='CR1 amplitude ÷ Q  (the usual assumption)', **MS)
ax2.plot(xt, nb['d_cr1_model'], 's-', color=C2, label='CR1 ÷ EB transfer function', **MS)
ax2.plot(xt, e['d_qs_invols'], '-', color=C1, lw=1.2, label='quasi-static', alpha=0.8)
ax2.set_yscale('log'); ax2.set_ylim(0.1, 40)
ax2.set_xlabel('distance from clamp (µm)'); ax2.set_ylabel('d_eff  (pm/V), log')
ax2.set_title('Why Q alone cannot convert a CR1 amplitude', loc='left'); ax2.legend(loc='lower right', fontsize=8)
ax2.set_xlim(35, 235)
fig.tight_layout(); fig.savefig('figC_d33_vs_position.png', dpi=150); plt.close(fig)

# ---------------------------------------------------------------- Fig D: cantilever transfer
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
xi = nb['xi_um']; zq_full, zc_full = nb['zp_qs_full'], nb['zp_cr1_full']
gs = nb['g_static']
sm = (1 / inv) / (gs * zq_full[np.argmin(np.abs(xi - xt[iL]))])            # normalise both to tip = 1 via model
ax1.plot(xi, zq_full / zq_full[np.argmin(np.abs(xi - xt[iL]))], color=C1, lw=1.2, alpha=0.8, label='EB quasi-static shape')
ax1.plot(xt, (1 / inv) / (1 / inv)[iL], 'o', color=C1, label='1/InvOLS(x) from force curves', **MS)
pc = e['P_cr1']; Ecal = nb['zc'][iL]
ax1.plot(xi, zc_full / zq_full[np.argmin(np.abs(xi - xt[iL]))] / np.max(zc_full / zq_full[np.argmin(np.abs(xi - xt[iL]))]), color=C2, lw=1.2, alpha=0.8, label='EB CR1 mode shape (norm.)')
pc_n = pc / pc.max()
ax1.plot(xt, pc_n, 's', color=C2, label='|P|_CR1(x)  (norm.; QPDI reads displacement directly)', **MS)
ax1.set_xlabel('distance from clamp (µm)'); ax1.set_ylabel('displacement response (norm.)')
ax1.set_title('Static and CR1 displacement shapes: data vs EB', loc='left'); ax1.legend(loc='upper left', fontsize=7.5)
ax1.axvline(L, color=MUTED, lw=0.8, ls=':'); ax1.text(L - 2, 0.95, 'free end', ha='right', fontsize=8, color=SEC)
ax1.set_xlim(0, 235); ax1.set_ylim(bottom=0)
ax2.plot(xt, e['E_meas'], 'o', color=C2, label='measured  |P|_CR1 / |P|_qs', **MS)
ax2.plot(xt, nb['E_mod'], '-', color=C2, lw=1.2, alpha=0.8, label='EB prediction (fit to f_CR1–3, Q, static shape only)')
ax2.plot(xt, j['E_mod'], '--', color=C3, lw=1.2, alpha=0.9, label='EB fit including CR1 shape')
ax2.axhline(float(e['Q_meas']), color=MUTED, lw=1, ls='--'); ax2.text(40, float(e['Q_meas']) * 1.12, f'Q = {float(e["Q_meas"]):.0f}', color=SEC, fontsize=8)
ax2.set_yscale('log'); ax2.set_ylim(1, 1000)
ax2.set_xlabel('distance from clamp (µm)'); ax2.set_ylabel('CR1 enhancement over quasi-static')
ax2.set_title('Resonance enhancement is a position, not a number', loc='left'); ax2.legend(loc='lower left', fontsize=8)
ax2.set_xlim(35, 235)
fig.tight_layout(); fig.savefig('figD_cantilever_transfer.png', dpi=150); plt.close(fig)
print('figures written')

"""Two-band fitting: what including flexural mode 2 buys."""
import numpy as np, pickle, matplotlib.pyplot as plt, matplotlib.ticker as T
import physrec as P, twoband as TB, ornl as O
import pickle as _pk
O.style()
P.use_library('out/eblib_x.npz', u_hi=3.45)
x, f, A = TB.prep_wide()
F = P.noise_floor(A)
res = pickle.load(open('out/twoband.pkl', 'rb'))
dB = lambda M: 20 * np.log10(M + F)
NS = [3, 4, 5, 6, 8, 12, 20]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 4.35),
                             gridspec_kw={'width_ratios': [1, 1.35]})
fig.subplots_adjust(left=.058, right=.985, top=.90, bottom=.135, wspace=.19)

# ---- left: fitted k1 vs n, A-only vs A+B
a1.axhspan(1000, 2154, color=O.GREEN, alpha=.10, lw=0, zorder=1)
a1.axhline(1310, color=O.GREEN, lw=1.3, ls=(0, (1, 2)), zorder=2)
a1.text(.98, 1330, 'Hertz 1310 N/m ', color='#004D21', fontsize=9.2,
        ha='right', va='bottom', transform=a1.get_yaxis_transform())
a1.axhline(1046, color=O.MAGENTA, lw=1.1, ls='--', zorder=2)
a1.text(.98, 960, 'mode-ratio-implied 1046 ', color=O.MAGENTA, fontsize=9.2,
        ha='right', va='top', transform=a1.get_yaxis_transform())
K2G = {}
for n in NS:
    sel = sorted(np.linspace(0, len(x) - 1, n).round().astype(int).tolist())
    K2G[n] = TB.rec2g(x, sel, A[sel], f, gp=False)['theta']['k1']
for mode, lab, c, ls, mk in (
        ('a', 'band A only (as benchmarked)', O.BLUE, '-', 's'),
        ('ab', 'A + B, one gain', O.GREEN, ':', 'D'),
        ('2g', 'A + B, per-band gain', O.GREEN, '-', 'D')):
    k = [K2G[n] if mode == '2g' else res[(mode, n)][1] for n in NS]
    a1.plot(NS, k, ls, color=c, lw=2.0, marker=mk, ms=5.5, mew=1.6,
            mec=O.WHITE, label=lab, zorder=4,
            alpha=.55 if mode == 'ab' else 1.0)
a1.set_xscale('log'); a1.set_yscale('log')
a1.set_xticks(NS); a1.set_yticks([300, 500, 800, 1300, 2000])
for A_ in (a1.get_xaxis(), a1.get_yaxis()):
    A_.set_major_formatter(T.FuncFormatter(lambda v, _: '%g' % v))
    A_.set_minor_locator(T.NullLocator())
a1.set_xlabel('positions measured  (n)')
a1.set_ylabel('fitted contact stiffness  (N/m)')
a1.set_ylim(260, 2400)
a1.legend(fontsize=9.3, labelcolor=O.INK2, loc='lower right')
O.clean(a1)
O.title(a1, 'Mode 2 + per-band gain: k₁ locks at ~1005 N/m from n = 3')

# ---- right: the wide-band spectrum at a held-out position, n = 4 (no GP)
n = 4
sel = res[('a', n)][0]
held = [i for i in range(len(x)) if i not in set(sel)]
iw = held[len(held) // 2]                      # a mid-lever held-out position
ref = dB(A).max()
a2.plot(f / 1e3, dB(A[iw]) - ref, '-', color=O.INK, lw=2.4, alpha=.30,
        label='measured')
a2.plot(f / 1e3, dB(res[('a', n)][2][iw]) - ref, '--', color=O.BLUE, lw=1.6,
        label=f'fit band A only  (k₁ = {res[("a",n)][1]:.0f} N/m)')
a2.plot(f / 1e3, dB(res[('ab', n)][2][iw]) - ref, ':', color=O.GREEN, lw=1.6,
        alpha=.6, label=f'A + B, one gain  (k₁ = {res[("ab",n)][1]:.0f} N/m)')
o2g = TB.rec2g(x, res[('a', n)][0], A[res[('a', n)][0]], f, gp=False)
a2.plot(f / 1e3, dB(o2g['A'][iw]) - ref, '-', color=O.GREEN, lw=1.7,
        label=f'A + B, per-band gain  (k₁ = {o2g["theta"]["k1"]:.0f} N/m)')
a2.annotate('band-A fit extrapolated: mode 2\nlands at 758 kHz — 144 kHz low,\n'
            'the real peak missed entirely',
            xy=(757, -3), xytext=(455, -13), fontsize=9.4, color=O.BLUE,
            ha='left', arrowprops=dict(arrowstyle='->', color=O.BLUE, lw=1.1))
a2.annotate('one gain: +19 kHz, +5 dB',
            xy=(921, -1.5), xytext=(700, -14), fontsize=9.2, color='#3F7A55',
            ha='left', arrowprops=dict(arrowstyle='->', color=O.GREEN,
                                       alpha=.6, lw=1.0))
a2.annotate('per-band gain: mode 2 at 902.5 kHz —\nwithin one frequency bin '
            'of the data',
            xy=(902, -7), xytext=(455, -20), fontsize=9.4, color='#004D21',
            ha='left', arrowprops=dict(arrowstyle='->', color=O.GREEN, lw=1.2))
a2.set_xlabel('frequency  (kHz)')
a2.set_ylabel('|Z|  (dB re max)')
a2.set_ylim(-72, 4)
a2.legend(fontsize=9.2, labelcolor=O.INK2, loc='lower left')
O.clean(a2)
O.title(a2, f'Held-out position x = {x[iw]:.1f} µm at n = {n} — no GP, '
            'physics only')
fig.savefig('fig/d_twoband.png', dpi=170)
print('wrote fig/d_twoband.png  pos', round(x[iw], 1))

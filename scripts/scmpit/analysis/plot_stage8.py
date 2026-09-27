import json

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, SEC, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9, 'axes.edgecolor': '#c3c2b7',
                     'axes.labelcolor': SEC, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.titlecolor': INK, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'legend.frameon': False,
                     'lines.linewidth': 1.6})

S = np.load('stage8_summary.npy', allow_pickle=True).item()
am, pm, walk, cal = S['morning'], S['closeout'], S['walk'], S['cal']
xw = np.array(walk['x_um']); fw = np.array(walk['f_closeout_Hz']); aw = np.array(walk['amp_closeout'])
xa = np.array(walk['x_morning_um']); fa = np.array(walk['f_morning_Hz']); aa = np.array(walk['amp_morning'])

fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.6))

# --- A: CR1 frequency along the beam -----------------------------------------
ax = axs[0, 0]
ax.plot(xa, fa / 1e3, 'o-', color=BLUE, ms=5, mec=SURF, mew=0.8, label='morning (stage 1, 12:52)')
ax.plot(xw, fw / 1e3, 's-', color=ORANGE, ms=5, mec=SURF, mew=0.8, label='close-out (stage 8, 22:19)')
ax.axhline(285.50, color=MUTED, ls='--', lw=1)
ax.text(xa[0], 285.9, 'pre-flight 285.50 kHz', fontsize=7.5, color=SEC)
ax.annotate('', xy=(195, 295.5), xytext=(195, 285.3),
            arrowprops=dict(arrowstyle='<->', color=SEC, lw=1.1))
ax.text(198, 290.2, '+10.46 kHz\n(+3.56 %)\nuniform', fontsize=8, color=INK, va='center')
ax.set_xlabel('distance from clamp (µm)'); ax.set_ylabel('CR1 frequency (kHz)')
ax.set_title('A.  CR1 shifted uniformly — the new contact state is stable', loc='left')
ax.legend(fontsize=8, loc='center left')
ax.set_ylim(284.0, 297.2)

# --- B: CR1 amplitude along the beam + ratio ---------------------------------
ax = axs[0, 1]
ax.plot(xa, aa * 1e3, 'o-', color=BLUE, ms=5, mec=SURF, mew=0.8, label='morning')
ax.plot(xw, aw * 1e3, 's-', color=ORANGE, ms=5, mec=SURF, mew=0.8, label='close-out')
ax.set_xlabel('distance from clamp (µm)'); ax.set_ylabel('CR1 peak |Z| (mV)')
ax.set_title('B.  The mode shape changed shape — the free-end node filled in', loc='left')
ax.legend(fontsize=8, loc='upper left')
ax2 = ax.twinx()
ax2.plot(xw, aw / aa, '^--', color=AQUA, ms=5, mec=SURF, mew=0.8, lw=1.2)
ax2.set_yscale('log'); ax2.set_ylabel('close-out / morning', color=AQUA)
ax2.tick_params(axis='y', colors=AQUA); ax2.grid(False)
ax2.axhline(1.0, color=AQUA, ls=':', lw=0.9)
ax2.annotate('free end ×10.4\n(stage 5 measured E there\nas ×7–9 stale)',
             xy=(232, 10.37), xytext=(55, 2.9), fontsize=7.5, color=SEC,
             arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))

# --- C: V_cpd across the day --------------------------------------------------
ax = axs[1, 0]
lab = ['stage 1\n12:52', 'stage 5\n20:06', 'stage 7\n21:16', 'stage 8\n22:19']
t = [12.87, 20.10, 21.27, 22.32]
v = [0.77, 1.33, 1.22, 1.191]
ax.plot(t, v, 'o-', color=BLUE, ms=7, mec=SURF, mew=1.0)
for ti, vi, li in zip(t, v, lab):
    ax.annotate(f'{li}\n{vi:+.2f} V', xy=(ti, vi), xytext=(0, 11), textcoords='offset points',
                ha='center', fontsize=7.5, color=INK)
ax.axvspan(18.6, 19.77, color=ORANGE, alpha=0.12, lw=0)
ax.text(19.18, 0.55, 'load ladder\n750–1500 nN', ha='center', fontsize=7.5, color=ORANGE)
ax.set_xlabel('time of day (instrument local)')
ax.set_ylabel('$V_{cpd}$ (V)')
ax.set_xlim(12.0, 23.4)
ax.set_ylim(0.45, 1.62)
ax.set_title('C.  Contact potential over the campaign — jumped, then settled', loc='left')

# --- D: bias dependence at position A, morning vs close-out ------------------
ax = axs[1, 1]
bias = np.array([-9, -6, -3, 0, 3, 6, 9], float)


def line(d, key_a, key_b):
    return np.real(d[key_a]) + np.real(d[key_b]) * bias


for d, c, nm, ls in ((am['cr1'], BLUE, 'morning', '-'), (pm['cr1'], ORANGE, 'close-out', '-')):
    a1 = abs(np.array([d['a1'] + d['b1'] * v for v in bias])) * 1e3
    a2 = abs(np.array([d['a2'] + d['b2'] * v for v in bias])) * 1e3
    ax.plot(bias, a1, ls, color=c, marker='o', ms=4, mec=SURF, mew=0.7, label=f'{nm}, domain 1')
    ax.plot(bias, a2, '--', color=c, marker='s', ms=4, mec=SURF, mew=0.7, label=f'{nm}, domain 2')
ax.set_xlabel('DC bias (V)'); ax.set_ylabel('CR1 |Z| from the fitted lines (mV)')
ax.set_title('D.  Same 14 conditions, 9.5 h apart — slopes hold, offsets move', loc='left')
ax.legend(fontsize=7.5, ncol=2, loc='upper center')

fig.suptitle('Stage 8 close-out calibration — end-of-campaign anchor against the morning baseline\n'
             'position A (154.9 µm), 500 nN, $V_{ac}$ = 1.000 V, identical conditions; '
             'free-air resonance also fell 63.80 → 58.93 kHz (−7.6 %)',
             fontsize=10, color=INK, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig('figN_stage8_closeout.png', dpi=150)
print('saved figN_stage8_closeout.png')

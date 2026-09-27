import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

d = np.load('load_dependence.npz')
loads, xt, amp, fpk, ph = d['loads'], d['xt'], d['amp'], d['fpk'], d['ph']
n_load, n_pos = amp.shape

INK, SEC, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9, 'axes.edgecolor': '#c3c2b7',
                     'axes.labelcolor': SEC, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.titlecolor': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'legend.frameon': False, 'lines.linewidth': 1.6})

pos_cmap = cm.get_cmap('Blues')
load_cmap = cm.get_cmap('Oranges')
pos_colors = [pos_cmap(0.35 + 0.55 * i / (n_pos - 1)) for i in range(n_pos)]
load_colors = [load_cmap(0.35 + 0.55 * i / (n_load - 1)) for i in range(n_load)]

# known artifact: position index 1 (77.7 um) at 50 nN -- tune locked onto free
# resonance (64 kHz), not CR1; amplitude in the CR1 window is a noise floor, not signal.
bad = {(0, 1)}   # (load_idx, pos_idx)

fig, axs = plt.subplots(2, 2, figsize=(11.5, 9))

# --- panel A: amplitude vs load, one line per position ---
ax = axs[0, 0]
for j in range(n_pos):
    y = amp[:, j].copy() * 1e3
    mask = np.array([(i, j) not in bad for i in range(n_load)])
    ax.plot(loads[mask], y[mask], 'o-', color=pos_colors[j], ms=4.5, mec=SURF, mew=0.8,
            label=f'{xt[j]:.0f} µm')
    if not mask.all():
        i0 = np.where(~mask)[0][0]
        ax.plot(loads[i0], y[i0], 'x', color=SEC, ms=6, mew=1.3)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('load (nN)'); ax.set_ylabel('CR1 peak |Z| (mV)')
ax.set_title('A.  Resonance amplitude grows with load at every position', loc='left')
ax.legend(title='position (base → tip)', fontsize=7, ncol=2, loc='upper left')

# --- panel B: CR1 frequency vs load, one line per position ---
ax = axs[0, 1]
for j in range(n_pos):
    y = fpk[:, j] / 1e3
    mask = np.array([(i, j) not in bad for i in range(n_load)])
    ax.plot(loads[mask], y[mask], 'o-', color=pos_colors[j], ms=4.5, mec=SURF, mew=0.8)
ax.set_xscale('log')
ax.set_xlabel('load (nN)'); ax.set_ylabel('CR1 peak frequency (kHz)')
ax.set_title('B.  A position-dependent split opens at 1000 nN, then closes again at 1500 nN', loc='left')
ax.axhline(285.5, color=MUTED, ls='--', lw=1)
ax.text(loads[0], 285.9, 'pre-flight CR1 (285.50 kHz)', fontsize=7.5, color=SEC)
ax.annotate('base (52 µm) up, tip (232 µm) down\n— 14.6 kHz spread mid-pass', xy=(1000, 288), xytext=(140, 292),
            fontsize=7.5, color=SEC, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))

# --- panel C: amplitude vs position, one line per load ---
ax = axs[1, 0]
for i in range(n_load):
    y = amp[i].copy() * 1e3
    mask = np.array([(i, j) not in bad for j in range(n_pos)])
    ax.plot(xt[mask], y[mask], 'o-', color=load_colors[i], ms=4.5, mec=SURF, mew=0.8,
            label=f'{loads[i]:.0f} nN')
ax.set_xlabel('distance from clamp (µm)'); ax.set_ylabel('CR1 peak |Z| (mV)')
ax.set_title('C.  Mode shape along the beam, at each load', loc='left')
ax.legend(title='load', fontsize=7.5, ncol=2, loc='upper left')

# --- panel D: amplitude normalized to the 50 nN pass, vs load, + Hertz n=1/3 guide ---
ax = axs[1, 1]
ref = amp[0].copy()
for j in range(n_pos):
    if (0, j) in bad:
        continue
    y = amp[:, j] / ref[j]
    ax.plot(loads, y, 'o-', color=pos_colors[j], ms=4.5, mec=SURF, mew=0.8)
guide = (loads / loads[0]) ** (1/3)
ax.plot(loads, guide, '--', color=SEC, lw=1.3, label=r'Hertz contact stiffness, $(F/F_0)^{1/3}$')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('load (nN)'); ax.set_ylabel('amplitude / amplitude(50 nN)')
ax.set_title('D.  Growth outpaces simple Hertzian contact-stiffness scaling', loc='left')
ax.legend(fontsize=8, loc='upper left')

fig.suptitle('Load ladder, spot 1, 0 V DC — CR1 amplitude and frequency vs. contact load, 8 positions × 7 loads (50–1500 nN)\n'
             'x marks a tune failure (CR1 suppressed near the pinned free-end at 50 nN, position 77.7 µm) excluded from the trend',
             fontsize=10, color=INK, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig('figG_load_dependence.png', dpi=150)
print('saved figG_load_dependence.png')

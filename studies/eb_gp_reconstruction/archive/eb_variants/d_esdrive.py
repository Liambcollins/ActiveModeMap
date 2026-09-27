"""EB spectra on the measured grid: contact (electromechanical) drive alone vs
with the distributed electrostatic term — the PFM drive model.

Uses the cached probe (out/antires_probe.pkl): EB at the two-band k1 = 1005,
z_mech (surface displacement through the contact spring) and z_es (distributed
electrostatic load on the lever) on the 101 calibrated positions x 412 band-A
bins. Mix: z = z_mech + alpha * s * z_es, alpha = 0.45 (in phase), s the
mode-1-peak normalisation between the two drives.
"""
import numpy as np, pickle, matplotlib.pyplot as plt
import ornl as O
from run_phys import prep

O.style()
d = pickle.load(open('out/antires_probe.pkl', 'rb'))
fm, xm, Zm, Ze = d['fm'], d['xm'], d['Zm'], d['Ze']
_, _, Am = prep()
Am = Am.T                                  # (nfreq, npos)
F = np.median(Am.min(axis=0))
dB = lambda M: 20 * np.log10(M + F)
s = np.abs(Zm).max() / np.abs(Ze).max()
ALPHA = 0.45
prof = Am.max(axis=1); ridge = prof >= .10 * prof.max()


def norm(M):
    # gain from RAW logs on the ridge (where data >> floor); adding F first
    # pins any model whose raw scale is below F to the floor
    g = np.exp(np.mean(np.log(Am[ridge]) - np.log(np.maximum(M[ridge], 1e-300))))
    return M * g


M0 = norm(np.abs(Zm))                      # contact drive only
M1 = norm(np.abs(Zm + ALPHA * s * Ze))     # + electrostatic
ME = norm(np.abs(ALPHA * s * Ze))          # the es term alone, same scale
ref = dB(Am).max()

fig, axes = plt.subplots(1, 4, figsize=(13.2, 4.1), sharey=False,
                         gridspec_kw={'width_ratios': [1, 1, 1, 1.15]})
fig.subplots_adjust(left=.052, right=.985, top=.87, bottom=.145, wspace=.24)

POS = [83, 92, 98]                         # x ~ 208, 217, 223 um
for ax, ip in zip(axes[:3], POS):
    ax.plot(fm / 1e3, dB(Am[:, ip]) - ref, '-', color=O.INK, lw=2.4,
            alpha=.30, label='measured (PFM)')
    ax.plot(fm / 1e3, dB(M0[:, ip]) - ref, '--', color=O.BLUE, lw=1.6,
            label='EB, contact drive only')
    ax.plot(fm / 1e3, dB(M1[:, ip]) - ref, '-', color=O.GREEN, lw=1.7,
            label='EB + electrostatic (α = 0.45)')
    ax.plot(fm / 1e3, dB(ME[:, ip]) - ref, ':', color=O.MAGENTA, lw=1.3,
            alpha=.8, label='the es term alone')
    ax.set_xlabel('frequency  (kHz)')
    ax.set_title(f'x = {xm[ip]:.1f} µm', loc='left', fontsize=10.5,
                 fontweight='600', color=O.INK)
    ax.set_ylim(-64, 3)
    O.clean(ax)
axes[0].set_ylabel('|response|  (dB re max)')
axes[0].legend(fontsize=8.2, labelcolor=O.INK2, loc='lower left')

# ---- right: the antiresonance branch vs position
ax = axes[3]


def fanti(Mat):
    out = np.full(Mat.shape[1], np.nan)
    for i in range(Mat.shape[1]):
        p = Mat[:, i]; j0 = int(np.argmax(p)); seg = p[j0 + 3:]
        if len(seg) > 5:
            k = int(np.argmin(seg))
            if 0 < k < len(seg) - 1:
                out[i] = fm[j0 + 3 + k]
    return out


for M, c, ls, lab, lw, al in ((Am, O.INK, '-', 'measured', 2.6, .30),
                              (M0, O.BLUE, '--', 'contact only', 1.7, 1.),
                              (M1, O.GREEN, '-', '+ es, α = 0.45', 1.9, 1.)):
    fa = fanti(M); mm = np.isfinite(fa) & (xm > 198) & (xm < 224.5)
    ax.plot(xm[mm], fa[mm] / 1e3, ls, color=c, lw=lw, alpha=al, label=lab)
ax.set_xlabel('laser position from clamp  (µm)')
ax.set_ylabel('antiresonance frequency  (kHz)')
ax.legend(fontsize=8.8, labelcolor=O.INK2, loc='upper right')
O.clean(ax)
ax.set_title('the branch, all positions', loc='left', fontsize=10.5,
             fontweight='600', color=O.INK)

fig.suptitle('PFM on PPLN: the null is where electromechanical and '
             'electrostatic responses cancel — EB at k₁ = 1005 N/m',
             x=.052, y=.965, ha='left', fontsize=12.5, fontweight='600',
             color=O.INK)
fig.savefig('fig/d_esdrive.png', dpi=170)
print('wrote fig/d_esdrive.png  positions', [round(xm[i], 1) for i in POS])

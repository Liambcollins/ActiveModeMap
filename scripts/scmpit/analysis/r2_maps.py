r"""R2 quantitative d33 maps, and the R1-vs-R2 null comparison.

Same conversion as R1's figK: every CR1 panel is amplitude / (V_ac x the in-situ
enhancement measured for THAT domain), applied piecewise through the domain
masks; quasi-static panels need only / V_ac. All d33 panels share one colour
scale so they are comparable by eye, which was the whole point of the figK
correction. Domain masks are anchored to image geometry (see r2_images.py).
"""
import json

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from igor2.binarywave import load as ibw_load
from sklearn.cluster import KMeans

UP = '/mnt/user-data/uploads/ActiveModeMap'
AMP_CH, PHASE_CH = 1, 3
INK, SEC, MUTED, SURF = '#0b0b0b', '#52514e', '#898781', '#fcfcfb'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9, 'axes.titlecolor': INK,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF})

R1_ROOT, R2_ROOT = f'{UP}/DomainsB_SCMPIT', f'{UP}/DomainsB_SCMPIT_R2'
R1F = {'A_qs': ('QImg0573.ibw', 0.0, 'qs', 1.0), 'A_cr0': ('QImg0574.ibw', 0.0, 'cr1', 1.0),
       'A_null': ('QImg0575.ibw', 0.77, 'cr1', 1.0), 'B_qs': ('QImg0578.ibw', 0.0, 'qs', 1.0),
       'B_cr0': ('QImg0579.ibw', 0.0, 'cr1', 1.0), 'B_null': ('QImg0580.ibw', 0.77, 'cr1', 1.0)}


def r2_frames():
    j = json.load(open(f'{R2_ROOT}/image_cr1_quant.json'))
    by = {im['name']: im for im in j['images']}
    g = lambda n: (by[n]['path'].split('\\')[-1], by[n]['bias_V'], by[n]['kind'], by[n]['drive_V'])
    return {'A_qs': g('A_qs20k_0V'), 'A_cr0': g('A_cr1_0V'), 'A_half': g('A_cr1_Vhalf'),
            'A_null': g('A_cr1_Vcpd'), 'B_qs': g('B_qs20k_0V'), 'B_cr0': g('B_cr1_0V'),
            'B_null': g('B_cr1_Vcpd')}, j['v_cpd_V']


def read(root, fn):
    d = np.asarray(ibw_load(f'{root}/{fn}')['wave']['wData'], float)
    return d[:, :, AMP_CH], d[:, :, PHASE_CH]


def masks(phase):
    v = np.column_stack([np.cos(np.radians(phase.ravel())), np.sin(np.radians(phase.ravel()))])
    lab = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(v).reshape(phase.shape)
    m0, m1 = lab == 0, lab == 1
    cols = np.arange(phase.shape[1])[None, :] * np.ones_like(phase)
    return (m0, m1) if cols[m0].mean() < cols[m1].mean() else (m1, m0)


def d33_map(amp, kind, vac, E1, E2, m1, m2):
    out = np.full_like(amp, np.nan)
    g1 = vac * (E1 if kind == 'cr1' else 1.0)
    g2 = vac * (E2 if kind == 'cr1' else 1.0)
    out[m1] = amp[m1] / g1 * 1e12
    out[m2] = amp[m2] / g2 * 1e12
    return out


def load_set(root, frames):
    """Returns per-position dict with masks, in-situ E, and each frame's maps."""
    out = {}
    for pos in ('A', 'B'):
        keys = [k for k in frames if k.startswith(pos + '_')]
        if not keys:
            continue
        _, rph = read(root, frames[f'{pos}_cr0'][0])
        m1, m2 = masks(rph)
        aq, _ = read(root, frames[f'{pos}_qs'][0])
        ac, _ = read(root, frames[f'{pos}_cr0'][0])
        E1 = float(np.median(ac[m1]) / np.median(aq[m1]))
        E2 = float(np.median(ac[m2]) / np.median(aq[m2]))
        ent = dict(m1=m1, m2=m2, E1=E1, E2=E2, frames={})
        for k in keys:
            fn, vdc, kind, vac = frames[k]
            amp, ph = read(root, fn)
            ent['frames'][k] = dict(d33=d33_map(amp, kind, vac, E1, E2, m1, m2), ph=ph,
                                    bias=vdc, kind=kind, vac=vac,
                                    md1=float(np.nanmedian(d33_map(amp, kind, vac, E1, E2, m1, m2)[m1])),
                                    md2=float(np.nanmedian(d33_map(amp, kind, vac, E1, E2, m1, m2)[m2])),
                                    ratio=float(np.median(amp[m1]) / np.median(amp[m2])))
        out[pos] = ent
    return out


def show(ax, img, vmin, vmax, cmap='magma'):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower',
                   extent=[0, 6, 0, 6], interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor('#c3c2b7')
    return im


R2F, VC2 = r2_frames()
S2 = load_set(R2_ROOT, R2F)
S1 = load_set(R1_ROOT, R1F)
VC1 = 0.77
VMIN, VMAX = 0, 15

# ======================================================= figP: R2 map set ====
fig, axs = plt.subplots(3, 4, figsize=(13.2, 10.2))
panels = [
    ('A', 'A_qs', 'quasi-static 20 kHz, 0 V'), ('A', 'A_cr0', 'CR1, 0 V'),
    ('A', 'A_half', f'CR1, +{VC2/2:.3f} V (half)'), ('A', 'A_null', f'CR1, +{VC2:.3f} V (null)'),
    ('B', 'B_qs', 'quasi-static 20 kHz, 0 V'), ('B', 'B_cr0', 'CR1, 0 V'),
    (None, None, None), ('B', 'B_null', f'CR1, +{VC2:.3f} V (null)'),
]
for ax, (pos, key, title) in zip(axs[:2].ravel(), panels):
    if pos is None:
        ax.axis('off'); continue
    f = S2[pos]['frames'][key]
    im = show(ax, f['d33'], VMIN, VMAX)
    ax.set_title(f'{pos} — {title}', fontsize=8.5, loc='left')
    ax.text(0.03, 0.04, f'D1 {f["md1"]:.2f}   D2 {f["md2"]:.2f} pm/V\nratio {f["ratio"]:.3f}',
            transform=ax.transAxes, fontsize=7.5, color='white', va='bottom',
            bbox=dict(facecolor='#00000066', edgecolor='none', pad=2.2))
axs[0, 0].set_ylabel(f'position A (148.8 µm)\nin-situ E = {S2["A"]["E1"]:.0f} / {S2["A"]["E2"]:.0f}',
                     fontsize=8.5, color=SEC)
axs[1, 0].set_ylabel(f'position B (free end)\nin-situ E = {S2["B"]["E1"]:.0f} / {S2["B"]["E2"]:.0f}',
                     fontsize=8.5, color=SEC)

for ax, (pos, key, lab) in zip(axs[2], [('A', 'A_cr0', 'A, CR1 0 V'), ('A', 'A_null', 'A, CR1 null'),
                                        ('B', 'B_cr0', 'B, CR1 0 V'), ('B', 'B_null', 'B, CR1 null')]):
    f = S2[pos]['frames'][key]
    imp = show(ax, f['ph'], -180, 180, cmap='twilight')
    ax.set_title(f'phase — {lab}', fontsize=8.5, loc='left')
axs[2, 0].set_ylabel('phase (deg)', fontsize=8.5, color=SEC)

fig.subplots_adjust(right=0.9)
cb = fig.colorbar(im, ax=axs[:2].ravel().tolist(), fraction=0.02, pad=0.015)
cb.set_label('$d_{33}$ (pm/V)', color=SEC)
cb2 = fig.colorbar(imp, ax=axs[2].tolist(), fraction=0.02, pad=0.015, ticks=[-180, -90, 0, 90, 180])
cb2.set_label('phase (deg)', color=SEC)
fig.suptitle('R2 quantitative PFM maps — every panel is amplitude ÷ ($V_{ac}$ × in-situ enhancement), '
             'one common 0–15 pm/V scale\n'
             f'6 µm frame, 256², 500 nN, $V_{{ac}}$ = 1 V; null = +{VC2:.3f} V measured from R2\'s own data',
             fontsize=10.5, color=INK, x=0.01, ha='left')
fig.savefig('/home/claude/scmpit_analysis/figP_r2_d33_maps.png', dpi=150,
            bbox_inches='tight', facecolor=SURF)
print('saved figP_r2_d33_maps.png')

# ================================== figQ: R1 vs R2, does the null equalise? ==
fig, axs = plt.subplots(2, 3, figsize=(10.6, 7.4))
rows = [('R1', S1, VC1, 'imported from the morning survey'),
        ('R2', S2, VC2, "measured from R2's own data")]
for r, (tag, S, vc, note) in enumerate(rows):
    for c, (key, title) in enumerate([('A_qs', 'quasi-static, 0 V'), ('A_cr0', 'CR1, 0 V'),
                                      ('A_null', f'CR1, +{vc:.3f} V')]):
        f = S['A']['frames'][key]
        im = show(axs[r, c], f['d33'], VMIN, VMAX)
        axs[r, c].set_title(f'{tag} — {title}', fontsize=9, loc='left')
        ok = abs(f['ratio'] - 1) < 0.05
        axs[r, c].text(0.03, 0.04,
                       f'D1 {f["md1"]:.2f}   D2 {f["md2"]:.2f} pm/V\nratio {f["ratio"]:.3f}'
                       + ('  ✓ equalised' if (c == 2 and ok) else
                          ('  ✗ not equalised' if c == 2 else '')),
                       transform=axs[r, c].transAxes, fontsize=7.8, color='white', va='bottom',
                       bbox=dict(facecolor='#00000066', edgecolor='none', pad=2.2))
    axs[r, 0].set_ylabel(f'{tag}\nnull {note}', fontsize=8.5, color=SEC)
cb = fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.022, pad=0.015)
cb.set_label('$d_{33}$ (pm/V)', color=SEC)
fig.suptitle('Does the bias null actually equalise the domains? Position A, identical frame and scale\n'
             'R1 used a $V_{cpd}$ imported from four hours earlier; R2 measured its own an hour before imaging',
             fontsize=10.5, color=INK, x=0.01, ha='left')
fig.savefig('/home/claude/scmpit_analysis/figQ_null_comparison.png', dpi=150,
            bbox_inches='tight', facecolor=SURF)
print('saved figQ_null_comparison.png')

for tag, S in (('R1', S1), ('R2', S2)):
    for pos in S:
        for k, f in S[pos]['frames'].items():
            print(f'  {tag} {k:8s} bias {f["bias"]:+6.3f}  d33 {f["md1"]:6.2f} / {f["md2"]:6.2f}  '
                  f'ratio {f["ratio"]:.3f}')

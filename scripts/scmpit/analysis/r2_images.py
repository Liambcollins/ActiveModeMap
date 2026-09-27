r"""Quantitative imaging, R1 vs R2: in-situ enhancement, d33, domain ratio, noise/SNR.

One code path for both campaigns. Domain mask from 2-means on (cos, sin) of the
phase of each position's highest-SNR CR1 frame, reused for every frame at that
position (all frames share one XY raster). Domain LABELS are then fixed by phase
sign, not by brightness -- R1 stage 7 showed the amplitude ranking can inflip
while the phase identity, set by the fixed polarisation, does not.

AmplitudeRetrace in these .ibw files is already in calibrated metres; no
InvOLS/32 factor (unlike the raw spectroscopy checkpoints).
"""
import json

import numpy as np
from igor2.binarywave import load as ibw_load
from scipy.ndimage import binary_erosion
from sklearn.cluster import KMeans

UP = '/mnt/user-data/uploads/ActiveModeMap'
AMP_CH, PHASE_CH = 1, 3

R1 = dict(root=f'{UP}/DomainsB_SCMPIT', frames=[
    ('A_qs20k_0V', 'QImg0573.ibw', 154.9, 0.0, 'qs', 1.0),
    ('A_cr1_0V', 'QImg0574.ibw', 154.9, 0.0, 'cr1', 1.0),
    ('A_cr1_Vcpd', 'QImg0575.ibw', 154.9, 0.77, 'cr1', 1.0),
    ('A_cr1_0V_repeat', 'QImg0576.ibw', 154.9, 0.0, 'cr1', 1.0),
    ('B_qs20k_0V', 'QImg0578.ibw', 232.0, 0.0, 'qs', 1.0),
    ('B_cr1_0V', 'QImg0579.ibw', 232.0, 0.0, 'cr1', 1.0),
    ('B_cr1_Vcpd', 'QImg0580.ibw', 232.0, 0.77, 'cr1', 1.0),
    ('B_qs20k_0V_2Vac', 'QImg0581.ibw', 232.0, 0.0, 'qs', 2.0),
], lowac=[('A_qs20k_0V_30mV', 'LoAC0575.ibw', 0.0, 'qs'),
          ('A_cr1_0V_30mV', 'LoAC0576.ibw', 0.0, 'cr1'),
          ('A_cr1_Vcpdtrue_30mV', 'LoAC0577.ibw', 1.33, 'cr1')])


def r2_spec():
    j = json.load(open(f'{UP}/DomainsB_SCMPIT_R2/image_cr1_quant.json'))
    fr = [(im['name'], im['path'].split('\\')[-1], im['stage_x_um'], im['bias_V'],
           im['kind'], im['drive_V']) for im in j['images']]
    k = json.load(open(f'{UP}/DomainsB_SCMPIT_R2/lowac_cr1_A.json'))
    lo = [(im['name'], im['path'].split('\\')[-1], im['bias_V'],
           'qs' if 'qs' in im['name'] else 'cr1') for im in k['images']]
    return dict(root=f'{UP}/DomainsB_SCMPIT_R2', frames=fr, lowac=lo, vcpd=j['v_cpd_V'])


def read(root, fn):
    w = ibw_load(f'{root}/{fn}')['wave']
    d = np.asarray(w['wData'], float)
    return d[:, :, AMP_CH], d[:, :, PHASE_CH]


def cmean(phase_deg):
    """Circular mean in degrees. A plain median of wrapped angles is meaningless:
    a domain sitting near +-180 averages to ~0 and the separation collapses."""
    z = np.exp(1j * np.radians(np.asarray(phase_deg, float).ravel()))
    return float(np.degrees(np.angle(z.mean())))


def masks_from(phase):
    """2-means on the phase unit vector, labelled by GEOMETRY.

    Both campaigns imaged the identical 6 um frame at identical offsets, so the
    same physical domain occupies the same pixels. Labelling by phase sign or by
    brightness is not stable across tip states -- R1 stage 7 showed the amplitude
    ranking inverting, and the absolute phase offset moves with drive frequency
    and tip state. The pixel geometry does not. Domain 1 := the cluster whose
    centroid sits at the smaller mean column index.
    """
    v = np.column_stack([np.cos(np.radians(phase.ravel())), np.sin(np.radians(phase.ravel()))])
    lab = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(v).reshape(phase.shape)
    m0, m1 = lab == 0, lab == 1
    cols = np.arange(phase.shape[1])[None, :] * np.ones_like(phase)
    return (m0, m1) if cols[m0].mean() < cols[m1].mean() else (m1, m0)


def stats(amp, phase, m1, m2):
    e1, e2 = binary_erosion(m1, iterations=4), binary_erosion(m2, iterations=4)
    if e1.sum() < 50: e1 = m1
    if e2.sum() < 50: e2 = m2
    a1, a2 = float(np.median(amp[m1])), float(np.median(amp[m2]))
    n1, n2 = float(np.std(amp[e1])), float(np.std(amp[e2]))
    p1, p2 = cmean(phase[m1]), cmean(phase[m2])
    sep = abs(p1 - p2)
    sep = 360 - sep if sep > 180 else sep
    noise = float(np.sqrt((n1 ** 2 + n2 ** 2) / 2))
    return dict(a1=a1, a2=a2, ratio=a1 / a2, p1=p1, p2=p2, sep=sep,
                noise=noise, contrast=abs(a1 - a2),
                snr=abs(a1 - a2) / noise if noise else float('nan'))


out = {}
for tag, spec in (('R1', R1), ('R2', r2_spec())):
    root = spec['root']
    print(f'\n{"="*78}\n{tag}\n{"="*78}')
    ent = {'frames': {}, 'lowac': {}}
    by_pos = {}
    for name, fn, x, vdc, kind, vac in spec['frames']:
        by_pos.setdefault(x, []).append((name, fn, vdc, kind, vac))
    for x, items in sorted(by_pos.items()):
        ref = next((i for i in items if i[3] == 'cr1' and i[2] == 0.0), items[0])
        _, rph = read(root, ref[1])
        m1, m2 = masks_from(rph)
        print(f'  position x={x} um   mask {m1.sum()} / {m2.sum()} px  (ref {ref[0]})')
        E = {}
        qs0 = next((i for i in items if i[3] == 'qs' and i[2] == 0.0 and i[4] == 1.0), None)
        cr0 = next((i for i in items if i[3] == 'cr1' and i[2] == 0.0), None)
        if qs0 and cr0:
            aq, _ = read(root, qs0[1]); ac, _ = read(root, cr0[1])
            sq = stats(aq, rph, m1, m2); sc = stats(ac, rph, m1, m2)
            E = {'E1': sc['a1'] / sq['a1'], 'E2': sc['a2'] / sq['a2']}
            print(f'    in-situ enhancement  E1 {E["E1"]:7.1f}   E2 {E["E2"]:7.1f}')
        for name, fn, vdc, kind, vac in items:
            amp, ph = read(root, fn)
            s = stats(amp, ph, m1, m2)
            gain1 = vac * (E.get('E1', 1.0) if kind == 'cr1' else 1.0)
            gain2 = vac * (E.get('E2', 1.0) if kind == 'cr1' else 1.0)
            s.update(d33_1=s['a1'] / gain1 * 1e12, d33_2=s['a2'] / gain2 * 1e12,
                     noise_ref=s['noise'] / (vac * (E.get('E1', 1.0) if kind == 'cr1' else 1.0)) * 1e12,
                     x=x, bias=vdc, kind=kind, vac=vac, **{'E1': E.get('E1'), 'E2': E.get('E2')})
            ent['frames'][name] = s
            print(f'    {name:20s} {vdc:+6.3f}V {vac:4.1f}Vac  '
                  f'd33 {s["d33_1"]:7.2f} / {s["d33_2"]:7.2f} pm/V  ratio {s["ratio"]:6.3f}  '
                  f'sep {s["sep"]:6.1f}d  SNR {s["snr"]:6.2f}  noise_ref {s["noise_ref"]:6.3f} pm/V')
    # ---- low-AC frames, converted with the trustworthy 1 V in-situ E at A ----
    EA = ent['frames'].get('A_cr1_0V', {})
    E1, E2 = EA.get('E1'), EA.get('E2')
    lo = spec['lowac']
    if lo and E1:
        _, rph = read(root, next(i[1] for i in lo if i[3] == 'cr1'))
        m1, m2 = masks_from(rph)
        print(f'  low-AC (30 mV) at A, converted with the 1 V in-situ E = {E1:.1f}/{E2:.1f}')
        for name, fn, vdc, kind in lo:
            amp, ph = read(root, fn)
            s = stats(amp, ph, m1, m2)
            g1 = 0.03 * (E1 if kind == 'cr1' else 1.0)
            g2 = 0.03 * (E2 if kind == 'cr1' else 1.0)
            s.update(d33_1=s['a1'] / g1 * 1e12, d33_2=s['a2'] / g2 * 1e12,
                     noise_ref=s['noise'] / g1 * 1e12, bias=vdc, kind=kind)
            ent['lowac'][name] = s
            print(f'    {name:24s} {vdc:+6.3f}V  d33 {s["d33_1"]:7.2f} / {s["d33_2"]:7.2f} pm/V  '
                  f'ratio {s["ratio"]:6.3f}  sep {s["sep"]:6.1f}d  SNR {s["snr"]:6.2f}  '
                  f'noise_ref {s["noise_ref"]:6.3f} pm/V')
    out[tag] = ent

json.dump(out, open('/home/claude/scmpit_analysis/r2_images.json', 'w'), indent=1, default=float)
print('\nsaved r2_images.json')

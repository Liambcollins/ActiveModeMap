r"""Stage-1 analysis: V_cpd and the channel ratio |b|/|P| per mode, SCM-PIT at 500 nN.

Method exactly as tipbias-survey-analysis-2026-09-18:
  Z_s(V) = P_s + b (V - V_cpd), P flips sign between domains, b does not.
  Complex least squares a_s + b_s*V per spot, per position, per frequency, then
      V_cpd = -(a1 + a2) / 2b,   P = (a1 - a2)/2,   b = (b1 + b2)/2.
Reported per mode over a narrow band around each measured contact resonance.

Read-only: runs off the checkpoint, never touches Igor.
"""
import json
import os
import sys

import numpy as np

FOLDER = sys.argv[1] if len(sys.argv) > 1 else '.'
CKPT = os.path.join(FOLDER, 'domains_bias_checkpoint_500nN_scmpit.npz')
PRE = os.path.join(FOLDER, 'preflight_result.json')
OUT = os.path.join(FOLDER, 'analysis', 'stage1_vcpd_channels.json')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

d = np.load(CKPT, allow_pickle=True)
F = d['freq_Hz']
Z = d['Z']                       # (n_cond, n_pos, n_freq)
X = d['x_um']
conds = json.loads(str(d['conditions']))
bias = np.array([float(c['bias_V']) for c in conds])
spot = np.array([int(c.get('spot') or 1) for c in conds])
print(f"phase convention: {d['phase_convention']}")

P_ = json.load(open(PRE))
MODES = {'CR1': P_['cr1_Hz'], 'CR2': P_['cr2_Hz'], 'CR3': P_['cr3_Hz']}
HALF = {'CR1': 4e3, 'CR2': 8e3, 'CR3': 25e3}

order = np.argsort(X)[::-1]                       # free end first, for printing
print(f'checkpoint: {Z.shape[0]} conditions x {Z.shape[1]} positions x {Z.shape[2]} freqs')
print(f'positions: {np.round(X[order], 1).tolist()}')
print(f'biases: {sorted(set(bias.tolist()))}   spots: {sorted(set(spot.tolist()))}')
print(f'non-finite spectra: {int((~np.isfinite(Z)).any(axis=2).sum())}')

results = {}
for name, f0 in MODES.items():
    m = np.abs(F - f0) <= HALF[name]
    if not m.any():
        continue
    # peak bin per position, spot 1, 0 V -- the mode may shift along x
    fits = {}
    for s in (1, 2):
        idx = np.where(spot == s)[0]
        V = bias[idx]
        A = np.stack([np.ones_like(V), V], axis=1)
        Zs = Z[idx][:, :, m]                                   # (nV, npos, nb)
        nV, npos, nb = Zs.shape
        coef, *_ = np.linalg.lstsq(A, Zs.reshape(nV, -1), rcond=None)
        fit = (A @ coef).reshape(nV, npos, nb)
        resid = np.sqrt((np.abs(Zs - fit) ** 2).mean(axis=0))
        scale = np.abs(Zs).mean(axis=0)
        fits[s] = dict(a=coef[0].reshape(npos, nb), b=coef[1].reshape(npos, nb),
                       rel_resid=resid / (scale + 1e-30))
    a1, b1 = fits[1]['a'], fits[1]['b']
    a2, b2 = fits[2]['a'], fits[2]['b']
    b = (b1 + b2) / 2.0
    Pz = (a1 - a2) / 2.0
    Vcpd = -((a1 + a2) / (2.0 * b)).real                       # (npos, nb)

    # weight by |b| inside the band: the estimate is meaningless where b ~ 0
    w = np.abs(b)
    wsum = w.sum(axis=1) + 1e-30
    vcpd_pos = (Vcpd * w).sum(axis=1) / wsum
    ratio_pos = (np.abs(b) * w).sum(axis=1) / ((np.abs(Pz) * w).sum(axis=1) + 1e-30)
    bratio = np.abs(b2).mean(axis=1) / (np.abs(b1).mean(axis=1) + 1e-30)
    # domain flip angle at V_cpd: arg(Z1/Z2) there == arg(P / -P) == 180 deg
    flip = np.angle((a1 + vcpd_pos[:, None] * b1) / (a2 + vcpd_pos[:, None] * b2), deg=True)
    flip_pos = np.angle((np.exp(1j * np.deg2rad(flip)) * w).sum(axis=1), deg=True)
    rr = np.maximum(fits[1]['rel_resid'], fits[2]['rel_resid']).mean(axis=1)

    results[name] = dict(
        f0_Hz=float(f0), band_Hz=[float(F[m][0]), float(F[m][-1])],
        x_um=[float(v) for v in X[order]],
        v_cpd_V=[float(v) for v in vcpd_pos[order]],
        ratio_b_over_P=[float(v) for v in ratio_pos[order]],
        b2_over_b1=[float(v) for v in bratio[order]],
        flip_deg=[float(v) for v in flip_pos[order]],
        rel_resid=[float(v) for v in rr[order]],
        v_cpd_median=float(np.median(vcpd_pos)), v_cpd_std=float(np.std(vcpd_pos)),
        ratio_median=float(np.median(ratio_pos)),
    )

    print(f'\n=== {name}  {f0 / 1e3:.2f} kHz  (band {F[m][0] / 1e3:.1f}-{F[m][-1] / 1e3:.1f} kHz) ===')
    print('    x_um    V_cpd    |b|/|P|   |b2/b1|   flip     resid')
    for j, i in enumerate(order):
        print(f'  {X[i]:7.1f}  {vcpd_pos[i]:+7.3f}   {ratio_pos[i]:7.3f}   {bratio[i]:6.3f}  '
              f'{flip_pos[i]:+7.1f}   {100 * rr[i]:5.1f} %')
    print(f'  V_cpd = {np.median(vcpd_pos):+.3f} +/- {np.std(vcpd_pos):.3f} V   '
          f'|b|/|P| median = {np.median(ratio_pos):.3f}')

PPPCONTAU = {'CR1': 1.64, 'CR2': 0.24, 'CR3': 0.12}
print('\n' + '=' * 72)
print('THE PREDICTION: |b|/|P| should be markedly below every PPP-CONTAu value')
print('  mode   PPP-CONTAu 15 nN   SCM-PIT 500 nN   ratio')
for name in results:
    r = results[name]['ratio_median']
    p = PPPCONTAU.get(name)
    print(f'  {name}      {p:12.3f}   {r:14.3f}   {r / p:6.2f}x' if p else
          f'  {name}      {"-":>12}   {r:14.3f}')
print('=' * 72)
vs = [results[n]['v_cpd_median'] for n in results]
print(f'V_cpd across modes: {[round(v, 3) for v in vs]}  '
      f'(PPP-CONTAu: +1.51 V, mode-independent)')

json.dump(results, open(OUT, 'w'), indent=1)
print(f'\nwritten -> {OUT}')

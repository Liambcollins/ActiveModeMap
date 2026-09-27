r"""Wideband low-rank reconstruction, R1 vs R2, one estimator for both.

The published R1 scores came from the repo's own `activemodemap.lowrank`. That
package needs scipy, which the device VM does not have, so this is an
independent re-implementation of the same idea: SVD the 8-position complex
matrix, keep r spatial/spectral factors, interpolate the spatial factors in x
with a cubic spline, and evaluate at the never-visited validation positions.

Absolute numbers may therefore differ slightly from the published R1 table. That
is fine and is stated in the write-up: the SAME estimator is applied to both
campaigns, so the R1-vs-R2 comparison is internally valid. As a check, the R1
column should land near the published 0.412 / 0.249 / 0.256 for ranks 4/5/6.
"""
import json

import numpy as np
from scipy.interpolate import CubicSpline

UP = '/mnt/user-data/uploads/ActiveModeMap'
SETS = {
    'R1': (f'{UP}/DomainsB_SCMPIT/20_wideband_fast/wideband_fast_8pos_500nN_0V_checkpoint.npz',
           f'{UP}/DomainsB_SCMPIT/21_wideband_validation/wideband_valid_7pos_500nN_0V_checkpoint.npz'),
    'R2': (f'{UP}/DomainsB_SCMPIT_R2/wideband_fast_8pos_500nN_0V_checkpoint.npz',
           f'{UP}/DomainsB_SCMPIT_R2/wideband_valid_7pos_500nN_0V_checkpoint.npz'),
}
BAND = (100e3, 1.9e6)                      # the band R1 scored on
CR = dict(cr1=(255e3, 330e3), cr2=(840e3, 960e3), cr3=(1650e3, 1900e3))
RANKS = (4, 5, 6)


def load(path):
    d = np.load(path, allow_pickle=True)
    F = np.asarray(d['freq_Hz'], float)
    Z = np.asarray(d['Z'])
    Z = Z[0] if Z.ndim == 3 else Z
    x = np.asarray(d['x_um'], float)
    o = np.argsort(x)
    return F, x[o], Z[o]


def reconstruct(x_tr, Z_tr, x_te, r):
    """Rank-r reconstruction of the complex field at x_te."""
    U, s, Vh = np.linalg.svd(Z_tr, full_matrices=False)
    A = U[:, :r] * s[:r]                       # (n_train, r) spatial weights
    out = np.zeros((x_te.size, Z_tr.shape[1]), complex)
    for k in range(r):
        re = CubicSpline(x_tr, A[:, k].real)(x_te)
        im = CubicSpline(x_tr, A[:, k].imag)(x_te)
        out += np.outer(re + 1j * im, Vh[k])
    return out


def score(truth, pred, F, m):
    t, p = truth[:, m], pred[:, m]
    nrmse = float(np.linalg.norm(t - p) / np.linalg.norm(t))
    at, ap = np.abs(t), np.abs(p)
    ok = at > 0
    db = 20 * np.log10(np.clip(ap[ok], 1e-30, None) / at[ok])
    ph = np.degrees(np.angle(p[ok] / t[ok]))
    return dict(nrmse=nrmse, med_abs_db=float(np.median(np.abs(db))),
                frac_within_3db=float(np.mean(np.abs(db) < 3.0)),
                phase_mae=float(np.mean(np.abs(ph))))


results = {}
for tag, (ftr, fte) in SETS.items():
    F, x_tr, Z_tr = load(ftr)
    F2, x_te, Z_te = load(fte)
    assert np.allclose(F, F2), f'{tag}: train and validation frequency grids differ'
    m = (F >= BAND[0]) & (F <= BAND[1])
    print(f'\n{tag}: train {x_tr.size} pos {x_tr.min():.1f}-{x_tr.max():.1f} um, '
          f'test {x_te.size} pos {x_te.min():.1f}-{x_te.max():.1f} um, {m.sum()} freqs in band')
    # keep the test set strictly inside the training hull (no extrapolation)
    inside = (x_te >= x_tr.min()) & (x_te <= x_tr.max())
    if not inside.all():
        print(f'   dropping {(~inside).sum()} test position(s) outside the training hull: '
              f'{x_te[~inside].tolist()}')
        x_te, Z_te = x_te[inside], Z_te[inside]
    ent = {}
    print(f'   {"rank":>4} {"NRMSE":>7} {"med|dB|":>8} {"<3dB":>6} {"phase MAE":>10}   CR1/CR2/CR3 NRMSE')
    for r in RANKS:
        pred = reconstruct(x_tr, Z_tr, x_te, r)
        s = score(Z_te, pred, F, m)
        per = {}
        for nm, b in CR.items():
            mb = (F >= b[0]) & (F <= b[1])
            per[nm] = score(Z_te, pred, F, mb)['nrmse']
        ent[r] = dict(**s, per_mode=per)
        print(f'   {r:4d} {s["nrmse"]:7.3f} {s["med_abs_db"]:8.2f} {100*s["frac_within_3db"]:5.0f}% '
              f'{s["phase_mae"]:9.1f}deg   {per["cr1"]:.3f} / {per["cr2"]:.3f} / {per["cr3"]:.3f}')
    results[tag] = dict(x_train=x_tr.tolist(), x_test=x_te.tolist(), scores=ent)

json.dump(results, open('/home/claude/scmpit_analysis/r2_wideband.json', 'w'), indent=1)
print('\nsaved r2_wideband.json')
print('\nR1 published (repo lowrank, same data): rank4 0.412 / rank5 0.249 / rank6 0.256')

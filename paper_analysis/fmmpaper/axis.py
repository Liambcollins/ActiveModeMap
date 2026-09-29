"""Position-axis calibration.

SCM-PIT-A Dense_Grid_B was commanded at 0.5 um, below the stage dead-band, so the
sweep under-travelled. The draft (Sec. S1) calibrates the axis with the InvOLS
ruler: the first ``n_anchor`` log rows are large-move anchor positions that are
unaffected by the dead-band; InvOLS is monotonic along the lever, so inverting
InvOLS(x) at the anchors maps every dense position onto the true axis.
Ported from Manuscript_FMM/gt.py (axis C) -- same algorithm, same numbers.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import uniform_filter1d


def invols_ruler_axis(log, x_dense, n_anchor: int = 10, x_max: float = 225.0) -> dict:
    anchors = log[log.i < n_anchor]
    dense = log[log.i >= n_anchor].sort_values("position_x_um")
    if len(dense) != len(x_dense) or not np.allclose(dense.position_x_um.values, x_dense):
        return dict(axis_note="log does not match dense positions; no axis correction")
    iv = dense.invols_m_per_V.values
    xa = anchors.position_x_um.values
    iva = anchors.invols_m_per_V.values
    o = np.argsort(iva)
    ruler = PchipInterpolator(np.log(iva[o]), xa[o], extrapolate=True)   # log InvOLS -> x
    lds = uniform_filter1d(np.log(iv), 5, mode="nearest")
    lds = np.maximum.accumulate(lds[::-1])[::-1]                          # non-increasing
    xt = np.clip(ruler(lds), xa.min(), x_max)
    xt = np.maximum.accumulate(xt)
    slope, off = np.polyfit(x_dense, xt, 1)
    return dict(x_true_um=xt, invols=iv, axis_fit=(float(slope), float(off)),
                axis_rms_um=float(np.std(xt - np.polyval([slope, off], x_dense))),
                anchors_x_um=xa, anchors_invols=iva)


def invols_interpolator(preflight: dict):
    """Log-linear interpolation of the pre-flight InvOLS(x) walk (SCM-PIT-B campaigns).

    Returns f(x_stage_um) -> InvOLS in m/V. Same method as r2an/s3_transfer.py.
    """
    xs = np.array(sorted(float(k) for k, v in preflight["invols_by_x"].items()
                         if v not in (None, "None")))
    vs = np.array([float(preflight["invols_by_x"][f"{k:.1f}"]) for k in xs])
    lv = np.log(vs)
    return lambda xq: np.exp(np.interp(np.asarray(xq, float), xs, lv))


def static_shape_fit(x_um, invols, L_bounds=(190.0, 300.0), x0_range=(-40.0, 40.0)):
    """Fit 1/InvOLS ~ (x-x0)^2 (3L-(x-x0)) (cantilever static deflection shape).

    Returns (x0, L, rms). Constrained so the laser cannot sit beyond the free end.
    """
    x_um = np.asarray(x_um, float); y = 1.0 / np.asarray(invols, float)
    best = None
    for x0 in np.arange(*x0_range, 0.1):
        s = x_um - x0
        if s.min() <= 0:
            continue
        for L in np.arange(max(L_bounds[0], s.max()), L_bounds[1], 0.5):
            m = s ** 2 * (3 * L - s)
            g = np.sum(y * m) / np.sum(m * m)
            rms = float(np.sqrt(np.mean((y / (g * m) - 1) ** 2)))
            if best is None or rms < best[2]:
                best = (float(x0), float(L), rms)
    return best

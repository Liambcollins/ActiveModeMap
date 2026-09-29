"""Physics-informed reconstruction arms (EB, EB+GP) through the ActiveModeMap package.

Probe-general version of ``scripts/eb_gp_dense/pkgrec.py`` (instrument repo, branch
feat/position-scale-and-reanalysis). One fitter for every probe:
``activemodemap.inference.PhysicsPosterior(fit_zeta=True, analytic_gain=True, fit_geometry=...)``
and ``activemodemap.hybrid.HybridSurrogate`` for the discrepancy GP.

Leakage-safe: an arm sees only (x_um, sel, Z[sel], freq) and returns the map on every x.

Data must be in the MODEL phase convention (``fmmpaper.io`` guarantees this) and on a
uniform frequency grid (use :func:`band_slice`).

    from fmmpaper import physrec
    zb, fb = physrec.band_slice(s.freq_Hz, s.Z[0], (50e3, 80e3), n_max=300)
    out = physrec.rec_eb(x_lever_um, sel, zb[sel], fb, physrec.GEOM["pppcontau"])
    out["Zrec"]  # EB+GP map, out["Zeb"] bare EB map, out["theta"] fitted parameters
"""
from __future__ import annotations

import numpy as np

from . import config

config.ensure_activemodemap()
from activemodemap.forward_model import EBForwardModel, ProbeGeometry  # noqa: E402
from activemodemap.hybrid import HybridSurrogate  # noqa: E402
from activemodemap.inference import PhysicsPosterior  # noqa: E402

_LAMBDA1_SQ = 1.87510407 ** 2

#: Probe presets. ``fit_geometry="setback"`` when the free resonance was measured (f0 is
#: then known); setback_bounds_um bracket the contact point. Values are the documented
#: ones: PPP-CONTAu from eb-gp-on-pppcontau-dense-2026-09-17 (+ setback 12.2 um fitted
#: in softprobe-recovery-vs-N), SCM-PIT-B from the R1/R2 blind fits (Phase 2d),
#: SCM-PIT-A from pkgrec (f0 never measured -> f0 free, contact 218-224 um).
GEOM = {
    "pppcontau": dict(L_um=445.0, f0_hz=13.649e3, k_lever=0.4057, setback_um=12.2,
                      tip_height_um=12.5, tip_mass_ratio=0.0018,
                      fit_geometry="setback", setback_bounds_um=(4.0, 25.0)),
    "scmpitB": dict(L_um=226.5, f0_hz=63.801e3, k_lever=1.697, setback_um=4.0,
                    tip_height_um=6.5, tip_mass_ratio=0.004,
                    fit_geometry="setback", setback_bounds_um=(1.0, 10.0)),
    "scmpitA": dict(L_um=232.0, f0_hz=60e3, k_lever=2.387, setback_um=10.1,
                    tip_height_um=12.5, tip_mass_ratio=0.004,
                    fit_geometry=True, setback_bounds_um=(8.0, 14.0), f0_bounds_hz=(50e3, 100e3)),
}
PRIOR_LO = np.array([1.0, 1.0, 0.5, -4.0, -4.0])
PRIOR_HI = np.array([5.0, 4.0, 2.5, 1.0, 1.0])


def band_slice(freq_Hz, Z, band, n_max=300):
    """Cut a band and decimate to at most ``n_max`` uniformly spaced bins (block mean).

    ``Z`` is (..., nfreq). Returns (Z_band, f_band) with a uniform grid, as the EB model needs.
    """
    m = (freq_Hz >= band[0]) & (freq_Hz <= band[1])
    f, z = np.asarray(freq_Hz)[m], np.asarray(Z)[..., m]
    k = max(1, int(np.ceil(f.size / n_max)))
    n = (f.size // k) * k
    f = f[:n].reshape(-1, k).mean(-1)
    z = z[..., :n].reshape(*z.shape[:-1], -1, k).mean(-1)
    fu = np.linspace(f[0], f[-1], f.size)          # tune grids are near-uniform (Hz rounding)
    if np.max(np.abs(fu - f)) > 0.5 * np.median(np.diff(f)):
        raise ValueError("frequency grid is far from uniform; resample explicitly")
    zi = lambda a: np.apply_along_axis(lambda r: np.interp(fu, f, r), -1, a)
    return zi(z.real) + 1j * zi(z.imag), fu


def build_model(freq_Hz, L_um, f0_hz, k_lever, setback_um, tip_height_um=12.5,
                tip_mass_ratio=0.004, n_modes=12, nx=481, **_):
    geom = ProbeGeometry(name="fmmpaper", f0_hz=f0_hz, k_lever=k_lever, L_um=L_um,
                         tip_setback_um=setback_um, tip_height_um=tip_height_um,
                         tilt_deg=11.0, tip_mass_ratio=tip_mass_ratio)
    fs = f0_hz / _LAMBDA1_SQ
    m = EBForwardModel(geom=geom, n_modes=n_modes, nx=nx, nf=len(freq_Hz),
                       omega_lo=float(freq_Hz[0] / fs), omega_hi=float(freq_Hz[-1] / fs))
    m.PRIOR_LO, m.PRIOR_HI = PRIOR_LO.copy(), PRIOR_HI.copy()
    return m


def _sigma(Z, half=6):
    from scipy.ndimage import uniform_filter1d
    sm = (uniform_filter1d(Z.real, 2 * half + 1, axis=-1)
          + 1j * uniform_filter1d(Z.imag, 2 * half + 1, axis=-1))
    a = np.abs(Z - sm).ravel()
    return float(np.median(a) + 1.4826 * np.median(np.abs(a - np.median(a))))


def rec_eb(x_um, sel, Z_sel, freq_Hz, geom: dict, gp=True, seed=0, n_modes=12):
    """Fit EB to the selected spectra; return bare-EB and EB+GP maps on every x.

    x_um: distance from the clamp (lever frame), all candidate positions.
    Z_sel: (len(sel), nfreq) complex, model phase convention, uniform freq_Hz.
    """
    x_um = np.asarray(x_um, float)
    sel = np.asarray(sel)
    L = geom["L_um"]
    scale = float(np.abs(Z_sel).max())
    Zs = np.asarray(Z_sel) / scale
    model = build_model(freq_Hz, n_modes=n_modes, **geom)
    xi = np.clip(x_um / L, 0, 1)
    data = [{"x": xi[i], "plus": Zs[k]} for k, i in enumerate(sel)]
    post = PhysicsPosterior(model, sigma=_sigma(Zs), rng=np.random.default_rng(seed),
                            fit_zeta=True, analytic_gain=True,
                            fit_geometry=geom.get("fit_geometry", "setback"),
                            setback_bounds_um=geom.get("setback_bounds_um"),
                            f0_bounds_hz=geom.get("f0_bounds_hz"))
    th = post.fit(data)
    cols = [int(np.argmin(np.abs(post.model.xi - v))) for v in xi]
    eb = post.predict_map()[:, cols].T
    Zrec = eb
    if gp:
        sur = HybridSurrogate(post).update(data)
        cm = sur.corrected_maps()
        Zrec = (cm[0] if isinstance(cm, (tuple, list)) else cm)[:, cols].T
    theta = dict(zip(post.names, map(float, th)))
    theta["zeta"] = 10 ** theta.get("log_zeta", np.log10(EBForwardModel.ZETA_DEFAULT))
    theta["k_ratio"] = 10 ** theta["log_alpha"]
    theta["red_chi2"] = float(post.red_chi2)
    return dict(Zrec=Zrec * scale, Zeb=eb * scale, theta=theta, gain=complex(post.gain) * scale)

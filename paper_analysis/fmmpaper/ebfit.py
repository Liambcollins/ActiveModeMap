"""Sparse EB reconstruction with a calibrated lever (the "v2" physics arm).

Replaces ``physrec.rec_eb`` for the paper. What changed and why:

* **Geometry is a lever constant.** Clamp offset c, effective length L, tip setback, tip
  height, kcone/k and the contact dashpot Q are calibrated ONCE per lever from a dense map
  with the joint multi-mode fit (``tools/calibrate_lever.py``, phase-2g method) and held
  fixed here. ``physrec.rec_eb`` refitted the setback on every sparse design and got a
  different value each time (1.5-7 um on one lever): a nuisance knob, not a measurement.
* **Only contact-state parameters are free per band:** k*/k (log_alpha), the intrinsic
  modal damping zeta, the electrostatic weight eps, and one complex gain solved analytically.
  These are the quantities a sparse capture is entitled to infer.
* **One objective.** Complex residuals only, weighted by the noise floor plus a fractional
  term. The old fitter ranked starts and did a first refinement on a log-amplitude cost,
  which prefers over-damped solutions whenever a design lands on a node (the N = 19 failure).
* **Physics-seeded starts.** log_alpha from the measured resonance frequency (1-D eigenvalue
  match), zeta from the measured linewidth, eps from the calibration; a few starts, and the
  best complex cost wins. No coarse scan, no reduced-chi2 guard.
* **EB+GP** adds a zero-mean complex GP over x on the residual at the measured positions with
  FIXED hyperparameters (length 0.1 L, 5 % nugget): with N < 10 points a learned length scale
  is noise, and PRESS selection was checked to be worse than the fixed prior at N <= 5.

Leakage: the arm sees (x, sel, Z[sel], f) and the lever calibration. The calibration comes
from a dense map of the same lever and is used as a lever constant, like a thermal tune.

    from fmmpaper import ebfit
    cal = ebfit.load_calibration("stiff")
    out = ebfit.rec_eb2(x_stage_um, sel, zb[sel], fb, cal, band="CR3")
    out["Zrec"], out["Zeb"], out["theta"]
"""
from __future__ import annotations

import json
import numpy as np
from scipy.optimize import least_squares

from . import config, recon, spectra
from . import jointeb as J

FREE_DEFAULT = ("log_alpha", "log_zeta", "eps")
BOUNDS = {"log_alpha": (1.0, 5.0), "log_kcone": (-1.0, 4.0), "log_zeta": (-4.0, -1.0), "eps": (-5.0, 5.0), "c_um": None}
X_SCALE = {"log_alpha": 0.1, "log_kcone": 0.3, "log_zeta": 0.2, "eps": 0.2, "c_um": 2.0}
C_HALF_RANGE_UM = 15.0   # "c_um" (clamp position in stage units, laser-frame drift) may move this far from the calibration


def load_calibration(probe: str, tag: str = ""):
    p = config.RESULTS_DIR / f"lever_calibration_{probe}{('_' + tag) if tag else ''}.json"
    return json.load(open(p))


def _sigma(Z, half=6):
    from scipy.ndimage import uniform_filter1d
    sm = uniform_filter1d(Z.real, 2 * half + 1, axis=-1) + 1j * uniform_filter1d(Z.imag, 2 * half + 1, axis=-1)
    a = np.abs(Z - sm).ravel()
    return float(np.median(a) + 1.4826 * np.median(np.abs(a - np.median(a))))


class _OneBand:
    """Single-band forward model with fixed lever geometry; free contact-state parameters."""

    def __init__(self, x_stage, sel, Z_sel, f_Hz, cal, band, free, dlf):
        self.x = np.asarray(x_stage, float); self.sel = np.asarray(sel)
        self.Z = np.asarray(Z_sel); self.f = np.asarray(f_Hz, float)
        self.P0 = dict(cal["P"]); self.free = tuple(free)
        bi = cal["bands"].index(band) if band in cal["bands"] else None
        self.dlf = float(cal["dlf"][bi]) if (dlf == "calib" and bi is not None) else float(dlf or 0.0)
        self.eps0 = float(cal["eps"][bi]) if bi is not None else 0.0
        self.zeta0 = float(cal["zeta"][bi]) if bi is not None else 0.002
        self.je = J.JointEB(self.x, [J.Band(band, self.f, self.Z)], cal["f0_hz"], n_modes=cal.get("n_modes", 14))
        scale = float(np.abs(self.Z).max()); self.scale = scale
        self.Zs = self.Z / scale
        self.sig = _sigma(self.Zs); self.frac = 0.03

    def P(self, v):
        P = dict(self.P0)
        for k, val in zip(self.free, v):
            if k in ("log_alpha", "log_kcone", "c_um"):
                P[k] = float(val)
        return P

    def unpack(self, v):
        d = dict(zip(self.free, v))
        zeta = 10 ** d.get("log_zeta", np.log10(self.zeta0)); eps = d.get("eps", self.eps0)
        return self.P(v), zeta, eps

    def maps(self, v, x=None):
        P, zeta, eps = self.unpack(v)
        (zp, ze), = self.je.band_maps(P, [zeta], x=x, dlf=[self.dlf])
        return zp + eps * ze

    def residual(self, v):
        m = self.maps(v, x=self.x[self.sel])
        g = np.vdot(m.ravel(), self.Zs.ravel()) / (np.vdot(m.ravel(), m.ravel()) + 1e-300)
        r = (g * m - self.Zs) / np.sqrt(self.sig ** 2 + (self.frac * np.abs(self.Zs)) ** 2)
        return np.concatenate([r.real.ravel(), r.imag.ravel()])

    def gain(self, v):
        m = self.maps(v, x=self.x[self.sel])
        return np.vdot(m.ravel(), self.Zs.ravel()) / (np.vdot(m.ravel(), m.ravel()) + 1e-300)

    # ------------------------------------------------------------- starts
    def alpha_starts(self, f_meas):
        """log_alpha values whose nearest model eigenfrequency matches the measured resonance."""
        grid = np.linspace(BOUNDS["log_alpha"][0] + 0.05, BOUNDS["log_alpha"][1] - 0.05, 120)
        err = []
        for la in grid:
            P = dict(self.P0); P["log_alpha"] = la
            fe = self.je.eig_freqs_Hz(P, nmax=8) * (1 + self.dlf)
            err.append(np.min(np.abs(np.log(fe / f_meas))))
        err = np.array(err)
        i = np.where((err <= np.roll(err, 1)) & (err <= np.roll(err, -1)) & (err < 0.05))[0]
        cands = sorted(grid[i], key=lambda la: err[np.argmin(np.abs(grid - la))])
        if not cands:
            cands = [grid[int(np.argmin(err))]]
        return [float(c) for c in cands[:2]]


def rec_eb2(x_stage_um, sel, Z_sel, f_Hz, cal, band, free=FREE_DEFAULT, dlf=0.0, gp=True,
            max_nfev=200, x_eval=None, gp_length_um=None):
    """Fit the contact state of one band from the selected spectra; return maps on every x.

    x_stage_um: stage positions of ALL candidate positions (the calibration frame).
    Z_sel: (len(sel), nf) complex, model phase convention, on the uniform grid f_Hz.
    Returns dict(Zrec [EB+GP], Zeb, theta, gain, cost, nfev, starts).
    """
    ob = _OneBand(x_stage_um, sel, Z_sel, f_Hz, cal, band, free, dlf)
    pk = spectra.peaks_along_x(ob.f, ob.Zs, (ob.f[0], ob.f[-1]))
    f_meas = float(np.nanmedian(pk["f_Hz"])); Qm = float(np.nanmedian(pk["Q"]))
    zeta_m = float(np.clip(0.5 / Qm if np.isfinite(Qm) and Qm > 0 else ob.zeta0, 1.5e-4, 0.05))
    la_starts = ob.alpha_starts(f_meas) if "log_alpha" in free else [ob.P0["log_alpha"]]
    z_starts = [np.log10(zeta_m * q) for q in (0.5, 1.0, 2.0)] if "log_zeta" in free else [None]
    e_starts = sorted({ob.eps0, 0.0, 0.5, -0.5}) if "eps" in free else [None]
    bnd = {k: (BOUNDS[k] if BOUNDS[k] is not None else (ob.P0["c_um"] - C_HALF_RANGE_UM, ob.P0["c_um"] + C_HALF_RANGE_UM)) for k in free}
    lo = np.array([bnd[k][0] for k in free]); hi = np.array([bnd[k][1] for k in free])
    xs = np.array([X_SCALE[k] for k in free])
    best, tried = None, 0
    for la in la_starts:
        for lz in z_starts:
            for e0 in e_starts:
                v0 = []
                for k in free:
                    v0.append({"log_alpha": la, "log_zeta": lz, "eps": e0, "log_kcone": ob.P0["log_kcone"], "c_um": ob.P0["c_um"]}[k])
                v0 = np.clip(np.array(v0, float), lo + 1e-6, hi - 1e-6)
                r = least_squares(ob.residual, v0, bounds=(lo, hi), method="trf", x_scale=xs,
                                  ftol=1e-9, xtol=1e-9, max_nfev=max_nfev)
                tried += 1
                if best is None or r.cost < best.cost:
                    best = r
    v = best.x; P, zeta, eps = ob.unpack(v); g = ob.gain(v)
    xe = ob.x if x_eval is None else np.asarray(x_eval, float)
    eb = g * ob.maps(v, x=xe) * ob.scale                       # (npos, nf)
    Zrec = eb
    gp_ell = np.nan
    if gp and len(sel) >= 2:
        # Discrepancy GP (Kennedy-O'Hagan): zero-mean complex GP over x on the residual at the
        # measured positions. Hyperparameters are FIXED (too few points to learn them): length
        # scale 0.1 L, the scale of mode-shape features, and a 5 % nugget, as in the package.
        R = np.asarray(Z_sel) - g * ob.maps(v, x=ob.x[sel]) * ob.scale
        gp_ell = gp_length_um if gp_length_um else 0.1 * float(ob.P0["L_um"])
        corr, sd = recon._gp_posterior(xe, ob.x[sel], R, gp_ell)
        Zrec = eb + corr
    n_res = 2 * ob.Zs.size
    theta = dict(zip(free, map(float, v)))
    theta.update(zeta=float(zeta), eps=float(eps), k_ratio=float(10 ** P["log_alpha"]), kcone_ratio=float(10 ** P["log_kcone"]),
                 setback_um=float(P["setback_um"]), c_um=float(P["c_um"]), dlf=ob.dlf, f_meas_Hz=f_meas, Q_meas=Qm,
                 red_chi2=float(2 * best.cost / n_res), gp_ell=float(gp_ell))
    return dict(Zrec=Zrec, Zeb=eb, theta=theta, gain=complex(g) * ob.scale, cost=float(best.cost),
                nfev=int(best.nfev), starts=tried, sd=(sd if gp and len(sel) >= 2 else None))


def fit_capture_bands(x_stage, Z, f, cal, bands_Hz, fit_frame_band="CR3", n_max=250, gp=True, x_eval=None,
                      kstar_from_higher_modes=True):
    """Fit every band of one capture (all rows of Z are measured positions). The clamp position c is
    refitted once on `fit_frame_band` (laser-frame drift), then every band gets its contact state.
    On CR1 alone eps and k* are degenerate whenever the electrostatic pathway dominates (soft lever
    at |V_dc| > 0), so with kstar_from_higher_modes CR1 takes k* from the median of the other bands
    and fits only zeta, eps and the gain. Returns (fits, cal_used)."""
    import json as _json
    sel = np.arange(len(x_stage)); cal = _json.loads(_json.dumps(cal)); fits = {}
    names = list(bands_Hz)
    if fit_frame_band:
        zb, fb = physrec_band_slice(f, Z, bands_Hz[fit_frame_band], n_max)
        o = rec_eb2(x_stage, sel, zb, fb, cal, fit_frame_band, free=("log_alpha", "log_zeta", "eps", "c_um"), gp=False)
        cal["P"]["c_um"] = o["theta"]["c_um"]
    for b in names:
        if b == "CR1" and kstar_from_higher_modes and len(names) > 1:
            continue
        zb, fb = physrec_band_slice(f, Z, bands_Hz[b], n_max)
        o = rec_eb2(x_stage, sel, zb, fb, cal, b, gp=gp, x_eval=x_eval); o["f"] = fb; fits[b] = o
    if "CR1" in names and kstar_from_higher_modes and len(names) > 1:
        # CR1 is fitted on an extended band that includes the antiresonance branch up to CR2: the
        # antiresonance is where the piezo and electrostatic pathways are resolved, so eps becomes
        # identifiable (on the resonance alone any admixture fits equally well).
        c1 = _json.loads(_json.dumps(cal)); c1["P"]["log_alpha"] = float(np.median([fits[b]["theta"]["log_alpha"] for b in fits]))
        lo1, hi1 = bands_Hz["CR1"]; hi_ext = 0.9 * bands_Hz[names[1]][0] if len(names) > 1 else hi1
        zb, fb = physrec_band_slice(f, Z, (0.7 * lo1, hi_ext), int(n_max * 5))   # keep the CR1 line resolved (~100 Hz bins)
        o = rec_eb2(x_stage, sel, zb, fb, c1, "CR1", free=("log_zeta", "eps"), gp=gp, x_eval=x_eval); o["f"] = fb
        o["theta"]["log_alpha"] = c1["P"]["log_alpha"]; fits["CR1"] = o
    return fits, cal


def physrec_band_slice(f, Z, band, n_max):
    from . import physrec
    return physrec.band_slice(f, Z, band, n_max=n_max)


def wideband_map(fits, cal, bands_Hz, x_eval, f_eval, gp_corr=True):
    """Evaluate the calibrated model over a whole frequency grid from per-band contact states: each
    frequency uses the (zeta, eps, gain, k*) of the nearest band, which reproduces the antiresonance
    branches between the resonances at never-visited positions (checked on the soft-probe live
    validation: 21.7 % over 100 Hz-1.1 MHz). With gp_corr the EB+GP correction of each band replaces
    the bare model inside that band."""
    names = list(bands_Hz); centers = np.array([np.mean(bands_Hz[b]) for b in names])
    near = np.argmin(np.abs(np.asarray(f_eval)[:, None] - centers[None, :]), axis=1)
    je = J.JointEB(np.asarray(x_eval, float), [J.Band("all", np.asarray(f_eval, float), None)], cal["f0_hz"], n_modes=cal.get("n_modes", 14))
    Zw = np.zeros((len(x_eval), len(f_eval)), complex)
    for i, b in enumerate(names):
        th = fits[b]["theta"]; P = dict(cal["P"]); P["log_alpha"] = th["log_alpha"]
        (zp, ze), = je.band_maps(P, [th["zeta"]], x=np.asarray(x_eval, float)); m = near == i
        Zw[:, m] = (fits[b]["gain"] * (zp + th["eps"] * ze))[:, m]
        if gp_corr and fits[b].get("Zrec") is not None and fits[b]["Zrec"].shape[0] == len(x_eval):
            fb = fits[b]["f"]; mb = (f_eval >= fb[0]) & (f_eval <= fb[-1])
            re = np.apply_along_axis(lambda r: np.interp(f_eval[mb], fb, r), 1, fits[b]["Zrec"].real)
            im = np.apply_along_axis(lambda r: np.interp(f_eval[mb], fb, r), 1, fits[b]["Zrec"].imag)
            Zw[:, mb] = re + 1j * im
    return Zw

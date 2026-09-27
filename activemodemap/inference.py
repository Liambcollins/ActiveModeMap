"""Physics-informed inference: posterior over EB parameters from measured spectra.

Staged fitting, built for sharp-resonance spectra where naive least squares
falls into broad-damping local minima:

  1. Global coarse scan over the mechanical params (alpha, kcone, Q[, zeta]) x a
     small grid of the amplitude params (A0, eps), ranked by a ROBUST
     log-amplitude cost (immune to phase decorrelation and peak dominance).
  2. Log-amplitude least-squares refinement from the top candidates (wide basin).
  3. Complex-residual polish with an SNR-aware noise model
     sigma_eff^2 = sigma^2 + (frac * |z|)^2   (additive floor + multiplicative),
     + Laplace approximation for the posterior covariance.

Warm starts reuse the previous MAP; a reduced-chi^2 guard triggers a global
rescan if an incremental fit degrades.

Parameter vector
----------------
    theta = [log_alpha, log_kcone, log_Q, log_eps, log_A0]      (always)
            + [log_zeta]                       if fit_zeta
            + [f0_kHz]                         if fit_geometry in (True, "f0")
            + [tip_setback_um]                 if fit_geometry in (True, "setback")

`fit_zeta` makes the intrinsic modal damping a parameter (see forward_model);
`fit_geometry` rebuilds the beam basis for the trial (f0, setback) on the same
Hz grid, which is what `scripts/eb_fit_real.py` did by hand. Prefer
fit_geometry="setback" when the free resonance has been measured: f0 is then a
known quantity, and letting it float only lets the fit hide a model-structure
error in the contact-to-free frequency ratio behind a wrong f0. `analytic_gain`
solves one complex gain g = <pred, meas> / <pred, pred> per evaluation instead
of fitting log_A0 (which is then pinned): it removes the zero-amplitude local
minimum a misaligned resonance otherwise falls into, and it absorbs the
instrument's global phase, which the model does not describe.

Data
----
`data` is a list of dicts with "x" (xi in [0, 1]) and a complex spectrum on the
model's Hz grid under "plus" and, for a domain pair, "minus". A single-domain
measurement simply omits "minus"; the piezo and electrostatic pathways are then
not separable and eps is a nuisance parameter.

The complex residual is only meaningful if the data carry the model's phase
convention — see `asylum.tune_to_complex` / `asylum.PHASE_CONVENTION`.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
from .forward_model import EBForwardModel

_DOMAINS = (("plus", +1.0), ("minus", -1.0))


class PhysicsPosterior:
    def __init__(self, model: EBForwardModel, sigma: float,
                 frac_noise: float = 0.03,
                 rng: np.random.Generator | None = None,
                 spot_fwhm_um: float = 0.0,
                 fit_zeta: bool = False,
                 fit_geometry: bool | str = False,
                 f0_bounds_hz: tuple[float, float] | None = None,
                 setback_bounds_um: tuple[float, float] | None = None,
                 analytic_gain: bool = False):
        self.base_model = model
        self.model = model                 # replaced by the fitted-geometry model after fit()
        self.sigma = sigma
        self.frac = frac_noise
        self.rng = rng or np.random.default_rng(1)
        self.fit_zeta = bool(fit_zeta)
        if fit_geometry not in (False, True, "f0", "setback"):
            raise ValueError("fit_geometry must be False, True, 'f0' or 'setback'")
        self.fit_geometry = fit_geometry
        self.fit_f0 = fit_geometry in (True, "f0")
        self.fit_setback = fit_geometry in (True, "setback")
        self.analytic_gain = bool(analytic_gain)
        self.gain = 1.0 + 0.0j

        lo, hi = list(model.PRIOR_LO[:5]), list(model.PRIOR_HI[:5])
        names = list(model.PARAM_NAMES[:5])
        if self.analytic_gain:              # log_A0 is pinned; the gain carries the scale
            lo[4], hi[4] = -0.01, 0.01
        if self.fit_zeta:
            if len(model.PRIOR_LO) >= 6:
                lo.append(model.PRIOR_LO[5]); hi.append(model.PRIOR_HI[5])
            else:
                lo.append(model.ZETA_PRIOR[0]); hi.append(model.ZETA_PRIOR[1])
            names.append(model.ZETA_NAME)
        self.n_core = len(names)
        g = model.geom
        if self.fit_f0:
            fb = f0_bounds_hz or (0.7 * g.f0_hz, 1.3 * g.f0_hz)
            lo.append(fb[0] / 1e3); hi.append(fb[1] / 1e3); names.append("f0_kHz")
        if self.fit_setback:
            sbb = setback_bounds_um or (max(0.5, 0.3 * g.tip_setback_um),
                                        3.0 * g.tip_setback_um)
            lo.append(sbb[0]); hi.append(sbb[1]); names.append("tip_setback_um")
        self.lo, self.hi = np.array(lo, float), np.array(hi, float)
        self.names = names
        # per-parameter step scale for the optimiser: log10 params ~0.3 decades,
        # f0 in kHz ~1% of its value, setback ~1 um
        xs = [0.3] * self.n_core
        if self.fit_f0:
            xs.append(0.01 * model.geom.f0_hz / 1e3)
        if self.fit_setback:
            xs.append(1.0)
        self.x_scale = np.array(xs, float)

        self.theta_map: np.ndarray | None = None
        self.cov: np.ndarray | None = None
        self.red_chi2 = np.inf
        dxi_um = model.geom.L_um * (model.xi[1] - model.xi[0])
        self.spot_sigma_pts = (spot_fwhm_um / 2.355) / dxi_um

    # ------------------------------------------------------------- utilities
    def _blur(self, z: np.ndarray) -> np.ndarray:
        if self.spot_sigma_pts <= 0.3:
            return z
        return (gaussian_filter1d(z.real, self.spot_sigma_pts, axis=1)
                + 1j * gaussian_filter1d(z.imag, self.spot_sigma_pts, axis=1))

    def _core(self, theta):
        return np.asarray(theta, float)[:self.n_core]

    def _model_for(self, theta) -> EBForwardModel:
        if not self.fit_geometry:
            return self.base_model
        th = np.asarray(theta, float)
        j = self.n_core
        f0 = sb = None
        if self.fit_f0:
            f0 = th[j] * 1e3; j += 1
        if self.fit_setback:
            sb = th[j]
        return self.base_model.with_geometry(f0_hz=f0, tip_setback_um=sb)

    @staticmethod
    def _domains(d):
        return [(k, s) for k, s in _DOMAINS if k in d]

    def _raw_columns(self, theta, data):
        """(pred, meas) per measured column, WITHOUT the analytic gain."""
        m = self._model_for(theta)
        resp = m.response(self._core(theta))
        zp, ze = self._blur(resp["piezo"]), self._blur(resp["elec"])
        A0 = 1.0 if self.analytic_gain else resp["A0"]
        out = []
        for d in data:
            i = int(np.argmin(np.abs(m.xi - d["x"])))
            for key, s in self._domains(d):
                out.append((A0 * (s * zp[:, i] + resp["eps"] * ze[:, i]), d[key]))
        return out

    def _solve_gain(self, cols):
        num = sum(np.vdot(p, mz) for p, mz in cols)
        den = sum(np.vdot(p, p) for p, _ in cols) + 1e-300
        return num / den

    def _pred_columns(self, theta, data):
        """Model spectra at the measured positions; list aligned with data."""
        cols = self._raw_columns(theta, data)
        if self.analytic_gain:
            g = self._solve_gain(cols)
            self.gain = g
            cols = [(g * p, mz) for p, mz in cols]
        return cols

    def _sigma_eff(self, meas):
        return np.sqrt(self.sigma ** 2 + (self.frac * np.abs(meas)) ** 2)

    # ------------------------------------------------------------ residuals
    def _residuals(self, theta, data):
        res = []
        for pred, meas in self._pred_columns(theta, data):
            se = self._sigma_eff(meas)
            r = (pred - meas) / se
            res.append(r.real)
            res.append(r.imag)
        return np.concatenate(res)

    def _logamp_residuals(self, theta, data):
        """Log-amplitude residuals. With `analytic_gain` the gain is solved in LOG
        space here (median offset), not as the complex least-squares gain: while
        the phases are still misaligned early in a fit the complex gain collapses
        toward zero and would make every log-amplitude residual enormous."""
        cols = (self._raw_columns(theta, data) if self.analytic_gain
                else self._pred_columns(theta, data))
        la_p = [np.log(np.abs(p) + self.sigma) for p, _ in cols]
        la_m = [np.log(np.abs(mz) + self.sigma) for _, mz in cols]
        if self.analytic_gain:
            off = np.median(np.concatenate(la_m) - np.concatenate(la_p))
            la_p = [x + off for x in la_p]
        return np.concatenate([p - mz for p, mz in zip(la_p, la_m)])

    # ----------------------------------------------------------- coarse scan
    def _coarse_scan(self, data, top_k: int = 3):
        lo, hi = self.lo, self.hi
        a_grid = np.linspace(lo[0] + 0.15, hi[0] - 0.1, 16)
        kc_grid = np.linspace(lo[1] + 0.15, hi[1] - 0.15, 7)
        q_grid = np.linspace(lo[2] + 0.15, hi[2] - 0.15, 5)
        eps_grid = np.linspace(lo[3] + 0.1, hi[3] - 0.1, 5)
        A0_grid = (np.array([0.0]) if self.analytic_gain
                   else np.linspace(lo[4] + 0.1, hi[4] - 0.1, 5))
        z_grid = (np.linspace(lo[5] + 0.1, hi[5] - 0.1, 5) if self.fit_zeta
                  else [None])
        if self.fit_zeta and len(z_grid) > 1:      # keep the scan affordable
            kc_grid = np.linspace(lo[1] + 0.15, hi[1] - 0.15, 5)
            q_grid = np.linspace(lo[2] + 0.15, hi[2] - 0.15, 4)

        mid = 0.5 * (lo + hi)
        if self.fit_geometry:                       # scan at the nominal geometry
            g = self.base_model.geom
            j = self.n_core
            if self.fit_f0:
                mid[j] = g.f0_hz / 1e3; j += 1
            if self.fit_setback:
                mid[j] = g.tip_setback_um
        m = self._model_for(mid)
        col_ix, signs, meas = [], [], []
        for d in data:
            i = int(np.argmin(np.abs(m.xi - d["x"])))
            for key, s in self._domains(d):
                col_ix.append(i); signs.append(s); meas.append(np.abs(d[key]))
        col_ix, signs = np.array(col_ix), np.array(signs)
        log_meas = np.log(np.array(meas) + self.sigma)

        results = []
        A0s, epss = 10.0 ** A0_grid, 10.0 ** eps_grid
        for a in a_grid:
            for kc in kc_grid:
                for q in q_grid:
                    for z in z_grid:
                        th = mid.copy()
                        th[0], th[1], th[2] = a, kc, q
                        if z is not None:
                            th[5] = z
                        resp = m.response(self._core(th))
                        zp = self._blur(resp["piezo"])[:, col_ix].T * signs[:, None]
                        ze = self._blur(resp["elec"])[:, col_ix].T
                        pred = (A0s[:, None, None, None]
                                * (zp[None, None] + epss[None, :, None, None]
                                   * ze[None, None]))
                        if self.analytic_gain:      # log-amplitude gain: median offset
                            la = np.log(np.abs(pred) + self.sigma)
                            off = np.median(log_meas[None, None] - la, axis=(2, 3))
                            la = la + off[..., None, None]
                        else:
                            la = np.log(np.abs(pred) + self.sigma)
                        cost = ((la - log_meas[None, None]) ** 2).sum(axis=(2, 3))
                        k_best = np.unravel_index(np.argmin(cost), cost.shape)
                        th = th.copy()
                        th[4] = A0_grid[k_best[0]]
                        th[3] = eps_grid[k_best[1]]
                        results.append((float(cost[k_best]), th))
        results.sort(key=lambda t: t[0])
        mech = [0, 1, 2] + ([5] if self.fit_zeta else [])
        starts, seen = [], []
        for cost, th in results:
            if not starts or all(np.linalg.norm(th[mech] - s[mech]) > 0.4 for s in seen):
                starts.append(th)
                seen.append(th)
            if len(starts) >= top_k:
                break
        return starts

    # ---------------------------------------------------------------- fitting
    def _polish(self, th0, data):
        """Log-amplitude refine -> complex polish. Returns (sol, cost)."""
        th0 = np.clip(np.asarray(th0, float), self.lo + 1e-9, self.hi - 1e-9)
        try:
            s1 = least_squares(self._logamp_residuals, th0, args=(data,),
                               bounds=(self.lo, self.hi), method="trf",
                               x_scale=self.x_scale, ftol=1e-8, max_nfev=150)
            s2 = least_squares(self._residuals, s1.x, args=(data,),
                               bounds=(self.lo, self.hi), method="trf",
                               x_scale=self.x_scale, ftol=1e-9, xtol=1e-9, max_nfev=300)
            return s2
        except Exception:
            return None

    def fit(self, data: list[dict], _rescanned: bool = False) -> np.ndarray:
        if self.theta_map is not None and not _rescanned:
            starts = [self.theta_map]         # warm start from previous step
        else:
            starts = self._coarse_scan(data)
        best, best_cost, best_jac = None, np.inf, None
        for th0 in starts:
            sol = self._polish(th0, data)
            if sol is not None and sol.cost < best_cost:
                best, best_cost, best_jac = sol.x, sol.cost, sol.jac
        if best is None:
            raise RuntimeError("physics fit failed from every start")
        n_res = sum(2 * d[k].size for d in data for k, _ in self._domains(d))
        self.red_chi2 = 2.0 * best_cost / n_res
        # goodness-of-fit guard: incremental fit degraded -> global rescan
        if self.red_chi2 > 4.0 and not _rescanned and self.theta_map is not None:
            self.theta_map = None
            return self.fit(data, _rescanned=True)
        self.theta_map = best
        self.model = self._model_for(best)
        if self.analytic_gain:
            self._pred_columns(best, data)            # refresh self.gain at the MAP
        JtJ = best_jac.T @ best_jac
        prior_prec = np.diag(1.0 / ((self.hi - self.lo) / 2.0) ** 2)
        self.cov = np.linalg.inv(JtJ + prior_prec)
        return best

    # ---------------------------------------------------------- posterior use
    def predict_map(self, theta=None, domain_sign: float = +1.0) -> np.ndarray:
        """Predicted complex map (nf, nx) for one domain, including blur, gain
        (analytic or A0) and eps — the object downstream consumers should use
        instead of reassembling it from `model.response`."""
        th = self.theta_map if theta is None else np.asarray(theta, float)
        m = self._model_for(th)
        resp = m.response(self._core(th))
        zp, ze = self._blur(resp["piezo"]), self._blur(resp["elec"])
        A0 = self.gain if self.analytic_gain else resp["A0"]
        return A0 * (domain_sign * zp + resp["eps"] * ze)

    def samples(self, n: int = 60) -> np.ndarray:
        L = np.linalg.cholesky(self.cov + 1e-12 * np.eye(len(self.theta_map)))
        s = self.theta_map[None, :] + self.rng.standard_normal(
            (n, len(self.theta_map))) @ L.T
        return np.clip(s, self.lo, self.hi)

    def predictive_maps(self, n_samples: int = 60):
        """Posterior-predictive mean/std of the |combined| maps (both domains,
        stacked) + posterior samples of the (D-NS, D-ESBS) positions in um."""
        ths = self.samples(n_samples)
        amps, spots = [], []
        for th in ths:
            m = self._model_for(th)
            resp = m.response(self._core(th))
            zpl = self.predict_map(th, +1.0)
            zmi = self.predict_map(th, -1.0)
            amps.append(np.concatenate([np.abs(zpl), np.abs(zmi)], axis=0))
            try:
                spots.append([(1 - m.find_dns(resp)) * m.geom.L_um,
                              (1 - m.find_desbs(resp)) * m.geom.L_um])
            except Exception:
                spots.append([np.nan, np.nan])
        amps = np.array(amps)
        return amps.mean(0), amps.std(0), np.array(spots)

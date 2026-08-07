"""Physics-informed inference: posterior over EB parameters from measured spectra.

Staged fitting, built for sharp-resonance spectra where naive least squares
falls into broad-damping local minima:

  1. Global coarse scan over the 3 mechanical params (alpha, kcone, Q) x a small
     grid of the 2 amplitude params (A0, eps), ranked by a ROBUST log-amplitude
     cost (immune to phase decorrelation and peak dominance).
  2. Log-amplitude least-squares refinement from the top candidates (wide basin).
  3. Complex-residual polish with an SNR-aware noise model
     sigma_eff^2 = sigma^2 + (frac * |z|)^2   (additive floor + multiplicative),
     + Laplace approximation for the posterior covariance.

Warm starts reuse the previous MAP; a reduced-chi^2 guard triggers a global
rescan if an incremental fit degrades.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
from .forward_model import EBForwardModel


class PhysicsPosterior:
    def __init__(self, model: EBForwardModel, sigma: float,
                 frac_noise: float = 0.03,
                 rng: np.random.Generator | None = None,
                 spot_fwhm_um: float = 0.0):
        self.model = model
        self.sigma = sigma
        self.frac = frac_noise
        self.rng = rng or np.random.default_rng(1)
        self.lo, self.hi = model.PRIOR_LO, model.PRIOR_HI
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

    def _pred_columns(self, theta, data):
        """Model spectra at the measured positions; list aligned with data."""
        resp = self.model.response(theta)
        zp, ze = self._blur(resp["piezo"]), self._blur(resp["elec"])
        out = []
        for d in data:
            i = int(np.argmin(np.abs(self.model.xi - d["x"])))
            for key, s in (("plus", +1.0), ("minus", -1.0)):
                out.append((resp["A0"] * (s * zp[:, i] + resp["eps"] * ze[:, i]),
                            d[key]))
        return out

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
        res = []
        for pred, meas in self._pred_columns(theta, data):
            res.append(np.log(np.abs(pred) + self.sigma)
                       - np.log(np.abs(meas) + self.sigma))
        return np.concatenate(res)

    # ----------------------------------------------------------- coarse scan
    def _coarse_scan(self, data, top_k: int = 3):
        a_grid = np.linspace(self.lo[0] + 0.15, self.hi[0] - 0.1, 16)
        kc_grid = np.linspace(self.lo[1] + 0.15, self.hi[1] - 0.15, 7)
        q_grid = np.linspace(self.lo[2] + 0.15, self.hi[2] - 0.15, 5)
        A0_grid = np.linspace(self.lo[4] + 0.1, self.hi[4] - 0.1, 5)
        eps_grid = np.linspace(self.lo[3] + 0.1, self.hi[3] - 0.1, 5)
        idx = [int(np.argmin(np.abs(self.model.xi - d["x"]))) for d in data]
        # measured log-amps, stacked (n_cols, nf)
        meas = np.array([np.abs(d[k]) for d in data for k in ("plus", "minus")])
        log_meas = np.log(meas + self.sigma)
        signs = np.array([+1.0, -1.0] * len(data))          # per column
        col_ix = np.repeat(idx, 2)

        mid = 0.5 * (self.lo + self.hi)
        results = []
        A0s = 10.0 ** A0_grid
        epss = 10.0 ** eps_grid
        for a in a_grid:
            for kc in kc_grid:
                for q in q_grid:
                    th = mid.copy()
                    th[0], th[1], th[2] = a, kc, q
                    resp = self.model.response(th)
                    zp = self._blur(resp["piezo"])[:, col_ix].T   # (ncol, nf)
                    ze = self._blur(resp["elec"])[:, col_ix].T
                    zp = zp * signs[:, None]
                    # broadcast over the (A0, eps) grid
                    pred = (A0s[:, None, None, None]
                            * (zp[None, None] + epss[None, :, None, None]
                               * ze[None, None]))
                    cost = ((np.log(np.abs(pred) + self.sigma)
                             - log_meas[None, None]) ** 2).sum(axis=(2, 3))
                    k_best = np.unravel_index(np.argmin(cost), cost.shape)
                    th = th.copy()
                    th[4] = A0_grid[k_best[0]]
                    th[3] = eps_grid[k_best[1]]
                    results.append((float(cost[k_best]), th))
        results.sort(key=lambda t: t[0])
        # keep top_k that are mechanically distinct (avoid near-duplicates)
        starts, seen = [], []
        for cost, th in results:
            if all(abs(th[0] - s) > 0.2 for s in seen) or not starts:
                starts.append(th)
                seen.append(th[0])
            if len(starts) >= top_k:
                break
        return starts

    # ---------------------------------------------------------------- fitting
    def _polish(self, th0, data):
        """Log-amplitude refine -> complex polish. Returns (sol, cost)."""
        try:
            s1 = least_squares(self._logamp_residuals, th0, args=(data,),
                               bounds=(self.lo, self.hi), method="trf",
                               x_scale=0.3, ftol=1e-8, max_nfev=150)
            s2 = least_squares(self._residuals, s1.x, args=(data,),
                               bounds=(self.lo, self.hi), method="trf",
                               x_scale=0.3, ftol=1e-9, xtol=1e-9, max_nfev=300)
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
        n_res = sum(4 * d["plus"].size for d in data)
        self.red_chi2 = 2.0 * best_cost / n_res
        # goodness-of-fit guard: incremental fit degraded -> global rescan
        if self.red_chi2 > 4.0 and not _rescanned and self.theta_map is not None:
            self.theta_map = None
            return self.fit(data, _rescanned=True)
        self.theta_map = best
        JtJ = best_jac.T @ best_jac
        prior_prec = np.diag(1.0 / ((self.hi - self.lo) / 2.0) ** 2)
        self.cov = np.linalg.inv(JtJ + prior_prec)
        return best

    # ---------------------------------------------------------- posterior use
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
            resp = self.model.response(th)
            zp, ze = self._blur(resp["piezo"]), self._blur(resp["elec"])
            zpl = resp["A0"] * (zp + resp["eps"] * ze)
            zmi = resp["A0"] * (-zp + resp["eps"] * ze)
            amps.append(np.concatenate([np.abs(zpl), np.abs(zmi)], axis=0))
            try:
                spots.append([(1 - self.model.find_dns(resp)) * self.model.geom.L_um,
                              (1 - self.model.find_desbs(resp)) * self.model.geom.L_um])
            except Exception:
                spots.append([np.nan, np.nan])
        amps = np.array(amps)
        return amps.mean(0), amps.std(0), np.array(spots)

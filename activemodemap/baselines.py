"""Model-agnostic baselines: random, equispaced, and a pure 2D-GP learner."""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


def next_random(rng, measured_x, xi_lo=0.15):
    return float(rng.uniform(xi_lo, 1.0))


def next_equispaced(step_idx, n_total, measured_x, xi_lo=0.15):
    """Progressive bisection ordering of an equispaced design on [xi_lo, 1]."""
    # van der Corput (base 2) sequence mapped onto the interval: 1/2, 1/4, 3/4, ...
    k, vdc, denom = step_idx + 1, 0.0, 1.0
    n = k
    while n:
        denom *= 2
        n, rem = divmod(n, 2)
        vdc += rem / denom
    return float(xi_lo + (1.0 - xi_lo) * vdc)


class GPMapLearner:
    """Pure GP over (x, f) on log-amplitude of both domain spectra."""

    def __init__(self, model, nf_sub=36, rng=None):
        self.model = model
        self.rng = rng or np.random.default_rng(2)
        self.if_sub = np.linspace(0, model.omega.size - 1, nf_sub).astype(int)
        self.X, self.y = [], []
        kern = (ConstantKernel(1.0) * RBF(length_scale=[0.15, 0.15, 0.5])
                + WhiteKernel(1e-3, noise_level_bounds=(1e-8, 1.0)))
        self.gp = GaussianProcessRegressor(kernel=kern, normalize_y=True)
        self._fitted = False

    def add(self, meas: dict, scale: float):
        f_n = self.model.omega[self.if_sub] / self.model.omega[-1]
        for key in ("plus", "minus"):
            amp = np.log10(np.abs(meas[key][self.if_sub]) / scale + 1e-4)
            dom = 0.0 if key == "plus" else 1.0  # domain as a third feature
            for fn, a in zip(f_n, amp):
                self.X.append([meas["x"], fn, dom])
                self.y.append(a)
        self.gp.fit(np.array(self.X), np.array(self.y))
        self._fitted = True

    def acquire(self, measured_x, n_cand=61, xi_lo=0.15):
        cand = np.linspace(xi_lo, 1.0, n_cand)
        mx = np.array(measured_x)
        cand = cand[np.min(np.abs(cand[:, None] - mx[None, :]), axis=1) > 0.01]
        f_n = self.model.omega[self.if_sub] / self.model.omega[-1]
        scores = []
        for c in cand:
            Xq = [[c, fn, d] for fn in f_n for d in (0.0, 1.0)]
            _, std = self.gp.predict(np.array(Xq), return_std=True)
            scores.append(std.sum())
        return float(cand[int(np.argmax(scores))])

    def predict_map(self, scale: float):
        """Predicted |combined| maps, both domains stacked: (2*nf, nx)."""
        f_n = self.model.omega / self.model.omega[-1]
        out = []
        for d in (0.0, 1.0):
            Xq = np.array([[x, fn, d] for fn in f_n for x in self.model.xi])
            mu = self.gp.predict(Xq)
            amp = (10.0 ** mu.reshape(self.model.omega.size,
                                      self.model.xi.size) - 1e-4) * scale
            out.append(np.clip(amp, 0, None))
        return np.concatenate(out, axis=0)

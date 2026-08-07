"""Hybrid surrogate: EB posterior + GP discrepancy, and model-free spot extraction.

Addresses the manuscript's finding that the EB model does not quantitatively
reproduce experiment/FEM near the tip. Two independent defenses:

1. Kennedy-O'Hagan correction: after the physics fit, the complex residual
   spectra r(x_i, f) = measured - EB(theta_MAP) are interpolated over x by a GP
   (one shared kernel, all frequencies/domains solved simultaneously), with a
   prior variance that GROWS toward the tip -- encoding the paper's observation
   that beam-model error concentrates in the near-tip/overhang region.
   Corrected map = EB(theta_MAP) + delta(x, f).

2. Model-free spot extraction from the corrected maps via the domain pair:
      piezo-like channel  P(x,f) = (plus - minus)/2
      electro  channel    E(x,f) = (plus + minus)/2
   D-NS  = near-tip minimum of |P| at the measured contact resonance,
   D-ESBS = near-tip minimum of |E| in the quasistatic band.
   The EB model then only STEERS sampling; the spots come from (corrected) data.

Acquisition 'hybrid': EIG while the physics posterior is still informative, then
local refinement -- measure at the predicted spots (bracketing them), which is
simultaneously where the GP correction is most needed.
"""

from __future__ import annotations

import numpy as np
from .forward_model import _parabolic_min
from .inference import PhysicsPosterior
from .acquisition import acquire_eig


# --------------------------------------------------------------- discrepancy GP
class DiscrepancyGP:
    """GP over x on complex residual spectra; shared kernel for all (f, domain).

    Prior std grows toward the tip: sd(x) = s0 * (0.3 + 0.7 * x^4), reflecting
    where beam-model error lives. Hyperparameters are fixed (few points; ML
    optimization is unstable with n < 10) except the scale s0, set from the
    residuals themselves.
    """

    def __init__(self, model, length_scale: float = 0.10, noise: float = 0.5):
        self.model = model
        self.l = length_scale
        self.noise = noise          # in per-column-normalized units

    @staticmethod
    def _sd_profile(x):
        return 0.3 + 0.7 * np.asarray(x) ** 4

    def _k(self, xa, xb):
        d = (np.asarray(xa)[:, None] - np.asarray(xb)[None, :]) / self.l
        return (self._sd_profile(xa)[:, None] * self._sd_profile(xb)[None, :]
                * np.exp(-0.5 * d ** 2))

    def fit(self, xs, R):
        """xs: (n,) positions; R: (n, m) whitened complex residuals
        (m = nf * n_domains, flattened). Each column (frequency/domain bin) is
        normalized to unit scale so resonance rows don't dominate the variance
        of quasistatic rows."""
        self.xs = np.asarray(xs)
        self.col_scale = np.sqrt(np.mean(np.abs(R) ** 2, axis=0)).clip(1.0)
        Rn = R / self.col_scale[None, :]
        K = self._k(self.xs, self.xs) + self.noise ** 2 * np.eye(len(xs))
        self.Kinv = np.linalg.inv(K)
        self.alpha = self.Kinv @ Rn
        return self

    def predict(self, xq):
        """mean: (nq, m) whitened correction; var: (nq, m) per-column variance."""
        Ks = self._k(np.asarray(xq), self.xs)
        mean = (Ks @ self.alpha) * self.col_scale[None, :]
        v_norm = np.clip(self._sd_profile(xq) ** 2
                         - np.einsum("ij,jk,ik->i", Ks, self.Kinv, Ks), 0, None)
        var = v_norm[:, None] * (self.col_scale ** 2)[None, :]
        return mean, var


# ------------------------------------------------------- model-free extraction
def _signed_zero_crossing(x, c, x_guess=None):
    """Null position from a complex profile via phase-rotated zero crossing.

    Robust to the laser-spot blur: a symmetric kernel preserves the zero of a
    locally-odd signed profile, whereas an |amplitude| minimum both broadens
    and collapses onto boundaries when the null is closer to the edge than the
    spot size. Picks the crossing nearest x_guess (physics prediction) if
    given, else nearest the free end; falls back to the amplitude minimum."""
    ph = np.angle(c[int(np.argmax(np.abs(c)))])
    v = (c * np.exp(-1j * ph)).real
    s = np.sign(v)
    idx = np.where(np.diff(s) != 0)[0]
    if idx.size == 0:
        return _parabolic_min(x, np.abs(c))
    xc = x[idx] + (x[idx + 1] - x[idx]) * v[idx] / (v[idx] - v[idx + 1])
    if x_guess is None:
        return float(xc[-1])
    return float(xc[int(np.argmin(np.abs(xc - x_guess)))])


def spots_from_maps(model, plus_map, minus_map, xi_min=0.85, qs_band=3,
                    i_res=None, guesses=(None, None)):
    """(D-NS, D-ESBS) in um-from-end from both-domain maps alone:
    P = (plus-minus)/2 is the piezo channel, E = (plus+minus)/2 electrostatic.
    The window xi >= xi_min should cover the data-dense refinement zone."""
    P = 0.5 * (plus_map - minus_map)
    E = 0.5 * (plus_map + minus_map)
    m = model.xi >= xi_min
    if i_res is None:
        i_mid = np.argmin(np.abs(model.xi - 0.5))
        i_res = int(np.argmax(np.abs(P[:, i_mid])))
    dns = _signed_zero_crossing(model.xi[m], P[i_res, m], guesses[0])
    e_qs = E[:qs_band, m].mean(axis=0)     # quasistatic band average (complex)
    desbs = _signed_zero_crossing(model.xi[m], e_qs, guesses[1])
    L = model.geom.L_um
    return (1.0 - dns) * L, (1.0 - desbs) * L


# ------------------------------------------------------------- hybrid surrogate
class HybridSurrogate:
    """EB posterior + discrepancy GP; corrected maps and spot posteriors."""

    def __init__(self, post: PhysicsPosterior):
        self.post = post
        self.model = post.model
        self.gp = None

    def update(self, data):
        m, post = self.model, self.post
        resp = m.response(post.theta_map)
        zp, ze = post._blur(resp["piezo"]), post._blur(resp["elec"])
        pred = {"plus": resp["A0"] * (zp + resp["eps"] * ze),
                "minus": resp["A0"] * (-zp + resp["eps"] * ze)}
        xs, rows = [], []
        for d in data:
            i = int(np.argmin(np.abs(m.xi - d["x"])))
            xs.append(m.xi[i])
            rows.append(np.concatenate([(d["plus"] - pred["plus"][:, i]),
                                        (d["minus"] - pred["minus"][:, i])])
                        / post.sigma)
        self.gp = DiscrepancyGP(m).fit(np.array(xs), np.array(rows))
        self._pred = pred
        return self

    def corrected_maps(self, xq=None, sample_rng=None):
        """EB(theta_MAP) + GP mean correction on the full grid; optionally one
        GP posterior sample instead of the mean (for uncertainty)."""
        m = self.model
        xq = m.xi if xq is None else xq
        mean, var = self.gp.predict(xq)
        if sample_rng is not None:
            mean = mean + np.sqrt(var) * (
                sample_rng.standard_normal(mean.shape)
                + 1j * sample_rng.standard_normal(mean.shape)) / np.sqrt(2)
        nf = m.omega.size
        corr = mean.T * self.post.sigma          # (2nf, nx)
        plus = self._pred["plus"] + corr[:nf]
        minus = self._pred["minus"] + corr[nf:]
        return plus, minus

    def spot_posterior(self, n_samples=40, rng=None):
        """Spot samples combining theta-posterior x GP-discrepancy uncertainty."""
        rng = rng or np.random.default_rng(3)
        # fix the resonance row from the mean map so samples share a reference,
        # and use the EB-posterior spots as crossing-selection guesses
        plus_m, minus_m = self.corrected_maps()
        P = 0.5 * (plus_m - minus_m)
        i_mid = np.argmin(np.abs(self.model.xi - 0.5))
        i_res = int(np.argmax(np.abs(P[:, i_mid])))
        L = self.model.geom.L_um
        dns_g, desbs_g = self.model.spots_um_from_end(self.post.theta_map)
        guesses = (1 - dns_g / L, 1 - desbs_g / L)
        out = []
        for _ in range(n_samples):
            plus, minus = self.corrected_maps(sample_rng=rng)
            try:
                out.append(spots_from_maps(self.model, plus, minus,
                                           i_res=i_res, guesses=guesses))
            except Exception:
                out.append((np.nan, np.nan))
        return np.array(out)


# ---------------------------------------------------------------- acquisition
def acquire_hybrid(post, surro, measured_x, step, rng):
    """EIG early; then bracket the predicted spots (local refinement)."""
    if step < 5 or surro is None:
        return acquire_eig(post, measured_x)
    spots = surro.spot_posterior(20, rng)
    dns = np.nanmedian(spots[:, 0])
    desbs = np.nanmedian(spots[:, 1])
    L = surro.model.geom.L_um
    targets = [1 - dns / L, 1 - desbs / L,
               1 - (dns + 2.0) / L, 1 - (desbs + 2.0) / L,
               1 - max(dns - 2.0, 0.5) / L]
    for t in targets:
        t = float(np.clip(t, 0.15, 1.0))
        if not measured_x or np.min(np.abs(np.array(measured_x) - t)) > 0.008:
            return t
    return float(np.clip(1 - (dns + rng.uniform(-3, 3)) / L, 0.15, 1.0))

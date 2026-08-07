"""Acquisition strategies: which detection position x to measure next.

Physics-informed strategies (use PhysicsPosterior):
  A 'variance'  - integrated posterior-predictive variance over frequency at x
  B 'eig'       - D-optimal expected information gain about theta (Laplace)
  C 'spot'      - expected reduction in posterior variance of the D-NS/D-ESBS

Baselines:
  'random', 'equispaced', 'gp' (model-agnostic 2D GP, in baselines.py)
"""

from __future__ import annotations

import numpy as np
from .inference import PhysicsPosterior


def _candidates(model, measured_x, n_cand=61, xi_lo=0.15, min_sep=0.01):
    cand = np.linspace(xi_lo, 1.0, n_cand)
    if measured_x:
        mx = np.array(measured_x)
        cand = cand[np.min(np.abs(cand[:, None] - mx[None, :]), axis=1) > min_sep]
    return cand


def acquire_variance(post: PhysicsPosterior, measured_x, n_samples=50) -> float:
    """A: x maximizing integrated predictive variance (both domains)."""
    m = post.model
    cand = _candidates(m, measured_x)
    idx = np.array([np.argmin(np.abs(m.xi - c)) for c in cand])
    maps = []
    for th in post.samples(n_samples):
        resp = m.response(th)
        zp = np.abs(m.measured(th, +1.0, resp)[:, idx])
        zm = np.abs(m.measured(th, -1.0, resp)[:, idx])
        maps.append(np.concatenate([zp, zm], axis=0))
    maps = np.array(maps)                     # (ns, 2nf, ncand)
    score = maps.var(axis=0).sum(axis=0)      # integrate variance over f
    return float(cand[int(np.argmax(score))])


def _jacobian_at_x(post: PhysicsPosterior, xi_val: float, dtheta=1e-3):
    """Whitened numeric Jacobian of the stacked (re, im, both-domain) spectrum
    at candidate position x wrt theta, evaluated at the MAP."""
    m = post.model
    i = int(np.argmin(np.abs(m.xi - xi_val)))
    th0 = post.theta_map

    def stacked(th):
        resp = m.response(th)
        zp = post._blur(resp["piezo"])[:, i]
        ze = post._blur(resp["elec"])[:, i]
        out = []
        for s in (+1.0, -1.0):
            z = resp["A0"] * (s * zp + resp["eps"] * ze)
            out.extend([z.real, z.imag])
        return np.concatenate(out)

    f0 = stacked(th0)
    # SNR-aware whitening using the MAP prediction as the expected signal
    nf = f0.size // 4
    amp = np.sqrt(f0[:nf] ** 2 + f0[nf:2 * nf] ** 2)
    amp2 = np.sqrt(f0[2 * nf:3 * nf] ** 2 + f0[3 * nf:] ** 2)
    se = np.concatenate([np.tile(post._sigma_eff(amp), 2),
                         np.tile(post._sigma_eff(amp2), 2)])
    J = np.empty((f0.size, th0.size))
    for k in range(th0.size):
        th = th0.copy()
        th[k] += dtheta
        J[:, k] = (stacked(th) - f0) / dtheta
    return J / se[:, None]


def acquire_eig(post: PhysicsPosterior, measured_x, n_cand=41) -> float:
    """B: D-optimal EIG  0.5 * logdet(I + Sigma J_w^T J_w)."""
    cand = _candidates(post.model, measured_x, n_cand=n_cand)
    best_x, best = cand[0], -np.inf
    p = len(post.theta_map)
    for c in cand:
        Jw = _jacobian_at_x(post, c)
        sign, logdet = np.linalg.slogdet(np.eye(p) + post.cov @ (Jw.T @ Jw))
        if sign > 0 and logdet > best:
            best, best_x = logdet, c
    return float(best_x)


def acquire_spot(post: PhysicsPosterior, measured_x, n_cand=41,
                 dtheta=1e-3) -> float:
    """C: minimize expected posterior variance of (D-NS, D-ESBS) positions."""
    m, th0 = post.model, post.theta_map
    # gradient of spot positions wrt theta
    s0 = np.array(m.spots_um_from_end(th0))
    G = np.empty((2, th0.size))
    for k in range(th0.size):
        th = th0.copy()
        th[k] += dtheta
        G[:, k] = (np.array(m.spots_um_from_end(th)) - s0) / dtheta
    cand = _candidates(m, measured_x, n_cand=n_cand)
    prec0 = np.linalg.inv(post.cov)
    best_x, best = cand[0], np.inf
    for c in cand:
        Jw = _jacobian_at_x(post, c)
        cov_new = np.linalg.inv(prec0 + Jw.T @ Jw)
        score = np.trace(G @ cov_new @ G.T)   # summed spot-position variance
        if score < best:
            best, best_x = score, c
    return float(best_x)


PHYSICS_STRATEGIES = {"variance": acquire_variance, "eig": acquire_eig,
                      "spot": acquire_spot}

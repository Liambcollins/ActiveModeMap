"""Forward model: cantilever response map R(x, f) under piezo + electrostatic excitation.

Stand-in two-pathway Euler-Bernoulli contact-resonance solver, implemented from the
description in Checa et al., "Dynamics of Null and Electrostatic Blind Spots for
Quantitative PFM" (Methods / Supplementary Note 1).

Modal (assumed-modes) formulation on the full clamped-free beam with the tip-sample
contact acting at an *interior* point xc = L - dx. Because the basis spans the whole
beam, the distal overhang with a true free end is handled automatically (this is the
same physics as the paper's two-segment solution, converged in the modal limit).

Design notes:
- All lengths scaled by L (xi in [0, 1]); frequencies scaled so the free fundamental
  is Omega_1 = 1.8751^2 = 3.516.  `freq_scale_hz` maps scaled Omega -> Hz.
- Interface is intentionally minimal (`response(theta) -> dict of complex maps`) so the
  user's CR2Beam / eb_cr_afm solver can replace this class without touching the rest
  of the framework: any object exposing `.response(theta)`, `.xi`, `.omega`,
  `.find_dns(...)`, `.find_desbs(...)` works.

Inferred parameters (theta, in log10 space unless noted):
    log_alpha  : contact stiffness ratio k* / k_lever          (k1 = k2 = k*)
    log_kcone  : tip-cone lateral stiffness ratio kcone/k_lever (series with k2)
    log_Q      : contact quality factor (sets dashpot c1)
    log_eps    : electrostatic-to-piezo drive ratio
    log_A0     : overall detection gain / signal scale
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

# clamped-free beam eigenvalues
_LAMBDAS = np.array(
    [1.87510407, 4.69409113, 7.85475744, 10.99554073, 14.13716839,
     17.27875953, 20.42035225, 23.56194490, 26.70353756, 29.84513021,
     32.98672286, 36.12831552, 39.26990817, 42.41150082])


def beam_modes(xi: np.ndarray, n_modes: int):
    """Clamped-free eigenmodes (normalized: int phi^2 dxi = 1) and derivatives."""
    lam = _LAMBDAS[:n_modes]
    phi = np.empty((n_modes, xi.size))
    dphi = np.empty_like(phi)
    for j, l in enumerate(lam):
        sig = (np.cosh(l) + np.cos(l)) / (np.sinh(l) + np.sin(l))
        phi[j] = (np.cosh(l * xi) - np.cos(l * xi)
                  - sig * (np.sinh(l * xi) - np.sin(l * xi)))
        dphi[j] = l * (np.sinh(l * xi) + np.sin(l * xi)
                       - sig * (np.cosh(l * xi) - np.cos(l * xi)))
    return lam, phi, dphi


@dataclass
class ProbeGeometry:
    """Fixed (non-inferred) probe geometry; defaults ~ BudgetSensors Multi75E-G."""
    name: str = "Multi75E-G"
    f0_hz: float = 75e3          # free fundamental resonance
    k_lever: float = 3.0         # N/m, cantilever stiffness
    L_um: float = 225.0          # cantilever length
    tip_setback_um: float = 11.0  # dx: tip contact point set back from free end
    tip_height_um: float = 17.0
    tilt_deg: float = 11.0       # mounting tilt + sample inclination
    tip_mass_ratio: float = 0.03  # tip mass / beam mass
    overhang_width_factor: float = 0.4  # reduced effective electrostatic width on overhang


class EBForwardModel:
    """theta -> complex response maps z(x, f) for piezo / electrostatic pathways."""

    PARAM_NAMES = ["log_alpha", "log_kcone", "log_Q", "log_eps", "log_A0"]
    # prior box in log10 space (alpha, kcone/k_lever, Q, eps, A0)
    PRIOR_LO = np.array([1.0, 1.5, 0.5, -1.0, -0.5])
    PRIOR_HI = np.array([5.0, 3.5, 2.0, 1.0, 0.5])

    def __init__(self, geom: ProbeGeometry | None = None,
                 n_modes: int = 12, nx: int = 241, nf: int = 161,
                 omega_lo: float = 0.5, omega_hi: float = 25.0):
        self.geom = geom or ProbeGeometry()
        g = self.geom
        self.n_modes = n_modes
        self.xi = np.linspace(0.0, 1.0, nx)
        self.omega = np.linspace(omega_lo, omega_hi, nf)
        self.freq_scale_hz = g.f0_hz / _LAMBDAS[0] ** 2  # Hz per scaled-Omega unit
        self.f_hz = self.omega * self.freq_scale_hz

        self.xi_c = 1.0 - g.tip_setback_um / g.L_um
        self.h = g.tip_height_um / g.L_um
        self.tilt = np.deg2rad(g.tilt_deg)

        lam, phi, dphi = beam_modes(self.xi, n_modes)
        self.lam, self.phi_grid, self.dphi_grid = lam, phi, dphi
        # basis at the contact point
        _, phc, dphc = beam_modes(np.array([self.xi_c]), n_modes)
        self.phi_c = phc[:, 0]
        self.dphi_c = dphc[:, 0]
        self.omega_j = lam ** 2                     # scaled modal frequencies
        # scaled lever stiffness: k_lever = 3 EI / L^3 -> 3.0 in scaled units
        self.k_lever_scaled = 3.0

        # distributed electrostatic load shape q(xi) ~ w_eff(xi) / d(xi)^2
        gap = self.h * np.cos(self.tilt) + (self.xi_c - self.xi) * np.sin(self.tilt)
        gap = np.clip(gap, 0.15 * self.h, None)
        w_eff = np.where(self.xi <= self.xi_c, 1.0, g.overhang_width_factor)
        qshape = w_eff / gap ** 2
        # modal projection of the distributed load (trapezoid over the grid)
        self.f_elec_modal = np.trapezoid(qshape[None, :] * self.phi_grid,
                                         self.xi, axis=1)
        self.f_elec_modal /= np.max(np.abs(self.f_elec_modal))

        # tilt-coupled lateral kinematics at apex: u_lat = h*z'(xc) + sin(tilt)*z(xc)
        self.b_lat = self.h * self.dphi_c + np.sin(self.tilt) * self.phi_c
        self.b_vert = self.phi_c

    # ------------------------------------------------------------------ core
    def response(self, theta: np.ndarray) -> dict:
        """Complex response maps for unit drives.

        Returns dict with 'piezo', 'elec' : (nf, nx) complex arrays,
        plus 'A0', 'eps' scalars. Measured (combined) response for a domain of
        sign s is  A0 * (s * piezo + eps * elec).
        """
        log_alpha, log_kcone, log_Q, log_eps, log_A0 = theta
        alpha, kcone_r = 10.0 ** log_alpha, 10.0 ** log_kcone
        Q, eps, A0 = 10.0 ** log_Q, 10.0 ** log_eps, 10.0 ** log_A0

        k1 = alpha * self.k_lever_scaled
        kcone = kcone_r * self.k_lever_scaled
        k2 = 1.0 / (1.0 / k1 + 1.0 / kcone)        # apex compliance in series
        # contact resonance estimate for damping scale
        c1 = np.sqrt(k1) / Q                        # heuristic dashpot scaling

        n, nf = self.n_modes, self.omega.size
        K = np.diag(self.omega_j ** 2).astype(complex)
        K += k1 * np.outer(self.b_vert, self.b_vert)
        K += k2 * np.outer(self.b_lat, self.b_lat)
        M = np.eye(n) + self.geom.tip_mass_ratio * np.outer(self.phi_c, self.phi_c)
        C = c1 * np.outer(self.b_vert, self.b_vert)
        C += np.diag(2 * 0.002 * self.omega_j)      # small intrinsic beam damping

        # batched frequency solve: (K + i w C - w^2 M) q = F
        w = self.omega[:, None, None]
        A = K[None] + 1j * w * C[None] - w ** 2 * M[None]

        F_p = ((k1 + 1j * self.omega[:, None] * c1) * self.b_vert[None, :])
        F_e = np.broadcast_to(self.f_elec_modal, (nf, n))
        rhs = np.stack([F_p, F_e], axis=-1)          # (nf, n, 2)
        sol = np.linalg.solve(A, rhs)                # (nf, n, 2)

        z_p = sol[..., 0] @ self.phi_grid            # (nf, nx)
        z_e = sol[..., 1] @ self.phi_grid
        # normalize pathways by their quasistatic free-end response: theta-smooth,
        # Q-independent, and makes eps the quasistatic electrostatic/piezo ratio
        z_p = z_p / np.abs(z_p[0, -1])
        z_e = z_e / np.abs(z_e[0, -1])
        return {"piezo": z_p, "elec": z_e, "A0": A0, "eps": eps}

    def measured(self, theta: np.ndarray, domain_sign: float = 1.0,
                 resp: dict | None = None) -> np.ndarray:
        r = resp or self.response(theta)
        return r["A0"] * (domain_sign * r["piezo"] + r["eps"] * r["elec"])

    # ------------------------------------------------------- derived features
    def contact_resonance(self, resp: dict) -> float:
        """First contact-resonance (scaled Omega) from mid-lever piezo response."""
        i_mid = np.argmin(np.abs(self.xi - 0.5))
        amp = np.abs(resp["piezo"][:, i_mid])
        return self.omega[int(np.argmax(amp))]

    def find_dns(self, resp: dict, xi_min: float = 0.6) -> float:
        """D-NS: near-tip minimum of on-resonance piezo displacement response."""
        w_r = self.contact_resonance(resp)
        i_f = int(np.argmin(np.abs(self.omega - w_r)))
        m = self.xi >= xi_min
        amp = np.abs(resp["piezo"][i_f, m])
        return _parabolic_min(self.xi[m], amp)

    def find_desbs(self, resp: dict, xi_min: float = 0.6,
                   omega_qs: float = 1.0) -> float:
        """D-ESBS: near-tip zero of the quasistatic electrostatic response."""
        i_f = int(np.argmin(np.abs(self.omega - omega_qs)))
        m = self.xi >= xi_min
        amp = np.abs(resp["elec"][i_f, m])
        return _parabolic_min(self.xi[m], amp)

    def spots_um_from_end(self, theta: np.ndarray) -> tuple[float, float]:
        """(D-NS, D-ESBS) as distance from the cantilever free end, in um."""
        r = self.response(theta)
        L = self.geom.L_um
        return ((1.0 - self.find_dns(r)) * L, (1.0 - self.find_desbs(r)) * L)


def _parabolic_min(x: np.ndarray, y: np.ndarray) -> float:
    """Sub-grid minimum via 3-point parabola around the discrete argmin."""
    i = int(np.argmin(y))
    if i == 0 or i == y.size - 1:
        return float(x[i])
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = (y0 - 2 * y1 + y2)
    if abs(denom) < 1e-30:
        return float(x[i])
    delta = 0.5 * (y0 - y2) / denom
    return float(x[i] + delta * (x[i + 1] - x[i]))

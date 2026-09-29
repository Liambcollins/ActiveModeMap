"""Joint multi-mode Euler-Bernoulli fit: ONE lever/contact geometry shared by every mode band.

Phase 2g. Single-band fits (``physrec``) return a different tip setback and contact stiffness
for every mode, so those parameters are not lever constants. This module asks whether one
geometry can describe all modes at once, and what has to be added if it cannot.

Physics is the package model (``activemodemap.forward_model.EBForwardModel.response``),
re-implemented here so that several frequency bands share one modal basis and so that two
terms the package lacks can be switched on:

* **frame**: positions enter as xi = (x_stage - c) / L_eff. The EB model only ever sees x/L,
  so a stage offset c and an effective length L_eff (lever length in stage units; it also
  absorbs a steady laser drift over a walk whose position and time are locked together) are
  the only frame parameters that matter. The static-shape InvOLS ruler measures c
  independently.
* **tip rotary inertia**: M += J * phi'(xc) phi'(xc)^T (J = I_tip / (m_beam L^2)), next to
  the lumped tip mass the package already has (mu = m_tip / m_beam).

Everything else is as in the package: modal (assumed-modes) clamped-free beam with the contact
at xi_c = 1 - setback/L; vertical contact spring k1 = alpha*k_lever with dashpot
c1 = sqrt(k1)/Q_c; lateral spring k2 = (1/k1 + 1/kcone)^-1 through the tilt/tip-height
kinematics; intrinsic modal damping 2*zeta*omega_j; tip (piezo) force plus distributed
electrostatic load. Timoshenko shear is NOT modelled.

Per band n the free nuisance parameters are zeta_n (intrinsic damping), eps_n (electrostatic
to piezo weight, signed) and one complex gain g_n solved analytically per evaluation.

Positions/frequencies: x in stage um, f in Hz; data in the MODEL phase convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

LAMBDAS = np.array([1.87510407, 4.69409113, 7.85475744, 10.99554073, 14.13716839,
                    17.27875953, 20.42035225, 23.56194490, 26.70353756, 29.84513021,
                    32.98672286, 36.12831552, 39.26990817, 42.41150082, 45.55309348,
                    48.69468613])


def beam_modes(xi, n):
    """Clamped-free modes (unit-norm) and d/dxi, in a cancellation-free form.

    Direct cosh(lx) - s*sinh(lx) loses all precision for l >~ 30 (terms ~e^l). With
    1 - s = (sin l - cos l - e^-l) / (sinh l + sin l):
        phi  = e^{-lx} + (1-s) sinh(lx) - cos(lx) + s sin(lx)
        phi' = l [ -e^{-lx} + (1-s) cosh(lx) + sin(lx) + s cos(lx) ]
    and (1-s) sinh(lx), (1-s) cosh(lx) are evaluated as ratios ~ e^{l(x-1)}.
    """
    lam = LAMBDAS[:n]
    xi = np.asarray(xi, float)
    phi = np.empty((n, xi.size)); dphi = np.empty_like(phi)
    for j, l in enumerate(lam):
        el = np.exp(-l)
        den = 0.5 * (1 - el * el) + np.sin(l) * el          # (sinh l + sin l) * e^{-l}
        one_m_s = (np.sin(l) - np.cos(l) - el)              # times 1/(sinh l + sin l)
        s = 1.0 - one_m_s * el / den
        ex = np.exp(l * (xi - 1.0))                          # e^{lx} e^{-l}
        emx = np.exp(-l * xi)
        sh_ratio = 0.5 * ex * (1 - np.exp(-2 * l * xi)) / den   # sinh(lx)/(sinh l + sin l)
        ch_ratio = 0.5 * ex * (1 + np.exp(-2 * l * xi)) / den
        phi[j] = emx + one_m_s * sh_ratio - np.cos(l * xi) + s * np.sin(l * xi)
        dphi[j] = l * (-emx + one_m_s * ch_ratio + np.sin(l * xi) + s * np.cos(l * xi))
    return lam, phi, dphi


# ------------------------------------------------------------------ parameter bookkeeping
#: name: (default, lo, hi). Geometry lengths in stage um; log10 for stiffness/Q/zeta.
SHARED = {
    "log_alpha": (3.0, 1.0, 5.0),        # k*/k_lever
    "log_kcone": (1.5, -1.0, 4.0),       # kcone/k_lever
    "log_Qc": (2.0, 0.3, 4.0),           # contact dashpot quality factor
    "setback_um": (12.0, 1.0, 40.0),     # contact point from the free end
    "tip_h_um": (12.5, 3.0, 25.0),       # tip height (lateral lever arm)
    "c_um": (0.0, -60.0, 60.0),          # stage position of the clamp
    "L_um": (445.0, 400.0, 560.0),       # lever length in stage units
    "mu": (0.0018, 0.0, 0.2),            # tip mass / beam mass
    "J": (0.0, 0.0, 0.02),               # tip rotary inertia / (m_beam L^2)
}
PER_BAND = {"log_zeta": (-2.5, -4.0, -1.0), "eps": (0.0, -20.0, 20.0), "dlf": (0.0, -0.04, 0.04)}


@dataclass
class Spec:
    """Which shared parameters are free; the rest are fixed at `fixed` (or SHARED default)."""
    free: tuple
    fixed: dict = field(default_factory=dict)
    name: str = ""
    band_freq: bool = False      # per-band frequency correction dlf_n: model evaluated at f*(1+dlf_n)
    shared_damping: bool = False  # one zeta and one eps for all bands (the old wideband fit)
    bounds: dict = field(default_factory=dict)  # per-parameter (lo, hi) overriding SHARED (other levers)


@dataclass
class Band:
    name: str
    f_Hz: np.ndarray
    Z: np.ndarray            # (npos, nf) complex, model phase convention


class JointEB:
    def __init__(self, x_stage, bands: list[Band], f0_hz, tilt_deg=11.0, n_modes=16,
                 nxq=241, overhang_width=0.4):
        self.x = np.asarray(x_stage, float)
        self.bands = bands
        self.fs = f0_hz / LAMBDAS[0] ** 2
        self.n = n_modes
        self.tilt = np.deg2rad(tilt_deg)
        self.xq = np.linspace(0, 1, nxq)
        _, self.phi_q, _ = beam_modes(self.xq, n_modes)
        self.omega_j = LAMBDAS[:n_modes] ** 2
        self.ow = overhang_width

    # ------------------------------------------------------------------ physics
    def _matrices(self, P):
        L = P["L_um"]
        xi_c = 1.0 - P["setback_um"] / L
        h = P["tip_h_um"] / L
        _, phc, dphc = beam_modes([xi_c], self.n)
        phc, dphc = phc[:, 0], dphc[:, 0]
        b_vert = phc
        b_lat = h * dphc + np.sin(self.tilt) * phc
        alpha, kc = 10 ** P["log_alpha"], 10 ** P["log_kcone"]
        k1 = 3.0 * alpha
        k2 = 1.0 / (1.0 / k1 + 1.0 / (3.0 * kc))
        K = np.diag(self.omega_j ** 2) + k1 * np.outer(b_vert, b_vert) + k2 * np.outer(b_lat, b_lat)
        M = np.eye(self.n) + P["mu"] * np.outer(phc, phc) + P["J"] * np.outer(dphc, dphc)
        c1 = np.sqrt(k1) / 10 ** P["log_Qc"]
        gap = h * np.cos(self.tilt) + (xi_c - self.xq) * np.sin(self.tilt)
        gap = np.clip(gap, 0.15 * h, None)
        q = np.where(self.xq <= xi_c, 1.0, self.ow) / gap ** 2
        trap = getattr(np, "trapezoid", None) or np.trapz
        fe = trap(q[None] * self.phi_q, self.xq, axis=1)
        fe /= np.max(np.abs(fe))
        return K, M, c1, k1, b_vert, fe

    def positions_xi(self, P, x=None):
        x = self.x if x is None else np.asarray(x, float)
        return (x - P["c_um"]) / P["L_um"]

    def band_maps(self, P, zeta_list, x=None, dlf=None):
        """Unit-drive piezo and electrostatic maps per band, each (npos, nf) complex,
        normalised by their quasistatic free-end response."""
        K, M, c1, k1, bv, fe = self._matrices(P)
        xi = np.clip(self.positions_xi(P, x), 0, 1)
        _, phx, _ = beam_modes(xi, self.n)
        _, ph1, _ = beam_modes([1.0], self.n)
        # quasistatic normalisation (omega -> 0)
        qp0 = np.linalg.solve(K, k1 * bv); qe0 = np.linalg.solve(K, fe)
        np0, ne0 = abs(qp0 @ ph1[:, 0]), abs(qe0 @ ph1[:, 0])
        out = []
        dlf = np.zeros(len(self.bands)) if dlf is None else dlf
        for b, zeta, d in zip(self.bands, zeta_list, dlf):
            w = b.f_Hz * (1.0 + d) / self.fs
            C = c1 * np.outer(bv, bv) + np.diag(2 * zeta * self.omega_j)
            A = K[None] + 1j * w[:, None, None] * C[None] - w[:, None, None] ** 2 * M[None]
            Fp = (k1 + 1j * w[:, None] * c1) * bv[None, :]
            Fe = np.broadcast_to(fe, (w.size, self.n))
            sol = np.linalg.solve(A, np.stack([Fp, Fe], -1))
            zp = (sol[..., 0] @ phx).T / np0
            ze = (sol[..., 1] @ phx).T / ne0
            out.append((zp, ze))
        return out

    def eig_freqs_Hz(self, P, nmax=8):
        K, M, *_ = self._matrices(P)
        ev = np.sort(np.real(np.linalg.eigvals(np.linalg.solve(M, K))))
        return np.sqrt(np.clip(ev, 0, None))[:nmax] * self.fs

    # ------------------------------------------------------------------ fitting
    def unpack(self, v, spec: Spec, n_bands):
        P = {k: spec.fixed.get(k, SHARED[k][0]) for k in SHARED}
        i = 0
        for k in spec.free:
            P[k] = v[i]; i += 1
        nz = 1 if spec.shared_damping else n_bands
        z = 10 ** np.asarray(v[i:i + nz]); i += nz
        e = np.asarray(v[i:i + nz]); i += nz
        if spec.shared_damping:
            z, e = np.repeat(z, n_bands), np.repeat(e, n_bands)
        d = np.asarray(v[i:i + n_bands]) if spec.band_freq else np.zeros(n_bands)
        return P, z, e, d

    def pack0(self, spec: Spec, n_bands, start=None):
        start = start or {}
        v = [start.get(k, spec.fixed.get(k, SHARED[k][0])) for k in spec.free]
        lo = [spec.bounds.get(k, SHARED[k][1:])[0] for k in spec.free]
        hi = [spec.bounds.get(k, SHARED[k][1:])[1] for k in spec.free]
        nz = 1 if spec.shared_damping else n_bands
        for key in ("log_zeta", "eps"):
            d0 = list(start.get(key, [PER_BAND[key][0]] * n_bands))
            v += [float(np.mean(d0))] if spec.shared_damping else d0[:nz]
            lo += [PER_BAND[key][1]] * nz; hi += [PER_BAND[key][2]] * nz
        if spec.band_freq:
            v += list(start.get("dlf", [0.0] * n_bands))
            lo += [PER_BAND["dlf"][1]] * n_bands; hi += [PER_BAND["dlf"][2]] * n_bands
        v = np.clip(np.array(v, float), np.array(lo) + 1e-9, np.array(hi) - 1e-9)
        return v, (np.array(lo), np.array(hi))

    def predict(self, P, zetas, eps, sel, x=None, gains=None, dlf=None):
        """Maps on positions x (default all) with complex gains solved on the `sel` rows."""
        maps_sel = self.band_maps(P, zetas, x=self.x[sel], dlf=dlf)
        maps_x = self.band_maps(P, zetas, x=x, dlf=dlf)
        out, gs = [], []
        for bi, (b, (zp, ze), (zpx, zex)) in enumerate(zip(self.bands, maps_sel, maps_x)):
            m = zp + eps[bi] * ze
            D = b.Z[sel]
            g = np.vdot(m.ravel(), D.ravel()) / np.vdot(m.ravel(), m.ravel()) if gains is None else gains[bi]
            gs.append(g)
            out.append(g * (zpx + eps[bi] * zex))
        return out, gs

    def residual(self, v, spec, sel, band_idx):
        nb = len(band_idx)
        P, z, e, dl = self.unpack(v, spec, nb)
        if P["L_um"] < self.x.max() - P["c_um"] - 0.01:
            return np.full(sum(2 * len(sel) * self.bands[i].f_Hz.size for i in band_idx), 10.0)
        sub = [self.bands[i] for i in band_idx]
        saved = self.bands; self.bands = sub
        try:
            maps = self.band_maps(P, z, x=self.x[sel], dlf=dl)
        finally:
            self.bands = saved
        r = []
        for (zp, ze), b, ei in zip(maps, sub, e):
            m = zp + ei * ze
            D = b.Z[sel]
            g = np.vdot(m.ravel(), D.ravel()) / np.vdot(m.ravel(), m.ravel())
            d = (g * m - D).ravel() / np.sqrt(np.mean(np.abs(D) ** 2))
            r += [d.real, d.imag]
        return np.concatenate(r)

    def fit(self, spec: Spec, sel, band_idx=None, starts=(None,), max_nfev=200, polish=True):
        band_idx = list(range(len(self.bands))) if band_idx is None else list(band_idx)
        best = None
        for st in starts:
            v0, bnds = self.pack0(spec, len(band_idx), st)
            try:
                r = least_squares(self.residual, v0, bounds=bnds, args=(spec, sel, band_idx),
                                  x_scale="jac", max_nfev=max_nfev)
            except Exception as exc:  # pragma: no cover
                print("start failed:", exc); continue
            if best is None or r.cost < best.cost:
                best = r
        P, z, e, dl = self.unpack(best.x, spec, len(band_idx))
        if polish and not spec.shared_damping:
            P, z, e, dl, best = self._polish(best, spec, sel, band_idx, P, z, e, dl, max_nfev)
        return dict(P=P, zeta=z, eps=e, dlf=dl, cost=float(best.cost), v=best.x, spec=spec,
                    band_idx=band_idx, nfev=int(best.nfev))

    def _polish(self, best, spec, sel, band_idx, P, z, e, dl, max_nfev):
        """Per-band nuisance re-fit (zeta, eps, dlf) from several starts with the geometry held,
        then one joint re-fit. Guards against a band settling in a wrong local minimum."""
        fixed = {k: P[k] for k in SHARED}
        z, e, dl = np.array(z, float), np.array(e, float), np.array(dl, float)
        for j, bi in enumerate(band_idx):
            sp = Spec(free=(), fixed=fixed, band_freq=spec.band_freq)
            cands = []
            for e0 in (e[j], 0.0, 0.5, -0.5):
                for dd in ((0.0, -0.003, 0.003) if spec.band_freq else (0.0,)):
                    st = {"log_zeta": [np.log10(z[j])], "eps": [e0]}
                    if spec.band_freq:
                        st["dlf"] = [dl[j] + dd]
                    v0, bnds = self.pack0(sp, 1, st)
                    r = least_squares(self.residual, v0, bounds=bnds, args=(sp, sel, [bi]),
                                      x_scale="jac", max_nfev=100)
                    cands.append(r)
            r = min(cands, key=lambda q: q.cost)
            _, zj, ej, dj = self.unpack(r.x, sp, 1)
            z[j], e[j], dl[j] = zj[0], ej[0], dj[0]
        st = {**{k: P[k] for k in spec.free}, "log_zeta": list(np.log10(z)), "eps": list(e), "dlf": list(dl)}
        v0, bnds = self.pack0(spec, len(band_idx), st)
        r = least_squares(self.residual, v0, bounds=bnds, args=(spec, sel, band_idx),
                          x_scale="jac", max_nfev=max_nfev)
        if r.cost < best.cost:
            best = r
            P, z, e, dl = self.unpack(r.x, spec, len(band_idx))
        return P, z, e, dl, best

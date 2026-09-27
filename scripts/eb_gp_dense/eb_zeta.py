"""EB model with the intrinsic modal damping zeta as a sixth inferred parameter.

The repo's `EBForwardModel.response` hard-codes `C += diag(2 * 0.002 * omega_j)`
(zeta = 0.002, Q_intrinsic = 250). On the PPP-CONTAu that is the binding constraint:
with alpha at the pinned limit the contact dashpot has no authority and the model
cannot make a Q ~ 120 line. Here zeta is inferred, in log10, alongside the five
existing parameters. Everything else is inherited.
"""
import numpy as np

from activemodemap.forward_model import EBForwardModel
from eb_gp_test import SingleDomainPosterior, SingleDomainHybrid, dns_um_from_end, subset_indices
from prep import OM1


class EBZetaModel(EBForwardModel):
    PARAM_NAMES = ["log_alpha", "log_kcone", "log_Q", "log_eps", "log_A0", "log_zeta"]

    def response(self, theta):
        log_alpha, log_kcone, log_Q, log_eps, log_A0, log_zeta = theta
        alpha, kcone_r = 10.0 ** log_alpha, 10.0 ** log_kcone
        Q, eps, A0 = 10.0 ** log_Q, 10.0 ** log_eps, 10.0 ** log_A0
        zeta = 10.0 ** log_zeta

        k1 = alpha * self.k_lever_scaled
        kcone = kcone_r * self.k_lever_scaled
        k2 = 1.0 / (1.0 / k1 + 1.0 / kcone)
        c1 = np.sqrt(k1) / Q

        n, nf = self.n_modes, self.omega.size
        K = np.diag(self.omega_j ** 2).astype(complex)
        K += k1 * np.outer(self.b_vert, self.b_vert)
        K += k2 * np.outer(self.b_lat, self.b_lat)
        M = np.eye(n) + self.geom.tip_mass_ratio * np.outer(self.phi_c, self.phi_c)
        C = c1 * np.outer(self.b_vert, self.b_vert)
        C += np.diag(2 * zeta * self.omega_j)          # <-- the only change

        w = self.omega[:, None, None]
        A = K[None] + 1j * w * C[None] - w ** 2 * M[None]
        F_p = ((k1 + 1j * self.omega[:, None] * c1) * self.b_vert[None, :])
        F_e = np.broadcast_to(self.f_elec_modal, (nf, n))
        rhs = np.stack([F_p, F_e], axis=-1)
        sol = np.linalg.solve(A, rhs)
        z_p = sol[..., 0] @ self.phi_grid
        z_e = sol[..., 1] @ self.phi_grid
        z_p = z_p / np.abs(z_p[0, -1])
        z_e = z_e / np.abs(z_e[0, -1])
        return {"piezo": z_p, "elec": z_e, "A0": A0, "eps": eps}

    def spots_um_from_end(self, theta):
        return super().spots_um_from_end(theta)


class ZetaPosterior(SingleDomainPosterior):
    """Coarse scan extended over zeta, then the inherited polish."""

    def _coarse_scan(self, data, top_k: int = 3):
        a_grid = np.linspace(self.lo[0] + 0.15, self.hi[0] - 0.1, 12)
        kc_grid = np.linspace(self.lo[1] + 0.15, self.hi[1] - 0.15, 5)
        q_grid = np.linspace(self.lo[2] + 0.15, self.hi[2] - 0.15, 4)
        z_grid = np.linspace(self.lo[5] + 0.1, self.hi[5] - 0.1, 6)
        A0_grid = np.linspace(self.lo[4] + 0.1, self.hi[4] - 0.1, 13)
        eps_grid = np.linspace(self.lo[3] + 0.1, self.hi[3] - 0.1, 9)
        col_ix = np.array([int(np.argmin(np.abs(self.model.xi - d["x"]))) for d in data])
        meas = np.array([np.abs(d["plus"]) for d in data])
        log_meas = np.log(meas + self.sigma)
        mid = 0.5 * (self.lo + self.hi)
        A0s, epss = 10.0 ** A0_grid, 10.0 ** eps_grid
        results = []
        for a in a_grid:
            for kc in kc_grid:
                for q in q_grid:
                    for z in z_grid:
                        th = mid.copy()
                        th[0], th[1], th[2], th[5] = a, kc, q, z
                        resp = self.model.response(th)
                        zp = self._blur(resp["piezo"])[:, col_ix].T
                        ze = self._blur(resp["elec"])[:, col_ix].T
                        pred = (A0s[:, None, None, None]
                                * (zp[None, None] + epss[None, :, None, None] * ze[None, None]))
                        cost = ((np.log(np.abs(pred) + self.sigma) - log_meas[None, None]) ** 2
                                ).sum(axis=(2, 3))
                        kb = np.unravel_index(np.argmin(cost), cost.shape)
                        th = th.copy(); th[4] = A0_grid[kb[0]]; th[3] = eps_grid[kb[1]]
                        results.append((float(cost[kb]), th))
        results.sort(key=lambda t: t[0])
        starts, seen = [], []
        for cost, th in results:
            if all(np.linalg.norm(th[[0, 1, 2, 5]] - s[[0, 1, 2, 5]]) > 0.4 for s in seen):
                starts.append(th); seen.append(th)
            if len(starts) >= top_k:
                break
        return starts


def build_zeta_model(L_um=445.0, f0_hz=13.649e3, k_lever=0.4057, tip_setback_um=10.0,
                     f_lo=50.2e3, f_hi=79.8e3, nf=601):
    from activemodemap.forward_model import ProbeGeometry
    geom = ProbeGeometry(name='PPP-CONTAu', f0_hz=f0_hz, k_lever=k_lever, L_um=L_um,
                         tip_setback_um=tip_setback_um, tip_height_um=12.5,
                         tilt_deg=11.0, tip_mass_ratio=0.0018)
    m = EBZetaModel(geom=geom, n_modes=12, nx=241, nf=nf,
                    omega_lo=OM1 * f_lo / f0_hz, omega_hi=OM1 * f_hi / f0_hz)
    # same widened box as the fixed-zeta test, plus zeta: 3e-4 .. 0.1 (Q_int 1700 .. 5)
    m.PRIOR_LO = np.array([1.0, 1.0, 0.5, -4.0, -4.0, -3.5])
    m.PRIOR_HI = np.array([5.0, 4.0, 2.5, 1.0, 1.0, -1.0])
    return m

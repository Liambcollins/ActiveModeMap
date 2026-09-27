"""zeta-fitted EB plus a global instrument phase as a seventh nuisance parameter.

`tune_to_complex` builds Z = amp * exp(i * phase_deg) from the raw Igor phase, and
neither the forward model nor the inference has any phase parameter. On the virtual
instrument the convention matches by construction; on the Cypher the lock-in phase
reference (panel PhaseOffset ~ 149 deg) enters the data directly. The complex-polish
stage then minimises a residual no parameter can remove, and shrinks A0 to do it.
"""
import numpy as np
from eb_zeta import EBZetaModel, ZetaPosterior, build_zeta_model
from eb_gp_test import SingleDomainHybrid
from activemodemap.hybrid import DiscrepancyGP


class EBPhaseModel(EBZetaModel):
    PARAM_NAMES = EBZetaModel.PARAM_NAMES + ["phase_deg"]

    def response(self, theta):
        r = super().response(np.asarray(theta)[:6])
        r["phase"] = np.exp(1j * np.deg2rad(theta[6]))
        return r


class PhasePosterior(ZetaPosterior):
    def _pred_columns(self, theta, data):
        resp = self.model.response(theta)
        zp, ze = self._blur(resp["piezo"]), self._blur(resp["elec"])
        g = resp["A0"] * resp["phase"]
        return [(g * (zp[:, int(np.argmin(np.abs(self.model.xi - d["x"])))]
                      + resp["eps"] * ze[:, int(np.argmin(np.abs(self.model.xi - d["x"])))]),
                 d["plus"]) for d in data]

    def _coarse_scan(self, data, top_k=3):
        # phase does not enter the log-amplitude cost; seed it analytically per start
        starts = super()._coarse_scan(data, top_k)
        out = []
        for th in starts:
            th = np.asarray(th, float).copy()
            th[6] = 0.0
            pred = np.concatenate([p for p, _ in self._pred_columns(th, data)])
            meas = np.concatenate([m for _, m in self._pred_columns(th, data)])
            th[6] = np.degrees(np.angle(np.vdot(pred, meas)))
            out.append(th)
        return out


class PhaseHybrid(SingleDomainHybrid):
    def update(self, data):
        m, post = self.model, self.post
        resp = m.response(post.theta_map)
        zp, ze = post._blur(resp["piezo"]), post._blur(resp["elec"])
        pred = resp["A0"] * resp["phase"] * (zp + resp["eps"] * ze)
        xs, rows = [], []
        for d in data:
            i = int(np.argmin(np.abs(m.xi - d["x"])))
            xs.append(m.xi[i]); rows.append((d["plus"] - pred[:, i]) / post.sigma)
        self.gp = DiscrepancyGP(m).fit(np.array(xs), np.array(rows))
        self._pred = pred
        return self


def build_phase_model(**kw):
    from activemodemap.forward_model import ProbeGeometry
    from prep import OM1
    L = kw.get('L_um', 445.0); f0 = 13.649e3
    geom = ProbeGeometry(name='PPP-CONTAu', f0_hz=f0, k_lever=0.4057, L_um=L,
                         tip_setback_um=kw.get('tip_setback_um', 10.0), tip_height_um=12.5,
                         tilt_deg=11.0, tip_mass_ratio=0.0018)
    m = EBPhaseModel(geom=geom, n_modes=12, nx=241, nf=601,
                     omega_lo=OM1 * 50.2e3 / f0, omega_hi=OM1 * 79.8e3 / f0)
    m.PRIOR_LO = np.array([1.0, 1.0, 0.5, -4.0, -4.0, -3.5, -400.0])
    m.PRIOR_HI = np.array([5.0, 4.0, 2.5, 1.0, 1.0, -1.0, 400.0])
    return m

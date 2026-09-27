"""EB + GP-discrepancy recovery tested against the 2026-09-17 dense sweep.

The dense sweep is SINGLE domain at a single bias, while `PhysicsPosterior` and
`HybridSurrogate` are written for a domain pair (plus/minus). The two subclasses
below use only the measured channel; everything else — the staged coarse-scan →
log-amplitude → complex-polish fit, the Laplace covariance, the discrepancy GP
with its tip-weighted prior — is the repo's own code, untouched.

With one domain the piezo and electrostatic pathways are not separable, so eps
is a nuisance parameter here: the fit sees only A0*(piezo + eps*elec).
"""
import numpy as np

from activemodemap.forward_model import EBForwardModel, ProbeGeometry
from activemodemap.inference import PhysicsPosterior
from activemodemap.hybrid import HybridSurrogate, DiscrepancyGP, _signed_zero_crossing
from activemodemap.lowrank import reconstruct_map, dns_from_map, spatial_null

from prep import load_dense, to_model_grid, estimate_sigma, OM1


# --------------------------------------------------------------------------- #
#  single-domain variants                                                     #
# --------------------------------------------------------------------------- #
class SingleDomainPosterior(PhysicsPosterior):
    def _pred_columns(self, theta, data):
        resp = self.model.response(theta)
        zp, ze = self._blur(resp["piezo"]), self._blur(resp["elec"])
        out = []
        for d in data:
            i = int(np.argmin(np.abs(self.model.xi - d["x"])))
            out.append((resp["A0"] * (zp[:, i] + resp["eps"] * ze[:, i]), d["plus"]))
        return out

    def _coarse_scan(self, data, top_k: int = 3):
        """The repo's coarse scan with the domain pair collapsed to one channel."""
        a_grid = np.linspace(self.lo[0] + 0.15, self.hi[0] - 0.1, 16)
        kc_grid = np.linspace(self.lo[1] + 0.15, self.hi[1] - 0.15, 7)
        q_grid = np.linspace(self.lo[2] + 0.15, self.hi[2] - 0.15, 5)
        A0_grid = np.linspace(self.lo[4] + 0.1, self.hi[4] - 0.1, 13)
        eps_grid = np.linspace(self.lo[3] + 0.1, self.hi[3] - 0.1, 9)
        col_ix = np.array([int(np.argmin(np.abs(self.model.xi - d["x"])))
                           for d in data])
        meas = np.array([np.abs(d["plus"]) for d in data])
        log_meas = np.log(meas + self.sigma)

        mid = 0.5 * (self.lo + self.hi)
        results = []
        A0s, epss = 10.0 ** A0_grid, 10.0 ** eps_grid
        for a in a_grid:
            for kc in kc_grid:
                for q in q_grid:
                    th = mid.copy()
                    th[0], th[1], th[2] = a, kc, q
                    resp = self.model.response(th)
                    zp = self._blur(resp["piezo"])[:, col_ix].T
                    ze = self._blur(resp["elec"])[:, col_ix].T
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
        starts, seen = [], []
        for cost, th in results:
            if all(np.linalg.norm(th[:3] - s[:3]) > 0.4 for s in seen):
                starts.append(th)
                seen.append(th)
            if len(starts) >= top_k:
                break
        return starts


class SingleDomainHybrid(HybridSurrogate):
    def update(self, data):
        m, post = self.model, self.post
        resp = m.response(post.theta_map)
        zp, ze = post._blur(resp["piezo"]), post._blur(resp["elec"])
        pred = resp["A0"] * (zp + resp["eps"] * ze)          # (nf, nx)
        xs, rows = [], []
        for d in data:
            i = int(np.argmin(np.abs(m.xi - d["x"])))
            xs.append(m.xi[i])
            rows.append((d["plus"] - pred[:, i]) / post.sigma)
        self.gp = DiscrepancyGP(m).fit(np.array(xs), np.array(rows))
        self._pred = pred
        return self

    def corrected_maps(self, xq=None, sample_rng=None):
        m = self.model
        xq = m.xi if xq is None else xq
        mean, var = self.gp.predict(xq)
        if sample_rng is not None:
            mean = mean + np.sqrt(var) * (
                sample_rng.standard_normal(mean.shape)
                + 1j * sample_rng.standard_normal(mean.shape)) / np.sqrt(2)
        return self._pred + mean.T * self.post.sigma          # (nf, nx)


# --------------------------------------------------------------------------- #
#  helpers                                                                    #
# --------------------------------------------------------------------------- #
def dns_um_from_end(x_um, Zmap_pos_by_freq, freq_Hz, L_um, guess_um=None):
    """D-NS from a (npos, nfreq) map, via the package's own estimators.

    Tries the antiresonance branch crossing first (`dns_from_map`), which is what
    the package prefers, and falls back to the signed zero crossing at the
    resonance row.
    """
    A = np.abs(Zmap_pos_by_freq)
    ires = int(np.argmax(A[np.argmin(np.abs(x_um - 0.5 * (x_um[0] + x_um[-1])))]))
    try:
        x_null = dns_from_map(x_um, Zmap_pos_by_freq, freq_Hz, ires,
                              guess_um=guess_um)
    except Exception:
        x_null = None
    if x_null is None or not np.isfinite(x_null):
        x_null = spatial_null(x_um, Zmap_pos_by_freq, freq_Hz, ires,
                              guess_um=guess_um)
    return (L_um - float(x_null)) if np.isfinite(x_null) else np.nan


def build_model(L_um=445.0, f0_hz=13.649e3, k_lever=0.4057,
                tip_setback_um=10.0, f_lo=50.2e3, f_hi=79.8e3, nf=601):
    geom = ProbeGeometry(name='PPP-CONT-AU', f0_hz=f0_hz, k_lever=k_lever,
                         L_um=L_um, tip_setback_um=tip_setback_um,
                         tip_height_um=12.0, tilt_deg=11.0, tip_mass_ratio=0.01)
    model = EBForwardModel(geom=geom, n_modes=12, nx=241, nf=nf,
                           omega_lo=OM1 * f_lo / f0_hz, omega_hi=OM1 * f_hi / f0_hz)
    # The shipped prior box is sized for a stiff PFM probe measured against its
    # quasistatic level. Three of the five parameters do not fit this case:
    #   log_Q   capped at 2.0 (Q <= 100); this lever tunes at Q ~ 130-150.
    #   log_A0  box [-0.5, 0.5] assumes data scaled to the QUASISTATIC response.
    #           These spectra are normalised to the resonance peak, which is ~Q
    #           times larger, so the needed A0 is ~1/Q ~ 1e-2 — off the box.
    #   log_eps box floor -1 (eps >= 0.1). This is a shaker-driven contact
    #           resonance at 0 V with no AC bias, so there is no electrostatic
    #           pathway at all and the honest value is eps -> 0.
    model.PRIOR_LO = np.array([1.0, 1.0, 0.5, -4.0, -4.0])
    model.PRIOR_HI = np.array([5.0, 4.0, 2.5, 1.0, 1.0])
    return model


def subset_indices(n_pos, n_take):
    """Equispaced positions across the measured span, endpoints included."""
    return np.unique(np.round(np.linspace(0, n_pos - 1, n_take)).astype(int))


def run(n_list=(4, 5, 6, 8, 10, 14, 20), seed=0, verbose=True):
    dense = load_dense()
    model = build_model(L_um=dense['L_um'])
    xi, Zg, scale = to_model_grid(dense, model)
    sigma = estimate_sigma(Zg)
    f_model = model.omega / OM1 * model.geom.f0_hz
    x_um = dense['x_um']
    L = dense['L_um']

    # ---- truth from all 75 measured positions -----------------------------
    truth_dns = dns_um_from_end(x_um, Zg, f_model, L)
    A_true = np.abs(Zg)
    norm = A_true.max()

    if verbose:
        print(f"dense truth: {x_um.size} positions, D-NS {truth_dns:.2f} um from the "
              f"free end; sigma/max = {sigma:.2e}")

    rows = []
    for n in n_list:
        idx = subset_indices(x_um.size, n)
        data = [{"x": xi[i], "plus": Zg[i]} for i in idx]

        post = SingleDomainPosterior(model, sigma=sigma,
                                     rng=np.random.default_rng(seed))
        post.fit(data)
        # --- EB only ---
        resp = model.response(post.theta_map)
        eb_full = resp["A0"] * (post._blur(resp["piezo"])
                                + resp["eps"] * post._blur(resp["elec"]))   # (nf,nx)
        cols = [int(np.argmin(np.abs(model.xi - v))) for v in xi]
        eb_at = eb_full[:, cols].T                                          # (npos,nf)

        # --- EB + GP discrepancy ---
        sur = SingleDomainHybrid(post).update(data)
        hy_full = sur.corrected_maps()
        hy_at = hy_full[:, cols].T

        # --- model-light low-rank at the same positions ---
        rank = min(4, len(idx) - 2) if len(idx) >= 6 else max(2, len(idx) - 1)
        lr = reconstruct_map(x_um, idx, Zg[idx], rank=rank)
        lr_at = lr['Zrec']

        def rmse(M):          # amplitude error, the metric loop.py reports
            return float(np.sqrt(np.mean((np.abs(M) - A_true) ** 2)) / norm)

        def crmse(M):         # complex error — what the discrepancy GP acts on
            return float(np.sqrt(np.mean(np.abs(M - Zg) ** 2)) / norm)

        r = dict(n=len(idx), rank=rank,
                 rmse_eb=rmse(eb_at), rmse_hy=rmse(hy_at), rmse_lr=rmse(lr_at),
                 crmse_eb=crmse(eb_at), crmse_hy=crmse(hy_at), crmse_lr=crmse(lr_at),
                 dns_eb=dns_um_from_end(x_um, eb_at, f_model, L, guess_um=L - truth_dns),
                 dns_hy=dns_um_from_end(x_um, hy_at, f_model, L, guess_um=L - truth_dns),
                 dns_lr=dns_um_from_end(x_um, lr_at, f_model, L, guess_um=L - truth_dns),
                 theta=post.theta_map.copy(),
                 cr_kHz=float(model.contact_resonance(resp) / OM1 * model.geom.f0_hz / 1e3))
        rows.append(r)
        if verbose:
            print(f"n={r['n']:3d} rk{rank}  cplxRMSE EB {r['crmse_eb']:.4f} "
                  f"EB+GP {r['crmse_hy']:.4f} LR {r['crmse_lr']:.4f} | "
                  f"ampRMSE {r['rmse_eb']:.4f}/{r['rmse_hy']:.4f}/{r['rmse_lr']:.4f} | "
                  f"D-NS {r['dns_eb']:6.2f}/{r['dns_hy']:6.2f}/{r['dns_lr']:6.2f} "
                  f"(truth {truth_dns:.2f})")
    return dict(rows=rows, truth_dns=truth_dns, x_um=x_um, f_model=f_model,
                Zg=Zg, sigma=sigma, model=model, L=L)


if __name__ == '__main__':
    run()

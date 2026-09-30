"""Synthetic identifiability test: can the lever geometry be calibrated WITHOUT a dense map?

Question (2026-09-29): the lever constants (L, clamp offset c, tip setback, tip height, cone
stiffness, tip mass mu, rotary inertia J) were fixed once from the R2 dense map. Could a free-air
walk (free resonances + free mode shapes at the same sparse positions) and/or a short load ladder
replace it?

Truth = the calibrated stiff lever (results/lever_calibration_stiff_mass.json). Synthetic data are
generated from the same JointEB physics with realistic noise, the geometry is refitted under each
data scenario from perturbed starts, and the score that matters is the TRANSFER error of E(x) on the
lever interior: a new contact state is generated, only the contact parameters are refitted with the
recovered geometry frozen (what pick_image_position.py does), and E_fit/E_true - 1 is reported.

Scenarios
  S1 sparse contact only        N=8 positions, CR1-CR3, one load           (the known degenerate case)
  S2 S1 + free-air walk         free f1..f3 and |phi_j| at the same 8 positions
  S3 S2 + load ladder           three loads (k*/k x 0.5, 1, 2), same geometry
  S4 dense contact only         N=30 positions, one load                    (what the paper did)

    python tools/sim_calib_freeair.py [--reps 3] [--starts 4] [--out results/sim_calib_freeair.json]
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fmmpaper import jointeb as J  # noqa: E402

GEOM = ["setback_um", "tip_h_um", "log_kcone", "c_um", "L_um", "mu", "J"]
J.SHARED["f0s"] = (1.0, 0.8, 1.25)      # shared free-beam frequency scale (see _scale)
CAL = json.load(open(Path(__file__).resolve().parents[1] / "results" / "lever_calibration_stiff_mass.json"))
P_TRUE = dict(CAL["P"]); P_TRUE["J"] = 2e-4                  # give J a non-trivial true value so its recovery is tested
F0 = CAL["f0_hz"]; NM = 12
ZETA_TRUE = [0.0028, 0.0012, 0.0005]; EPS_TRUE = [-0.01, -0.05, -0.07]
X_SPARSE = np.round(np.linspace(78.0, 232.0, 8), 1)         # R2 survey span, stage um
X_DENSE = np.round(np.linspace(78.0, 232.0, 30), 1)
X_EVAL = np.arange(90.0, 226.0, 1.0)                          # lever interior for the E score (xi ~ 0.27-0.89)
SNR_AMP = 0.03                                                # complex noise, fraction of band RMS
FREE_F_NOISE, FREE_A_NOISE = 5e-4, 0.03                       # free-air: frequency, shape amplitude
LOADS = {"S1": [1.0], "S2": [1.0], "S3": [0.5, 1.0, 2.0], "S4": [1.0]}
rng = np.random.default_rng(0)
MISMATCH = 0.0        # model error: truth evaluated with per-mode frequency offsets (0, -1, +2 %) x MISMATCH on CR2/CR3 (Timoshenko/non-uniform lever stand-in)
DLF = np.array([0.0, -0.01, 0.02])


def _scale(je, P):
    """Optional shared free-beam frequency scale f0s (the thermal f0 includes the tip mass; f0s frees it)."""
    je.fs = F0 / J.LAMBDAS[0] ** 2 * P.get("f0s", 1.0)


def eig(P, nmax=3):
    je = J.JointEB(X_SPARSE, [], F0, n_modes=NM); _scale(je, P)
    return je.eig_freqs_Hz(P, nmax)


def bands_for(P):
    f = eig(P, 3)
    return [np.linspace(0.85 * fn, 1.15 * fn, 120) for fn in f]


def make_je(x, P_bands):
    return J.JointEB(x, [J.Band(f"CR{i+1}", f, None) for i, f in enumerate(P_bands)], F0, n_modes=NM)


def contact_maps(je, P, zetas, eps, x, dlf=None):
    _scale(je, P)
    out = je.band_maps(P, zetas, x=x, dlf=dlf)
    return [zp + e * ze for (zp, ze), e in zip(out, eps)]


def free_air(P, x):
    """Free resonances and |mode shape| at x from the same matrices with the contact removed."""
    Q = dict(P, log_alpha=-6.0, log_kcone=-6.0)
    je = J.JointEB(x, [], F0, n_modes=NM); _scale(je, Q)
    K, M, *_ = je._matrices(Q)
    ev, V = np.linalg.eig(np.linalg.solve(M, K)); o = np.argsort(np.real(ev)); ev, V = np.real(ev[o]), np.real(V[:, o])
    f = np.sqrt(np.clip(ev[:3], 0, None)) * je.fs
    _, phx, _ = J.beam_modes(np.clip(je.positions_xi(Q, x), 0, 1), NM)
    shapes = np.abs(V[:, :3].T @ phx)                         # (3, npos)
    shapes /= np.sqrt(np.mean(shapes ** 2, axis=1, keepdims=True))
    return f, shapes


def synth_contact(x, P, load_factors, seed):
    r = np.random.default_rng(seed); caps = []
    for lf in load_factors:
        Pl = dict(P, log_alpha=P["log_alpha"] + np.log10(lf))
        fb = bands_for(Pl); je = make_je(x, fb)
        Z = contact_maps(je, Pl, ZETA_TRUE, EPS_TRUE, x, dlf=MISMATCH * DLF)
        Zn = []
        for z in Z:
            s = SNR_AMP * np.sqrt(np.mean(np.abs(z) ** 2))
            Zn.append(z + s * (r.standard_normal(z.shape) + 1j * r.standard_normal(z.shape)) / np.sqrt(2))
        caps.append(dict(lf=lf, fb=fb, Z=Zn, sigma=[SNR_AMP * np.sqrt(np.mean(np.abs(z) ** 2)) for z in Z]))
    return caps


def synth_free(x, P, seed):
    r = np.random.default_rng(seed + 100); f, sh = free_air(P, x)
    return f * (1 + FREE_F_NOISE * r.standard_normal(3)), sh + FREE_A_NOISE * r.standard_normal(sh.shape)


# ---------------------------------------------------------------- joint residual
class Problem:
    def __init__(self, x, caps, free_data, geom_free, P_fix):
        self.x, self.caps, self.free_data, self.geom_free, self.P_fix = x, caps, free_data, geom_free, dict(P_fix)
        self.jes = [make_je(x, c["fb"]) for c in caps]
        self.nb = 3

    def unpack(self, v):
        P = dict(self.P_fix); i = 0
        for k in self.geom_free: P[k] = v[i]; i += 1
        per = []
        for _ in self.caps:
            la, lq = v[i], v[i + 1]; i += 2
            lz = v[i:i + self.nb]; i += self.nb
            ep = v[i:i + self.nb]; i += self.nb
            per.append((la, lq, 10 ** np.asarray(lz), np.asarray(ep)))
        return P, per

    def pack(self, P0, per0):
        v, lo, hi = [], [], []
        for k in self.geom_free:
            v.append(P0[k]); lo.append(J.SHARED[k][1]); hi.append(J.SHARED[k][2])
        for la, lq, z, e in per0:
            v += [la, lq]; lo += [1.0, 0.3]; hi += [5.0, 4.0]
            v += list(np.log10(z)); lo += [-4.0] * self.nb; hi += [-1.0] * self.nb
            v += list(e); lo += [-5.0] * self.nb; hi += [5.0] * self.nb
        return np.array(v, float), (np.array(lo), np.array(hi))

    def residual(self, v):
        P, per = self.unpack(v); res = []
        for je, cap, (la, lq, z, e) in zip(self.jes, self.caps, per):
            Pl = dict(P, log_alpha=la, log_Qc=lq)
            try:
                m = contact_maps(je, Pl, z, e, self.x)
            except np.linalg.LinAlgError:
                return np.full(self._n, 1e3)
            for mb, D, s in zip(m, cap["Z"], cap["sigma"]):
                g = np.vdot(mb.ravel(), D.ravel()) / max(np.vdot(mb.ravel(), mb.ravel()).real, 1e-30)
                r = (g * mb - D) / s
                res += [r.real.ravel(), r.imag.ravel()]
        if self.free_data is not None:
            f_meas, sh_meas = self.free_data; f, sh = free_air(P, self.x)
            res.append((f - f_meas) / (FREE_F_NOISE * f_meas))
            res.append(((sh - sh_meas) / FREE_A_NOISE).ravel())
        out = np.concatenate(res); self._n = out.size
        return out


def fit(prob, P0, per0, max_nfev=400):
    v0, (lo, hi) = prob.pack(P0, per0); v0 = np.clip(v0, lo + 1e-9, hi - 1e-9)
    prob._n = prob.residual(v0).size
    r = least_squares(prob.residual, v0, bounds=(lo, hi), max_nfev=max_nfev, x_scale="jac")
    P, per = prob.unpack(r.x)
    return P, per, 2 * r.cost / prob._n


def E_profile(P, alpha, Qc, x):
    """E(x) = |Z_CR1|peak / |Z_QS| of the pure piezo response (two-domain P) for a contact state."""
    Pl = dict(P, log_alpha=alpha, log_Qc=Qc)
    f1 = eig(Pl, 1)[0]; fb = [np.linspace(0.9 * f1, 1.1 * f1, 200), np.array([30e3])]
    je = J.JointEB(x, [J.Band("CR1", fb[0], None), J.Band("QS", fb[1], None)], F0, n_modes=NM); _scale(je, Pl)
    (zp1, _), (zq, _) = je.band_maps(Pl, [ZETA_TRUE[0], ZETA_TRUE[0]], x=x)
    return np.abs(zp1).max(1) / np.abs(zq)[:, 0]


def transfer_score(P_geom, seed):
    """New contact state; refit contact-only with P_geom frozen from a sparse capture; E error on the interior."""
    P_new = dict(P_TRUE, log_alpha=P_TRUE["log_alpha"] + np.log10(1.4), log_Qc=P_TRUE["log_Qc"] - 0.1, c_um=P_TRUE["c_um"] + 1.0)
    cap = synth_contact(X_SPARSE, P_new, [1.0], seed + 500)
    prob = Problem(X_SPARSE, cap, None, ["c_um"], P_geom)
    per0 = [(P_geom.get("log_alpha", 2.9), P_geom.get("log_Qc", 1.9), np.array([3e-3, 1e-3, 5e-4]), np.zeros(3))]
    Pf, per, _ = fit(prob, dict(P_geom, c_um=P_geom["c_um"]), per0, max_nfev=200)
    la, lq = per[0][0], per[0][1]
    E_fit = E_profile(Pf, la, lq, X_EVAL); E_true = E_profile(P_new, P_new["log_alpha"], P_new["log_Qc"], X_EVAL)
    err = E_fit / E_true - 1
    return dict(median_pct=float(100 * np.median(np.abs(err))), p90_pct=float(100 * np.percentile(np.abs(err), 90)),
                max_pct=float(100 * np.abs(err).max()), f_cr1_fit_kHz=float(eig(dict(Pf, log_alpha=la), 1)[0] / 1e3))


def perturbed_start(r):
    P0 = dict(P_TRUE)
    P0["setback_um"] = float(np.clip(P_TRUE["setback_um"] * r.uniform(0.5, 3.0), 1.1, 39))
    P0["tip_h_um"] = float(np.clip(P_TRUE["tip_h_um"] * r.uniform(0.6, 2.0), 3.1, 24))
    P0["log_kcone"] = P_TRUE["log_kcone"] + r.uniform(-0.7, 0.7)
    P0["c_um"] = P_TRUE["c_um"] + r.uniform(-8, 8); P0["L_um"] = P_TRUE["L_um"] * r.uniform(0.93, 1.07)
    P0["mu"] = float(np.clip(P_TRUE["mu"] * r.uniform(0.3, 3), 0, 0.19)); P0["J"] = float(r.uniform(0, 1e-3))
    return P0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=3); ap.add_argument("--starts", type=int, default=4)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "sim_calib_freeair.json"))
    ap.add_argument("--scen", default="S1,S2,S3,S4")
    ap.add_argument("--mismatch", type=float, default=0.0, help="1.0 = truth has -1/+2 %% CR2/CR3 frequency offsets the EB model cannot produce")
    a = ap.parse_args()
    global MISMATCH; MISMATCH = a.mismatch
    # bounds for L in this stiff frame (SHARED defaults are the soft lever's)
    J.SHARED["L_um"] = (218.0, 150.0, 300.0)
    print("truth:", {k: round(P_TRUE[k], 4) for k in GEOM}, "f_contact kHz", np.round(eig(P_TRUE) / 1e3, 1),
          "f_free kHz", np.round(free_air(P_TRUE, X_SPARSE)[0] / 1e3, 2))
    # ---- sensitivity of E to the setback with alpha refit to hold CR1 (Liam: interior imaging, setback may not matter)
    sens = {}
    f1_true = eig(P_TRUE, 1)[0]; E0 = E_profile(P_TRUE, P_TRUE["log_alpha"], P_TRUE["log_Qc"], X_EVAL)
    for sb in (1.5, 2.7, 6.0, 10.0, 15.0):
        Ps = dict(P_TRUE, setback_um=sb)
        grid = np.linspace(2.0, 4.5, 400); fg = np.array([eig(dict(Ps, log_alpha=g), 1)[0] for g in grid])
        la = float(grid[np.argmin(np.abs(fg - f1_true))])
        E = E_profile(Ps, la, P_TRUE["log_Qc"], X_EVAL); e = E / E0 - 1
        sens[sb] = dict(log_alpha=la, median_pct=float(100 * np.median(np.abs(e))), max_pct=float(100 * np.abs(e).max()))
        print(f"  setback {sb:5.1f} um (alpha refit so CR1 unchanged, k*/k {10**la:.0f}): E interior change median {sens[sb]['median_pct']:.1f} %, max {sens[sb]['max_pct']:.1f} %")
    ideal = transfer_score(P_TRUE, 7)
    print("transfer score with the TRUE geometry (noise floor of the metric):", ideal)

    results = dict(truth={k: P_TRUE[k] for k in GEOM}, setback_sensitivity=sens, ideal_transfer=ideal, scenarios={})
    for sc in a.scen.split(","):
        t0 = time.time(); rows = []
        for rep in range(a.reps):
            x = X_DENSE if sc == "S4" else X_SPARSE
            caps = synth_contact(x, P_TRUE, LOADS[sc], seed=rep)
            free_data = synth_free(x, P_TRUE, seed=rep) if sc in ("S2", "S3") else None
            best = None
            for st in range(a.starts):
                r = np.random.default_rng(1000 * rep + st); P0 = perturbed_start(r)
                per0 = [(P_TRUE["log_alpha"] + np.log10(c["lf"]) + r.uniform(-0.15, 0.15), P_TRUE["log_Qc"] + r.uniform(-0.3, 0.3),
                         np.array([3e-3, 1e-3, 5e-4]), np.zeros(3)) for c in caps]
                prob = Problem(x, caps, free_data, GEOM, P_TRUE)
                try:
                    Pf, per, chi2 = fit(prob, P0, per0)
                except Exception as e:
                    print("   fit failed:", e); continue
                if best is None or chi2 < best[2]: best = (Pf, per, chi2)
            Pf, per, chi2 = best
            ts = transfer_score(Pf, rep)
            rows.append(dict(rep=rep, chi2=chi2, geom={k: Pf[k] for k in GEOM}, transfer=ts))
            print(f"{sc} rep{rep}: chi2 {chi2:.2f}  " + "  ".join(f"{k} {Pf[k]:.3g}" for k in GEOM) +
                  f"  | E transfer median {ts['median_pct']:.1f} % p90 {ts['p90_pct']:.1f} % max {ts['max_pct']:.1f} %")
        g = {k: (float(np.mean([r["geom"][k] for r in rows])), float(np.std([r["geom"][k] for r in rows]))) for k in GEOM}
        results["scenarios"][sc] = dict(rows=rows, geom_mean_sd=g, minutes=(time.time() - t0) / 60,
                                        transfer_median_pct=float(np.mean([r["transfer"]["median_pct"] for r in rows])),
                                        transfer_max_pct=float(np.mean([r["transfer"]["max_pct"] for r in rows])))
        print(f"== {sc}: geometry mean+-sd " + "  ".join(f"{k} {g[k][0]:.3g}+-{g[k][1]:.2g}" for k in GEOM) +
              f"  | E transfer median {results['scenarios'][sc]['transfer_median_pct']:.1f} %, max {results['scenarios'][sc]['transfer_max_pct']:.1f} %  ({(time.time()-t0)/60:.1f} min)")
        json.dump(results, open(a.out, "w"), indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

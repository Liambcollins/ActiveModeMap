"""Check 2: does the split between contact damping (dashpot, Q_c) and intrinsic modal damping (zeta) set the E level?
Reproduces xprobe_cr1model.py's survey contact fit, then sweeps log_Qc and, at each value, re-solves zeta so the model CR1 Q
stays at the fitted value; reports how E/Q (= M) changes, and the survey-fit misfit along the sweep (does the data pick the split?).
    python tools/check_damping_split.py <cal_tag> [--apex]"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import brentq
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
tag = sys.argv[1]; apex = "--apex" in sys.argv
from fmmpaper import jointeb as J, ebfit, registry, io as fio
from fmmpaper.spectra import q_factor
if apex:
    _o = J.JointEB._matrices
    def _m(self, P):
        K, M, c1, k1, bv, fe = _o(self, P); L = P["L_um"]; xi_c = 1 - P["setback_um"] / L; h = P["tip_h_um"] / L
        _, phc, dphc = J.beam_modes([xi_c], self.n); phc, dphc = phc[:, 0], dphc[:, 0]
        bv2 = np.cos(self.tilt) * phc - h * np.sin(self.tilt) * dphc; k1 = 3.0 * 10 ** P["log_alpha"]
        return K - k1 * np.outer(bv, bv) + k1 * np.outer(bv2, bv2), M, np.sqrt(k1) / 10 ** P["log_Qc"], k1, bv2, fe
    J.JointEB._matrices = _m
import calibrate_freeair as CF, sim_calib_freeair as S
X = pd.read_csv(HERE.parent / "results/r3r4/xprobe_table.csv"); out = {}
for lab, ds, sh in (("SCM-PIT R1 (500 nN)", "scmpitB_r1_bias", "R1"), ("SCM-PIT R2 (500 nN)", "scmpitB_r2_bias", "R2")):
    ref = ebfit.load_calibration("stiff", tag); P_ref = dict(ref["P"]); S.F0 = ref["f0_hz"]
    probe = registry.PROBES[CF.PROBE_KEY["stiff"]]; xs, fs, Zs, src = CF.survey_P(fio.path_of(ds), probe)
    bands = {k: v for k, v in probe.bands_Hz.items() if k in ("CR1", "CR2", "CR3")}
    cap = CF.capture_from(xs, fs, Zs, bands); o = np.argsort(xs); xs = xs[o]; cap["Z"] = [z[o] for z in cap["Z"]]
    J.SHARED["c_um"] = (P_ref["c_um"], P_ref["c_um"] - 30, P_ref["c_um"] + 30); J.SHARED["L_um"] = (P_ref["L_um"], 0.6 * P_ref["L_um"], 1.4 * P_ref["L_um"])
    best = None
    for c0 in (P_ref["c_um"] - 8, P_ref["c_um"], P_ref["c_um"] + 8):
        r = CF.fit_contact(xs, [cap], dict(P_ref, c_um=c0), ["c_um"], starts=1)
        if best is None or r[2] < best[2]: best = r
    Pf, per, chi2 = best; la, lq, z = per[0][0], per[0][1], per[0][2]
    def M_of(lq_, z_):
        Pl = dict(Pf, log_alpha=la, log_Qc=lq_); f1 = S.eig(Pl, 1)[0]; fb = np.linspace(0.96 * f1, 1.04 * f1, 1600)
        je = J.JointEB(xs, [J.Band("CR1", fb, None), J.Band("QS", np.array([30e3]), None)], S.F0, n_modes=S.NM); S._scale(je, Pl)
        (zp, _), (zq, _) = je.band_maps(Pl, [z_[0], z_[0]], x=xs)
        a = np.abs(zp); k = a.argmax(1); Q = np.array([q_factor(fb, a[i], k[i]) for i in range(len(xs))]); E = a.max(1) / np.abs(zq)[:, 0]
        return E, Q
    z = np.atleast_1d(z); E0, Q0 = M_of(lq, z); Qt = float(np.median(Q0)); print("   Q0 per stop", np.round(Q0,1), "E0", np.round(E0,1), flush=True)
    T = X[X.dataset == lab]; Mm = np.array([ (T.M_meas_corr.values)[np.argmin(np.abs(T.x_um.values - xx))] for xx in xs]); used = np.array([T.used.values[np.argmin(np.abs(T.x_um.values - xx))] for xx in xs])
    rows = []
    for lq_ in np.linspace(lq - 1.2, lq + 2.0, 9):
        # zeta that keeps the model CR1 Q at Qt (None if impossible: contact damping alone already too strong)
        g = lambda lz: float(np.median(M_of(lq_, np.array([10 ** lz]))[1])) - Qt
        lo_, hi_ = g(-6.0), g(-2.2)
        if not (np.isfinite(lo_) and np.isfinite(hi_) and lo_ > 0 > hi_):
            rows.append(dict(log_Qc=float(lq_), feasible=False, Qmax_minus_target=float(lo_), Qmin_minus_target=float(hi_))); continue
        lz = brentq(g, -6.0, -2.2, xtol=1e-3); E, Q = M_of(lq_, np.array([10 ** lz]))
        Mmod = E / Q; ratio = Mm / Mmod
        # dashpot share of the CR1 loss: 1/Q = 1/Qc_eff + 2 zeta  -> share = 1 - 2*zeta*Q
        rows.append(dict(log_Qc=float(lq_), zeta=float(10 ** lz), Q=float(np.median(Q)), 
                         ratio_median=float(np.median(ratio[used])), ratio_range=[float(ratio[used].min()), float(ratio[used].max())]))
    out[sh] = dict(fit=dict(log_Qc=float(lq), zeta=float(z[0]), Q=Qt, chi2=float(chi2)), sweep=rows)
    print(f"== {tag} {sh}: fitted log_Qc {lq:.2f} zeta {z[0]:.2e} Q {Qt:.0f}", flush=True)
    for rr in rows: print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rr.items()}, flush=True)
json.dump(out, open(HERE.parent / f"results/r3r4/check_damping_split_{tag}.json", "w"), indent=1)

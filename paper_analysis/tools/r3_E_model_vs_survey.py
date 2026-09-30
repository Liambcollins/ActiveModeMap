"""Model E(x), Q and M = E/Q from a calibrate_freeair result vs the survey's in-situ E(x) (R3, Multi75-G).
    python tools/r3_E_model_vs_survey.py results/r3r4/calfa_<tag>.json [...]
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from fmmpaper import jointeb as J
from fmmpaper.spectra import q_factor
import sim_calib_freeair as S
S.F0 = 72900.0
surv = json.load(open(HERE.parent / "results/r3r4/r3_survey.json")); t = surv["table"]
xs = np.array([r["x_um"] for r in t]); E_meas = np.array([r["E"] for r in t]); E_corr = np.array([r["E_corr_no232"] for r in t]); Q_meas = np.array([r["Q"] for r in t])
def model(P, la, lq, z1, x):
    Pl = dict(P, log_alpha=la, log_Qc=lq); f1 = S.eig(Pl, 1)[0]; fb = np.linspace(0.97 * f1, 1.03 * f1, 1200)
    je = J.JointEB(x, [J.Band("CR1", fb, None), J.Band("QS", np.array([30e3]), None)], S.F0, n_modes=S.NM); S._scale(je, Pl)
    (zp, _), (zq, _) = je.band_maps(Pl, [z1, z1], x=x)
    a = np.abs(zp); k = a.argmax(1)
    Q = np.array([q_factor(fb, a[i], k[i]) for i in range(len(x))])
    return a.max(1) / np.abs(zq)[:, 0], Q, fb[k]
out = {}
for p in sys.argv[1:]:
    d = json.load(open(p)); P = d["contact_fit"]["P"]; pc = d["contact_fit"]["per_capture"][0]
    E, Q, fpk = model(P, pc["log_alpha"], pc["log_Qc"], pc["zeta"][0], xs)
    xe = np.arange(60, 236, 2.0); Ee, Qe, _ = model(P, pc["log_alpha"], pc["log_Qc"], pc["zeta"][0], xe)
    M_meas, M_corr, M_mod = E_meas / Q_meas, E_corr / Q_meas, E / Q
    tag = Path(p).stem
    print(f"== {tag}  chi2 {d['contact_fit']['chi2']:.0f}  " + "  ".join(f"{k} {P[k]:.4g}" for k in ["setback_um", "tip_h_um", "log_kcone", "c_um", "L_um", "mu", "J", "f0s"]))
    print("   x      E_meas  E_corr  E_model   Q_meas Q_model   M_meas M_corr M_model  M_model/M_corr  f_CR1 model kHz")
    for i in range(len(xs)):
        print(f"  {xs[i]:6.1f} {E_meas[i]:7.1f} {E_corr[i]:7.1f} {E[i]:8.1f}  {Q_meas[i]:6.0f} {Q[i]:6.0f}   {M_meas[i]:6.3f} {M_corr[i]:6.3f} {M_mod[i]:6.3f}   {M_mod[i]/M_corr[i]:6.3f}   {fpk[i]/1e3:.2f}")
    inner = xs < 230
    r = M_mod[inner] / M_corr[inner]; r2 = M_mod[inner] / M_meas[inner]
    print(f"   M_model/M_corr on 69-205 um: median {np.median(r):.3f}, spread (max/min) {r.max()/r.min():.3f};  vs uncorrected: median {np.median(r2):.3f}, spread {r2.max()/r2.min():.3f}")
    out[tag] = dict(P=P, per=pc, x=xs.tolist(), E_model=E.tolist(), Q_model=Q.tolist(), M_model=M_mod.tolist(), M_meas=M_meas.tolist(), M_corr=M_corr.tolist(),
                    x_eval=xe.tolist(), E_eval=Ee.tolist(), Q_eval=Qe.tolist(), ratio_corr_median=float(np.median(r)), ratio_corr_spread=float(r.max() / r.min()))
json.dump(out, open(HERE.parent / "results/r3r4/r3_E_model_vs_survey.json", "w"), indent=1, default=float)

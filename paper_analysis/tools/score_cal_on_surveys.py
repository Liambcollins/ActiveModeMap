"""Score a stiff-lever calibration on the SCM-PIT R1/R2 surveys: generic CR1 model (xprobe_cr1model.py, clamp + contact
refit per survey, geometry frozen) -> E_meas/E_model at the interior stops.  python tools/score_cal_on_surveys.py <cal_tag> [--apex]"""
import sys, json, runpy
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
tag = sys.argv[1]; apex = "--apex" in sys.argv
from fmmpaper import jointeb as J
if apex:
    _orig = J.JointEB._matrices
    def _m(self, P):
        K, M, c1, k1, bv, fe = _orig(self, P)
        L = P["L_um"]; xi_c = 1.0 - P["setback_um"] / L; h = P["tip_h_um"] / L
        _, phc, dphc = J.beam_modes([xi_c], self.n); phc, dphc = phc[:, 0], dphc[:, 0]
        bv2 = np.cos(self.tilt) * phc - h * np.sin(self.tilt) * dphc; k1 = 3.0 * 10 ** P["log_alpha"]
        K = K - k1 * np.outer(bv, bv) + k1 * np.outer(bv2, bv2)
        return K, M, np.sqrt(k1) / 10 ** P["log_Qc"], k1, bv2, fe
    J.JointEB._matrices = _m
X = pd.read_csv(HERE.parent / "results/r3r4/xprobe_table.csv"); res = {}
for lab, ds, short in (("SCM-PIT R1 (500 nN)", "scmpitB_r1_bias", "R1"), ("SCM-PIT R2 (500 nN)", "scmpitB_r2_bias", "R2")):
    out = f"cal_{tag}_{short}"
    sys.argv = ["xprobe_cr1model.py", "stiff", tag, ds, out]
    runpy.run_path(str(HERE / "xprobe_cr1model.py"), run_name="__main__")
    m = json.load(open(HERE.parent / f"results/r3r4/xprobe_cr1model_{out}.json"))
    t = X[X.dataset == lab]; xm = np.array(m["x"]); Mm = np.array(m["M_model"])
    Mmod = np.array([Mm[np.argmin(np.abs(xm - xx))] for xx in t.x_um.values])
    r = t.M_meas_corr.values / Mmod; u = t.used.values & np.isfinite(r)
    res[short] = dict(median=float(np.median(r[u])), range=[float(r[u].min()), float(r[u].max())], per_stop=[round(float(v), 3) for v in r], c_fit=m["c_fit"],
                      d33_cr1_median=float(np.median((t.d33_cr1_model.values * t.M_model.values / Mmod)[u])))
    print(f"### {tag} {short}: E_meas/E_model median {res[short]['median']:.3f} range {res[short]['range'][0]:.2f}-{res[short]['range'][1]:.2f}  per stop {res[short]['per_stop']}  d_eff(CR1) median {res[short]['d33_cr1_median']:.2f}", flush=True)
json.dump(res, open(HERE.parent / f"results/r3r4/score_cal_{tag}.json", "w"), indent=1)

"""Check 3: fixed-bin vs peak-tracked CR1 in E_meas (SCM-PIT R1/R2). Same CPD correction (delta from the QS fit) for both."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from fmmpaper import io as fio, domains
R = HERE.parent / "results/r3r4"; X = pd.read_csv(R / "xprobe_table.csv"); out = {}
for lab, key, sh in (("SCM-PIT R1 (500 nN)", "scmpitB_r1_bias", "R1"), ("SCM-PIT R2 (500 nN)", "scmpitB_r2_bias", "R2")):
    T = X[X.dataset == lab].reset_index(drop=True); invmap = dict(zip(np.round(T.x_um.values, 1), T.invols_nm.values * 1e-9)); u = T.used.values
    s = fio.load(key); res = {}
    m = json.load(open(R / f"xprobe_cr1model_cal_apex_semfixed_{sh}.json")); xm = np.array(m["x"]); Mm = np.array(m["M_model"])
    for trk in (False, True):
        t = domains.transfer_table(s, lambda x: invmap[round(float(x), 1)], 1.0, cr_band="CR1", track=trk)
        t = t.iloc[[int(np.argmin(np.abs(t.x_um.values - xx))) for xx in T.x_um.values]].reset_index(drop=True)
        cs = domains.cpd_split(t, mask=u)
        Q = t.Q.values; Mmeas = cs["E_corr"] / Q
        Mapex = np.array([Mm[np.argmin(np.abs(xm - xx))] for xx in T.x_um.values])
        r_gen = Mmeas / T.M_model.values; r_apx = Mmeas / Mapex
        res["tracked" if trk else "fixed_bin"] = dict(E_corr=[round(float(v), 1) for v in cs["E_corr"]], Q=[round(float(v), 1) for v in Q],
            f_spread_Hz=[round(float(v)) for v in t.f_cr_spread_Hz.values], ratio_generic_median=float(np.median(r_gen[u])), ratio_apexSEM_median=float(np.median(r_apx[u])),
            ratio_apexSEM_range=[float(r_apx[u].min()), float(r_apx[u].max())], delta_V=cs["delta_V"], d_eff_qs=cs["d33_uniform"],
            d_eff_cr_apexSEM_median=float(np.median((cs["P_cr_corr"] / (Mapex * Q))[u])))
        print(sh, "tracked" if trk else "fixed", {k: v for k, v in res["tracked" if trk else "fixed_bin"].items() if k not in ("E_corr",)}, flush=True)
    out[sh] = res
json.dump(out, open(R / "check_peak_tracking.json", "w"), indent=1)

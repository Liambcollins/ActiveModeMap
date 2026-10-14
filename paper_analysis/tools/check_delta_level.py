"""Check 1: does the domain-potential correction delta set the E level?  E_meas(delta) = |Pc - delta*bc| / |Pq - delta*bq|.
Compares with the beam-model E (generic calibration and apex/SEM calibration) on SCM-PIT R1/R2 and reports the delta that
would bring median E_meas/E_model to 1, plus what that delta does to the QS flatness and the CR1-route d_eff."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from fmmpaper import io as fio, domains
R = HERE.parent / "results/r3r4"; X = pd.read_csv(R / "xprobe_table.csv"); out = {}
for lab, key, sh in (("SCM-PIT R1 (500 nN)", "scmpitB_r1_bias", "R1"), ("SCM-PIT R2 (500 nN)", "scmpitB_r2_bias", "R2")):
    T = X[X.dataset == lab].reset_index(drop=True)
    invmap = dict(zip(np.round(T.x_um.values, 1), T.invols_nm.values * 1e-9))
    s = fio.load(key); t = domains.transfer_table(s, lambda x: invmap[round(float(x), 1)], 1.0, cr_band="CR1")
    order = [int(np.argmin(np.abs(t.x_um.values - xx))) for xx in T.x_um.values]; t = t.iloc[order].reset_index(drop=True)
    u = T.used.values; conv = np.ones(len(T))      # P, b already in pm/V
    Pq = t.Pq_re + 1j * t.Pq_im; bq = t.bq_re + 1j * t.bq_im; Pc = t.Pc_re + 1j * t.Pc_im; bc = t.bc_re + 1j * t.bc_im
    Pq, bq, Pc, bc = [np.asarray(v) for v in (Pq, bq, Pc, bc)]
    cs = domains.cpd_split(t.assign(), mask=u); d_fit = cs["delta_V"]
    rho = bc / Pc
    models = {"generic": T.M_model.values * T.Q.values}
    m = json.load(open(R / f"xprobe_cr1model_cal_apex_semfixed_{sh}.json")); xm = np.array(m["x"]); Mm = np.array(m["M_model"])
    models["apex_SEM"] = np.array([Mm[np.argmin(np.abs(xm - xx))] for xx in T.x_um.values]) * T.Q.values
    res = dict(delta_fit_V=d_fit, rho_cr=[round(float(abs(v)), 4) for v in rho], rho_cr_phase_deg=[round(float(np.degrees(np.angle(v))), 1) for v in rho])
    deltas = np.linspace(-3, 3, 601)
    for mk, Em in models.items():
        med = []; sd_qs = []; dcr = []; dqs = []
        for d in deltas:
            Eq = np.abs(Pc - d * bc) / np.abs(Pq - d * bq); r = Eq / Em
            med.append(np.median(r[u])); qs = np.abs(Pq - d * bq) * conv; cr = np.abs(Pc - d * bc) * conv / Em
            sd_qs.append(np.std(qs[u]) / np.mean(qs[u])); dqs.append(np.mean(qs[u])); dcr.append(np.median(cr[u]))
        med = np.array(med); sd_qs = np.array(sd_qs); i_fit = int(np.argmin(np.abs(deltas - d_fit)))
        cross = deltas[np.argmin(np.abs(med - 1))] if np.nanmin(np.abs(med - 1)) < 0.02 else None
        res[mk] = dict(ratio_at_fit=float(med[i_fit]), d_eff_qs_at_fit=float(dqs[i_fit]), d_eff_cr_at_fit=float(dcr[i_fit]),
                       ratio_at_0=float(med[300]), d_eff_qs_at_0=float(dqs[300]), d_eff_cr_at_0=float(dcr[300]),
                       delta_for_ratio_1=None if cross is None else float(cross))
        if cross is not None:
            j = int(np.argmin(np.abs(deltas - cross)))
            res[mk].update(qs_sd_pct_at_ratio1=float(100 * sd_qs[j]), qs_sd_pct_at_fit=float(100 * sd_qs[i_fit]), d_eff_qs_at_ratio1=float(dqs[j]), d_eff_cr_at_ratio1=float(dcr[j]))
        # sensitivity d(ratio)/d(delta) at the fit
        res[mk]["dratio_ddelta_per_V"] = float((med[i_fit + 10] - med[i_fit - 10]) / (deltas[i_fit + 10] - deltas[i_fit - 10]))
    out[sh] = res; print(sh, json.dumps(res, indent=1))
json.dump(out, open(R / "check_delta_level.json", "w"), indent=1)

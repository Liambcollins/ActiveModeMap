"""Cross-probe d33 with one pipeline: two-domain survey -> transfer table (QS 15-45 kHz / CR1 peak) ->
domain-CPD split (uniform d33) -> (1) QS d33 corrected, (2) QS d33 at the electrostatic blind spot (raw, no
correction needed), (3) CR1 check: |P_CR1, corrected| / (M_model * Q_measured), M_model = E/Q of the calibrated
lever (xprobe_cr1model.py). InvOLS: each stop's own force curve (stiff, Multi75) or tune note (soft).
    python tools/xprobe_d33.py <data root>
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from fmmpaper import registry, domains, calib, forcecurve as fc, io as fio
root = Path(sys.argv[1]); OUT = HERE.parent / "results/r3r4"
def fc_map(folder, files, xs):
    v = {}
    for x, f in zip(xs, files):
        r = fc.invols_fc(folder / f); v[round(x, 1)] = r["invols"] if np.isfinite(r["invols"]) else r["invols_ret"]
    return lambda x, _v=v: _v[round(float(x), 1)]
F6 = json.load(open(HERE.parent / "results/fig6_main.json"))["values"]["runs"]["R2"]["table"]
inv_r2 = {round(r["x_um"], 1): r["invols"] for r in F6}
D = {
 "SCM-PIT R1 (500 nN)": dict(key="scmpitB_r1_bias", model="stiffR1", load=500,
        inv=fc_map(root / "DomainsB_SCMPIT/10_bias_survey", [f"FSCMPIT{n:04d}.ibw" for n in range(44, 59, 2)], [232.0, 206.3, 180.6, 154.9, 129.1, 103.4, 77.7, 52.0])),
 "SCM-PIT R2 (500 nN)": dict(key="scmpitB_r2_bias", model="stiffR2", load=500, inv=lambda x: inv_r2[round(float(x), 1)]),
 "PPP-CONTAu (15 nN)": dict(key="ppp_bias_15nN", model="soft15", load=15, inv=calib.tune_invols(root / "DomainsBPPPCONTAU/02_domains_bias_15nN")[0], drop_tip=True),
 "PPP-CONTAu (250 nN)": dict(key="ppp_bias_250nN", model="soft250", load=250, inv=calib.tune_invols(root / "DomainsBPPPCONTAU/07_bias_survey_250nN")[0], drop_tip=True),
 "Multi75-G R3 (350 nN)": dict(key=None, path=root / "DomainsB_SCMPIT_R3/10_bias_survey/domains_bias_checkpoint_500nN_scmpit.npz", probe="multi75g", model="m75", load=350,
        inv=fc_map(root / "DomainsB_SCMPIT_R3", [f"FSCMPIT{n:04d}.ibw" for n in range(59, 72, 2)], [232.0, 204.9, 177.7, 150.6, 123.4, 96.3, 69.1]), drop_tip=True),
}
m75 = json.load(open(OUT / "r3_E_model_vs_survey.json"))["calfa_mu0.026"]; m75_P = json.load(open(OUT / "calfa_mu0.026.json"))["contact_fit"]["P"]
rows, summ = [], []
for lab, d in D.items():
    s = fio.load(d["key"]) if d["key"] else fio.load_series_file(d["path"], registry.PROBES[d["probe"]])
    t = domains.transfer_table(s, d["inv"], 1.0, cr_band="CR1")
    if d["model"] == "m75":
        M = np.array(m75["M_model"]); c = m75_P["c_um"]; Lc = m75_P["L_um"] - m75_P["setback_um"]
    else:
        mm = json.load(open(OUT / f"xprobe_cr1model_{d['model']}.json")); M = np.array(mm["M_model"])
        cal = json.load(open(HERE.parent / f"results/lever_calibration_{mm['probe']}_mass.json"))["P"]; c = mm["c_fit"]; Lc = cal["L_um"] - cal["setback_um"]
    xi = (t.x_um.values - c) / Lc
    keep = (xi < 0.97) if d.get("drop_tip") else np.ones(len(t), bool)
    if lab.startswith("Multi75"): keep = t.x_um.values < 230
    cs = domains.cpd_split(t, mask=keep)
    d33c = cs["d33_qs_corr"]; d33cr = cs["P_cr_corr"] / (M * t.Q.values)
    ib = int(np.argmin(t.ratio_qs.values))            # electrostatic blind spot (smallest |b|/|P| in QS), any stop
    for i in range(len(t)):
        rows.append(dict(dataset=lab, load_nN=d["load"], x_um=t.x_um[i], xi=xi[i], used=bool(keep[i]), invols_nm=t.invols[i] * 1e9, Q=t.Q[i], E=t.E[i],
                         ratio_qs=t.ratio_qs[i], d33_qs_raw=t.d33_qs[i], d33_qs_corr=d33c[i], M_model=M[i], M_meas_corr=cs["E_corr"][i] / t.Q[i],
                         d33_cr1_model=d33cr[i], d33_cr_over_Q=t.d33_cr_over_Q[i]))
    k = keep
    summ.append(dict(dataset=lab, load_nN=d["load"], n=int(k.sum()), dV_domains_V=cs["dV_domains_V"], d33_uniform=cs["d33_uniform"],
                     raw_median=float(np.median(t.d33_qs[k])), raw_max_over_min=float(t.d33_qs[k].max() / t.d33_qs[k].min()),
                     corr_sd_pct=float(100 * np.std(d33c[k]) / np.mean(d33c[k])),
                     blind_spot_x=float(t.x_um[ib]), blind_spot_ratio=float(t.ratio_qs[ib]), d33_blind_spot_raw=float(t.d33_qs[ib]),
                     cr1_model_median=float(np.median(d33cr[k])), cr1_model_sd_pct=float(100 * np.std(d33cr[k]) / np.mean(d33cr[k])),
                     cr1_model_max_over_min=float(d33cr[k].max() / d33cr[k].min()), cr_over_Q_max_over_min=float(t.d33_cr_over_Q[k].max() / t.d33_cr_over_Q[k].min())))
T = pd.DataFrame(rows); S_ = pd.DataFrame(summ)
T.to_csv(OUT / "xprobe_table.csv", index=False); S_.to_csv(OUT / "xprobe_summary.csv", index=False)
pd.set_option("display.width", 250); print(S_.round(3).to_string()); print(T.round(3).to_string())

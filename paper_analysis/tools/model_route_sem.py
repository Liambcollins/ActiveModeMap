"""One model-route table for every survey (SEM-geometry recipe, peak-tracked E). Writes results/r3r4/model_route_sem.json/.csv.

Model: xprobe_semgeom_blind_v2 --geom sem fits (results/r3r4/xprobe_semgeom_v2_sem_tracked.json) for the metal-coated levers,
adama_blind_fit --fix_geom --sig_static 1e3 (smooth) for the Adama. Lever frame: SCM-PIT R2 uses the survey static-ruler
registration (clamp at stage 24.1 um, as Fig. 3/4); the others use the frame of xprobe_table_tracked.csv (xi * L_eff).
Measured E: transfer_table(track=True), domain-potential corrected (M_meas_corr * Q) from xprobe_table_tracked.csv; Adama from
its own peak-tracked table. d_eff(model) = |P_CR1| S / (G E_model) [x r(x) stuck/sliding on the Adama only].
    python tools/model_route_sem.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import fmmpaper as F  # noqa: E402
from fmmpaper import eb_fit, registry  # noqa: E402
import eb_smooth  # noqa: E402,F401

R = F.config.RESULTS_DIR; OUT = R / "r3r4"
X = pd.read_csv(OUT / "xprobe_table_tracked.csv")
J = json.load(open(OUT / "xprobe_semgeom_v2_sem_tracked.json"))
PK = {"SCM-PIT": "scmpitB", "PPP-CONTAu": "pppcontau", "Multi75-G": "multi75g"}
SEM = {"SCM-PIT": dict(L=227.1, sb=10.25, h=11.7), "PPP-CONTAu": dict(L=461.6, sb=23.5, h=14.3), "Multi75-G": dict(L=234.7, sb=16.3, h=12.8), "Adama": dict(L=246.7, sb=21.2, h=14.6)}
FRAME_X0 = {"SCM-PIT R2 (500 nN)": 24.1}      # stage position of the clamp where a registration exists
rows, summ = [], {}
for lab, j in J.items():
    t = X[X.dataset == lab].sort_values("x_um"); used = t.used.values.copy()
    pr = lab.split(" ")[0]; g = SEM[pr]
    if lab in FRAME_X0: xl = t.x_um.values - FRAME_X0[lab]
    else:
        _, c = np.polyfit(t.xi.values, t.x_um.values, 1); xl = t.x_um.values - c
    tip = g["L"] - g["sb"]; used &= xl < tip - 10.0      # interior: more than 10 um from the tip
    p = np.array([j["params"][k] for k in eb_fit.PARAMS[:5]]); cr = np.array(j["cr_Hz"]); L = float(j["L_um"])
    pb = registry.PROBES[PK[pr]]; st = eb_fit.ContactState(lab, cr, float(j["Q1"]), np.array([]), np.array([]), f0_free_Hz=pb.f0_free_Hz, k_lever=pb.k_N_per_m, L_um=L)
    fk = eb_fit.peaks(p, st); Qm = eb_fit.q_model(p, st, fk[0])
    E_model = eb_fit.enhancement(p, st, xl)[0]; xg = np.linspace(2, L - 1, 300); E_grid = eb_fit.enhancement(p, st, xg)[0]
    E_meas = (t.M_meas_corr * t.Q).values; E_gen = (t.M_model * t.Q).values
    d33_model = t.d33_cr1_model.values * E_gen / E_model          # CR1 / E_model (no stuck/sliding factor on metal levers)
    if lab in FRAME_X0:   # SCM-PIT R2: use the peak-aligned two-domain maps of Fig. 3 (results/fig_e_routes_sem_summary.json) so Fig. 3/4 and the tables share one dataset
        FS = json.load(open(R / "fig_e_routes_sem_summary.json")); assert np.allclose(FS["x_lever"], xl, atol=0.2)
        E_meas = np.array(FS["E_meas_tracked"]); d33_model = np.array(FS["d33_cr_model"]); E_model = np.array(FS["E_model_pts"])
        t = t.assign(d33_qs_corr=np.array(FS["d33_qs_stat"]))
    ratio = E_meas / E_model
    k = used & np.isfinite(ratio)
    for i in range(len(xl)):
        rows.append(dict(dataset=lab, probe=pr, x_stage_um=t.x_um.values[i], x_lever_um=xl[i], used=bool(used[i]), Q=t.Q.values[i], E_meas=E_meas[i], E_model=E_model[i],
                         E_generic=E_gen[i], ratio=ratio[i], d33_cr1_model=d33_model[i], d33_qs_corr=t.d33_qs_corr.values[i], d33_qs_raw=t.d33_qs_raw.values[i]))
    dm = d33_model[k]
    summ[lab] = dict(probe=pr, geometry=dict(L_um=L, setback_um=float(p[2]), tip_h_um=float(p[4])), k_ratio=float(10 ** p[0]), kcone_ratio=float(10 ** p[1]), zeta=float(10 ** p[3]),
                     cr_meas_kHz=(cr / 1e3).tolist(), cr_model_kHz=(fk / 1e3).tolist(), cr_err_pct=(100 * (fk / cr - 1)).round(2).tolist(), Q_meas=float(j["Q1"]), Q_model=float(Qm),
                     ratio_median=float(np.median(ratio[k])), ratio_min=float(ratio[k].min()), ratio_max=float(ratio[k].max()), n_interior=int(k.sum()),
                     d33_model_median=float(np.median(dm)), d33_model_min=float(dm.min()), d33_model_max=float(dm.max()), d33_model_sd_pct=float(100 * dm.std() / dm.mean()),
                     d33_qs_corr=float(np.median(t.d33_qs_corr.values[k])), generic_ratio_median=float(np.median((E_meas / E_gen)[k])),
                     x_grid=xg.tolist(), E_grid=E_grid.tolist(), frame_x0_stage=float(FRAME_X0.get(lab, np.nan)) if lab in FRAME_X0 else float(t.x_um.values[0] - xl[0]))
    print(f"{lab:24s} CR err {summ[lab]['cr_err_pct']} Q {Qm:.0f}/{j['Q1']:.0f} k*/k {10**p[0]:.0f} kc/k {10**p[1]:.1f} | ratio {np.median(ratio[k]):.3f} ({ratio[k].min():.2f}-{ratio[k].max():.2f}) n={k.sum()} | d_eff {np.median(dm):.2f} ({dm.min():.2f}-{dm.max():.2f}, sd {100*dm.std()/dm.mean():.1f}%) vs QS {np.median(t.d33_qs_corr.values[k]):.2f}")

# ---- Adama (smooth SEM fit rescored at x0 = -20)
A = json.load(open(R / "adama/adama_R1_semgeom_nostatic_smooth_x0m20_score.json")); AJ = json.load(open(R / "adama/adama_R1_semgeom_nostatic_smooth_blind_fit.json"))
AT = pd.read_csv(R / "adama/adama_R1_semgeom_nostatic_smooth_table.csv")
xl = np.array(A["x_lever"]); used = xl < (246.7 - 21.2) - 10.0; lab = "Adama R1 (750 nN)"
ratio = np.array(A["ratio"]); dm_all = np.array(A["d33_cr1_rE"]); dq = np.array(A["d33_qs_r"]); r = np.array(A["r"])
for i in range(len(xl)):
    rows.append(dict(dataset=lab, probe="Adama", x_stage_um=AT.x_um.values[i], x_lever_um=xl[i], used=bool(used[i]), Q=AT.Q.values[i], E_meas=A["E_tracked"][i], E_model=A["E_model"][i],
                     E_generic=np.nan, ratio=ratio[i], d33_cr1_model=dm_all[i], d33_qs_corr=dq[i], d33_qs_raw=AT.d33_qs.values[i], r_stuck_slide=r[i]))
p = np.array(AJ["p"]); st = eb_fit.ContactState(lab, np.array(AJ["cr_Hz"]), AJ["Q1"], np.array([]), np.array([]), f0_free_Hz=AJ["f0_free_Hz"], k_lever=AJ["k_lever"], L_um=AJ["L_um"])
xg = np.linspace(2, AJ["L_um"] - 1, 300); E_grid = eb_fit.enhancement(p, st, xg)[0]
k = used; sm = AJ["summary"]
summ[lab] = dict(probe="Adama", geometry=dict(L_um=AJ["L_um"], setback_um=21.2, tip_h_um=14.6), k_ratio=sm["k_ratio"], kcone_ratio=sm["kcone_ratio"], zeta=sm["zeta"],
                 cr_meas_kHz=(np.array(AJ["cr_Hz"]) / 1e3).tolist(), cr_model_kHz=sm["cr_model_kHz"], cr_err_pct=sm["cr_err_pct"], Q_meas=AJ["Q1"], Q_model=A["Q_model"],
                 ratio_median=float(np.median(ratio[k])), ratio_min=float(ratio[k].min()), ratio_max=float(ratio[k].max()), n_interior=int(k.sum()),
                 d33_model_median=float(np.median(dm_all[k])), d33_model_min=float(dm_all[k].min()), d33_model_max=float(dm_all[k].max()), d33_model_sd_pct=float(100 * dm_all[k].std() / dm_all[k].mean()),
                 d33_qs_corr=float(np.median(dq[k])), generic_ratio_median=None, x_grid=xg.tolist(), E_grid=E_grid.tolist(), frame_x0_stage=-20.0, stuck_sliding=True)
print(f"{lab:24s} CR err {sm['cr_err_pct']} | ratio {np.median(ratio[k]):.3f} ({ratio[k].min():.2f}-{ratio[k].max():.2f}) | CR1/(rE) {np.median(dm_all[k]):.2f} vs QS/r {np.median(dq[k]):.2f}")
pd.DataFrame(rows).to_csv(OUT / "model_route_sem.csv", index=False)
json.dump(summ, open(OUT / "model_route_sem.json", "w"), indent=1, default=float)
print("saved", OUT / "model_route_sem.json")

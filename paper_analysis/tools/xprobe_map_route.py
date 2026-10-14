"""Reconstructed-map route (E_map) for every survey: the Fig. 3c pipeline applied across probes.

Per survey: two-domain P_raw(x, f) and b(x, f) from the bias survey; domain-CPD correction P = P_raw - delta*b with delta
from the QS uniform-d_eff split (metal-coated levers; not applied to the Adama); EB+GP reconstruction of the CR1 and QS
P maps (calibrated lever as interpolant, GP on the residual); E_map(x) = max|P_CR1| / mean|P_QS|; leave-one-out E at every
stop; stationary d_eff by the map route = |P_CR1| InvOLS / G / E_loo (Adama: additionally / r_stuck_slide).
    python tools/xprobe_map_route.py [--only SCM-PIT]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from fmmpaper import registry, domains, ebfit, physrec, io as fio, config  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--only", default=""); ap.add_argument("--no_loo", action="store_true"); ap.add_argument("--tracked", action="store_true", help="peak-aligned CR1 maps, SEM-recipe model column, Adama smooth fit, R2 survey frame")
a = ap.parse_args()
root = config.data_root(); R = HERE.parent / "results"; OUT = R / "r3r4"
X = pd.read_csv(OUT / "xprobe_table.csv"); SB = pd.read_csv(OUT / "xprobe_semgeom_blind_table.csv")
AD = pd.read_csv(R / "adama/adama_R1_stuck_slide_table.csv"); ADJ = json.load(open(R / "adama/adama_R1_stuck_slide.json"))
ad_fit = json.load(open(R / ("adama/adama_R1_semgeom_nostatic_smooth_blind_fit.json" if a.tracked else "adama/adama_R1_semgeom_nostatic_blind_fit.json")))
MR = pd.read_csv(OUT / "model_route_sem.csv") if a.tracked else None
if a.tracked: X = pd.read_csv(OUT / "xprobe_table_tracked.csv")
cal_adama = dict(probe="adama", f0_hz=ad_fit["f0_free_Hz"], n_modes=14, bands=["CR1"], zeta=[ad_fit["params"]["log_zeta"] and 10 ** ad_fit["params"]["log_zeta"]], eps=[0.0], dlf=[0.0],
                 P=dict(log_alpha=ad_fit["params"]["log_alpha"], log_kcone=ad_fit["params"]["log_kcone"], log_Qc=2.3, setback_um=21.2, tip_h_um=14.6,
                        c_um=ADJ["x0_stage_um"], L_um=ADJ["L_um"], mu=0.03, J=0.0))
D = {"SCM-PIT R1 (500 nN)": ("scmpitB", root / "DomainsB_SCMPIT/10_bias_survey/domains_bias_checkpoint_500nN_scmpit.npz", ("stiff", "mass")),
     "SCM-PIT R2 (500 nN)": ("scmpitB", root / "DomainsB_SCMPIT_R2/10_bias_survey/domains_bias_checkpoint_500nN_scmpit.npz", ("stiff", "mass")),
     "PPP-CONTAu (15 nN)": ("pppcontau", root / "DomainsBPPPCONTAU/02_domains_bias_15nN/domains_bias_checkpoint.npz", ("soft", "mass")),
     "PPP-CONTAu (250 nN)": ("pppcontau", root / "DomainsBPPPCONTAU/07_bias_survey_250nN/domains_bias_checkpoint_250nN.npz", ("soft", "mass")),
     "Multi75-G R3 (350 nN)": ("multi75g", root / "DomainsB_SCMPIT_R3/10_bias_survey/domains_bias_checkpoint_500nN_scmpit.npz", ("multi75g", "sem")),
     "Adama R1 (750 nN)": ("adama", root / "DomainsB_ADAMA_R1/10_bias_survey/domains_bias_checkpoint_500nN_scmpit.npz", None)}
rows, summ = [], {}
for lab, (pk, path, caltag) in D.items():
    if a.only and a.only not in lab: continue
    probe = registry.PROBES[pk]; G = probe.amp_divisor
    s = fio.load_series_file(path, probe)
    dec = domains.decompose_series(s, band=None, align_band="CR1" if a.tracked else None); Praw, b, f = dec["P"], dec["b"], s.freq_Hz
    if pk == "adama":
        t = AD.sort_values("x_um").reset_index(drop=True); used = t.used.values; inv = t.invols.values; c_stage = ADJ["x0_stage_um"]
        delta = 0.0; cal = cal_adama; rfac = t.r_stuck_slide.values; E_gen = np.full(len(t), np.nan); E_sem = t.E_model_stuck.values
        if a.tracked:
            mr = MR[MR.dataset == "Adama R1 (750 nN)"].sort_values("x_stage_um"); E_sem = mr.E_model.values; rfac = mr.r_stuck_slide.values; c_stage = -20.0; cal["P"]["c_um"] = -20.0
    else:
        t = X[X.dataset == lab].sort_values("x_um").reset_index(drop=True); used = t.used.values; inv = t.invols_nm.values * 1e-9
        Lc, c_stage = np.polyfit(t.xi.values, t.x_um.values, 1)
        if a.tracked and lab == "SCM-PIT R2 (500 nN)": c_stage = 24.1      # survey static-ruler registration (Fig. 3/4 frame)
        tt = domains.transfer_table(s, lambda x, _i=dict(zip(np.round(t.x_um.values, 1), inv)): _i[round(float(x), 1)], 1.0, cr_band="CR1", track=a.tracked)
        cs = domains.cpd_split(tt, mask=used); delta = cs["delta_V"]
        cal = ebfit.load_calibration(*caltag); cal = json.loads(json.dumps(cal)); cal["P"]["c_um"] = float(c_stage)
        rfac = np.ones(len(t)); E_gen = t.M_model.values * t.Q.values
        sb = SB[SB.dataset == lab].sort_values("x_um"); E_sem = sb.E_sem_blind.values
        if a.tracked: mr = MR[MR.dataset == lab].sort_values("x_stage_um"); E_sem = mr.E_model.values; E_gen = mr.E_generic.values
    assert np.allclose(np.sort(s.x_um), t.x_um.values, atol=0.6)
    Pcor = Praw - delta * b
    bands = {"CR1": probe.bands_Hz["CR1"], "QS": probe.qs_band_Hz}
    zc, fc = physrec.band_slice(f, Pcor, bands["CR1"], n_max=250); zq, fq = physrec.band_slice(f, Pcor, bands["QS"], n_max=100)
    xs = s.x_um; n = len(xs)
    def recon(keep, xe):
        oc = ebfit.rec_eb2(xs, keep, zc[keep], fc, cal, "CR1", x_eval=xe)
        cq = json.loads(json.dumps(cal)); cq["P"]["log_alpha"] = oc["theta"]["log_alpha"]
        oq = ebfit.rec_eb2(xs, keep, zq[keep], fq, cq, "QS", free=("eps",), x_eval=xe)
        return oc["Zrec"], oq["Zrec"]
    xg = np.linspace(xs.min() - 10, xs.max() + 5, 200)
    Zc_g, Zq_g = recon(np.arange(n), xg); E_map = np.abs(Zc_g).max(1) / np.abs(Zq_g).mean(1)
    E_meas = np.abs(zc).max(1) / np.abs(zq).mean(1)
    E_loo = np.full(n, np.nan)
    if not a.no_loo:
        for j in range(n):
            keep = np.array([i for i in range(n) if i != j]); zc_j, zq_j = recon(keep, xs[[j]])
            E_loo[j] = np.abs(zc_j[0]).max() / np.abs(zq_j[0]).mean()
    conv = inv / G * 1e12
    d33_qs = np.abs(zq).mean(1) * conv / rfac; d33_cr_loo = np.abs(zc).max(1) * conv / rfac / E_loo
    d33_cr_gen = np.abs(zc).max(1) * conv / rfac / E_gen; d33_cr_sem = np.abs(zc).max(1) * conv / rfac / E_sem
    k = used & (np.isfinite(E_loo) | a.no_loo)
    def med_rng(v):
        v = v[k]; v = v[np.isfinite(v)]
        return f"{np.median(v):.2f} ({v.min():.2f}–{v.max():.2f})" if v.size else "—"
    print(f"\n== {lab}: delta {delta:+.3f} V, clamp stage {c_stage:+.1f}")
    print(f"   E_meas/E_loo  {med_rng(E_meas/E_loo)} | E_meas/E_generic {med_rng(E_meas/E_gen) if np.isfinite(E_gen).any() else '—'} | E_meas/E_sem {med_rng(E_meas/E_sem)}")
    print(f"   d_eff QS corrected {med_rng(d33_qs)} | CR1/E_loo (map) {med_rng(d33_cr_loo)} | CR1/E_generic {med_rng(d33_cr_gen) if np.isfinite(E_gen).any() else '—'} | CR1/E_sem {med_rng(d33_cr_sem)}")
    for i in range(n):
        rows.append(dict(dataset=lab, x_um=xs[i], used=bool(used[i]), E_meas=E_meas[i], E_loo=E_loo[i], E_generic=E_gen[i], E_sem=E_sem[i], r_stuck_slide=rfac[i],
                         d33_qs_corr=d33_qs[i], d33_cr_map=d33_cr_loo[i], d33_cr_generic=d33_cr_gen[i], d33_cr_sem=d33_cr_sem[i]))
    summ[lab] = dict(delta_V=float(delta), clamp_stage_um=float(c_stage), x_grid=xg.tolist(), E_map_grid=E_map.tolist(),
                     E_meas_over_loo=med_rng(E_meas / E_loo), d33_qs=med_rng(d33_qs), d33_cr_map=med_rng(d33_cr_loo),
                     d33_cr_generic=(med_rng(d33_cr_gen) if np.isfinite(E_gen).any() else None), d33_cr_sem=med_rng(d33_cr_sem))
tag = "_tracked" if a.tracked else ""
pd.DataFrame(rows).to_csv(OUT / f"xprobe_map_route_table{tag}.csv", index=False)
json.dump(summ, open(OUT / f"xprobe_map_route{tag}.json", "w"), indent=1)
print("wrote", OUT / f"xprobe_map_route_table{tag}.csv")

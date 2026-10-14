"""SEM-geometry blind lever model for every survey (the Adama recipe applied to all probes).

Per survey: contact-resonance frequencies (two-domain P, median peak over stops per band) and the CR1 Q (median over
interior stops) are the only fit inputs. Geometry is FIXED from the SEM (L, tip set-back, tip height); free: k*/k,
k_cone/k, zeta. Withheld and used to score: the measured (CPD-corrected) E(x) and the force-curve InvOLS shape.
Also reports the stuck/sliding static-shape ratio r(x) implied by the fitted lateral stiffness.
    python tools/xprobe_semgeom_blind.py [--starts 6 --nfev 40]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from fmmpaper import registry, domains, eb_fit, io as fio, config, spectra  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--starts", type=int, default=6); ap.add_argument("--nfev", type=int, default=40); ap.add_argument("--only", default=""); ap.add_argument("--free_sb", action="store_true", help="L and tip height from SEM; set-back free (rectangular-beam effective value)"); ap.add_argument("--geom", default="sem", choices=["sem", "nominal", "nominal_f0"], help="sem: electron microscopy; nominal: vendor datasheet L, tip height midpoint, tip at the end (set-back 0.5 um); nominal_f0: also the datasheet f0 and k")
a = ap.parse_args()
root = config.data_root(); R = HERE.parent / "results"; OUT = R / "r3r4"
X = pd.read_csv(OUT / "xprobe_table.csv")
# SEM geometry (cross-probe note 2026-09-30; Adama 2026-10-01). set-back = tip apex to pointed end; tip height from side view.
SEM = {"scmpitB": dict(L=227.1, sb=10.25, h=11.7), "pppcontau": dict(L=461.6, sb=23.5, h=14.3), "multi75g": dict(L=234.7, sb=16.3, h=12.8)}
# vendor datasheets (nominal): Nanosensors SCM-PIT L 225 um, tip height 10-15 um, f0 75 kHz, k 2.8 N/m; Nanosensors PPP-CONTAu L 450 um,
# tip height 10-15 um, f0 13 kHz, k 0.2 N/m; BudgetSensors Multi75-G L 225 um, tip height 17 um, f0 75 kHz, k 3 N/m. No vendor gives the
# tip set-back; "tip at the very end" is taken as 0.5 um.
NOMINAL = {"scmpitB": dict(L=225.0, sb=0.5, h=12.5, f0=75e3, k=2.8), "pppcontau": dict(L=450.0, sb=0.5, h=12.5, f0=13e3, k=0.2), "multi75g": dict(L=225.0, sb=0.5, h=17.0, f0=75e3, k=3.0)}
if a.geom != "sem": SEM = NOMINAL
D = {"SCM-PIT R1 (500 nN)": ("scmpitB", root / "DomainsB_SCMPIT/10_bias_survey/domains_bias_checkpoint_500nN_scmpit.npz"),
     "SCM-PIT R2 (500 nN)": ("scmpitB", root / "DomainsB_SCMPIT_R2/10_bias_survey/domains_bias_checkpoint_500nN_scmpit.npz"),
     "PPP-CONTAu (15 nN)": ("pppcontau", root / "DomainsBPPPCONTAU/02_domains_bias_15nN/domains_bias_checkpoint.npz"),
     "PPP-CONTAu (250 nN)": ("pppcontau", root / "DomainsBPPPCONTAU/07_bias_survey_250nN/domains_bias_checkpoint_250nN.npz"),
     "Multi75-G R3 (350 nN)": ("multi75g", root / "DomainsB_SCMPIT_R3/10_bias_survey/domains_bias_checkpoint_500nN_scmpit.npz")}
rows = []; summary = {}
for lab, (pk, path) in D.items():
    if a.only and a.only not in lab: continue
    probe = registry.PROBES[pk]; g = SEM[pk]
    s = fio.load_series_file(path, probe)
    d = domains.decompose_series(s, band=None); P = d["P"]; f = s.freq_Hz
    cr = []
    for b in ("CR1", "CR2", "CR3"):
        if b not in probe.bands_Hz: break
        lo, hi = probe.bands_Hz[b]
        if hi > f.max(): break
        fk = [spectra.peak(f, P[i], (lo, hi))["f_Hz"] for i in range(P.shape[0])]
        amp = [spectra.peak(f, P[i], (lo, hi))["amp"] for i in range(P.shape[0])]
        w = np.array(amp) > 0.3 * max(amp)           # ignore stops near a node of that mode
        cr.append(float(np.median(np.array(fk)[w])))
    cr = np.array(cr)
    t = X[X.dataset == lab].sort_values("xi"); used = t.used.values
    Lc_cal, c_cal = np.polyfit(t.xi.values, t.x_um.values, 1)      # calibrated frame: x_um = c + xi*(L_cal - sb_cal)
    xl = t.x_um.values - c_cal                                       # lever frame from the calibrated clamp position
    Q1 = float(np.nanmedian(t.Q.values[used]))
    f0_use, k_use = (g["f0"], g["k"]) if a.geom == "nominal_f0" else (probe.f0_free_Hz, probe.k_N_per_m)
    st = eb_fit.ContactState(lab, cr, Q1, np.array([]), np.array([]), f0_free_Hz=f0_use, k_lever=k_use, L_um=g["L"])
    lo, hi = eb_fit.LO.copy(), eb_fit.HI.copy(); lo[4], hi[4] = g["h"] - 1e-3, g["h"] + 1e-3
    if not a.free_sb: lo[2], hi[2] = g["sb"] - 1e-3, g["sb"] + 1e-3
    starts = [[la, lk, g["sb"] * (0.5 if a.free_sb else 1.0), np.log10(3e-3), g["h"]] for la in (2.6, 3.0, 3.4) for lk in (0.5, 1.5, 2.5, 3.5)][:a.starts]
    print(f"\n== {lab}: CR {np.round(cr/1e3,1)} kHz ({len(cr)} modes), Q1 {Q1:.0f}, SEM L {g['L']} sb {g['sb']} h {g['h']}, clamp at stage {c_cal:+.1f}")
    r = eb_fit.fit(st, use_nodes=False, starts=starts, lo=lo, hi=hi, sig_f=0.015, sig_static=1e3, sig_Q=0.05, max_nfev=a.nfev)
    if r is None: print("   fit failed"); continue
    p = r.x; sm = eb_fit.summarize(p, st)
    E_model = eb_fit.enhancement(p, st, xl)[0]
    xt = g["L"] - p[2]; p_sl = p.copy(); p_sl[1] = -2.0
    rr = (eb_fit.static_shape(p, st, xl) / eb_fit.static_shape(p, st, np.array([xt]))[0]) / (eb_fit.static_shape(p_sl, st, xl) / eb_fit.static_shape(p_sl, st, np.array([xt]))[0])
    E_meas = t.M_meas_corr.values * t.Q.values; E_gen = t.M_model.values * t.Q.values
    ratio = E_meas / E_model; k = used & np.isfinite(ratio)
    # static ruler check (withheld): sliding-model shape vs 1/InvOLS
    y = 1 / t.invols_nm.values[used]; z = eb_fit.static_shape(p_sl, st, xl[used]); gain = np.sum(y * z) / np.sum(z * z); st_rms = float(np.sqrt(np.mean(np.log(y / (gain * z)) ** 2)))
    print(f"   fit: CR err % {sm['cr_err_pct']}  Q {sm['Q_model']:.0f}/{Q1:.0f}  k*/k {sm['k_ratio']:.0f}  kcone/k {sm['kcone_ratio']:.1f}  setback {sm['setback_um']:.1f}  zeta {sm['zeta']:.2e}  cost {r.cost:.3f}")
    print(f"   E_meas/E_model (SEM blind): median {np.median(ratio[k]):.3f}, range {ratio[k].min():.2f}-{ratio[k].max():.2f} | generic: median {np.median((E_meas/E_gen)[k]):.3f}, range {(E_meas/E_gen)[k].min():.2f}-{(E_meas/E_gen)[k].max():.2f}")
    print(f"   stuck/slide r(x): {np.round(rr[used], 3)}   sliding static ruler rms {100*st_rms:.1f} %")
    for i in range(len(xl)):
        rows.append(dict(dataset=lab, x_um=t.x_um.values[i], x_lever_um=xl[i], used=bool(used[i]), Q=t.Q.values[i], E_meas_corr=E_meas[i], E_generic=E_gen[i], E_sem_blind=E_model[i], r_stuck_slide=rr[i],
                         d33_cr1_sem_blind=t.d33_cr1_model.values[i] * E_gen[i] / E_model[i] / rr[i]))
    summary[lab] = dict(cr_Hz=cr.tolist(), Q1=Q1, params=dict(zip(eb_fit.PARAMS[:5], p.tolist())), summary=sm, clamp_stage_um=float(c_cal), sem=g,
                        ratio_median=float(np.median(ratio[k])), ratio_range=[float(ratio[k].min()), float(ratio[k].max())],
                        generic_ratio_median=float(np.median((E_meas / E_gen)[k])), static_rms_sliding_pct=100 * st_rms)
pd.DataFrame(rows).to_csv(OUT / f"xprobe_semgeom_blind{'_freesb' if a.free_sb else ''}{'' if a.geom == 'sem' else '_' + a.geom}_table.csv", index=False)
json.dump(summary, open(OUT / f"xprobe_semgeom_blind{'_freesb' if a.free_sb else ''}{'' if a.geom == 'sem' else '_' + a.geom}.json", "w"), indent=1, default=float)
print("wrote", OUT / f"xprobe_semgeom_blind{'_freesb' if a.free_sb else ''}{'' if a.geom == 'sem' else '_' + a.geom}_table.csv")

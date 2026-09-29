"""Live FMM capture (stiff probe B, R2) reconstructed with EB+GP instead of low-rank, and d33 from it.

Steps
-----
1. Fit the contact state (k*/k, zeta, eps, gain per band) to the 8 wideband spectra of the live
   capture with the calibrated lever (ebfit.rec_eb2); the clamp position c is refitted per capture
   because the laser frame drifts (phase 2d), everything else is the lever calibration.
2. Score EB and EB+GP on the 7 never-visited validation positions, against low-rank (rank 5), on
   the CR bands (matched span) and on the full band.
3. Enhancement along the lever from the live capture: E(x) = |Z(x, f_CR1)| / |Z(x, QS)| from the
   EB+GP maps (CR1 band and a quasi-static band both fitted+corrected), from bare EB, and from the
   piezo pathway of the calibrated model with the live contact state (no gains: the blind-EB analogue).
4. d33 at the bias-survey positions: two-domain CR1 amplitude divided by each E, against the
   quasi-static route and the archived blind-EB route (Fig. 6 pipeline, CPD-corrected).

    python tools/live_ebgp_d33.py
"""
from __future__ import annotations

import datetime as dt, json, sys, time
from pathlib import Path

import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fmmpaper as F  # noqa: E402
from fmmpaper import axis, calib, domains, ebfit, eb_fit, physrec, recon, spectra  # noqa: E402
from fmmpaper import jointeb as J  # noqa: E402

L_UM, VAC = 226.5, 1.0
CAMP, D0 = "DomainsB_SCMPIT_R2", dt.datetime(2026, 9, 20, 22)
X0_DENSE = 28.1                      # ruler x0 during the dense map, the calibration's frame
BANDS = ["CR1", "CR2", "CR3"]


def interp_rows(ft, ff, M):
    return (np.apply_along_axis(lambda r: np.interp(ft, ff, r), 1, M.real)
            + 1j * np.apply_along_axis(lambda r: np.interp(ft, ff, r), 1, M.imag))


def fit_capture(d, cal, x_eval, bands, extra_bands=None, fit_c=True, n_max=250):
    """Per-band EB+GP on a capture; maps on x_eval (stage um, same frame as d.x_um).
    Returns dict(band -> rec_eb2 output + f), the fitted c, and full-grid EB / EB+GP / sd arrays."""
    x, f, Z = d.x_um, d.freq_Hz, d.Z[0]
    sel = np.arange(len(x)); out = {}
    cal = json.loads(json.dumps(cal))
    if fit_c:   # frame from the capture itself: CR3 has two nodes in range and pins c best
        zb, fb = physrec.band_slice(f, Z, d.probe.bands_Hz["CR3"], n_max=n_max)
        o = ebfit.rec_eb2(x, sel, zb, fb, cal, "CR3", free=("log_alpha", "log_zeta", "eps", "c_um"), gp=False)
        cal["P"]["c_um"] = o["theta"]["c_um"]
    for b in bands:
        zb, fb = physrec.band_slice(f, Z, d.probe.bands_Hz[b], n_max=n_max)
        o = ebfit.rec_eb2(x, sel, zb, fb, cal, b, x_eval=x_eval); o["f"] = fb; out[b] = o
    for name, rng in (extra_bands or {}).items():   # e.g. the quasi-static band, with k* from CR1
        zb, fb = physrec.band_slice(f, Z, rng, n_max=n_max)
        c2 = json.loads(json.dumps(cal)); c2["P"]["log_alpha"] = out["CR1"]["theta"]["log_alpha"]
        o = ebfit.rec_eb2(x, sel, zb, fb, c2, name, free=("eps",), x_eval=x_eval); o["f"] = fb; out[name] = o
    return out, cal["P"]["c_um"]


def stitch(fits, f, nx, bands):
    eb = np.full((nx, f.size), np.nan + 0j); ebgp = eb.copy(); sd = np.full((nx, f.size), np.nan); span = np.zeros(f.size, bool)
    for b in bands:
        o = fits[b]; m = (f >= o["f"][0]) & (f <= o["f"][-1]); span |= m
        eb[:, m] = interp_rows(f[m], o["f"], o["Zeb"]); ebgp[:, m] = interp_rows(f[m], o["f"], o["Zrec"])
        sd[:, m] = np.apply_along_axis(lambda r: np.interp(f[m], o["f"], r), 1, o["sd"])
    return eb, ebgp, sd, span


def score(Zp, Zt, f, bands_Hz, span):
    ok = ~np.isnan(Zp[0]); span = span & ok
    r = dict(nrmse_span=recon.nrmse(Zp[:, span], Zt[:, span]))
    for b, rng in bands_Hz.items():
        m = (f >= rng[0]) & (f <= rng[1]) & ok; r[f"nrmse_{b}"] = recon.nrmse(Zp[:, m], Zt[:, m])
    return r


def main():
    cal = ebfit.load_calibration("stiff", "mass")
    d, v = F.load("scmpitB_r2_wb_fast"), F.load("scmpitB_r2_wb_valid")
    tl = calib.timeline(CAMP, D0); an = calib.frame_anchors(CAMP, tl, L_UM).set_index("stage")
    x0_d, x0_v, x0_s = (float(an.loc[k, "x0"]) for k in ("20_wideband_fast", "21_wideband_validation", "10_bias_survey"))
    print(f"ruler x0: dense {X0_DENSE}, design {x0_d:.2f}, validation {x0_v:.2f}, bias survey {x0_s:.2f} um")
    c_ruler = cal["P"]["c_um"] - (X0_DENSE - x0_d)
    # ---------------------------------------------------------------- 1-2. live fit + validation
    f = d.freq_Hz; bands_Hz = dict(d.probe.bands_Hz)
    xv_design_frame = v.x_um - (x0_v - x0_d)                     # validation positions in the design's frame
    rows = {}
    for tag, fit_c in (("c fitted from capture", True), ("c from ruler drift", False)):
        c0 = json.loads(json.dumps(cal)); c0["P"]["c_um"] = c_ruler
        t0 = time.time(); fits, c_used = fit_capture(d, c0, xv_design_frame, BANDS, fit_c=fit_c)
        eb, ebgp, sd, span = stitch(fits, f, len(v.x_um), BANDS)
        rows[tag] = dict(c_um=c_used, sec=time.time() - t0,
                         EB=score(eb, v.Z[0], f, bands_Hz, span), **{"EB+GP": score(ebgp, v.Z[0], f, bands_Hz, span)},
                         theta={b: {k: fits[b]["theta"][k] for k in ("k_ratio", "zeta", "eps", "red_chi2")} for b in BANDS})
        print(f"[{tag}] c = {c_used:.2f} um ({time.time()-t0:.1f} s)  " + "  ".join(f"{b}: k*/k {fits[b]['theta']['k_ratio']:.0f} zeta {fits[b]['theta']['zeta']:.4f}" for b in BANDS))
        for arm in ("EB", "EB+GP"):
            r = rows[tag][arm]; print(f"    {arm:6s} validation NRMSE: CR bands {r['nrmse_span']:.1f} %   " + "  ".join(f"{b} {r['nrmse_' + b]:.1f}" for b in BANDS))
        if fit_c: best_fits, best_c = fits, c_used
    # low-rank reference on the same validation set
    for r_ in (4, 5, 6):
        Zp, _ = recon.cross_capture(d.x_um, d.Z[0], v.x_um, arm="lowrank", rank=r_)
        sc = score(Zp, v.Z[0], f, bands_Hz, span); full = recon.nrmse(Zp, v.Z[0])
        per = "  ".join(f"{b} {sc['nrmse_' + b]:.1f}" for b in BANDS)
        print(f"    low-rank r={r_}: CR bands {sc['nrmse_span']:.1f} %  ({per});  full 100 Hz-2 MHz {full:.1f} %")
        rows[f"lowrank_r{r_}"] = dict(span=sc, full=full)
    # ---------------------------------------------------------------- 3. E along the lever from the live capture
    s = F.load("scmpitB_r2_bias")
    pre = axis.invols_interpolator(F.load("scmpitB_r2_preflight"))
    inv_fn = lambda x: float(np.atleast_1d(pre(x))[0]) * float(calib.factor_at(tl, "10_bias_survey", x)[0])
    t = domains.transfer_table(s, inv_fn, VAC); t["x_lever_um"] = t.x_um - x0_s
    x_stage_design = t.x_lever_um.values + x0_d                    # survey positions in the design frame
    qs = d.probe.qs_band_Hz
    c0 = json.loads(json.dumps(cal)); c0["P"]["c_um"] = best_c
    fits, _ = fit_capture(d, c0, x_stage_design, ["CR1"], extra_bands={"QS": tuple(qs)}, fit_c=False)
    def E_of(key):
        zc = np.abs(fits["CR1"][key]).max(1); zq = np.abs(fits["QS"][key]).mean(1); return zc / zq
    t["E_fmm"] = E_of("Zrec"); t["E_eb_live"] = E_of("Zeb")
    # piezo pathway only, one scale, live contact state (blind-EB analogue)
    P = dict(c0["P"]); P["log_alpha"] = fits["CR1"]["theta"]["log_alpha"]; zeta = fits["CR1"]["theta"]["zeta"]
    fcr = float(np.median(t.f_cr_Hz)); fq = np.linspace(qs[0], qs[1], 20)
    je = J.JointEB(x_stage_design, [J.Band("CR1", np.array([fcr]), None), J.Band("QS", fq, None)], cal["f0_hz"], n_modes=cal["n_modes"])
    (zp_c, _), (zp_q, _) = je.band_maps(P, [zeta, zeta])
    t["E_model_live"] = np.abs(zp_c[:, 0]) / np.abs(zp_q).mean(1)
    # archived blind-EB E (Fig. 6 central variant) for reference
    c6 = json.loads((F.config.RESULTS_DIR / "_cache_fig6.json").read_text())["R2"]["central"]
    st = eb_fit.ContactState("R2", np.zeros(3), 0.0, np.array([]), np.array([]), L_um=L_UM)
    t["E_eb_blind"] = eb_fit.enhancement(np.array(c6["p"]), st, t.x_lever_um.values)[0]
    contact = L_UM - c6["summary"]["setback_um"]; t["interior"] = t.x_lever_um < contact - 3
    # ---------------------------------------------------------------- 4. d33 routes (Fig. 6 CPD correction)
    cs = domains.cpd_split(t, mask=t.interior)
    t["d33_qs_raw"] = t.d33_qs; t["E_raw"] = t.E
    t["d33_qs"] = cs["d33_qs_corr"]; t["E_meas"] = cs["E_corr"]
    routes = {"quasi-static (corrected)": t.d33_qs.values}
    for key, lab in (("E_fmm", "CR1 / E from live EB+GP map"), ("E_eb_live", "CR1 / E from live EB (per-band gains)"),
                     ("E_model_live", "CR1 / E model, live contact state"), ("E_eb_blind", "CR1 / E blind EB (Fig. 6)")):
        t["d33_" + key] = cs["P_cr_corr"] / t[key]; routes[lab] = t["d33_" + key].values
    m = t.interior.values
    print(f"\ndomain CPD term delta = {1e3*cs['delta_V']:+.0f} mV; uniform QS d33 {cs['d33_uniform']:.2f} pm/V; interior positions {m.sum()} of {len(t)}")
    print("\nE along the lever (interior positions):")
    cols = ["x_lever_um", "E_meas", "E_fmm", "E_eb_live", "E_model_live", "E_eb_blind"]
    print(t.loc[m, cols].round(1).to_string(index=False))
    print("\nratio E_meas / E:", {k: f"{(t.E_meas / t[k])[m].median():.2f} (range {(t.E_meas / t[k])[m].min():.2f}-{(t.E_meas / t[k])[m].max():.2f})"
                                  for k in ("E_fmm", "E_eb_live", "E_model_live", "E_eb_blind")})
    print("\nd33 routes (pm/V, interior positions):")
    summ = []
    for lab, dd in routes.items():
        dd = np.asarray(dd)[m]
        summ.append(dict(route=lab, median=np.median(dd), spread=dd.max() / dd.min(), sd_pct=100 * dd.std() / np.median(dd)))
        print(f"  {lab:42s} median {np.median(dd):5.2f}  spread x{dd.max()/dd.min():.2f}  sd {100*dd.std()/np.median(dd):4.1f} %")
    # ---------------------------------------------------------------- 5. electrostatics removed: two-domain P map, leave-one-out
    # The live capture is single-domain at 0 V, so its E carries the electrostatic channel (+-35 % at the clamp end,
    # opposite sign on the two domains). A two-domain capture cancels it: P = (Z1 - Z2)/2 per frequency. Build that
    # map from the survey's own spectra and predict E at each position from the other seven (never-visited test).
    Pc = domains.decompose_series(s, band="CR1"); Pq = domains.decompose_series(s, band="QS")
    xs_stage = t.x_lever_um.values + x0_s
    c_s = json.loads(json.dumps(cal)); c_s["P"]["c_um"] = best_c + (x0_s - x0_d)
    loo = []
    for j in range(len(xs_stage)):
        keep = np.array([i for i in range(len(xs_stage)) if i != j])
        zc, fc = physrec.band_slice(Pc["freq_Hz"], Pc["P"], s.probe.bands_Hz["CR1"], n_max=250)
        oc = ebfit.rec_eb2(xs_stage, keep, zc[keep], fc, c_s, "CR1", x_eval=xs_stage[[j]])
        cq = json.loads(json.dumps(c_s)); cq["P"]["log_alpha"] = oc["theta"]["log_alpha"]
        zq, fq_ = physrec.band_slice(Pq["freq_Hz"], Pq["P"], tuple(qs), n_max=100)
        oq = ebfit.rec_eb2(xs_stage, keep, zq[keep], fq_, cq, "QS", free=("eps",), x_eval=xs_stage[[j]])
        E_eb = np.abs(oc["Zeb"][0]).max() / np.abs(oq["Zeb"][0]).mean()
        E_gp = np.abs(oc["Zrec"][0]).max() / np.abs(oq["Zrec"][0]).mean()
        E_true = np.abs(zc[j]).max() / np.abs(zq[j]).mean()
        conv = t.invols[j] / s.probe.amp_divisor / VAC * 1e12          # raw V -> pm/V
        loo.append(dict(x_lever_um=float(t.x_lever_um[j]), E_true=E_true, E_meas_raw=float(t.E_raw[j]), E_loo_eb=E_eb, E_loo_ebgp=E_gp,
                        d33_qs_raw=float(t.d33_qs_raw[j]), d33_loo_eb_raw=float(t.P_cr[j] * conv / E_eb), d33_loo_ebgp_raw=float(t.P_cr[j] * conv / E_gp),
                        d33_qs=float(t.d33_qs[j]), d33_loo_eb=float(cs["P_cr_corr"][j] / E_eb), d33_loo_ebgp=float(cs["P_cr_corr"][j] / E_gp),
                        k_ratio=oc["theta"]["k_ratio"], zeta=oc["theta"]["zeta"]))
    loo = pd.DataFrame(loo)
    print("\nTwo-domain P map, leave-one-out (E and d33 at each position predicted from the other seven):")
    print(loo[["x_lever_um", "E_true", "E_meas_raw", "E_loo_eb", "E_loo_ebgp", "d33_qs_raw", "d33_loo_eb_raw", "d33_loo_ebgp_raw", "d33_qs", "d33_loo_eb", "d33_loo_ebgp"]].round(2).to_string(index=False))
    for k in ("d33_loo_eb", "d33_loo_ebgp", "d33_loo_eb_raw"):
        dd = loo[k].values; print(f"  {k}: median {np.median(dd):.2f} pm/V, spread x{dd.max()/dd.min():.2f}, sd {100*dd.std()/np.median(dd):.1f} %;  E ratio true/pred median {np.median(loo.E_true/loo[k.replace('d33','E').replace('_raw','')]):.2f} (range {(loo.E_true/loo[k.replace('d33','E').replace('_raw','')]).min():.2f}-{(loo.E_true/loo[k.replace('d33','E').replace('_raw','')]).max():.2f})")
    out = dict(x0=dict(dense=X0_DENSE, design=x0_d, validation=x0_v, survey=x0_s), c_ruler=c_ruler, c_fitted=best_c,
               loo_two_domain=loo.to_dict("records"),
               validation=rows, cpd=dict(delta_V=cs["delta_V"], d33_uniform=cs["d33_uniform"]),
               table=t[["x_lever_um", "interior", "invols", "Q", "P_cr", "P_qs", "f_cr_Hz"] + cols[1:] + ["d33_qs_raw", "d33_qs"] + ["d33_" + k for k in ("E_fmm", "E_eb_live", "E_model_live", "E_eb_blind")]].to_dict("records"),
               d33_summary=summ)
    p = F.config.RESULTS_DIR / "live_ebgp_d33_stiff_r2.json"
    json.dump(out, open(p, "w"), indent=1, default=float); print("\nwrote", p)


if __name__ == "__main__":
    main()

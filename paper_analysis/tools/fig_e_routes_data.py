"""Data for the two-route E calibration figure (stiff probe B, run R2). Writes results/fig_e_routes.npz.

Beam-model route: the blind EB fit (Fig. 6 'central', cached) sees only CR1-CR3 frequencies, Q1 and the
shape of 1/InvOLS(x); E_model(x) = |z(f_CR1, x)| / |z(f_QS, x)| from its mode shapes.
Map route: the 8-position two-domain P map, EB+GP reconstructed over the CR1 and QS bands;
E_map(x) = max_f |P_CR1| / mean_f |P_QS|. Both are checked against the measured two-domain E at the
8 survey positions, then used to calibrate the CPD-corrected CR1 image at position A.

    python tools/fig_e_routes_data.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fmmpaper as F  # noqa: E402
from fmmpaper import calib, domains, ebfit, eb_fit, images, physrec, spectra  # noqa: E402
import fig5_d33 as F5  # noqa: E402

out = {}
cal = ebfit.load_calibration("stiff", "mass")
s, t, x0_s, tl, an = F5.survey_table()
xl = t.x_lever_um.values; xs_stage = xl + x0_s
live = json.load(open(F.config.RESULTS_DIR / "live_ebgp_d33_stiff_r2.json"))
c_s = json.loads(json.dumps(cal)); c_s["P"]["c_um"] = live["c_fitted"] + (x0_s - live["x0"]["design"])
L = F5.L_UM
out.update(x_survey=xl, L_um=L, x0_survey=x0_s, freq_Hz=s.freq_Hz)

# ---- raw spectra at the 8 positions (0 V, spot 1 and 2)
i1 = int(np.intersect1d(s.where(bias_V=0), s.where(spot=1))[0]); i2 = int(np.intersect1d(s.where(bias_V=0), s.where(spot=2))[0])
out.update(Z_spot1=s.Z[i1], Z_spot2=s.Z[i2])
fr = np.array([float(np.median(spectra.peaks_along_x(s.freq_Hz, s.Z[i1], s.probe.bands_Hz[b])["f_Hz"])) for b in ("CR1", "CR2", "CR3")])
out["bands_Hz"] = np.array([s.probe.bands_Hz[b] for b in ("CR1", "CR2", "CR3")]); out["qs_band_Hz"] = np.array(s.probe.qs_band_Hz)

# ---- beam-model route: blind EB fit (cached central), its inputs and outputs
c6 = json.load(open(F.config.RESULTS_DIR / "_cache_fig6.json"))["R2"]["central"]; p = np.array(c6["p"])
keep = xl < L - 10
st = eb_fit.ContactState("R2 survey", fr, float(np.nanmedian(t.Q)), xl[keep], t.invols.values[keep], L_um=L)
fk = eb_fit.peaks(p, st); Qm = eb_fit.q_model(p, st, fk[0])
xg = np.linspace(0, L, 400)
zq_g = eb_fit.static_shape(p, st, xg); zq_pts = eb_fit.static_shape(p, st, st.x_static_um)
y = 1 / st.invols_static; g = np.sum(y * zq_pts) / np.sum(zq_pts ** 2)
E_model_g, zqs, zcr, f1 = eb_fit.enhancement(p, st, xg)
E_model_pts = eb_fit.enhancement(p, st, xl)[0]
# model CR1 line shape at mid-lever (for the Q panel), and model spectrum for the frequency panel
m, r = eb_fit._model(p, st, np.linspace(fk[0] - 6e3, fk[0] + 6e3, 801)); ix = np.argmin(np.abs(m.xi - 0.55))
out.update(cr_meas_Hz=fr, cr_model_Hz=fk, Q_meas=st.Q1, Q_model=Qm, Q_all=t.Q.values,
           x_static=st.x_static_um, invols_static=st.invols_static, static_gain=g, x_grid=xg,
           static_model_grid=zq_g * g, z_qs_grid=zqs, z_cr1_grid=zcr, E_model_grid=E_model_g, E_model_pts=E_model_pts,
           Q_line_f=m.f_hz, Q_line_amp=np.abs(r["piezo"][:, ix]), Q_line_x=m.xi[ix] * L, p_blind=p)
mm, rr = eb_fit._model(p, st, np.linspace(150e3, 2.1e6, 3000)); out.update(model_spec_f=mm.f_hz, model_spec_amp=np.abs(rr["piezo"][:, ix]))

# ---- map route: two-domain P map over CR1 and QS bands, reconstructed on the grid
Pc = domains.decompose_series(s, band="CR1"); Pq = domains.decompose_series(s, band="QS")
zc, fc = physrec.band_slice(Pc["freq_Hz"], Pc["P"], s.probe.bands_Hz["CR1"], n_max=250)
zq, fq = physrec.band_slice(Pq["freq_Hz"], Pq["P"], tuple(s.probe.qs_band_Hz), n_max=100)
sel = np.arange(len(xs_stage)); xe = xg + x0_s
oc = ebfit.rec_eb2(xs_stage, sel, zc, fc, c_s, "CR1", x_eval=xe)
cq = json.loads(json.dumps(c_s)); cq["P"]["log_alpha"] = oc["theta"]["log_alpha"]
oq = ebfit.rec_eb2(xs_stage, sel, zq, fq, cq, "QS", free=("eps",), x_eval=xe)
E_map_g = np.abs(oc["Zrec"]).max(1) / np.abs(oq["Zrec"]).mean(1)
E_meas = np.abs(zc).max(1) / np.abs(zq).mean(1)
E_true, E_eb_loo, E_gp_loo, _ = F5.e_from_pmap(s, xs_stage, c_s)
out.update(P_cr1_meas=zc, f_cr1=fc, P_qs_meas=zq, f_qs=fq, P_cr1_rec=oc["Zrec"], P_qs_rec=oq["Zrec"],
           E_map_grid=E_map_g, E_meas=E_meas, E_loo=E_gp_loo, P_full=domains.decompose_series(s)["P"])

# ---- image at A: CPD-corrected CR1 / E (both routes), QS reference
root = F.config.data_root() / "DomainsB_SCMPIT_R2/50_quant_imaging"
J = json.loads((root / "image_cr1_quant.json").read_text()); ims = {im["name"]: im for im in J["images"]}
x0i = float(calib.x0_at(an, np.datetime64(J["started"].replace(" ", "T")))[0])
cs = domains.cpd_split(t, mask=np.ones(len(t), bool))
cplx = lambda a, b: t[a].values + 1j * t[b].values
rc = cplx("bc_re", "bc_im") / cplx("Pc_re", "Pc_im")
ri = lambda r_, x: np.interp(x, xl, r_.real) + 1j * np.interp(x, xl, r_.imag)
delta = cs["delta_V"]; Vbar = float(J["v_cpd_V"]); V1, V2 = Vbar - delta, Vbar + delta
X = 154.9 - x0i
rd = lambda n: images.read(root / ims[n]["path"].split(chr(92))[-1], ims[n]["drive_V"])
zc0 = rd("A_cr1_0V"); vname = next(n for n in ims if n.startswith("A_cr1_V") and abs(ims[n]["bias_V"] - Vbar) < 1e-6); zcv = rd(vname)
m1, m2 = images.domain_masks(zc0)
pc = images.cpd_correct(zcv, m1, m2, Vbar, V1, V2, ri(rc, X), delta)
Emap_A = float(np.interp(X, xg, E_map_g)); Emod_A = float(np.interp(X, xg, E_model_g))
e1, e2 = images.interior(m1), images.interior(m2)
out.update(img_cr1_raw=np.abs(zc0), img_cr1_corr=np.abs(pc), img_m1=m1, img_e1=e1, img_e2=e2, x_A=X, E_map_A=Emap_A, E_model_A=Emod_A,
           d33_spec=cs["d33_uniform"], img_scan_um=float(ims["A_cr1_0V"].get("scan_um", 6.0)))
d_map = np.abs(pc) / Emap_A; d_mod = np.abs(pc) / Emod_A
print(f"CR model {fk/1e3} meas {fr/1e3}; Q {Qm:.0f}/{st.Q1:.0f}; x_A {X:.1f}; E_map(A) {Emap_A:.0f}, E_model(A) {Emod_A:.0f}")
print("E_meas/E_map at pts", np.round(E_meas / np.interp(xl, xg, E_map_g), 3), "E_meas/E_model", np.round(E_meas / E_model_pts, 3))
print(f"image A d33 / E_map: {np.median(d_map[e1]):.2f} {np.median(d_map[e2]):.2f}; / E_model: {np.median(d_mod[e1]):.2f} {np.median(d_mod[e2]):.2f}; spec {cs['d33_uniform']:.2f}")
np.savez_compressed(F.config.RESULTS_DIR / "fig_e_routes.npz", **{k: np.asarray(v) for k, v in out.items()})
print("saved")

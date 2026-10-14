"""Fig. 3 data, peak-tracked and SEM-geometry version (2026-10-14). Writes results/fig_e_routes_sem.npz.

Starts from results/fig_e_routes.npz (raw spectra, image, frame) and replaces
 * the map route by a PEAK-TRACKED two-domain decomposition: every bias condition's CR1-band spectrum is
   frequency-aligned to the 0 V / domain-1 CR1 peak at that position before the per-frequency bias fit, so the
   intercept spectrum P(f) is not blurred by the 0.2-0.8 kHz bias shift of CR1 (Methods: peak-tracked E).
   The QS band is unchanged. Domain-potential correction P = P_raw - delta*b with delta from the QS channel
   (unchanged), EB+GP reconstruction of the CR1 and QS maps on the grid, leave-one-out, stationary d_eff.
 * the model route by the single SEM-geometry recipe (xprobe_semgeom_blind_v2 --geom sem, peak-tracked table):
   L, set-back, tip height fixed at SEM; k*, kcone, zeta from CR1-CR3 + Q1; static profile withheld (shown as a check).
   Evaluated in the survey lever frame (clamp at stage x0 = 24.1 um, static-ruler registration).
    python tools/fig_e_routes_tracked.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import fmmpaper as F  # noqa: E402
from fmmpaper import domains, ebfit, eb_fit, physrec, spectra, registry, io as fio  # noqa: E402
import eb_smooth  # noqa: E402,F401

R = F.config.RESULTS_DIR
D = dict(np.load(R / "fig_e_routes.npz", allow_pickle=True))
probe = registry.PROBES["scmpitB"]
s = fio.load_series_file(F.config.data_root() / "DomainsB_SCMPIT_R2/10_bias_survey/domains_bias_checkpoint_500nN_scmpit.npz", probe)
f = s.freq_Hz; x0_s = float(D["x0_survey"]); xl = D["x_survey"]; xs = xl + x0_s; xg = D["x_grid"]; n = len(xs)
assert np.allclose(s.x_um, xs, atol=0.2), (s.x_um, xs)
i1, V1, i2, V2 = domains.split_spots(s)
iref = int(np.intersect1d(s.where(bias_V=0.0), i1)[0])
band = probe.bands_Hz["CR1"]; qs_band = tuple(probe.qs_band_Hz)
m_cr = (f >= band[0] - 5e3) & (f <= band[1] + 5e3)

# ---- peak-aligned CR1-band spectra
Zal = s.Z.copy()
shifts = np.zeros((s.Z.shape[0], n)); fref = np.zeros(n); Qtr = np.zeros(n)
for j in range(n):
    pk0 = spectra.peak(f, s.Z[iref, j], band); fref[j] = pk0["f_Hz"]; Qs = []
    for c in range(s.Z.shape[0]):
        pk = spectra.peak(f, s.Z[c, j], band); df = fref[j] - pk["f_Hz"]; shifts[c, j] = df; Qs.append(pk["Q"])
        z = s.Z[c, j, m_cr]; fb = f[m_cr]
        Zal[c, j, m_cr] = np.interp(fb - df, fb, z.real) + 1j * np.interp(fb - df, fb, z.imag)
    Qtr[j] = float(np.nanmedian(Qs))
dec = domains.decompose(V1, Zal[i1], V2, Zal[i2]); Praw, b = dec["P"], dec["b"]
dec0 = domains.decompose(V1, s.Z[i1], V2, s.Z[i2])
delta = float(D["delta_V"]); Pcor = Praw - delta * b; Pcor0 = dec0["P"] - delta * dec0["b"]
mq = (f >= qs_band[0]) & (f <= qs_band[1]); mb = (f >= band[0]) & (f <= band[1])
E_meas_tracked = np.abs(Pcor[:, mb]).max(1) / np.abs(Pcor[:, mq]).mean(1)
E_meas_fixed = np.abs(Pcor0[:, mb]).max(1) / np.abs(Pcor0[:, mq]).mean(1)
print("CR1 shift between conditions (Hz, ptp per position):", np.round(np.ptp(shifts, axis=0), 0))
print("E tracked", np.round(E_meas_tracked, 0)); print("E fixed-spectrum", np.round(E_meas_fixed, 0), "ratio", np.round(E_meas_tracked / E_meas_fixed, 3))

# ---- reconstruction of the aligned, corrected maps
zc, fc = physrec.band_slice(f, Pcor, band, n_max=250); zq, fq = physrec.band_slice(f, Pcor, qs_band, n_max=100)
cal = ebfit.load_calibration("stiff", "mass"); live = json.load(open(R / "live_ebgp_d33_stiff_r2.json"))
c_s = json.loads(json.dumps(cal)); c_s["P"]["c_um"] = live["c_fitted"] + (x0_s - live["x0"]["design"])


def recon(keep, xe):
    oc = ebfit.rec_eb2(xs, keep, zc[keep], fc, c_s, "CR1", x_eval=xe)
    cq = json.loads(json.dumps(c_s)); cq["P"]["log_alpha"] = oc["theta"]["log_alpha"]
    oq = ebfit.rec_eb2(xs, keep, zq[keep], fq, cq, "QS", free=("eps",), x_eval=xe)
    return oc["Zrec"], oq["Zrec"]


Zc_g, Zq_g = recon(np.arange(n), xg + x0_s)
E_map = np.abs(Zc_g).max(1) / np.abs(Zq_g).mean(1)
E_meas_dec = np.abs(zc).max(1) / np.abs(zq).mean(1)        # on the decimated grid (reconstruction convention)
E_loo = np.zeros(n)
for j in range(n):
    keep = np.array([i for i in range(n) if i != j]); a, q = recon(keep, xs[[j]])
    E_loo[j] = np.abs(a[0]).max() / np.abs(q[0]).mean()
inv = np.interp(xl, D["x_static"], D["invols_static"]) if False else None
T = pd.DataFrame(json.load(open(R / "fig6_main.json"))["values"]["runs"]["R2"]["table"])
conv = T.invols.values / 32.0 * 1e12
d33_qs = np.abs(zq).mean(1) * conv
d33_cr = np.abs(zc).max(1) * conv / E_loo
print("E_meas (decimated)/E_loo", np.round(E_meas_dec / E_loo, 3), "| median", np.median(E_meas_dec / E_loo).round(3))
print("stationary QS", np.round(d33_qs, 2), "median", np.median(d33_qs).round(2)); print("CR1/E_loo", np.round(d33_cr, 2), "median", np.median(d33_cr).round(2))

# ---- model route: SEM recipe in the survey frame
J = json.load(open(R / "r3r4/xprobe_semgeom_v2_sem_tracked.json"))["SCM-PIT R2 (500 nN)"]
p = np.array([J["params"][k] for k in eb_fit.PARAMS[:5]]); L = float(J["L_um"]); cr = np.array(J["cr_Hz"])
st = eb_fit.ContactState("R2 SEM", cr, float(J["Q1"]), np.array([]), np.array([]), L_um=L)
fk = eb_fit.peaks(p, st); Qm = eb_fit.q_model(p, st, fk[0])
E_model_g, zqs, zcr, f1 = eb_fit.enhancement(p, st, xg); E_model_pts = eb_fit.enhancement(p, st, xl)[0]
psl = p.copy(); psl[1] = -2.0
zq_g = eb_fit.static_shape(psl, st, xg); zq_pts = eb_fit.static_shape(psl, st, D["x_static"]); y = 1 / D["invols_static"]; g = np.sum(y * zq_pts) / np.sum(zq_pts ** 2)
static_rms = float(np.sqrt(np.mean(np.log(y / (g * zq_pts)) ** 2)))
m_, r_ = eb_fit._model(p, st, np.linspace(fk[0] - 6e3, fk[0] + 6e3, 801)); ix = np.argmin(np.abs(m_.xi - 0.55))
ratio = E_meas_tracked / E_model_pts
d33_cr_model = np.abs(zc).max(1) * conv / E_model_pts
print(f"model CR {np.round(fk/1e3,1)} meas {np.round(cr/1e3,1)} err % {np.round(100*(fk/cr-1),1)}; Q {Qm:.0f}/{J['Q1']:.0f}; static check rms {100*static_rms:.1f} %")
print("E_meas(tracked)/E_model", np.round(ratio, 3), "| interior median", np.median(ratio[:-1]).round(3), "range", ratio[:-1].min().round(3), ratio[:-1].max().round(3))
print("CR1/E_model d_eff", np.round(d33_cr_model, 2), "| median", np.median(d33_cr_model).round(2), "sd %", (100 * d33_cr_model.std() / d33_cr_model.mean()).round(1), "| QS uniform", float(D["d33_uniform"]))

# ---- image at A
xA = float(D["x_A"]); E_map_A = float(np.interp(xA, xg, E_map)); E_model_A = float(np.interp(xA, xg, E_model_g))
img = D["img_cr1_corr"] / E_map_A; d1, d2 = float(np.median(img[D["img_e1"]])), float(np.median(img[D["img_e2"]]))
E_img = float(D["E_image_A"]); print(f"E_map(A) {E_map_A:.0f} (was {float(D['E_map_cor_A']):.0f}); E_model(A) {E_model_A:.0f}; image domains {d1:.2f}/{d2:.2f}; E_image/E_survey {E_img/E_map_A:.3f}; implied {np.mean([d1,d2])/(E_img/E_map_A):.2f}")

D.update({k: np.asarray(v) for k, v in dict(
    P_cr1_cor=zc, P_qs_cor=zq, P_cr1_rec_cor=Zc_g, P_qs_rec_cor=Zq_g, E_map_cor_grid=E_map, E_meas_cor=E_meas_tracked, E_meas_dec=E_meas_dec,
    E_meas_fixed=E_meas_fixed, E_loo_cor=E_loo, d33_qs_stat=d33_qs, d33_cr_stat=d33_cr, E_map_cor_A=E_map_A, E_survey_A=E_map_A,
    img_d1=d1, img_d2=d2, cr1_shift_ptp_Hz=np.ptp(shifts, axis=0), Q_tracked=Qtr,
    cr_model_Hz=fk, Q_model=Qm, Q_meas=float(J["Q1"]), E_model_grid=E_model_g, E_model_pts=E_model_pts, E_model_A=E_model_A,
    z_qs_grid=zqs, z_cr1_grid=zcr, static_model_grid=zq_g * g, static_check_rms=static_rms, Q_line_f=m_.f_hz, Q_line_amp=np.abs(r_["piezo"][:, ix]),
    p_blind=p, ratio_model=ratio, d33_cr_model=d33_cr_model, L_sem_um=L, sem_setback_um=p[2], sem_tip_h_um=p[4]).items()})
np.savez_compressed(R / "fig_e_routes_sem.npz", **D)
json.dump(dict(E_meas_tracked=E_meas_tracked.tolist(), E_meas_fixed=E_meas_fixed.tolist(), E_loo=E_loo.tolist(), E_map_grid_at_stops=np.interp(xl, xg, E_map).tolist(),
               d33_qs_stat=d33_qs.tolist(), d33_cr_loo=d33_cr.tolist(), E_model_pts=E_model_pts.tolist(), ratio_model=ratio.tolist(), d33_cr_model=d33_cr_model.tolist(),
               cr_model_Hz=fk.tolist(), cr_meas_Hz=cr.tolist(), Q_model=Qm, Q_meas=float(J["Q1"]), static_check_rms_pct=100 * static_rms,
               E_map_A=E_map_A, E_model_A=E_model_A, img_d1=d1, img_d2=d2, E_image_A=E_img, x_lever=xl.tolist(), x0_stage=x0_s),
          open(R / "fig_e_routes_sem_summary.json", "w"), indent=1)
print("saved results/fig_e_routes_sem.npz")

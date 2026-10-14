"""Domain-CPD-corrected map route for the two-route E figure (adds to results/fig_e_routes.npz).

P_raw = (a1 - a2)/2 from the two-domain bias fit removes the electrostatic term common to both domains;
a domain-dependent contact potential leaves delta*b(x, f) in P_raw. delta is fitted in the QS channel
(uniform d33, domains.cpd_split) and removed at every frequency: P_corr = P_raw - delta*b.
The CR1 and QS maps, E_map and a leave-one-out matched validation are rebuilt from P_corr.

Input results/r2_bias_Pb.npz: per-frequency P_raw, b from the R2 bias survey (same linear fit as
domains.decompose), exported because the data drive was not mounted.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fmmpaper as F  # noqa: E402
from fmmpaper import domains, ebfit, physrec  # noqa: E402

R = F.config.RESULTS_DIR
D = dict(np.load(R / "fig_e_routes.npz", allow_pickle=True))
B = np.load(R / "r2_bias_Pb.npz")
T = pd.DataFrame(json.load(open(R / "fig6_main.json"))["values"]["runs"]["R2"]["table"])
assert np.allclose(B["x_um"], T.x_um.values)
Praw, b, f = B["P"].astype(complex), B["b"].astype(complex), B["freq_Hz"]
print("P_raw vs saved P_full: max rel diff", float(np.max(np.abs(Praw - D["P_full"])) / np.max(np.abs(D["P_full"]))))
cs = domains.cpd_split(T, mask=np.ones(len(T), bool)); delta = cs["delta_V"]
Pcor = Praw - delta * b
bands = {"CR1": tuple(D["bands_Hz"][0]), "QS": tuple(D["qs_band_Hz"])}
zc_r, fc = physrec.band_slice(f, Praw, bands["CR1"], n_max=250); zq_r, fq = physrec.band_slice(f, Praw, bands["QS"], n_max=100)
zc, _ = physrec.band_slice(f, Pcor, bands["CR1"], n_max=250); zq, _ = physrec.band_slice(f, Pcor, bands["QS"], n_max=100)
cal = ebfit.load_calibration("stiff", "mass")
live = json.load(open(R / "live_ebgp_d33_stiff_r2.json")); x0_s = float(D["x0_survey"])
c_s = json.loads(json.dumps(cal)); c_s["P"]["c_um"] = live["c_fitted"] + (x0_s - live["x0"]["design"])
xl = D["x_survey"]; xs = xl + x0_s; xg = D["x_grid"]; n = len(xs)


def recon(keep, xe):
    oc = ebfit.rec_eb2(xs, keep, zc[keep], fc, c_s, "CR1", x_eval=xe)
    cq = json.loads(json.dumps(c_s)); cq["P"]["log_alpha"] = oc["theta"]["log_alpha"]
    oq = ebfit.rec_eb2(xs, keep, zq[keep], fq, cq, "QS", free=("eps",), x_eval=xe)
    return oc["Zrec"], oq["Zrec"]


Zc_g, Zq_g = recon(np.arange(n), xg + x0_s)
E_map = np.abs(Zc_g).max(1) / np.abs(Zq_g).mean(1)
E_meas = np.abs(zc).max(1) / np.abs(zq).mean(1)
E_loo = np.zeros(n)
for j in range(n):
    keep = np.array([i for i in range(n) if i != j]); a, q = recon(keep, xs[[j]])
    E_loo[j] = np.abs(a[0]).max() / np.abs(q[0]).mean()
conv = T.invols.values / 32.0 * 1e12                        # pm per V of drive, Vac = 1 V
d33_qs = np.abs(zq).mean(1) * conv                          # stationary QS, corrected
d33_cr = np.abs(zc).max(1) * conv / E_loo                   # stationary CR1 / E_map predicted from the other 7
E_map_A = float(np.interp(D["x_A"], xg, E_map))
img = D["img_cr1_corr"] / E_map_A
imgs = pd.DataFrame(json.load(open(R / "fig6_main.json"))["values"]["images"]); ia = imgs[(imgs.run == "R2") & (imgs.pos == "A")].iloc[0]
out = dict(P_raw_spec=Praw, P_cor_spec=Pcor, delta_V=delta, d33_uniform=cs["d33_uniform"], P_cr1_cor=zc, P_qs_cor=zq, P_cr1_raw=zc_r, P_qs_raw=zq_r,
           P_cr1_rec_cor=Zc_g, P_qs_rec_cor=Zq_g, E_map_cor_grid=E_map, E_meas_cor=E_meas, E_loo_cor=E_loo,
           d33_qs_stat=d33_qs, d33_cr_stat=d33_cr, E_map_cor_A=E_map_A,
           img_d1=float(np.median(img[D["img_e1"]])), img_d2=float(np.median(img[D["img_e2"]])),
           E_image_A=float(ia.E_image), E_survey_A=float(ia.E_survey))
D.update({k: np.asarray(v) for k, v in out.items()})
np.savez_compressed(R / "fig_e_routes.npz", **D)
print(f"delta {delta:.3f} V, uniform d33 {cs['d33_uniform']:.2f}")
print("E_meas/E_loo (corrected)", np.round(E_meas / E_loo, 3))
print("E_meas/E_model", np.round(E_meas / D["E_model_pts"], 3))
print("d33 stationary QS", np.round(d33_qs, 2)); print("d33 stationary CR1/E_loo", np.round(d33_cr, 2))
print(f"medians QS {np.median(d33_qs):.2f}, CR1/E_loo {np.median(d33_cr):.2f}; E_map(A) {E_map_A:.0f}; image {out['img_d1']:.2f}/{out['img_d2']:.2f}; image E {ia.E_image:.0f} vs survey {ia.E_survey:.0f}")

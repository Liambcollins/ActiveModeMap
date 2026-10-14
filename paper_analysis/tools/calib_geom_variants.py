"""Dense-map lever calibration (as calibrate_lever.py, stiff/R2, mass free) with geometry variants (2026-10-13).
  --fix k=v ...   hold shared parameters fixed (e.g. L_um=227.1 tip_h_um=11.7 setback_um=10.25)
  --apex          vertical contact acts at the tilted tip APEX: b_vert = cos(t)*phi(xc) - h*sin(t)*phi'(xc)
                  (package model applies it at the tip base; with 11 deg tilt the apex sits h*sin(t) towards the clamp)
Writes results/lever_calibration_stiff_<tag>.json (same format as calibrate_lever.py).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import fmmpaper as F
from fmmpaper import jointeb as J, physrec, recon, spectra
import calibrate_lever as CL
ap = argparse.ArgumentParser(); ap.add_argument("--fix", nargs="*", default=[]); ap.add_argument("--apex", action="store_true")
ap.add_argument("--tag", required=True); ap.add_argument("--max_nfev", type=int, default=300); ap.add_argument("--probe", default="stiff")
a = ap.parse_args()
if a.apex:
    _orig = J.JointEB._matrices
    def _m(self, P):
        K, M, c1, k1, bv, fe = _orig(self, P)
        L = P["L_um"]; xi_c = 1.0 - P["setback_um"] / L; h = P["tip_h_um"] / L
        _, phc, dphc = J.beam_modes([xi_c], self.n); phc, dphc = phc[:, 0], dphc[:, 0]
        bv2 = np.cos(self.tilt) * phc - h * np.sin(self.tilt) * dphc
        alpha = 10 ** P["log_alpha"]; k1 = 3.0 * alpha
        K = K - k1 * np.outer(bv, bv) + k1 * np.outer(bv2, bv2)
        c1 = np.sqrt(k1) / 10 ** P["log_Qc"]
        return K, M, c1, k1, bv2, fe
    J.JointEB._matrices = _m
cfg = CL.PROBES[a.probe]; fixed = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.fix}
J.PER_BAND["dlf"] = (0.0, -0.04, 0.04)
s = F.load(cfg["key"]); x = s.x_um
bands = []
for b in cfg["bands"]:
    z, f = physrec.band_slice(s.freq_Hz, s.Z[0], s.probe.bands_Hz[b], n_max=250); bands.append(J.Band(b, f, z))
je = J.JointEB(x, bands, cfg["f0_hz"], n_modes=14)
sel = recon.select_equispaced(x, 30); ho = recon.held_out(len(x), sel)
zeta0 = []
for b in bands:
    pk = spectra.peaks_along_x(b.f_Hz, b.Z[sel], (b.f_Hz[0], b.f_Hz[-1])); zeta0.append(float(np.clip(0.5 / np.nanmedian(pk["Q"]), 1e-4, 0.05)))
free = tuple(k for k in CL.GEO + ("mu", "J") if k not in fixed)
bounds = dict(cfg["bounds"]); bounds.update(mu=(0.0, 0.1), J=(0.0, 0.02)); bounds["setback_um"] = (0.3, 25.0); bounds["tip_h_um"] = (3.0, 20.0)
fx = {"mu": cfg["mu"], "J": 0.0}; fx.update(fixed)
spec = J.Spec(free=free, fixed=fx, band_freq=True, bounds=bounds)
starts = []
for sb in (cfg["seeds"]["setback_um"] if "setback_um" not in fixed else (fixed["setback_um"],)):
    for la in cfg["seeds"]["log_alpha"]:
        st = dict(cfg["start"]); st.update({k: v for k, v in fixed.items()})
        if "setback_um" not in fixed: st["setback_um"] = sb if "L_um" not in fixed else sb + 7.0
        if "tip_h_um" not in fixed and "L_um" in fixed: st["tip_h_um"] = 11.7
        if "L_um" in fixed: st["L_um"] = fixed["L_um"]
        st.update(log_alpha=la, log_Qc=2.5, mu=cfg["mu"], J=1e-4, log_zeta=list(np.log10(zeta0)), eps=[0.0] * len(bands))
        starts.append(st)
t0 = time.time(); fit = je.fit(spec, sel, starts=starts, max_nfev=a.max_nfev)
maps, gains = je.predict(fit["P"], fit["zeta"], fit["eps"], sel, x=x, dlf=fit["dlf"])
rows = []
for b, Mp in zip(bands, maps):
    D = b.Z; e = float(np.sqrt(np.mean(np.abs(Mp[ho] - D[ho]) ** 2) / np.mean(np.abs(D[ho]) ** 2)))
    pm = spectra.peaks_along_x(b.f_Hz, Mp, (b.f_Hz[0], b.f_Hz[-1])); pd_ = spectra.peaks_along_x(b.f_Hz, D, (b.f_Hz[0], b.f_Hz[-1]))
    nm = spectra.nodes_from_profile(x, pm["amp"]); nd = [n for n in spectra.nodes_from_profile(x, pd_["amp"]) if x.min() + 5 <= n <= x.max() - 5]
    ne = [min(abs(t - q) for q in nm) for t in nd] if nm else [np.inf] * len(nd)
    rows.append(dict(band=b.name, nrmse=e, nodes_dense=[round(v, 1) for v in nd], node_err_um=[round(v, 2) for v in ne]))
P = {k: float(v) for k, v in fit["P"].items()}
tipx = P["L_um"] - P["setback_um"]; teff = tipx - (P["tip_h_um"] * np.tan(np.deg2rad(11.0)) if a.apex else 0.0)
print(f"[{a.tag}] {time.time()-t0:.0f}s cost {fit['cost']:.1f} | L {P['L_um']:.1f} sb {P['setback_um']:.2f} h {P['tip_h_um']:.2f} c {P['c_um']:.2f} "
      f"kcone/k {10**P['log_kcone']:.1f} k*/k {10**P['log_alpha']:.0f} mu {P['mu']:.4f} J {P['J']:.2e} | tip base from clamp {tipx:.1f}, vertical load point {teff:.1f} | "
      + " ".join(f"{r['band']} {r['nrmse']:.3f} nodes±{r['node_err_um']}" for r in rows), flush=True)
out = dict(probe=a.probe, dataset=cfg["key"], f0_hz=cfg["f0_hz"], n_modes=14, N_fit=30, sel=list(map(int, sel)), free=list(free), fixed=fx, apex=a.apex,
           P=P, zeta=[float(v) for v in fit["zeta"]], eps=[float(v) for v in fit["eps"]], dlf=[float(v) for v in fit["dlf"]], bands=cfg["bands"], score=rows, cost=fit["cost"])
json.dump(out, open(F.config.RESULTS_DIR / f"lever_calibration_{a.probe}_{a.tag}.json", "w"), indent=1)

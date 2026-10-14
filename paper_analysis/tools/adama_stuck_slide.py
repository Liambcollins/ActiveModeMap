"""Adama R1: lateral pinning of the diamond tip, and what it does to the QS calibration.

Contact state from the frequencies-only blind fit (SEM geometry fixed; results/adama/adama_R1_semgeom_nostatic_blind_fit.json):
k*/k ~ 1270, k_cone/k ~ 440 -- a laterally PINNED tip. The force-curve InvOLS ruler is taken with the tip sliding
(the sample pushes the tip up; the beam is point-loaded), so its shape is the k_cone -> 0 static shape of the same
lever. Here:
  1. the lever frame x0 (stage um of the clamp) is fitted from the pre-flight InvOLS shape with the SLIDING model;
  2. the stuck/sliding static-shape ratio r(x) = phi_stuck(x) / phi_slide(x) (displacement detection) is computed;
  3. QS d33 is re-calibrated: d33 = |P_qs| InvOLS_fc / G / r(x);
  4. the blind E(x) (stuck model) is scored against the peak-tracked survey E, and CR1 d33 = |P_cr| InvOLS_fc / G / r / E_model.
    python tools/adama_stuck_slide.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from fmmpaper import eb_fit, registry, config  # noqa: E402

OUT = HERE.parent / "results/adama"; probe = registry.PROBES["adama"]; G = probe.amp_divisor
fit = json.load(open(OUT / "adama_R1_semgeom_nostatic_blind_fit.json")); p = np.array(fit["p"]); L = fit["L_um"]
tab = pd.read_csv(OUT / "adama_R1_semgeom_nostatic_table.csv")
pf = json.load(open(config.data_root() / "DomainsB_ADAMA_R1/01_preflight/preflight_result.json"))
ib = pf["invols_by_x"]; xs_pf = np.array([float(k) for k in ib]); inv_pf = np.array([ib[k] for k in ib]); o = np.argsort(xs_pf); xs_pf, inv_pf = xs_pf[o], inv_pf[o]
st = eb_fit.ContactState("R1", np.array(fit["cr_Hz"]), fit["Q1"], np.array([]), np.array([]), f0_free_Hz=fit["f0_free_Hz"], k_lever=fit["k_lever"], L_um=L)
p_slide = p.copy(); p_slide[1] = -2.0          # k_cone/k = 0.01: laterally free tip

def static(pp, x): return eb_fit.static_shape(pp, st, x)
def rms_static(x0, pp):
    xl = xs_pf - x0; m = xl < L - 21.2 - 2.0     # ruler points beyond the tip excluded (laser past the tip)
    y = 1 / inv_pf[m]; z = static(pp, xl[m]); g = np.sum(y * z) / np.sum(z * z)
    return float(np.sqrt(np.mean(np.log(y / (g * z)) ** 2))), int(m.sum())

print("x0 scan (static ruler vs model), rms log-residual %:   x0   sliding   stuck")
scan = []
for x0 in np.arange(-30, 2.1, 2.0):
    rs, n = rms_static(x0, p_slide); rk, _ = rms_static(x0, p); scan.append((x0, rs, rk, n))
    print(f"   {x0:6.1f}   {100*rs:6.2f}   {100*rk:6.2f}   (n={n})")
x0 = min(scan, key=lambda r: r[1])[0]
print(f"lever frame: clamp at stage {x0:+.1f} um (sliding model); SEM geometry puts the pointed end at stage {x0 + L:.1f} um, the tip at {x0 + L - 21.2:.1f} um")

xl = tab.x_um.values - x0
phi_stuck = static(p, xl); phi_slide = static(p_slide, xl)
# normalise both shapes to unit TIP displacement (what the sample imposes in both experiments)
xt = L - 21.2
r = (phi_stuck / static(p, np.array([xt]))[0]) / (phi_slide / static(p_slide, np.array([xt]))[0])
E_model = eb_fit.enhancement(p, st, xl)[0]
d33_qs_raw = tab.d33_qs.values; d33_qs_corr = d33_qs_raw / r
d33_cr1 = tab.P_cr.values * tab.invols.values / G * 1e12 / r / E_model
print("\n stage   lever   r=stuck/slide   d33_qs raw   d33_qs /r    E_trk   E_model  E_trk/E_mod   d33_cr1 (=P_cr InvOLS/G/r/E_model)")
for i in range(len(xl)):
    print(f"{tab.x_um[i]:6.1f} {xl[i]:7.1f}   {r[i]:8.3f}      {d33_qs_raw[i]:7.2f}     {d33_qs_corr[i]:7.2f}   {tab.E[i]:7.1f} {E_model[i]:8.1f}   {tab.E[i]/E_model[i]:6.3f}     {d33_cr1[i]:6.2f}")
k = xl < xt - 8      # interior: exclude the stop at/after the tip
def spread(v): return float(v[k].max() / v[k].min())
print(f"\ninterior ({k.sum()} stops): QS raw median {np.median(d33_qs_raw[k]):.2f} pm/V spread x{spread(d33_qs_raw):.2f}; "
      f"stuck/slide-corrected median {np.median(d33_qs_corr[k]):.2f} spread x{spread(d33_qs_corr):.2f}, sd {100*np.std(d33_qs_corr[k])/np.mean(d33_qs_corr[k]):.1f} %; "
      f"E_trk/E_model median {np.median((tab.E.values/E_model)[k]):.3f} range {(tab.E.values/E_model)[k].min():.2f}-{(tab.E.values/E_model)[k].max():.2f}; "
      f"CR1 d33 median {np.median(d33_cr1[k]):.2f} spread x{spread(d33_cr1):.2f}")
xg = np.linspace(20, L - 1, 400)
out = tab.assign(x_lever_um=xl, r_stuck_slide=r, d33_qs_stuckcorr=d33_qs_corr, E_model_stuck=E_model, d33_cr1_stuck=d33_cr1, used=k)
out.to_csv(OUT / "adama_R1_stuck_slide_table.csv", index=False)
json.dump(dict(x0_stage_um=float(x0), L_um=L, tip_stage_um=float(x0 + xt), p_stuck=p.tolist(), p_slide=p_slide.tolist(), params=fit["params"], summary=fit["summary"],
               x_grid=xg.tolist(), r_grid=((static(p, xg) / static(p, np.array([xt]))[0]) / (static(p_slide, xg) / static(p_slide, np.array([xt]))[0])).tolist(),
               E_grid=eb_fit.enhancement(p, st, xg)[0].tolist(), scan=[dict(x0=a, rms_slide=b, rms_stuck=c) for a, b, c, _ in scan]),
          open(OUT / "adama_R1_stuck_slide.json", "w"), indent=1)
print("wrote", OUT / "adama_R1_stuck_slide_table.csv")

"""Adama R1 (diamond tip): transfer table from the two-domain survey + blind Euler-Bernoulli fit.

Inputs (all R1, 30 Sep 2026, 750 nN):
  * 10_bias_survey/domains_bias_checkpoint_500nN_scmpit.npz   two-domain survey, 8 positions
  * 01_preflight/preflight_result.json                       CR1-CR3 at the free end, InvOLS at 20 positions
The blind fit sees CR1-CR3, Q1 and the SHAPE of 1/InvOLS(x) (pre-flight ruler, 20 positions); the lever-frame
origin x0 (stage um of the clamp) is scanned, because stage != lever on this probe. L is the SEM value (246.7 um).
Amplitudes and E are withheld and used only to score the result.

    python tools/adama_blind_fit.py [--mass 0.03] [--f0 58410]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from fmmpaper import registry, domains, eb_fit, io as fio, config  # noqa: E402
import eb_smooth  # noqa: E402,F401  smooth peak/Q evaluation (2026-10-13)

ap = argparse.ArgumentParser(); ap.add_argument("--run", default="R1"); ap.add_argument("--x0_grid", default="-30,5,2.5")
ap.add_argument("--L", type=float, default=246.7); ap.add_argument("--tag", default=""); ap.add_argument("--starts", type=int, default=6); ap.add_argument("--free_f0", action="store_true"); ap.add_argument("--fix_geom", action="store_true", help="setback and tip height fixed at the SEM values (21.2 / 14.6 um)"); ap.add_argument("--sig_static", type=float, default=0.02); ap.add_argument("--nfev", type=int, default=40)
a = ap.parse_args()
root = config.data_root(); R = root / f"DomainsB_ADAMA_{a.run}"; OUT = HERE.parent / "results/adama"; OUT.mkdir(exist_ok=True)
probe = registry.PROBES["adama"]
pf = json.load(open(R / "01_preflight/preflight_result.json"))
ib = pf["invols_by_x"]; xs_pf = np.array([float(k) for k in ib]); inv_pf = np.array([ib[k] for k in ib]); o = np.argsort(xs_pf); xs_pf, inv_pf = xs_pf[o], inv_pf[o]
inv_fn = lambda x: float(np.exp(np.interp(float(x), xs_pf, np.log(inv_pf))))
cr = np.array([pf["cr1_Hz"], pf["cr2_Hz"], pf["cr3_Hz"]], float)
k_lever = pf["k_N_per_m"]; f0 = pf["f0_free_Hz"]

s = fio.load_series_file(R / "10_bias_survey/domains_bias_checkpoint_500nN_scmpit.npz", probe)
t_fix = domains.transfer_table(s, inv_fn, 1.0, cr_band="CR1")
t_trk = domains.transfer_table(s, inv_fn, 1.0, cr_band="CR1", track=True)
Q1 = float(np.nanmedian(t_trk.Q.values[t_trk.x_um < 220]))
print(f"{a.run}: CR {cr/1e3} kHz, k {k_lever}, f0 {f0/1e3} kHz, Q1 (survey median, interior) {Q1:.0f}")
print(t_trk[["x_um", "invols", "f_cr_Hz", "f_cr_spread_Hz", "Q", "E", "d33_qs", "ratio_qs", "vcpd_qs", "flip_qs"]].round(3).to_string())
print("fixed-frequency E:", np.round(t_fix.E.values, 1))

# ---- blind fit with the clamp position scanned (static ruler excludes the last 10 um: laser past the tip)
lo, hi, st = [float(v) for v in a.x0_grid.split(",")]
res = []
for x0 in np.arange(lo, hi + 1e-9, st):
    xl = xs_pf - x0; keep = xl < a.L - 3.0
    stt = eb_fit.ContactState(f"Adama {a.run} x0={x0:g}", cr, Q1, xl[keep], inv_pf[keep], f0_free_Hz=f0, k_lever=k_lever, L_um=a.L)
    sb0, th0 = (21.2, 14.6) if a.fix_geom else (8.0, 14.6)
    starts = [[la, lk, sb0, np.log10(3e-3), th0] for la in (2.7, 3.0, 3.3) for lk in (2.0, 3.0)][:a.starts]
    lo, hi = eb_fit.LO.copy(), eb_fit.HI.copy()
    if a.fix_geom: lo[2], hi[2], lo[4], hi[4] = 21.2 - 1e-3, 21.2 + 1e-3, 14.6 - 1e-3, 14.6 + 1e-3
    r = eb_fit.fit(stt, use_nodes=False, starts=starts, free_f0=a.free_f0, lo=lo, hi=hi, sig_f=0.015, sig_static=a.sig_static, sig_Q=0.05, max_nfev=a.nfev)
    if r is None: continue
    sm = eb_fit.summarize(r.x, stt); res.append((x0, r.cost, r.x, sm))
    print(f"  x0 {x0:6.1f}: cost {r.cost:8.3f}  CR err % {sm['cr_err_pct']}  static rms {sm['static_rms_pct']:.1f} %  Q {sm['Q_model']:.0f}/{Q1:.0f}  "
          f"k*/k {sm['k_ratio']:.0f} f0s {sm['f0_scale']:.3f} kcone/k {sm['kcone_ratio']:.1f} setback {sm['setback_um']:.1f} tip_h {sm['tip_height_um']:.1f} zeta {sm['zeta']:.2e}")
x0, cost, p, sm = min(res, key=lambda r: r[1])
print("\nBEST x0 =", x0, json.dumps({k: v for k, v in sm.items() if k not in ("nodes_model_um", "nodes_meas_um")}, default=float))
xl_s = s.x_um - x0
stt = eb_fit.ContactState("best", cr, Q1, xs_pf - x0, inv_pf, f0_free_Hz=f0, k_lever=k_lever, L_um=a.L)
E_model, zq, zc, f1 = eb_fit.enhancement(p, stt, xl_s); Qm = eb_fit.q_model(p, stt, f1)
xg = np.linspace(max(5, xl_s.min() - 20), a.L - 1, 300); E_g = eb_fit.enhancement(p, stt, xg)[0]
tab = t_trk.assign(x_lever_um=xl_s, E_fixed=t_fix.E.values, E_model=E_model, M_model=E_model / Qm, Q_model=Qm,
                   d33_cr1_model=t_trk.P_cr.values * t_trk.invols.values / probe.amp_divisor * 1e12 / E_model)
print("\nscore (withheld): x_lever  E_tracked  E_fixed  E_model  E_trk/E_model   d33_qs   d33_cr1_model")
for _, r in tab.iterrows():
    print(f"  {r.x_lever_um:6.1f}  {r.E:8.1f} {r.E_fixed:8.1f} {r.E_model:8.1f}   {r.E/r.E_model:6.3f}     {r.d33_qs:6.2f}   {r.d33_cr1_model:6.2f}")
tag = a.tag or (a.run + ("_f0free" if a.free_f0 else "") + ("_semgeom" if a.fix_geom else "") + ("_nostatic" if a.sig_static > 1 else ""))
tab.to_csv(OUT / f"adama_{tag}_table.csv", index=False)
json.dump(dict(run=a.run, L_um=a.L, x0_stage_um=x0, p=p.tolist(), params=dict(zip(eb_fit.PARAMS, p.tolist())), summary=sm, Q1=Q1, cr_Hz=cr.tolist(),
               k_lever=k_lever, f0_free_Hz=f0, x_lever_um=xl_s.tolist(), E_model=E_model.tolist(), Q_model=Qm, M_model=(E_model / Qm).tolist(),
               x_grid=xg.tolist(), E_grid=E_g.tolist(), scan=[dict(x0=r[0], cost=r[1]) for r in res]),
          open(OUT / f"adama_{tag}_blind_fit.json", "w"), indent=1, default=float)
print("wrote", OUT / f"adama_{tag}_blind_fit.json")

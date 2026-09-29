"""One-time lever calibration: one geometry for every mode, from a dense map (phase 2g method).

Fits the shared lever/contact geometry (clamp offset c, effective length L, tip setback, tip
height, k*/k, kcone/k, contact-dashpot Q) jointly to all bands of a dense map with the joint
EB model (fmmpaper.jointeb), per band only zeta, eps, a complex gain and a small frequency
correction dlf. Writes results/lever_calibration_<probe>.json.

    python tools/calibrate_lever.py stiff          # SCM-PIT B, R2 dense map
    python tools/calibrate_lever.py stiff --key scmpitB_r1_dense --tag r1   # same lever, run R1
    python tools/calibrate_lever.py soft           # PPP-CONTAu (reproduces phase 2g M1f)
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fmmpaper as F  # noqa: E402
from fmmpaper import jointeb as J, physrec, recon, spectra  # noqa: E402

PROBES = {
    "stiff": dict(key="scmpitB_r2_dense", f0_hz=63.801e3, bands=["CR1", "CR2", "CR3"], mu=0.004,
                  start=dict(c_um=28.1, L_um=226.5, setback_um=5.0, tip_h_um=6.5, log_alpha=3.0, log_kcone=1.9),
                  bounds=dict(c_um=(10.0, 45.0), L_um=(200.0, 260.0), setback_um=(1.0, 20.0), tip_h_um=(3.0, 15.0)),
                  seeds=dict(setback_um=(3.0, 6.0, 10.0), log_alpha=(2.6, 3.0, 3.4))),
    "soft": dict(key="ppp_dense_1um", f0_hz=13.649e3, bands=["CR1", "CR2", "CR3", "CR4", "CR5"], mu=0.0018,
                 start=dict(c_um=-19.0, L_um=468.0, setback_um=12.5, tip_h_um=14.6, log_alpha=3.4, log_kcone=1.9),
                 bounds={}, seeds=dict(setback_um=(8.0, 12.5, 18.0), log_alpha=(3.0, 3.4, 3.8))),
}
GEO = ("log_alpha", "log_kcone", "log_Qc", "setback_um", "tip_h_um", "c_um", "L_um")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", choices=PROBES); ap.add_argument("--key"); ap.add_argument("--tag", default="")
    ap.add_argument("--N", type=int, default=30); ap.add_argument("--max_nfev", type=int, default=300)
    ap.add_argument("--dlf_max", type=float, default=0.04, help="bound on the per-band frequency correction")
    ap.add_argument("--no_dlf", action="store_true", help="no per-band frequency correction at all")
    ap.add_argument("--free_mass", action="store_true", help="fit tip mass ratio mu and rotary inertia J too")
    a = ap.parse_args()
    cfg = PROBES[a.probe]; key = a.key or cfg["key"]
    J.PER_BAND["dlf"] = (0.0, -a.dlf_max, a.dlf_max)
    s = F.load(key); x = s.x_um
    bands = []
    for b in cfg["bands"]:
        z, f = physrec.band_slice(s.freq_Hz, s.Z[0], s.probe.bands_Hz[b], n_max=250)
        bands.append(J.Band(b, f, z))
    je = J.JointEB(x, bands, cfg["f0_hz"], n_modes=14)
    sel = recon.select_equispaced(x, a.N); ho = recon.held_out(len(x), sel)
    # measured resonance and Q per band (starts for zeta)
    zeta0 = []
    for b in bands:
        pk = spectra.peaks_along_x(b.f_Hz, b.Z[sel], (b.f_Hz[0], b.f_Hz[-1]))
        zeta0.append(float(np.clip(0.5 / np.nanmedian(pk["Q"]), 1e-4, 0.05)))
    free = GEO + (("mu", "J") if a.free_mass else ())
    bounds = dict(cfg["bounds"]); bounds.update(mu=(0.0, 0.1), J=(0.0, 0.02))
    spec = J.Spec(free=free, fixed={"mu": cfg["mu"], "J": 0.0}, band_freq=not a.no_dlf, bounds=bounds)
    starts = []
    for sb in cfg["seeds"]["setback_um"]:
        for la in cfg["seeds"]["log_alpha"]:
            st = dict(cfg["start"]); st.update(setback_um=sb, log_alpha=la, log_Qc=2.5, mu=cfg["mu"], J=1e-4,
                                              log_zeta=list(np.log10(zeta0)), eps=[0.0] * len(bands))
            starts.append(st)
    t0 = time.time()
    fit = je.fit(spec, sel, starts=starts, max_nfev=a.max_nfev)
    print(f"joint fit: {time.time()-t0:.0f} s, cost {fit['cost']:.1f}, nfev {fit['nfev']}")
    maps, gains = je.predict(fit["P"], fit["zeta"], fit["eps"], sel, x=x, dlf=fit["dlf"])
    rows = []
    for b, M in zip(bands, maps):
        D = b.Z
        e = float(np.sqrt(np.mean(np.abs(M[ho] - D[ho]) ** 2) / np.mean(np.abs(D[ho]) ** 2)))
        pm = spectra.peaks_along_x(b.f_Hz, M, (b.f_Hz[0], b.f_Hz[-1])); pd_ = spectra.peaks_along_x(b.f_Hz, D, (b.f_Hz[0], b.f_Hz[-1]))
        nm = spectra.nodes_from_profile(x, pm["amp"])
        nd = [n for n in spectra.nodes_from_profile(x, pd_["amp"]) if x.min() + 5 <= n <= x.max() - 5]
        ne = [min(abs(t - q) for q in nm) for t in nd] if nm else [np.inf] * len(nd)
        rows.append(dict(band=b.name, nrmse=e, nodes_dense=[round(v, 1) for v in nd], node_err_um=[round(v, 2) for v in ne],
                         f_res_kHz=float(np.median(pd_["f_Hz"]) / 1e3)))
        print(f"  {b.name}: held-out NRMSE {e:.3f}, nodes {np.round(nd,1).tolist()} err {np.round(ne,1).tolist()} um")
    P = {k: float(v) for k, v in fit["P"].items()}
    print("geometry:", {k: round(P[k], 4) for k in free})
    print("zeta:", np.round(fit["zeta"], 5).tolist(), "eps:", np.round(fit["eps"], 3).tolist(), "dlf %:", np.round(100 * fit["dlf"], 2).tolist())
    out = dict(probe=a.probe, dataset=key, f0_hz=cfg["f0_hz"], n_modes=14, N_fit=a.N, sel=list(map(int, sel)), free=list(free),
               P=P, zeta=[float(v) for v in fit["zeta"]], eps=[float(v) for v in fit["eps"]], dlf=[float(v) for v in fit["dlf"]],
               bands=cfg["bands"], score=rows, cost=fit["cost"])
    tag = f"_{a.tag}" if a.tag else ""
    p = F.config.RESULTS_DIR / f"lever_calibration_{a.probe}{tag}.json"
    json.dump(out, open(p, "w"), indent=1); print("wrote", p)


if __name__ == "__main__":
    main()

"""Validate the calibrated-lever EB fitter (ebfit.rec_eb2) against the old one (physrec.rec_eb).

Per band and N: held-out NRMSE of EB and EB+GP, node error, fitted k*/k and zeta, fit time.
Old-fitter numbers come from results/_cache_02s.json and results/_cache_explorer/ (no refits).

    python tools/validate_ebfit.py --probe stiff --cal mass
"""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fmmpaper as F  # noqa: E402
from fmmpaper import ebfit, physrec, recon, spectra  # noqa: E402

KEYS = {"stiff": ("scmpitB_r2_dense", 28.1), "soft": ("ppp_dense_1um", -19.0)}
NS = [3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 19, 20, 30]


def node_err(x, f, Z, M, edge=5.0):
    f0, _ = spectra.on_resonance_profile(f, Z, (f.min(), f.max())); j = int(np.argmin(np.abs(f - f0)))
    a0, a = np.abs(Z[:, j]), np.abs(M[:, j])
    inb = lambda n: x.min() + edge <= n <= x.max() - edge
    nd0 = [n for n in spectra.nodes_from_profile(x, a0 / a0.max()) if inb(n)]
    nd = [n for n in spectra.nodes_from_profile(x, a / a0.max()) if inb(n)]
    if not nd0:
        return float("nan")
    return float(max(min(abs(t - q) for q in nd) for t in nd0)) if nd else float("inf")


def old_numbers(probe, band, N, x, sel):
    """(EB, EB+GP) held-out NRMSE of the old fitter, if cached."""
    if probe == "stiff":
        c = json.load(open(F.config.RESULTS_DIR / "_cache_02s.json")).get(f"{band}|{N}")
        if c:
            return 100 * c["EB"]["nrmse"], 100 * c["EB+GP"]["nrmse"], c["theta"]["tip_setback_um"]
    key = hashlib.md5(f"{probe}|{band}|{list(map(int, sel))}".encode()).hexdigest()[:12]
    p = F.config.RESULTS_DIR / "_cache_explorer" / f"{probe}_{band}_{key}.npz"
    if p.exists():
        d = np.load(p, allow_pickle=True); th = json.loads(str(d["theta"]))
        return d["Zeb"], d["Zrec"], th["tip_setback_um"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="stiff"); ap.add_argument("--cal", default="mass")
    ap.add_argument("--N", nargs="+", type=int, default=NS); ap.add_argument("--bands", nargs="+")
    a = ap.parse_args()
    key, x0 = KEYS[a.probe]; s = F.load(key); x, f, Z = s.x_um, s.freq_Hz, s.Z[0]
    cal = ebfit.load_calibration(a.probe, a.cal)
    bands = a.bands or list(s.probe.bands_Hz)
    rows = []
    for band in bands:
        zb, fb = physrec.band_slice(f, Z, dict(s.probe.bands_Hz)[band], n_max=250)
        print(f"\n{band}   N   EB v2   EB+GP v2 | EB v1   EB+GP v1 | node v2  k*/k   zeta    eps   chi2   s   (v1 setback)")
        for N in a.N:
            sel = recon.select_equispaced(x, N); h = recon.held_out(len(x), sel)
            t0 = time.time(); o = ebfit.rec_eb2(x, sel, zb[sel], fb, cal, band); dt = time.time() - t0
            th = o["theta"]
            r = dict(band=band, N=N, eb=recon.nrmse(o["Zeb"][h], zb[h]), ebgp=recon.nrmse(o["Zrec"][h], zb[h]),
                     node_um=node_err(x - x0, fb, zb, o["Zrec"]), k_ratio=th["k_ratio"], zeta=th["zeta"], eps=th["eps"],
                     chi2=th["red_chi2"], sec=dt, eb_v1=np.nan, ebgp_v1=np.nan, setback_v1=np.nan)
            old = old_numbers(a.probe, band, N, x, sel)
            if old is not None:
                if isinstance(old[0], np.ndarray):
                    r.update(eb_v1=recon.nrmse(old[0][h], zb[h]), ebgp_v1=recon.nrmse(old[1][h], zb[h]), setback_v1=old[2])
                else:
                    r.update(eb_v1=old[0], ebgp_v1=old[1], setback_v1=old[2])
            rows.append(r)
            print(f"      {N:3d}  {r['eb']:5.1f}   {r['ebgp']:5.1f}    | {r['eb_v1']:5.1f}   {r['ebgp_v1']:5.1f}    | {r['node_um']:5.1f}  {r['k_ratio']:6.0f}  {r['zeta']:.4f} {r['eps']:6.3f} {r['chi2']:5.1f} {dt:4.1f}   ({r['setback_v1']:.1f})")
    out = F.config.RESULTS_DIR / f"validate_ebfit_{a.probe}_{a.cal}.json"
    json.dump(dict(probe=a.probe, cal=a.cal, calibration=cal["P"], rows=rows), open(out, "w"), indent=1, default=float)
    print("wrote", out)


if __name__ == "__main__":
    main()

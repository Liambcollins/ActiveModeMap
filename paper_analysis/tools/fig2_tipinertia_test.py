"""Cheap test: SEM geometry + free tip mass (mu) and rotary inertia (J) per band in the Fig. 2 beam-model fit (no dense map).
Does freeing tip inertia bring CR3-CR5 of the soft lever back? Reports per-band EB and hybrid error vs N and the fitted mu, J.
    python tools/fig2_tipinertia_test.py soft [N ...]
"""
import sys, json, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
import fmmpaper as F
from fmmpaper import ebfit, physrec, recon, semprior
PROBE = sys.argv[1] if len(sys.argv) > 1 else "soft"; NS = [int(a) for a in sys.argv[2:]] or [6, 12]
P = {"soft": dict(key="ppp_dense_1um", x0=-19.0), "stiff": dict(key="scmpitB_r2_dense", x0=28.1)}[PROBE]
s = F.load(P["key"]); xs, f, Z = s.x_um, s.freq_Hz, s.Z[0]; x = xs - P["x0"]
bands = dict(s.probe.bands_Hz); names = list(bands); masks = {b: (f >= bands[b][0]) & (f <= bands[b][1]) for b in names}
cal, free0 = semprior.calibration(PROBE, P["x0"])
VAR = {"SEM": free0, "SEM + mu": free0 + ("mu",), "SEM + mu + J": free0 + ("mu", "J")}
interp = lambda ft, ff, M: (np.apply_along_axis(lambda r: np.interp(ft, ff, r), 1, M.real) + 1j * np.apply_along_axis(lambda r: np.interp(ft, ff, r), 1, M.imag))
res = {}
for N in NS:
    sel = recon.select_equispaced(x, N); h = recon.held_out(len(x), sel)
    for vn, free in VAR.items():
        t0 = time.time(); row = {}
        for b in names:
            zb, fb = physrec.band_slice(f, Z, bands[b], n_max=250)
            o = ebfit.rec_eb2(xs, sel, zb[sel], fb, cal, b, free=free); m = masks[b]
            eb = interp(f[m], fb, o["Zeb"]); gp = interp(f[m], fb, o["Zrec"])
            row[b] = dict(EB=recon.nrmse(eb[h], Z[h][:, m]), GP=recon.nrmse(gp[h], Z[h][:, m]), mu=o["theta"].get("mu"), J=o["theta"].get("J"), k=o["theta"]["k_ratio"], kc=o["theta"]["kcone_ratio"], c=o["theta"]["c_um"])
        res[f"N={N} | {vn}"] = row
        print(f"N={N:2d} {vn:14s} EB " + " ".join(f"{b} {row[b]['EB']:5.1f}" for b in names) + " | GP " + " ".join(f"{row[b]['GP']:4.1f}" for b in names)
              + " | mu " + " ".join(f"{(row[b]['mu'] if row[b]['mu'] is not None else 0):.4f}" for b in names) + " | J " + " ".join(f"{(row[b]['J'] if row[b]['J'] is not None else 0):.1e}" for b in names) + f" ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(HERE.parent / f"results/fig2_tipinertia_test_{PROBE}.json", "w"), indent=1, default=float)

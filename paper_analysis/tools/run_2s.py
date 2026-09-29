"""Stiff probe (SCM-PIT-B, R2 dense map, 161 pos at 1 um): per-band learning curves for Fig. 2.
Same protocol as notebook 02a (soft probe). Cached in results/_cache_02s.json (resumable).
    python tools/run_2s.py
"""
import json, sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
import fmmpaper as F
from fmmpaper import physrec, recon, spectra

X0 = 28.1                     # clamp position during the R2 dense map (mean over node times, notebook 05a)
BANDS = ["CR1", "CR2", "CR3"]
NS = [3, 4, 5, 6, 8, 10, 14, 20, 30]
GEOM = physrec.GEOM["scmpitB"]
s = F.load("scmpitB_r2_dense")
x = s.x_um - X0
band_data = {b: physrec.band_slice(s.freq_Hz, s.Z[0], s.probe.bands_Hz[b], n_max=250) for b in BANDS}

def nodes_of(Zb, fb):
    pk = spectra.peaks_along_x(fb, Zb, (fb[0], fb[-1]))
    return spectra.nodes_from_profile(x, pk["amp"])

truth = {b: [n for n in nodes_of(zb, fb) if x.min() + 5 <= n <= x.max() - 5] for b, (zb, fb) in band_data.items()}
CACHE = F.config.RESULTS_DIR / "_cache_02s.json"
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
cache["_truth"] = truth; cache["_x0"] = X0

def err(M, zb, h):
    return float(np.sqrt(np.mean(np.abs(M[h] - zb[h]) ** 2)) / np.sqrt(np.mean(np.abs(zb[h]) ** 2)))

def node_err(M, fb, t):
    nd = nodes_of(M, fb)
    if not t:
        return None
    if not nd:
        return float("inf")
    return float(np.mean([min(abs(a - m) for m in nd) for a in t]))

for n in NS:
    for b, (zb, fb) in band_data.items():
        key = f"{b}|{n}"
        if key in cache:
            continue
        sel = recon.select_equispaced(x, n); h = recon.held_out(len(x), sel)
        t0 = time.time()
        rec = {"sel": [int(i) for i in sel]}
        try:
            o = physrec.rec_eb(x, sel, zb[sel], fb, GEOM)
            maps = {"EB": o["Zeb"], "EB+GP": o["Zrec"]}
            rec["theta"] = o["theta"]
        except Exception as e:
            maps = {}; rec["error"] = repr(e)
        maps["GP"] = recon.rec_gp(x, sel, zb[sel])["Zrec"]
        maps["low-rank"] = recon.rec_lowrank(x, sel, zb[sel], rank=min(n, 6))["Zrec"]
        for arm, M in maps.items():
            rec[arm] = dict(nrmse=err(M, zb, h), node_err_um=node_err(M, fb, truth[b]))
        rec["sec"] = time.time() - t0
        cache[key] = rec
        CACHE.write_text(json.dumps(cache))
        print(f"{key:7s} {rec['sec']:5.0f}s " + " ".join(f"{a} {rec[a]['nrmse']:.3f}" for a in maps if a in rec), flush=True)
print("done", truth)

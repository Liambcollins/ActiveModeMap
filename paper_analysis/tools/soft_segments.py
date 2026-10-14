"""PPP-CONTAu dense map, geometry fixed: (a) per-position complex gain, (b) contact state (k*, zeta, eps) refit in x-segments.
Tests whether the misfit is a position-dependent contact state / gain rather than a global per-band one."""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
import fmmpaper as F
from fmmpaper import ebfit, physrec
probe = sys.argv[1] if len(sys.argv) > 1 else "soft"; nseg = int(sys.argv[2]) if len(sys.argv) > 2 else 3
key = {"soft": "ppp_dense_1um", "stiff": "scmpitB_r2_dense"}[probe]
s = F.load(key); xs, f, Z = s.x_um, s.freq_Hz, s.Z[0]
cal = ebfit.load_calibration(probe, "mass"); out = {}
nr = lambda m, z: float(np.linalg.norm(m - z) / np.linalg.norm(z))
for b, rng in dict(s.probe.bands_Hz).items():
    zb, fb = physrec.band_slice(f, Z, rng, n_max=250)
    sel = np.arange(len(xs))
    o = ebfit.rec_eb2(xs, sel, zb, fb, cal, b, gp=False, max_nfev=150); m = o["Zeb"]
    g = (np.conj(m) * zb).sum(1) / ((np.abs(m) ** 2).sum(1) + 1e-30)
    r = dict(global_=nr(m, zb), rowgain=nr(g[:, None] * m, zb), rowgain_amp_range=[float(np.percentile(np.abs(g), 5)), float(np.percentile(np.abs(g), 95))])
    edges = np.linspace(0, len(xs), nseg + 1).astype(int); mm = np.zeros_like(zb); ks = []
    for i in range(nseg):
        ii = np.arange(edges[i], edges[i + 1])
        oi = ebfit.rec_eb2(xs, ii, zb[ii], fb, cal, b, gp=False, max_nfev=150)
        mm[ii] = oi["Zeb"][ii]; ks.append((float(oi["theta"]["k_ratio"]), float(oi["theta"]["zeta"]), float(oi["theta"]["eps"]), float(xs[ii].min()), float(xs[ii].max())))
    r.update(segments=nr(mm, zb), seg_theta=ks)
    out[b] = r
    print(b, f"global {r['global_']:.3f}  rowgain {r['rowgain']:.3f}  {nseg}-seg {r['segments']:.3f}  k*/k per seg", [round(k[0]) for k in ks], "zeta", [round(k[1],4) for k in ks], flush=True)
json.dump(out, open(HERE.parent / f"results/soft_segments_{probe}_{nseg}.json", "w"), indent=1)

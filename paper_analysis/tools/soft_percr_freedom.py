"""PPP-CONTAu dense map, geometry fixed: which extra per-band contact freedoms improve the beam-model match?
Variants (free per band, all dense positions): base=(alpha,zeta,eps); +kcone; +Qc; +kcone+Qc.
Metric: complex nrmse of best-gain model vs dense map over band; plus resonance-frequency track error (lw)."""
import sys, json, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
import fmmpaper as F
from fmmpaper import ebfit, physrec
_rec = ebfit.rec_eb2
probe = sys.argv[1] if len(sys.argv) > 1 else "soft"
key = {"soft": "ppp_dense_1um", "stiff": "scmpitB_r2_dense"}[probe]
s = F.load(key); xs, f, Z = s.x_um, s.freq_Hz, s.Z[0]
cal = ebfit.load_calibration(probe, "mass")
V = {"base": ("log_alpha", "log_zeta", "eps"), "kcone": ("log_alpha", "log_zeta", "eps", "log_kcone"),
     "Qc": ("log_alpha", "log_zeta", "eps", "log_Qc"), "kcone+Qc": ("log_alpha", "log_zeta", "eps", "log_kcone", "log_Qc")}
out = {}; sel = np.arange(len(xs))
for b, rng in dict(s.probe.bands_Hz).items():
    zb, fb = physrec.band_slice(f, Z, rng, n_max=250)
    for vn, free in V.items():
        t0 = time.time()
        o = ebfit.rec_eb2(xs, sel, zb, fb, cal, b, free=free, gp=False, max_nfev=150)
        m = o["Zeb"]; nr = float(np.linalg.norm(m - zb) / np.linalg.norm(zb))
        # per-position amplitude ratio at the peak
        ar = np.abs(m).max(1) / np.abs(zb).max(1)
        fp = fb[np.abs(m).argmax(1)]; fd = fb[np.abs(zb).argmax(1)]
        lw = fb[np.argmax(np.abs(zb).mean(0))] / max(o["theta"].get("Q_meas", 100), 1)
        out[f"{b}|{vn}"] = dict(nrmse=nr, amp_med=float(np.median(ar)), amp_p5=float(np.percentile(ar, 5)), amp_p95=float(np.percentile(ar, 95)),
                                 df_lw_rms=float(np.sqrt(np.mean(((fp - fd) / lw) ** 2))), cost=o["cost"], theta={k: float(v) for k, v in o["theta"].items()})
        print(b, vn, f"nrmse {nr:.3f} amp {np.percentile(ar,5):.2f}-{np.percentile(ar,95):.2f} df_lw {out[f'{b}|{vn}']['df_lw_rms']:.2f} k*/k {o['theta']['k_ratio']:.0f} kc {o['theta']['kcone_ratio']:.1f} cost {o['cost']:.0f} {time.time()-t0:.0f}s", flush=True)
json.dump(out, open(HERE.parent / f"results/soft_percr_freedom_{probe}.json", "w"), indent=1)

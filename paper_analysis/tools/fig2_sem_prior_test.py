"""Does Fig. 2 need the dense-map geometry? Rerun the Fig. 2 scoring (EB, EB+GP vs N) with the SEM geometry as the only
lever prior (L, set-back, tip height from SEM; clamp position free per band within +-15 um of the registered value;
per-band contact state k*, zeta, eps, gain from the sparse spectra as always). Compare with the dense-map calibration.
    python tools/fig2_sem_prior_test.py soft|stiff
"""
import sys, json, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
import fmmpaper as F
from fmmpaper import ebfit, physrec, recon
PROBE = sys.argv[1]
P = {"soft": dict(key="ppp_dense_1um", x0=-19.0, sem=dict(L=461.6, sb=23.5, h=14.3)),
     "stiff": dict(key="scmpitB_r2_dense", x0=28.1, sem=dict(L=227.1, sb=10.25, h=11.7))}[PROBE]
s = F.load(P["key"]); xs, f, Z = s.x_um, s.freq_Hz, s.Z[0]; x = xs - P["x0"]
bands = dict(s.probe.bands_Hz); names = list(bands)
masks = {b: (f >= bands[b][0]) & (f <= bands[b][1]) for b in names}; span = np.zeros(f.size, bool)
for m in masks.values(): span |= m
cal0 = ebfit.load_calibration(PROBE, "mass")
cal_sem = json.loads(json.dumps(cal0)); cal_sem["P"].update(L_um=P["sem"]["L"], setback_um=P["sem"]["sb"], tip_h_um=P["sem"]["h"], c_um=P["x0"])
cal_sem_nok = json.loads(json.dumps(cal_sem)); cal_sem_nok["P"]["log_kcone"] = 0.0; cal_sem_nok["P"]["log_Qc"] = 2.0   # no dense-fit contact knowledge either
VAR = {"dense-map geometry": (cal0, ebfit.FREE_DEFAULT), "SEM geometry, c free": (cal_sem, ebfit.FREE_DEFAULT + ("c_um",)),
       "SEM geometry + generic kcone/Qc, c free": (cal_sem_nok, ebfit.FREE_DEFAULT + ("c_um", "log_kcone"))}
interp = lambda ft, ff, M: (np.apply_along_axis(lambda r: np.interp(ft, ff, r), 1, M.real) + 1j * np.apply_along_axis(lambda r: np.interp(ft, ff, r), 1, M.imag))
res = {}
for N in (4, 6, 9, 12):
    sel = recon.select_equispaced(x, N); h = recon.held_out(len(x), sel)
    for vn, (cal, free) in VAR.items():
        t0 = time.time(); eb = np.full(Z.shape, np.nan + 0j); ebgp = eb.copy(); per = {}
        for b in names:
            zb, fb = physrec.band_slice(f, Z, bands[b], n_max=250)
            o = ebfit.rec_eb2(xs, sel, zb[sel], fb, cal, b, free=free)
            m = masks[b]; eb[:, m] = interp(f[m], fb, o["Zeb"]); ebgp[:, m] = interp(f[m], fb, o["Zrec"])
            per[b] = dict(EB=recon.nrmse(eb[h][:, m], Z[h][:, m]), EBGP=recon.nrmse(ebgp[h][:, m], Z[h][:, m]), c_um=o["theta"].get("c_um"), kcone=o["theta"].get("kcone_ratio"))
        r = dict(EB=recon.nrmse(eb[h][:, span], Z[h][:, span]), EBGP=recon.nrmse(ebgp[h][:, span], Z[h][:, span]), per_band=per)
        res[f"N={N} | {vn}"] = r
        print(f"N={N:2d} {vn:42s} EB {r['EB']:5.1f}%  EB+GP {r['EBGP']:5.1f}%  | per band EB+GP " + " ".join(f"{b} {per[b]['EBGP']:.0f}" for b in names) + f"  c {per['CR1']['c_um']}  ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(HERE.parent / f"results/fig2_sem_prior_test_{PROBE}.json", "w"), indent=1, default=float)

"""CR1 cross-check model for one survey: calibrated lever geometry (frozen), clamp position c and contact state
(k*, Q_c, zeta, eps per band) refitted to the survey's own two-domain spectra. Returns model M(x) = E/Q at the survey stops.
    python tools/xprobe_cr1model.py <probe stiff|soft> <cal tag> <dataset key> <out tag>
"""
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import calibrate_freeair as CF, sim_calib_freeair as S
from fmmpaper import jointeb as J, ebfit, registry, io as fio
from fmmpaper.spectra import q_factor
probe_key, cal_tag, ds, tag = sys.argv[1:5]; t0 = time.time()
ref = ebfit.load_calibration(probe_key, cal_tag); P_ref = dict(ref["P"]); S.F0 = ref["f0_hz"]
probe = registry.PROBES[CF.PROBE_KEY[probe_key]]
path = fio.path_of(ds)
xs, fs, Zs, src = CF.survey_P(path, probe)
bands = {k: v for k, v in probe.bands_Hz.items() if k in ("CR1", "CR2", "CR3")}
cap = CF.capture_from(xs, fs, Zs, bands); o = np.argsort(xs); xs = xs[o]; cap["Z"] = [z[o] for z in cap["Z"]]
J.SHARED["c_um"] = (P_ref["c_um"], P_ref["c_um"] - 30, P_ref["c_um"] + 30)
J.SHARED["L_um"] = (P_ref["L_um"], 0.6 * P_ref["L_um"], 1.4 * P_ref["L_um"])
best = None
for c0 in (P_ref["c_um"] - 8, P_ref["c_um"], P_ref["c_um"] + 8):
    Pf, per, chi2 = CF.fit_contact(xs, [cap], dict(P_ref, c_um=c0), ["c_um"], starts=1)
    if best is None or chi2 < best[2]: best = (Pf, per, chi2)
Pf, per, chi2 = best; la, lq, z = per[0][0], per[0][1], per[0][2]
Pl = dict(Pf, log_alpha=la, log_Qc=lq); f1 = S.eig(Pl, 1)[0]; fb = np.linspace(0.96 * f1, 1.04 * f1, 1600)
je = J.JointEB(xs, [J.Band("CR1", fb, None), J.Band("QS", np.array([np.mean(probe.qs_band_Hz)]), None)], S.F0, n_modes=S.NM); S._scale(je, Pl)
(zp, _), (zq, _) = je.band_maps(Pl, [z[0], z[0]], x=xs)
a = np.abs(zp); k = a.argmax(1); Q = np.array([q_factor(fb, a[i], k[i]) for i in range(len(xs))]); E = a.max(1) / np.abs(zq)[:, 0]
out = dict(dataset=ds, probe=probe_key, cal=cal_tag, c_fit=Pf["c_um"], c_cal=P_ref["c_um"], chi2=chi2, log_alpha=la, f_CR1_model_kHz=f1 / 1e3,
           x=xs.tolist(), E_model=E.tolist(), Q_model=Q.tolist(), M_model=(E / Q).tolist(), minutes=(time.time() - t0) / 60)
json.dump(out, open(HERE.parent / f"results/r3r4/xprobe_cr1model_{tag}.json", "w"), indent=1, default=float)
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items() if k not in ("x", "E_model", "Q_model", "M_model")}))
print(np.round(E / Q, 3))

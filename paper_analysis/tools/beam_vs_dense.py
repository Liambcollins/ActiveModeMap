"""Where does the calibrated beam model fail? Compare dense maps with the Euler-Bernoulli model, band by band.
Geometry fixed from the dense-map calibration (lever_calibration_{soft,stiff}_mass.json). Per band the contact state
(k*, zeta, electrostatic weight eps, complex gain) is refit using ALL dense positions ('best case') and using the 30
calibration positions ('cal'). Saves results/beam_vs_dense_{probe}.npz.
    python tools/beam_vs_dense.py soft|stiff
"""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
import fmmpaper as F
from fmmpaper import ebfit, physrec
probe = sys.argv[1]
key = {"soft": "ppp_dense_1um", "stiff": "scmpitB_r2_dense"}[probe]
s = F.load(key); xs, f, Z = s.x_um, s.freq_Hz, s.Z[0]
cal = ebfit.load_calibration(probe, "mass")
bands = dict(s.probe.bands_Hz)
out = dict(x=xs, c_um=cal["P"]["c_um"], L_um=cal["P"]["L_um"], setback_um=cal["P"]["setback_um"], bands=np.array(list(bands)))
allsel = np.arange(len(xs)); calsel = np.array(cal["sel"])
for b, rng in bands.items():
    zb, fb = physrec.band_slice(f, Z, rng, n_max=250)
    for tag, sel in (("all", allsel), ("cal", calsel)):
        t0 = time.time()
        o = ebfit.rec_eb2(xs, sel, zb[sel], fb, cal, b, gp=False)
        out[f"{b}_f"] = fb; out[f"{b}_Z"] = zb; out[f"{b}_{tag}"] = o["Zeb"]
        out[f"{b}_{tag}_theta"] = np.array([float(v) for v in np.atleast_1d(o["theta"]).ravel()]) if not isinstance(o["theta"], dict) else np.array(list(o["theta"].values()), float)
        print(probe, b, tag, f"{time.time()-t0:.0f}s", "theta", o["theta"], flush=True)
np.savez(HERE.parent / f"results/beam_vs_dense_{probe}.npz", **out)
print("saved")

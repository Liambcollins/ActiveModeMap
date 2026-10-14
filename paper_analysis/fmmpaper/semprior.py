"""SEM-geometry lever prior for the sparse reconstructions (no dense-map calibration).

    cal, free = semprior.calibration("soft", x0_stage=-19.0)
    ebfit.rec_eb2(xs, sel, Z_sel, f, cal, band, free=free)

Geometry (L, set-back, tip height) from the SEM images (Supplementary Note 1); tip mass and rotary inertia at the package
defaults; cone stiffness and contact dashpot at generic values and k_cone refitted per band; clamp position c free within
+-15 um of the registered value; per-band contact state (k*, zeta, eps, complex gain) from the sparse spectra, as always.
The only dense-map quantity retained is the free resonance f0 (a measured tune).
"""
from __future__ import annotations
import json
from . import config, ebfit

SEM = {"soft": dict(L_um=461.6, setback_um=23.5, tip_h_um=14.3), "stiff": dict(L_um=227.1, setback_um=10.25, tip_h_um=11.7),
       "multi75g": dict(L_um=234.7, setback_um=16.3, tip_h_um=12.8)}
FREE = ebfit.FREE_DEFAULT + ("c_um", "log_kcone")


def calibration(probe: str, x0_stage: float, tag: str = "mass"):
    cal0 = ebfit.load_calibration(probe, tag)
    cal = json.loads(json.dumps(cal0)); P = cal["P"]
    P.update(SEM[probe]); P.update(c_um=float(x0_stage), log_kcone=0.0, log_Qc=2.0, mu=0.0018, J=0.0)
    cal["zeta"] = [0.002] * len(cal["bands"]); cal["eps"] = [0.0] * len(cal["bands"]); cal["dlf"] = [0.0] * len(cal["bands"])
    cal["prior"] = "SEM"; cal.pop("score", None); cal.pop("sel", None); cal.pop("free", None)
    return cal, FREE

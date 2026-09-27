#!/usr/bin/env python3
"""Mini-ladder: does tip-column compliance cap the FEM mode ratio f2/f1?

Both FEM lateral conditions saturate below the measured f2/f1 = 3.080
(frictionless at ~2.66, isotropic at ~2.94 by k1 = 5000 N/m), while EB with a
rigid tip reaches it. The prime suspect is the elastic tip column: raising
e_tip stiffens it toward the rigid-tip limit. This runs a 2 x 2 x 3 sweep
(lateral x k1 x e_tip) = 12 exports, ~3-25 min depending on tab throttling.

Run from fem_eb_comparison\\py\\ (next to run_fem_ladder.py):

    python run_fem_etip.py --stl <ABSOLUTE STL PATH> --out ..\\data_etip

Interpretation: if f2/f1 climbs toward ~3.1-3.2 as e_tip rises, the tip column
explains the saturation and the FEM geometry is vindicated; if it barely moves,
the cap lives elsewhere (taper / overhang / attach node).
"""
import argparse, json, os, sys

from afemulator_client import AFeMulator, AFeMulatorError
from run_fem_sweep import LATERAL, Setter, TILT_DEG, Z_DRIVE_PM
import api_map
from run_fem_ladder import QUIET, set_extra, flexural_khz, retry

K1S = [1247.7, 5000.0]
ETIPS = [130.0, 400.0, 1300.0]          # GPa; 130 is the shipped default
LATS = ["frictionless", "isotropic"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=True)
    ap.add_argument("--out", default="../data_etip")
    ap.add_argument("--log-damping", type=float, default=4.775)
    a = ap.parse_args()

    app = AFeMulator()
    print("ping:", app.ping())
    mapping = api_map.resolve(app.get_params(), None)
    setp = Setter(app, mapping, verify=True)
    os.makedirs(a.out, exist_ok=True)
    app.wait_ready(); app.set_autorun(False)
    app.load_stl(a.stl)
    setp(tilt=TILT_DEG, z_disp=Z_DRIVE_PM, **QUIET)
    setp(freq_low=0.0, freq_high=2000.0)
    set_extra(app, log_damping=a.log_damping)

    rows = []
    for lat in LATS:
        k2_of = LATERAL[lat]
        for k1 in K1S:
            setp(k_z=k1, k_x=float(k2_of(k1)), k_y=float(k2_of(k1)))
            for et in ETIPS:
                set_extra(app, e_tip_gpa=et)
                retry(lambda: app.compute("fea"), f"compute {lat} k1={k1:g} et={et:g}")
                fx = flexural_khz(app)
                name = f"fem_etip_{lat}_k1_{k1:g}_et{et:g}.h5"
                retry(lambda: app.export(os.path.join(a.out, name)),
                      f"export {name}")
                r = dict(lateral=lat, k1=k1, e_tip_gpa=et, file=name,
                         f1_khz=fx[0], f2_khz=fx[1], ratio=fx[1] / fx[0])
                rows.append(r)
                print(f"  {lat:12s} k1={k1:6.0f} e_tip={et:5.0f}  "
                      f"f1={fx[0]:7.2f}  f2={fx[1]:8.2f}  f2/f1={r['ratio']:.3f}",
                      flush=True)
    with open(os.path.join(a.out, "etip_sweep.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print("\nmeasured f2/f1 = 3.080  |  wrote", os.path.join(a.out, "etip_sweep.json"))


if __name__ == "__main__":
    main()

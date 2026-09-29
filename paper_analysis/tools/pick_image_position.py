"""Where to put the laser for a quantitative resonance image: E(x) from a sparse capture.

Run on the instrument PC right after a sparse capture (two-domain bias survey preferred, or a
single-domain 0 V capture) and before imaging. Fits the calibrated-lever model (fmmpaper.ebfit)
to the capture, reconstructs the CR1 and quasi-static bands on a 1 um grid, and prints E(x) =
|Z_CR1| / |Z_QS| along the lever with recommended imaging positions: large E, and flat enough
in x that a micrometre of laser drift changes E by less than a few per cent.

    python tools/pick_image_position.py <capture.npz> --probe stiff [--tol_pct 3] [--drift_um 2]
    python tools/pick_image_position.py .../domains_bias_checkpoint_500nN_scmpit.npz --probe stiff

Input formats (as written by the campaign scripts): a two-domain bias-survey checkpoint with
'conditions' (bias_V, spot) -> uses P = (Z_spot1 - Z_spot2)/2 at each bias and fits the
bias-independent part; or a single-condition wideband checkpoint -> uses Z at 0 V of one domain
(E then carries the electrostatic channel: reported with a warning).

Outputs a table and results/pick_image_position_<stem>.json; E at any chosen position can be read
from the table. Positions are STAGE coordinates (what you set on the laser stage) and, for
reference, distance from the clamp.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fmmpaper as F  # noqa: E402
from fmmpaper import domains, ebfit, physrec, registry  # noqa: E402
from fmmpaper.io import load_series_file  # noqa: E402

PROBE_KEY = {"stiff": "scmpitB", "soft": "pppcontau"}


def two_domain_P(s):
    """P(x, f) = bias-independent piezoresponse from a two-domain survey (decompose per frequency)."""
    c = s.conditions
    if "spot" not in c or c.spot.nunique() < 2:
        return None
    Pc = domains.decompose_series(s, band=None)
    return Pc["P"], Pc["freq_Hz"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture"); ap.add_argument("--probe", default="stiff", choices=PROBE_KEY)
    ap.add_argument("--cal", default="mass", help="lever calibration tag (results/lever_calibration_<probe>_<tag>.json)")
    ap.add_argument("--tol_pct", type=float, default=3.0, help="max change of E over +-drift_um to call a position stable")
    ap.add_argument("--drift_um", type=float, default=2.0)
    ap.add_argument("--min_E", type=float, default=50.0)
    a = ap.parse_args()
    probe = registry.PROBES[PROBE_KEY[a.probe]]
    s = load_series_file(a.capture, probe)
    cal = ebfit.load_calibration(a.probe, a.cal)
    x = s.x_um; f = s.freq_Hz; bands = dict(probe.bands_Hz); qs = tuple(probe.qs_band_Hz)
    PF = two_domain_P(s)
    if PF is not None:
        Z, fZ = PF; src = "two-domain P (electrostatic channel removed)"
    else:
        c = s.conditions
        i0 = 0
        if "bias_V" in c:
            i0 = int(np.argmin(np.abs(c.bias_V.values)))
        Z, fZ = s.Z[i0], f; src = "single domain at %s V: E includes the electrostatic channel (up to +-35 %% at the clamp end)" % (c.bias_V.iloc[i0] if "bias_V" in c else "?")
    print(f"capture: {a.capture}\n  {len(x)} positions, stage {x.min():.1f}-{x.max():.1f} um; source: {src}")
    # frame + contact state from the highest band, then CR1 and QS
    sel = np.arange(len(x)); xg = np.linspace(x.min(), x.max(), int(np.ptp(x)) + 1)
    hi_band = list(bands)[-1]
    zb, fb = physrec.band_slice(fZ, Z, bands[hi_band], n_max=250)
    o = ebfit.rec_eb2(x, sel, zb, fb, cal, hi_band, free=("log_alpha", "log_zeta", "eps", "c_um"), gp=False)
    cal = json.loads(json.dumps(cal)); cal["P"]["c_um"] = o["theta"]["c_um"]
    zc, fc = physrec.band_slice(fZ, Z, bands["CR1"], n_max=250)
    oc = ebfit.rec_eb2(x, sel, zc, fc, cal, "CR1", x_eval=xg)
    cq = json.loads(json.dumps(cal)); cq["P"]["log_alpha"] = oc["theta"]["log_alpha"]
    zq, fq = physrec.band_slice(fZ, Z, qs, n_max=100)
    oq = ebfit.rec_eb2(x, sel, zq, fq, cq, "QS", free=("eps",), x_eval=xg)
    E = np.abs(oc["Zrec"]).max(1) / np.abs(oq["Zrec"]).mean(1)                 # data-corrected value
    E_eb = np.abs(oc["Zeb"]).max(1) / np.abs(oq["Zeb"]).mean(1)                 # smooth model: used for the slope
    E_meas = np.abs(zc).max(1) / np.abs(zq).mean(1)
    fcr = float(oc["theta"]["f_meas_Hz"])
    print(f"  frame: clamp at stage {cal['P']['c_um']:.1f} um; k*/k {oc['theta']['k_ratio']:.0f}, CR1 {fcr/1e3:.1f} kHz, chi2 {oc['theta']['red_chi2']:.1f}")
    print("  measured E at the capture positions: " + "  ".join(f"{xx:.0f}:{e:.0f}" for xx, e in zip(x, E_meas)))
    # stability: max relative change of E within +-drift_um
    d = int(round(a.drift_um)); dE = np.array([100 * (E_eb[max(0, i - d):i + d + 1].max() / E_eb[max(0, i - d):i + d + 1].min() - 1) for i in range(len(xg))])
    edge = (xg > xg.min() + 5) & (xg < xg.max() - 5)                            # the slope is not known at the span edges
    ok = (dE <= a.tol_pct) & (E >= a.min_E) & edge
    tab = pd.DataFrame(dict(stage_um=xg, from_clamp_um=xg - cal["P"]["c_um"], E=E, dE_pct_over_drift=dE, stable=ok))
    # recommended: the stable position with the largest E, and the stable position nearest each antinode
    rec = tab[tab.stable].sort_values("E", ascending=False)
    print(f"\nE(x) along the lever (every 10 um; full table in results/):")
    print(tab.iloc[::10][["stage_um", "from_clamp_um", "E", "dE_pct_over_drift", "stable"]].round(1).to_string(index=False))
    if len(rec):
        best = rec.iloc[0]
        print(f"\nRECOMMENDED: stage {best.stage_um:.0f} um ({best.from_clamp_um:.0f} um from the clamp): E = {best.E:.0f}, changes {best.dE_pct_over_drift:.1f} % over +-{a.drift_um:.0f} um")
        alt = rec[np.abs(rec.stage_um - best.stage_um) > 20].head(2)
        for _, r in alt.iterrows():
            print(f"  alternative: stage {r.stage_um:.0f} um, E = {r.E:.0f}, {r.dE_pct_over_drift:.1f} % over +-{a.drift_um:.0f} um")
    else:
        print("\nno position meets the stability criterion; loosen --tol_pct or --min_E")
    print("\nAvoid: positions within 10 um of a CR1 node or of the tip end, where E is small and changes fastest.")
    out = F.config.RESULTS_DIR / f"pick_image_position_{Path(a.capture).stem}.json"
    json.dump(dict(capture=str(a.capture), source=src, c_um=cal["P"]["c_um"], theta_CR1=oc["theta"], f_CR1_Hz=fcr,
                   table=tab.to_dict("records"), recommended=(rec.iloc[0].to_dict() if len(rec) else None)), open(out, "w"), indent=1, default=float)
    print("wrote", out)


if __name__ == "__main__":
    main()

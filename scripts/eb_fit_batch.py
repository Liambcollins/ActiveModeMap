#!/usr/bin/env python
"""Batch physics-informed EB fits with the SHORT-SETBACK BATCH HYPOTHESIS.

Liam's observation: this batch of probes doesn't always have a null/ESBS in
the accessible span — or if it does, it sits at the very end of the tip. The
first EB fit (Demo4) independently landed on setback ≈ 3.6 µm vs the 11 µm
nominal, which puts the modal null at (or past) the free end. This script
treats that as the working hypothesis and tests it properly:

  1. Fit every dense run with geometry (f0, tip setback) FREE. Pool the fitted
     setbacks -> batch geometry estimate.
  2. Classify the null on each run's raw branch with classify_null: an
     approaching-but-not-crossing branch is reported as "at or beyond free
     end" WITH A BOUND, not as a failure.
  3. Refit each run with the setback PINNED to the pooled batch value; if the
     residual barely moves, one shared geometry explains the whole batch.
  4. Evaluate the fitted EB model on a fine grid to the physical end of the
     beam: where does the MODEL put the null, and does it agree with the
     bound from (2)?

    python scripts/eb_fit_batch.py <data_root> --out <outdir>
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from activemodemap.lowrank import classify_null
from eb_fit_real import EBFitter, load, fit, branch_null

BLUE, GREEN, RED, GOLD, GREY = "#2a78d6", "#2f9e5f", "#e34948", "#e8a13c", "#8a8f98"
BANDS = {"mode A": (330e3, 470e3), "mode B": (1090e3, 1240e3)}


def branch_of(x, Zb, fb):
    """Raw antiresonance branch fa(x) and the on-resonance index/frequency."""
    ires = int(np.argmax(np.abs(Zb[np.argmax(np.abs(Zb).max(1))])))
    fres = float(fb[ires])
    fa = np.full(x.size, np.nan)
    for i in range(x.size):
        j = int(np.argmin(np.abs(Zb[i])))
        if 0 < j < fb.size - 1:
            fa[i] = float(fb[j])
    return fa, fres, ires


def fit_run(folder, band, setback_fixed=None, xdec=4):
    """One EB fit; optionally with the setback pinned (batch geometry)."""
    xs, fb, Zd = load(folder, band=band, xdec=xdec)
    fitter = EBFitter(xs, fb)
    if setback_fixed is None:
        sol = fit(fitter, Zd, verbose=False)
    else:
        # pinch the setback bounds around the batch value
        from scipy.optimize import least_squares
        s = float(setback_fixed)
        best = None
        for f0k in np.arange(45.0, 105.1, 5.0):
            for la in np.arange(2.0, 4.41, 0.3):
                for le in (-2.0, -1.0, -0.3, 0.3):
                    p0 = [la, np.log10(2000/3), 2.35, le, f0k, s]
                    try:
                        Zp, _ = fitter.predict(p0)
                    except Exception:
                        continue
                    jm = int(np.argmax(np.abs(Zp).max(0)))
                    jd = int(np.argmax(np.abs(Zd).max(0)))
                    if abs(fitter.F[jm] - fitter.F[jd]) > 8e3:
                        continue
                    r = fitter.residuals(p0, Zd)
                    c = float(r @ r)
                    if best is None or c < best[0]:
                        best = (c, p0)
        if best is None:
            raise RuntimeError("no geometry aligned the contact resonance")
        lo = [1.0, 1.5, 1.8, -3.0, 40.0, max(s - 0.05, 0.5)]
        hi = [5.0, 3.5, 3.0,  1.0, 110.0, s + 0.05]
        sol = least_squares(fitter.residuals, best[1], args=(Zd, None),
                            bounds=(lo, hi), xtol=1e-12, diff_step=0.02,
                            max_nfev=400)
    # amplitude-scale for reporting
    Zp, _ = fitter.predict(sol.x)
    g = float(np.exp(np.median(np.log(np.abs(Zd) + 1e-12)
                               - np.log(np.abs(Zp) + 1e-12))))
    Zp = Zp * g
    rel = float(np.median(np.abs(np.abs(Zp) - np.abs(Zd)).mean(1)
                          / (np.abs(Zd).mean(1) + 1e-30)))
    return dict(xs=xs, fb=fb, Zd=Zd, fitter=fitter, sol=sol, Zp=Zp,
                gain=g, rel=rel)


def null_report(xs, fb, Zd, fitter=None, sol=None, gain=1.0, L=225.0):
    """classify_null on the raw data branch AND (if given) on the fitted EB
    model evaluated on a fine grid to the physical end of the beam."""
    fa_d, fres_d, _ = branch_of(xs, Zd, fb)
    data_cls = classify_null(xs, fa_d, fres_d)
    model_cls = None
    if fitter is not None and sol is not None:
        xq = np.arange(xs.min(), L + 0.25, 0.5)
        Zq = fitter.predict(sol.x, xq)[0] * gain
        fa_m, fres_m, _ = branch_of(xq, Zq, fb)
        model_cls = classify_null(xq, fa_m, fres_m)
        model_cls["fa_Hz"] = None            # keep JSON light
    for c in (data_cls, model_cls):
        if c is not None:
            c.pop("fa_Hz", None)
    return data_cls, model_cls, fa_d, fres_d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root"); ap.add_argument("--out", default="reanalysis_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out = {}

    # discover dense runs
    runs = []
    import re, contextlib, io
    _X = re.compile(r"_X(\d+)_")
    for folder in sorted(glob.glob(os.path.join(a.root, "*/"))):
        name = os.path.basename(folder.rstrip("/"))
        xs = sorted({int(m.group(1)) / 1000.0
                     for p in glob.glob(os.path.join(folder, "Tune_*.txt"))
                     for m in [_X.search(p)] if m})
        if len(xs) >= 40 and np.median(np.diff(xs)) <= 1.2:
            runs.append((name, folder))
    print("dense runs:", [n for n, _ in runs])

    # ---------- pass 1: geometry free ----------
    fits = {}
    for name, folder in runs:
        for band_name, band in BANDS.items():
            # skip bands the tunes don't reach
            probe = glob.glob(os.path.join(folder, "Tune_*.txt"))[0]
            from activemodemap.asylum import read_tune_txt, tune_to_complex
            with contextlib.redirect_stdout(io.StringIO()):
                fprobe, _ = tune_to_complex(read_tune_txt(probe))
            if fprobe.max() < band[1]:
                continue
            key = f"{name}/{band_name}"
            try:
                r = fit_run(folder, band)
            except Exception as e:
                print(f"{key}: fit failed ({e})")
                out[key] = dict(error=str(e))
                continue
            p = r["sol"].x
            dcls, mcls, fa_d, fres_d = null_report(
                r["xs"], r["fb"], r["Zd"], r["fitter"], r["sol"], r["gain"])
            fits[key] = r
            out[key] = dict(
                n_pos=int(r["xs"].size), rel_resid=r["rel"],
                params=dict(log_alpha=float(p[0]), log_kcone=float(p[1]),
                            log_Q=float(p[2]), log_eps=float(p[3]),
                            f0_kHz=float(p[4]), setback_um=float(p[5])),
                data_null=dcls, model_null=mcls,
                f_res_kHz=fres_d / 1e3,
                span_um=[float(r["xs"].min()), float(r["xs"].max())])
            print(f"{key}: resid {100*r['rel']:.0f}%  setback {p[5]:.1f} µm  "
                  f"f0 {p[4]:.0f} kHz | data null: {dcls['status']} "
                  f"({dcls.get('x_null_um') or dcls.get('x_bound_um'):.1f} µm)"
                  f" | model null: {mcls['status']} "
                  f"({(mcls.get('x_null_um') if mcls['status']=='crossed' else mcls.get('x_bound_um')) or float('nan'):.1f} µm)")

    # ---------- pooled batch setback (mode A fits only, weighted by 1/resid) --
    sbs = [(out[k]["params"]["setback_um"], out[k]["rel_resid"])
           for k in out if "params" in out[k] and k.endswith("mode A")]
    if sbs:
        w = np.array([1.0 / max(r, 1e-3) for _, r in sbs])
        vals = np.array([s for s, _ in sbs])
        batch_sb = float(np.sum(vals * w) / np.sum(w))
        out["batch"] = dict(setback_um=batch_sb,
                            setback_values=[float(v) for v in vals],
                            spread_um=float(vals.max() - vals.min()) if len(vals) > 1 else 0.0,
                            nominal_um=11.0)
        print(f"\nBATCH setback: {batch_sb:.1f} µm "
              f"(values {np.round(vals,1)}, nominal 11.0)")

        # ---------- pass 2: setback pinned to the batch value ----------
        for key in list(fits):
            name, band_name = key.split("/")
            folder = dict(runs)[name]
            try:
                r2 = fit_run(folder, BANDS[band_name], setback_fixed=batch_sb)
            except Exception as e:
                out[key]["pinned"] = dict(error=str(e))
                continue
            p2 = r2["sol"].x
            d2, m2, _, _ = null_report(r2["xs"], r2["fb"], r2["Zd"],
                                       r2["fitter"], r2["sol"], r2["gain"])
            out[key]["pinned"] = dict(
                rel_resid=r2["rel"], f0_kHz=float(p2[4]),
                setback_um=float(p2[5]), model_null=m2,
                resid_penalty=float(r2["rel"] - out[key]["rel_resid"]))
            print(f"{key} pinned@{batch_sb:.1f}: resid {100*r2['rel']:.0f}% "
                  f"(free was {100*out[key]['rel_resid']:.0f}%)")

    # ---------- figure: branches + classification per run ----------
    keys = [k for k in fits]
    if keys:
        fig, axes = plt.subplots(1, len(keys), figsize=(5.0 * len(keys), 3.9),
                                 squeeze=False)
        for ax, key in zip(axes[0], keys):
            r = fits[key]
            fa_d, fres_d, _ = branch_of(r["xs"], r["Zd"], r["fb"])
            dcls = out[key]["data_null"]
            ax.plot(r["xs"], fa_d / 1e3, ".", color=GREEN, ms=3.5,
                    label="raw branch fa(x)")
            xq = np.arange(r["xs"].min(), 225.0 + 0.25, 0.5)
            Zq = r["fitter"].predict(r["sol"].x, xq)[0] * r["gain"]
            fa_m, fres_m, _ = branch_of(xq, Zq, r["fb"])
            ax.plot(xq, fa_m / 1e3, color=BLUE, lw=1.3, label="EB-model branch")
            ax.axhline(fres_d / 1e3, color="k", ls=":", lw=1.1, label="f_res")
            ax.axvline(r["xs"].max(), color=GREY, lw=1.0)
            if dcls["status"] == "crossed":
                ax.axvline(dcls["x_null_um"], color=RED, ls="--", lw=1.4,
                           label=f"null {dcls['x_null_um']:.1f} µm")
                title2 = f"null IN SPAN at {dcls['x_null_um']:.1f} µm"
            elif dcls["status"] == "at_or_beyond_end":
                ax.axvline(dcls["x_bound_um"], color=RED, ls="--", lw=1.4,
                           label=f"bound {dcls['x_bound_um']:.1f} µm")
                title2 = (f"null AT/BEYOND END (bound {dcls['x_bound_um']:.1f} µm, "
                          f"gap {dcls['gap_Hz']/1e3:.1f} kHz)")
            else:
                title2 = "no approach to f_res in span"
            sb = out[key]["params"]["setback_um"]
            ax.set_title(f"{key}\nfit setback {sb:.1f} µm — {title2}", fontsize=9)
            ax.set_xlabel("position (µm)"); ax.set_ylabel("frequency (kHz)")
            ax.legend(fontsize=7.5, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "eb_fit_batch.png"), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    with open(os.path.join(a.out, "eb_fit_batch.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"wrote {a.out}/eb_fit_batch.json + eb_fit_batch.png")


if __name__ == "__main__":
    main()

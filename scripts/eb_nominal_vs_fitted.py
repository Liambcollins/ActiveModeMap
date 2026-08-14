#!/usr/bin/env python
"""EB model, FITTED geometry vs NOMINAL geometry, on the dense runs.

Nominal = the ProbeGeometry defaults (BudgetSensors Multi75E-G): f0 = 75 kHz,
tip setback = 11 µm. Fitted = f0 and setback free (eb_fit_batch.py). Both
variants fit the remaining physics (alpha, kcone, Q, eps) with the same
machinery — analytic gain, log-amplitude residuals, resonance-alignment gate —
so the ONLY difference is whether the geometry is taken from the spec sheet
or from the data. The comparison quantifies what believing the spec sheet
costs on this batch.

    python scripts/eb_nominal_vs_fitted.py <data_root> --out <outdir>
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from activemodemap.lowrank import classify_null
from eb_fit_real import EBFitter, load, fit
from eb_fit_batch import branch_of

BLUE, GREEN, RED, GOLD, GREY = "#2a78d6", "#2f9e5f", "#e34948", "#e8a13c", "#8a8f98"
NOM_F0_KHZ, NOM_SB_UM = 75.0, 11.0


def fit_nominal(fitter, Zd):
    """Fit alpha/kcone/Q/eps with geometry pinned to the spec sheet."""
    jd = int(np.argmax(np.abs(Zd).max(0)))
    fres_dat = fitter.F[jd]
    best = None
    for la in np.arange(1.6, 4.61, 0.15):
        for lk in (2.5, 2.824, 3.1):
            for le in (-2.0, -1.0, -0.3, 0.3):
                p0 = [la, lk, 2.35, le, NOM_F0_KHZ, NOM_SB_UM]
                try:
                    Zp, _ = fitter.predict(p0)
                except Exception:
                    continue
                jm = int(np.argmax(np.abs(Zp).max(0)))
                if abs(fitter.F[jm] - fres_dat) > 8e3:
                    continue
                r = fitter.residuals(p0, Zd)
                c = float(r @ r)
                if best is None or c < best[0]:
                    best = (c, p0)
    gate_relaxed = False
    if best is None:
        # nominal geometry cannot even align the resonance within 8 kHz —
        # that is itself a result; retry with a loose 30 kHz gate so the
        # least-squares still has a starting point to characterise the misfit
        gate_relaxed = True
        for la in np.arange(1.6, 4.61, 0.15):
            for le in (-2.0, -0.3):
                p0 = [la, 2.824, 2.35, le, NOM_F0_KHZ, NOM_SB_UM]
                try:
                    Zp, _ = fitter.predict(p0)
                except Exception:
                    continue
                jm = int(np.argmax(np.abs(Zp).max(0)))
                if abs(fitter.F[jm] - fres_dat) > 30e3:
                    continue
                r = fitter.residuals(p0, Zd)
                c = float(r @ r)
                if best is None or c < best[0]:
                    best = (c, p0)
    if best is None:
        raise RuntimeError("nominal geometry cannot align the contact resonance")
    eps = 1e-4
    lo = [1.0, 1.5, 1.8, -3.0, NOM_F0_KHZ - eps, NOM_SB_UM - eps]
    hi = [5.0, 3.5, 3.0,  1.0, NOM_F0_KHZ + eps, NOM_SB_UM + eps]
    sol = least_squares(fitter.residuals, best[1], args=(Zd, None),
                        bounds=(lo, hi), xtol=1e-12, diff_step=0.02,
                        max_nfev=400)
    return sol, gate_relaxed


def evaluate(fitter, sol, Zd, fb, L=225.0):
    Zp, _ = fitter.predict(sol.x)
    g = float(np.exp(np.median(np.log(np.abs(Zd) + 1e-12)
                               - np.log(np.abs(Zp) + 1e-12))))
    Zp = Zp * g
    rel = float(np.median(np.abs(np.abs(Zp) - np.abs(Zd)).mean(1)
                          / (np.abs(Zd).mean(1) + 1e-30)))
    # end-region residual (last 20 µm) — where the null physics lives
    tail = fitter.x >= fitter.x.max() - 20
    rel_end = float(np.median(np.abs(np.abs(Zp[tail]) - np.abs(Zd[tail])).mean(1)
                              / (np.abs(Zd[tail]).mean(1) + 1e-30)))
    jd = int(np.argmax(np.abs(Zd).max(0)))
    jm = int(np.argmax(np.abs(Zp).max(0)))
    xq = np.arange(fitter.x.min(), L + 0.25, 0.5)
    Zq = fitter.predict(sol.x, xq)[0] * g
    fa_m, fres_m, _ = branch_of(xq, Zq, fb)
    cls = classify_null(xq, fa_m, fres_m)
    return dict(gain=g, Zp=Zp, Zq=Zq, xq=xq, rel=rel, rel_end=rel_end,
                fres_model_kHz=float(fitter.F[jm] / 1e3),
                fres_data_kHz=float(fitter.F[jd] / 1e3),
                fres_err_kHz=float((fitter.F[jm] - fitter.F[jd]) / 1e3),
                null=cls, fa_model=fa_m)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root"); ap.add_argument("--out", default="reanalysis_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    RUNS = [("Demo4", (330e3, 470e3)), ("Demo6Fullgrid", (330e3, 470e3))]
    out = {}
    fig, axes = plt.subplots(len(RUNS), 3, figsize=(14.6, 3.8 * len(RUNS)),
                             squeeze=False)
    for row, (name, band) in enumerate(RUNS):
        folder = os.path.join(a.root, name)
        xs, fb, Zd = load(folder, band=band)
        fitter = EBFitter(xs, fb)
        sol_fit = fit(fitter, Zd, verbose=False)
        ev_fit = evaluate(fitter, sol_fit, Zd, fb)
        sol_nom, relaxed = fit_nominal(fitter, Zd)
        ev_nom = evaluate(fitter, sol_nom, Zd, fb)
        fa_d, fres_d, ires = branch_of(xs, Zd, fb)
        cls_d = classify_null(xs, fa_d, fres_d)

        def _null_str(c):
            if c["status"] == "crossed":
                return f"crossed {c['x_null_um']:.1f}"
            if c["status"] == "at_or_beyond_end":
                return f"beyond-end bound {c['x_bound_um']:.1f}"
            return "no null"

        out[name] = dict(
            fitted=dict(params=list(map(float, sol_fit.x)), rel=ev_fit["rel"],
                        rel_end=ev_fit["rel_end"],
                        fres_err_kHz=ev_fit["fres_err_kHz"],
                        null=dict(ev_fit["null"])),
            nominal=dict(params=list(map(float, sol_nom.x)), rel=ev_nom["rel"],
                         rel_end=ev_nom["rel_end"],
                         fres_err_kHz=ev_nom["fres_err_kHz"],
                         gate_relaxed=relaxed,
                         null=dict(ev_nom["null"])),
            data_null=dict(cls_d), f_res_data_kHz=fres_d / 1e3)
        print(f"{name}: FITTED  f0={sol_fit.x[4]:.1f} sb={sol_fit.x[5]:.1f} "
              f"resid {100*ev_fit['rel']:.0f}% (end {100*ev_fit['rel_end']:.0f}%) "
              f"f_res err {ev_fit['fres_err_kHz']:+.1f} kHz "
              f"null {_null_str(ev_fit['null'])}")
        print(f"{name}: NOMINAL f0=75.0 sb=11.0 "
              f"resid {100*ev_nom['rel']:.0f}% (end {100*ev_nom['rel_end']:.0f}%) "
              f"f_res err {ev_nom['fres_err_kHz']:+.1f} kHz "
              f"null {_null_str(ev_nom['null'])}"
              + ("  [8 kHz gate relaxed to 30 kHz]" if relaxed else ""))
        print(f"{name}: DATA    f_res {fres_d/1e3:.1f} kHz, "
              f"null {_null_str(cls_d)}")

        # ---- panels: on-res profile | branch | near-tip spectrum ----
        ax = axes[row]
        ires_d = int(np.argmax(np.abs(Zd[np.argmax(np.abs(Zd).max(1))])))
        ax[0].semilogy(xs, np.abs(Zd[:, ires_d]), ".", color="k", ms=4,
                       label="data")
        for ev, c, lb in ((ev_fit, BLUE, "fitted geometry"),
                          (ev_nom, RED, "nominal geometry")):
            ax[0].semilogy(ev["xq"], np.abs(ev["Zq"][:, ires_d]), color=c,
                           lw=1.5, label=f"{lb} ({100*ev['rel']:.0f}%)")
        ax[0].set_title(f"{name}: on-resonance |Z|", fontsize=10)
        ax[0].set_xlabel("position (µm)"); ax[0].set_ylabel("|Z| (V)")
        ax[0].legend(fontsize=8, frameon=False)

        ax[1].plot(xs, fa_d / 1e3, ".", color="k", ms=3.5, label="data branch")
        for ev, c, lb in ((ev_fit, BLUE, "fitted"), (ev_nom, RED, "nominal")):
            ax[1].plot(ev["xq"], ev["fa_model"] / 1e3, color=c, lw=1.4,
                       label=f"{lb}: {_null_str(ev['null'])} µm")
        ax[1].axhline(fres_d / 1e3, color="k", ls=":", lw=1.0)
        ax[1].axvline(225, color=GREY, lw=0.9)
        ax[1].set_title("antiresonance branch fa(x) & null placement",
                        fontsize=10)
        ax[1].set_xlabel("position (µm)"); ax[1].set_ylabel("frequency (kHz)")
        ax[1].legend(fontsize=8, frameon=False)

        j = int(np.argmin(np.abs(xs - (xs.max() - 15))))
        ax[2].semilogy(fb / 1e3, np.abs(Zd[j]), color="k", lw=1.0,
                       label=f"data x={xs[j]:.0f}")
        for ev, c, lb in ((ev_fit, BLUE, "fitted"), (ev_nom, RED, "nominal")):
            ax[2].semilogy(fb / 1e3, np.abs(ev["Zp"][j]), color=c, lw=1.4,
                           label=lb)
        ax[2].set_title("spectrum 15 µm from the end: notch shape", fontsize=10)
        ax[2].set_xlabel("frequency (kHz)")
        ax[2].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "eb_nominal_vs_fitted.png"), dpi=150,
                bbox_inches="tight")
    with open(os.path.join(a.out, "eb_nominal_vs_fitted.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"wrote {a.out}/eb_nominal_vs_fitted.png/.json")


if __name__ == "__main__":
    main()

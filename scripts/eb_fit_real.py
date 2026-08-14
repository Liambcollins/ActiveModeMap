#!/usr/bin/env python
"""First test of the physics-informed Euler-Bernoulli fit on REAL dense data.

Fits EBForwardModel to a dense run's mode-A band, extending the usual theta
(log_alpha, log_kcone, log_Q, log_eps, log_A0) with the geometry the synthetic
pipeline takes as known but a real probe does not: the free resonance f0 and
the tip setback dx (the overhang length past the contact point — decisive here,
because the measured null sits ON the overhang), plus a global complex phase.

Two tests:
  FIT    — all dense positions: residual, f_res match, null location vs raw truth.
  EXTRAP — fit only x >= x_split, predict the held-out base side; compare with
           the Chebyshev-4 fit given the same restricted positions. This is the
           one thing the low-rank model cannot do by construction.

    python scripts/eb_fit_real.py <data_root>/<run> --out <outdir>
"""
from __future__ import annotations
import argparse, contextlib, glob, io, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from activemodemap.asylum import read_tune_txt, tune_to_complex
from activemodemap.lowrank import band_mask, chebyshev_basis
from activemodemap.forward_model import EBForwardModel, ProbeGeometry

_X = re.compile(r"_X(\d+)_")
BLUE, GREEN, RED, GOLD = "#2a78d6", "#2f9e5f", "#e34948", "#e8a13c"


def load(folder, band=(330e3, 470e3), fdec=2, xdec=4):
    d, buf = {}, io.StringIO()
    with contextlib.redirect_stdout(buf):
        for p in sorted(glob.glob(os.path.join(folder, "Tune_*.txt"))):
            m = _X.search(p)
            if m:
                f, Z = tune_to_complex(read_tune_txt(p))
                d[int(m.group(1)) / 1000.0] = (f, Z)
    xs = np.array(sorted(d))
    keep = np.array([i for i in range(xs.size)
                     if (i > 0 and xs[i] - xs[i-1] <= 1.2)
                     or (i < xs.size - 1 and xs[i+1] - xs[i] <= 1.2)])
    xs = xs[keep]
    F0 = d[xs[0]][0]
    lo = max(d[x][0].min() for x in xs); hi = min(d[x][0].max() for x in xs)
    F = F0[(F0 >= lo) & (F0 <= hi)]
    Z = np.array([np.interp(F, d[x][0], d[x][1].real)
                  + 1j * np.interp(F, d[x][0], d[x][1].imag) for x in xs])
    bm = band_mask(F, band)
    return xs[::xdec], F[bm][::fdec], Z[::xdec][:, bm][:, ::fdec]


class EBFitter:
    """EB model with (f0, setback) geometry as fit parameters.

    The model is rebuilt when geometry changes (cached), theta evaluated inside.
    Parameter vector p = [log_alpha, log_kcone, log_Q, log_eps, f0_kHz,
    setback_um]. The overall complex gain is NOT a fit parameter: it is solved
    analytically per evaluation (g = <Zm,Zd>/<Zm,Zm>), which removes the
    zero-amplitude local minimum a misaligned resonance otherwise falls into.
    """
    def __init__(self, x_um, F_Hz, L_um=225.0, nx=241, n_modes=14):
        self.x, self.F, self.L = x_um, F_Hz, L_um
        self.nx, self.nm = nx, n_modes
        self._cache = {}

    def _model(self, f0_hz, setback):
        key = (round(f0_hz, 0), round(setback, 2))
        if key not in self._cache:
            g = ProbeGeometry(f0_hz=f0_hz, L_um=self.L, tip_setback_um=setback)
            fs = f0_hz / 1.87510407 ** 2
            om = self.F / fs
            self._cache[key] = EBForwardModel(
                geom=g, n_modes=self.nm, nx=self.nx, nf=self.F.size,
                omega_lo=float(om.min()), omega_hi=float(om.max()))
            if len(self._cache) > 200:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[key]

    def predict(self, p, x_um=None, Zdat=None):
        th = np.array([p[0], p[1], p[2], p[3], 0.0], float)   # log_A0 = 0
        m = self._model(p[4] * 1e3, p[5])
        Zm = m.measured(th, +1.0)                    # (nf, nx_model)
        x = self.x if x_um is None else x_um
        idx = np.clip(np.round(np.asarray(x) / self.L * (self.nx - 1)).astype(int),
                      0, self.nx - 1)
        Zp = Zm[:, idx].T
        if Zdat is not None:                          # analytic complex gain
            g = np.vdot(Zp, Zdat) / (np.vdot(Zp, Zp) + 1e-300)
            Zp = Zp * g
        return Zp, m

    def residuals(self, p, Zdat, x_um=None):
        """LOG-AMPLITUDE residuals. Real tunes carry an instrumental
        phase-vs-frequency background (drive-path delay) the EB model does not
        describe; a complex-valued fit stalls at ~100% residual because no
        single complex gain can align the phases. The nulls and the notch live
        in |Z|, so fit |Z| — the same choice inference.py makes with
        _logamp_residuals. Gain solved analytically in log space."""
        Zp, _ = self.predict(p, x_um)
        la_p = np.log(np.abs(Zp) + 1e-12)
        la_d = np.log(np.abs(Zdat) + 1e-12)
        return (la_p - la_d - np.median(la_d - la_p) * (-1.0)).ravel() \
            if False else (la_p + np.median(la_d - la_p) - la_d).ravel()


def branch_null(x, Zb, fb):
    ires = int(np.argmax(np.abs(Zb[np.argmax(np.abs(Zb).max(1))])))
    fres = fb[ires]
    fa = np.full(x.size, np.nan)
    for i in range(x.size):
        j = int(np.argmin(np.abs(Zb[i])))
        if 0 < j < fb.size - 1:
            fa[i] = fb[j]
    g = fa - fres; ok = np.isfinite(g); pair = ok[:-1] & ok[1:]
    idx = np.where(pair & (np.sign(g[:-1]) != np.sign(g[1:])))[0]
    if idx.size == 0:
        return np.nan, fres
    xc = x[idx] + (x[idx+1]-x[idx]) * g[idx] / (g[idx]-g[idx+1])
    return float(xc[-1]), float(fres)


def fit(fitter, Zdat, x_um=None, verbose=True, fres_dat=None):
    """Coarse (f0, alpha, setback) scan with resonance-alignment gate, then LS.

    The gate is what makes this work on real data: combinations whose model
    contact resonance misses the measured f_res by > 8 kHz are rejected before
    scoring, so the optimizer starts aligned instead of discovering that a
    zero-gain fit is locally cheapest.
    """
    if fres_dat is None:
        j = int(np.argmax(np.abs(Zdat).max(0)))
        fres_dat = fitter.F[j]
    best = None
    for f0k in np.arange(45.0, 105.1, 5.0):
        for sb in (5.0, 11.0, 18.0):
            for la in np.arange(2.0, 4.41, 0.3):
                for le in (-2.0, -1.0, -0.3, 0.3):
                    p0 = [la, np.log10(2000/3), 2.35, le, f0k, sb]
                    try:
                        Zp, _ = fitter.predict(p0, x_um)
                    except Exception:
                        continue
                    jm = int(np.argmax(np.abs(Zp).max(0)))
                    if abs(fitter.F[jm] - fres_dat) > 8e3:
                        continue
                    r = fitter.residuals(p0, Zdat, x_um)
                    c = float(r @ r)
                    if best is None or c < best[0]:
                        best = (c, p0)
    if best is None:
        raise RuntimeError("no geometry aligned the contact resonance")
    lo = [1.0, 1.5, 1.8, -3.0, 40.0, 1.0]
    hi = [5.0, 3.5, 3.0,  1.0, 110.0, 30.0]
    sol = least_squares(fitter.residuals, best[1], args=(Zdat, x_um),
                        bounds=(lo, hi), xtol=1e-12, diff_step=0.02, max_nfev=400)
    if verbose:
        n = ["log_a", "log_kc", "log_Q", "log_eps", "f0_kHz", "setback_um"]
        print("  fit:", ", ".join(f"{k}={v:.2f}" for k, v in zip(n, sol.x)))
    return sol


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_folder"); ap.add_argument("--out", default="reanalysis_out")
    ap.add_argument("--split-um", type=float, default=150.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    xs, fb, Zd = load(a.run_folder)
    name = os.path.basename(a.run_folder.rstrip("/"))
    print(f"{name}: {xs.size} positions x {fb.size} freqs "
          f"({fb[0]/1e3:.0f}-{fb[-1]/1e3:.0f} kHz)")
    truth_x, truth_f = branch_null(xs, Zd, fb)
    print(f"  raw truth: null x={truth_x:.1f}, f_res={truth_f/1e3:.1f} kHz")
    out = {"run": name, "truth_null_x": truth_x, "truth_fres_kHz": truth_f/1e3}

    fitter = EBFitter(xs, fb)
    # ---------- FIT on everything ----------
    sol = fit(fitter, Zd)
    Zp, m = fitter.predict(sol.x)
    Zp = Zp * float(np.exp(np.median(np.log(np.abs(Zd) + 1e-12)
                                     - np.log(np.abs(Zp) + 1e-12))))
    rel = float(np.median(np.abs(np.abs(Zp) - np.abs(Zd)).mean(1)
                          / (np.abs(Zd).mean(1) + 1e-30)))
    xq = np.arange(xs.min(), xs.max() + 0.25, 0.5)
    _Zm = fitter.predict(sol.x)[0]
    _g = float(np.exp(np.median(np.log(np.abs(Zd) + 1e-12)
                                - np.log(np.abs(_Zm) + 1e-12))))
    Zq = fitter.predict(sol.x, xq)[0] * _g
    eb_null, eb_fres = branch_null(xq, Zq, fb)
    out["full_fit"] = dict(params=list(map(float, sol.x)), rel_resid=rel,
                           null_x=eb_null, null_err_um=float(abs(eb_null-truth_x))
                           if np.isfinite(eb_null) else None,
                           fres_kHz=eb_fres/1e3)
    print(f"  FULL fit: rel resid {100*rel:.0f}%, f_res {eb_fres/1e3:.1f} kHz, "
          f"null x={eb_null:.1f} (err {abs(eb_null-truth_x):.1f} µm)")

    # ---------- EXTRAPOLATION: fit tip side only, predict base side ----------
    tipside = xs >= a.split_um
    base = ~tipside
    sol2 = fit(EBFitter(xs[tipside], fb), Zd[tipside], verbose=False)
    _ft = EBFitter(xs[tipside], fb)
    _gm = _ft.predict(sol2.x)[0]
    _g2 = float(np.exp(np.median(np.log(np.abs(Zd[tipside]) + 1e-12)
                                 - np.log(np.abs(_gm) + 1e-12))))
    Zbase_eb = EBFitter(xs, fb).predict(sol2.x, xs[base])[0] * _g2
    rel_eb = float(np.median(np.abs(np.abs(Zbase_eb) - np.abs(Zd[base])).mean(1)
                             / (np.abs(Zd[base]).mean(1) + 1e-30)))
    # Chebyshev-4 given the same tip-side positions, extrapolated to base side
    Ball = chebyshev_basis(np.concatenate([xs[tipside], xs[base]]), 4)
    coef, *_ = np.linalg.lstsq(Ball[:tipside.sum()], Zd[tipside], rcond=None)
    Zbase_ch = Ball[tipside.sum():] @ coef
    rel_ch = float(np.median(np.abs(np.abs(Zbase_ch) - np.abs(Zd[base])).mean(1)
                             / (np.abs(Zd[base]).mean(1) + 1e-30)))
    out["extrapolation"] = dict(split_um=a.split_um,
                                n_fit=int(tipside.sum()), n_holdout=int(base.sum()),
                                eb_rel_err=rel_eb, cheb4_rel_err=rel_ch)
    print(f"  EXTRAP (fit x>={a.split_um:.0f}, predict x<{a.split_um:.0f}): "
          f"EB {100*rel_eb:.0f}% vs cheb4 {100*rel_ch:.0f}% median rel err")

    # ---------- figure ----------
    ires = int(np.argmax(np.abs(Zd[np.argmax(np.abs(Zd).max(1))])))
    fig, ax = plt.subplots(1, 3, figsize=(14.2, 3.7))
    ax[0].semilogy(xs, np.abs(Zd[:, ires]), ".", color="k", ms=4, label="data")
    ax[0].semilogy(xq, np.abs(Zq[:, ires]), color=BLUE, lw=1.6, label="EB fit")
    if np.isfinite(eb_null):
        ax[0].axvline(eb_null, color=BLUE, ls="--", lw=1.2)
    ax[0].axvline(truth_x, color=RED, ls=":", lw=1.6)
    ax[0].set_title(f"{name}: on-resonance |Z|, EB fit (resid {100*rel:.0f}%)\n"
                    f"null: EB {eb_null:.1f} vs raw {truth_x:.1f} µm", fontsize=9.5)
    ax[0].set_xlabel("position (µm)"); ax[0].legend(fontsize=8, frameon=False)
    j = int(np.argmin(np.abs(xs - xs.max() + 20)))
    ax[1].semilogy(fb/1e3, np.abs(Zd[j]), color="k", lw=1.0, label=f"data x={xs[j]:.0f}")
    ax[1].semilogy(fb/1e3, np.abs(Zp[j]), color=BLUE, lw=1.4, label="EB fit")
    ax[1].set_title("spectrum near the tip: notch shape", fontsize=9.5)
    ax[1].set_xlabel("frequency (kHz)"); ax[1].legend(fontsize=8, frameon=False)
    ax[2].semilogy(xs[base], np.abs(Zd[base][:, ires]), ".", color="k", ms=5, label="held-out data")
    ax[2].semilogy(xs[base], np.abs(Zbase_eb[:, ires]), "o", color=GREEN, ms=4,
                   mfc="none", label=f"EB extrap ({100*rel_eb:.0f}%)")
    ax[2].semilogy(xs[base], np.abs(Zbase_ch[:, ires]), "x", color=GOLD, ms=5,
                   label=f"cheb4 extrap ({100*rel_ch:.0f}%)")
    ax[2].set_title(f"extrapolation to x < {a.split_um:.0f} µm\n"
                    f"(both fit only x ≥ {a.split_um:.0f})", fontsize=9.5)
    ax[2].set_xlabel("position (µm)"); ax[2].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"{name}_eb_fit.png"), dpi=150, bbox_inches="tight")
    with open(os.path.join(a.out, f"{name}_eb_fit.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"wrote {name}_eb_fit.png/.json")


if __name__ == "__main__":
    main()

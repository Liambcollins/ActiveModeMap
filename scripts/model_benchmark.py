#!/usr/bin/env python
"""Model benchmark + ES-null search + piezo/ES deconvolution validation.

Uses the dense sweeps as ground truth to score reconstruction models the
experiment could have used instead of the global Chebyshev low-rank fit:

  * cheb4 / cheb6 — global Chebyshev basis (cheb4 = what the loop runs today)
  * spline        — natural cubic spline through the sampled positions
  * gp            — Gaussian-process regression in x (RBF kernel, one n×n solve
                    shared by every frequency column; lengthscale by LOO-CV)

Each model sees the SAME position subsets (uniform, and uniform with 3 extra
points in the last 8 µm — "tip-refined"), is fit over the mode band, and is
scored against the held-out dense positions on (i) median relative map error
and (ii) D-NS error vs the raw dense truth. Mode B is scored on map error where
swept (its nulls are still open until the dense sweep completes).

Also: finds EVERY spatial null of the electrostatic channel along the full
span (not only near the tip), and runs the deconvolution validity battery on
the two-domain data (notch preservation, channel correlation, gain-imbalance
sensitivity).

    python scripts/model_benchmark.py <data_root> --out <outdir>
"""
from __future__ import annotations
import argparse, contextlib, glob, io, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from activemodemap.asylum import read_tune_txt, tune_to_complex
from activemodemap.lowrank import (band_mask, chebyshev_basis, reconstruct_map,
                                   resonance_index)
from activemodemap.series import separate_domains, separate_channels

_X = re.compile(r"_X(\d+)_")
BLUE, GREEN, RED, GOLD, PURP = "#2a78d6", "#2f9e5f", "#e34948", "#e8a13c", "#8458b3"


def load_tunes(folder):
    d, buf = {}, io.StringIO()
    with contextlib.redirect_stdout(buf):
        for p in sorted(glob.glob(os.path.join(folder, "Tune_*.txt"))):
            m = _X.search(p)
            if m:
                f, Z = tune_to_complex(read_tune_txt(p))
                d[int(m.group(1)) / 1000.0] = (f, Z)
    xs = np.array(sorted(d))
    F0 = d[xs[0]][0]
    lo = max(d[x][0].min() for x in xs); hi = min(d[x][0].max() for x in xs)
    F = F0[(F0 >= lo) & (F0 <= hi)]
    Z = np.array([np.interp(F, d[x][0], d[x][1].real)
                  + 1j * np.interp(F, d[x][0], d[x][1].imag) for x in xs])
    return xs, F, Z


# ---------------- models ---------------------------------------------------- #
def fit_cheb(xs, Zs, xq, rank):
    B = chebyshev_basis(np.concatenate([xs, xq]), min(rank, len(xs)))
    Bs, Bq = B[:len(xs)], B[len(xs):]
    coef, *_ = np.linalg.lstsq(Bs, Zs, rcond=None)
    return Bq @ coef


def fit_spline(xs, Zs, xq):
    cs = CubicSpline(xs, Zs, axis=0)
    return cs(np.clip(xq, xs.min(), xs.max()))


def fit_gp(xs, Zs, xq):
    """RBF GP, complex targets; one kernel solve serves all frequency columns.
    Lengthscale by leave-one-out CV on the sampled positions."""
    def K(a, b, ell):
        return np.exp(-0.5 * ((a[:, None] - b[None, :]) / ell) ** 2)
    best, best_err = None, np.inf
    for ell in (8.0, 15.0, 30.0, 60.0, 120.0):
        Km = K(xs, xs, ell) + 1e-6 * np.eye(len(xs))
        Ki = np.linalg.inv(Km)
        # LOO residual (Rasmussen 5.10-5.12), applied per frequency then pooled
        alpha = Ki @ Zs
        loo = alpha / np.diag(Ki)[:, None]
        err = float(np.median(np.abs(loo)))
        if err < best_err:
            best_err, best = err, ell
    Km = K(xs, xs, best) + 1e-6 * np.eye(len(xs))
    return K(xq, xs, best) @ np.linalg.solve(Km, Zs), best


def dns_raw(x, Zb, fb):
    """Branch-crossing null from a map sampled on x (edge-guarded)."""
    ires = int(np.argmax(np.abs(Zb[np.argmax(np.abs(Zb).max(1))])))
    fres = fb[ires]
    fa = np.full(x.size, np.nan)
    for i in range(x.size):
        j = int(np.argmin(np.abs(Zb[i])))
        if 0 < j < fb.size - 1:
            fa[i] = fb[j]
    g = fa - fres; ok = np.isfinite(g)
    pair = ok[:-1] & ok[1:]
    idx = np.where(pair & (np.sign(g[:-1]) != np.sign(g[1:])))[0]
    if idx.size == 0:
        return np.nan
    xc = x[idx] + (x[idx + 1] - x[idx]) * g[idx] / (g[idx] - g[idx + 1])
    return float(xc[-1])


def all_spatial_nulls(x, M, fb, min_depth=3.0):
    """Every position where the (rotated) response changes sign along x, at the
    band's strongest frequency — the whole span, not just the tip side.
    `min_depth`: |Z| must dip at least this factor below its neighbourhood."""
    ires = int(np.argmax(np.abs(M[np.argmax(np.abs(M).max(1))])))
    z = M[:, ires]
    v = (z * np.exp(-1j * np.angle(z[int(np.argmax(np.abs(z)))]))).real
    out = []
    for i in np.where(np.diff(np.sign(v)) != 0)[0]:
        xc = x[i] + (x[i + 1] - x[i]) * v[i] / (v[i] - v[i + 1])
        lo, hi = max(0, i - 6), min(x.size, i + 7)
        depth = np.median(np.abs(z[lo:hi])) / (min(abs(z[i]), abs(z[i + 1])) + 1e-30)
        if depth >= min_depth:
            out.append((float(xc), float(depth)))
    return out, float(fb[ires])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root"); ap.add_argument("--out", default="reanalysis_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    results = {"benchmark": {}, "es_nulls": {}, "deconv": {}}

    # ================= 1. model benchmark on the dense grids ================ #
    cases = []
    for run, band, mode in (("Demo4", (330e3, 470e3), "A"),
                            ("Demo6Fullgrid", (330e3, 470e3), "A"),
                            ("Demo6Fullgrid", (1090e3, 1240e3), "B")):
        folder = os.path.join(a.root, run)
        if not os.path.isdir(folder):
            continue
        xs, F, Z = load_tunes(folder)
        # dense-only subset as truth
        keep = np.array([i for i in range(xs.size)
                         if (i > 0 and xs[i]-xs[i-1] <= 1.2)
                         or (i < xs.size-1 and xs[i+1]-xs[i] <= 1.2)])
        xd = xs[keep]
        bm = band_mask(F, band)
        if bm.sum() < 32:
            continue
        fb = F[bm][::4]                      # decimate for speed; notch ≳15 pts wide
        Zd = Z[keep][:, bm][:, ::4]
        truth_dns = dns_raw(xd, Zd, fb)
        cases.append((run, mode, xd, fb, Zd, truth_dns))
        print(f"[truth] {run} mode {mode}: dense null x={truth_dns:.1f} "
              f"({xd.max()-truth_dns:.1f} µm from end), {xd.size} positions")

    ns = [5, 6, 8, 10, 12, 16]
    models = ["cheb4", "cheb6", "spline", "gp"]
    schemes = ["uniform", "tip-refined"]
    for run, mode, xd, fb, Zd, truth_dns in cases:
        key = f"{run}:{mode}"
        results["benchmark"][key] = {"truth_dns_x": truth_dns, "n": ns, "models": {}}
        for scheme in schemes:
            for model in models:
                map_err, dns_err = [], []
                for n in ns:
                    if scheme == "uniform":
                        sel = np.unique(np.linspace(0, xd.size - 1, n).astype(int))
                    else:
                        base = np.unique(np.linspace(0, xd.size - 1, max(n - 3, 2)).astype(int))
                        tip = np.where(xd >= xd.max() - 8.0)[0]
                        extra = tip[np.unique(np.linspace(0, tip.size - 1, 3).astype(int))]
                        sel = np.unique(np.concatenate([base, extra]))
                    xs_, Zs_ = xd[sel], Zd[sel]
                    try:
                        if model == "cheb4":
                            Zq = fit_cheb(xs_, Zs_, xd, 4)
                        elif model == "cheb6":
                            Zq = fit_cheb(xs_, Zs_, xd, 6)
                        elif model == "spline":
                            Zq = fit_spline(xs_, Zs_, xd)
                        else:
                            Zq, _ = fit_gp(xs_, Zs_, xd)
                        hold = np.setdiff1d(np.arange(xd.size), sel)
                        rel = np.abs(Zq[hold] - Zd[hold]).mean(1) / (np.abs(Zd[hold]).mean(1) + 1e-30)
                        map_err.append(float(np.median(rel)))
                        d = dns_raw(xd, Zq, fb)
                        dns_err.append(float(abs(d - truth_dns)) if np.isfinite(d) else np.nan)
                    except Exception:
                        map_err.append(np.nan); dns_err.append(np.nan)
                results["benchmark"][key]["models"][f"{model}|{scheme}"] = dict(
                    map_err=map_err, dns_err=dns_err)
        print(f"[bench] {key} done")

    # figure: one row per case — map err + D-NS err vs n
    fig, ax = plt.subplots(len(cases), 2, figsize=(12.6, 3.3 * len(cases)), squeeze=False)
    colors = {"cheb4": BLUE, "cheb6": PURP, "spline": GOLD, "gp": GREEN}
    for r, (run, mode, *_rest) in enumerate(cases):
        key = f"{run}:{mode}"
        for mk, res in results["benchmark"][key]["models"].items():
            model, scheme = mk.split("|")
            ls = "-" if scheme == "uniform" else "--"
            ax[r][0].semilogy(ns, res["map_err"], ls, color=colors[model], lw=1.6,
                              label=f"{model} ({scheme})" if r == 0 else None)
            ax[r][1].plot(ns, res["dns_err"], ls, color=colors[model], lw=1.6)
        ax[r][0].set_ylabel(f"{run}\nmode {mode}\nmedian rel. map error")
        ax[r][1].set_ylabel("|D-NS error| (µm)")
        ax[r][1].set_ylim(0, 12)
        for c in (0, 1):
            ax[r][c].set_xlabel("number of measured positions")
    ax[0][0].legend(fontsize=7.5, ncol=2, frameon=False)
    ax[0][0].set_title("held-out map error (solid: uniform, dashed: +3 tip-refine)", fontsize=10)
    ax[0][1].set_title("null-location error vs dense truth", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "model_benchmark.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ================= 2. ES nulls over the WHOLE span ====================== #
    for name, kind in (("Domains1", "domains"), ("Bias Dependenceb", "bias")):
        ck = glob.glob(os.path.join(a.root, name, "*checkpoint*.npz"))
        if not ck:
            continue
        d = np.load(ck[0], allow_pickle=False)
        conds = json.loads(str(d["conditions"]))
        ser = dict(x_um=np.asarray(d["x_um"]), freq_Hz=np.asarray(d["freq_Hz"]),
                   Z=np.asarray(d["Z"]), conditions=[type("C", (), c)() for c in conds])
        ch = (separate_domains(ser, 1, 2) if kind == "domains"
              else separate_channels(ser, load_nN=conds[0]["load_nN"]))
        F = ch["freq_Hz"]; xs = ser["x_um"]
        entry = {}
        for band, lbl in (((330e3, 470e3), "modeA"), ((1090e3, 1240e3), "modeB")):
            bm = band_mask(F, band)
            if bm.sum() < 16:
                continue
            # reconstruct E on a fine grid with the best cheap model (spline)
            xq = np.arange(xs.min(), xs.max() + 0.25, 0.5)
            Eq = fit_spline(xs, ch["elec"][:, bm][:, ::4], xq)
            nulls, fused = all_spatial_nulls(xq, Eq, F[bm][::4])
            entry[lbl] = dict(f_Hz=fused, nulls=[dict(x_um=n[0],
                              from_end_um=float(xs.max()-n[0]), depth=n[1])
                              for n in nulls])
            print(f"[ES] {name} {lbl} @ {fused/1e3:.0f} kHz: "
                  + (", ".join(f"x={n[0]:.1f} (depth {n[1]:.0f}x)" for n in nulls)
                     or "no deep nulls"))
        results["es_nulls"][name] = entry

    # ================= 3. deconvolution validity (Domains1) ================= #
    ck = glob.glob(os.path.join(a.root, "Domains1", "*checkpoint*.npz"))
    if ck:
        d = np.load(ck[0], allow_pickle=False)
        conds = json.loads(str(d["conditions"]))
        ser = dict(x_um=np.asarray(d["x_um"]), freq_Hz=np.asarray(d["freq_Hz"]),
                   Z=np.asarray(d["Z"]), conditions=[type("C", (), c)() for c in conds])
        xs, F = ser["x_um"], ser["freq_Hz"]
        bm = band_mask(F, (330e3, 470e3)); fb = F[bm]
        Zu, Zd_ = ser["Z"][0][:, bm], ser["Z"][1][:, bm]
        ch = separate_domains(ser, 1, 2)
        P, E = ch["piezo"][:, bm], ch["elec"][:, bm]
        ires = resonance_index(fb, P)
        v = {}
        # (1) notch preservation: antiresonance contrast in P vs raw Z_up
        def contrast(M):
            A = np.abs(M).max(0)
            j = int(np.argmax(A))
            w = (fb > fb[j] + 5e3) & (fb < fb[j] + 90e3)
            return float(A[j] / (A[w].min() + 1e-30))
        v["notch_contrast_P"] = contrast(P); v["notch_contrast_Zup"] = contrast(Zu)
        # (2) channel independence: |corr| of on-res spatial profiles
        pr, er = np.abs(P[:, ires]), np.abs(E[:, ires])
        v["onres_corr_P_E"] = float(np.corrcoef(pr, er)[0, 1])
        # (3) gain-imbalance sensitivity: rescale Z_down by g and redo
        g = np.median(np.abs(Zu[:, ires]) / (np.abs(Zd_[:, ires]) + 1e-30))
        v["gain_ratio_up_down"] = float(g)
        Pg, Eg = (Zu - g * Zd_) / 2, (Zu + g * Zd_) / 2
        xq = np.arange(xs.min(), xs.max() + 0.25, 0.5)
        for tag, EE in (("raw", E), ("gain_corrected", Eg)):
            Eq = fit_spline(xs, EE[:, ::2], xq)
            nulls, _ = all_spatial_nulls(xq, Eq, fb[::2], min_depth=2.0)
            v[f"es_nulls_{tag}"] = [round(n[0], 1) for n in nulls]
        # (4) elec fraction + (5) EB-synthetic numbers quoted from prior validation
        v["elec_frac"] = float(ch["elec_frac"])
        v["eb_synthetic_err_um"] = {"dns": 0.48, "desbs": 0.00}
        results["deconv"] = v
        print("[deconv]", json.dumps(v, indent=None)[:400])

        fig, ax = plt.subplots(1, 3, figsize=(13.6, 3.6))
        ax[0].semilogy(xs, np.abs(Zu[:, ires]), "o-", color=BLUE, ms=4, label="|Z| domain ↑")
        ax[0].semilogy(xs, np.abs(Zd_[:, ires]), "s-", color=GREEN, ms=4, label="|Z| domain ↓")
        ax[0].set_title("raw on-resonance response, both domains", fontsize=10)
        ax[0].legend(fontsize=8, frameon=False); ax[0].set_xlabel("position (µm)")
        ax[1].semilogy(xs, pr, "o-", color=BLUE, ms=4, label="|P| = |Z↑−Z↓|/2")
        ax[1].semilogy(xs, er, "s-", color=GREEN, ms=4, label="|E| = |Z↑+Z↓|/2")
        ax[1].set_title(f"separated channels (corr = {v['onres_corr_P_E']:+.2f})", fontsize=10)
        ax[1].legend(fontsize=8, frameon=False); ax[1].set_xlabel("position (µm)")
        A_up, A_P = np.abs(Zu).max(0), np.abs(P).max(0)
        ax[2].semilogy(fb/1e3, A_up/A_up.max(), color=BLUE, lw=1.2, label="raw Z↑ (max over x)")
        ax[2].semilogy(fb/1e3, A_P/A_P.max(), color=RED, lw=1.2, label="piezo channel P")
        ax[2].set_title(f"antiresonance notch preserved in P\n(contrast {v['notch_contrast_P']:.0f}× vs {v['notch_contrast_Zup']:.0f}× raw)", fontsize=9.5)
        ax[2].legend(fontsize=8, frameon=False); ax[2].set_xlabel("frequency (kHz)")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "deconv_validation.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    with open(os.path.join(a.out, "model_benchmark.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    print("wrote model_benchmark.json + figures")


if __name__ == "__main__":
    main()

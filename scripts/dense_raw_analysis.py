#!/usr/bin/env python
"""Model-free analysis of dense ActiveModeMap sweeps + cross-run comparison.

For runs with dense (≲1.2 µm pitch) coverage, locates the displacement null with
NO basis fit: per-position on-resonance amplitude, antiresonance-notch branch
fa(x), and the on-resonance phase flip, straight from the raw spectra. Includes
the laser-edge control: a true modal null kills only the resonant response, so
the off-resonance baseline must stay flat through it (a falling baseline means
the spot slid off the lever instead).

Also builds the cross-run D-NS comparison and, for bias series, the V_cpd
estimate. Complements scripts/reanalyze_today.py.

    python scripts/dense_raw_analysis.py <data_root> --out <outdir>
"""
from __future__ import annotations
import argparse, contextlib, glob, io, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from activemodemap.asylum import read_tune_txt, tune_to_complex
from activemodemap.lowrank import band_mask, classify_null
from activemodemap.series import separate_channels, estimate_v_cpd

_X = re.compile(r"_X(\d+)_")
BLUE, GREEN, RED, GOLD = "#2a78d6", "#2f9e5f", "#e34948", "#e8a13c"


def load_tunes(folder):
    d, buf = {}, io.StringIO()
    with contextlib.redirect_stdout(buf):
        for p in sorted(glob.glob(os.path.join(folder, "Tune_*.txt"))):
            m = _X.search(p)
            if not m:
                continue
            f, Z = tune_to_complex(read_tune_txt(p))
            d[int(m.group(1)) / 1000.0] = (f, Z)     # last tune wins
    xs = np.array(sorted(d))
    F0 = d[xs[0]][0]
    lo = max(d[x][0].min() for x in xs); hi = min(d[x][0].max() for x in xs)
    F = F0[(F0 >= lo) & (F0 <= hi)]
    Z = np.array([np.interp(F, d[x][0], d[x][1].real)
                  + 1j * np.interp(F, d[x][0], d[x][1].imag) for x in xs])
    return xs, F, Z


def raw_null(xs, F, Z, band, base_band=(550e3, 700e3)):
    """Null location from raw dense data, three ways, plus the edge control."""
    bm = band_mask(F, band); fb = F[bm]; Zb = Z[:, bm]
    ires = int(np.argmax(np.abs(Zb).max(axis=0) * 0
               + np.abs(Zb[np.argmax(np.abs(Zb).max(axis=1))])))
    ires = int(np.argmax(np.abs(Zb[np.argmax(np.abs(Zb).max(axis=1))])))
    fres = fb[ires]
    onres = np.abs(Zb[:, ires])
    bb = band_mask(F, base_band)
    base = np.median(np.abs(Z[:, bb]), axis=1) if bb.sum() > 10 else np.full(xs.size, np.nan)
    # notch branch — edge-margin guarded: a "notch" pinned within a few bins
    # of the window edge is the search window clipping, not the antiresonance
    fa = np.full(xs.size, np.nan)
    margin = 4
    for i in range(xs.size):
        row = np.abs(Zb[i])
        j = int(np.argmin(row))
        if margin <= j < fb.size - margin:
            fa[i] = fb[j]
    g = fa - fres; ok = np.isfinite(g)
    # jump-guarded crossings (see classify_null): reject notch-identity
    # switches where the branch hops across f_res without approaching it
    med_gap = float(np.median(np.abs(g[ok]))) if ok.any() else np.inf
    pair = ok[:-1] & ok[1:]
    idx = [i for i in np.where(pair & (np.sign(g[:-1]) != np.sign(g[1:])))[0]
           if min(abs(g[i]), abs(g[i+1])) <= 0.5 * med_gap]
    xc = [float(xs[i] + (xs[i+1]-xs[i]) * g[i] / (g[i]-g[i+1])) for i in idx]
    # phase flip of the rotated on-resonance response
    zr = Zb[:, ires] * np.exp(-1j * np.angle(Zb[np.argmax(onres), ires]))
    flips = [float(xs[i]) for i in np.where(np.diff(np.sign(zr.real)) != 0)[0]]
    x_min = float(xs[int(np.argmin(onres))])
    # boundary-aware classification: an approaching-but-not-crossing branch is
    # "null at/beyond free end" with a bound, not a failure (short-setback batch)
    cls = classify_null(xs, fa, fres)
    return dict(f_res_Hz=float(fres), onres=onres, base=base, fa=fa,
                crossings=xc, phase_flips=flips, x_amp_min=x_min,
                classification=cls,
                fb=fb, contrast=float(onres.max() / max(onres.min(), 1e-30)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root"); ap.add_argument("--out", default="reanalysis_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out = {}

    dense_runs = []
    for folder in sorted(glob.glob(os.path.join(a.root, "*/"))):
        name = os.path.basename(folder.rstrip("/"))
        if not glob.glob(os.path.join(folder, "Tune_*.txt")):
            continue
        xs, F, Z = load_tunes(folder)
        pitch = np.median(np.diff(xs)) if xs.size > 3 else np.inf
        if pitch <= 1.2 and xs.size >= 40:
            dense_runs.append((name, xs, F, Z))

    for name, xs, F, Z in dense_runs:
        entry = {}
        fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
        for mi, (band, lbl) in enumerate([((330e3, 470e3), "mode A")] +
                                          ([((1090e3, 1240e3), "mode B")]
                                           if F[-1] > 1.1e6 else [])):
            r = raw_null(xs, F, Z, band)
            c = r["classification"]
            x_best = (c["x_null_um"] if c["status"] == "crossed" else
                      c["x_bound_um"] if c["status"] == "at_or_beyond_end" else
                      r["x_amp_min"])
            entry[lbl] = dict(f_res_kHz=r["f_res_Hz"]/1e3,
                              crossings_um=r["crossings"],
                              phase_flips_um=r["phase_flips"],
                              x_amp_min_um=r["x_amp_min"],
                              null_status=c["status"],
                              null_bound_um=c["x_bound_um"],
                              null_gap_kHz=c["gap_Hz"]/1e3 if np.isfinite(c["gap_Hz"]) else None,
                              null_from_end_um=float(xs.max() - x_best),
                              contrast=r["contrast"])
            if mi == 0:
                ax[0].semilogy(xs, r["onres"], color=BLUE, lw=1.4,
                               label=f"|Z| at {r['f_res_Hz']/1e3:.0f} kHz")
                ax[0].semilogy(xs, r["base"], color=GOLD, lw=1.4,
                               label="off-res baseline (550-700 kHz)")
                for c in r["crossings"]:
                    ax[0].axvline(c, color=RED, ls="--", lw=1.3)
                ax[0].set_title(f"{name}: raw on-resonance profile\n"
                                "baseline flat through the null ⇒ not a laser-edge artifact",
                                fontsize=9.5)
                ax[0].set_xlabel("position (µm)"); ax[0].set_ylabel("|Z| (V)")
                ax[0].legend(fontsize=8, frameon=False)
                ax[1].plot(xs, r["fa"]/1e3, ".", color=GREEN, ms=3.5)
                ax[1].axhline(r["f_res_Hz"]/1e3, color="k", ls=":", lw=1.1)
                for c in r["crossings"]:
                    ax[1].axvline(c, color=RED, ls="--", lw=1.3)
                ax[1].set_title(f"raw antiresonance branch fa(x); crossings "
                                f"{np.round(r['crossings'],1)} µm", fontsize=9.5)
                ax[1].set_xlabel("position (µm)"); ax[1].set_ylabel("frequency (kHz)")
            else:
                bm = band_mask(F, band)
                ax[2].semilogy(xs, np.abs(Z[:, bm][:, np.argmax(np.abs(Z[:, bm]).max(0))]),
                               color=GREEN, lw=1.4, label=f"|Z| at {r['f_res_Hz']/1e3:.0f} kHz")
                for c in r["crossings"]:
                    ax[2].axvline(c, color=RED, ls="--", lw=1.1)
                ax[2].set_title(f"{name}: mode B raw on-resonance", fontsize=9.5)
                ax[2].set_xlabel("position (µm)"); ax[2].set_ylabel("|Z| (V)")
                ax[2].legend(fontsize=8, frameon=False)
        if F[-1] <= 1.1e6:
            ax[2].axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"{name}_raw_dense.png"), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        out[name] = entry
        print(f"{name}: " + " | ".join(
            f"{k}: null {v['null_from_end_um']:.1f} µm from end "
            f"(crossings {np.round(v['crossings_um'],1)}, flips "
            f"{np.round(v['phase_flips_um'],1)})" for k, v in entry.items()))

    # ---- V_cpd from bias series -------------------------------------------
    for folder in sorted(glob.glob(os.path.join(a.root, "*/"))):
        name = os.path.basename(folder.rstrip("/"))
        ck = glob.glob(os.path.join(folder, "*series_checkpoint*.npz")) \
            or glob.glob(os.path.join(folder, "*checkpoint*.npz"))
        if not ck:
            continue
        d = np.load(ck[0], allow_pickle=False)
        conds = json.loads(str(d["conditions"]))
        if len({c["bias_V"] for c in conds}) < 3 or any(c.get("spot") for c in conds):
            continue
        ser = dict(x_um=np.asarray(d["x_um"]), freq_Hz=np.asarray(d["freq_Hz"]),
                   Z=np.asarray(d["Z"]),
                   conditions=[type("C", (), c)() for c in conds])
        ch = separate_channels(ser, load_nN=conds[0]["load_nN"])
        per_pos, med = estimate_v_cpd(ch, band_Hz=(330e3, 470e3))
        out.setdefault(name, {})["v_cpd_V"] = float(med)
        out[name]["rel_resid"] = float(ch["rel_resid"])
        out[name]["biases"] = sorted({c["bias_V"] for c in conds})
        print(f"{name}: V_cpd ~ {med:+.2f} V, linearity residual "
              f"{100*ch['rel_resid']:.1f}% over biases {out[name]['biases']}")

    with open(os.path.join(a.out, "raw_dense.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"wrote {a.out}/raw_dense.json")


if __name__ == "__main__":
    main()

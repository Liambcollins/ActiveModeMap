#!/usr/bin/env python
"""Re-analyse a day's ActiveModeMap runs using EVERY eigenmode in the tune window.

Walks a directory of run folders (each holding the Tune_*.txt files Igor saved,
plus any series/domain checkpoint .npz) and, per run:

  1. loads every spectrum (positions from the _X<nm>_ filename field; bias from
     the DC tag, sample spot from the _S tag; checkpoints used when present,
     since their grids are already aligned),
  2. auto-detects ALL eigenmodes in the window from the position-maximum
     amplitude spectrum,
  3. per mode: band-limited low-rank reconstruction from every available
     position (active-learning + dense alike), D-NS branch crossings, and a
     rank-sensitivity spread (the honest uncertainty),
  4. bias series: complex linear fit in V -> piezo/electrostatic channels,
     D-NS + D-ESBS per mode, V_cpd estimate, linearity residual,
  5. two-domain runs: difference/sum -> D-NS + D-ESBS per mode,
  6. writes per-run figures and a machine-readable summary.json.

Usage:
    python scripts/reanalyze_today.py <data_root> --out <outdir> [--rank 4]

Pure Python; no Igor needed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from activemodemap.asylum import read_tune_txt, tune_to_complex          # noqa: E402
from activemodemap.lowrank import (reconstruct_map, dns_from_map,        # noqa: E402
                                   resonance_index, band_mask, dns_branch,
                                   spatial_null, classify_null)
from activemodemap.series import separate_channels, separate_domains     # noqa: E402

_XPAT = re.compile(r"_X(\d+)_")
_DCPAT = re.compile(r"_DC([mp])(\d+p\d+)V")
_SPAT = re.compile(r"_S(\d+)_")
_LPAT = re.compile(r"_L(\d+)nN")

C_MAP = "viridis"
C_PIEZO, C_ELEC, C_MARK = "#2a78d6", "#2f9e5f", "#e34948"


# --------------------------------------------------------------------------- #
#  Loading                                                                    #
# --------------------------------------------------------------------------- #
def parse_name(fn):
    m = _XPAT.search(fn)
    if not m:
        return None
    x = int(m.group(1)) / 1000.0
    dc = _DCPAT.search(fn)
    bias = 0.0
    if dc:
        bias = float(dc.group(2).replace("p", ".")) * (-1 if dc.group(1) == "m" else 1)
    sp = _SPAT.search(fn)
    ld = _LPAT.search(fn)
    return dict(x_um=x, bias_V=bias, spot=int(sp.group(1)) if sp else None,
                load_nN=float(ld.group(1)) if ld else np.nan)


def load_run(folder):
    """-> dict(kind, x_um, freq_Hz, Z[cond, pos, f], conds, meta) or None."""
    ck = sorted(glob.glob(os.path.join(folder, "*checkpoint*.npz")))
    if ck:
        d = np.load(ck[0], allow_pickle=False)
        conds = json.loads(str(d["conditions"]))
        Z = np.asarray(d["Z"])
        kind = ("domains" if any(c.get("spot") for c in conds)
                else ("bias" if len({c["bias_V"] for c in conds}) > 1 else "map"))
        return dict(kind=kind, x_um=np.asarray(d["x_um"], float),
                    freq_Hz=np.asarray(d["freq_Hz"], float), Z=Z, conds=conds,
                    src=os.path.basename(ck[0]))

    paths = sorted(glob.glob(os.path.join(folder, "Tune_*.txt")))
    if not paths:
        return None
    groups = {}                      # (bias, spot) -> {x: (f, Z)}
    for p in paths:
        meta = parse_name(os.path.basename(p))
        if meta is None:
            continue
        try:
            f, Z = tune_to_complex(read_tune_txt(p))
        except Exception as e:
            print(f"    skipping {os.path.basename(p)}: {e}")
            continue
        key = (meta["bias_V"], meta["spot"])
        # keep the LAST tune at a repeated position (later = re-measured)
        groups.setdefault(key, {})[meta["x_um"]] = (f, Z)
    if not groups:
        return None
    # common positions across groups; common frequency grid across everything
    pos_sets = [set(g) for g in groups.values()]
    xs = np.array(sorted(set.intersection(*pos_sets)))
    all_fz = [fz for g in groups.values() for fz in g.values()]
    F0 = all_fz[0][0]
    lo = max(float(f.min()) for f, _ in all_fz)
    hi = min(float(f.max()) for f, _ in all_fz)
    F = F0[(F0 >= lo - 1e-9) & (F0 <= hi + 1e-9)]
    keys = sorted(groups, key=lambda k: (k[1] or 0, k[0]))
    Z = np.empty((len(keys), xs.size, F.size), complex)
    for i, k in enumerate(keys):
        for j, x in enumerate(xs):
            f, z = groups[k][x]
            if f.shape == F.shape and np.allclose(f, F):
                Z[i, j] = z
            else:
                Z[i, j] = np.interp(F, f, z.real) + 1j * np.interp(F, f, z.imag)
    conds = [dict(bias_V=k[0], spot=k[1],
                  load_nN=parse_name(os.path.basename(paths[0]))["load_nN"])
             for k in keys]
    kind = ("domains" if any(c["spot"] for c in conds)
            else ("bias" if len(conds) > 1 else "map"))
    n_extra = sum(len(g) for g in groups.values()) - len(keys) * xs.size
    return dict(kind=kind, x_um=xs, freq_Hz=F, Z=Z, conds=conds,
                src=f"{len(paths)} tune files"
                    + (f" ({n_extra} at non-shared positions dropped)" if n_extra else ""))


# --------------------------------------------------------------------------- #
#  Mode detection                                                             #
# --------------------------------------------------------------------------- #
def detect_modes(F, Zref, min_prom_rel=0.05, min_sep_Hz=150e3):
    """Eigenmode peaks from the position-maximum amplitude spectrum.

    The max over positions is used rather than the mean so a mode that is weak
    at most positions but strong somewhere still registers. Peaks must stand
    `min_prom_rel` of the global maximum above their surroundings and be
    `min_sep_Hz` apart. Returns [(f_peak, (band_lo, band_hi)), ...]; each band
    is asymmetric (further above than below) because the antiresonance sits
    above the resonance, and bands are clipped at midpoints between modes.
    """
    A = np.abs(Zref).max(axis=0)
    k = max(F.size // 400, 3)
    As = np.convolve(A, np.ones(k) / k, mode="same")
    gmax = As.max()
    cand = []
    w = max(int(min_sep_Hz / np.median(np.diff(F))) // 2, 3)
    for i in range(w, F.size - w):
        if As[i] == As[i - w:i + w + 1].max() and As[i] > min_prom_rel * gmax:
            base = np.median(As[max(0, i - 6 * w):i + 6 * w])
            if As[i] > 3 * base:
                cand.append(i)
    # merge near-duplicates
    peaks = []
    for i in cand:
        if not peaks or F[i] - F[peaks[-1]] > min_sep_Hz:
            peaks.append(i)
        elif As[i] > As[peaks[-1]]:
            peaks[-1] = i
    out = []
    for n, i in enumerate(peaks):
        fpk = float(F[i])
        lo, hi = fpk - 60e3, fpk + 110e3
        if n > 0:
            lo = max(lo, (fpk + F[peaks[n - 1]]) / 2)
        if n < len(peaks) - 1:
            hi = min(hi, (fpk + F[peaks[n + 1]]) / 2)
        lo, hi = max(lo, float(F[0])), min(hi, float(F[-1]))
        out.append((fpk, (lo, hi)))
    return out


# --------------------------------------------------------------------------- #
#  Per-mode reconstruction                                                    #
# --------------------------------------------------------------------------- #
def analyze_mode(x_um, F, Zp, band, rank, span_um):
    """Low-rank reconstruction of one condition's map in one mode band."""
    x_um = np.asarray(x_um, float)
    xg = np.arange(x_um.min(), x_um.max() + 0.25, 0.5)
    sel = [int(np.argmin(np.abs(xg - x))) for x in x_um]
    bm = band_mask(F, band)
    if bm.sum() < 16:
        return None
    fb, Zb = F[bm], Zp[:, bm]
    r = min(rank, len(sel) - 2) if len(sel) > 4 else min(rank, len(sel))
    rec = reconstruct_map(xg, sel, Zb, r)
    ires = resonance_index(fb, Zb)
    dns = dns_from_map(xg, rec["Zrec"], fb, ires)
    cross, fa = dns_branch(xg, rec["Zrec"], fb, ires)
    cls = classify_null(xg, fa, fb[ires])
    sens = {}
    for rr in (r - 1, r, r + 1):
        if 2 <= rr <= len(sel) - 1:
            rc = reconstruct_map(xg, sel, Zb, rr)
            sens[rr] = dns_from_map(xg, rc["Zrec"], fb,
                                    resonance_index(fb, Zb))
    vals = [v for v in sens.values() if np.isfinite(v)]
    spread = (max(vals) - min(vals)) if len(vals) > 1 else np.nan
    return dict(x_grid=xg, freq=fb, Zrec=rec["Zrec"], rank=r,
                f_res_Hz=float(fb[ires]), dns_um=float(dns),
                dns_from_end_um=float(span_um - dns),
                crossings_um=[float(c) for c in cross],
                null_status=cls["status"],
                null_bound_um=cls["x_bound_um"],
                null_gap_kHz=(cls["gap_Hz"] / 1e3
                              if np.isfinite(cls["gap_Hz"]) else None),
                rank_spread_um=float(spread), n_pos=len(sel))


# --------------------------------------------------------------------------- #
#  Figures                                                                    #
# --------------------------------------------------------------------------- #
def fig_map(run_name, res_by_mode, span, out_png):
    n = len(res_by_mode)
    fig, ax = plt.subplots(1, n, figsize=(5.6 * n, 3.6), squeeze=False)
    for a, (fpk, r) in zip(ax[0], res_by_mode.items()):
        ext = [r["x_grid"][0], r["x_grid"][-1], r["freq"][0] / 1e3, r["freq"][-1] / 1e3]
        a.imshow(np.log10(np.abs(r["Zrec"]).T + 1e-14), origin="lower",
                 aspect="auto", extent=ext, cmap=C_MAP)
        if np.isfinite(r["dns_um"]):
            a.axvline(r["dns_um"], color=C_MARK, ls="--", lw=1.6)
        a.set_title(f"{run_name} · mode {fpk / 1e3:.0f} kHz\n"
                    f"D-NS {r['dns_from_end_um']:.1f} µm from tip "
                    f"(±{r['rank_spread_um'] / 2:.1f}, {r['n_pos']} pos)",
                    fontsize=10)
        a.set_xlabel("position (µm)"); a.set_ylabel("frequency (kHz)")
    fig.tight_layout(); fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_channels(run_name, ch_res, span, out_png):
    modes = list(ch_res)
    fig, ax = plt.subplots(2, len(modes), figsize=(5.6 * len(modes), 6.6), squeeze=False)
    for j, fpk in enumerate(modes):
        for i, key in enumerate(("dns", "desbs")):
            r = ch_res[fpk][key]
            a = ax[i][j]
            ext = [r["x_grid"][0], r["x_grid"][-1], r["freq"][0] / 1e3, r["freq"][-1] / 1e3]
            a.imshow(np.log10(np.abs(r["Zrec"]).T + 1e-14), origin="lower",
                     aspect="auto", extent=ext, cmap=C_MAP)
            a.axvline(r["value_um"], color=C_MARK, ls="--", lw=1.6)
            lbl = "piezo → D-NS" if key == "dns" else "electrostatic → D-ESBS"
            a.set_title(f"{run_name} · {fpk / 1e3:.0f} kHz · {lbl}\n"
                        f"{span - r['value_um']:.1f} µm from tip", fontsize=9.5)
            a.set_xlabel("position (µm)"); a.set_ylabel("frequency (kHz)")
    fig.tight_layout(); fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def channel_pair(x_um, F, P, E, band, rank, span):
    """D-NS from the piezo channel, D-ESBS from the electrostatic channel."""
    xg = np.arange(x_um.min(), x_um.max() + 0.25, 0.5)
    sel = [int(np.argmin(np.abs(xg - x))) for x in x_um]
    bm = band_mask(F, band); fb = F[bm]
    out = {}
    r = min(rank, len(sel) - 2) if len(sel) > 4 else min(rank, len(sel))
    for key, M, est in (("dns", P, "branch"), ("desbs", E, "null")):
        Mb = M[:, bm]
        rec = reconstruct_map(xg, sel, Mb, r)
        ires = resonance_index(fb, Mb)
        v = (dns_from_map(xg, rec["Zrec"], fb, ires) if est == "branch"
             else spatial_null(xg, rec["Zrec"], fb, ires))
        out[key] = dict(x_grid=xg, freq=fb, Zrec=rec["Zrec"], value_um=float(v),
                        from_end_um=float(span - v), f_res_Hz=float(fb[ires]))
    return out


# --------------------------------------------------------------------------- #
#  Driver                                                                     #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    ap.add_argument("--out", default="reanalysis_out")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--span-um", type=float, default=225.0,
                    help="probe length; from-tip distances use this")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    summary = {}

    for folder in sorted(glob.glob(os.path.join(a.root, "*/"))):
        name = os.path.basename(folder.rstrip("/"))
        run = load_run(folder)
        if run is None:
            continue
        F, xs = run["freq_Hz"], run["x_um"]
        print(f"\n=== {name}: kind={run['kind']}, {xs.size} positions, "
              f"{F.size} freq pts ({F[0]/1e3:.0f}-{F[-1]/1e3:.0f} kHz), "
              f"{len(run['conds'])} condition(s)  [{run['src']}]")
        ref = run["Z"][0]
        modes = detect_modes(F, ref)
        print("    modes:", ", ".join(f"{f/1e3:.1f} kHz" for f, _ in modes))
        entry = dict(kind=run["kind"], n_positions=int(xs.size),
                     x_range_um=[float(xs.min()), float(xs.max())],
                     n_freq=int(F.size), conds=run["conds"], modes={})

        # per-condition, per-mode maps (condition 0 shown; all stored)
        res_by_mode = {}
        for fpk, band in modes:
            r = analyze_mode(xs, F, ref, band, a.rank, a.span_um)
            if r is None:
                continue
            res_by_mode[fpk] = r
            entry["modes"][f"{fpk/1e3:.1f}kHz"] = {
                k: r[k] for k in ("f_res_Hz", "dns_um", "dns_from_end_um",
                                  "crossings_um", "rank_spread_um", "n_pos", "rank",
                                  "null_status", "null_bound_um", "null_gap_kHz")}
            print(f"      {fpk/1e3:7.1f} kHz: D-NS {r['dns_from_end_um']:6.2f} um "
                  f"from tip (±{r['rank_spread_um']/2:.2f}), "
                  f"crossings {np.round(r['crossings_um'],1)}")
        if res_by_mode:
            fig_map(name, res_by_mode, a.span_um,
                    os.path.join(a.out, f"{name}_map.png"))

        # channel separation
        ser = dict(x_um=xs, freq_Hz=F, Z=run["Z"],
                   conditions=[type("C", (), c)() for c in run["conds"]])
        try:
            if run["kind"] == "bias":
                ch = separate_channels(ser, load_nN=run["conds"][0]["load_nN"])
                entry["linearity_rel_resid"] = ch["rel_resid"]
                print(f"    bias fit over {len(run['conds'])} biases: "
                      f"peak residual {100*ch['rel_resid']:.1f}% of peak")
            elif run["kind"] == "domains":
                sp = sorted({c["spot"] for c in run["conds"]})
                ch = separate_domains(ser, sp[0], sp[1])
                entry["elec_frac"] = ch["elec_frac"]
            else:
                ch = None
            if ch is not None:
                ch_res = {}
                for fpk, band in modes:
                    ch_res[fpk] = channel_pair(xs, F, ch["piezo"], ch["elec"],
                                               band, a.rank, a.span_um)
                    e = entry["modes"].setdefault(f"{fpk/1e3:.1f}kHz", {})
                    e["dns_piezo_from_end_um"] = ch_res[fpk]["dns"]["from_end_um"]
                    e["desbs_from_end_um"] = ch_res[fpk]["desbs"]["from_end_um"]
                    print(f"      {fpk/1e3:7.1f} kHz channels: "
                          f"D-NS {ch_res[fpk]['dns']['from_end_um']:6.2f} | "
                          f"D-ESBS {ch_res[fpk]['desbs']['from_end_um']:6.2f} um from tip")
                fig_channels(name, ch_res, a.span_um,
                             os.path.join(a.out, f"{name}_channels.png"))
        except Exception as e:
            print(f"    channel separation failed: {type(e).__name__}: {e}")
            entry["channel_error"] = str(e)

        summary[name] = entry

    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\nwrote {a.out}/summary.json and figures")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Cross-run D-NS comparison figure, batch-hypothesis aware.

Left: mode-A null distance from the free end for every run — dense model-free
results as filled circles, sparse reconstructions as open circles with rank
spread, "at/beyond end" classifications as left-pointing arrows into the
shaded beyond-the-end region, EB-model nulls as diamonds.
Right: the two dense mode-A profiles overlaid near the end + mode B's interior
node, the practical in-span alternative for this batch.

    python scripts/comparison_figure.py --out /tmp/analysis_out
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, GREEN, RED, GOLD, GREY, NAVY = ("#2a78d6", "#2f9e5f", "#e34948",
                                      "#e8a13c", "#8a8f98", "#1e2761")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/analysis_out")
    a = ap.parse_args()
    S = json.load(open(os.path.join(a.out, "summary.json")))
    R = json.load(open(os.path.join(a.out, "raw_dense.json")))
    E = json.load(open(os.path.join(a.out, "eb_nominal_vs_fitted.json")))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.4),
                                  gridspec_kw=dict(width_ratios=[1.45, 1.0]))

    # ---------------- left: every run's mode-A result -------------------
    rows = []          # (label, value_from_end, err, kind)
    # dense, model-free
    for name in ("Demo4", "Demo6Fullgrid"):
        if name in R and "mode A" in R[name]:
            v = R[name]["mode A"]
            pitch = 1.0 if name == "Demo4" else 0.5
            lbl = f"{name} dense ({'1000' if name=='Demo4' else '2000'} nN)"
            rows.append((lbl, v["null_from_end_um"], pitch, "dense"))
    # EB model nulls with NOMINAL (spec-sheet) geometry — the fitted-geometry
    # variants are degenerate at the end region and not shown (see
    # eb_nominal_vs_fitted.py)
    for key, lbl in (("Demo4", "EB, nominal geom (Demo4)"),
                     ("Demo6Fullgrid", "EB, nominal geom (Demo6)")):
        if key in E and "nominal" in E[key]:
            m = E[key]["nominal"]["null"]
            if m["status"] == "crossed":
                rows.append((lbl, 225.0 - m["x_null_um"], 0.5, "eb"))
            elif m["status"] == "at_or_beyond_end":
                rows.append((lbl, 225.0 - m["x_bound_um"], 0.5, "eb_beyond"))
    # sparse reconstructions
    for run, e in S.items():
        if not isinstance(e, dict) or "modes" not in e:
            continue
        if run in ("Demo4", "Demo6Fullgrid"):
            continue                      # dense versions already shown
        for mk, v in e["modes"].items():
            if not mk.startswith("38"):   # mode A only
                continue
            st = v.get("null_status")
            if st == "crossed":
                rows.append((run, v["dns_from_end_um"],
                             max(v.get("rank_spread_um", 0) / 2, 0.2), "sparse"))
            elif st == "at_or_beyond_end" and np.isfinite(v.get("null_bound_um", np.nan)):
                rows.append((run, e["x_range_um"][1] - v["null_bound_um"], 0.5,
                             "sparse_beyond"))
            else:
                rows.append((run, np.nan, 0, "inconclusive"))
    rows = rows[::-1]
    ys = np.arange(len(rows))
    ax.axvspan(-9, 0, color=RED, alpha=0.08)
    ax.axvline(0, color=RED, lw=1.2)
    ax.text(-4.5, len(rows) / 2.0, "beyond\nfree end", ha="center",
            va="center", fontsize=8.5, color=RED)
    for y, (lbl, v, err, kind) in zip(ys, rows):
        if kind == "inconclusive" or not np.isfinite(v):
            ax.text(20.5, y, "no in-span crossing — sampling can't bound it",
                    va="center", fontsize=8, color=GREY, style="italic")
            ax.plot([20], [y], marker="x", color=GREY, ms=6)
            continue
        if v > 31:                      # off-axis (e.g. Demo1's morning frame)
            ax.annotate(f"→ {v:.0f} µm (truncated-tune morning run, own frame)",
                        xy=(31, y), xytext=(20.5, y), va="center", fontsize=8,
                        color=GREY, style="italic")
            continue
        c = dict(dense=BLUE, eb=GREEN, eb_beyond=GREEN, sparse=GOLD,
                 sparse_beyond=GOLD)[kind]
        mk = dict(dense="o", eb="D", eb_beyond="D", sparse="o",
                  sparse_beyond="o")[kind]
        mfc = c if kind in ("dense", "eb", "eb_beyond") else "none"
        ax.errorbar([v], [y], xerr=err, fmt=mk, color=c, mfc=mfc, ms=7,
                    capsize=3, lw=1.4)
        if kind.endswith("beyond"):
            ax.annotate("", xy=(v - 2.2, y), xytext=(v, y),
                        arrowprops=dict(arrowstyle="->", color=c, lw=1.4))
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("mode-A null distance from free end (µm)")
    ax.set_xlim(-9, 32)
    ax.set_title("Every run, one axis — nulls cluster AT the end\n"
                 "(dense = filled blue; EB model = green diamonds; sparse = open gold; "
                 "arrows = at/beyond-end bound)", fontsize=10)
    ax.grid(axis="x", alpha=0.25)

    # ------- right: both modes' branch tails close AT the end (Demo6) -------
    import glob, io, contextlib, re
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dense_raw_analysis import load_tunes, raw_null
    xs, F, Z = load_tunes("/tmp/today/Demo6Fullgrid")
    rA = raw_null(xs, F, Z, (330e3, 470e3))
    rB = raw_null(xs, F, Z, (1090e3, 1240e3))
    sl = xs >= 205
    gA = (rA["fa"] - rA["f_res_Hz"]) / 1e3
    gB = (rB["fa"] - rB["f_res_Hz"]) / 1e3
    ax2.axhline(0, color="k", lw=1.0)
    ax2.plot(xs[sl], gA[sl], "o-", color=BLUE, ms=3, lw=1.1,
             label="mode A: fa − f_res")
    ax2.plot(xs[sl], gB[sl], "s-", color=GREEN, ms=3, lw=1.1,
             label="mode B: fa − f_res")
    ax2.axvline(225, color=RED, lw=1.2)
    ax2.axvspan(225, 232, color=RED, alpha=0.08)
    # mode A crossing marker
    cA = rA["classification"]
    if cA["status"] == "crossed":
        ax2.plot([cA["x_null_um"]], [0], "o", color=BLUE, ms=9, mfc="none", mew=2)
        ax2.annotate(f"mode A crosses at {cA['x_null_um']:.1f} µm\n"
                     f"({225-cA['x_null_um']:.1f} µm from end)",
                     xy=(cA["x_null_um"], 0), xytext=(206, 26), fontsize=9,
                     color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE))
    # mode B extrapolated bound
    cB = rB["classification"]
    if cB["status"] == "at_or_beyond_end":
        okB = np.isfinite(gB) & (xs >= 213)
        pf = np.polyfit(xs[okB], gB[okB], 1)
        xe = np.linspace(xs[okB][-1], cB["x_bound_um"], 20)
        ax2.plot(xe, np.polyval(pf, xe), "--", color=GREEN, lw=1.2)
        ax2.plot([cB["x_bound_um"]], [0], "s", color=GREEN, ms=9, mfc="none", mew=2)
        ax2.annotate(f"mode B never crosses in span —\n"
                     f"bound {cB['x_bound_um']:.1f} µm, "
                     f"{cB['x_bound_um']-225:.1f} µm BEYOND the end",
                     xy=(cB["x_bound_um"], 0), xytext=(211.5, 9), fontsize=9,
                     color=GREEN, arrowprops=dict(arrowstyle="->", color=GREEN))
    ax2.text(225.4, 55, "beyond\nfree end", fontsize=8, color=RED)
    ax2.set_xlim(205, 232)
    ax2.set_xlabel("position (µm)"); ax2.set_ylabel("fa − f_res (kHz)")
    ax2.legend(fontsize=8.5, frameon=False, loc="lower left")
    ax2.set_title("Demo6 dense, model-free: BOTH modes' branches close at the end\n"
                  "mode A crosses 1.5 µm inside; mode B extrapolates ~5 µm beyond",
                  fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "comparison_dns.png"), dpi=150,
                bbox_inches="tight")
    print("wrote comparison_dns.png")


if __name__ == "__main__":
    main()

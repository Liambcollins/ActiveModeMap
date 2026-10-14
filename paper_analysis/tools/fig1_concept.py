"""Fig. 1: concept of fast mode mapping and the dense ground truth (soft probe, five modes).

(a) schematic; (b) soft-probe dense map (the dark curved bands are the antiresonance branches); (c) stiff-probe map;
(d) a conventional single-position sweep; (e) fixed-frequency spatial profiles (on resonance and on an
antiresonance branch): the detected response depends strongly on where the laser sits; (f) time to map.

    python tools/fig1_concept.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fmmpaper as F  # noqa: E402
from fmmpaper import spectra, schematic, plotting as fp, style3 as S3  # noqa: E402

X0 = {"stiff": 28.1, "soft": -19.0}
TIMES = pd.DataFrame([dict(probe="soft probe", dense_min=410.6, n_dense=346, sparse_min=13.8, n_sparse=12),
                      dict(probe="stiff probe", dense_min=192.0, n_dense=161, sparse_min=9.6, n_sparse=8)])


def antiresonance_branches(x, f, Z, bands, depth_dB=8.0, smooth_bins=15, max_step_frac=0.03, min_len=25):
    """Antiresonance branch in each gap between consecutive CR bands, traced by continuity.

    Per position the candidate is the deepest minimum of the smoothed dB spectrum inside the gap that is
    at least depth_dB below the gap's median level; the branch is grown from the deepest point outward,
    accepting a candidate only if it moves by less than max_step_frac of the gap per position (this
    rejects fixed-frequency features such as weak torsional modes, which do not move with x). Segments
    shorter than min_len positions are dropped. Returns a list of (x, f_AR) arrays with NaN gaps."""
    A = 20 * np.log10(np.abs(Z) + 1e-15); A = uniform_filter1d(A, smooth_bins, axis=1)
    names = list(bands); out = []
    for b0, b1 in zip(names[:-1], names[1:]):
        lo, hi = bands[b0][1], bands[b1][0]; m = (f > lo) & (f < hi)
        if m.sum() < 10:
            continue
        seg = A[:, m]; fm = f[m]; gap = hi - lo
        # local minima per position (candidate list), with depth
        cands = []
        for i in range(len(x)):
            r = seg[i]; loc = np.where((r[1:-1] < r[:-2]) & (r[1:-1] < r[2:]))[0] + 1
            dep = np.median(r) - r[loc]; ok = dep >= depth_dB
            cands.append(list(zip(fm[loc][ok], dep[ok])))
        far = np.full(len(x), np.nan)
        # seed: the globally deepest candidate
        best = max(((d, i, fq) for i, c in enumerate(cands) for fq, d in c), default=None)
        if best is None:
            out.append((x, far)); continue
        _, i0, f0 = best; far[i0] = f0
        for direction in (1, -1):
            prev = f0; i = i0 + direction
            while 0 <= i < len(x):
                c = [fq for fq, d in cands[i] if abs(fq - prev) < max_step_frac * gap]
                if not c:
                    break
                prev = min(c, key=lambda q: abs(q - prev)); far[i] = prev; i += direction
        # drop fixed-frequency runs (a feature that does not move with x is not an antiresonance branch)
        good = ~np.isnan(far)
        if good.sum() >= min_len:
            df = np.abs(np.gradient(np.where(good, far, np.nan)))
            flat = uniform_filter1d(np.where(good, df, 0.0), 15) < 0.0005 * gap
            far[flat & good] = np.nan
        if np.sum(~np.isnan(far)) < min_len:
            far[:] = np.nan
        out.append((x, far))
    return out


def load(probe):
    key = {"stiff": "scmpitB_r2_dense", "soft": "ppp_dense_1um"}[probe]
    s = F.load(key); return dict(s=s, x=s.x_um - X0[probe], f=s.freq_Hz, Z=s.Z[0], bands=dict(s.probe.bands_Hz))


def plot(path, cmap=fp.MAP_CMAP, ink="w", formats=("png", "pdf"), fontsize=None):
    """ink: colour of band labels and the sweep-position line drawn on the maps (use "k" on light colormaps)."""
    S3.setup() if fontsize else fp.setup()
    if fontsize:
        plt.rcParams.update({k: fontsize for k in ("font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize")})
    S, T = load("soft"), load("stiff")
    W = fp.WIDTH_IN["double"]; fig = plt.figure(figsize=(W, W * (1.05 if fontsize else 0.92)))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.58 if fontsize else 0.42, 1.15, 0.85], hspace=0.45 if fontsize else 0.38)
    # (a) schematic
    axa = fig.add_subplot(gs[0]); schematic.draw(axa); fp.panel_label(axa, "a", dx=0.0, dy=0.95)
    # (b, c) maps
    g1 = gs[1].subgridspec(1, 3, width_ratios=[1.55, 1.0, 0.035], wspace=0.22)
    axb = fig.add_subplot(g1[0, 0]); axc = fig.add_subplot(g1[0, 1])
    im = fp.map_db(axb, S["x"], S["f"], S["Z"], fmin=40, fmax=1090, colorbar=False, cmap=cmap)
    def band_labels(ax, D, fmax, dy):
        xr = D["x"].max() - 0.015 * np.ptp(D["x"]); names = list(D["bands"])
        for b in names:
            fc = np.mean(D["bands"][b]) / 1e3; top = (b == names[-1]) and (fontsize or fc + 4 * dy > fmax)
            if fontsize:  # offset in points so the label clears the resonance line at any font size
                ax.annotate(b, (xr, fc), xytext=(0, -4 if top else 4), textcoords="offset points", color=ink, fontsize=5.5, ha="right", va="top" if top else "bottom")
            else:
                ax.text(xr, fc - dy if top else fc + dy, b, color=ink, fontsize=5.5, ha="right", va="top" if top else "bottom")
    band_labels(axb, S, 1090, 12)
    axb.set_title("soft probe · 346 positions, 1 µm pitch · 411 min", fontsize=6.5)
    axb.set_xlabel("distance from clamp x (µm)"); axb.set_ylabel("frequency (kHz)")
    fp.map_db(axc, T["x"], T["f"], T["Z"], fmin=240, fmax=1910, colorbar=False, cmap=cmap)
    band_labels(axc, T, 1910, 25)
    axc.set_title("stiff probe · 161 positions · 192 min", fontsize=6.5); axc.set_xlabel("distance from clamp x (µm)"); axc.set_ylabel("")
    if fontsize:
        S3.boxes(axb); S3.boxes(axc)
    cb = fig.colorbar(im, cax=fig.add_subplot(g1[0, 2])); cb.set_label("|Z| (dB re max)", fontsize=6)
    fp.panel_label(axb, "b", dx=-0.1); fp.panel_label(axc, "c", dx=-0.1)
    # (d) single sweep, (e) fixed-frequency profiles, (f) timing
    g2 = gs[2].subgridspec(1, 3, width_ratios=[1.1, 1.1, 0.8], wspace=0.4) if not fontsize else gs[2].subgridspec(1, 3, width_ratios=[0.95, 1.5, 0.8], wspace=0.42)
    axd, axe, axf = (fig.add_subplot(g2[0, i]) for i in range(3))
    ipos = int(np.argmin(np.abs(S["x"] - 300))); ref = np.abs(S["Z"]).max()
    axd.plot(S["f"] / 1e3, 20 * np.log10(np.abs(S["Z"][ipos]) / ref + 1e-9), color="k", lw=0.6)
    axd.set_xlim(40, 1090); axd.set_ylim(-75, 3); axd.set_xlabel("frequency (kHz)"); axd.set_ylabel("|Z| (dB re max)")
    axd.set_title(("" if fontsize else "one ") + f"conventional sweep, x = {S['x'][ipos]:.0f} µm", fontsize=6.5)
    axb.axvline(S["x"][ipos], color=ink, lw=0.5, ls="--")
    for b in S["bands"]:
        pk = spectra.peak(S["f"], S["Z"][ipos], S["bands"][b])
        last = b == list(S["bands"])[-1]
        axd.text(pk["f_Hz"] / 1e3 + (-15 if last else 15), 20 * np.log10(pk["amp"] / ref) - 2, b, fontsize=5, ha="right" if last else "left", va="top", color="0.3")
    # profiles: CR1, CR3, CR5 on resonance, plus one antiresonance branch frequency
    picks = [("CR1", None), ("CR3", None), ("CR5", None)]
    cols = {"CR1": S3.CR1, "CR3": S3.QS, "CR5": S3.MODEL} if fontsize else {"CR1": fp.C["cr1"], "CR3": fp.C["cr3"], "CR5": fp.C["cr5"]}
    for b, _ in picks:
        f0, v = spectra.on_resonance_profile(S["f"], S["Z"], S["bands"][b]); a = np.abs(v) / np.abs(v).max()
        axe.plot(S["x"], a, color=cols[b], lw=0.9, label=f"{b}, {f0/1e3:.0f} kHz")
    # antiresonance: a frequency in the CR3-CR4 gap where the branch sweeps across the lever
    far = antiresonance_branches(S["x"], S["f"], S["Z"], S["bands"])[2][1]      # gap CR3-CR4
    f_ar = float(np.nanmedian(far)); j = int(np.argmin(np.abs(S["f"] - f_ar))); a = np.abs(S["Z"][:, j]); a = a / a.max()
    axe.plot(S["x"], a, color="0.35", lw=0.9, ls="--", label=f"{f_ar/1e3:.0f} kHz, antiresonance")
    axe.set_xlim(S["x"].min(), S["x"].max()); axe.set_ylim(0, 1.25); axe.set_xlabel("distance from clamp x (µm)"); axe.set_ylabel("|Z| at fixed f (norm.)")
    if fontsize:  # legend inside, in headroom above the normalised profiles
        axe.set_ylim(0, 1.5); axe.set_yticks(np.arange(0, 1.01, 0.2))
        axe.legend(loc="upper center", ncol=2, handlelength=0.9, handletextpad=0.35, columnspacing=0.5, borderaxespad=0.2, frameon=False)
    else:
        axe.legend(fontsize=5.2, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.5, columnspacing=0.8, frameon=False)
    if fontsize:
        axd.set_xticks(np.arange(200, 1001, 200))
    yy = np.arange(len(TIMES))
    axf.barh(yy + 0.19, TIMES.dense_min, height=0.36, color=S3.LIGHT if fontsize else "0.6", label="dense reference")
    axf.barh(yy - 0.19, TIMES.sparse_min, height=0.36, color=S3.MAP if fontsize else fp.C["ebgp"], label="live sparse capture")
    for i, r in TIMES.iterrows():
        axf.text(r.sparse_min * 1.15, i - 0.19, f"N = {r.n_sparse}: ×{r.dense_min / r.sparse_min:.0f} faster", va="center", fontsize=5.5)
        axf.text(r.dense_min * 1.15, i + 0.19, f"N = {r.n_dense}", va="center", fontsize=5.5, color="0.35")
    axf.set_xscale("log"); axf.set_xlim(3, 3e4); axf.set_ylim(1.5, -0.9); axf.set_yticks(yy); axf.set_yticklabels(TIMES.probe, fontsize=6)
    if fontsize:
        axf.set_xticks([1e1, 1e2, 1e3, 1e4])
    axf.set_xlabel("acquisition time (min)"); axf.legend(loc="lower left" if fontsize else "lower center", bbox_to_anchor=(-0.02 if fontsize else 0.5, 1.0), fontsize=5.2, handlelength=1.2, frameon=False, ncol=1)
    for ax, lab in ((axd, "d"), (axe, "e"), (axf, "f")):
        fp.panel_label(ax, lab, dx=-0.18 if lab != "f" else -0.3)
    if fontsize:  # schematic: move "tip" right of the tip so it clears the distance-axis label
        for t in axa.texts:
            if t.get_text() == "tip":
                x, _ = t.get_position(); t.set_position((x + 0.85, -0.42)); t.set_ha("left")
            elif t.get_text().startswith("V$_{dc}$"):
                x, _ = t.get_position(); t.set_position((x, -1.05))
    if fontsize:  # raise every non-bold text (ticks, labels, legends, annotations, schematic) to one size
        for t in fig.findobj(matplotlib.text.Text):
            if t.get_text() and t.get_fontweight() != "bold":
                t.set_fontsize(fontsize)
    path = Path(path)
    for ext in formats:
        out = path.with_suffix("." + ext); fig.savefig(out, dpi=600); print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmap", default=fp.MAP_CMAP)
    ap.add_argument("--ink", default=None, help="label/line colour on the maps (default: w for magma, k otherwise)")
    ap.add_argument("--out", default=None, help="output path stem (default figures/Fig1_concept[_<cmap>])")
    ap.add_argument("--formats", default="png,pdf")
    ap.add_argument("--fontsize", type=float, default=None, help="set all text to this size (7 = Nature max)")
    a = ap.parse_args()
    ink = a.ink or ("w" if a.cmap == fp.MAP_CMAP else "k")
    out = a.out or F.config.FIG_DIR / ("Fig1_concept" + ("" if a.cmap == fp.MAP_CMAP else f"_{a.cmap}") + ".png")
    plot(out, cmap=a.cmap, ink=ink, formats=tuple(a.formats.split(",")), fontsize=a.fontsize)

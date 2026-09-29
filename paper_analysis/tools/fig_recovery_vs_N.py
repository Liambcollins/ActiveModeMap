"""Figure: sparse recovery vs N for EB, EB+GP and low-rank on the soft probe, all five modes.

(a) ground-truth dense map with the four equispaced designs;
(b) where each arm fails: per-position error in every CR band, for N = 3, 6, 9, 12;
(c) recovered spectra at one held-out position (CR1, CR3, CR5) for N = 3 and N = 12, with the
    EB+GP posterior band;
(d) recovered mode shapes at the same bands and N.

    python tools/fig_recovery_vs_N.py            # computes (cached in results/fig_recovery_vs_N.npz), plots
    python tools/fig_recovery_vs_N.py --replot   # plot only
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fmmpaper as F  # noqa: E402
from fmmpaper import ebfit, physrec, recon, spectra, plotting as fp  # noqa: E402

NS = [4, 6, 9, 12]
ARMS = ["EB", "EB+GP", "low-rank"]
ARM_C = {"EB": fp.C["eb"], "EB+GP": fp.C["ebgp"], "low-rank": fp.C["lowrank"]}
ARM_LS = {"EB": "-", "EB+GP": "-", "low-rank": "--"}
SHOW_N = [4, 12]
PROBES = {   # key, clamp in stage um (lever frame = stage - x0), calibration, bands shown in (c,d), map f-range (kHz), x* target, n positions label
    "soft": dict(key="ppp_dense_1um", x0=-19.0, cal="soft", show=["CR1", "CR3", "CR5"], frange=(40, 1090), xstar=270.0, label="346 positions, 1 µm pitch"),
    "stiff": dict(key="scmpitB_r2_dense", x0=28.1, cal="stiff", show=["CR1", "CR2", "CR3"], frange=(240, 1910), xstar=120.0, label="161 positions, 1 µm pitch"),
}
PROBE = "soft"
CACHE = lambda: F.config.RESULTS_DIR / f"fig_recovery_vs_N_{PROBE}.npz"


def compute(cal_tag="mass", rank=6):
    P = PROBES[PROBE]
    s = F.load(P["key"]); xs, f, Z = s.x_um, s.freq_Hz, s.Z[0]; x = xs - P["x0"]
    bands = dict(s.probe.bands_Hz); names = list(bands)
    cal = ebfit.load_calibration(P["cal"], cal_tag)
    masks = {b: (f >= bands[b][0]) & (f <= bands[b][1]) for b in names}
    span = np.zeros(f.size, bool)
    for m in masks.values(): span |= m
    out = dict(x=x, f=f, Z=Z, span=span, names=np.array(names), Zrec={}, sd={}, sel={}, theta={})
    interp = lambda ft, ff, M: (np.apply_along_axis(lambda r: np.interp(ft, ff, r), 1, M.real)
                                + 1j * np.apply_along_axis(lambda r: np.interp(ft, ff, r), 1, M.imag))
    for N in NS:
        sel = recon.select_equispaced(x, N); out["sel"][N] = np.array(sel)
        t0 = time.time()
        eb = np.full(Z.shape, np.nan + 0j); ebgp = eb.copy(); sd = np.full(Z.shape, np.nan)
        for b in names:
            zb, fb = physrec.band_slice(f, Z, bands[b], n_max=250)
            o = ebfit.rec_eb2(xs, sel, zb[sel], fb, cal, b)
            m = masks[b]
            eb[:, m] = interp(f[m], fb, o["Zeb"]); ebgp[:, m] = interp(f[m], fb, o["Zrec"])
            sd[:, m] = np.apply_along_axis(lambda r: np.interp(f[m], fb, r), 1, o["sd"])
            out["theta"][(N, b)] = o["theta"]
        lr = recon.rec_lowrank(x, sel, Z[sel], rank=min(rank, N))["Zrec"]
        out["Zrec"][N] = {"EB": eb, "EB+GP": ebgp, "low-rank": lr}; out["sd"][N] = sd
        h = recon.held_out(len(x), sel)
        print(f"N={N}: {time.time()-t0:.1f} s  " + "  ".join(f"{a} {recon.nrmse(out['Zrec'][N][a][h][:, span], Z[h][:, span]):.1f}%" for a in ARMS), flush=True)
    return out


def save(out):
    np.savez_compressed(CACHE(), x=out["x"], f=out["f"], Z=out["Z"], span=out["span"], names=out["names"],
                        **{f"rec_{N}_{a}": out["Zrec"][N][a] for N in NS for a in ARMS},
                        **{f"sd_{N}": out["sd"][N] for N in NS}, **{f"sel_{N}": out["sel"][N] for N in NS})


def load():
    d = np.load(CACHE(), allow_pickle=True)
    out = dict(x=d["x"], f=d["f"], Z=d["Z"], span=d["span"], names=list(d["names"]), Zrec={}, sd={}, sel={})
    for N in NS:
        out["Zrec"][N] = {a: d[f"rec_{N}_{a}"] for a in ARMS}; out["sd"][N] = d[f"sd_{N}"]; out["sel"][N] = d[f"sel_{N}"]
    return out


def pick_position(out):
    """A held-out position (for every N) near the target where the shown bands are all well above their nodes."""
    P = PROBES[PROBE]; x_target = P["xstar"]
    x, f, Z = out["x"], out["f"], out["Z"]
    used = set(np.concatenate([out["sel"][N] for N in NS]).tolist())
    bands = dict(F.load(P["key"]).probe.bands_Hz)
    ok = np.ones(len(x), bool)
    for b in P["show"]:
        m = (f >= bands[b][0]) & (f <= bands[b][1]); f0, prof = spectra.on_resonance_profile(f, Z, bands[b])
        a = np.abs(prof); ok &= a > 0.45 * a.max()
    cand = [i for i in np.argsort(np.abs(x - x_target)) if ok[i] and i not in used]
    return int(cand[0])


def plot(out, path, err_scale="log"):
    P = PROBES[PROBE]; SHOW_BANDS = P["show"]
    x, f, Z, span, names = out["x"], out["f"], out["Z"], out["span"], out["names"]
    bands = dict(F.load(P["key"]).probe.bands_Hz)
    masks = {b: (f >= bands[b][0]) & (f <= bands[b][1]) for b in names}
    ipos = pick_position(out); fp.setup()
    W = fp.WIDTH_IN["double"]; fig = plt.figure(figsize=(W, W * 1.18))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 0.06, 0.66, 0.66], hspace=0.42)
    # ---------------------------------------------------------------- (a) + (b)
    top = gs[0].subgridspec(1, 2, width_ratios=[0.9, 2.1], wspace=0.22)
    ga = top[0].subgridspec(2, 1, height_ratios=[0.16, 1], hspace=0.05)
    axd = fig.add_subplot(ga[0]); axa = fig.add_subplot(ga[1])
    fp.map_db(axa, x, f, Z, colorbar=False, fmin=P["frange"][0], fmax=P["frange"][1])
    axa.set_xlabel("distance from clamp x (µm)"); axa.set_ylabel("frequency (kHz)")
    for b in names:
        axa.text(x.max() - 2, np.mean(bands[b]) / 1e3, b, color="w", fontsize=5.5, ha="right", va="center")
    axa.axvline(x[ipos], color="w", lw=0.6, ls=":")
    for k, N in enumerate(NS):
        axd.plot(x[out["sel"][N]], np.full(N, k), "v", color="k", ms=3.2, mec="none")
        axd.text(x.min() - 4, k, f"N = {N}", ha="right", va="center", fontsize=5.5)
    axd.set_xlim(axa.get_xlim()); axd.set_ylim(-0.7, len(NS) - 0.3); axd.axis("off"); axd.invert_yaxis()
    axd.set_title(f"ground truth: {P['label']}", fontsize=6.5, pad=2)
    fp.panel_label(axd, "a", dx=-0.2, dy=1.15)
    # (b) error strips
    gb = top[1].subgridspec(len(NS), len(ARMS), hspace=0.18, wspace=0.08)
    from matplotlib.colors import Normalize
    norm = LogNorm(3, 100) if err_scale == "log" else Normalize(0, 60); ims = None
    for i, N in enumerate(NS):
        h = recon.held_out(len(x), out["sel"][N])
        for j, a in enumerate(ARMS):
            ax = fig.add_subplot(gb[i, j]); zr = out["Zrec"][N][a]
            # absolute error at each position, relative to the band's RMS amplitude over the whole lever
            # (a per-position denominator would blow up at the nodes, where |Z| sits on the noise floor)
            E = np.array([[100 * np.sqrt(np.mean(np.abs(zr[p][masks[b]] - Z[p][masks[b]]) ** 2) / np.mean(np.abs(Z[:, masks[b]]) ** 2))
                           for p in range(len(x))] for b in names])
            xe = np.concatenate([[x[0] - 0.5], 0.5 * (x[1:] + x[:-1]), [x[-1] + 0.5]])
            Eplot = np.clip(E, 3, 100) if err_scale == "log" else np.clip(E, 0, 60)
            ims = ax.pcolormesh(xe, np.arange(len(names) + 1), Eplot, cmap="Reds", norm=norm, shading="flat", rasterized=True)
            for xs in x[out["sel"][N]]: ax.axvline(xs, color="k", lw=0.35, alpha=0.5)
            tot = recon.nrmse(zr[h][:, span], Z[h][:, span])
            ax.text(0.98, 0.93, f"{tot:.0f} %", transform=ax.transAxes, ha="right", va="top", fontsize=6,
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
            ax.set_yticks(np.arange(len(names)) + 0.5); ax.set_yticklabels(names if j == 0 else [], fontsize=5)
            ax.tick_params(length=0); ax.set_xticks([])
            if i == 0: ax.set_title(a, color=ARM_C[a], fontsize=7, pad=3)
            if j == 0: ax.set_ylabel(f"N = {N}", fontsize=6.5, labelpad=14, rotation=0, va="center")
            if i == len(NS) - 1:
                ax.set_xticks([150, 250, 350, 450] if PROBE == "soft" else [50, 100, 150, 200]); ax.tick_params(axis="x", length=2, labelsize=5.5)
                if j == 1: ax.set_xlabel("distance from clamp x (µm)")
            if i == 0 and j == 0: fp.panel_label(ax, "b", dx=-0.3, dy=1.3)
    cax = fig.add_axes([0.915, 0.70, 0.012, 0.2])
    cb = fig.colorbar(ims, cax=cax); cb.set_label("error at position, % of band RMS", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    if err_scale == "log": cb.set_ticks([3, 10, 30, 100]); cb.set_ticklabels(["≤3", "10", "30", "≥100"])
    else: cb.set_ticks([0, 20, 40, 60]); cb.set_ticklabels(["0", "20", "40", "≥60"])
    # ---------------------------------------------------------------- (c) spectra
    gc = gs[2].subgridspec(len(SHOW_N), len(SHOW_BANDS), hspace=0.12, wspace=0.1)
    for i, N in enumerate(SHOW_N):
        for j, b in enumerate(SHOW_BANDS):
            ax = fig.add_subplot(gc[i, j]); m = masks[b]; fk = f[m] / 1e3
            ref = np.abs(Z[ipos][m]).max()
            db = lambda z: 20 * np.log10(np.abs(z) / ref + 1e-6)
            ax.plot(fk, db(Z[ipos][m]), color="k", lw=1.0, label="measured")
            for a in ARMS:
                zr = out["Zrec"][N][a][ipos][m]
                ax.plot(fk, db(zr), ARM_LS[a], color=ARM_C[a], lw=0.9, label=a)
            sd = out["sd"][N][ipos][m]; zg = np.abs(out["Zrec"][N]["EB+GP"][ipos][m])
            ax.fill_between(fk, db(np.clip(zg - 2 * sd, 1e-9, None)), db(zg + 2 * sd), color=ARM_C["EB+GP"], alpha=0.18, lw=0, label="EB+GP ±2σ")
            ax.set_ylim(-42, 4); ax.set_xlim(fk.min(), fk.max())
            if j == 0: ax.set_ylabel("|Z| (dB re peak)"); ax.text(-0.36, 0.5, f"N = {N}", transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=7, fontweight="bold")
            else: ax.set_yticklabels([])
            if i == 0: ax.set_title(b, fontsize=7, pad=2); ax.set_xticklabels([])
            else: ax.set_xlabel("frequency (kHz)")
            e = {a: recon.nrmse(out["Zrec"][N][a][ipos][m], Z[ipos][m]) for a in ARMS}
            ax.text(0.03, 0.05, "  ".join(f"{a} {e[a]:.0f}%" for a in ARMS), transform=ax.transAxes, fontsize=5, va="bottom")
            if i == 0 and j == 0: fp.panel_label(ax, "c", dx=-0.42, dy=1.2)
            if i == 0 and j == 1: ax.text(0.5, 1.24, f"recovered spectra at a held-out position, x = {x[ipos]:.0f} µm", transform=ax.transAxes, fontsize=6.5, ha="center")
    # ---------------------------------------------------------------- (d) mode shapes
    gd = gs[3].subgridspec(len(SHOW_N), len(SHOW_BANDS), hspace=0.12, wspace=0.1)
    for i, N in enumerate(SHOW_N):
        for j, b in enumerate(SHOW_BANDS):
            ax = fig.add_subplot(gd[i, j]); m = masks[b]
            f0, prof = spectra.on_resonance_profile(f, Z, bands[b]); jf = int(np.argmin(np.abs(f - f0)))
            a0 = np.abs(Z[:, jf]); nrm = a0.max()
            ax.plot(x, a0 / nrm, color="k", lw=1.0)
            for a in ARMS:
                ax.plot(x, np.abs(out["Zrec"][N][a][:, jf]) / nrm, ARM_LS[a], color=ARM_C[a], lw=0.9)
            sd = out["sd"][N][:, jf]; zg = np.abs(out["Zrec"][N]["EB+GP"][:, jf])
            ax.fill_between(x, np.clip(zg - 2 * sd, 0, None) / nrm, (zg + 2 * sd) / nrm, color=ARM_C["EB+GP"], alpha=0.18, lw=0)
            ax.plot(x[out["sel"][N]], a0[out["sel"][N]] / nrm, "v", color="k", ms=3.2, mec="none")
            ax.set_ylim(0, 1.45); ax.set_xlim(x.min(), x.max()); ax.set_yticks([0, 0.5, 1])
            if j == 0: ax.set_ylabel("|Z| (norm.)"); ax.text(-0.36, 0.5, f"N = {N}", transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=7, fontweight="bold")
            else: ax.set_yticklabels([])
            if i == 0: ax.set_title(f"{b}, {f0/1e3:.0f} kHz", fontsize=7, pad=2); ax.set_xticklabels([])
            else: ax.set_xlabel("distance from clamp x (µm)")
            if i == 0 and j == 0: fp.panel_label(ax, "d", dx=-0.42, dy=1.2)
            if i == 0 and j == 1: ax.text(0.5, 1.24, "recovered mode shapes at the band resonance", transform=ax.transAxes, fontsize=6.5, ha="center")
    handles = [Line2D([], [], color="k", lw=1, label="measured (dense)")] + \
              [Line2D([], [], color=ARM_C[a], ls=ARM_LS[a], lw=1, label=a) for a in ARMS] + \
              [Patch(fc=ARM_C["EB+GP"], alpha=0.18, label="EB+GP ±2σ"), Line2D([], [], marker="v", color="k", ls="", ms=3.5, label="measured positions")]
    axl = fig.add_subplot(gs[1]); axl.axis("off")
    axl.legend(handles=handles, loc="center", ncol=6, frameon=False, fontsize=6, handlelength=2.2, columnspacing=1.6)
    fig.savefig(path, dpi=600); fig.savefig(str(path).replace(".png", ".pdf")); plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--replot", action="store_true"); ap.add_argument("--probe", default="soft", choices=PROBES)
    ap.add_argument("--err_scale", default="log", choices=["log", "linear"])
    a = ap.parse_args(); PROBE = a.probe
    if a.replot and CACHE().exists():
        out = load()
    else:
        out = compute(); save(out)
    plot(out, F.config.FIG_DIR / f"fig_recovery_vs_N_{PROBE}{'' if a.err_scale == 'log' else '_linear'}.png", err_scale=a.err_scale)

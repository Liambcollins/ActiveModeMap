"""Supplementary Fig. 1 (2026-10-14): what the lever-geometry prior does to the sparse reconstructions.

Held-out complex error vs N for the beam model and the hybrid with (i) the SEM geometry as the only prior (clamp position
free, k_cone refitted; fmmpaper.semprior) and (ii) an effective geometry calibrated from thirty positions of the dense map
(the earlier choice), on both dense-mapped probes; polynomial arm for reference. Per-band error at N = 12 for both priors.
    python tools/figS1_geometry_prior.py            # compute (cached results/figS1_geometry_prior.npz) + plot
    python tools/figS1_geometry_prior.py --replot
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fmmpaper as F  # noqa: E402
from fmmpaper import ebfit, physrec, recon, semprior, style3 as S  # noqa: E402

NS = [3, 4, 5, 6, 8, 10, 12, 16, 20]
PROBES = {"soft": dict(key="ppp_dense_1um", x0=-19.0, cal="soft", name="PPP-CONTAu"), "stiff": dict(key="scmpitB_r2_dense", x0=28.1, cal="stiff", name="SCM-PIT")}
CACHE = F.config.RESULTS_DIR / "figS1_geometry_prior.npz"
interp = lambda ft, ff, M: (np.apply_along_axis(lambda r: np.interp(ft, ff, r), 1, M.real) + 1j * np.apply_along_axis(lambda r: np.interp(ft, ff, r), 1, M.imag))


def compute():
    out = {}
    for pk, P in PROBES.items():
        s = F.load(P["key"]); xs, f, Z = s.x_um, s.freq_Hz, s.Z[0]; x = xs - P["x0"]
        bands = dict(s.probe.bands_Hz); names = list(bands)
        masks = {b: (f >= bands[b][0]) & (f <= bands[b][1]) for b in names}; span = np.zeros(f.size, bool)
        for m in masks.values(): span |= m
        cal_d, free_d = ebfit.load_calibration(P["cal"], "mass"), ebfit.FREE_DEFAULT
        cal_s, free_s = semprior.calibration(P["cal"], P["x0"])
        out[f"{pk}|names"] = np.array(names)
        for N in NS:
            sel = recon.select_equispaced(x, N); h = recon.held_out(len(x), sel); t0 = time.time()
            lr = recon.rec_lowrank(x, sel, Z[sel], rank=min(6, N))["Zrec"]
            out[f"{pk}|poly|{N}"] = recon.nrmse(lr[h][:, span], Z[h][:, span])
            for prior, cal, free in (("dense", cal_d, free_d), ("sem", cal_s, free_s)):
                eb = np.full(Z.shape, np.nan + 0j); gp = eb.copy()
                for b in names:
                    zb, fb = physrec.band_slice(f, Z, bands[b], n_max=250)
                    o = ebfit.rec_eb2(xs, sel, zb[sel], fb, cal, b, free=free); m = masks[b]
                    eb[:, m] = interp(f[m], fb, o["Zeb"]); gp[:, m] = interp(f[m], fb, o["Zrec"])
                    out[f"{pk}|{prior}|EB|{N}|{b}"] = recon.nrmse(eb[h][:, m], Z[h][:, m]); out[f"{pk}|{prior}|GP|{N}|{b}"] = recon.nrmse(gp[h][:, m], Z[h][:, m])
                out[f"{pk}|{prior}|EB|{N}"] = recon.nrmse(eb[h][:, span], Z[h][:, span]); out[f"{pk}|{prior}|GP|{N}"] = recon.nrmse(gp[h][:, span], Z[h][:, span])
            print(f"{pk} N={N:2d}  dense EB {out[f'{pk}|dense|EB|{N}']:.1f} GP {out[f'{pk}|dense|GP|{N}']:.1f} | SEM EB {out[f'{pk}|sem|EB|{N}']:.1f} GP {out[f'{pk}|sem|GP|{N}']:.1f} | poly {out[f'{pk}|poly|{N}']:.1f}  ({time.time()-t0:.0f}s)", flush=True)
    np.savez(CACHE, **{k: np.asarray(v) for k, v in out.items()})
    return out


def plot(out):
    S.setup(); fig = S.figure(70)
    axs = [fig.add_axes([0.07, 0.17, 0.26, 0.68]), fig.add_axes([0.40, 0.17, 0.26, 0.68]), fig.add_axes([0.73, 0.17, 0.25, 0.68])]
    STY = {("sem", "EB"): dict(color=S.MODEL, ls="-", label="beam model, SEM geometry"), ("dense", "EB"): dict(color=S.MODEL, ls="--", label="beam model, dense-map geometry"),
           ("sem", "GP"): dict(color=S.MAP, ls="-", label="hybrid, SEM geometry"), ("dense", "GP"): dict(color=S.MAP, ls="--", label="hybrid, dense-map geometry")}
    for ax, pk in zip(axs[:2], ("soft", "stiff")):
        for (prior, arm), st in STY.items():
            ax.plot(NS, [float(out[f"{pk}|{prior}|{arm}|{N}"]) for N in NS], marker="o", ms=2.4, mew=0, lw=0.9, **st)
        ax.plot(NS, [float(out[f"{pk}|poly|{N}"]) for N in NS], marker="o", ms=2.4, mew=0, lw=0.9, color=S.POLY, ls=":", label="polynomial")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(2.7, 22); ax.set_ylim(7, 120)
        ax.set_xticks([3, 6, 12, 20]); ax.set_xticklabels(["3", "6", "12", "20"]); ax.set_yticks([10, 20, 50, 100]); ax.set_yticklabels(["10", "20", "50", "100"]); ax.minorticks_off()
        ax.set_xlabel("number of sampled positions N"); ax.set_title(PROBES[pk]["name"] + ", held-out positions of the dense map", color=S.PROBE[PROBES[pk]["name"]], pad=4)
    axs[0].set_ylabel("complex error (%)"); S.label(axs[0], "a", dx=-0.22); S.label(axs[1], "b", dx=-0.2)
    # (c) per band at N = 12, both probes
    ax = axs[2]; N = 12; xpos = 0; ticks = []; labels = []
    for pk in ("soft", "stiff"):
        names = list(out[f"{pk}|names"])
        for b in names:
            for k, (prior, arm) in enumerate((("dense", "EB"), ("sem", "EB"), ("dense", "GP"), ("sem", "GP"))):
                v = float(out[f"{pk}|{prior}|{arm}|{N}|{b}"]); st = STY[(prior, arm)]
                ax.bar(xpos + k * 0.2 - 0.3, v, 0.18, color=st["color"], alpha=1.0 if prior == "sem" else 0.45, lw=0)
            ticks.append(xpos); labels.append(f"{b}\n{PROBES[pk]['name'].split('-')[0]}"); xpos += 1
        xpos += 0.6
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=6); ax.set_ylabel("complex error at N = 12 (%)"); ax.set_ylim(0, 105)
    ax.text(0.98, 0.95, "pale: dense-map geometry\nfull: SEM geometry", transform=ax.transAxes, ha="right", va="top", fontsize=6.3, color=S.GREY)
    S.label(ax, "c", dx=-0.2)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ms=2.4, mew=0, lw=0.9, **st) for st in STY.values()] + [Line2D([], [], marker="o", ms=2.4, mew=0, lw=0.9, color=S.POLY, ls=":", label="polynomial")],
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01), fontsize=6.3, columnspacing=1.2, handlelength=3.0)
    S.save(fig, "FigS1_geometry_prior")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--replot", action="store_true"); a = ap.parse_args()
    out = dict(np.load(CACHE, allow_pickle=True)) if (a.replot and CACHE.exists()) else compute()
    plot(out)

"""Fig. 3: how the reconstruction methods compare, and why.

(a) error vs N on one band (soft CR1) and (b) on all five bands (matched span): EB, EB+GP, GP,
    low-rank, with the three floors (EB model floor; low-rank basis floor = fit on every position;
    SVD rank-3 floor); (c) stiff probe, all bands, including a data-only library basis learned
    from an earlier dense map of the same lever (R1 -> R2); (d) sampling strategies at fixed N:
    equispaced, random (50 draws), node-avoiding (physics-aware, from the calibrated model only);
    (e) the contact state the physics arm infers is flat in N (k*/k, zeta); (f) what transfers to a
    separate capture: the stiff live test.

    python tools/fig3_methods.py            # compute (cached in results/fig3_methods.json) + plot
    python tools/fig3_methods.py --replot
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fmmpaper as F  # noqa: E402
from fmmpaper import ebfit, physrec, recon, plotting as fp  # noqa: E402
from fmmpaper import jointeb as J  # noqa: E402

NS = [3, 4, 5, 6, 8, 10, 12, 15, 20, 30]
N_RANDOM = 50
ARMS = ["EB", "EB+GP", "GP", "low-rank"]
C = {"EB": fp.C["eb"], "EB+GP": fp.C["ebgp"], "GP": fp.C["gp"], "low-rank": fp.C["lowrank"], "library": "#7570b3"}
LS = {"EB": "-", "EB+GP": "-", "GP": ":", "low-rank": "--", "library": "-."}
CACHE = F.config.RESULTS_DIR / "fig3_methods.json"
PROBES = {"soft": dict(key="ppp_dense_1um", x0=-19.0, cal="soft"), "stiff": dict(key="scmpitB_r2_dense", x0=28.1, cal="stiff")}


def band_data(s):
    f, Z = s.freq_Hz, s.Z[0]; out = {}
    for b, rng in s.probe.bands_Hz.items():
        out[b] = physrec.band_slice(f, Z, rng, n_max=250)
    return out


def arms_on(x_stage, sel, bd, cal, bands, h):
    """Held-out NRMSE per arm per band and matched-span, plus EB theta."""
    res = {a: {} for a in ARMS}; theta = {}; num = {a: 0.0 for a in ARMS}; den = 0.0
    for b in bands:
        zb, fb = bd[b]
        o = ebfit.rec_eb2(x_stage, sel, zb[sel], fb, cal, b)
        maps = {"EB": o["Zeb"], "EB+GP": o["Zrec"], "GP": recon.rec_gp(x_stage, sel, zb[sel])["Zrec"],
                "low-rank": recon.rec_lowrank(x_stage, sel, zb[sel], rank=min(6, len(sel)))["Zrec"]}
        theta[b] = dict(k_ratio=o["theta"]["k_ratio"], zeta=o["theta"]["zeta"], eps=o["theta"]["eps"])
        e = np.sum(np.abs(zb[h]) ** 2); den += e
        for a, M in maps.items():
            res[a][b] = recon.nrmse(M[h], zb[h]); num[a] += np.sum(np.abs(M[h] - zb[h]) ** 2)
    for a in ARMS:
        res[a]["all"] = 100 * np.sqrt(num[a] / den)
    return res, theta


def node_avoiding_design(x_stage, N, cal, bands_Hz, s):
    """Physics-aware design from the calibrated model only: among equispaced designs shifted by an offset,
    take the one whose positions have the largest minimum predicted on-resonance amplitude over all bands
    (i.e. the design that stays away from every predicted node)."""
    P = dict(cal["P"]); je = J.JointEB(x_stage, [], cal["f0_hz"], n_modes=cal["n_modes"])
    fe = je.eig_freqs_Hz(P, nmax=8); names = list(bands_Hz)
    fres = [fe[np.argmin(np.abs(fe - np.mean(bands_Hz[b])))] for b in names]
    bands = [J.Band(b, np.array([fr]), None) for b, fr in zip(names, fres)]
    je.bands = bands
    zetas = [float(z) for z in cal["zeta"]]
    maps = je.band_maps(P, zetas, x=x_stage)
    amp = np.array([np.abs(zp[:, 0]) / np.abs(zp[:, 0]).max() for zp, _ in maps])   # (nb, nx)
    best = None; n = len(x_stage)
    for off in np.linspace(0, 1, 21)[:-1]:
        idx = np.round((np.arange(N) + off) * (n - 1) / (N - 1 + off * 2)).astype(int) if N > 1 else [n // 2]
        idx = np.clip(np.round(np.linspace(0, n - 1, N + 1)[:-1] + off * (n - 1) / N), 0, n - 1).astype(int)
        score = amp[:, idx].min()
        if best is None or score > best[0]:
            best = (score, sorted(set(idx.tolist())))
    return best[1]


def compute():
    out = {}
    for probe, P in PROBES.items():
        s = F.load(P["key"]); x = s.x_um; bd = band_data(s); bands = list(s.probe.bands_Hz); cal = ebfit.load_calibration(P["cal"], "mass")
        rec = dict(bands=bands, equi={}, random={}, nodeavoid={}, floors={})
        t0 = time.time()
        for N in NS:
            sel = recon.select_equispaced(x, N); h = recon.held_out(len(x), sel)
            r, th = arms_on(x, sel, bd, cal, bands, h); rec["equi"][N] = dict(res=r, theta=th, sel=list(map(int, sel)))
            print(f"{probe} equi N={N}: " + " ".join(f"{a} {r[a]['all']:.1f}" for a in ARMS), flush=True)
        for N in (4, 6, 9, 12):
            sel = node_avoiding_design(x, N, cal, s.probe.bands_Hz, s); h = recon.held_out(len(x), sel)
            r, th = arms_on(x, sel, bd, cal, bands, h); rec["nodeavoid"][N] = dict(res=r, sel=list(map(int, sel)))
            print(f"{probe} node-avoiding N={N}: " + " ".join(f"{a} {r[a]['all']:.1f}" for a in ARMS), flush=True)
            rng = np.random.default_rng(0); draws = []
            for k in range(N_RANDOM):
                sel = sorted(rng.choice(len(x), N, replace=False).tolist()); h = recon.held_out(len(x), sel)
                r, _ = arms_on(x, sel, bd, cal, bands, h); draws.append({a: r[a]["all"] for a in ARMS})
            rec["random"][N] = draws
            print(f"{probe} random N={N} ({time.time()-t0:.0f} s): medians " + " ".join(f"{a} {np.median([d[a] for d in draws]):.1f}" for a in ARMS), flush=True)
        # floors per band and matched-span
        fl = {"lowrank_r6": {}, "lowrank_r15": {}, "svd_r3": {}}
        num = {k: 0.0 for k in fl}; den = 0.0
        for b in bands:
            zb, fb = bd[b]; den += np.sum(np.abs(zb) ** 2)
            for k, r_ in (("lowrank_r6", 6), ("lowrank_r15", 15)):
                M = recon.rec_lowrank(x, np.arange(len(x)), zb, rank=r_)["Zrec"]; fl[k][b] = recon.nrmse(M, zb); num[k] += np.sum(np.abs(M - zb) ** 2)
            U, S, Vh = np.linalg.svd(zb, full_matrices=False); M = (U[:, :3] * S[:3]) @ Vh[:3]
            fl["svd_r3"][b] = recon.nrmse(M, zb); num["svd_r3"] += np.sum(np.abs(M - zb) ** 2)
        for k in fl:
            fl[k]["all"] = 100 * np.sqrt(num[k] / den)
        rec["floors"] = fl
        out[probe] = rec
    # stiff: library basis from the R1 dense map (earlier contact state) on R2 equispaced designs
    s2 = F.load("scmpitB_r2_dense"); s1 = F.load("scmpitB_r1_dense"); x2 = s2.x_um - 28.1; x1 = s1.x_um - 12.3
    bd2 = band_data(s2); lib = {}
    for N in NS:
        sel = recon.select_equispaced(s2.x_um, N); h = recon.held_out(len(x2), sel); num = 0.0; den = 0.0; per = {}
        for b, rng in s2.probe.bands_Hz.items():
            zb, fb = bd2[b]; z1, f1 = physrec.band_slice(s1.freq_Hz, s1.Z[0], rng, n_max=250)
            U, S, Vh = np.linalg.svd(z1, full_matrices=False); r = 3
            Ux = np.array([np.interp(x2, x1, U[:, k].real) + 1j * np.interp(x2, x1, U[:, k].imag) for k in range(r)]).T
            Cc, *_ = np.linalg.lstsq(Ux[sel], zb[sel], rcond=None); M = Ux @ Cc
            per[b] = recon.nrmse(M[h], zb[h]); num += np.sum(np.abs(M[h] - zb[h]) ** 2); den += np.sum(np.abs(zb[h]) ** 2)
        per["all"] = 100 * np.sqrt(num / den); lib[N] = per
        print(f"stiff library(R1->R2) N={N}: {per['all']:.1f}", flush=True)
    out["stiff"]["library_r1"] = lib
    out["transfer_live_stiff"] = {"EB": 15.1, "EB+GP": 18.9, "low-rank r=5": 19.1, "library, same contact": 12.8, "library, earlier contact": 17.1}
    json.dump(out, open(CACHE, "w"), indent=1, default=float)
    return out


def plot(out, path):
    fp.setup(); W = fp.WIDTH_IN["double"]
    fig, axs = plt.subplots(2, 3, figsize=(W, W * 0.62)); axs = axs.ravel()
    soft, stiff = out["soft"], out["stiff"]

    def curves(ax, rec, key, title, floors=True, library=None):
        for a in ARMS:
            y = [rec["equi"][str(N)]["res"][a][key] for N in NS]
            ax.plot(NS, y, LS[a], color=C[a], lw=1.1, marker="o", ms=2.6, mec="none", label=a)
        if library:
            ax.plot(NS, [library[str(N)][key] for N in NS], LS["library"], color=C["library"], lw=1.1, marker="s", ms=2.4, mec="none", label="library basis (earlier map)")
        if floors:
            fl = rec["floors"]
            ax.axhline(fl["lowrank_r15"][key], color=C["low-rank"], lw=0.7, ls=(0, (1, 1))); ax.text(34, fl["lowrank_r15"][key], "low-rank\nbasis floor", color=C["low-rank"], fontsize=4.8, va="center")
            ax.axhline(fl["svd_r3"][key], color="0.4", lw=0.7, ls=(0, (1, 1))); ax.text(34, fl["svd_r3"][key], "rank-3\nSVD floor", color="0.4", fontsize=4.8, va="center")
            eb_floor = np.median([rec["equi"][str(N)]["res"]["EB"][key] for N in NS[-4:]])
            ax.axhline(eb_floor, color=C["EB"], lw=0.7, ls=(0, (1, 1))); ax.text(34, eb_floor, "EB model\nfloor", color=C["EB"], fontsize=4.8, va="center")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([3, 5, 10, 20, 30]); ax.set_xticklabels(["3", "5", "10", "20", "30"])
        ax.set_ylim(1, 150); ax.set_xlim(2.7, 60); ax.set_xlabel("number of measured positions N"); ax.set_ylabel("held-out NRMSE (%)")
        ax.set_title(title, fontsize=6.5)
    curves(axs[0], soft, "CR1", "soft probe, one band (CR1)")
    curves(axs[1], soft, "all", "soft probe, five bands")
    curves(axs[2], stiff, "all", "stiff probe, three bands", library=stiff["library_r1"])
    axs[0].legend(fontsize=5.2, loc="lower left", handlelength=2); axs[2].legend(fontsize=5.2, loc="lower left", handlelength=2)
    # (d) sampling strategies, soft, all bands
    ax = axs[3]; Nd = [4, 6, 12]; w = 0.26
    for j, a in enumerate(("EB+GP", "low-rank", "GP")):
        xs = np.arange(len(Nd)) + (j - 1) * w
        eq = [soft["equi"][str(N)]["res"][a]["all"] for N in Nd]; na = [soft["nodeavoid"][str(N)]["res"][a]["all"] for N in Nd]
        rnd = [[d[a] for d in soft["random"][str(N)]] for N in Nd]
        ax.boxplot(rnd, positions=xs, widths=w * 0.8, showfliers=False, patch_artist=True,
                   boxprops=dict(fc=C[a], alpha=0.25, ec=C[a], lw=0.6), medianprops=dict(color=C[a], lw=0.9), whiskerprops=dict(color=C[a], lw=0.6), capprops=dict(color=C[a], lw=0.6))
        ax.plot(xs, eq, "o", color=C[a], ms=3.2, mec="k", mew=0.4, label=f"{a}: equispaced" if j == 0 else None)
        ax.plot(xs, na, "^", color=C[a], ms=3.4, mec="k", mew=0.4, label=f"{a}: node-avoiding" if j == 0 else None)
    ax.set_yscale("log"); ax.set_ylim(5, 400); ax.set_xticks(range(len(Nd))); ax.set_xticklabels([f"N = {N}" for N in Nd]); ax.set_ylabel("held-out NRMSE, five bands (%)")
    ax.set_title("sampling strategy, soft (box: 50 random)", fontsize=6.5)
    h = [Line2D([], [], marker="o", color="0.5", ls="", ms=3.2, mec="k", mew=0.4, label="equispaced"), Line2D([], [], marker="^", color="0.5", ls="", ms=3.4, mec="k", mew=0.4, label="node-avoiding (from the model)")] + \
        [Line2D([], [], color=C[a], lw=2, alpha=0.5, label=a) for a in ("EB+GP", "low-rank", "GP")]
    ax.legend(handles=h, fontsize=5, loc="upper right", ncol=1, handlelength=1.4, framealpha=0.9, frameon=True, edgecolor="none")
    # (e) parameters vs N
    ax = axs[4]; bands = soft["bands"]
    for b in bands:
        k = [soft["equi"][str(N)]["theta"][b]["k_ratio"] for N in NS]
        ax.plot(NS, k, "-", color=fp.C[b.lower()], lw=0.9, marker="o", ms=2.2, mec="none", label=b)
    for b in stiff["bands"]:
        k = [stiff["equi"][str(N)]["theta"][b]["k_ratio"] for N in NS]
        ax.plot(NS, k, "--", color=fp.C[b.lower()], lw=0.9, marker="s", ms=2.2, mec="none")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([3, 5, 10, 20, 30]); ax.set_xticklabels(["3", "5", "10", "20", "30"])
    ax.set_xlabel("number of measured positions N"); ax.set_ylabel("fitted contact stiffness k*/k")
    ax.set_ylim(300, 6000); ax.legend(fontsize=5, ncol=3, loc="lower right", handlelength=1.4, title="solid: soft   dashed: stiff", title_fontsize=5)
    ax.set_title("inferred contact stiffness vs N", fontsize=6.5)
    # (f) transfer to a separate capture
    ax = axs[5]; T = out["transfer_live_stiff"]; labs = list(T); vals = [T[k] for k in labs]
    cols = [C["EB"], C["EB+GP"], C["low-rank"], C["library"], C["library"]]
    ax.barh(range(len(labs)), vals, color=cols, height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.4, i, f"{v:.1f} %", va="center", fontsize=5.5)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=5.5); ax.invert_yaxis(); ax.set_xlim(0, 26)
    ax.set_xlabel("NRMSE at 7 never-visited positions (%)"); ax.set_title("stiff live capture → separate capture", fontsize=6.5)
    for ax, lab in zip(axs, "abcdef"):
        fp.panel_label(ax, lab, dx=-0.2 if lab != "f" else -0.55)
    fig.tight_layout(w_pad=1.5, h_pad=1.8)
    fig.savefig(path, dpi=600); fig.savefig(str(path).replace(".png", ".pdf")); plt.close(fig); print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--replot", action="store_true"); a = ap.parse_args()
    out = json.load(open(CACHE)) if a.replot and CACHE.exists() else json.loads(json.dumps(compute(), default=float))
    plot(out, F.config.FIG_DIR / "Fig3_methods.png")

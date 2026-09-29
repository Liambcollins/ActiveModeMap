"""2x2 panel per (N, arm): reconstructed map, NRMSE along the lever, transfer function at one
position, on-resonance profile. Stiff probe (R2 dense), one band, equispaced designs.

    python tools/grid_arms_vs_N.py --band CR3 --N 3 5 7 9 13 15 19
"""
import argparse, json, hashlib, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fmmpaper as F
from fmmpaper import spectra, recon, plotting as fp, physrec, ebfit
fp.setup()

X0 = {"stiff": 28.1, "soft": -19.0}
KEYS = {"stiff": ("scmpitB_r2_dense", "scmpitB"), "soft": ("ppp_dense_1um", "pppcontau")}
ARMS = ["EB", "GP", "EB+GP"]
STYLE = {"GP": (fp.C["gp"], ":"), "EB": (fp.C["eb"], "-"), "EB+GP": (fp.C["ebgp"], "-")}
CACHE = F.config.RESULTS_DIR / "_cache_explorer"; CACHE.mkdir(exist_ok=True)


def eb_v2(probe, band, sel, x_stage, zb, fb, cal_tag):
    cal = ebfit.load_calibration(probe, cal_tag)
    t0 = time.time(); out = ebfit.rec_eb2(x_stage, sel, zb[sel], fb, cal, band)
    th = out["theta"]
    print(f"  EB v2 fit N={len(sel)}: {time.time()-t0:.1f} s, k*/k {th['k_ratio']:.0f}, zeta {th['zeta']:.4f}, chi2 {th['red_chi2']:.1f}", flush=True)
    return out


def eb_cached(probe, band, sel, x, zb, fb, geom):
    key = hashlib.md5(f"{probe}|{band}|{list(map(int, sel))}".encode()).hexdigest()[:12]
    p = CACHE / f"{probe}_{band}_{key}.npz"
    if p.exists():
        d = np.load(p, allow_pickle=True)
        return dict(Zrec=d["Zrec"], Zeb=d["Zeb"], theta=json.loads(str(d["theta"])))
    t0 = time.time(); out = physrec.rec_eb(x, sel, zb[sel], fb, geom)
    np.savez_compressed(p, Zrec=out["Zrec"], Zeb=out["Zeb"], theta=json.dumps(out["theta"]))
    print(f"  EB fit N={len(sel)}: {time.time()-t0:.0f} s, setback {out['theta']['tip_setback_um']:.2f} um, "
          f"chi2 {out['theta']['red_chi2']:.2f}", flush=True)
    return out


def panel(x, f, Z, sel, h, zr, arm, N, band, ipos, theta, out_png):
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    ref = np.abs(Z).max(); nr = recon.nrmse(zr[h], Z[h])
    # (a) reconstructed map
    ax = axs[0, 0]; fp.map_db(ax, x, f, zr, ref=ref, colorbar=False); fp.mark_positions(ax, x[sel])
    ax.set_title(f"{arm} map, N = {N}, held-out NRMSE {nr:.1f} %"); ax.set_xlabel("distance from clamp (µm)")
    # (b) NRMSE along the lever
    ax = axs[0, 1]; c, ls = STYLE[arm]
    err = [recon.nrmse(zr[i], Z[i]) for i in range(len(x))]
    ax.plot(x, err, ls, color=c, lw=1.2)
    for xs in x[sel]: ax.axvline(xs, color="0.85", lw=0.6, zorder=0)
    ax.axvline(x[ipos], color="k", lw=0.8, ls="--")
    ax.set_yscale("log"); ax.set_ylim(1, 300); ax.set_xlabel("distance from clamp (µm)"); ax.set_ylabel("NRMSE (%)")
    ax.set_title("error along the lever (grey: measured positions)")
    # (c) transfer function at ipos
    axa = axs[1, 0]; axp = axa.twinx()
    axa.plot(f / 1e3, 20 * np.log10(np.abs(Z[ipos]) + 1e-12), color="k", lw=1, label="measured")
    axa.plot(f / 1e3, 20 * np.log10(np.abs(zr[ipos]) + 1e-12), ls, color=c, lw=1.2, label=arm)
    axp.plot(f / 1e3, np.degrees(np.angle(Z[ipos])), color="k", lw=0.5, alpha=0.35)
    axp.plot(f / 1e3, np.degrees(np.angle(zr[ipos])), ls, color=c, lw=0.6, alpha=0.6)
    axp.set_ylim(-200, 200); axp.set_ylabel("phase (deg, faint)")
    axa.set_xlabel("frequency (kHz)"); axa.set_ylabel("|Z| (dB V)"); axa.legend(fontsize=8, loc="lower right")
    tag = "measured" if ipos in sel else "held out"
    axa.set_title(f"transfer function at x = {x[ipos]:.0f} µm ({tag}), NRMSE {recon.nrmse(zr[ipos], Z[ipos]):.1f} %")
    # (d) on-resonance profile
    ax = axs[1, 1]
    f0, _ = spectra.on_resonance_profile(f, Z, (f.min(), f.max())); j = int(np.argmin(np.abs(f - f0)))
    a0 = np.abs(Z[:, j]); nrm = a0.max(); a = np.abs(zr[:, j])
    ax.plot(x, a0 / nrm, color="k", lw=1.4, label="dense"); ax.plot(x, a / nrm, ls, color=c, lw=1.2, label=arm)
    ax.plot(x[sel], a0[sel] / nrm, "v", color="k", ms=7, label="measured positions")
    inb = lambda n: x.min() + 5 <= n <= x.max() - 5
    nd0 = [n for n in spectra.nodes_from_profile(x, a0 / nrm) if inb(n)]
    nd = [n for n in spectra.nodes_from_profile(x, a / nrm) if inb(n)]
    for n in nd0: ax.axvline(n, color="0.8", lw=0.8, ls=":")
    for n in nd: ax.plot(n, 0.02, "v", color=c, ms=6)
    nerr = [min(abs(n - m) for m in nd0) for n in nd] if nd0 and nd else []
    ax.set_ylim(0, 1.6); ax.set_xlabel("distance from clamp (µm)"); ax.set_ylabel(f"|Z| at {f0/1e3:.0f} kHz (norm.)")
    ax.set_title(f"on-resonance profile, {band}"); ax.legend(fontsize=8, loc="upper left")
    ax.text(0.01, -0.22, f"dense nodes {np.round(nd0, 1).tolist()}  |  {arm} nodes {np.round(nd, 1).tolist()}"
            + (f"  |err| {np.round(nerr, 1).tolist()} µm" if nerr else ""), transform=ax.transAxes, fontsize=8, va="top")
    sup = f"stiff probe, {band}, equispaced N = {N}, arm {arm}"
    if theta is not None and arm != "GP":
        if "tip_setback_um" in theta:
            sup += f"   (EB v1 fit: setback {theta['tip_setback_um']:.1f} µm, k*/k {theta['k_ratio']:.0f}, ζ {theta['zeta']:.4f}, χ²ᵣ {theta['red_chi2']:.1f})"
        else:
            sup += f"   (EB v2, calibrated lever: k*/k {theta['k_ratio']:.0f}, ζ {theta['zeta']:.4f}, ε {theta['eps']:.3f}, χ²ᵣ {theta['red_chi2']:.1f})"
    fig.suptitle(sup, fontsize=10); fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=150); plt.close(fig)
    return dict(N=N, arm=arm, nrmse_heldout=nr, nrmse_at_pos=recon.nrmse(zr[ipos], Z[ipos]),
                nodes=np.round(nd, 2).tolist(), node_err=np.round(nerr, 2).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="stiff"); ap.add_argument("--band", default="CR3")
    ap.add_argument("--N", nargs="+", type=int, default=[3, 5, 7, 9, 13, 15, 19])
    ap.add_argument("--xpos", type=float, default=120.0, help="TF position, um from clamp")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fitter", default="v2", choices=["v1", "v2"], help="v2: calibrated lever (ebfit); v1: physrec.rec_eb")
    ap.add_argument("--cal", default="mass", help="lever calibration tag for v2")
    a = ap.parse_args()
    key, g = KEYS[a.probe]; s = F.load(key)
    x_stage = s.x_um; x, f, Z = s.x_um - X0[a.probe], s.freq_Hz, s.Z[0]
    zb, fb = physrec.band_slice(f, Z, dict(s.probe.bands_Hz)[a.band], n_max=250)
    geom = physrec.GEOM[g]
    out = pathlib.Path(a.out or F.config.FIG_DIR / f"grid_{a.probe}_{a.band}_{a.fitter}"); out.mkdir(parents=True, exist_ok=True)
    ipos = int(np.argmin(np.abs(x - a.xpos)))
    rows = []
    for N in a.N:
        sel = recon.select_equispaced(x, N); h = recon.held_out(len(x), sel)
        print(f"N={N}: positions {np.round(x[sel]).astype(int).tolist()}", flush=True)
        eb = (eb_v2(a.probe, a.band, sel, x_stage, zb, fb, a.cal) if a.fitter == "v2"
              else eb_cached(a.probe, a.band, sel, x, zb, fb, geom))
        recs = {"EB": eb["Zeb"], "EB+GP": eb["Zrec"], "GP": recon.rec_gp(x, sel, zb[sel])["Zrec"]}
        for arm in ARMS:
            rows.append(panel(x, fb, zb, np.array(sel), h, recs[arm], arm, N, a.band, ipos, eb["theta"],
                              out / f"N{N:02d}_{arm.replace('+', 'p')}.png"))
    json.dump(rows, open(out / "summary.json", "w"), indent=1)
    print("\nN   arm     held-out NRMSE   node err (um)")
    for r in rows: print(f"{r['N']:<3} {r['arm']:<6} {r['nrmse_heldout']:6.1f} %        {r['node_err']}")
    print("wrote", out)


if __name__ == "__main__":
    main()

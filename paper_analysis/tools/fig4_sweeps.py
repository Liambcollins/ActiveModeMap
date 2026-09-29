"""Fig. 4: fast parameter sweeps on the soft probe — the full mode-shape dynamics vs bias and load
from 8-position captures.

Data: two-domain bias surveys at 15 nN and 250 nN: 8 positions x 7 V_dc (-9..+9 V) x 2 domains,
wideband. Every (load, bias, domain) is one 8-position live capture; each is reconstructed with the
calibrated EB arm (+GP inside the bands) and evaluated on the dense position grid over the whole band.

(a) reconstructed wideband maps at -9, 0, +9 V (250 nN, domain 1): the antiresonance branches move
    with V_dc; (b) measured and reconstructed spectra at one position vs V_dc around CR1; (c, d) CR1
    and CR3 on-resonance mode shapes vs V_dc (the shape changes only where the electrostatic pathway
    matters); (e) the fitted electrostatic weight eps(V_dc) per mode on both domains; (f) load: k*/k
    and resonance frequencies at 15 vs 250 nN.

    python tools/fig4_sweeps.py [--replot]
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
from fmmpaper import ebfit, physrec, recon, spectra, plotting as fp  # noqa: E402

PROBES = {
    # second sweep dimension: load for the soft probe; for the stiff probe the two runs R1/R2 of the same lever
    # (same 500 nN load, contact changed by the load ladder between them: the "contact state" comparison)
    "soft": dict(x0=-19.0, cal="soft", sets={"15 nN": "ppp_bias_15nN", "250 nN": "ppp_bias_250nN"}, main="250 nN", other="15 nN",
                 set_label="load", cal_x0=-19.0, x0_of={"15 nN": -19.0, "250 nN": -19.0}, frange=(40, 1090), spec=(40e3, 250e3), shapes=("CR1", "CR3"),
                 other_x=15.0, main_x=250.0, other_axis="load (nN)"),
    "stiff": dict(x0=28.1, cal="stiff", sets={"R1": "scmpitB_r1_bias", "R2": "scmpitB_r2_bias"}, main="R2", other="R1",
                  set_label="run", cal_x0=28.1, x0_of={"R1": 5.5, "R2": 24.1}, frange=(240, 1910), spec=(200e3, 1000e3), shapes=("CR1", "CR3"),
                  other_x=1.0, main_x=2.0, other_axis="run"),
}
PROBE = "soft"
CACHE = lambda: F.config.RESULTS_DIR / f"fig4_sweeps_{PROBE}.npz"
DEC = 8                                  # frequency decimation for the wideband maps


def compute():
    P = PROBES[PROBE]; cal0 = ebfit.load_calibration(P["cal"], "mass"); LOADS = P["sets"]
    s0 = F.load(LOADS[P["main"]]); xg = np.linspace(s0.x_um.min(), s0.x_um.max(), int(np.ptp(s0.x_um)) + 1)   # evaluation grid (stage um), 1 um pitch
    out = dict(xg=xg, theta={}, maps={}, fits_sel={}, x8=None, f=None)
    for load, key in LOADS.items():
        s = F.load(key); x8 = s.x_um; f = s.freq_Hz; bands = dict(s.probe.bands_Hz); out["x8"] = x8; out["f"] = f
        fg = f[::DEC]; out["fg"] = fg
        cond = s.conditions
        cal = json.loads(json.dumps(cal0)); cal["P"]["c_um"] = cal0["P"]["c_um"] - (P["cal_x0"] - P["x0_of"][load])   # ruler drift to this set's frame
        t0 = time.time()
        for i in range(len(cond)):
            V, spot = float(cond.bias_V.iloc[i]), int(cond.spot.iloc[i])
            Z = s.Z[i]
            fits, c_used = ebfit.fit_capture_bands(x8, Z, f, cal, bands, x_eval=xg)
            out["theta"][(load, V, spot)] = {b: dict(fits[b]["theta"], gain=fits[b]["gain"], c_um=c_used["P"]["c_um"]) for b in bands}
            out["maps"][(load, V, spot)] = ebfit.wideband_map(fits, c_used, bands, xg, fg, gp_corr=True).astype(np.complex64)
            out["fits_sel"][(load, V, spot)] = {b: fits[b]["Zeb"][[int(np.argmin(np.abs(xg - xx))) for xx in x8]] for b in bands}
            print(f"{load} V={V:+.0f} domain {spot}: c {c_used['P']['c_um']:.1f}  " + " ".join(f"{b} k*{fits[b]['theta']['k_ratio']:.0f} e{fits[b]['theta']['eps']:+.2f}" for b in bands) + f"  ({time.time()-t0:.0f} s)", flush=True)
    return out


def save(out):
    keys = list(out["maps"]); np.savez_compressed(CACHE(), xg=out["xg"], x8=out["x8"], f=out["f"], fg=out["fg"],
        keys=np.array([f"{k[0]}|{k[1]}|{k[2]}" for k in keys]),
        **{f"map_{i}": out["maps"][k] for i, k in enumerate(keys)},
        theta=json.dumps({f"{k[0]}|{k[1]}|{k[2]}": {b: {kk: (complex(vv).real, complex(vv).imag) if kk == "gain" else float(vv) for kk, vv in th.items()} for b, th in v.items()} for k, v in out["theta"].items()}))


def load_cache():
    d = np.load(CACHE(), allow_pickle=True); keys = [tuple(k.split("|")) for k in d["keys"]]
    out = dict(xg=d["xg"], x8=d["x8"], f=d["f"], fg=d["fg"], maps={}, theta={})
    th = json.loads(str(d["theta"]))
    for i, k in enumerate(keys):
        kk = (k[0], float(k[1]), int(k[2])); out["maps"][kk] = d[f"map_{i}"]
        out["theta"][kk] = {b: {a: (complex(*v) if a == "gain" else v) for a, v in t.items()} for b, t in th["|".join(k)].items()}
    return out


def plot(out, path):
    fp.setup(); W = fp.WIDTH_IN["double"]; P = PROBES[PROBE]; MAIN, OTHER = P["main"], P["other"]; X0 = P["x0_of"][MAIN]; LOADS = P["sets"]
    s = F.load(LOADS[MAIN]); bands = dict(s.probe.bands_Hz); names = list(bands); cond = s.conditions
    xg, x8, f, fg = out["xg"] - X0, out["x8"] - X0, out["f"], out["fg"]
    Vs = sorted(set(float(v) for v in cond.bias_V)); cmap = plt.get_cmap("coolwarm"); vc = lambda V: cmap((V + 9) / 18)
    fig = plt.figure(figsize=(W, W * 0.95))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.85, 0.85], hspace=0.42)
    # (a) maps at -9 / 0 / +9 V
    ga = gs[0].subgridspec(1, 4, width_ratios=[1, 1, 1, 0.04], wspace=0.15)
    ref = np.abs(out["maps"][(MAIN, 0.0, 1)]).max(); shapes = P["shapes"]
    for j, V in enumerate((-9.0, 0.0, 9.0)):
        ax = fig.add_subplot(ga[0, j]); im = fp.map_db(ax, xg, fg, out["maps"][(MAIN, V, 1)], ref=ref, fmin=P["frange"][0], fmax=P["frange"][1], colorbar=False)
        fp.mark_positions(ax, x8)
        ax.set_title(f"V$_{{dc}}$ = {V:+.0f} V", fontsize=6.5, color=vc(V)); ax.set_xlabel("distance from clamp x (µm)")
        if j == 0: ax.set_ylabel("frequency (kHz)")
        else: ax.set_yticklabels([])
        if j == 0: fp.panel_label(ax, "a", dx=-0.22)
    cb = fig.colorbar(im, cax=fig.add_subplot(ga[0, 3])); cb.set_label("|Z| (dB re max)", fontsize=6)
    fig.text(0.5, 0.955, f"full mode-shape dynamics from 8 positions per bias (triangles), {PROBE} probe, {MAIN}, domain 1", ha="center", fontsize=6.5)
    # (b) spectra vs bias at one position, (c) CR1 shapes, (d) CR3 shapes
    gb = gs[1].subgridspec(1, 3, wspace=0.35)
    axb, axc, axd = (fig.add_subplot(gb[0, i]) for i in range(3))
    ip = 3; jg = int(np.argmin(np.abs(xg - x8[ip])))       # 4th survey position
    m = (f >= P["spec"][0]) & (f <= P["spec"][1]); mg = (fg >= P["spec"][0]) & (fg <= P["spec"][1]); refb = None
    for V in Vs:
        i = int(np.intersect1d(s.where(bias_V=V), s.where(spot=1))[0]); z = s.Z[i, ip]
        if refb is None: refb = np.abs(z[m]).max()
        axb.plot(f[m] / 1e3, 20 * np.log10(np.abs(z[m]) / refb + 1e-9), color=vc(V), lw=0.6)
        axb.plot(fg[mg] / 1e3, 20 * np.log10(np.abs(out["maps"][(MAIN, V, 1)][jg][mg]) / refb + 1e-9), "--", color=vc(V), lw=0.6, alpha=0.8)
    axb.set_xlabel("frequency (kHz)"); axb.set_ylabel("|Z| (dB)"); axb.set_title(f"x = {x8[ip]:.0f} µm: measured, reconstructed (dashed)", fontsize=6)
    axb.set_ylim(-60, 3)
    for ax, b in ((axc, shapes[0]), (axd, shapes[1])):
        mb = (fg >= bands[b][0]) & (fg <= bands[b][1]); nrm = None
        for V in Vs:
            M = out["maps"][(MAIN, V, 1)]; a = np.abs(M[:, mb]).max(1)          # peak amplitude in the band at every position
            nrm = a.max() if nrm is None else nrm
            ax.plot(xg, a / nrm, color=vc(V), lw=0.8)
            i = int(np.intersect1d(s.where(bias_V=V), s.where(spot=1))[0])
            amp8 = np.abs(s.Z[i][:, ::DEC][:, mb]).max(1)                    # same frequency grid as the maps
            ax.plot(x8, amp8 / nrm, "v", color=vc(V), ms=2.6, mec="none")
        ax.set_xlabel("distance from clamp x (µm)"); ax.set_ylabel(f"|Z| at {b} peak (norm.)"); ax.set_title(f"{b} mode shape vs V$_{{dc}}$", fontsize=6.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-9, 9)); cbv = fig.colorbar(sm, ax=axd, fraction=0.05, pad=0.02); cbv.set_label("V$_{dc}$ (V)", fontsize=6)
    # (e) eps(V) per mode, both domains; (f) load
    ge = gs[2].subgridspec(1, 3, wspace=0.35)
    axe, axf, axg = (fig.add_subplot(ge[0, i]) for i in range(3))
    for b in names:
        for spot, mk, ls in ((1, "o", "-"), (2, "s", "--")):
            e = [out["theta"][(MAIN, V, spot)][b]["eps"] for V in Vs]
            axe.plot(Vs, e, ls, marker=mk, color=fp.C[b.lower()], ms=2.6, lw=0.8, mec="none", label=f"{b}" if spot == 1 else None)
    axe.axhline(0, color="0.7", lw=0.5); axe.set_xlabel("V$_{dc}$ (V)"); axe.set_ylabel("fitted electrostatic weight ε"); axe.set_title(f"ε(V$_{{dc}}$), {MAIN}: domain 1 (solid), domain 2 (dashed)", fontsize=6.5)
    axe.legend(fontsize=5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.3), handlelength=1.2, columnspacing=0.8, frameon=False)
    # (f) k*/k vs V for both loads; (g) resonance frequencies vs load
    for load, mk in ((OTHER, "o"), (MAIN, "s")):
        for b in names:
            k = [out["theta"][(load, V, 1)][b]["k_ratio"] for V in Vs]
            axf.plot(Vs, k, "-" if load == MAIN else ":", marker=mk, color=fp.C[b.lower()], ms=2.4, lw=0.8, mec="none")
    axf.set_yscale("log"); axf.set_xlabel("V$_{dc}$ (V)"); axf.set_ylabel("fitted k*/k"); axf.set_title(f"contact stiffness: {OTHER} (dotted) vs {MAIN} (solid)", fontsize=6.5)
    for b in names:
        fr = {load: np.mean([out["theta"][(load, V, 1)][b]["f_meas_Hz"] for V in Vs]) for load in LOADS}
        axg.plot([P["other_x"], P["main_x"]], [fr[OTHER] / fr[MAIN] * 100 - 100, 0], "o-", color=fp.C[b.lower()], ms=2.6, lw=0.8, mec="none", label=b)
    if PROBE == "soft":
        axg.set_xscale("log"); axg.set_xlabel("load (nN)"); axg.set_title("resonance frequencies vs load", fontsize=6.5)
    else:
        axg.set_xticks([P["other_x"], P["main_x"]]); axg.set_xticklabels([OTHER, MAIN]); axg.set_xlim(0.5, 2.5); axg.set_xlabel("run (same lever, contact changed by the load ladder)"); axg.set_title("resonance frequencies, R1 vs R2", fontsize=6.5)
    axg.set_ylabel(f"resonance shift re {MAIN} (%)"); axg.legend(fontsize=5, ncol=2)
    for ax, lab in ((axb, "b"), (axc, "c"), (axd, "d"), (axe, "e"), (axf, "f"), (axg, "g")):
        fp.panel_label(ax, lab, dx=-0.22)
    fig.savefig(path, dpi=600); fig.savefig(str(path).replace(".png", ".pdf")); plt.close(fig); print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--replot", action="store_true"); ap.add_argument("--probe", default="soft", choices=PROBES)
    a = ap.parse_args(); PROBE = a.probe
    if a.replot and CACHE().exists():
        out = load_cache()
    else:
        out = compute(); save(out)
    plot(out, F.config.FIG_DIR / ("Fig4_sweeps.png" if PROBE == "soft" else f"Fig4_sweeps_{PROBE}.png"))

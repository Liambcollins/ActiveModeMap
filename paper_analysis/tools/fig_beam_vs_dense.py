"""Figure: where the calibrated beam model departs from the dense maps (SCM-PIT R2, PPP-CONTAu), per resonance band.
Reads results/beam_vs_dense_{stiff,soft}.npz (tools/beam_vs_dense.py). Rows: (1) resonance frequency along the lever, as an
offset in measured linewidths; (2) on-resonance amplitude profile, measured vs model; (3) model / measured peak amplitude;
plus representative CR1 spectra. Model = 'all' fit (contact state per band fitted to every dense position; geometry fixed).
"""
import sys, json
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
HERE = Path(__file__).resolve().parent; R = HERE.parent / "results"
MEAS, MOD = "#1a1a1a", "#b2182b"
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False, "font.family": "DejaVu Sans"})

def peak_track(f, Z, nsm=9):
    A = np.abs(Z); As = uniform_filter1d(A, nsm, axis=1); k = As.argmax(1); n = A.shape[1]
    fp = np.empty(len(k)); ap = np.empty(len(k)); ph = np.empty(len(k)); df = f[1] - f[0]
    for i, j in enumerate(k):
        j = int(np.clip(j, 1, n - 2)); y0, y1, y2 = np.log(As[i, j - 1:j + 2]); den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        fp[i] = f[j] + d * df; ap[i] = np.exp(y1 - 0.25 * (y0 - y2) * d)
        zs = uniform_filter1d(Z[i].real, nsm) + 1j * uniform_filter1d(Z[i].imag, nsm); ph[i] = np.angle(zs[j])
    return fp, ap, ph

def summarize(probe):
    d = np.load(R / f"beam_vs_dense_{probe}.npz"); x = d["x"] - d["c_um"]; tip = d["L_um"] - d["setback_um"]
    res = dict(x=x, tip=float(tip), bands={})
    for b in d["bands"]:
        f, Z, M = d[f"{b}_f"], d[f"{b}_Z"], d[f"{b}_all"]
        fm, am, pm = peak_track(f, Z); fo, ao, po = peak_track(f, M)
        A = np.abs(uniform_filter1d(np.abs(Z), 9, axis=1)).max(0)
        # linewidth from the median measured Q
        th = d[f"{b}_all_theta"]; Q = float(th[list(json.loads('["log_alpha","log_zeta","eps","zeta","k_ratio","kcone_ratio","setback_um","c_um","dlf","f_meas_Hz","Q_meas"]')).index("Q_meas")])
        lw = np.nanmedian(fm) / Q
        nrm = np.sqrt(np.mean(am ** 2))
        res["bands"][str(b)] = dict(f=f, Z=Z, M=M, fm=fm, fo=fo, am=am, ao=ao, pm=pm, po=po, lw=lw, Q=Q, kr=float(th[4]), nrm=nrm)
    return res

S = {p: summarize(p) for p in ("stiff", "soft")}
names = {"stiff": "SCM-PIT R2 (161 positions, 500 nN)", "soft": "PPP-CONTAu (346 positions, 100 nN)"}
fig = plt.figure(figsize=(7.2, 9.6))
gs = fig.add_gridspec(2, 1, hspace=0.32)
out = {}
for gi, p in enumerate(("stiff", "soft")):
    r = S[p]; bands = list(r["bands"]); nb = len(bands)
    sub = gs[gi].subgridspec(5, 5, hspace=0.2, wspace=0.45, height_ratios=[1, 1.2, 0.9, 0.55, 1.1])
    x = r["x"]; out[p] = {}
    for j, b in enumerate(bands):
        B = r["bands"][b]; inside = x < r["tip"] - 2
        a1 = fig.add_subplot(sub[0, j]); a2 = fig.add_subplot(sub[1, j], sharex=a1); a3 = fig.add_subplot(sub[2, j], sharex=a1)
        ok = (B["am"] / B["nrm"] > 0.15) & (B["ao"] / B["nrm"] > 0.15)      # peak tracking is meaningless at a node
        dfl = np.where(ok, (B["fo"] - B["fm"]) / B["lw"], np.nan)
        a1.plot(x[ok], (B["fm"][ok] - np.nanmedian(B["fm"])) / B["lw"], ".", color=MEAS, ms=1.6, label="measured")
        a1.plot(x, np.where(ok, (B["fo"] - np.nanmedian(B["fm"])) / B["lw"], np.nan), "-", color=MOD, lw=1.2, label="beam model")
        a2.semilogy(x, B["am"] / B["nrm"], ".", color=MEAS, ms=1.6); a2.semilogy(x, B["ao"] / B["nrm"], "-", color=MOD, lw=1.2)
        ratio = np.where(ok, B["ao"] / B["am"], np.nan)
        a3.axhspan(0.9, 1.1, color="#e6e6e6", lw=0); a3.axhline(1, color="#9a9a9a", lw=0.6)
        a3.plot(x, ratio, "-", color=MOD, lw=1.0); a3.set_ylim(0.4, 1.8)
        for a in (a1, a2, a3):
            a.axvline(r["tip"], color="#9a9a9a", lw=0.6, ls=":")
            if a is not a3: plt.setp(a.get_xticklabels(), visible=False)
        a1.set_title(f"{b}  {np.nanmedian(B['fm'])/1e3:.0f} kHz\nk*/k {B['kr']:.0f}", fontsize=7)
        a3.set_xlabel("x from clamp (µm)")
        if j == 0:
            a1.set_ylabel("f_res − median\n(linewidths)"); a2.set_ylabel("|Z| at resonance\n(rel. RMS)"); a3.set_ylabel("model / measured\npeak amplitude")
        inside = inside & ok; q = lambda v: float(np.nanpercentile(v[inside], 50))
        out[p][b] = dict(freq_offset_lw_median=q(dfl), freq_offset_lw_range=[float(np.nanpercentile(dfl[inside], 5)), float(np.nanpercentile(dfl[inside], 95))],
                         amp_ratio_median=q(ratio), amp_ratio_p5_p95=[float(np.nanpercentile(ratio[inside], 5)), float(np.nanpercentile(ratio[inside], 95))], k_ratio=B["kr"], Q=B["Q"])
    # CR1 spectra at three positions
    B = r["bands"][bands[0]]; idx = [int(np.argmin(np.abs(x - v * r["tip"]))) for v in (0.25, 0.55, 0.85)]
    for j, i in enumerate(idx):
        a = fig.add_subplot(sub[4, j])
        a.plot(B["f"] / 1e3, np.abs(B["Z"][i]) / B["nrm"], color=MEAS, lw=0.7); a.plot(B["f"] / 1e3, np.abs(B["M"][i]) / B["nrm"], color=MOD, lw=1.0)
        a.set_title(f"CR1, x = {x[i]:.0f} µm", fontsize=7); fc = B["f"][np.argmax(np.abs(B["Z"][i]))]; a.set_xlim(fc/1e3 - 6 * B["lw"]/1e3, fc/1e3 + 6 * B["lw"]/1e3); a.set_xlabel("f (kHz)")
        if j == 0: a.set_ylabel("|Z| (rel.)")
    ax = fig.add_subplot(sub[4, 3:]); ax.axis("off")
    ax.text(0.0, 0.95, names[p], fontsize=8, weight="bold", va="top")
    ax.text(0.0, 0.62, "dots: measured dense map\nline: beam model (geometry from\ncalibration, contact state per band\nfitted to every position)\ndotted: tip; shaded: ±10 %", fontsize=6.5, va="top", color="#444444")
fig.text(0.01, 0.985, "a", weight="bold", fontsize=9); fig.text(0.01, 0.485, "b", weight="bold", fontsize=9)
for ext in ("png", "pdf"): fig.savefig(HERE.parent / f"figures/FigS_beam_vs_dense.{ext}", dpi=300, bbox_inches="tight")
json.dump(out, open(R / "beam_vs_dense_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))

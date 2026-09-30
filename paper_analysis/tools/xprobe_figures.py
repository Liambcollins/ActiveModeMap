"""Cross-probe figures: workflow on SCM-PIT R2, d33 across probes, imaging vs spectroscopy."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from fmmpaper import io as fio, domains
from fmmpaper.spectra import peak
R = HERE.parent / "results/r3r4"; F = HERE.parent / "figures/r3r4"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]; INK2 = "#52514e"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.5, "axes.edgecolor": INK2, "lines.linewidth": 1.6, "legend.frameon": False, "figure.dpi": 150})
def tag(ax, s): ax.text(-0.15, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")
X = pd.read_csv(R / "xprobe_table.csv"); S = pd.read_csv(R / "xprobe_summary.csv"); I = pd.read_csv(R / "xprobe_imaging.csv")
which = sys.argv[1:] or ["w", "x", "i"]

if "w" in which:   # workflow, SCM-PIT R2
    s = fio.load("scmpitB_r2_bias"); f6 = pd.DataFrame(json.load(open(HERE.parent / "results/fig6_main.json"))["values"]["runs"]["R2"]["table"])
    j = int(np.argmin(np.abs(s.x_um - 117.7))); i1, V1, i2, V2 = domains.split_spots(s); qm = s.band("QS")
    iref = int(np.intersect1d(s.where(bias_V=0), i1)[0]); pk = peak(s.freq_Hz, s.Z[iref, j], s.probe.bands_Hz["CR1"])
    conv = f6.invols[f6.x_um.round(1) == round(s.x_um[j], 1)].iloc[0] / 32 * 1e12
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.3))
    for sp, ii, VV, c in ((1, i1, V1, C[0]), (2, i2, V2, C[1])):
        zq = s.Z[ii, j][:, qm].mean(-1) * conv; o = np.argsort(VV)
        ax[0].plot(VV[o], np.real(zq[o]), "o-", color=c, label=f"domain {sp}")
    ax[0].axhline(0, color=INK2, lw=0.6); ax[0].set(xlabel="DC bias (V)", ylabel="QS response, in-phase (pm/V)", title=f"1. two domains, bias sweep (x = {s.x_um[j]:.0f} µm)")
    ax[0].legend(loc="lower right"); ax[0].text(0.03, 0.97, "P = half the gap between the lines\nb = common slope (electrostatic)", transform=ax[0].transAxes, fontsize=6.5, color=INK2, va="top"); tag(ax[0], "a")
    xl = f6.x_lever_um
    ax[1].plot(xl, f6.d33_qs_raw, "o-", color=C[0], label="QS, as measured"); ax[1].plot(xl, f6.d33_qs, "s-", color=C[1], label="QS, domain-CPD corrected")
    ax[1].set(xlabel="distance from clamp (µm)", ylabel="d33 (pm/V)", ylim=(0, 11), title="2. quasi-static d33 along the lever"); ax[1].legend(loc="lower left"); tag(ax[1], "b")
    ax[2].semilogy(xl, f6.E, "o-", color=C[0], label="E = |P_CR1| / |P_QS|"); ax[2].semilogy(xl, f6.Q, "s-", color=C[3], label="Q of CR1")
    ax[2].set(xlabel="distance from clamp (µm)", ylabel="gain at CR1", title="3. resonance gain E is not Q"); ax[2].legend(); tag(ax[2], "c")
    ax[3].plot(xl, f6.d33_qs, "s-", color=C[1], label="QS, corrected")
    ax[3].plot(xl, f6.d33_cr_eb, "o-", color=C[2], label="CR1 ÷ E from lever model")
    ax[3].plot(xl, np.clip(f6.d33_cr_over_Q, 0, 11), "^:", color=INK2, lw=1, label="CR1 ÷ Q (wrong)")
    ax[3].set(xlabel="distance from clamp (µm)", ylabel="d33 (pm/V)", ylim=(0, 11), title="4. CR1 route checks QS"); ax[3].legend(loc="lower left", fontsize=6.5); tag(ax[3], "d")
    fig.tight_layout(); fig.savefig(F / "fig_x_workflow_scmpit.png"); print("w")

if "x" in which:   # cross-probe
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6), gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
    order = ["PPP-CONTAu (15 nN)", "PPP-CONTAu (250 nN)", "Multi75-G R3 (350 nN)", "SCM-PIT R1 (500 nN)", "SCM-PIT R2 (500 nN)"]
    for k, ds in enumerate(order):
        r = S[S.dataset == ds].iloc[0]; t = X[(X.dataset == ds) & X.used]
        ax[0].plot(k - 0.2, r.raw_median, "o", mfc="none", color=INK2, ms=6, label="QS as measured (median)" if k == 0 else None)
        ax[0].vlines(k - 0.2, t.d33_qs_raw.min(), t.d33_qs_raw.max(), color=INK2, lw=0.8)
        ax[0].plot(k, r.d33_uniform, "s", color=C[1], ms=7, label="QS, domain-CPD corrected" if k == 0 else None)
        ax[0].vlines(k, t.d33_qs_corr.min(), t.d33_qs_corr.max(), color=C[1], lw=1.2)
        ax[0].plot(k + 0.2, r.d33_blind_spot_raw, "D", color=C[0], ms=6, label="QS at the blind spot (no correction)" if k == 0 else None)
        ax[0].plot(k + 0.4, r.cr1_model_median, "o", color=C[2], ms=6, label="CR1 ÷ lever-model E (median, range)" if k == 0 else None)
        ax[0].vlines(k + 0.4, t.d33_cr1_model.min(), min(t.d33_cr1_model.max(), 15), color=C[2], lw=1.2)
    ax[0].set(xticks=range(5), ylabel="d33 (pm/V)", ylim=(0, 19), title="d33 by route, each probe"); ax[0].set_xticklabels([o.replace(" (", "\n(") for o in order], fontsize=6.5)
    ax[0].legend(fontsize=6.3, loc="upper center", ncol=2); tag(ax[0], "a")
    for k, ds in enumerate(order):
        t = X[(X.dataset == ds) & X.used]; ax[1].plot(t.xi, t.d33_qs_raw, "o:", color=C[k], lw=0.8, ms=3); ax[1].plot(t.xi, t.d33_qs_corr, "s-", color=C[k], ms=4, label=ds)
    ax[1].set(xlabel="position along lever (0 = clamp, 1 = tip)", ylabel="QS d33 (pm/V)", ylim=(0, 16), title="raw (dotted) vs corrected (solid)"); ax[1].legend(fontsize=6); tag(ax[1], "b")
    L = [("PPP-CONTAu", 15, 5.92, C[0]), ("PPP-CONTAu", 250, 6.91, C[0]), ("Multi75-G", 350, 7.09, C[2]), ("SCM-PIT", 500, 7.27, C[6]), ("SCM-PIT", 500, 7.58, C[6])]
    ld = json.load(open(HERE.parent / "results/fig6_main.json"))["values"]["load"]
    for nm, l, v, c in L: ax[2].plot(l, v, "s", color=c, ms=7)
    ax[2].plot([r["load_nN"] for r in ld], [r["d33_corr"] for r in ld], "o--", color=C[6], mfc="none", label="SCM-PIT load series at one spot")
    ax[2].plot([350, 700], [7.08, 9.34], "^-", color=C[2], mfc="none", label="Multi75-G QS images (R4, not corrected)")
    ax[2].plot([350, 700], [7.29, 8.38], "v--", color=C[2], mfc="none", label="Multi75-G R4 bracket (700: conditional)")
    for nm, c in (("PPP-CONTAu survey", C[0]), ("Multi75-G survey", C[2]), ("SCM-PIT surveys", C[6])): ax[2].plot([], [], "s", color=c, label=nm)
    ax[2].set(xlabel="load (nN)", ylabel="d33 (pm/V)", ylim=(4, 10.5), title="d33 against load, all probes"); ax[2].legend(fontsize=6); tag(ax[2], "c")
    fig.tight_layout(); fig.savefig(F / "fig_x_crossprobe.png"); print("x")

if "i" in which:   # imaging vs spectroscopy, horizontal
    kinds = [("QS image", "QS image", C[0]), ("CR1 image", "CR1 image, fixed frequency", C[1]), ("BE image", "BE image (tracks the resonance)", C[2])]
    rows = []
    for key, name, c in kinds:
        for _, r in I[I.kind.str.startswith(key)].iterrows():
            if np.isfinite(r["ratio"]): rows.append((name, c, r["probe"].replace(" (", ", ").rstrip(")") + ", " + r["where"].split(" (")[0], r["ratio"]))
    fig, ax = plt.subplots(figsize=(7.5, 4.6)); y = 0; yt = []; yl = []; last = None
    for name, c, lab, v in rows:
        if name != last:
            y += 0.6; ax.text(0.555, y, name, color=c, fontweight="bold", fontsize=8, va="center"); y += 1; last = name
        ax.plot(v, y, "o", color=c, ms=7); yt.append(y); yl.append(lab); y += 1
    ax.axvspan(0.95, 1.05, color="#e6e5e0", zorder=0); ax.axvline(1, color=INK2, lw=0.8)
    ax.set(yticks=yt, xlabel="image ÷ spectroscopy (same convention, same lever position)", xlim=(0.55, 1.2), title="Imaging vs spectroscopy")
    ax.set_yticklabels(yl, fontsize=6.5); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(F / "fig_x_imaging.png"); print("i")

"""Figure: why measure on resonance (SCM-PIT AC series, 30 mV images, noise ratios across data sets)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from fmmpaper import beimg
root = Path(sys.argv[1]); R = HERE.parent / "results/r3r4"; F = HERE.parent / "figures/r3r4"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]; INK2 = "#52514e"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.5, "axes.edgecolor": INK2, "lines.linewidth": 1.6, "legend.frameon": False, "figure.dpi": 150})
def tag(ax, s): ax.text(-0.16, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")
A = json.load(open(R / "xprobe_sensitivity_ac.json"))["ac_series"]; Cq = pd.read_csv(R / "xprobe_sensitivity_images.csv"); B = pd.read_csv(R / "xprobe_sensitivity_survey.csv")
fig = plt.figure(figsize=(13, 7.4)); gs = fig.add_gridspec(2, 12, height_ratios=[1, 0.9], hspace=0.45, wspace=1.2)
ax0 = fig.add_subplot(gs[0, 0:4]); ax1 = fig.add_subplot(gs[0, 4:8]); ax4 = fig.add_subplot(gs[0, 8:12])
axm = [fig.add_subplot(gs[1, 3 * i:3 * i + 3]) for i in range(4)]
a = A["R2"]; t = pd.DataFrame(a["table"]); conv = a["invols"] / 32 * 1e12
ax0.loglog(t.vac, t.P_qs_bin * conv, "o-", color=C[0], label="QS (one bin, 30 kHz)"); ax0.loglog(t.vac, t.P_cr * conv / a["E"], "s-", color=C[1], label="CR1 peak ÷ E")
ax0.axhline(a["fits"]["P_qs_bin"]["n0"] * conv, color=C[0], ls=":", lw=1); ax0.axhline(a["fits"]["P_cr"]["n0"] * conv / a["E"], color=C[1], ls=":", lw=1)
ax0.text(0.0022, a["fits"]["P_qs_bin"]["n0"] * conv * 1.25, "QS noise floor", color=C[0], fontsize=6.5); ax0.text(0.0022, a["fits"]["P_cr"]["n0"] * conv / a["E"] * 1.25, "CR1 noise floor ÷ E", color=C[1], fontsize=6.5)
ax0.set(xlabel="drive V_ac (V)", ylabel="two-domain response, as tip displacement (pm)", title="SCM-PIT R2, one spot, 2 mV – 2 V"); ax0.legend(loc="lower right", fontsize=6.5); tag(ax0, "a")
for run, mk, ls in (("R2", "o", "-"), ("R1", "^", "--")):
    a = A[run]; t = pd.DataFrame(a["table"])
    ax1.semilogx(t.vac, t.d33_qs_bin, mk + ls, color=C[0], ms=4, lw=1, label=f"QS one bin ({run})"); ax1.semilogx(t.vac, t.d33_cr, mk + ls, color=C[1], ms=4, lw=1, label=f"CR1 ({run})")
ref = np.median([pd.DataFrame(A[r]["table"]).query("vac>=0.5").d33_cr.median() for r in A])
ax1.axhspan(0.9 * ref, 1.1 * ref, color="#e6e5e0", zorder=0)
for run in ("R2",):
    for col, c in (("P_qs_bin", C[0]), ("P_cr", C[1])):
        v = A[run]["fits"][col]["vac_snr10"]; ax1.axvline(v, color=c, lw=0.8, ls=":")
ax1.set(xlabel="drive V_ac (V)", ylabel="d33 (pm/V)", ylim=(0, 25), title="d33 recovered against drive"); ax1.legend(fontsize=6, ncol=2, loc="upper right"); tag(ax1, "b")
ax1.text(0.03, 0.03, f"SNR 10 per bin at {A['R2']['fits']['P_cr']['vac_snr10']*1e3:.0f} mV (CR1)\nvs {A['R2']['fits']['P_qs_bin']['vac_snr10']*1e3:.0f} mV (QS)", transform=ax1.transAxes, fontsize=6.5, color=INK2)
E = 265.5; ext = [0, 6, 0, 6]; d = root / "DomainsB_SCMPIT_R2"
maps = [("QS, 1 V", d / "50_quant_imaging/QImg0708.ibw", 1.0), ("CR1 ÷ E, 1 V", d / "50_quant_imaging/QImg0711.ibw", E),
        ("QS, 30 mV", d / "60_lowac_30mV/LoAC0710.ibw", 1.0), ("CR1 ÷ E, 30 mV", d / "60_lowac_30mV/LoAC0712.ibw", E)]
for k, (ttl, pth, sc) in enumerate(maps):
    q = beimg.load_qs(pth); im = axm[k].imshow(q["amp_m"] * 1e12 / q["drive_V"] / sc, cmap="Blues", vmin=0, vmax=16, origin="lower", extent=ext)
    axm[k].set(title=ttl, xlabel="µm"); axm[k].grid(False)
    if k: axm[k].set_yticklabels([])
tag(axm[0], "c"); fig.colorbar(im, ax=axm, fraction=0.02, pad=0.01, label="|d33| per pixel (pm/V)")
rows = [(r.case.replace("SCM-PIT ", "").replace(" (BE 8 ms)", ", BE"), r.gain_noise, C[1] if "30 mV" in r.case else (C[2] if "BE" in r.case else C[0])) for r in Cq.itertuples()]
rows += [(f"{d.split(' (')[0].replace('Multi75-G','M75')} survey spectra, 1 V", B[(B.dataset == d) & B.used].gain.median(), C[4]) for d in B.dataset.unique()]
for i, (lab, g, c) in enumerate(rows): ax4.barh(i, g, color=c, height=0.7)
ax4.axvline(1, color=INK2, lw=0.8); ax4.set_xscale("log"); ax4.set(yticks=range(len(rows)), xlabel="QS noise ÷ resonance noise (both in pm/V)", title="noise advantage of resonance")
ax4.set_yticklabels([r[0] for r in rows], fontsize=6); ax4.invert_yaxis(); ax4.grid(axis="y", visible=False); tag(ax4, "d")
fig.savefig(F / "fig_x_sensitivity.png", bbox_inches="tight"); print("ok")

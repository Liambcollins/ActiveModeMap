"""Fig. 6 (v3): calibrated resonance response at low drive, SCM-PIT R2 position A.

All values in the domain-CPD-corrected convention. The AC series stores the raw two-domain P; the
domain-CPD term is proportional to P at every drive (|b|/|P| is drive-independent, Supplementary Note 3),
so each channel is corrected by its survey factor at the same lever position:
    QS:  f_QS = d_QS,corr / d_QS,raw,      CR1: f_CR1 = d_CR1,corr / d_CR1,raw,
and the CR1 enhancement is moved to the corrected E with E_raw(x_A)/E_corr(x_A).
References: the corrected uniform QS value (7.58, flat by construction of delta) and the QS value at the
electrostatic blind spot near the tip (7.77), which needs no correction.

    python tools/fig6_v3.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fmmpaper as F  # noqa: E402
from fmmpaper import style3 as S, imgv3  # noqa: E402

R = F.config.RESULTS_DIR
A = json.load(open(R / "r3r4/xprobe_sensitivity_ac.json"))["ac_series"]
SUM = pd.read_csv(R / "r3r4/xprobe_summary.csv").set_index("dataset")
T = pd.DataFrame(json.load(open(R / "fig6_main.json"))["values"]["runs"]["R2"]["table"])
Cq = pd.read_csv(R / "r3r4/xprobe_sensitivity_images.csv"); B = pd.read_csv(R / "r3r4/xprobe_sensitivity_survey.csv")
r2 = SUM.loc["SCM-PIT R2 (500 nN)"]; D_UNI, D_TIP = float(r2.d33_uniform), float(r2.d33_blind_spot_raw)
IM = imgv3.position_A(A["R2"]["E"])
X = IM["X"]; it = lambda c: float(np.interp(X, T.x_lever_um, T[c]))
fQ = it("d33_qs") / it("d33_qs_raw"); fC = it("d33_cr_eb") / it("d33_cr_eb_raw"); fE = it("E_raw") / it("E")
print(f"x_A {X:.1f}: f_QS {fQ:.3f}, f_CR1 {fC:.3f}, E_raw/E_corr {fE:.3f}")

S.setup(); fig = S.figure(128)
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.02], hspace=0.42)
g1 = gs[0].subgridspec(1, 4, width_ratios=[1, 1, 0.62, 0.7], wspace=0.45)
ax0, ax1, ax2 = fig.add_subplot(g1[0, 0]), fig.add_subplot(g1[0, 1]), fig.add_subplot(g1[0, 3])
# (a) response vs drive, as tip displacement, with noise floors
a = A["R2"]; t = pd.DataFrame(a["table"]); conv = a["invols"] / 32 * 1e12
yq = t.P_qs_bin * conv * fQ; yc = t.P_cr * conv / a["E"] * fC * fE
ax0.loglog(t.vac, yq, "o-", color=S.QS, ms=3, lw=0.9); ax0.loglog(t.vac, yc, "s-", color=S.CR1, ms=3, lw=0.9)
nq = a["fits"]["P_qs_bin"]["n0"] * conv * fQ; nc = a["fits"]["P_cr"]["n0"] * conv / a["E"] * fC * fE
ax0.axhline(nq, color=S.QS, ls=":", lw=0.8); ax0.axhline(nc, color=S.CR1, ls=":", lw=0.8)
ax0.text(2.0, nq / 1.4, "QS noise floor", ha="right", va="top", fontsize=6.5); ax0.text(2.0, nc / 1.4, "CR1 noise floor", ha="right", va="top", fontsize=6.5)
ax0.text(0.0025, 1.05, "QS", color=S.QS, va="bottom"); ax0.text(0.028, 0.075, "CR1 ÷ E", color=S.CR1, va="center")
ax0.set_xlabel("drive $V_{ac}$ (V)"); ax0.set_ylabel("two-domain displacement (pm)"); S.label(ax0, "a", dx=-0.3)
# (b) d_eff vs drive, both runs, against the two references
ax1.axhspan(0.9 * D_UNI, 1.1 * D_UNI, color="#eeeeee", lw=0, zorder=0)
ax1.axhline(D_UNI, color=S.GREY, lw=0.8, ls="--", zorder=1); ax1.axhline(D_TIP, color=S.INK, lw=0.8, ls=":", zorder=1)
for run, mk, ls, mfc in (("R2", "o", "-", None), ("R1", "^", "--", "white")):
    a = A[run]; t = pd.DataFrame(a["table"])
    ax1.semilogx(t.vac, t.d33_qs_bin * fQ, mk + ls, color=S.QS, ms=3, lw=0.8, mfc=mfc or S.QS)
    ax1.semilogx(t.vac, t.d33_cr * fC * fE, mk + ls, color=S.CR1, ms=3, lw=0.8, mfc=mfc or S.CR1)
vq, vc = A["R2"]["fits"]["P_qs_bin"]["vac_snr10"], A["R2"]["fits"]["P_cr"]["vac_snr10"]
ax1.axvline(vc, color=S.CR1, lw=0.6, ls=":"); ax1.axvline(vq, color=S.QS, lw=0.6, ls=":")
ax1.set_ylim(0, 20); ax1.set_xlabel("drive $V_{ac}$ (V)"); ax1.set_ylabel(S.deff_label())
ax1.text(vc * 1.1, 0.5, f"{vc*1e3:.0f} mV", color=S.CR1, va="bottom", fontsize=6.5)
ax1.text(vq * 1.1, 0.5, f"{vq*1e3:.0f} mV", color=S.QS, va="bottom", fontsize=6.5)
from matplotlib.lines import Line2D
ax1.legend(handles=[Line2D([], [], color=S.QS, marker="o", ms=3, lw=0.8, label="QS, one bin"), Line2D([], [], color=S.CR1, marker="o", ms=3, lw=0.8, label="CR1 ÷ E"),
                    Line2D([], [], color=S.GREY, marker="o", ls="-", ms=3, lw=0.8, label="R2"), Line2D([], [], color=S.GREY, marker="^", mfc="white", ls="--", ms=3, lw=0.8, label="R1"),
                    Line2D([], [], color=S.GREY, ls="--", lw=0.8, label=f"survey {D_UNI:.2f}"), Line2D([], [], color=S.INK, ls=":", lw=0.8, label=f"tip {D_TIP:.2f}")],
           loc="upper right", ncol=1, fontsize=6.3, borderaxespad=0.1, handlelength=1.5)
S.label(ax1, "b", dx=-0.27)
fig.text(0, 0, "")
# (c) noise advantage of resonance; rows group comparable acquisitions (individual sets as dots)
surv = [B[(B.dataset == d) & B.used].gain.median() for d in B.dataset.unique()]
c1v = [r.gain_noise for r in Cq.itertuples() if "1 V" in r.case]; cbe = [r.gain_noise for r in Cq.itertuples() if "BE" in r.case]
c30 = {r.case.replace("SCM-PIT ", "").replace(", 30 mV", ""): r.gain_noise for r in Cq.itertuples() if "30 mV" in r.case}
rows = [("R1 drive series", [A["R1"]["gain_bin"]]), ("R2 drive series", [A["R2"]["gain_bin"]]),
        (f"surveys, 1 V ({len(surv)})", surv), ("R1 image, 30 mV", [c30["R1 A"]]), ("R2 image, 30 mV", [c30["R2 A"]]),
        (f"images 1 V, fixed f ({len(c1v)})", c1v), (f"images 1 V, BE ({len(cbe)})", cbe)]
for k, (lab, gs_) in enumerate(rows):
    lo, hi = min(gs_ + [1]), max(gs_ + [1])
    ax2.plot([min(gs_), max(gs_)], [k, k], color=S.LIGHT, lw=2.2, solid_capstyle="round", zorder=1)
    for g in gs_: ax2.plot(g, k, "o", color=S.CR1 if g > 1 else S.QS, ms=3.6, zorder=2)
ax2.axvline(1, color=S.INK, lw=0.6); ax2.set_xscale("log"); ax2.set_xlim(0.3, 40); ax2.set_ylim(len(rows) - 0.4, -0.6)
ax2.set_yticks(range(len(rows))); ax2.set_yticklabels([r[0] for r in rows]); ax2.tick_params(axis="y", length=0)
ax2.set_xlabel("QS noise ÷ CR1 noise"); ax2.spines["left"].set_visible(False)
ax2.set_xticks([0.5, 1, 2, 5, 10, 20]); ax2.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
ax2.text(0.9, -0.75, "QS better", ha="right", va="bottom", color=S.QS, fontsize=6.5); ax2.text(1.1, -0.75, "CR1 better", ha="left", va="bottom", color=S.CR1, fontsize=6.5)
S.label(ax2, "c", dx=-1.05, dy=1.04)
# (d) signed images at A
g2 = gs[1].subgridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.06], wspace=0.12)
VM = 12; ext = [0, 6, 0, 6]; panels = [("1V", "qs", "QS, 1 V"), ("1V", "cr", "CR1 ÷ E, 1 V"), ("30mV", "qs", "QS, 30 mV"), ("30mV", "cr", "CR1 ÷ E, 30 mV")]
for k, (tag, key, ttl) in enumerate(panels):
    ax = fig.add_subplot(g2[0, k]); S.boxes(ax); h = S.sym_im(ax, IM[tag][key], VM, extent=ext)
    ax.set_title(ttl); ax.set_xticks([0, 2, 4, 6]); ax.set_yticks([0, 2, 4, 6])
    ax.set_xlabel("µm")
    if k: ax.set_yticklabels([])
    else: ax.set_ylabel("µm"); S.label(ax, "d", dx=-0.3, dy=1.04)
cb = fig.colorbar(h, cax=fig.add_subplot(g2[0, 4]), ticks=[-12, -6, 0, 6, 12]); cb.set_label("in-phase " + S.deff_label())
S.save(fig, "Fig6_low_drive")
json.dump(dict(f_QS=fQ, f_CR1=fC, E_raw_over_corr=fE, d_uniform=D_UNI, d_tip=D_TIP,
               images={k: dict(qs_med=list(IM[k]["qs_med"]), cr_med=list(IM[k]["cr_med"])) for k in ("1V", "30mV")}),
          open(R / "fig6_v3.json", "w"), indent=1, default=float)

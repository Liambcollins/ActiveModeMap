"""Fig. 5 (v3): response calibration across four probe types.

(a) d_eff per survey by route: QS as measured, QS after the correction that the probe needs (domain-CPD on the three
metal-coated levers; stuck/sliding tip-pinning on the diamond Adama), QS at the electrostatic blind spot near the tip,
and CR1 / lever-model E (median and range over positions);
(b) QS d_eff along the lever before (open, dotted) and after (filled) correction, lever position as x/L;
(c) corrected d_eff against load.

    python tools/fig5_v3.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from matplotlib.lines import Line2D
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fmmpaper as F  # noqa: E402
from fmmpaper import style3 as S  # noqa: E402

R = F.config.RESULTS_DIR
X = pd.read_csv(R / "r3r4/xprobe_table.csv"); SM = pd.read_csv(R / "r3r4/xprobe_summary.csv").set_index("dataset")
LD = json.load(open(R / "fig6_main.json"))["values"]["load"]
# Adama R1 (diamond tip, 750 nN): blind EB fit with SEM geometry, stuck/sliding static correction (tools/adama_stuck_slide.py)
AD = pd.read_csv(R / "adama/adama_R1_stuck_slide_table.csv"); ADJ = json.load(open(R / "adama/adama_R1_stuck_slide.json"))
AD_LAB = "Adama R1 (750 nN)"; AD_L = ADJ["L_um"] - 21.2
ad_rows = [dict(dataset=AD_LAB, load_nN=750, x_um=r.x_um, xi=r.x_lever_um / AD_L, used=bool(r.used), d33_qs_raw=r.d33_qs, d33_qs_corr=r.d33_qs_stuckcorr,
                d33_cr1_model=r.d33_cr1_stuck, ratio_qs=r.ratio_qs) for r in AD.itertuples()]
X = pd.concat([X, pd.DataFrame(ad_rows)], ignore_index=True)
k = AD.used.values; ib = int(np.argmin(AD.ratio_qs.values))
SM.loc[AD_LAB] = dict(load_nN=750, raw_median=float(np.median(AD.d33_qs[k])), d33_uniform=float(np.median(AD.d33_qs_stuckcorr[k])),
                      d33_blind_spot_raw=float(AD.d33_qs[ib]), cr1_model_median=float(np.median(AD.d33_cr1_stuck[k])))

ORDER = ["PPP-CONTAu (15 nN)", "PPP-CONTAu (250 nN)", "Multi75-G R3 (350 nN)", "SCM-PIT R1 (500 nN)", "SCM-PIT R2 (500 nN)", AD_LAB]
probe = lambda d: d.split(" ")[0]
MK = {"PPP-CONTAu (15 nN)": "o", "PPP-CONTAu (250 nN)": "s", "Multi75-G R3 (350 nN)": "D", "SCM-PIT R1 (500 nN)": "^", "SCM-PIT R2 (500 nN)": "o", AD_LAB: "v"}
TICK = {"PPP-CONTAu (15 nN)": "15 nN", "PPP-CONTAu (250 nN)": "250 nN", "Multi75-G R3 (350 nN)": "R3\n350 nN",
        "SCM-PIT R1 (500 nN)": "R1\n500 nN", "SCM-PIT R2 (500 nN)": "R2\n500 nN", AD_LAB: "R1\n750 nN"}

S.setup(); fig = S.figure(64)
axa = fig.add_axes([0.06, 0.24, 0.43, 0.6]); axb = fig.add_axes([0.565, 0.24, 0.175, 0.6]); axc = fig.add_axes([0.815, 0.24, 0.175, 0.6])
# (a)
off = {"raw": -0.27, "corr": -0.09, "tip": 0.09, "model": 0.27}
for k_, d in enumerate(ORDER):
    r = SM.loc[d]; t = X[(X.dataset == d) & X.used]; adama = d == AD_LAB
    axa.vlines(k_ + off["raw"], t.d33_qs_raw.min(), t.d33_qs_raw.max(), color=S.GREY, lw=0.8)
    axa.plot(k_ + off["raw"], r.raw_median, "o", color=S.GREY, mfc="white", ms=4)
    axa.vlines(k_ + off["corr"], t.d33_qs_corr.min(), t.d33_qs_corr.max(), color=S.QS, lw=1.0)
    axa.plot(k_ + off["corr"], r.d33_uniform, "v" if adama else "o", color=S.QS, ms=4.6 if adama else 4)
    axa.plot(k_ + off["tip"], r.d33_blind_spot_raw, "*", color=S.INK, ms=6.5)
    axa.vlines(k_ + off["model"], t.d33_cr1_model.min(), min(t.d33_cr1_model.max(), 14), color=S.MODEL, lw=1.0)
    axa.plot(k_ + off["model"], r.cr1_model_median, "s", color=S.MODEL, ms=3.6)
axa.set_xticks(range(len(ORDER))); axa.set_xticklabels([TICK[d] for d in ORDER]); axa.tick_params(axis="x", length=0)
axa.set_ylim(0, 16.5)
for xx, nm in ((0.5, "PPP-CONTAu"), (2, "Multi75-G"), (3.5, "SCM-PIT"), (5, "Adama")):
    axa.text(xx, -0.24, nm, transform=axa.get_xaxis_transform(), ha="center", va="top", color=S.PROBE[nm])
axa.set_xlim(-0.5, len(ORDER) - 0.5); axa.set_ylabel(S.deff_label())
for k_ in range(1, len(ORDER)): axa.axvline(k_ - 0.5, color="#eeeeee", lw=0.6, zorder=0)
axa.legend(handles=[Line2D([], [], color=S.GREY, marker="o", mfc="white", ls="", ms=4, label="QS, as measured"),
                    Line2D([], [], color=S.QS, marker="o", ls="", ms=4, label="QS, domain-CPD corrected"),
                    Line2D([], [], color=S.QS, marker="v", ls="", ms=4.6, label="QS, tip-pinning corrected"),
                    Line2D([], [], color=S.INK, marker="*", ls="", ms=6.5, label="QS at tip blind spot"),
                    Line2D([], [], color=S.MODEL, marker="s", ls="", ms=3.6, label="CR1 ÷ lever-model E")],
           loc="lower center", ncol=3, columnspacing=0.9, handletextpad=0.4, bbox_to_anchor=(0.5, 1.0), fontsize=6.3)
axa.text(1 + off["model"] + 0.07, 11.6, "model fails", fontsize=6.5, ha="left", va="center", color=S.MODEL)
axa.text(5 + off["raw"] - 0.1, 13.6, "×1.7 along\nthe lever", fontsize=6.0, ha="right", va="center", color=S.GREY, linespacing=1.0)
S.label(axa, "a", dx=-0.115)
# (b)
for d in ORDER:
    t = X[(X.dataset == d) & X.used]; c = S.PROBE[probe(d)]; m = MK[d]
    axb.plot(t.xi, t.d33_qs_raw, m + ":", color=c, mfc="white", ms=3, lw=0.7)
    axb.plot(t.xi, t.d33_qs_corr, m + "-", color=c, ms=3, lw=0.9)
axb.set_xlabel("laser position x / L"); axb.set_ylabel(S.deff_label("QS")); axb.set_ylim(0, 16.5); axb.set_xlim(0.15, 1.05)
axb.text(0.2, 1.2, "open, dotted: raw · filled: corrected", color=S.GREY, fontsize=6.3)
axb.legend(handles=[Line2D([], [], color=S.PROBE[probe(d)], marker=MK[d], ls="-", ms=3, lw=0.9,
                           label={"PPP-CONTAu (15 nN)": "15 nN", "PPP-CONTAu (250 nN)": "250 nN", "Multi75-G R3 (350 nN)": "350 nN", "SCM-PIT R1 (500 nN)": "R1", "SCM-PIT R2 (500 nN)": "R2", AD_LAB: "Adama 750 nN"}[d]) for d in ORDER],
           loc="upper right", fontsize=6.0, handlelength=1.4, borderaxespad=0.0, bbox_to_anchor=(1.02, 1.04), labelspacing=0.25)
S.label(axb, "b", dx=-0.34)
# (c)
SV = [("PPP-CONTAu", 15, SM.loc[ORDER[0]].d33_uniform), ("PPP-CONTAu", 250, SM.loc[ORDER[1]].d33_uniform), ("Multi75-G", 350, SM.loc[ORDER[2]].d33_uniform),
      ("SCM-PIT", 500, SM.loc[ORDER[3]].d33_uniform), ("SCM-PIT", 500, SM.loc[ORDER[4]].d33_uniform), ("Adama", 750, SM.loc[AD_LAB].d33_uniform)]
axc.plot([r["load_nN"] for r in LD], [r["d33_corr"] for r in LD], "o--", color=S.PROBE["SCM-PIT"], mfc="white", ms=3.5, lw=0.8)
axc.plot([350, 700], [7.29, 8.38], "D--", color=S.PROBE["Multi75-G"], mfc="white", ms=3.2, lw=0.8)
for nm, l, v in SV: axc.plot(l, v, "v" if nm == "Adama" else "s", color=S.PROBE[nm], ms=4.6 if nm == "Adama" else 4.2, zorder=3)
axc.set_xlabel("load (nN)"); axc.set_ylabel(S.deff_label()); axc.set_xlim(-30, 800); axc.set_ylim(5, 9.5)
for nm, (xx, yy, ha) in {"PPP-CONTAu": (60, 5.85, "left"), "Multi75-G": (560, 8.55, "right"), "SCM-PIT": (290, 8.05, "right"), "Adama": (700, 5.75, "right")}.items():
    axc.text(xx, yy, nm, color=S.PROBE[nm], ha=ha)
axc.legend(handles=[Line2D([], [], color=S.GREY, marker="s", ls="", ms=4.2, label="survey, corrected"),
                    Line2D([], [], color=S.GREY, marker="o", mfc="white", ls="--", ms=3.5, lw=0.8, label="one spot / bracket")],
           loc="upper left", fontsize=6.3, bbox_to_anchor=(0.0, 1.04))
S.label(axc, "c", dx=-0.34)
S.save(fig, "Fig5_cross_probe")

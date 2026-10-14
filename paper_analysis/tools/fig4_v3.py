"""Fig. 4 (v3): domain separation and calibration of CR1 on SCM-PIT R2 (500 nN).

(a) in-phase QS response vs DC bias on the two domains at one lever position; (b) QS d_eff along the
lever before and after the domain-CPD correction, with the electrostatic blind spot near the tip;
(c) the CR1 enhancement E (corrected two-domain) against Q; (d) d_eff along the lever: corrected QS,
CR1 / blind-EB E_model, and CR1 / Q.

    python tools/fig4_v3.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fmmpaper as F  # noqa: E402
from fmmpaper import style3 as S  # noqa: E402

R = F.config.RESULTS_DIR
T = pd.DataFrame(json.load(open(R / "fig6_main.json"))["values"]["runs"]["R2"]["table"])
Bq = np.load(R / "r2_bias_qs.npz"); SUM = pd.read_csv(R / "r3r4/xprobe_summary.csv").set_index("dataset").loc["SCM-PIT R2 (500 nN)"]
x = T.x_lever_um.values; D_UNI, D_TIP = float(SUM.d33_uniform), float(SUM.d33_blind_spot_raw)

S.setup(); fig = S.figure(56)
axs = [fig.add_axes([0.06 + 0.245 * k, 0.2, 0.18, 0.7]) for k in range(4)]
# (a) bias sweep at the position nearest 118 um (lever)
j = int(np.argmin(np.abs(x - 116.5))); conv = T.invols.values[j] / 32 * 1e12
V, sp, Z = Bq["V"], Bq["spot"], Bq["Zqs"][:, j] * conv
fit = {}
for s_ in (1, 2):
    m = sp == s_; A_ = np.vstack([np.ones(m.sum()), V[m]]).T; fit[s_] = np.linalg.lstsq(A_, Z[m], rcond=None)[0]
P = (fit[1][0] - fit[2][0]) / 2; rot = np.exp(-1j * np.angle(P))
ax = axs[0]; vv = np.linspace(-9.5, 9.5, 2)
for s_, c, lab in ((1, S.D1, "domain 1"), (2, S.D2, "domain 2")):
    m = sp == s_; o = np.argsort(V[m])
    ax.plot(vv, np.real((fit[s_][0] + fit[s_][1] * vv) * rot), color=c, lw=0.8)
    ax.plot(V[m][o], np.real(Z[m][o] * rot), "o", color=c, ms=3.2)
    ax.text(9.5, np.real((fit[s_][0] + fit[s_][1] * 9.5) * rot) + (2.5 if s_ == 1 else -2.5), lab, color=c, ha="right", va="bottom" if s_ == 1 else "top")
ax.axhline(0, color=S.LIGHT, lw=0.6, zorder=0); ax.axvline(0, color=S.LIGHT, lw=0.6, zorder=0)
ax.annotate("", xy=(0, np.real(fit[1][0] * rot)), xytext=(0, np.real(fit[2][0] * rot)), arrowprops=dict(arrowstyle="<->", lw=0.7, color=S.INK, shrinkA=0, shrinkB=0))
ax.text(0.6, np.real((fit[1][0] + fit[2][0]) / 2 * rot), "2P$_{raw}$", va="center")
ax.set_xlabel("DC bias (V)"); ax.set_ylabel("QS response, in-phase (pm V$^{-1}$)"); ax.set_xticks([-9, -6, -3, 0, 3, 6, 9])
ax.text(9.5, -33, f"x = {x[j]:.0f} µm", ha="right", va="bottom"); S.label(ax, "a", dx=-0.32)
# (b) QS along the lever, raw vs corrected, blind spot
ax = axs[1]
ax.plot(x, T.d33_qs_raw, "o-", color=S.GREY, mfc="white", ms=3.2, lw=0.8)
ax.plot(x, T.d33_qs, "o-", color=S.QS, ms=3.2, lw=0.9)
ax.plot(x[-1], D_TIP, "*", color=S.INK, ms=7, zorder=5)
ax.text(70, 9.95, "raw P$_{raw}$", color=S.GREY, va="bottom"); ax.text(60, 7.15, f"corrected P (median {D_UNI:.2f})", color=S.QS, va="top")
ax.annotate(f"blind spot\n{D_TIP:.2f}", (x[-1], D_TIP), xytext=(x[-1] - 6, 9.2), ha="center", va="bottom", arrowprops=dict(arrowstyle="-", lw=0.5, color=S.INK))
ax.set_ylim(6, 10.8); ax.set_xlim(35, 220); ax.set_xlabel("distance from clamp x (µm)"); ax.set_ylabel(S.deff_label("QS"))
S.label(ax, "b", dx=-0.3)
# (c) E vs Q
ax = axs[2]
ax.semilogy(x, T.E, "o-", color=S.CR1, ms=3.2, lw=0.9); ax.semilogy(x, T.Q, "s--", color=S.GREY, ms=3, lw=0.8)
ax.text(60, 700, "E = |P$_{CR1}$| / |P$_{QS}$|", color=S.CR1, va="bottom"); ax.text(195, T.Q.values[-1] * 1.18, "Q", color=S.GREY, ha="center", va="bottom")
ax.set_xlim(35, 220); ax.set_ylim(20, 1500); ax.set_xlabel("distance from clamp x (µm)"); ax.set_ylabel("CR1 enhancement")
ax.set_yticks([20, 50, 100, 200, 500, 1000]); ax.set_yticklabels(["20", "50", "100", "200", "500", "1000"]); ax.minorticks_off()
S.label(ax, "c", dx=-0.3)
# (d) d_eff routes
ax = axs[3]
ax.axhline(D_UNI, color=S.GREY, lw=0.6, ls="--", zorder=0)
ax.plot(x, T.d33_qs, "o-", color=S.QS, ms=3.2, lw=0.9)
ax.plot(x, T.d33_cr_eb, "s-", color=S.MODEL, ms=3.2, lw=0.9)
cq = np.clip(T.d33_cr_over_Q, 0, 12.5); ax.plot(x, cq, "^:", color=S.GREY, ms=3, lw=0.8)
ax.plot(x[-1], D_TIP, "*", color=S.INK, ms=7, zorder=5)
ax.text(205, 7.75 + 0.55, "QS", color=S.QS, ha="right", va="bottom")
ax.text(42, 0.6, f"CR1 ÷ E$_{{model}}$ (median {np.median(T.d33_cr_eb):.2f})", color=S.MODEL, va="bottom")
ax.text(150, 11.0, "CR1 ÷ Q", color=S.GREY, ha="left")
ax.set_ylim(0, 12.5); ax.set_xlim(35, 220); ax.set_xlabel("distance from clamp x (µm)"); ax.set_ylabel(S.deff_label())
S.label(ax, "d", dx=-0.3)
S.save(fig, "Fig4_domain_calibration")
print("model median", np.median(T.d33_cr_eb), "sd %", 100 * np.std(T.d33_cr_eb) / np.median(T.d33_cr_eb))

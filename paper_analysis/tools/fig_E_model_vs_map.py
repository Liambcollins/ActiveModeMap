"""E(x) by lever model vs E(x) by reconstruction/measurement, one panel per probe survey.

Measured/reconstructed E: |P_CR1| / |P_QS| from the two-domain survey (domain-CPD corrected for the metal-coated
levers, as in Fig. 5; peak-tracked for the Adama). Model E: generic calibrated lever model (xprobe_cr1model, M_model x Q);
for SCM-PIT R2 also the blind EB fit of Fig. 3, and for the Adama the frequencies-only blind fit with SEM geometry.
    python tools/fig_E_model_vs_map.py
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
X = pd.read_csv(R / "r3r4/xprobe_table.csv")
SB = pd.read_csv(R / "r3r4/xprobe_semgeom_blind_table.csv")   # SEM geometry fixed, frequencies + Q only
PAN = [("PPP-CONTAu (15 nN)", "PPP-CONTAu", "15 nN", "soft15"), ("PPP-CONTAu (250 nN)", "PPP-CONTAu", "250 nN", "soft250"),
       ("Multi75-G R3 (350 nN)", "Multi75-G", "R3, 350 nN", "m75"), ("SCM-PIT R1 (500 nN)", "SCM-PIT", "R1, 500 nN", "stiffR1"),
       ("SCM-PIT R2 (500 nN)", "SCM-PIT", "R2, 500 nN", "stiffR2"), ("Adama R1 (750 nN)", "Adama", "R1, 750 nN", None)]
S.setup(); fig = S.figure(104)
axs = [fig.add_axes([0.07 + (i % 3) * 0.315, 0.555 - (i // 3) * 0.475, 0.25, 0.33]) for i in range(6)]
for ax, (ds, pr, tag, mtag) in zip(axs, PAN):
    c = S.PROBE[pr]
    if pr == "Adama":
        T = pd.read_csv(R / "adama/adama_R1_stuck_slide_table.csv"); J = json.load(open(R / "adama/adama_R1_stuck_slide.json"))
        xl = T.x_lever_um.values; Em = T.E.values; Emod = T.E_model_stuck.values; used = T.used.values
        ax.plot(J["x_grid"], J["E_grid"], "-", color=S.MODEL, lw=0.9)
        ax.plot(xl[used], Emod[used], "s", color=S.MODEL, ms=3.2, mfc="white")
        ax.plot(xl, Em, "o", color=S.MAP, ms=3.6)
        ax.plot(xl[~used], Em[~used], "o", color=S.MAP, ms=3.6, mfc="white")
        lab = "blind EB (freq. + Q, SEM geometry)"; Lc = J["L_um"] - 21.2
    else:
        t = X[X.dataset == ds].sort_values("xi"); used = t.used.values
        Lc, cpos = np.polyfit(t.xi.values, t.x_um.values, 1)      # x_um = c + xi * Lc  (Lc = L - setback, tip at xi = 1)
        xl = t.xi.values * Lc
        Em = t.M_meas_corr.values * t.Q.values            # corrected measured E
        Emod = t.M_model.values * t.Q.values               # model M x measured Q (as used for Fig. 5)
        ax.plot(xl[used], Emod[used], "s-", color=S.MODEL, ms=3.2, mfc="white", lw=0.9)
        ax.plot(xl[used], Em[used], "o", color=S.MAP, ms=3.6)
        ax.plot(xl[~used], Em[~used], "o", color=S.MAP, ms=3.6, mfc="white")
        lab = "generic lever model (M_model · Q)"
        sb = SB[SB.dataset == ds].sort_values("x_lever_um"); su = sb.used.values
        ax.plot(sb.x_lever_um.values[su], sb.E_sem_blind.values[su], "^", color=S.POLY, ms=3.4, mfc="white")
        rs = (sb.E_meas_corr.values / sb.E_sem_blind.values)[su]
        if mtag == "stiffR2":
            d = np.load(R / "fig_e_routes.npz", allow_pickle=True)
            ax.plot(d["x_grid"], d["E_model_grid"], "--", color=S.CR1, lw=0.9)
            ax.plot(d["x_grid"], d["E_map_cor_grid"], "-", color=S.MAP, lw=0.9, alpha=0.8)
            lab += " · dashed: blind EB · teal line: EB+GP map"
    ax.set_yscale("log"); ax.set_ylim(3, 800); ax.set_xlim(0, Lc * 1.08)
    ax.axvline(Lc, color=S.LIGHT, lw=0.6, zorder=0)
    ax.set_title(f"{pr}  ·  {tag}", color=c, pad=10, fontsize=7)
    ax.text(0.5, 1.02, lab, transform=ax.transAxes, fontsize=6.0, color=S.MODEL, ha="center", va="bottom")
    ax.set_xlabel("x from clamp (µm)")
    if ax in (axs[0], axs[3]): ax.set_ylabel("enhancement E")
    k = used & np.isfinite(Em) & np.isfinite(Emod)
    r = Em[k] / Emod[k]
    txt = f"E_meas / E_model\nmedian {np.median(r):.2f}, range {r.min():.2f}–{r.max():.2f}"
    if pr != "Adama": txt += f"\nSEM-geometry blind: {np.median(rs):.2f}, {rs.min():.2f}–{rs.max():.2f}"
    ax.text(0.03, 0.05, txt, transform=ax.transAxes, fontsize=6.3, color=S.INK, va="bottom")
fig.legend(handles=[Line2D([], [], color=S.MAP, marker="o", ls="", ms=3.6, label="E measured / reconstructed: |P_CR1| / |P_QS|"),
                    Line2D([], [], color=S.MAP, marker="o", ls="", ms=3.6, mfc="white", label="stop not used (tip region)"),
                    Line2D([], [], color=S.MODEL, marker="s", ls="-", ms=3.2, mfc="white", lw=0.9, label="E lever model"),
                    Line2D([], [], color=S.CR1, ls="--", lw=0.9, label="E blind EB (Fig. 3, SCM-PIT R2)"),
                    Line2D([], [], color=S.MAP, ls="-", lw=0.9, label="E reconstructed map (EB+GP, corrected)"),
                    Line2D([], [], color=S.POLY, marker="^", ls="", ms=3.4, mfc="white", label="E blind, SEM geometry fixed (freq. + Q)")],
           loc="upper center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, 1.0), columnspacing=1.2)
S.save(fig, "FigS_E_model_vs_map")

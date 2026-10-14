"""Supplementary Fig. 8 (2026-10-14): E(x) by the model route and by measurement, one panel per probe survey.

Measured E: |P_CR1| / |P_QS| from the two-domain survey, peak-tracked and domain-potential corrected (Adama: peak-tracked,
stuck/sliding handled in d_eff). Model route (squares, line): the single SEM-geometry recipe (tools/model_route_sem.py).
Open triangles: the dense-map / free-air calibrated generic model, for comparison. SCM-PIT R2 also shows the EB+GP map.
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
MR = pd.read_csv(R / "r3r4/model_route_sem.csv"); MJ = json.load(open(R / "r3r4/model_route_sem.json"))
PAN = [("PPP-CONTAu (15 nN)", "PPP-CONTAu", "15 nN"), ("PPP-CONTAu (250 nN)", "PPP-CONTAu", "250 nN"), ("Multi75-G R3 (350 nN)", "Multi75-G", "R3, 350 nN"),
       ("SCM-PIT R1 (500 nN)", "SCM-PIT", "R1, 500 nN"), ("SCM-PIT R2 (500 nN)", "SCM-PIT", "R2, 500 nN"), ("Adama R1 (750 nN)", "Adama", "R1, 750 nN")]
TIP = {"PPP-CONTAu": 461.6 - 23.5, "Multi75-G": 234.7 - 16.3, "SCM-PIT": 227.1 - 10.25, "Adama": 246.7 - 21.2}
S.setup(); fig = S.figure(104)
axs = [fig.add_axes([0.07 + (i % 3) * 0.315, 0.555 - (i // 3) * 0.475, 0.25, 0.33]) for i in range(6)]
for ax, (ds, pr, tag) in zip(axs, PAN):
    c = S.PROBE[pr]; t = MR[MR.dataset == ds].sort_values("x_lever_um"); j = MJ[ds]; used = t.used.values
    xl, Em, Emod = t.x_lever_um.values, t.E_meas.values, t.E_model.values
    ax.plot(j["x_grid"], j["E_grid"], "-", color=S.MODEL, lw=0.9)
    ax.plot(xl[used], Emod[used], "s", color=S.MODEL, ms=3.2, mfc="white")
    if np.isfinite(t.E_generic.values).any():
        ax.plot(xl[used], t.E_generic.values[used], "^", color=S.POLY, ms=3.4, mfc="white")
    ax.plot(xl[used], Em[used], "o", color=S.MAP, ms=3.6); ax.plot(xl[~used], Em[~used], "o", color=S.MAP, ms=3.6, mfc="white")
    lab = f"SEM geometry; k*/k {j['k_ratio']:.0f}, k$_{{cone}}$/k {j['kcone_ratio']:.0f}; CR err {', '.join(f'{e:+.1f}' for e in j['cr_err_pct'])} %"
    if ds == "SCM-PIT R2 (500 nN)":
        d = np.load(R / "fig_e_routes_sem.npz", allow_pickle=True)
        ax.plot(d["x_grid"], d["E_map_cor_grid"], "-", color=S.MAP, lw=0.9, alpha=0.8)
    Lc = TIP[pr]
    ax.set_yscale("log"); ax.set_ylim(3, 1200); ax.set_xlim(0, Lc * 1.08); ax.axvline(Lc, color=S.LIGHT, lw=0.6, zorder=0)
    ax.set_title(f"{pr}  ·  {tag}", color=c, pad=10, fontsize=7)
    ax.text(0.5, 1.02, lab, transform=ax.transAxes, fontsize=5.3, color=S.MODEL, ha="center", va="bottom")
    ax.set_xlabel("x from clamp (µm)")
    if ax in (axs[0], axs[3]): ax.set_ylabel("enhancement E")
    txt = f"E_meas / E_model: median {j['ratio_median']:.2f}, range {j['ratio_min']:.2f}–{j['ratio_max']:.2f}"
    if j.get("generic_ratio_median") is not None: txt += f"\ngeneric model: {j['generic_ratio_median']:.2f}"
    ax.text(0.03, 0.05, txt, transform=ax.transAxes, fontsize=6.3, color=S.INK, va="bottom")
fig.legend(handles=[Line2D([], [], color=S.MAP, marker="o", ls="", ms=3.6, label="E measured: |P_CR1| / |P_QS|, peak-tracked, corrected"),
                    Line2D([], [], color=S.MAP, marker="o", ls="", ms=3.6, mfc="white", label="stop within 10 µm of the tip or beyond (not scored)"),
                    Line2D([], [], color=S.MODEL, marker="s", ls="-", ms=3.2, mfc="white", lw=0.9, label="E model route (SEM geometry; CR frequencies + Q only)"),
                    Line2D([], [], color=S.POLY, marker="^", ls="", ms=3.4, mfc="white", label="E generic model (dense-map / free-air geometry), comparison"),
                    Line2D([], [], color=S.MAP, ls="-", lw=0.9, label="E reconstructed map (EB+GP, SCM-PIT R2)")],
           loc="upper center", ncol=3, fontsize=6.3, bbox_to_anchor=(0.5, 1.0), columnspacing=1.2)
S.save(fig, "FigS_E_model_vs_map")

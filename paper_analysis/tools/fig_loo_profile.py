"""Leave-one-out interpolation error of the reconstructed-map enhancement along the lever, all surveys.
E_meas/E_LOO at every survey stop against x/L (tip at 1). Open symbols: stops at or beyond the last interior
position (extrapolation). Right panel: the same against distance from the tip in um.
    python tools/fig_loo_profile.py
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
T = pd.read_csv(R / "r3r4/xprobe_map_route_table.csv"); SB = pd.read_csv(R / "r3r4/xprobe_semgeom_blind_table.csv")
AD = pd.read_csv(R / "adama/adama_R1_stuck_slide_table.csv"); ADJ = json.load(open(R / "adama/adama_R1_stuck_slide.json"))
SEM = {"SCM-PIT": (227.1, 10.25), "PPP-CONTAu": (461.6, 23.5), "Multi75-G": (234.7, 16.3), "Adama": (ADJ["L_um"], 21.2)}
MK = {"SCM-PIT R1 (500 nN)": "^", "SCM-PIT R2 (500 nN)": "o", "PPP-CONTAu (15 nN)": "o", "PPP-CONTAu (250 nN)": "s", "Multi75-G R3 (350 nN)": "D", "Adama R1 (750 nN)": "v"}
LAB = {"SCM-PIT R1 (500 nN)": "SCM-PIT R1", "SCM-PIT R2 (500 nN)": "SCM-PIT R2", "PPP-CONTAu (15 nN)": "PPP-CONTAu 15 nN", "PPP-CONTAu (250 nN)": "PPP-CONTAu 250 nN",
       "Multi75-G R3 (350 nN)": "Multi75-G R3", "Adama R1 (750 nN)": "Adama R1"}
S.setup(); fig = S.figure(62)
axa = fig.add_axes([0.07, 0.2, 0.5, 0.68]); axb = fig.add_axes([0.66, 0.2, 0.32, 0.68])
rows = []
for ds, g in T.groupby("dataset", sort=False):
    g = g.sort_values("x_um"); pr = ds.split(" ")[0]; L, sb = SEM[pr]; xt = L - sb
    if pr == "Adama": xl = AD.sort_values("x_um").x_lever_um.values
    else:
        m = SB[SB.dataset == ds].set_index("x_um").x_lever_um; xl = np.array([m[v] for v in g.x_um])
    r = (g.E_meas / g.E_loo).values; xi = xl / xt; last = np.zeros(len(r), bool); last[-1] = True   # outermost stop: extrapolation
    c = S.PROBE[pr]; mk = MK[ds]
    for ax, xx in ((axa, xi), (axb, xt - xl)):
        ax.plot(xx[~last], r[~last], mk, color=c, ms=4, mec="white", mew=0.4, zorder=3)
        ax.plot(xx[last], r[last], mk, color=c, ms=4, mfc="white", zorder=3)
        ax.plot(xx, r, "-", color=c, lw=0.6, alpha=0.6, zorder=2)
    rows.append(dict(dataset=ds, interior_median=float(np.median(r[~last])), interior_min=float(r[~last].min()), interior_max=float(r[~last].max()), last_stop=float(r[last][0])))
for ax in (axa, axb):
    ax.axhline(1, color=S.INK, lw=0.6); ax.axhspan(0.95, 1.05, color="#eeeeee", zorder=0)
    ax.set_yscale("log"); ax.set_ylim(0.4, 2.8); ax.set_yticks([0.5, 0.7, 1, 1.4, 2]); ax.set_yticklabels(["0.5", "0.7", "1", "1.4", "2"]); ax.minorticks_off()
axa.set_xlabel("laser position x / L (tip at 1)"); axa.set_ylabel("E$_{meas}$ / E$_{LOO}$"); axa.set_xlim(0.1, 1.12); axa.axvline(1, color=S.LIGHT, lw=0.6)
axa.text(0.12, 1.052, "±5 %", color=S.GREY, fontsize=6.3, va="bottom")
axb.set_xlabel("distance from the tip (µm)"); axb.set_xlim(-25, 400); axb.invert_xaxis(); axb.axvline(0, color=S.LIGHT, lw=0.6)
axa.legend(handles=[Line2D([], [], color=S.PROBE[d.split(" ")[0]], marker=MK[d], ls="", ms=4, label=LAB[d]) for d in MK] +
                   [Line2D([], [], color=S.GREY, marker="o", ls="", ms=4, mfc="white", label="outermost stop (extrapolation)")],
           loc="upper left", ncol=4, fontsize=6.2, bbox_to_anchor=(0.0, 1.0), columnspacing=0.9, handletextpad=0.4, borderaxespad=0.2)
S.label(axa, "a", dx=-0.1); S.label(axb, "b", dx=-0.2)
pd.DataFrame(rows).to_csv(R / "r3r4/loo_profile_summary.csv", index=False); print(pd.DataFrame(rows).round(3).to_string())
S.save(fig, "FigS_loo_profile")

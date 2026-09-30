"""R3 held-out walk: rank-r low-rank reconstruction from the 7 survey positions (0 V, spot 1), scored on the
7 held-out midpoints measured 3 h later (after imaging).
  (a) positions as labelled; (b) positions from InvOLS (results/r3r4/positions.csv: the walk sat ~12 um toward
  the free end of its labels); (c) = (b) with the prediction shifted in frequency by the median CR1 / CR2 shift
  between the two captures (the contact state changed during imaging: CR1 +1.4 kHz, Q 335 -> 275).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from fmmpaper import registry, recon
from fmmpaper.io import load_series_file
from fmmpaper.spectra import peak
root = Path(sys.argv[1] if len(sys.argv) > 1 else "D:/ActiveModeMap") / "DomainsB_SCMPIT_R3"
pr = registry.PROBES["multi75g"]; B = pr.bands_Hz
s = load_series_file(root / "10_bias_survey/domains_bias_checkpoint_500nN_scmpit.npz", pr)
v = load_series_file(root / "21_wideband_validation/wideband_valid_7pos_500nN_0V_checkpoint.npz", pr)
pos = pd.read_csv(HERE.parent / "results/r3r4/positions.csv")
xs_eq = np.array([pos[(pos.stage == "survey") & np.isclose(pos.x_label, x)].x_equiv.iloc[0] for x in s.x_um])
xv_eq = np.array([pos[(pos.stage == "held-out walk") & np.isclose(pos.x_label, x)].x_equiv.iloc[0] for x in v.x_um])
i0 = int(s.where(bias_V=0, spot=1)[0]); m = s.freq_Hz >= 100e3
Zd = s.Z[i0][:, m]; Zv = v.Z[0][:, m]; f = s.freq_Hz[m]
def shift(Zp, df_by_band):
    out = Zp.copy()
    for b, df in df_by_band.items():
        mb = (f >= B[b][0]) & (f <= B[b][1])
        for j in range(Zp.shape[0]):
            out[j, mb] = np.interp(f[mb] - df, f[mb], Zp[j, mb].real) + 1j * np.interp(f[mb] - df, f[mb], Zp[j, mb].imag)
    return out
res = {}
for case, xd, xv, align in (("a_labels", s.x_um, v.x_um, False), ("b_invols_x", xs_eq, xv_eq, False), ("c_invols_x_fshift", xs_eq, xv_eq, True)):
    rows = []
    for r in range(2, 8):
        Zp, _ = recon.cross_capture(xd, Zd, xv, arm="lowrank", rank=r)
        dfb = {}
        if align:
            for b in B:
                dfb[b] = float(np.median([peak(f, Zv[j], B[b])["f_Hz"] - peak(f, Zp[j], B[b])["f_Hz"] for j in range(len(xv))]))
            Zp = shift(Zp, dfb)
        sc = recon.score(Zp, Zv, f, bands=B); rows.append(dict(rank=r, **sc, **{f"df_{k}_Hz": val for k, val in dfb.items()}))
    best = min(rows, key=lambda d: d["nrmse"]); res[case] = dict(rows=rows, best=best)
    print(case, "best rank", best["rank"], {k: round(val, 3) for k, val in best.items() if isinstance(val, float)})
# per-position CR1 comparison at the best rank of (b)
rb = res["b_invols_x"]["best"]["rank"]; Zp, _ = recon.cross_capture(xs_eq, Zd, xv_eq, arm="lowrank", rank=rb); per = []
for j in range(len(xv_eq)):
    a = peak(f, Zv[j], B["CR1"]); b = peak(f, Zp[j], B["CR1"])
    per.append(dict(x_label=float(v.x_um[j]), x_equiv=float(xv_eq[j]), f_meas=a["f_Hz"], f_pred=b["f_Hz"], Q_meas=a["Q"], Q_pred=b["Q"], A_meas=a["amp"], A_pred=b["amp"]))
print(pd.DataFrame(per).round(4).to_string())
json.dump(dict(cases=res, per_position_b=per, x_design_equiv=xs_eq.tolist(), x_valid_equiv=xv_eq.tolist()), open(HERE.parent / "results/r3r4/r3_heldout.json", "w"), indent=1, default=float)

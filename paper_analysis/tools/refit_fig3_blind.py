"""Fig. 3c blind beam fit (SCM-PIT R2): free k*, k_cone, set-back, zeta, tip height; inputs CR1-3 frequencies, CR1 Q and the
1/InvOLS(x) shape; no PFM amplitude. Refitted with the smooth peak/Q evaluation (tools/eb_smooth.py), using the fixed-bin Q and
the peak-tracked Q, and scored against fixed-bin and peak-tracked E_meas; the cached published parameters are re-evaluated too."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import least_squares
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from fmmpaper import eb_fit, io as fio, spectra
R = HERE.parent / "results/r3r4"; L = 226.5; lab = "SCM-PIT R2 (500 nN)"
Tf = pd.read_csv(R / "xprobe_table.csv"); Tt = pd.read_csv(R / "xprobe_table_tracked.csv")
Tf = Tf[Tf.dataset == lab].sort_values("x_um").reset_index(drop=True); Tt = Tt[Tt.dataset == lab].sort_values("x_um").reset_index(drop=True)
s = fio.load("scmpitB_r2_bias")
i1 = int(np.intersect1d(s.where(bias_V=0), s.where(spot=1))[0])
fr = np.array([float(np.median(spectra.peaks_along_x(s.freq_Hz, s.Z[i1], s.probe.bands_Hz[b])["f_Hz"])) for b in ("CR1", "CR2", "CR3")])
_, c = np.polyfit(Tf.xi.values, Tf.x_um.values, 1); xl = Tf.x_um.values - c; keep = xl < L - 10; u = Tf.used.values
inv = Tf.invols_nm.values * 1e-9
p_pub = np.array([2.8656367734515715, -0.6411012199990145, 5.214132684585675, -2.586580768586415, 7.835476821246848])
lo = np.array([1.5, -2.0, 0.3, -3.5, 3.0]); hi = np.array([5.0, 4.0, 40.0, -1.0, 25.0])
def score(p, st, tag):
    sm = eb_fit.summarize(p, st); E = eb_fit.enhancement(p, st, xl)[0]
    rf = (Tf.M_meas_corr.values * Tf.Q.values) / E; rt = (Tt.M_meas_corr.values * Tt.Q.values) / E
    Pf = Tf.d33_cr1_model.values * Tf.M_model.values * Tf.Q.values; Pt = Tt.d33_cr1_model.values * Tt.M_model.values * Tt.Q.values   # corrected |P_CR1| in pm/V
    o = dict(tag=tag, p=p.tolist(), k_ratio=sm["k_ratio"], kcone_ratio=sm["kcone_ratio"], setback_um=sm["setback_um"], tip_height_um=sm["tip_height_um"], zeta=sm["zeta"],
             cr_err_pct=sm["cr_err_pct"], Q_model=sm["Q_model"], Q_input=st.Q1, static_rms_pct=sm["static_rms_pct"],
             ratio_fixed_median=float(np.median(rf[u])), ratio_fixed_range=[float(rf[u].min()), float(rf[u].max())],
             ratio_tracked_median=float(np.median(rt[u])), ratio_tracked_range=[float(rt[u].min()), float(rt[u].max())],
             d_eff_fixed_median=float(np.median((Pf / E)[u])), d_eff_tracked_median=float(np.median((Pt / E)[u])), d_eff_tracked_sd_pct=float(100 * np.std((Pt / E)[u]) / np.mean((Pt / E)[u])))
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items() if k != "p"}), flush=True); return o
out = []
st0 = eb_fit.ContactState("R2", fr, float(np.nanmedian(Tf.Q.values)), xl[keep], inv[keep], L_um=L)
out.append(score(p_pub, st0, "published params, grid evaluation"))
import eb_smooth  # noqa: F401  (patches eb_fit.peaks / q_model)
out.append(score(p_pub, st0, "published params, smooth evaluation"))
for qtag, T in (("fixed-bin Q", Tf), ("tracked Q", Tt)):
    st = eb_fit.ContactState("R2", fr, float(np.nanmedian(T.Q.values)), xl[keep], inv[keep], L_um=L)
    best = None
    for p0 in [p_pub] + [[la, lk, sb, np.log10(2.5e-3), th] for la in (2.7, 3.0) for lk in (-0.5, 1.0, 2.5) for sb, th in ((5.0, 8.0), (10.0, 11.7))]:
        r = least_squares(lambda p: eb_fit.residual(p, st, False), np.clip(p0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), diff_step=1e-3, max_nfev=120)
        if best is None or r.cost < best.cost: best = r
    out.append(score(best.x, st, f"refit smooth, {qtag}, cost {best.cost:.4f}"))
json.dump(out, open(R / "refit_fig3_blind.json", "w"), indent=1)

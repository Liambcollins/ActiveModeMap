"""Recompute the model-route numbers with peak-tracked CR1 (each bias condition/domain read at its own CR1 peak).
Same InvOLS per stop, same mask and same domain-CPD fit (delta from the QS channel, unchanged by tracking) as xprobe_d33.py.
Writes results/r3r4/xprobe_table_tracked.csv (xprobe_table columns, with Q, E, M_meas_corr, d33_cr1_model replaced) and
results/r3r4/recompute_tracked_summary.json (fixed-bin vs tracked, generic calibrated lever model)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from fmmpaper import io as fio, domains, registry, config
R = HERE.parent / "results/r3r4"; X = pd.read_csv(R / "xprobe_table.csv"); root = config.data_root()
SRC = {"SCM-PIT R1 (500 nN)": dict(key="scmpitB_r1_bias"), "SCM-PIT R2 (500 nN)": dict(key="scmpitB_r2_bias"),
       "PPP-CONTAu (15 nN)": dict(key="ppp_bias_15nN"), "PPP-CONTAu (250 nN)": dict(key="ppp_bias_250nN"),
       "Multi75-G R3 (350 nN)": dict(path=root / "DomainsB_SCMPIT_R3/10_bias_survey/domains_bias_checkpoint_500nN_scmpit.npz", probe="multi75g")}
rows, summ = [], {}
for lab, d in SRC.items():
    T = X[X.dataset == lab].reset_index(drop=True); u = T.used.values
    invmap = dict(zip(np.round(T.x_um.values, 1), T.invols_nm.values * 1e-9))
    s = fio.load(d["key"]) if "key" in d else fio.load_series_file(d["path"], registry.PROBES[d["probe"]])
    res = {}
    for trk in (False, True):
        t = domains.transfer_table(s, lambda x: invmap[round(float(x), 1)], 1.0, cr_band="CR1", track=trk)
        t = t.iloc[[int(np.argmin(np.abs(t.x_um.values - xx))) for xx in T.x_um.values]].reset_index(drop=True)
        cs = domains.cpd_split(t, mask=u)
        E = cs["E_corr"]; Q = t.Q.values; Pcr = cs["P_cr_corr"]
        Em = T.M_model.values * Q                          # generic model: M = E/Q of the calibrated lever, scaled by the measured Q
        ratio = E / Em; dcr = Pcr / Em
        res["tracked" if trk else "fixed"] = dict(delta_V=cs["delta_V"], d_eff_qs_corr=float(np.median(cs["d33_qs_corr"][u])),
            Q_median=float(np.median(Q[u])), ratio_median=float(np.median(ratio[u])), ratio_range=[float(ratio[u].min()), float(ratio[u].max())],
            d_eff_cr1_generic_median=float(np.median(dcr[u])), d_eff_cr1_generic_range=[float(dcr[u].min()), float(dcr[u].max())],
            cr1_trend_max_over_min=float(dcr[u].max() / dcr[u].min()), f_shift_Hz_median=float(np.median(t.f_cr_spread_Hz.values[u])))
        if trk:
            T2 = T.copy(); T2["Q"] = Q; T2["E"] = t.E.values; T2["M_meas_corr"] = E / Q; T2["d33_cr1_model"] = dcr; T2["f_cr_spread_Hz"] = t.f_cr_spread_Hz.values
            rows.append(T2)
    summ[lab] = res
    f, k = res["fixed"], res["tracked"]
    print(f"{lab:24s} QS corr {f['d_eff_qs_corr']:.2f} | E_meas/E_model {f['ratio_median']:.3f} -> {k['ratio_median']:.3f} ({k['ratio_range'][0]:.2f}-{k['ratio_range'][1]:.2f}) | "
          f"CR1/E_generic {f['d_eff_cr1_generic_median']:.2f} -> {k['d_eff_cr1_generic_median']:.2f} ({k['d_eff_cr1_generic_range'][0]:.2f}-{k['d_eff_cr1_generic_range'][1]:.2f}, x{k['cr1_trend_max_over_min']:.2f}) | "
          f"Q {f['Q_median']:.0f}->{k['Q_median']:.0f} | CR1 shift {k['f_shift_Hz_median']:.0f} Hz", flush=True)
pd.concat(rows + [X[~X.dataset.isin(SRC)]]).to_csv(R / "xprobe_table_tracked.csv", index=False)
json.dump(summ, open(R / "recompute_tracked_summary.json", "w"), indent=1)

"""Held-out walk (d): which survey bias/domain does the 0 V walk look like? Z_s(V) = a_s + b_s V fitted per
position and frequency on the survey (spot s), evaluated at V_eff, low-rank transferred to the walk positions
(InvOLS positions), shifted by the CR1/CR2 frequency change, scored. V_eff != 0 means the contact potential moved."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from fmmpaper import registry, recon, domains
from fmmpaper.io import load_series_file
import pandas as pd
root = Path(sys.argv[1]) / "DomainsB_SCMPIT_R3"; pr = registry.PROBES["multi75g"]; B = pr.bands_Hz
s = load_series_file(root / "10_bias_survey/domains_bias_checkpoint_500nN_scmpit.npz", pr)
v = load_series_file(root / "21_wideband_validation/wideband_valid_7pos_500nN_0V_checkpoint.npz", pr)
H = json.load(open(HERE.parent / "results/r3r4/r3_heldout.json")); xs, xv = np.array(H["x_design_equiv"]), np.array(H["x_valid_equiv"])
dfb = {k.split("_")[1]: val for k, val in H["cases"]["c_invols_x_fshift"]["best"].items() if k.startswith("df_")}
m = (s.freq_Hz >= 300e3); f = s.freq_Hz[m]; Zv = v.Z[0][:, m]
i1, V1, i2, V2 = domains.split_spots(s)
fits = {1: domains._linfit(V1, s.Z[i1][:, :, m])[:2], 2: domains._linfit(V2, s.Z[i2][:, :, m])[:2]}
def shift(Zp):
    out = Zp.copy()
    for b, df in dfb.items():
        mb = (f >= B[b][0]) & (f <= B[b][1])
        for j in range(Zp.shape[0]):
            out[j, mb] = np.interp(f[mb] - df, f[mb], Zp[j, mb].real) + 1j * np.interp(f[mb] - df, f[mb], Zp[j, mb].imag)
    return out
rows = []
for sp in (1, 2):
    a, b = fits[sp]
    for Ve in np.arange(-9, 9.01, 0.5):
        Zd = a + b * Ve
        for r in (4, 5):
            Zp, _ = recon.cross_capture(xs, Zd, xv, arm="lowrank", rank=r); Zp = shift(Zp)
            sc = recon.score(Zp, Zv, f, bands=B); rows.append(dict(spot=sp, V_eff=Ve, rank=r, **{k: sc[k] for k in ("nrmse", "nrmse_CR1", "nrmse_CR2", "within_3dB", "phase_mae_deg")}))
t = pd.DataFrame(rows); best = t.loc[t.groupby("spot").nrmse.idxmin()]
print(best.round(3).to_string()); print(t[(t["rank"] == 5) & (t.spot == 1)][["V_eff", "nrmse", "nrmse_CR1", "nrmse_CR2"]].round(1).to_string())
json.dump(dict(scan=rows, best=best.to_dict(orient="records")), open(HERE.parent / "results/r3r4/r3_heldout_bias.json", "w"), indent=1, default=float)

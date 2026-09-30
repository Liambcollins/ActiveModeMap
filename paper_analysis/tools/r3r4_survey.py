"""R3 two-domain bias survey (Multi75-G, 350 nN): d33_qs(x), V_cpd, in-situ E(x), domain-CPD correction.

InvOLS per survey position comes from the survey's OWN force curves (FSCMPIT0059-0071), recomputed
from the saved Z/deflection (fmmpaper.forcecurve), not from the pre-flight walk.
    python tools/r3r4_survey.py [--root D:/ActiveModeMap]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from fmmpaper import registry, domains, forcecurve as fc
from fmmpaper.io import load_series_file

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="D:/ActiveModeMap"); a = ap.parse_args()
R3 = Path(a.root) / "DomainsB_SCMPIT_R3"; OUT = HERE.parent / "results" / "r3r4"; OUT.mkdir(parents=True, exist_ok=True)
probe = registry.PROBES["multi75g"]
# survey force curves in acquisition order 232 -> 69.1 (log: position 1..7); 42 um failed (FSCMPIT0073 not saved)
FC = {232.0: "FSCMPIT0059.ibw", 204.9: "FSCMPIT0061.ibw", 177.7: "FSCMPIT0063.ibw", 150.6: "FSCMPIT0065.ibw",
      123.4: "FSCMPIT0067.ibw", 96.3: "FSCMPIT0069.ibw", 69.1: "FSCMPIT0071.ibw"}
inv = {}
for x, f in FC.items():
    r = fc.invols_fc(R3 / f)
    inv[x] = dict(file=f, ext=r["invols"], ret=r["invols_ret"], note_prior=r["note_invols"])
pre = json.load(open(R3 / "01_preflight" / "preflight_result.json"))
xp = np.array(sorted(float(k) for k in pre["invols_by_x"])); ip = np.array([float(pre["invols_by_x"][f"{v:.1f}"]) for v in xp])
def invols_fn(x):
    e = inv[float(x)]["ext"]; r = inv[float(x)]["ret"]
    return e if np.isfinite(e) else r            # 232: the extend fit fails (prior InvOLS 18x too big), use retract
s = load_series_file(R3 / "10_bias_survey" / "domains_bias_checkpoint_500nN_scmpit.npz", probe)
print(s)
t = domains.transfer_table(s, invols_fn, vac_V=1.0, cr_band="CR1")
t["invols_preflight_interp"] = np.interp(t.x_um, xp, ip)
cs_all = domains.cpd_split(t)
cs_in = domains.cpd_split(t, mask=(t.x_um < 230).values)    # without the free-end point (electrostatic blind spot, V_cpd ill-defined)
for k, cs in (("all", cs_all), ("no232", cs_in)):
    t[f"d33_qs_corr_{k}"] = cs["d33_qs_corr"]; t[f"E_corr_{k}"] = cs["E_corr"]
# residual of the uniform-d33 model: how well does a constant D + delta*b(x) describe P_qs(x)?
Pq = t.Pq_re.values + 1j * t.Pq_im.values; bq = t.bq_re.values + 1j * t.bq_im.values
for k, cs in (("all", cs_all), ("no232", cs_in)):
    D = np.median(Pq - cs["delta_V"] * bq)
    cs["resid_rms_pmV"] = float(np.sqrt(np.mean(np.abs(Pq - cs["delta_V"] * bq - D) ** 2)))
# alternative: per-domain CPD from the two spots' own linear fits (V_s = -Re(a_s/b_s))
qm = s.band("QS"); i1, V1, i2, V2 = domains.split_spots(s)
rows = []
for j, x in enumerate(s.x_um):
    zq = s.Z[:, j, qm].mean(-1)
    a1, b1, r1 = domains._linfit(V1, zq[i1]); a2, b2, r2 = domains._linfit(V2, zq[i2])
    rows.append(dict(x=float(x), V1=float(np.real(-a1 / b1)), V2=float(np.real(-a2 / b2)), resid1=float(r1), resid2=float(r2),
                     absb1=float(abs(b1)), absb2=float(abs(b2))))
pd_ = __import__("pandas").DataFrame(rows)
print(t[["x_um", "invols", "invols_preflight_interp", "f_cr_Hz", "Q", "E", "d33_qs", "vcpd_qs", "vcpd_cr", "ratio_qs", "flip_qs", "flip_cr",
         "d33_qs_corr_all", "d33_qs_corr_no232", "E_corr_no232"]].round(3).to_string())
print(pd_.round(3).to_string())
for k, cs in (("all", cs_all), ("no232", cs_in)):
    print(k, {kk: (round(v, 3) if isinstance(v, float) else None) for kk, v in cs.items() if isinstance(v, float)})
t.to_csv(OUT / "r3_survey_table.csv", index=False)
json.dump(dict(invols_forcecurves=inv, cpd_split_all={k: v for k, v in cs_all.items() if isinstance(v, float)},
               cpd_split_no232={k: v for k, v in cs_in.items() if isinstance(v, float)}, per_domain_cpd=rows,
               table=t.to_dict(orient="records")), open(OUT / "r3_survey.json", "w"), indent=1, default=float)

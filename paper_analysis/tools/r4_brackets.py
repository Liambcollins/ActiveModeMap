"""R4 two-domain bias brackets at the imaging position (label 166 um, InvOLS-equivalent ~158 um).
Valid bracket -> E, Q, M = E/Q, V_cpd, d33_qs. Brackets whose 'domains' did not flip (both spots on one domain)
are also reduced single-domain: p = a + b*V_ref with V_ref the V_cpd of the valid bracket (conditional E)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from fmmpaper import registry, domains, forcecurve as fc
from fmmpaper.io import load_series_file
from fmmpaper.spectra import peak
root = Path(sys.argv[1]) / "DomainsB_SCMPIT_R4"; pr = registry.PROBES["multi75g"]
BR = {"before_350": ("FSCMPIT0495.ibw", 350), "after_350": ("FSCMPIT0499.ibw", 350), "after_700": ("FSCMPIT0500.ibw", 700)}
rows, out = [], {}
for tag, (fcf, load) in BR.items():
    inv = fc.invols_fc(root / fcf)["invols"]
    s = load_series_file(root / "20_brackets" / f"bracket_{tag}_checkpoint.npz", pr)
    t = domains.transfer_table(s, lambda x: inv, vac_V=1.0).iloc[0].to_dict()
    # single-domain view per spot: QS and CR complex vs bias
    i1, V1, i2, V2 = domains.split_spots(s); qm = s.band("QS")
    iref = int(np.intersect1d(s.where(bias_V=0), i1)[0]); pk = peak(s.freq_Hz, s.Z[iref, 0], pr.bands_Hz["CR1"])
    per = {}
    for sp, ii, VV in ((1, i1, V1), (2, i2, V2)):
        zq = s.Z[ii, 0][:, qm].mean(-1); zc = s.Z[ii, 0, pk["k"]]
        aq, bq, _ = domains._linfit(VV, zq); ac, bc, _ = domains._linfit(VV, zc)
        per[sp] = dict(aq=aq, bq=bq, ac=ac, bc=bc, phase_qs0_deg=float(np.degrees(np.angle(aq))), Vnull_qs=float(domains.v0_null_bias(aq, bq, 0.0)))
    conv = inv / 32 * 1e12
    row = dict(bracket=tag, load_nN=load, invols_nm=inv * 1e9, f_CR1_kHz=t["f_cr_Hz"] / 1e3, Q=t["Q"], E=t["E"], M=t["E"] / t["Q"], vcpd_qs=t["vcpd_qs"], vcpd_cr=t["vcpd_cr"],
               d33_qs=t["d33_qs"], flip_qs=t["flip_qs"], flip_cr=t["flip_cr"], ratio_qs=t["ratio_qs"], ratio_cr=t["ratio_cr"],
               spot_phase_diff_deg=float((per[1]["phase_qs0_deg"] - per[2]["phase_qs0_deg"]) % 360), valid=bool(abs(t["flip_qs"] - 180) < 40))
    rows.append(row); out[tag] = dict(row=row, per_spot={k: {kk: (str(vv) if isinstance(vv, complex) else vv) for kk, vv in v.items()} for k, v in per.items()})
V_ref = [r for r in rows if r["valid"]][0]["vcpd_qs"]
for r, tag in zip(rows, BR):
    s = out[tag]["per_spot"]; conv = r["invols_nm"] * 1e-9 / 32 * 1e12
    for sp in (1, 2):
        aq, bq, ac, bc = (complex(s[sp][k]) for k in ("aq", "bq", "ac", "bc"))
        pq, pc = aq + bq * V_ref, ac + bc * V_ref
        r[f"s{sp}_d33_qs_at_Vref"] = abs(pq) * conv; r[f"s{sp}_E_at_Vref"] = abs(pc) / abs(pq); r[f"s{sp}_M_at_Vref"] = abs(pc) / abs(pq) / r["Q"]
t = pd.DataFrame(rows); print(t.round(3).T.to_string())
t.to_csv(HERE.parent / "results/r3r4/r4_brackets.csv", index=False)
json.dump(dict(V_ref=V_ref, brackets=rows), open(HERE.parent / "results/r3r4/r4_brackets.json", "w"), indent=1, default=float)

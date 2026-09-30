"""Where was the laser really? Every saved force curve gives InvOLS; the R3 pre-flight walk gives InvOLS(x)
(10 um steps, 14:57-15:21, same laser session as the survey). Inverting it assigns each capture an
'InvOLS-equivalent' position on the pre-flight axis. Output: results/r3r4/positions.json/.csv
"""
import json, sys, re
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from fmmpaper import forcecurve as fc
root = Path(sys.argv[1] if len(sys.argv) > 1 else "D:/ActiveModeMap"); R3 = root / "DomainsB_SCMPIT_R3"
pre = json.load(open(R3 / "01_preflight/preflight_result.json"))
xp = np.array(sorted(float(k) for k in pre["invols_by_x"])); ip = np.array([float(pre["invols_by_x"][f"{v:.1f}"]) for v in xp])
def x_equiv(inv):   # log-linear interpolation of the monotone InvOLS(x); outside the walk -> nan
    li = np.log(ip)[::-1]; return float(np.interp(np.log(inv), li, xp[::-1], left=np.nan, right=np.nan))
rows = []
def add(run, stage, label_x, f):
    r = fc.invols_fc(f); inv = r["invols"] if np.isfinite(r["invols"]) else r["invols_ret"]
    rows.append(dict(run=run, stage=stage, x_label=label_x, file=f.name, time=r["kv"].get("Time"), invols_nm=inv * 1e9,
                     x_equiv=x_equiv(inv), offset_um=x_equiv(inv) - label_x if label_x is not None else np.nan))
for i, x in enumerate([232, 222, 212, 202, 192, 182, 172, 162, 152, 142, 132, 122, 112, 102, 92, 82, 72, 62, 52, 42]):
    add("R3", "preflight", float(x), R3 / f"FSCMPIT{18 + 2*i:04d}.ibw")
for x, n in zip([232.0, 204.9, 177.7, 150.6, 123.4, 96.3, 69.1], range(59, 72, 2)):
    add("R3", "survey", x, R3 / f"FSCMPIT{n:04d}.ibw")
add("R3", "imaging (16:13)", 96.3, R3 / "FSCMPIT0076.ibw"); add("R3", "imaging (16:14)", 96.3, R3 / "FSCMPIT0078.ibw")
for x, n in zip([218.4, 191.3, 164.1, 137.0, 109.8, 82.7, 55.6], range(79, 92, 2)):
    add("R3", "held-out walk", x, R3 / f"FSCMPIT{n:04d}.ibw")
for i in range(186):
    f = R3 / f"FSCMPIT{108 + 2*i:04d}.ibw"
    if f.exists(): add("R3", "dense map", 232.0 - i, f)
t = pd.DataFrame(rows); out = HERE.parent / "results/r3r4"
t.to_csv(out / "positions.csv", index=False)
print(t[t.stage != "dense map"].round(1).to_string())
d = t[t.stage == "dense map"]
for lo, hi in ((200, 233), (140, 200), (90, 140), (47, 90)):
    m = (d.x_label >= lo) & (d.x_label < hi); print(f"dense map {lo}-{hi}: offset median {np.nanmedian(d.offset_um[m]):+.1f} um  (n {m.sum()})")
g = t.groupby("stage").offset_um.agg(["median", "min", "max", "count"]); print(g.round(1))
json.dump(dict(preflight_x=xp.tolist(), preflight_invols_nm=(ip * 1e9).tolist(), summary=g.reset_index().to_dict(orient="records")), open(out / "positions.json", "w"), indent=1, default=float)

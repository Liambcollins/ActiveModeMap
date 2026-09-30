"""Dense map (R3, 350 nN, 0 V, one domain) as a held-out check of the free-air + sparse lever model:
CR1/CR2 peak-amplitude profiles along the lever on the InvOLS-equivalent axis vs the model (geometry frozen,
k* refit so CR1 matches the dense map's CR1). Also: the node of CR2."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from fmmpaper import jointeb as J
import sim_calib_freeair as S
S.F0 = 72900.0
D = np.load(HERE.parent / "results/r3r4/r3_dense_reduced.npz"); pos = pd.read_csv(HERE.parent / "results/r3r4/positions.csv")
dm = pos[pos.stage == "dense map"].set_index("x_label")
xl = D["x_label"]; xe = np.array([dm.x_equiv.get(v, np.nan) for v in xl]); ok = np.isfinite(xe)
A1, A2 = np.abs(D["zpk_CR1"]), np.abs(D["zpk_CR2"]); f1, f2 = np.median(D["fpk_CR1"]), np.median(D["fpk_CR2"])
res = dict(f_CR1_kHz=f1 / 1e3, f_CR2_kHz=f2 / 1e3, ratio=f2 / f1)
def node(x, a, lo, hi):
    m = (x > lo) & (x < hi); i = np.argmin(a[m]); return float(x[m][i])
res["dense_CR2_node_equiv"] = node(xe[ok], A2[ok], 60, 225); res["dense_CR2_node_label"] = node(xl, A2, 60, 225)
out = {}
for tag in ("calfa_mu0.026", "calfa_muprior0.003", "calfa_freemu"):
    d = json.load(open(HERE.parent / f"results/r3r4/{tag}.json")); P = d["contact_fit"]["P"]; pc = d["contact_fit"]["per_capture"][0]
    g = np.linspace(pc["log_alpha"] - 0.3, pc["log_alpha"] + 0.3, 61); fg = np.array([S.eig(dict(P, log_alpha=v, log_Qc=pc["log_Qc"]), 1)[0] for v in g]); la = float(np.interp(f1, fg, g))
    Pl = dict(P, log_alpha=la, log_Qc=pc["log_Qc"]); fc = S.eig(Pl, 2)
    x = np.arange(40, 236, 1.0)
    je = J.JointEB(x, [J.Band("CR1", np.array([fc[0]]), None), J.Band("CR2", np.array([fc[1]]), None)], S.F0, n_modes=S.NM); S._scale(je, Pl)
    (p1, e1), (p2, e2) = je.band_maps(Pl, [1e-9, 1e-9], x=x)          # undamped shapes at the eigenfrequencies
    m1 = np.abs(p1[:, 0]); m2 = np.abs(p2[:, 0])
    # scale each model profile to the dense data by least squares on the InvOLS axis
    a1 = np.interp(xe[ok], x, m1); a2 = np.interp(xe[ok], x, m2)
    s1 = np.dot(a1, A1[ok]) / np.dot(a1, a1); s2 = np.dot(a2, A2[ok]) / np.dot(a2, a2)
    r1 = np.sqrt(np.mean((s1 * a1 - A1[ok]) ** 2)) / np.sqrt(np.mean(A1[ok] ** 2)); r2 = np.sqrt(np.mean((s2 * a2 - A2[ok]) ** 2)) / np.sqrt(np.mean(A2[ok] ** 2))
    out[tag] = dict(log_alpha=la, f_model_kHz=(fc / 1e3).tolist(), CR2_node_model=node(x, m2, 60, 225), nrmse_CR1_profile=float(r1), nrmse_CR2_profile=float(r2),
                    x=x.tolist(), m1=(s1 * m1).tolist(), m2=(s2 * m2).tolist())
    print(tag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[tag].items() if k not in ("x", "m1", "m2")})
print(res)
json.dump(dict(summary=res, models=out, x_label=xl.tolist(), x_equiv=xe.tolist(), A1=A1.tolist(), A2=A2.tolist()), open(HERE.parent / "results/r3r4/r3_dense_vs_model.json", "w"), indent=1, default=float)

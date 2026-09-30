"""Per-domain d33 from BE (BEPyAE) and QS (Asylum) frames, R3 and R4 (Multi75-G).

BE:  d33 = A / M * InvOLS / 32 / V_ac        (estimator 'A/M'; A = SHO amplitude = peak/Q_pix)
     d33 = A * Q_pix / (M * Q_ref) * ...      (estimator 'peak/E'; peak from the fit, E = M * Q_ref of the calibration state)
QS:  d33 = AmplitudeRetrace (m) / V_ac        (already calibrated with the scan's InvOLS; rescaled if that InvOLS was wrong)
Two-domain contrast = |<z>_a - <z>_b| / 2 (complex medians); it keeps the domain-CPD term delta*b but removes the common one.
    python tools/r3r4_frames.py <data root> R3|R4
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from fmmpaper import beimg
root = Path(sys.argv[1]); run = sys.argv[2]; OUT = HERE.parent / "results/r3r4"; OUT.mkdir(exist_ok=True, parents=True)
cal = json.load(open(OUT / f"{run.lower()}_frame_calibration.json"))     # M, Q_ref, InvOLS per contact state (written by r3r4_calib.py)

def qpix_slopes(A, Q, lab):
    s = {}
    for k in (0, 1):
        m = (lab == k) & np.isfinite(A) & np.isfinite(Q) & (A > 0) & (Q > 20)
        if m.sum() < 200: continue
        lq, la = np.log(Q[m]), np.log(A[m])
        s[k] = dict(exp_A_vs_Q=float(np.polyfit(lq, la, 1)[0]), exp_peak_vs_Q=float(np.polyfit(lq, la + lq, 1)[0]),
                    Q_p10=float(np.percentile(Q[m], 10)), Q_p90=float(np.percentile(Q[m], 90)))
    return s

rows = []
todo = [dict(fr, state=stt) for fr in cal["frames"] for stt in (fr["states"] if "states" in fr else [fr.get("state", "")])]
cache = {}
for fr in todo:
    p = root / fr["path"]
    if fr["kind"] == "BE":
        if p not in cache:
            b = beimg.load_be(p); lab, cen = beimg.split_domains(b["phi"])
            if not (120 < (cen[0] - cen[1]) % 360 < 240): lab = np.zeros_like(lab)   # no antiparallel pair: one domain in the window
            cache[p] = (b, lab, beimg.domain_stats(b["A"], b["phi"], lab, extra=dict(Q=b["Q"], f=b["f"], peak=b["A"] * b["Q"])), qpix_slopes(b["A"], b["Q"], lab))
        b, lab, st, qs_ = cache[p]
        c = cal["states"][fr["state"]]; conv = fr["invols"] / 32 / fr["vac"] * 1e12
        row = dict(frame=fr["name"], kind="BE", state=fr["state"], load_nN=fr["load_nN"], bias_V=fr["bias_V"], pixel_ms=1e3 * b["pixel_time_s"], window=str(b["attrs"].get("window", "")), npx=int(b["A"].shape[0]),
                   invols_nm=fr["invols"] * 1e9, M=c["M"], Q_ref=c["Q_ref"])
        for k in (0, 1):
            d = st[k]
            row.update({f"d{k}_frac": d["frac"], f"d{k}_d33_A_over_M": d["amp_med"] / c["M"] * conv,
                        f"d{k}_d33_peak_over_E": d["peak_med"] / (c["M"] * c["Q_ref"]) * conv, f"d{k}_Q": d["Q_med"], f"d{k}_f_kHz": d["f_med"] / 1e3})
        row["contrast_A_over_M"] = st.get("contrast", np.nan) / c["M"] * conv; row["flip_deg"] = st.get("flip_deg", np.nan)
        row["single_domain"] = bool(st[1]["frac"] < 0.08)
        row["d33_best"] = row["d0_d33_A_over_M"] if row["single_domain"] else row["contrast_A_over_M"]
        for k, v in qs_.items(): row.update({f"d{k}_{kk}": vv for kk, vv in v.items()})
    else:
        q = beimg.load_qs(p); ph = np.radians(q["phase_deg"]); lab, cen = beimg.split_domains(ph)
        if not (120 < (cen[0] - cen[1]) % 360 < 240): lab = np.zeros_like(lab)
        scale = fr.get("invols_true", q["invols"]) / q["invols"]
        st = beimg.domain_stats(q["amp_m"] * scale, ph, lab)
        conv = 1e12 / q["drive_V"]
        row = dict(frame=fr["name"], kind="QS", state=fr["state"], load_nN=fr["load_nN"], bias_V=fr["bias_V"], invols_nm=q["invols"] * 1e9 * scale,
                   invols_scan_nm=q["invols"] * 1e9, f_drive_kHz=q["f_drive"] / 1e3, scan_um=q["scan_size"] * 1e6, setpoint_V=q["setpoint_V"])
        for k in (0, 1):
            row.update({f"d{k}_frac": st[k]["frac"], f"d{k}_d33": st[k]["amp_med"] * conv})
        row["contrast"] = st.get("contrast", np.nan) * conv; row["flip_deg"] = st.get("flip_deg", np.nan); row["single_domain"] = bool(st[1]["frac"] < 0.08)
        row["d33_best"] = row["d0_d33"] if row["single_domain"] else row["contrast"]
    rows.append(row); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
t = pd.DataFrame(rows); t.to_csv(OUT / f"{run.lower()}_frames.csv", index=False)
json.dump(dict(calibration=cal, frames=rows), open(OUT / f"{run.lower()}_frames.json", "w"), indent=1, default=float)

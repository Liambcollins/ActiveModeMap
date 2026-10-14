"""SEM-based blind lever fits, v2 (2026-10-13).

Same inputs and scoring as xprobe_semgeom_blind.py (fit to CR frequencies + CR1 Q only; E(x) and the InvOLS profile withheld),
with two changes:
 1. Smooth objective. The v1 fit read model peak frequencies and Q off fixed frequency grids, so the residual was piecewise
    constant and least_squares stopped at its starting point for zeta (and often k_cone). Here each peak is refined on a local
    fine grid with parabolic interpolation, and Q uses interpolated half-power crossings.
 2. Geometry variants:
    sem      L, set-back, tip height all fixed at SEM values.
    tipfix   clamp-to-tip distance a = L_SEM - sb_SEM fixed (the tip position is what SEM measures well); only the overhang
             beyond the tip (effective set-back) is free in [0.3, sb_SEM], so L_eff = a + sb_eff. Rationale: a tapered end carries
             less mass and stiffness than a rectangular overhang of the same length.
    sbfree   L fixed at SEM, set-back free in [0.3, 40] (v1 --free_sb, i.e. the tip is allowed to move).
    apex     L, tip height fixed at SEM; set-back fixed at SEM set-back + h*tan(11 deg), i.e. the vertical load acts at the
             tilted tip APEX rather than at the tip base (the apex sits h*tan(t) ~ 2.3-2.8 um nearer the clamp).
    python tools/xprobe_semgeom_blind_v2.py --geom tipfix
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from fmmpaper import registry, domains, eb_fit, io as fio, config, spectra  # noqa: E402

_peaks0 = eb_fit.peaks
def peaks_smooth(p, st, f_lo=None, f_hi=None, n=2500):
    fk = _peaks0(p, st, f_lo, f_hi, n)
    if fk is None: return None
    out = []
    for f in fk:
        ff = np.linspace(0.985 * f, 1.015 * f, 401)
        m, r = eb_fit._model(p, st, ff)
        i = np.argmin(np.abs(m.xi - 0.55)); a = np.abs(r["piezo"][:, i]); k = int(np.argmax(a))
        if 0 < k < a.size - 1:
            y0, y1, y2 = np.log(a[k - 1:k + 2]); den = y0 - 2 * y1 + y2
            d = 0.5 * (y0 - y2) / den if den != 0 else 0.0
            out.append(m.f_hz[k] + d * (m.f_hz[1] - m.f_hz[0]))
        else: out.append(m.f_hz[k])
    return np.sort(np.array(out))

def q_smooth(p, st, f1):
    for span in (0.04, 0.12):
        ff = np.linspace(f1 * (1 - span), f1 * (1 + span), 2401)
        m, r = eb_fit._model(p, st, ff)
        i = np.argmin(np.abs(m.xi - 0.55)); a = np.abs(r["piezo"][:, i]); j = int(np.argmax(a)); h = a[j] / np.sqrt(2)
        lo = j
        while lo > 0 and a[lo] > h: lo -= 1
        hi = j
        while hi < a.size - 1 and a[hi] > h: hi += 1
        if lo > 0 and hi < a.size - 1: break
    fl = np.interp(h, [a[lo], a[lo + 1]], [ff[lo], ff[lo + 1]]); fh = np.interp(h, [a[hi], a[hi - 1]], [ff[hi], ff[hi - 1]])
    return f1 / (fh - fl)

eb_fit.peaks = peaks_smooth; eb_fit.q_model = q_smooth

ap = argparse.ArgumentParser(); ap.add_argument("--geom", default="sem", choices=["sem", "tipfix", "sbfree", "apex"]); ap.add_argument("--nfev", type=int, default=150)
ap.add_argument("--only", default=""); ap.add_argument("--tag", default=""); ap.add_argument("--table", default="xprobe_table.csv")
a = ap.parse_args()
root = config.data_root(); R = HERE.parent / "results"; OUT = R / "r3r4"
X = pd.read_csv(OUT / a.table)
SEM = {"scmpitB": dict(L=227.1, sb=10.25, h=11.7), "pppcontau": dict(L=461.6, sb=23.5, h=14.3), "multi75g": dict(L=234.7, sb=16.3, h=12.8)}
D = {"SCM-PIT R1 (500 nN)": ("scmpitB", root / "DomainsB_SCMPIT/10_bias_survey/domains_bias_checkpoint_500nN_scmpit.npz"),
     "SCM-PIT R2 (500 nN)": ("scmpitB", root / "DomainsB_SCMPIT_R2/10_bias_survey/domains_bias_checkpoint_500nN_scmpit.npz"),
     "PPP-CONTAu (15 nN)": ("pppcontau", root / "DomainsBPPPCONTAU/02_domains_bias_15nN/domains_bias_checkpoint.npz"),
     "PPP-CONTAu (250 nN)": ("pppcontau", root / "DomainsBPPPCONTAU/07_bias_survey_250nN/domains_bias_checkpoint_250nN.npz"),
     "Multi75-G R3 (350 nN)": ("multi75g", root / "DomainsB_SCMPIT_R3/10_bias_survey/domains_bias_checkpoint_500nN_scmpit.npz")}
CACHE = OUT / "xprobe_semgeom_v2_inputs.json"
cache = json.load(open(CACHE)) if CACHE.exists() else {}
rows = []; summary = {}
for lab, (pk, path) in D.items():
    if a.only and a.only not in lab: continue
    probe = registry.PROBES[pk]; g = SEM[pk]
    if lab in cache: cr = np.array(cache[lab])
    else:
        s = fio.load_series_file(path, probe); d = domains.decompose_series(s, band=None); P = d["P"]; f = s.freq_Hz; cr = []
        for b in ("CR1", "CR2", "CR3"):
            if b not in probe.bands_Hz: break
            lo_, hi_ = probe.bands_Hz[b]
            if hi_ > f.max(): break
            pkk = [spectra.peak(f, P[i], (lo_, hi_)) for i in range(P.shape[0])]
            fk = np.array([q["f_Hz"] for q in pkk]); amp = np.array([q["amp"] for q in pkk]); w = amp > 0.3 * amp.max()
            cr.append(float(np.median(fk[w])))
        cr = np.array(cr); cache[lab] = cr.tolist(); json.dump(cache, open(CACHE, "w"))
    t = X[X.dataset == lab].sort_values("xi"); used = t.used.values
    _, c_cal = np.polyfit(t.xi.values, t.x_um.values, 1); xl = t.x_um.values - c_cal
    Q1 = float(np.nanmedian(t.Q.values[used]))
    lo, hi = eb_fit.LO.copy(), eb_fit.HI.copy(); lo[4], hi[4] = g["h"] - 1e-3, g["h"] + 1e-3
    a_tip = g["L"] - g["sb"]
    if a.geom == "sem": lo[2], hi[2] = g["sb"] - 1e-3, g["sb"] + 1e-3
    elif a.geom == "apex":   # vertical load acts at the tilted apex, h*tan(11 deg) nearer the clamp: effective set-back = SEM set-back + h tan(t)
        sb_apex = g["sb"] + g["h"] * np.tan(np.deg2rad(11.0)); lo[2], hi[2] = sb_apex - 1e-3, sb_apex + 1e-3
    elif a.geom == "tipfix": lo[2], hi[2] = 0.3, g["sb"]
    sb_starts = [g["sb"]] if a.geom == "sem" else [g["sb"] + g["h"] * np.tan(np.deg2rad(11.0))] if a.geom == "apex" else ([0.5 * g["sb"]] if a.geom == "tipfix" else [0.5 * g["sb"]])
    from scipy.optimize import least_squares
    def make(Lc):
        st = eb_fit.ContactState(lab, cr, Q1, np.array([]), np.array([]), f0_free_Hz=probe.f0_free_Hz, k_lever=probe.k_N_per_m, L_um=Lc)
        def res(p):
            if a.geom == "tipfix": st.L_um = a_tip + p[2]
            return eb_fit.residual(p, st, False, sig_f=0.015, sig_static=1e3, sig_Q=0.05)[np.r_[0:len(cr), -1]]
        return st, res
    cands = []
    for sb0 in sb_starts:
        for la in (2.6, 3.0, 3.4):
            for lk in (0.3, 1.5, 2.7):
                st, res = make((a_tip + sb0) if a.geom == "tipfix" else g["L"])
                r = least_squares(res, [la, lk, sb0, np.log10(3e-3), g["h"]], bounds=(lo, hi), diff_step=1e-3, max_nfev=25)
                cands.append(r.x)
    best = None
    for x0 in sorted(cands, key=lambda x: float(np.sum(make((a_tip + x[2]) if a.geom == "tipfix" else g["L"])[1](x) ** 2)))[:2]:
        st, res = make((a_tip + x0[2]) if a.geom == "tipfix" else g["L"])
        r = least_squares(res, x0, bounds=(lo, hi), diff_step=1e-3, max_nfev=a.nfev)
        if a.geom == "tipfix": st.L_um = a_tip + r.x[2]
        if best is None or r.cost < best[0].cost: best = (r, st)
    r, st = best; p = r.x; sm = eb_fit.summarize(p, st)
    E_model = eb_fit.enhancement(p, st, xl)[0]
    xt = st.L_um - p[2]; p_sl = p.copy(); p_sl[1] = -2.0
    rr = (eb_fit.static_shape(p, st, xl) / eb_fit.static_shape(p, st, np.array([xt]))[0]) / (eb_fit.static_shape(p_sl, st, xl) / eb_fit.static_shape(p_sl, st, np.array([xt]))[0])
    E_meas = t.M_meas_corr.values * t.Q.values; E_gen = t.M_model.values * t.Q.values
    ratio = E_meas / E_model; k = used & np.isfinite(ratio)
    y = 1 / t.invols_nm.values[used]; z = eb_fit.static_shape(p_sl, st, xl[used]); gg = np.sum(y * z) / np.sum(z * z); st_rms = float(np.sqrt(np.mean(np.log(y / (gg * z)) ** 2)))
    zr = ratio[k]; xr = xl[k]; slope = np.polyfit(xr / st.L_um, zr, 1)[0]
    print(f"== {lab} [{a.geom}] L {st.L_um:.1f} sb {p[2]:.1f} h {p[4]:.1f} | CR err % {sm['cr_err_pct']} Q {sm['Q_model']:.0f}/{Q1:.0f} k*/k {sm['k_ratio']:.0f} kcone/k {sm['kcone_ratio']:.1f} zeta {sm['zeta']:.2e} cost {r.cost:.3f} nfev {r.nfev}")
    print(f"   E_meas/E_model median {np.median(zr):.3f} range {zr.min():.2f}-{zr.max():.2f} (trend dRatio/d(x/L) {slope:+.2f}) | generic median {np.median((E_meas/E_gen)[k]):.3f} | r(x) {rr[used].min():.2f}-{rr[used].max():.2f} | sliding ruler rms {100*st_rms:.1f}%")
    d_sem = t.d33_cr1_model.values * E_gen / E_model / rr
    for i in range(len(xl)):
        rows.append(dict(dataset=lab, geom=a.geom, x_um=t.x_um.values[i], x_lever_um=xl[i], used=bool(used[i]), Q=t.Q.values[i], E_meas_corr=E_meas[i], E_generic=E_gen[i], E_model=E_model[i], ratio=ratio[i], r_stuck_slide=rr[i], d33_cr1_model=d_sem[i], d33_qs_corr=t.d33_qs_corr.values[i] if "d33_qs_corr" in t else np.nan))
    summary[lab] = dict(geom=a.geom, L_um=st.L_um, cr_Hz=cr.tolist(), Q1=Q1, params=dict(zip(eb_fit.PARAMS[:5], p.tolist())), summary=sm,
                        ratio_median=float(np.median(zr)), ratio_range=[float(zr.min()), float(zr.max())], ratio_trend=float(slope),
                        generic_ratio_median=float(np.median((E_meas / E_gen)[k])), r_range=[float(rr[used].min()), float(rr[used].max())],
                        static_rms_sliding_pct=100 * st_rms, d33_cr1_median=float(np.nanmedian(d_sem[k])), d33_cr1_range=[float(np.nanmin(d_sem[k])), float(np.nanmax(d_sem[k]))])
pd.DataFrame(rows).to_csv(OUT / f"xprobe_semgeom_v2_{a.geom}{a.tag}_table.csv", index=False)
json.dump(summary, open(OUT / f"xprobe_semgeom_v2_{a.geom}{a.tag}.json", "w"), indent=1, default=float)
